#!/usr/bin/env python3
"""
资金效率分析 CLI

用法:
  python scripts/cli/capital_efficiency_cli.py              # 组合效率总览
  python scripts/cli/capital_efficiency_cli.py 300059       # 单仓效率分析
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
from datetime import datetime
from typing import Dict, List
from scripts.analyzers.capital_efficiency import CapitalEfficiencyAnalyzer
from scripts.core.tushare_client import TushareClient

ETF_PREFIXES = ('51', '15', '56', '58', '52', '50', '11', '12', '13', '16', '18')


def is_etf(code: str) -> bool:
    return code.startswith(ETF_PREFIXES)


def get_etf_tscode(code: str) -> str:
    suffix = 'SH' if code.startswith(('5', '11')) else 'SZ'
    return f"{code}.{suffix}"


def load_portfolio():
    """从 portfolio.xlsx 和 capital.xlsx 加载持仓和资金数据"""
    import openpyxl

    # 加载持仓
    portfolio_path = os.path.join(os.path.dirname(__file__), '../../config/portfolio.xlsx')
    holdings = []
    if os.path.exists(portfolio_path):
        wb = openpyxl.load_workbook(portfolio_path)
        ws = wb.active
        headers = [cell.value for cell in ws[1]]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            h = dict(zip(headers, row))
            code = str(h.get('股票代码', '')).strip()
            if not code:
                continue
            holdings.append({
                'code': code,
                'name': str(h.get('股票名称', '')).strip(),
                'cost': float(h.get('买入价格', 0) or 0),
                'shares': int(h.get('持仓数量', 0) or 0),
                'buy_date': h.get('买入日期', ''),
                'stop_loss': float(h.get('止损价', 0) or 0),
                'take_profit': float(h.get('止盈价', 0) or 0),
                'position_pct': float(h.get('仓位%', 0) or 0),
                'type': str(h.get('持仓类型', '')).strip(),
                'notes': str(h.get('备注', '')).strip(),
            })
        wb.close()

    # 加载资金
    capital_path = os.path.join(os.path.dirname(__file__), '../../config/capital.xlsx')
    capital = {'total': 500000, 'cash': 50000}
    if os.path.exists(capital_path):
        wb = openpyxl.load_workbook(capital_path)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            key = str(row[0]).strip()
            val = row[1]
            if key == '总资金':
                capital['total'] = float(val)
            elif key == '可用资金':
                capital['cash'] = float(val)
        wb.close()

    return holdings, capital


def fetch_tech_data(tc: TushareClient, code: str) -> Dict:
    """获取技术指标数据用于效率评分"""
    import numpy as np
    tech = {}

    try:
        # ETF用fund_daily，股票用get_kline
        if is_etf(code):
            ts_code = get_etf_tscode(code)
            raw = tc.get_fund_daily(ts_code=ts_code)
            if raw is None or len(raw) < 30:
                return tech
            raw = raw.sort_values('trade_date').reset_index(drop=True)
            close = raw['close'].values.astype(float)
            vol = raw['vol'].values.astype(float) if 'vol' in raw.columns else None
        else:
            df = tc.get_kline(code, freq='daily', start=None, end=None)
            if df is None or len(df) < 30:
                return tech
            close = df['close'].values.astype(float)
            vol = df['volume'].values.astype(float) if 'volume' in df.columns else None

        # 简单趋势判断
        ma5 = close[-5:].mean()
        ma20 = close[-20:].mean()
        ma60 = close[-min(60, len(close)):].mean() if len(close) >= 30 else ma20
        threshold = 0.01  # 1%阈值，避免微小差异被判为多/空

        if ma5 > ma20 * (1 + threshold) and ma20 > ma60 * (1 + threshold):
            tech['ma_alignment'] = 'bullish'
        elif ma5 < ma20 * (1 - threshold) and ma20 < ma60 * (1 - threshold):
            tech['ma_alignment'] = 'bearish'
        else:
            tech['ma_alignment'] = 'mixed'

        # 周线趋势（用日线模拟：每5天取样）
        if len(close) >= 50:
            weekly_close = close[::5]
            if len(weekly_close) >= 10:
                wma5 = weekly_close[-5:].mean()
                wma10 = weekly_close[-min(10, len(weekly_close)):].mean()
                if wma5 > wma10 * (1 + threshold):
                    tech['weekly_trend'] = 'bullish'
                elif wma5 < wma10 * (1 - threshold):
                    tech['weekly_trend'] = 'bearish'
                else:
                    tech['weekly_trend'] = 'neutral'

        # 日线趋势
        if ma5 > ma20 * (1 + threshold):
            tech['daily_trend'] = 'bullish'
        elif ma5 < ma20 * (1 - threshold):
            tech['daily_trend'] = 'bearish'
        else:
            tech['daily_trend'] = 'neutral'

        # RSI
        deltas = close[-15:] - close[-16:-1] if len(close) >= 16 else close[1:] - close[:-1]
        gains = deltas[deltas > 0].mean() if (deltas > 0).any() else 0
        losses = (-deltas[deltas < 0]).mean() if (deltas < 0).any() else 0.001
        rs = gains / losses if losses > 0 else 100
        tech['daily_rsi'] = 100 - (100 / (1 + rs))

        # MACD柱状线变化方向
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        dif = ema12 - ema26
        if len(dif) >= 2:
            tech['daily_macd_hist_change'] = dif[-1] - dif[-2]
        else:
            tech['daily_macd_hist_change'] = 0

        # 量比
        if vol is not None and len(vol) >= 10:
            avg_vol = vol[-10:].mean()
            tech['volume_ratio'] = vol[-1] / avg_vol if avg_vol > 0 else 1.0
        else:
            tech['volume_ratio'] = 1.0

    except Exception as e:
        print(f"  获取{code}技术数据失败: {e}")

    return tech


def _ema(data, period):
    """计算EMA"""
    import numpy as np
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    multiplier = 2.0 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = data[i] * multiplier + ema[i-1] * (1 - multiplier)
    return ema


def main():
    parser = argparse.ArgumentParser(description='资金效率分析')
    parser.add_argument('symbol', nargs='?', help='股票代码（可选，不填则分析全组合）')
    args = parser.parse_args()

    analyzer = CapitalEfficiencyAnalyzer()
    tc = TushareClient()
    holdings, capital = load_portfolio()

    if not holdings:
        print("未找到持仓数据，请确认 config/portfolio.xlsx 存在")
        return

    # 获取实时价格（股票从daily_basic，ETF从fund_daily）
    print("获取实时行情...", end='', flush=True)
    price_map = {}

    # 股票价格
    realtime = tc.get_daily_basic()
    if realtime is not None:
        for _, r in realtime.iterrows():
            code = str(r.get('ts_code', '')).replace('.SZ', '').replace('.SH', '').replace('.BJ', '')
            price_map[code] = r.get('close', 0)

    # ETF价格（daily_basic不含ETF，使用get_fund_daily）
    etf_codes = [h['code'] for h in holdings if is_etf(h['code'])]
    for code in etf_codes:
        if code not in price_map or price_map[code] == 0:
            try:
                ts_code = get_etf_tscode(code)
                df = tc.get_fund_daily(ts_code=ts_code)
                if df is not None and len(df) > 0:
                    df = df.sort_values('trade_date')
                    price_map[code] = float(df.iloc[-1]['close'])
            except Exception as e:
                print(f"  获取ETF {code}价格失败: {e}")

    print(" 完成")

    # 构建持仓列表
    positions = []
    for h in holdings:
        code = h['code']
        cost = h['cost']
        shares = h['shares']
        price = price_map.get(code, cost)
        market_value = price * shares
        pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0

        holding_data = {
            'code': code,
            'name': h['name'],
            'cost': cost,
            'price': price,
            'shares': shares,
            'market_value': market_value,
            'buy_date': h.get('buy_date', ''),
            'pnl_pct': pnl_pct,
        }

        # 获取技术数据
        tech = fetch_tech_data(tc, code)

        result = analyzer.score_position(holding_data, tech)
        result['cost'] = cost
        result['shares'] = shares
        result['price'] = price
        positions.append(result)

    if args.symbol:
        # 单仓分析
        target = args.symbol.strip()
        matched = [p for p in positions if p['code'] == target]
        if not matched:
            print(f"未找到持仓 {target}")
            return
        p = matched[0]
        g = p['grade_info']
        print(f"\n{'='*60}")
        print(f"💰 资金效率分析: {p['name']}({p['code']})")
        print(f"{'='*60}")
        print(f"  持仓: {p['shares']}股@{p.get('cost',0):.3f} → 现价{p.get('price',0):.3f}")
        print(f"  盈亏: {p['pnl_pct']:+.1f}% | 市值: {p['market_value']:,.0f}元")
        print(f"  效率分: {g['color']}{p['grade']}({p['score']}) | {g['label']}")
        print(f"  趋势{p['trend_score']} + 动能{p['momentum_score']} + 回报{p['return_score']}")
        print(f"  年化资金回报率: {p['annualized_return_pct']:+.1f}%")
        print(f"  持仓天数: {p['holding_days']}天")
        print(f"  建议: {g['action']}")
        if p['is_dead_capital']:
            print(f"\n  ⚠️ 死资本警告：{p['market_value']:,.0f}元被锁在低效亏损仓位")
            print(f"  → 如果止损释放，可用于更高效率标的")
    else:
        # 组合总览
        total = capital.get('total', 500000)
        cash = capital.get('cash', 30000)
        result = analyzer.score_portfolio(positions, total, cash)
        print(f"\n{analyzer.format_portfolio_report(result)}")

        # 详细列表
        print(f"\n{'─'*60}")
        print(f"{'持仓':<20} {'效率':>6} {'等级':>4} {'市值':>10} {'盈亏':>8} {'年化回报':>8}")
        print(f"{'─'*60}")
        for p in result['positions_sorted']:
            name = p['name'][:8]
            gi = p['grade_info']
            print(f"{name:<20} {p['score']:>6.1f} {gi['color']}{p['grade']:>3} "
                  f"{p['market_value']:>10,.0f} {p['pnl_pct']:>+7.1f}% "
                  f"{p['annualized_return_pct']:>+7.1f}%")


if __name__ == '__main__':
    main()
