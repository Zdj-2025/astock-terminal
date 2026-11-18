#!/usr/bin/env python3
"""持仓深度分析 - 2025/05/20"""
import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scripts.core.tushare_client import TushareClient
from scripts.screener.trend_scanner import TrendScanner
from scripts.screener.gain_momentum_scorer import GainMomentumScorer

client = TushareClient()
scanner = TrendScanner(client)
scorer = GainMomentumScorer(client)

# 持仓定义
holdings = [
    {'name': '数据港', 'code': '603881', 'shares': 3260, 'cost_broker': 27.802, 'cost_base': 21.902, 'price': 40.79, 'type': 'stock'},
    {'name': '半导体ETF', 'code': '512480', 'shares': 20000, 'cost_broker': 1.589, 'cost_base': 1.589, 'price': 2.227, 'type': 'etf'},
    {'name': '芯片ETF', 'code': '512760', 'shares': 36000, 'cost_broker': 0.297, 'cost_base': 0.297, 'price': 1.201, 'type': 'etf'},
    {'name': '君禾股份', 'code': '603617', 'shares': 5000, 'cost_broker': 7.648, 'cost_base': 7.648, 'price': 7.71, 'type': 'stock'},
    {'name': '联化科技', 'code': '002250', 'shares': 2000, 'cost_broker': 15.125, 'cost_base': 15.125, 'price': 15.06, 'type': 'stock'},
    {'name': 'AI智能ETF', 'code': '159819', 'shares': 10000, 'cost_broker': 0.210, 'cost_base': 0.210, 'price': 2.002, 'type': 'etf'},
    {'name': '兰卫医学', 'code': '301060', 'shares': 1000, 'cost_broker': 10.815, 'cost_base': 10.815, 'price': 10.770, 'type': 'stock'},
]

TOTAL_CAPITAL = 487686

def get_kline_data(code, htype='stock'):
    """获取日线数据"""
    try:
        if htype == 'etf':
            # ETF用fund_daily
            df = client.pro.fund_daily(ts_code=code, start_date='20250101', end_date='20260520')
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date').reset_index(drop=True)
                df = df.rename(columns={'pre_close': 'pre_close', 'change': 'change', 'pct_chg': 'pct_chg'})
                return df
            else:
                # fallback to daily
                df = client.pro.daily(ts_code=code, start_date='20250101', end_date='20260520')
                if df is not None and len(df) > 0:
                    df = df.sort_values('trade_date').reset_index(drop=True)
                return df
        else:
            df = client.get_kline(code, freq='D', start='20250101', adj='qfq')
            return df
    except Exception as e:
        print(f"  获取{code}日线失败: {e}")
        return None

def get_weekly_data(code, htype='stock'):
    """获取周线数据"""
    try:
        if htype == 'etf':
            df = client.pro.fund_daily(ts_code=code, start_date='20241001', end_date='20260520')
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date').reset_index(drop=True)
                # 重采样为周线
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date')
                weekly = df.resample('W').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'
                }).dropna()
                weekly = weekly.reset_index()
                return weekly
            return None
        else:
            df = client.get_kline(code, freq='W', start='20241001', adj='qfq')
            return df
    except Exception as e:
        print(f"  获取{code}周线失败: {e}")
        return None

def calc_ma(close_arr, period):
    """计算均线"""
    if len(close_arr) < period:
        return None
    return float(np.mean(close_arr[-period:]))

def analyze_position_safe(scanner, close_arr):
    """位置安全检查"""
    try:
        safe, reason = scanner.check_position_safe(np.array(close_arr))
        return safe, reason
    except Exception as e:
        return None, f"检查失败: {e}"

def analyze_single(h):
    """分析单只持仓"""
    result = {}
    code = h['code']
    name = h['name']
    htype = h['type']
    
    print(f"\n{'='*60}")
    print(f"分析: {name} ({code})")
    print(f"{'='*60}")
    
    # 1. 日线数据
    df = get_kline_data(code, htype)
    if df is None or len(df) == 0:
        print(f"  ⚠️ 无法获取日线数据，跳过")
        return None
    
    print(f"  日线数据: {len(df)}条, {df.iloc[0].get('trade_date', 'N/A')} ~ {df.iloc[-1].get('trade_date', 'N/A')}")
    
    # 标准化列名
    close_col = 'close'
    if close_col not in df.columns:
        print(f"  可用列: {list(df.columns)}")
        return None
    
    close_arr = df[close_col].values.astype(float)
    
    # 2. 趋势v2评分
    trend_result = None
    try:
        if htype == 'stock':
            trend_result = scanner._scan_single(df, code, include_weekly=False)
        else:
            trend_result = scanner._scan_single(df, code, include_weekly=False)
        
        if trend_result:
            print(f"  趋势v2: grade={trend_result.get('trend_grade', 'N/A')}, score={trend_result.get('trend_v2_score', 'N/A')}")
            if 'trend_v2_factors' in trend_result:
                print(f"  因子: {trend_result['trend_v2_factors']}")
        result['trend'] = trend_result
    except Exception as e:
        print(f"  趋势分析失败: {e}")
        result['trend'] = None
    
    # 3. 增幅/动量评分
    momentum_result = None
    try:
        if htype == 'stock':
            momentum_result = scorer.score(code)
            if momentum_result:
                print(f"  动量: score={momentum_result.get('total_score', momentum_result.get('score', 'N/A'))}")
                if isinstance(momentum_result, dict):
                    for k, v in momentum_result.items():
                        if k not in ('detail', 'factors', 'data'):
                            print(f"    {k}: {v}")
        else:
            print(f"  动量: ETF跳过")
        result['momentum'] = momentum_result
    except Exception as e:
        print(f"  动量分析失败: {e}")
        result['momentum'] = None
    
    # 4. 位置安全
    safe, reason = analyze_position_safe(scanner, close_arr)
    print(f"  位置安全: {'✅安全' if safe else '⚠️不安全'} - {reason}")
    result['position_safe'] = (safe, reason)
    
    # 5. 均线和支撑压力
    mas = {}
    for period in [5, 10, 20, 60, 120, 250]:
        ma_val = calc_ma(close_arr, period)
        if ma_val is not None:
            mas[f'MA{period}'] = round(ma_val, 3)
            current = h['price']
            diff_pct = (current - ma_val) / ma_val * 100
            print(f"  MA{period}: {ma_val:.3f} (偏离{diff_pct:+.1f}%)")
    result['mas'] = mas
    
    # 6. 周线趋势
    df_w = get_weekly_data(code, htype)
    weekly_trend = None
    if df_w is not None and len(df_w) > 10:
        try:
            w_close = df_w['close'].values.astype(float)
            # 简单判断周线趋势
            w_ma5 = calc_ma(w_close, 5)
            w_ma10 = calc_ma(w_close, 10)
            w_ma20 = calc_ma(w_close, 20)
            last_close = w_close[-1]
            
            if w_ma5 and w_ma10 and w_ma20:
                if last_close > w_ma5 > w_ma10 > w_ma20:
                    weekly_trend = '多头排列'
                elif last_close > w_ma5 and w_ma5 > w_ma10:
                    weekly_trend = '偏多'
                elif last_close < w_ma5 < w_ma10:
                    weekly_trend = '偏空'
                elif last_close < w_ma5 < w_ma10 < w_ma20:
                    weekly_trend = '空头排列'
                else:
                    weekly_trend = '震荡'
                print(f"  周线趋势: {weekly_trend}")
                print(f"  周线MA5={w_ma5:.3f} MA10={w_ma10:.3f} MA20={w_ma20:.3f}")
        except Exception as e:
            print(f"  周线分析失败: {e}")
    else:
        print(f"  周线数据不足: {len(df_w) if df_w is not None else 0}条")
    result['weekly_trend'] = weekly_trend
    
    # 7. RSI
    try:
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
        print(f"  RSI(14): {rsi:.1f}")
        result['rsi'] = round(rsi, 1)
    except:
        result['rsi'] = None
    
    # 8. MACD
    try:
        ema12 = pd.Series(close_arr).ewm(span=12).mean()
        ema26 = pd.Series(close_arr).ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        macd_hist = (dif - dea) * 2
        result['macd'] = {
            'dif': round(float(dif.iloc[-1]), 3),
            'dea': round(float(dea.iloc[-1]), 3),
            'hist': round(float(macd_hist.iloc[-1]), 3),
            'prev_hist': round(float(macd_hist.iloc[-2]), 3) if len(macd_hist) > 1 else 0
        }
        macd_trend = '金叉' if result['macd']['hist'] > 0 and result['macd']['prev_hist'] <= 0 else \
                     '死叉' if result['macd']['hist'] < 0 and result['macd']['prev_hist'] >= 0 else \
                     '红柱扩大' if result['macd']['hist'] > result['macd']['prev_hist'] > 0 else \
                     '红柱缩小' if 0 < result['macd']['hist'] < result['macd']['prev_hist'] else \
                     '绿柱扩大' if result['macd']['hist'] < result['macd']['prev_hist'] < 0 else \
                     '绿柱缩小'
        print(f"  MACD: DIF={result['macd']['dif']}, DEA={result['macd']['dea']}, 柱={result['macd']['hist']} ({macd_trend})")
        result['macd_trend'] = macd_trend
    except Exception as e:
        result['macd'] = None
        result['macd_trend'] = None
    
    # 9. 成交量分析
    try:
        vol_col = 'vol' if 'vol' in df.columns else 'volume'
        if vol_col in df.columns:
            vol_arr = df[vol_col].values.astype(float)
            vol_ma5 = calc_ma(vol_arr, 5)
            vol_ma20 = calc_ma(vol_arr, 20)
            last_vol = vol_arr[-1]
            vol_ratio = last_vol / vol_ma5 if vol_ma5 and vol_ma5 > 0 else 0
            print(f"  量比(vs MA5): {vol_ratio:.2f}, vol_ma5={vol_ma5:.0f}, vol_ma20={vol_ma20:.0f}")
            result['volume'] = {'ratio': round(vol_ratio, 2), 'ma5': vol_ma5, 'ma20': vol_ma20}
    except:
        pass
    
    # 10. 涨跌幅统计
    try:
        recent_5d = (close_arr[-1] - close_arr[-6]) / close_arr[-6] * 100 if len(close_arr) > 5 else 0
        recent_20d = (close_arr[-1] - close_arr[-21]) / close_arr[-21] * 100 if len(close_arr) > 20 else 0
        print(f"  近5日涨跌: {recent_5d:+.2f}%, 近20日涨跌: {recent_20d:+.2f}%")
        result['change_5d'] = round(recent_5d, 2)
        result['change_20d'] = round(recent_20d, 2)
    except:
        pass
    
    result['holding'] = h
    result['close_arr'] = close_arr
    return result

# 主分析
print("="*60)
print("持仓深度分析 - 2026/05/20")
print("="*60)
print(f"总资金池: {TOTAL_CAPITAL:,.0f}")

all_results = []
for h in holdings:
    r = analyze_single(h)
    if r:
        all_results.append(r)

# 仓位占比分析
print("\n" + "="*60)
print("仓位占比分析")
print("="*60)
total_position = 0
for r in all_results:
    h = r['holding']
    mv = h['shares'] * h['price']
    pct = mv / TOTAL_CAPITAL * 100
    pnl_pct = (h['price'] - h['cost_broker']) / h['cost_broker'] * 100
    pnl_amt = (h['price'] - h['cost_broker']) * h['shares']
    total_position += mv
    print(f"  {h['name']}({h['code']}): 市值{mv:,.0f} 占比{pct:.1f}% 浮盈{pnl_amt:+,.0f}({pnl_pct:+.1f}%)")

cash = TOTAL_CAPITAL - total_position
print(f"\n  总持仓: {total_position:,.0f} ({total_position/TOTAL_CAPITAL*100:.1f}%)")
print(f"  现金: {cash:,.0f} ({cash/TOTAL_CAPITAL*100:.1f}%)")

# 做T分析
print("\n" + "="*60)
print("做T持仓分析")
print("="*60)
t_cost = 41.125
t_shares = 1000  # 500+500
t_price = 40.79
t_pnl = (t_price - t_cost) * t_shares
print(f"  数据港做T: 1000股 @ {t_cost} -> 现价{t_price}, 浮亏{t_pnl:+,.0f}")
print(f"  建议: 做T部分浮亏不大，可等反弹后出T，或补T降成本")

print("\n✅ 分析完成")
