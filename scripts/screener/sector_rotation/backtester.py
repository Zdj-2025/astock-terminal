"""
科技板块轮动监控系统 - 回测引擎 v1
逐日滑窗计算信号，评估1年历史有效性
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from scripts.screener.sector_rotation.config import SECTOR_DEFS, SECTORS, init_sectors
from scripts.screener.sector_rotation.scorer import score_sector
from scripts.screener.sector_rotation.signal_detector import detect_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 回测数据缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.backtest_cache')


def _rate_limit():
    time.sleep(0.35)  # ~170次/分钟


def fetch_backtest_data(start_date: str, end_date: str, force: bool = False):
    """
    批量拉取所有板块的回测数据并缓存
    每个板块：ETF日线 + 个股资金流（用于汇总）
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    pro = ts.pro_api()

    if not SECTORS:
        init_sectors()

    sectors = SECTORS if SECTORS else SECTOR_DEFS

    for name, config in sectors.items():
        cache_file = os.path.join(CACHE_DIR, f"{name}_{start_date}_{end_date}.pkl")
        if os.path.exists(cache_file) and not force:
            logger.info(f"  {name}: 使用缓存")
            continue

        logger.info(f"  拉取 {name}...")
        data = {}

        # ETF日线
        etf_code = config.get('etf_primary', '')
        if etf_code:
            _rate_limit()
            df = pro.fund_daily(ts_code=etf_code, start_date=start_date, end_date=end_date)
            data['etf_daily'] = df.sort_values('trade_date').reset_index(drop=True) if df is not None and not df.empty else pd.DataFrame()
        else:
            data['etf_daily'] = pd.DataFrame()

        # 个股资金流（取权重前5只，减少API调用）
        stock_codes = list(config.get('stocks', {}).keys())[:5]
        if not stock_codes:
            stock_codes = config.get('seed_stocks', [])[:5]

        all_flows = []
        all_stocks = {}
        for code in stock_codes:
            _rate_limit()
            df_flow = pro.moneyflow(ts_code=code, start_date=start_date, end_date=end_date)
            if df_flow is not None and not df_flow.empty:
                all_flows.append(df_flow.sort_values('trade_date').reset_index(drop=True))

            _rate_limit()
            df_stock = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df_stock is not None and not df_stock.empty:
                all_stocks[code] = df_stock.sort_values('trade_date').reset_index(drop=True)

        # 汇总板块资金流
        if all_flows:
            combined = pd.concat(all_flows, ignore_index=True)
            sector_flow = combined.groupby('trade_date').agg({
                'net_mf_amount': 'sum',
                'buy_elg_amount': 'sum',
                'sell_elg_amount': 'sum',
                'buy_lg_amount': 'sum',
                'sell_lg_amount': 'sum',
                'buy_md_amount': 'sum',
                'sell_md_amount': 'sum',
                'buy_sm_amount': 'sum',
                'sell_sm_amount': 'sum',
            }).reset_index().sort_values('trade_date')
            sector_flow['elg_net_amount'] = sector_flow['buy_elg_amount'] - sector_flow['sell_elg_amount']
            sector_flow['total_amount'] = (
                sector_flow['buy_elg_amount'] + sector_flow['sell_elg_amount'] +
                sector_flow['buy_lg_amount'] + sector_flow['sell_lg_amount'] +
                sector_flow['buy_md_amount'] + sector_flow['sell_md_amount'] +
                sector_flow['buy_sm_amount'] + sector_flow['sell_sm_amount']
            ) / 2
            data['sector_moneyflow'] = sector_flow
        else:
            data['sector_moneyflow'] = pd.DataFrame()

        data['stocks_daily'] = all_stocks

        # 保存缓存
        pd.to_pickle(data, cache_file)
        logger.info(f"  {name}: 已缓存 ({len(data.get('etf_daily', []))}条ETF, {len(all_flows)}只个股资金流)")

    logger.info("数据拉取完成")


def load_backtest_data(start_date: str, end_date: str):
    """加载缓存数据"""
    sectors = SECTORS if SECTORS else SECTOR_DEFS
    all_data = {}
    for name in sectors:
        cache_file = os.path.join(CACHE_DIR, f"{name}_{start_date}_{end_date}.pkl")
        if os.path.exists(cache_file):
            all_data[name] = pd.read_pickle(cache_file)
        else:
            logger.warning(f"缓存不存在: {name}")
    return all_data


def run_backtest(start_date: str, end_date: str, forward_days: list = [5, 10, 20]):
    """
    逐日滑窗回测
    对于每个交易日：
      1. 用截至当日的数据计算评分和信号
      2. 记录信号
      3. 计算未来N天的涨幅（用于验证信号有效性）

    Returns:
        trades: list of dicts with signal info and forward returns
    """
    logger.info("加载回测数据...")
    all_data = load_backtest_data(start_date, end_date)

    # 获取交易日历（用任一ETF的日期）
    sample_etf = None
    for name, data in all_data.items():
        if not data.get('etf_daily', pd.DataFrame()).empty:
            sample_etf = data['etf_daily']
            break

    if sample_etf is None:
        logger.error("无ETF数据")
        return []

    trade_dates = sorted(sample_etf['trade_date'].unique())
    # 去掉前30天（需要预热期）
    trade_dates = [d for d in trade_dates if d >= start_date]
    if len(trade_dates) < 30:
        logger.error("交易日不足30天")
        return []

    logger.info(f"回测期间: {trade_dates[0]} ~ {trade_dates[-1]}, 共{len(trade_dates)}个交易日")

    # 逐日计算
    all_signals = []
    for i, date in enumerate(trade_dates):
        if i < 29:  # 至少需要30天数据
            continue

        # 截至当日的数据窗口
        for name, data in all_data.items():
            etf = data.get('etf_daily', pd.DataFrame())
            flow = data.get('sector_moneyflow', pd.DataFrame())
            stocks = data.get('stocks_daily', {})

            if etf.empty:
                continue

            # 截至当日的数据
            etf_cut = etf[etf['trade_date'] <= date].copy()
            flow_cut = flow[flow['trade_date'] <= date].copy() if not flow.empty else pd.DataFrame()
            stocks_cut = {}
            for code, sdf in stocks.items():
                cut = sdf[sdf['trade_date'] <= date].copy()
                if not cut.empty:
                    stocks_cut[code] = cut

            # 评分
            score_result = score_sector(etf_cut, flow_cut, stocks_cut)
            signal_result = detect_signal(score_result)

            # 未来N天涨幅
            future_returns = {}
            for fd in forward_days:
                future_idx = trade_dates.index(date) + fd if date in trade_dates else -1
                if future_idx < len(trade_dates):
                    future_date = trade_dates[future_idx]
                    future_row = etf[etf['trade_date'] == future_date]
                    if not future_row.empty and not etf_cut.empty:
                        future_close = future_row['close'].iloc[0]
                        current_close = etf_cut['close'].iloc[-1]
                        future_returns[f'ret_{fd}d'] = round((future_close / current_close - 1) * 100, 2)
                    else:
                        future_returns[f'ret_{fd}d'] = None
                else:
                    future_returns[f'ret_{fd}d'] = None

            all_signals.append({
                'date': date,
                'sector': name,
                'signal': signal_result['signal'],
                'score': score_result['total_score'],
                'pv_score': score_result['price_volume']['score'],
                'fund_score': score_result['fund_flow']['score'],
                'tech_score': score_result['technicals']['score'],
                'breadth_score': score_result['breadth']['score'],
                **future_returns,
            })

        if (i + 1) % 20 == 0:
            logger.info(f"  回测进度: {i+1}/{len(trade_dates)}")

    logger.info(f"回测完成，共{len(all_signals)}条信号记录")
    return all_signals


def analyze_backtest_results(signals: list, forward_days: list = [5, 10, 20]):
    """分析回测结果"""
    if not signals:
        print("无回测数据")
        return

    df = pd.DataFrame(signals)

    print("=" * 80)
    print("📊 板块轮动回测报告")
    print("=" * 80)

    # 1. 各信号的胜率和平均收益
    print("\n📌 一、信号有效性分析")
    print(f"{'信号':<8} {'次数':<8} ", end="")
    for fd in forward_days:
        print(f"{'胜率'+str(fd)+'d':<12} {'平均'+str(fd)+'d':<12}", end="")
    print()

    signal_types = ['关注', '潜伏', '启动', '加速', '过热', '冷却', '观望']
    for sig in signal_types:
        subset = df[df['signal'] == sig]
        if subset.empty:
            continue
        print(f"{sig:<8} {len(subset):<8} ", end="")
        for fd in forward_days:
            col = f'ret_{fd}d'
            valid = subset[col].dropna()
            if len(valid) > 0:
                win_rate = (valid > 0).sum() / len(valid) * 100
                avg_ret = valid.mean()
                print(f"{win_rate:.1f}%{'':>5} {avg_ret:+.2f}%{'':>5}", end="")
            else:
                print(f"{'N/A':<12} {'N/A':<12}", end="")
        print()

    # 2. 各板块信号效果
    print("\n📌 二、各板块启动/潜伏信号效果")
    actionable = df[df['signal'].isin(['关注', '潜伏', '启动'])]
    if actionable.empty:
        print("  无可操作信号")
    else:
        for sector in actionable['sector'].unique():
            sub = actionable[actionable['sector'] == sector]
            print(f"\n  {sector} ({len(sub)}次信号):")
            for fd in forward_days:
                col = f'ret_{fd}d'
                valid = sub[col].dropna()
                if len(valid) > 0:
                    win_rate = (valid > 0).sum() / len(valid) * 100
                    avg_ret = valid.mean()
                    print(f"    {fd}日: 胜率{win_rate:.0f}% 平均{avg_ret:+.1f}%", end="")
                    # 按信号类型细分
                    for sig in ['关注', '潜伏', '启动']:
                        sig_sub = sub[sub['signal'] == sig][col].dropna()
                        if len(sig_sub) > 0:
                            print(f" | {sig}({len(sig_sub)}){(sig_sub>0).sum()/len(sig_sub)*100:.0f}%", end="")
                    print()

    # 3. 信号升级的效果
    print("\n📌 三、信号连续性分析")
    for sector in df['sector'].unique():
        sub = df[df['sector'] == sector].sort_values('date')
        # 找从观望→关注/潜伏/启动的转换点
        signals_seq = sub['signal'].tolist()
        dates_seq = sub['date'].tolist()
        upgrades = []
        for j in range(1, len(signals_seq)):
            prev = signals_seq[j - 1]
            curr = signals_seq[j]
            if prev in ('观望', '冷却') and curr in ('关注', '潜伏', '启动'):
                upgrades.append((dates_seq[j], curr))

        if upgrades:
            # 看升级后10日涨幅
            upgrade_rets = []
            for d, sig in upgrades:
                row = sub[sub['date'] == d]
                if not row.empty and pd.notna(row.iloc[0].get('ret_10d')):
                    upgrade_rets.append(row.iloc[0]['ret_10d'])

            if upgrade_rets:
                wr = sum(1 for r in upgrade_rets if r > 0) / len(upgrade_rets) * 100
                avg = np.mean(upgrade_rets)
                print(f"  {sector}: {len(upgrades)}次升级, 10日胜率{wr:.0f}%, 平均{avg:+.1f}%")

    # 4. 与基准对比
    print("\n📌 四、策略 vs 基准")
    # 基准：等权持有10个ETF
    baseline_rets = {}
    for fd in forward_days:
        col = f'ret_{fd}d'
        all_valid = df[df['signal'] == '观望'][col].dropna()
        signal_valid = df[df['signal'].isin(['关注', '潜伏', '启动'])][col].dropna()
        if len(all_valid) > 0 and len(signal_valid) > 0:
            print(f"  {fd}日: 观望期平均{all_valid.mean():+.1f}% vs 信号期平均{signal_valid.mean():+.1f}%")

    # 5. 资金动量IC测试
    print("\n📌 五、评分因子IC分析")
    for dim, col_name in [('总分', 'score'), ('价格量能', 'pv_score'), ('资金流', 'fund_score'),
                           ('技术', 'tech_score'), ('广度', 'breadth_score')]:
        for fd in [10]:
            ret_col = f'ret_{fd}d'
            valid = df[[col_name, ret_col]].dropna()
            if len(valid) > 10:
                ic = valid[col_name].corr(valid[ret_col], method='spearman')
                print(f"  {dim} vs {fd}日收益 IC={ic:+.3f} (n={len(valid)})")

    print("\n" + "=" * 80)

    # 保存详细结果
    sd = signals[0]['date'] if signals else 'unknown'
    ed = signals[-1]['date'] if signals else 'unknown'
    result_file = os.path.join(CACHE_DIR, f'backtest_results_{sd}_{ed}.csv')
    df.to_csv(result_file, index=False)
    logger.info(f"详细结果已保存: {result_file}")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='20250601')
    parser.add_argument('--end', default='20260531')
    parser.add_argument('--fetch-only', action='store_true', help='只拉数据不回测')
    parser.add_argument('--force', action='store_true', help='强制重新拉数据')
    args = parser.parse_args()

    logger.info(f"回测区间: {args.start} ~ {args.end}")

    # 1. 拉数据
    fetch_backtest_data(args.start, args.end, force=args.force)

    if args.fetch_only:
        logger.info("仅拉数据模式，退出")
        sys.exit(0)

    # 2. 回测
    signals = run_backtest(args.start, args.end)

    # 3. 分析
    if signals:
        analyze_backtest_results(signals)
