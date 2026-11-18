#!/usr/bin/env python3
"""弹簧起涨全量滚动回测 v3 — 大盘择时+止损优化。

改进：
1. 全量滚动回测（每月月初扫一次，共5个月=5轮，扩大样本）
2. 大盘择时：上证指数日线趋势过滤（MA20方向+MACD状态）
3. 止损策略回测（固定-3%止损 vs 不止损）
4. 严格无未来数据：所有指标只用到scan_date

用法:
  PYTHONPATH=. python scripts/research/spring_backtest_v3.py
  PYTHONPATH=. python scripts/research/spring_backtest_v3.py --top 5 --stop-loss 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============ 指标计算（与spring_scanner.py完全一致） ============

def calc_squeeze_ratio(close: pd.Series, period: int = 20, lookback: int = 60) -> float:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    width = ((ma + 2 * std) - (ma - 2 * std)) / ma * 100
    avg_width = width.rolling(lookback).mean()
    ratio = width / avg_width
    return ratio.iloc[-1] if len(ratio) > 0 else np.nan


def calc_macd_red_ratio(close: pd.Series, lookback=20) -> float:
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    recent = macd.tail(lookback)
    if len(recent) == 0:
        return 0
    return (recent > 0).sum() / len(recent)


def calc_vol_surge(volume: pd.Series, lookback=20) -> float:
    if len(volume) < lookback + 5:
        return 0
    return volume.tail(5).mean() / max(volume.tail(lookback).mean(), 1)


def check_red_fading(close: pd.Series) -> bool:
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    if len(macd) < 8:
        return False
    recent_3 = macd.iloc[-3:].values
    prev_3 = macd.iloc[-6:-3].values
    if all(r > 0 for r in recent_3) and all(r > 0 for r in prev_3):
        avg_r = recent_3.mean()
        avg_p = prev_3.mean()
        if avg_p > 0 and avg_r < avg_p * 0.6:
            return True
    return False


def calc_spring_score(red_ratio, vol_surge, squeeze, above_ma20, ret5, red_fading) -> int:
    s = 0
    if red_ratio >= 1.0: s += 30
    elif red_ratio > 0.9: s += 25
    elif red_ratio > 0.8: s += 20
    elif red_ratio > 0.7: s += 10

    if vol_surge > 2.0: s += 30
    elif vol_surge > 1.7: s += 25
    elif vol_surge > 1.5: s += 20
    elif vol_surge > 1.3: s += 12

    if squeeze < 0.4: s += 10
    elif squeeze < 0.6: s += 7
    elif squeeze < 0.8: s += 3

    if above_ma20: s += 5
    if 0 < ret5 <= 2: s += 5
    elif 2 < ret5 <= 3: s += 2
    if red_fading: s -= 15
    return s


# ============ 大盘择时 ============

def get_market_regime(index_df: pd.DataFrame, scan_idx: int) -> str:
    """判断大盘趋势状态。

    Returns:
        'bull': 多头（MA20向上 + MACD金叉）→ 适合买入
        'neutral': 中性（MA20走平或MACD不明显）→ 谨慎
        'bear': 空头（MA20向下 + MACD死叉）→ 不开仓
    """
    sub = index_df.iloc[:scan_idx + 1]
    if len(sub) < 30:
        return 'neutral'

    close = sub['close']

    # MA20方向
    ma20 = close.rolling(20).mean()
    if len(ma20) < 5 or ma20.isna().any():
        return 'neutral'
    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100

    # MACD状态
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=9, adjust=False).mean()

    dif_now = dif.iloc[-1]
    dea_now = dea.iloc[-1]
    dif_prev = dif.iloc[-2] if len(dif) > 1 else dif_now

    golden = dif_now > dea_now  # 金叉
    dead = dif_now < dea_now    # 死叉
    ma20_up = ma20_slope > 0.3  # MA20向上（阈值：5天涨0.3%以上）
    ma20_down = ma20_slope < -0.3

    if ma20_up and golden:
        return 'bull'
    elif ma20_down and dead:
        return 'bear'
    else:
        return 'neutral'


# ============ 核心扫描（无未来数据） ============

def scan_at_date(code: str, df: pd.DataFrame, scan_idx: int) -> Optional[dict]:
    sub = df.iloc[:scan_idx + 1]
    if len(sub) < 80:
        return None

    close = sub['close']
    volume = sub.get('volume', sub.get('vol', None))
    if volume is None:
        return None

    # 压缩比
    squeeze = calc_squeeze_ratio(close)
    if pd.isna(squeeze) or squeeze > 1.2:
        return None

    # 红柱占比
    red_ratio = calc_macd_red_ratio(close)
    if red_ratio < 0.7:
        return None

    # 量比
    vol_surge = calc_vol_surge(volume)
    if pd.isna(vol_surge) or vol_surge < 1.3:
        return None

    # 涨幅约束
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
    if ret20 > 10.0 or ret5 > 3.0:
        return None

    # MA20方向确认
    ma20 = close.rolling(20).mean().iloc[-1]
    if pd.isna(ma20):
        return None
    above_ma20 = close.iloc[-1] > ma20
    if not above_ma20:
        return None

    # 红柱衰竭
    red_fading = check_red_fading(close)

    # 评分
    score = calc_spring_score(red_ratio, vol_surge, squeeze, above_ma20, ret5, red_fading)

    return {
        'code': code,
        'scan_close': round(close.iloc[-1], 2),
        'spring_score': score,
        'macd_red_pct': round(red_ratio * 100, 1),
        'vol_surge': round(vol_surge, 2) if not pd.isna(vol_surge) else 0,
        'squeeze_ratio': round(squeeze, 3),
        'ret5': round(ret5, 2),
        'ret20': round(ret20, 2),
        'above_ma20': above_ma20,
        'red_fading': red_fading,
    }


# ============ 回测主逻辑 ============

def run_backtest(
    data_root: str = 'data/store',
    top_n: int = 10,
    stop_loss_pct: float = 3.0,
    min_score: int = 50,
    market_filter: str = 'all',  # 'all', 'no_bear', 'bull_only'
    hold_days: list = None,
) -> pd.DataFrame:

    if hold_days is None:
        hold_days = [1, 3, 5, 7, 10]

    store = Path(data_root)

    # 加载上证指数
    index_path = store / 'kline' / '000001' / 'D.parquet'
    if not index_path.exists():
        print('❌ 找不到上证指数数据(000001)')
        return pd.DataFrame()
    index_df = pd.read_parquet(index_path)
    if 'date' in index_df.columns:
        index_df = index_df.sort_values('date').reset_index(drop=True)
        index_df['dt'] = pd.to_datetime(index_df['date'])
    elif 'trade_date' in index_df.columns:
        index_df = index_df.sort_values('trade_date').reset_index(drop=True)
        index_df['dt'] = pd.to_datetime(index_df['trade_date'])
    else:
        print('❌ 上证指数日期列无法识别')
        return pd.DataFrame()

    # 股票池
    codes = sorted([
        d.name for d in (store / 'kline').iterdir()
        if d.is_dir() and (d / 'D.parquet').exists() and d.name != '000001'
    ])

    # 扫描日期：每月2个时间点（月初+月中），1-5月共10轮
    scan_dates = []
    for month in range(1, 6):
        scan_dates.append(f'2026-{month:02d}-08')   # 月初
        scan_dates.append(f'2026-{month:02d}-22')   # 月中

    print(f'📅 扫描日期: {scan_dates}')
    print(f'📊 持有天数: {hold_days}')
    print(f'🏥 股票池: {len(codes)}只')
    print(f'🎯 每轮取Top{top_n}, 最低{min_score}分')
    print(f'🛡️ 止损: -{stop_loss_pct}%')
    print(f'🌍 大盘过滤: {market_filter}')
    print()

    all_results = []

    for scan_date_str in scan_dates:
        scan_ts = pd.Timestamp(scan_date_str)
        print(f'--- {scan_date_str} ---')

        # 大盘择时
        idx_mask = index_df['dt'] <= scan_ts
        if not idx_mask.any():
            print('  ⚠️ 无指数数据，跳过')
            continue
        idx_scan = idx_mask[idx_mask].index[-1]
        regime = get_market_regime(index_df, idx_scan)

        if market_filter == 'no_bear' and regime == 'bear':
            print(f'  🐻 大盘空头，跳过')
            continue
        if market_filter == 'bull_only' and regime != 'bull':
            print(f'  😐 大盘{regime}，跳过(仅做多)')
            continue

        print(f'  🌍 大盘: {regime}')

        # 扫描
        date_results = []
        for code in codes:
            try:
                dp = store / 'kline' / code / 'D.parquet'
                df = pd.read_parquet(dp)
                if len(df) < 100:
                    continue

                if 'date' in df.columns:
                    df = df.sort_values('date').reset_index(drop=True)
                    df['dt'] = pd.to_datetime(df['date'])
                elif 'trade_date' in df.columns:
                    df = df.sort_values('trade_date').reset_index(drop=True)
                    df['dt'] = pd.to_datetime(df['trade_date'])
                else:
                    continue

                mask = df['dt'] <= scan_ts
                if not mask.any():
                    continue
                scan_idx = mask[mask].index[-1]
                max_hold = max(hold_days)
                if scan_idx + max_hold >= len(df):
                    continue

                result = scan_at_date(code, df, scan_idx)
                if result is None:
                    continue
                if result['spring_score'] < min_score:
                    continue

                # 持有收益
                scan_close = df['close'].iloc[scan_idx]
                max_loss = 0.0  # 最大回撤
                stopped = False
                stop_day = None

                for d in range(1, max_hold + 1):
                    future_close = df['close'].iloc[scan_idx + d]
                    ret = (future_close / scan_close - 1) * 100

                    # 追踪最大回撤
                    if ret < max_loss:
                        max_loss = ret

                    # 止损检查（盘中最低价触及止损线也算）
                    if not stopped and ret <= -stop_loss_pct:
                        stopped = True
                        stop_day = d

                    if d in hold_days:
                        # 如果已止损，用止损日的收益
                        if stopped and stop_day and stop_day <= d:
                            result[f'ret_{d}d'] = round(-stop_loss_pct, 2)
                            result[f'ret_{d}d_raw'] = round(ret, 2)  # 原始收益也记录
                        else:
                            result[f'ret_{d}d'] = round(ret, 2)

                result['max_drawdown'] = round(max_loss, 2)
                result['stopped'] = stopped
                result['stop_day'] = stop_day
                result['scan_date'] = scan_date_str
                result['market_regime'] = regime
                date_results.append(result)

            except Exception:
                continue

        # 排序取TopN
        if date_results:
            date_df = pd.DataFrame(date_results).sort_values('spring_score', ascending=False)
            top = date_df.head(top_n)
            print(f'  扫出{len(date_df)}只，取Top{top_n}')
            for _, r in top.iterrows():
                rets = ' '.join([f'{d}日={r.get(f"ret_{d}d", "N/A"):>+6.1f}%' for d in hold_days])
                fade = '⚠️' if r['red_fading'] else ''
                stop_tag = f'🛑第{r["stop_day"]}日' if r['stopped'] else ''
                print(f'  {r["code"]} 分={r["spring_score"]:.0f} 红柱={r["macd_red_pct"]:.0f}% 量比={r["vol_surge"]:.1f} {fade} | {rets} {stop_tag}')
            all_results.extend(top.to_dict('records'))
        else:
            print('  无信号')
        print()

    # ============ 汇总统计 ============
    if not all_results:
        print('📭 无回测结果')
        return pd.DataFrame()

    df_all = pd.DataFrame(all_results)

    print('=' * 80)
    print('📊 回测汇总')
    print('=' * 80)
    print(f'总信号数: {len(df_all)}, 扫描轮次: {len(scan_dates)}')
    print()

    # 分层统计（按弹簧分）
    for threshold_label, threshold in [('≥60分', 60), ('≥50分(推荐)', 50), ('全部', 0)]:
        subset = df_all[df_all['spring_score'] >= threshold] if threshold > 0 else df_all
        n = len(subset)
        if n == 0:
            continue
        print(f'--- {threshold_label} (n={n}) ---')

        for d in hold_days:
            col = f'ret_{d}d'
            vals = subset[col].dropna()
            if len(vals) == 0:
                continue
            win = (vals > 0).sum() / len(vals) * 100
            avg = vals.mean()
            med = vals.median()
            print(f'  {d:>2}日: 胜率={win:.0f}% 均值={avg:+.1f}% 中位={med:+.1f}% 最大亏损={vals.min():+.1f}%')
        print()

    # 止损效果
    stopped = df_all[df_all['stopped'] == True]
    not_stopped = df_all[df_all['stopped'] == False]
    print(f'--- 止损统计 (-{stop_loss_pct}%) ---')
    print(f'  触发止损: {len(stopped)}/{len(df_all)} ({len(stopped)/len(df_all)*100:.0f}%)')
    if len(stopped) > 0:
        # 止损后的原始收益（如果不止损会怎样）
        for d in hold_days:
            raw_col = f'ret_{d}d_raw'
            stop_col = f'ret_{d}d'
            if raw_col in stopped.columns and stop_col in stopped.columns:
                raw_vals = stopped[raw_col].dropna()
                stop_vals = stopped[stop_col].dropna()
                if len(raw_vals) > 0 and len(stop_vals) > 0:
                    print(f'  {d}日止损组: 止损后={stop_vals.mean():+.1f}% vs 不止损={raw_vals.mean():+.1f}%')
    if len(not_stopped) > 0:
        for d in hold_days:
            col = f'ret_{d}d'
            vals = not_stopped[col].dropna()
            if len(vals) > 0:
                print(f'  {d}日未止损组(n={len(vals)}): 均值={vals.mean():+.1f}% 胜率={( vals > 0).sum()/len(vals)*100:.0f}%')
    print()

    # 大盘环境对比
    print('--- 大盘环境对比 ---')
    for regime in ['bull', 'neutral', 'bear']:
        sub = df_all[df_all['market_regime'] == regime]
        if len(sub) == 0:
            continue
        for d in [1, 3, 5]:
            col = f'ret_{d}d'
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f'  {regime:>8} (n={len(sub)}): {d}日胜率={(vals>0).sum()/len(vals)*100:.0f}% 均值={vals.mean():+.1f}%')
    print()

    # 最优持有天数
    print('--- 最优持有天数(≥50分) ---')
    best_sub = df_all[df_all['spring_score'] >= 50]
    best_day = None
    best_sharpe = -999
    for d in hold_days:
        col = f'ret_{d}d'
        vals = best_sub[col].dropna()
        if len(vals) < 3:
            continue
        avg = vals.mean()
        std = vals.std()
        sharpe = avg / std if std > 0 else 0
        win = (vals > 0).sum() / len(vals) * 100
        print(f'  {d}日: 胜率={win:.0f}% 均值={avg:+.2f}% 波动={std:.2f}% 夏普比={sharpe:.2f}')
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_day = d
    if best_day:
        print(f'  🏆 最优: {best_day}日 (夏普比={best_sharpe:.2f})')

    # 保存
    out_path = Path('data/backtest_results')
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f'spring_v3_rolling_{market_filter}.csv'
    df_all.to_csv(out_file, index=False)
    print(f'\n📁 已保存: {out_file}')

    return df_all


def main():
    parser = argparse.ArgumentParser(description='弹簧起涨v3全量滚动回测')
    parser.add_argument('--top', type=int, default=10, help='每轮TopN')
    parser.add_argument('--stop-loss', type=float, default=3.0, help='止损百分比')
    parser.add_argument('--min-score', type=int, default=50, help='最低弹簧分')
    parser.add_argument('--market-filter', default='no_bear', choices=['all', 'no_bear', 'bull_only'],
                        help='大盘过滤: all=不过滤, no_bear=避开空头, bull_only=仅做多')
    parser.add_argument('--data-root', default='data/store')
    args = parser.parse_args()

    run_backtest(
        data_root=args.data_root,
        top_n=args.top,
        stop_loss_pct=args.stop_loss,
        min_score=args.min_score,
        market_filter=args.market_filter,
    )


if __name__ == '__main__':
    main()
