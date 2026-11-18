#!/usr/bin/env python3
"""弹簧起涨扫描器回测 — 防过拟合验证。

随机选取1-5月的某一天，用v2逻辑扫描，然后看1/3/7/15个交易日的收益率。
严格避免未来数据：所有计算只用到scan_date为止的数据。

用法:
  PYTHONPATH=. python scripts/research/spring_backtest.py
  PYTHONPATH=. python scripts/research/spring_backtest.py --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calc_bollinger_width(close: pd.Series, period: int = 20) -> pd.Series:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    width = ((ma + 2 * std) - (ma - 2 * std)) / ma * 100
    return width


def calc_squeeze_ratio(close: pd.Series, period: int = 20, lookback: int = 60) -> pd.Series:
    width = calc_bollinger_width(close, period)
    avg_width = width.rolling(lookback).mean()
    return width / avg_width


def calc_macd_red_ratio(close: pd.Series, fast=12, slow=26, signal=9, lookback=20) -> float:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    recent = macd.tail(lookback)
    if len(recent) == 0:
        return 0
    return (recent > 0).sum() / len(recent)


def calc_vol_surge(volume: pd.Series, lookback=20) -> float:
    if len(volume) < lookback + 5:
        return 0
    recent_5 = volume.tail(5).mean()
    avg_20 = volume.tail(lookback).mean()
    if avg_20 == 0:
        return 0
    return recent_5 / avg_20


def check_red_fading(close: pd.Series) -> bool:
    """检查MACD红柱是否在衰竭（近3根 vs 前3根，衰减>40%）"""
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
        avg_recent = recent_3.mean()
        avg_prev = prev_3.mean()
        if avg_prev > 0 and avg_recent < avg_prev * 0.6:
            return True
    return False


def scan_at_date(
    code: str,
    df: pd.DataFrame,
    scan_idx: int,
) -> dict | None:
    """在指定索引位置执行弹簧扫描逻辑（无未来数据）。

    Args:
        code: 股票代码
        df: 完整K线数据（但只用到scan_idx及之前）
        scan_idx: 扫描日期对应的行索引

    Returns:
        扫描结果dict 或 None（不满足条件）
    """
    # 只取到scan_idx的数据（严格无未来数据）
    sub = df.iloc[:scan_idx + 1].copy()
    if len(sub) < 80:
        return None

    close = sub['close']
    volume = sub['volume'] if 'volume' in sub.columns else sub['vol']
    high = sub['high']
    low = sub['low']

    # 1. 压缩比
    squeeze = calc_squeeze_ratio(close)
    if len(squeeze) == 0 or squeeze.isna().all():
        return None
    current_squeeze = squeeze.iloc[-1]
    if pd.isna(current_squeeze) or current_squeeze > 1.2:
        return None

    # 2. MACD红柱占比
    red_ratio = calc_macd_red_ratio(close, lookback=20)
    if red_ratio < 0.7:
        return None

    # 3. 量比
    vol_surge = calc_vol_surge(volume)
    if vol_surge < 1.3:
        return None

    # 4. 涨幅约束
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
    if ret20 > 10.0 or ret5 > 3.0:
        return None

    # 5. 方向确认
    ma20 = close.rolling(20).mean()
    current_ma20 = ma20.iloc[-1]
    if pd.isna(current_ma20):
        return None
    above_ma20 = close.iloc[-1] > current_ma20
    if not above_ma20:
        return None

    # 6. 红柱衰竭检测
    red_fading = check_red_fading(close)

    # 7. 弹簧评分（v2权重）
    spring_score = 0

    if red_ratio >= 1.0:
        spring_score += 30
    elif red_ratio > 0.9:
        spring_score += 25
    elif red_ratio > 0.8:
        spring_score += 20
    elif red_ratio > 0.7:
        spring_score += 10

    if vol_surge > 2.0:
        spring_score += 30
    elif vol_surge > 1.7:
        spring_score += 25
    elif vol_surge > 1.5:
        spring_score += 20
    elif vol_surge > 1.3:
        spring_score += 12

    if current_squeeze < 0.4:
        spring_score += 10
    elif current_squeeze < 0.6:
        spring_score += 7
    elif current_squeeze < 0.8:
        spring_score += 3

    if above_ma20:
        spring_score += 5

    if 0 < ret5 <= 2:
        spring_score += 5
    elif 2 < ret5 <= 3:
        spring_score += 2

    if red_fading:
        spring_score -= 15

    ma20_dist = (close.iloc[-1] / current_ma20 - 1) * 100

    return {
        'code': code,
        'scan_close': round(close.iloc[-1], 2),
        'spring_score': spring_score,
        'squeeze_ratio': round(current_squeeze, 3),
        'macd_red_pct': round(red_ratio * 100, 1),
        'vol_surge': round(vol_surge, 2),
        'ret5': round(ret5, 2),
        'ret20': round(ret20, 2),
        'above_ma20': above_ma20,
        'ma20_dist': round(ma20_dist, 1),
        'red_fading': red_fading,
    }


def run_backtest(
    data_root: str = 'data/store',
    seed: int = 42,
    hold_days: list = None,
) -> pd.DataFrame:
    """执行回测。每月随机选一天扫描，追踪持有收益。"""

    if hold_days is None:
        hold_days = [1, 3, 7, 15]

    rng = np.random.RandomState(seed)
    store = Path(data_root)

    codes = sorted([
        d.name for d in (store / 'kline').iterdir()
        if d.is_dir() and (d / 'D.parquet').exists()
    ])

    # 生成每月的随机日期（1月到5月）
    months = [(2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5)]
    scan_dates = []
    for year, month in months:
        # 每月随机选一个交易日（选月中附近的日期，避免月初月末异常）
        day = rng.randint(5, 25)  # 5-24号
        scan_dates.append(f'{year}-{month:02d}-{day:02d}')

    print(f'📅 回测扫描日期: {scan_dates}')
    print(f'📊 持有天数: {hold_days}')
    print(f'🏥 股票池: {len(codes)}只')
    print()

    all_results = []

    for scan_date_str in scan_dates:
        scan_date = pd.Timestamp(scan_date_str)
        print(f'--- 扫描日: {scan_date_str} ---')

        date_results = []

        for code in codes:
            try:
                dp = store / 'kline' / code / 'D.parquet'
                df = pd.read_parquet(dp)
                if len(df) < 100:
                    continue

                # 标准化日期列
                if 'date' in df.columns:
                    df = df.sort_values('date').reset_index(drop=True)
                    df['dt'] = pd.to_datetime(df['date'])
                elif 'trade_date' in df.columns:
                    df = df.sort_values('trade_date').reset_index(drop=True)
                    df['dt'] = pd.to_datetime(df['trade_date'])
                else:
                    continue

                # 找到scan_date对应的行索引（<=scan_date的最大日期）
                mask = df['dt'] <= scan_date
                if not mask.any():
                    continue
                scan_idx = mask[mask].index[-1]

                # 确保有足够的后续数据计算持有收益
                total_rows = len(df)
                max_hold = max(hold_days)
                if scan_idx + max_hold >= total_rows:
                    continue

                # 执行扫描（无未来数据）
                result = scan_at_date(code, df, scan_idx)
                if result is None:
                    continue

                # 计算持有收益（未来数据，但这是验证用的，不影响扫描逻辑）
                scan_close = df['close'].iloc[scan_idx]
                for d in hold_days:
                    future_idx = scan_idx + d
                    if future_idx < total_rows:
                        future_close = df['close'].iloc[future_idx]
                        ret = (future_close / scan_close - 1) * 100
                        result[f'ret_{d}d'] = round(ret, 2)
                    else:
                        result[f'ret_{d}d'] = None

                result['scan_date'] = scan_date_str
                date_results.append(result)

            except Exception as e:
                continue

        if date_results:
            date_df = pd.DataFrame(date_results)
            date_df = date_df.sort_values('spring_score', ascending=False)
            top_n = date_df.head(10)  # 每次取Top10
            print(f'  扫出 {len(date_df)} 只信号，取Top10')
            for _, r in top_n.iterrows():
                rets = ' '.join([f'{d}日={r.get(f"ret_{d}d", "N/A"):>+6.1f}%' for d in hold_days])
                fade = '⚠️' if r['red_fading'] else ''
                print(f'  {r["code"]} 分={r["spring_score"]:.0f} 红柱={r["macd_red_pct"]:.0f}% 量比={r["vol_surge"]:.1f} ret5={r["ret5"]:+.1f}% {fade} | {rets}')
            all_results.extend(top_n.to_dict('records'))
        else:
            print('  无信号')

        print()

    # 汇总统计
    if not all_results:
        print('📭 回测无结果')
        return pd.DataFrame()

    df_all = pd.DataFrame(all_results)

    print('=' * 70)
    print('📊 回测汇总')
    print('=' * 70)
    print(f'总信号数: {len(df_all)}')
    print()

    # 按弹簧分分层统计
    for threshold in [60, 50, 40, 0]:
        subset = df_all[df_all['spring_score'] >= threshold] if threshold > 0 else df_all
        if len(subset) == 0:
            continue
        label = f'≥{threshold}分' if threshold > 0 else '全部'
        print(f'--- 弹簧分 {label} (n={len(subset)}) ---')

        for d in hold_days:
            col = f'ret_{d}d'
            vals = subset[col].dropna()
            if len(vals) == 0:
                continue
            win_rate = (vals > 0).sum() / len(vals) * 100
            avg_ret = vals.mean()
            median_ret = vals.median()
            max_ret = vals.max()
            min_ret = vals.min()
            print(f'  {d:>2}日: 胜率={win_rate:.0f}% 均值={avg_ret:+.1f}% 中位={median_ret:+.1f}% 最好={max_ret:+.1f}% 最差={min_ret:+.1f}%')
        print()

    # 红柱衰竭 vs 不衰竭对比
    fading = df_all[df_all['red_fading'] == True]
    not_fading = df_all[df_all['red_fading'] == False]
    if len(fading) > 0 and len(not_fading) > 0:
        print('--- 红柱衰竭 vs 不衰竭 ---')
        for label, subset in [('⚠️衰竭', fading), ('✅正常', not_fading)]:
            rets = [subset[f'ret_{d}d'].dropna().mean() for d in hold_days]
            win = [(subset[f'ret_{d}d'].dropna() > 0).sum() / max(len(subset[f'ret_{d}d'].dropna()), 1) * 100 for d in hold_days]
            ret_str = ' '.join([f'{d}日={r:+.1f}%(胜{w:.0f}%)' for d, r, w in zip(hold_days, rets, win)])
            print(f'  {label} (n={len(subset)}): {ret_str}')
        print()

    # 保存详细结果
    out_path = Path('data/backtest_results')
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f'spring_v2_backtest_{seed}.csv'
    df_all.to_csv(out_file, index=False)
    print(f'📁 详细结果已保存: {out_file}')

    return df_all


def main():
    parser = argparse.ArgumentParser(description='弹簧起涨扫描器v2回测')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--data-root', default='data/store')
    args = parser.parse_args()

    run_backtest(data_root=args.data_root, seed=args.seed)


if __name__ == '__main__':
    main()
