#!/usr/bin/env python3
"""V3+通道融合回测 — 全市场逐日回测。

核心思路：在v3评分（二阶导+位置+量能）基础上，
融入通道位置、B级信号（通道下半+f''拐点+企稳）、底背离，
看融合评分的胜率是否有提升。

⚠️ 防未来数据铁律：
  1. Savgol滤波器 shift(window//2)
  2. 信号检测只用 close[:i+1]
  3. 收益从次日开盘价计算
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import argrelextrema, savgol_filter

SG_W = 21
SG_P = 3
PROJECT = Path(__file__).parent.parent.parent


def safe_savgol_deriv(ln_p, window=SG_W, poly=SG_P):
    """计算因果f'和f''(2026-09-10: 原中心化+shift滞后, 改用权威 causal_savgol pos=0)."""
    w = min(window, len(ln_p) // 4 * 2 + 1)
    if w < 5:
        return np.zeros(len(ln_p)), np.zeros(len(ln_p))
    from scripts.core.derivatives import causal_savgol
    return causal_savgol(ln_p, w, poly, deriv=1), causal_savgol(ln_p, w, poly, deriv=2)


def compute_channel(close, high, low, n):
    """简单布林通道（MA20±2std），返回位置百分比。"""
    if n < 20:
        return 50.0, False
    ma = np.mean(close[n-20:n])
    std = np.std(close[n-20:n])
    if std < 1e-8:
        return 50.0, False
    upper = ma + 2 * std
    lower = ma - 2 * std
    pct = (close[n-1] - lower) / (upper - lower) * 100
    return pct, True


def compute_v3_score(f1, f2, close, high, volume, n):
    """计算v3核心分(0-10)。

    CI = 0.40×f''趋势 + 0.20×量能 + 0.40×位置
    """
    if n < 60:
        return 0.0

    # 趋势分(0-4)
    f1v = f1[n-1]
    f2v = f2[n-1]
    f2_rising = f2[n-1] > f2[max(0, n-3)]

    if f1v > 0.05 and f2v > 0 and f2_rising:
        trend = 4.0
    elif f1v > 0.05 and f2v > 0:
        trend = 3.5
    elif f1v > 0.02 and f2v > 0:
        trend = 3.0
    elif f1v > 0 and f2v > 0:
        trend = 2.5
    elif f1v > 0:
        trend = 2.0
    elif f2v > 0 and f2_rising:
        trend = 1.5  # 拐点
    elif f2v > 0:
        trend = 1.0
    else:
        trend = 0.0

    # 位置分(0-4): 距180日高点
    lookback = min(180, n)
    dh180 = (close[n-1] / np.max(high[n-lookback:n]) - 1) * 100
    if dh180 < -30:
        pos = 4.0
    elif dh180 < -20:
        pos = 3.0
    elif dh180 < -10:
        pos = 2.0
    elif dh180 < -5:
        pos = 1.0
    else:
        pos = 0.0

    # 量能分(0-2)
    vol_score = 0.0
    if volume is not None and n >= 20:
        avg_vol = np.mean(volume[n-20:n])
        if avg_vol > 0:
            vol_ratio = volume[n-1] / avg_vol
            if vol_ratio > 1.5:
                vol_score = 2.0
            elif vol_ratio > 1.2:
                vol_score = 1.5
            elif vol_ratio > 1.0:
                vol_score = 1.0
            else:
                vol_score = 0.5

    # CI加权
    score = 0.40 * trend + 0.40 * pos + 0.20 * vol_score
    return score


def detect_daily_divergence(dif, low, n):
    """日线底背离（只用前n个数据）。"""
    if n < 40:
        return False
    sub_dif = dif[:n]
    sub_low = low[:n]
    lows_idx = argrelextrema(sub_low, np.less, order=10)[0]
    if len(lows_idx) >= 2:
        l1, l2 = lows_idx[-2], lows_idx[-1]
        if sub_low[l2] < sub_low[l1] and sub_dif[l2] > sub_dif[l1]:
            return True
    return False


def backtest_single(close, low, high, opens, volume, min_days=120):
    """单只股票逐日回测，返回逐日分数和收益。"""
    n = len(close)
    if n < min_days + 25:
        return None

    # 预计算MACD
    s = pd.Series(close)
    e12 = s.ewm(span=12).mean().values
    e26 = s.ewm(span=26).mean().values
    dif = e12 - e26

    # 预计算二阶导(防泄漏)
    ln_p = np.log(close)
    f1, f2 = safe_savgol_deriv(ln_p)

    results = []
    for i in range(min_days, n - 20):
        ni = i + 1  # 用前ni个数据
        price = close[i]

        # v3核心分
        v3 = compute_v3_score(f1, f2, close, high, volume, ni)

        # 通道位置
        ch_pct, ch_valid = compute_channel(close, high, low, ni)

        # B级信号: 通道下半(<35%) + f''拐点 + 阳线企稳
        b_signal = False
        if ch_valid and ch_pct < 35:
            f2_turn = any(
                f2[i-k] > 0 and f2[max(0, i-k-1)] <= 0
                for k in range(min(3, i - min_days))
            )
            stable = close[i] > opens[i]  # 阳线
            if f2_turn and stable:
                b_signal = True

        # 底背离
        divergence = detect_daily_divergence(dif, low, ni)

        # 融合加分
        bonus = 0.0
        if b_signal:
            bonus += 1.5
        if divergence:
            bonus += 0.5
        if ch_valid and ch_pct < 20:
            bonus += 0.5  # 通道极底部额外加分

        total = v3 + bonus

        # 收益（次日开盘价入场）
        entry = opens[i + 1]
        ret5 = (close[min(i+5, n-1)] / entry - 1) * 100
        ret10 = (close[min(i+10, n-1)] / entry - 1) * 100
        ret20 = (close[i + 20] / entry - 1) * 100

        results.append({
            'v3': v3,
            'total': total,
            'ch_pct': ch_pct if ch_valid else np.nan,
            'b_signal': b_signal,
            'divergence': divergence,
            'ret5': ret5,
            'ret10': ret10,
            'ret20': ret20,
        })

    return results


def main():
    data_dir = PROJECT / 'data' / 'store' / 'kline'

    total_stocks = 0
    valid_stocks = 0
    all_results = []

    for code_dir in sorted(data_dir.iterdir()):
        if not code_dir.is_dir():
            continue
        code = code_dir.name
        if len(code) != 6 or not code.isdigit():
            continue
        if code.startswith('688'):
            continue

        f = code_dir / 'D.parquet'
        if not f.exists():
            continue
        total_stocks += 1

        try:
            df = pd.read_parquet(f)
            if len(df) < 140:
                continue

            # 兼容多种列名格式
            if 'trade_date' in df.columns:
                dates = df['trade_date'].values
            elif 'date' in df.columns:
                dates = df['date'].values
            else:
                continue

            close = df['close'].values.astype(float)
            low = df['low'].values.astype(float)
            high = df['high'].values.astype(float)
            opens = df['open'].values.astype(float)
            volume = df['vol'].values.astype(float) if 'vol' in df.columns else (
                df['volume'].values.astype(float) if 'volume' in df.columns else None)

            if str(dates[0]) > str(dates[-1]):
                close = close[::-1]; low = low[::-1]; high = high[::-1]
                opens = opens[::-1]; dates = dates[::-1]
                if volume is not None:
                    volume = volume[::-1]

            result = backtest_single(close, low, high, opens, volume)
            if result is None:
                continue

            valid_stocks += 1
            for r in result:
                r['code'] = code
            all_results.extend(result)

        except Exception as e:
            continue

    print(f"全市场回测: {valid_stocks}/{total_stocks}只")
    print(f"总信号数: {len(all_results)}")
    print()

    if not all_results:
        print("无结果")
        return

    df = pd.DataFrame(all_results)

    # ====== 分组统计 ======
    print("=" * 75)
    print("📊 V3+通道融合 — 全市场回测报告")
    print("=" * 75)

    # 1. 纯v3评分分组
    print("\n── 纯V3评分 vs 20日胜率 ──")
    for threshold in [5, 6, 7, 8]:
        sub = df[df['v3'] >= threshold]
        if len(sub) == 0:
            continue
        wr5 = (sub['ret5'] > 0).mean() * 100
        wr10 = (sub['ret10'] > 0).mean() * 100
        wr20 = (sub['ret20'] > 0).mean() * 100
        avg20 = sub['ret20'].mean()
        print(f"  v3≥{threshold}: {len(sub):>6d}次 | 5日{wr5:.0f}% 10日{wr10:.0f}% 20日{wr20:.0f}% | 平均{avg20:+.1f}%")

    # 2. 融合分(total=v3+bonus)分组
    print("\n── V3+通道融合分 vs 20日胜率 ──")
    for threshold in [5, 6, 7, 8, 9, 10]:
        sub = df[df['total'] >= threshold]
        if len(sub) == 0:
            continue
        wr5 = (sub['ret5'] > 0).mean() * 100
        wr10 = (sub['ret10'] > 0).mean() * 100
        wr20 = (sub['ret20'] > 0).mean() * 100
        avg20 = sub['ret20'].mean()
        print(f"  融合≥{threshold}: {len(sub):>6d}次 | 5日{wr5:.0f}% 10日{wr10:.0f}% 20日{wr20:.0f}% | 平均{avg20:+.1f}%")

    # 3. B级信号单独看
    print("\n── B级信号(通道下半+f''拐点+阳线) ──")
    b_df = df[df['b_signal']]
    if len(b_df) > 0:
        for threshold in [0, 3, 5]:
            sub = b_df[b_df['v3'] >= threshold]
            if len(sub) == 0:
                continue
            wr20 = (sub['ret20'] > 0).mean() * 100
            avg20 = sub['ret20'].mean()
            print(f"  B信号+v3≥{threshold}: {len(sub):>5d}次 | 20日胜率{wr20:.0f}% | 平均{avg20:+.1f}%")
    else:
        print("  无B级信号")

    # 4. 底背离单独看
    print("\n── 底背离信号 ──")
    div_df = df[df['divergence']]
    if len(div_df) > 0:
        for threshold in [0, 3, 5]:
            sub = div_df[div_df['v3'] >= threshold]
            if len(sub) == 0:
                continue
            wr20 = (sub['ret20'] > 0).mean() * 100
            avg20 = sub['ret20'].mean()
            print(f"  底背离+v3≥{threshold}: {len(sub):>5d}次 | 20日胜率{wr20:.0f}% | 平均{avg20:+.1f}%")
    else:
        print("  无底背离信号")

    # 5. 通道位置分组
    print("\n── 通道位置 vs 胜率 ──")
    ch_valid = df[~df['ch_pct'].isna()]
    for lo, hi, label in [(0, 20, '极底部(0-20%)'), (20, 40, '下半(20-40%)'),
                          (40, 60, '中部(40-60%)'), (60, 80, '上半(60-80%)'),
                          (80, 101, '极顶部(80-100%)')]:
        sub = ch_valid[(ch_valid['ch_pct'] >= lo) & (ch_valid['ch_pct'] < hi)]
        if len(sub) == 0:
            continue
        wr20 = (sub['ret20'] > 0).mean() * 100
        avg20 = sub['ret20'].mean()
        print(f"  {label}: {len(sub):>5d}次 | 20日胜率{wr20:.0f}% | 平均{avg20:+.1f}%")

    # 6. 黄金组合: v3>=5 + 通道底部 + B级/底背离
    print("\n── 黄金组合 ──")
    combos = [
        ("v3≥5+通道<30%", (df['v3'] >= 5) & (df['ch_pct'] < 30)),
        ("v3≥5+通道<30%+B级", (df['v3'] >= 5) & (df['ch_pct'] < 30) & df['b_signal']),
        ("v3≥5+通道<30%+底背离", (df['v3'] >= 5) & (df['ch_pct'] < 30) & df['divergence']),
        ("v3≥6+通道<30%", (df['v3'] >= 6) & (df['ch_pct'] < 30)),
        ("v3≥6+通道<30%+B级", (df['v3'] >= 6) & (df['ch_pct'] < 30) & df['b_signal']),
        ("v3≥6+通道<40%", (df['v3'] >= 6) & (df['ch_pct'] < 40)),
        ("v3≥7+通道<40%", (df['v3'] >= 7) & (df['ch_pct'] < 40)),
    ]
    for label, mask in combos:
        sub = df[mask & ~df['ch_pct'].isna()]
        if len(sub) == 0:
            continue
        wr20 = (sub['ret20'] > 0).mean() * 100
        avg20 = sub['ret20'].mean()
        wr5 = (sub['ret5'] > 0).mean() * 100
        wr10 = (sub['ret10'] > 0).mean() * 100
        print(f"  {label}: {len(sub):>5d}次 | 5日{wr5:.0f}% 10日{wr10:.0f}% 20日{wr20:.0f}% | {avg20:+.1f}%")

    # 7. 保存完整数据
    out_path = PROJECT / 'data' / 'v3_channel_backtest.csv'
    df.to_csv(out_path, index=False)
    print(f"\n完整数据已保存: {out_path}")
    print(f"({len(df)}行 × {len(df.columns)}列)")


if __name__ == '__main__':
    main()
