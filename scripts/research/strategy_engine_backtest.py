#!/usr/bin/env python3
"""策略引擎回测 — 对比V3原始评分 vs 新策略层评分。

目标：
1. 底部反转策略：新策略 vs V3原始底部反转(L2/L3) 胜率和收益对比
2. 主升浪策略：新策略 vs V3原始主升浪 胜率和收益对比
3. 确认新策略引擎不劣于原有系统

股票池：1800只，每日采样
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

DATA_ROOT = Path("data/store")
STOCK_POOL = Path("data/stock_pool.csv")
SG_WINDOW = 21
SG_POLY = 3


def load_stock_pool():
    df = pd.read_csv(STOCK_POOL, dtype={'code': str})
    return df['code'].astype(str).str.zfill(6).tolist()


def load_kline(code):
    path = DATA_ROOT / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    return df


def score_bottom_new(f1, f2, dh180, vol_ratio, has_accel_confirm=False):
    """新底部反转评分(策略引擎)。"""
    score = 0.0
    # f''(0-3)
    if f2 > 0.05: score += 3.0
    elif f2 > 0.02: score += 2.5
    elif f2 > 0.01: score += 2.0
    elif f2 > 0: score += 1.0
    # f'(0-2)
    if f1 >= 0.1: score += 2.0
    elif f1 >= 0: score += 1.5
    elif f1 >= -0.1: score += 0.5
    # 放量(0-2)
    if vol_ratio > 1.3: score += 2.0
    elif vol_ratio > 1.0: score += 1.5
    elif vol_ratio > 0.8: score += 0.5
    # 位置(0-2)
    if dh180 < -30: score += 2.0
    elif dh180 < -20: score += 1.5
    elif dh180 < -15: score += 1.0
    elif dh180 < -10: score += 0.5
    # 分钟(0-1)
    if has_accel_confirm: score += 1.0
    return round(score, 1)


def score_uptrend_new(f1, f2, dh180, vol_ratio, above_ma20):
    """新主升浪评分(策略引擎)。"""
    score = 0.0
    # f'(0-3)
    if f1 > 0.3: score += 3.0
    elif f1 > 0.15: score += 2.5
    elif f1 > 0.05: score += 2.0
    elif f1 > 0: score += 1.0
    # f''(0-2)
    if f2 > 0.05: score += 2.0
    elif f2 > 0.02: score += 1.5
    elif f2 > 0.01: score += 1.0
    elif f2 > 0: score += 0.5
    # 放量(0-2)
    if vol_ratio > 1.3: score += 2.0
    elif vol_ratio > 1.0: score += 1.5
    elif vol_ratio > 0.8: score += 0.5
    # MA20(0-1)
    if above_ma20: score += 1.0
    # 位置(0-2)
    if dh180 < -15: score += 2.0
    elif dh180 < -5: score += 1.5
    elif dh180 < 0: score += 1.0
    elif dh180 < 5: score += 0.5
    return round(score, 1)


def run():
    print("=" * 70)
    print("策略引擎回测 — 新策略层 vs V3原始")
    print("=" * 70)

    codes = load_stock_pool()
    print(f"股票池: {len(codes)} 只\n")
    print("[1/2] 采集数据...")

    records = []
    for i, code in enumerate(codes):
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(codes)}")
        df = load_kline(code)
        if df is None or len(df) < 280:
            continue
        close = df['close'].values.astype(float)
        vc = 'vol' if 'vol' in df.columns else 'volume'
        if vc not in df.columns:
            continue
        volume = df[vc].values.astype(float)
        n = len(close)

        ln_p = np.log(close)
        f1 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=1) * 100
        f2 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=2) * 10000

        for t in range(250, n - 20):
            c = close[:t+1]
            v = volume[:t+1]
            if t + 20 >= n:
                break
            fwd_ret = (close[t+20] / close[t] - 1) * 100

            dh180 = (c[-1] / np.max(c[-180:]) - 1) * 100 if t >= 180 else 0
            vol5 = np.mean(v[-5:])
            vol20 = np.mean(v[-20:]) if len(v) >= 20 else vol5
            vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0
            ma20 = np.mean(c[-20:])
            above_ma20 = c[-1] > ma20

            records.append({
                'code': code,
                'f1': f1[t],
                'f2': f2[t],
                'dh180': dh180,
                'vol_ratio': vol_ratio,
                'above_ma20': above_ma20,
                'fwd_ret20': fwd_ret,
            })

    df = pd.DataFrame(records)
    print(f"  总样本: {len(df):,}\n")

    # ── 新策略评分 ──
    print("[2/2] 策略对比")
    print("=" * 80)

    # ── 底部反转对比 ──
    print("\n" + "─" * 80)
    print("底部反转：新策略 vs V3原始L2/L3")
    print("─" * 80)

    # 新策略评分
    df['new_bottom'] = df.apply(
        lambda r: score_bottom_new(r['f1'], r['f2'], r['dh180'], r['vol_ratio']),
        axis=1
    )

    # V3原始L2条件
    old_l2 = df[(df['f2'] > 0.01) & (df['f1'] >= 0) & (df['dh180'] < -20)]
    # V3原始L3条件
    old_l3 = df[(df['f2'] > 0.01) & (df['f1'] >= 0) & (df['dh180'] < -25) & (df['vol_ratio'] > 1.0)]
    # 新策略命中(match条件: f2>0.01 + f1>=0 + dh180<-15)
    new_match = df[(df['f2'] > 0.01) & (df['f1'] >= 0) & (df['dh180'] < -15)]

    # 新策略按分数分组
    new_high = df[(df['new_bottom'] >= 7.5) & (df['dh180'] < -15)]
    new_mid = df[(df['new_bottom'] >= 5.5) & (df['new_bottom'] < 7.5) & (df['dh180'] < -15)]

    print(f"\n  {'策略':30s} | {'样本':>7s} | {'胜率':>6s} | {'收益':>8s} | {'盈亏比':>6s}")
    print("  " + "-" * 70)
    for name, sub in [
        ("V3-L2(f''>0+f'>=0+dh180<-20)", old_l2),
        ("V3-L3(L2+放量+dh180<-25)", old_l3),
        ("新策略-命中(f''>0+f'>=0+dh180<-15)", new_match),
        ("新策略-高分(>=7.5)", new_high),
        ("新策略-中分(5.5-7.5)", new_mid),
    ]:
        if len(sub) < 50:
            continue
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        avg_w = sub[sub['fwd_ret20'] > 0]['fwd_ret20'].mean() if (sub['fwd_ret20'] > 0).any() else 0
        avg_l = abs(sub[sub['fwd_ret20'] <= 0]['fwd_ret20'].mean()) if (sub['fwd_ret20'] <= 0).any() else 1
        plr = avg_w / avg_l if avg_l > 0 else 999
        print(f"  {name:30s} | {len(sub):>7,} | {wr:>5.1f}% | {ar:>+7.2f}% | {plr:>5.2f}")

    # ── 主升浪对比 ──
    print(f"\n{'─' * 80}")
    print("主升浪：新策略 vs V3原始")
    print("─" * 80)

    df['new_uptrend'] = df.apply(
        lambda r: score_uptrend_new(r['f1'], r['f2'], r['dh180'], r['vol_ratio'], r['above_ma20']),
        axis=1
    )

    old_up = df[(df['f1'] > 0) & (df['f2'] > 0.01) & (df['above_ma20'])]
    old_up_vol = old_up[old_up['vol_ratio'] > 1.0]
    new_up_high = df[(df['new_uptrend'] >= 8.0) & (df['above_ma20'])]
    new_up_mid = df[(df['new_uptrend'] >= 6.0) & (df['new_uptrend'] < 8.0) & (df['above_ma20'])]

    print(f"\n  {'策略':30s} | {'样本':>7s} | {'胜率':>6s} | {'收益':>8s} | {'盈亏比':>6s}")
    print("  " + "-" * 70)
    for name, sub in [
        ("V3-主升浪(f'>0+f''>0+>MA20)", old_up),
        ("V3-主升浪+放量", old_up_vol),
        ("新策略-高分(>=8.0)", new_up_high),
        ("新策略-中分(6.0-8.0)", new_up_mid),
    ]:
        if len(sub) < 50:
            continue
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        avg_w = sub[sub['fwd_ret20'] > 0]['fwd_ret20'].mean() if (sub['fwd_ret20'] > 0).any() else 0
        avg_l = abs(sub[sub['fwd_ret20'] <= 0]['fwd_ret20'].mean()) if (sub['fwd_ret20'] <= 0).any() else 1
        plr = avg_w / avg_l if avg_l > 0 else 999
        print(f"  {name:30s} | {len(sub):>7,} | {wr:>5.1f}% | {ar:>+7.2f}% | {plr:>5.2f}")

    # ── 新策略分数 vs 20日收益 分组 ──
    print(f"\n{'─' * 80}")
    print("新策略分数分组胜率（底部反转 & 主升浪）")
    print("─" * 80)

    for strategy_name, col, condition in [
        ("底部反转", "new_bottom", df['dh180'] < -15),
        ("主升浪", "new_uptrend", df['above_ma20']),
    ]:
        sub = df[condition]
        print(f"\n  {strategy_name} ({col})")
        print(f"  {'分数段':15s} | {'样本':>7s} | {'胜率':>6s} | {'收益':>8s}")
        print("  " + "-" * 45)
        for lo, hi, label in [
            (0, 3, "0-3(低)"),
            (3, 5, "3-5"),
            (5, 6, "5-6"),
            (6, 7, "6-7"),
            (7, 8, "7-8"),
            (8, 10, "8-10(高)"),
        ]:
            g = sub[(sub[col] >= lo) & (sub[col] < hi)]
            if len(g) < 30:
                continue
            wr = (g['fwd_ret20'] > 0).mean() * 100
            ar = g['fwd_ret20'].mean()
            print(f"  {label:15s} | {len(g):>7,} | {wr:>5.1f}% | {ar:>+7.2f}%")

    print("\n" + "=" * 70)
    print("回测完成")
    print("=" * 70)


if __name__ == "__main__":
    run()
