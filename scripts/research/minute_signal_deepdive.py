#!/usr/bin/env python3
"""分钟级信号深挖 — f''拐点+f'>0 信号验证。

核心发现：60min f''拐点 + f'>0 = 63%胜率
深挖目标：
  1. 最优lookback窗口（5/10/15/20根）
  2. f'阈值敏感性（0/0.05/0.1/0.2）
  3. 按市场状态拆分（均线多头/空头/交叉）
  4. 时间衰减曲线（信号后1-40根）
  5. 跟60min底背离对比，谁更强？
  6. 组合效果：f''拐+f'>0+60背离
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, argrelextrema

warnings.filterwarnings("ignore")

DATA_ROOT = Path("./data/store/kline")
SG_WIN, SG_POLY = 21, 3


def compute_macd(close):
    s = pd.Series(close)
    ema12 = s.ewm(span=12, adjust=False).mean().values
    ema26 = s.ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd = (dif - dea) * 2
    return dif, dea, macd


def detect_bottom_divergence(close, low, macd, lookback=80, order=3):
    n = len(close)
    if n < lookback:
        return False
    rec_low = low[-lookback:]
    rec_macd = macd[-lookback:]
    lows_idx = argrelextrema(rec_low, np.less, order=order)[0]
    if len(lows_idx) < 2:
        return False
    i1, i2 = lows_idx[-2], lows_idx[-1]
    return rec_low[i2] < rec_low[i1] and rec_macd[i2] > rec_macd[i1]


def check_ma_alignment(close):
    if len(close) < 20:
        return "insufficient"
    s = pd.Series(close)
    ma5 = s.rolling(5).mean().values[-1]
    ma10 = s.rolling(10).mean().values[-1]
    ma20 = s.rolling(20).mean().values[-1]
    if ma5 > ma10 > ma20:
        return "bull"
    elif ma5 < ma10 < ma20:
        return "bear"
    return "cross"


def scan_deep(minute_codes):
    all_rows = []
    t0 = time.time()

    for idx, code in enumerate(minute_codes):
        try:
            path60 = DATA_ROOT / code / "60min.parquet"
            if not path60.exists():
                continue
            df60 = pd.read_parquet(path60)
            df60["datetime"] = pd.to_datetime(df60["datetime"])
            df60 = df60.sort_values("datetime").reset_index(drop=True)
            close = df60["close"].values.astype(float)
            low = df60["low"].values.astype(float)
            n = len(close)
            if n < 100:
                continue

            dif, dea, macd = compute_macd(close)

            if n >= SG_WIN:
                ln_p = np.log(close)
                f1 = savgol_filter(ln_p, SG_WIN, SG_POLY, deriv=1) * 100
                f2 = savgol_filter(ln_p, SG_WIN, SG_POLY, deriv=2) * 10000
            else:
                continue

            margin = max(80, SG_WIN)
            sample_range = range(margin, n - 40, 10)  # 每10根采样

            for i in sample_range:
                seg_close = close[:i+1]
                seg_low = low[:i+1]
                seg_macd = macd[:i+1]

                # f''拐点检测（不同窗口）
                f2_crosses = {}
                for lookback in [3, 5, 7, 10]:
                    if i >= lookback:
                        f2_past = np.mean(f2[i-lookback:i])
                        f2_now = f2[i]
                        f2_crosses[f"cross_lb{lookback}"] = int(f2_past < 0 and f2_now > 0)

                bd60 = int(detect_bottom_divergence(seg_close, seg_low, seg_macd))
                ma_state = check_ma_alignment(seg_close)

                # 多个未来收益窗口
                rets = {}
                for fwd in [1, 3, 5, 10, 15, 20, 30, 40]:
                    j = min(i + fwd, n - 1)
                    rets[f"ret{fwd}"] = round((close[j] / close[i] - 1) * 100, 2)

                row = {
                    "code": code,
                    "datetime": df60.iloc[i]["datetime"],
                    "close": close[i],
                    "f1": round(f1[i], 4),
                    "f2": round(f2[i], 4),
                    "bd60": bd60,
                    "ma": ma_state,
                    **f2_crosses,
                    **rets,
                }
                all_rows.append(row)
        except Exception:
            continue

        if (idx + 1) % 300 == 0:
            print(f"  {idx+1}/{len(minute_codes)} ({time.time()-t0:.0f}s) {len(all_rows):,}条")

    df = pd.DataFrame(all_rows)
    print(f"\n扫描完成: {len(df):,}条, {df['code'].nunique()}只")
    return df


def analyze_deep(df):
    sep = "=" * 70

    # ============================================================
    print(f"\n{sep}")
    print("【一、f''拐点窗口敏感性】")
    print(sep)
    print(f"找最优lookback窗口（f''从负转正的判定窗口）\n")

    for lb in [3, 5, 7, 10]:
        col = f"cross_lb{lb}"
        # 整体
        mask_all = df[col] == 1
        sub = df[mask_all]
        if len(sub) < 20:
            print(f"  lb={lb}: 样本不足")
            continue

        # f'>0 的子集
        mask_up = mask_all & (df["f1"] > 0)
        sub_up = df[mask_up]

        print(f"  lb={lb}: 全部={len(sub):,} ret20_wr={(sub['ret20']>0).mean()*100:.1f}% avg={sub['ret20'].mean():+.2f}%")
        if len(sub_up) >= 20:
            print(f"         f'>0: {len(sub_up):,} ret20_wr={(sub_up['ret20']>0).mean()*100:.1f}% avg={sub_up['ret20'].mean():+.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【二、f'阈值敏感性（lb=5, f''拐点+f'>threshold）】")
    print(sep)

    col = "cross_lb5"
    for threshold in [0, 0.02, 0.05, 0.1, 0.15, 0.2]:
        mask = (df[col] == 1) & (df["f1"] > threshold)
        sub = df[mask]
        if len(sub) < 10:
            print(f"  f'>{threshold}: 样本不足")
            continue
        print(f"  f'>{threshold:.2f}: n={len(sub):>5,}  ret5_wr={(sub['ret5']>0).mean()*100:.1f}%  ret20_wr={(sub['ret20']>0).mean()*100:.1f}%  ret20_avg={sub['ret20'].mean():+.2f}%  ret40_wr={(sub['ret40']>0).mean()*100:.1f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【三、f''拐+f'>0 按市场状态拆分】")
    print(sep)

    mask_signal = (df["cross_lb5"] == 1) & (df["f1"] > 0)
    for state in ["bull", "bear", "cross"]:
        mask_state = mask_signal & (df["ma"] == state)
        sub = df[mask_state]
        if len(sub) < 10:
            print(f"  {state}: 样本不足")
            continue
        print(f"  {state:>6s}: n={len(sub):>5,}  ret20_wr={(sub['ret20']>0).mean()*100:.1f}%  avg={sub['ret20'].mean():+.2f}%  ret40_wr={(sub['ret40']>0).mean()*100:.1f}%  avg={sub['ret40'].mean():+.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【四、时间衰减曲线 — 信号后第N根收益】")
    print(sep)

    # f''拐+f'>0
    mask_signal = (df["cross_lb5"] == 1) & (df["f1"] > 0)
    sub_sig = df[mask_signal]
    # 60min底背离
    sub_bd = df[df["bd60"] == 1]
    # 基准
    sub_base = df[(df["cross_lb5"] == 0) & (df["bd60"] == 0)]

    print(f"\n  N根  f2拐+f1>0       60min背离      无信号")
    for fwd in [1, 3, 5, 10, 15, 20, 30, 40]:
        col = f"ret{fwd}"
        sig_wr = (sub_sig[col]>0).mean()*100 if len(sub_sig)>10 else 0
        bd_wr = (sub_bd[col]>0).mean()*100 if len(sub_bd)>10 else 0
        base_wr = (sub_base[col]>0).mean()*100 if len(sub_base)>10 else 0
        sig_avg = sub_sig[col].mean() if len(sub_sig)>10 else 0
        bd_avg = sub_bd[col].mean() if len(sub_bd)>10 else 0
        base_avg = sub_base[col].mean() if len(sub_base)>10 else 0
        print(f"  {fwd:>3}   {sig_wr:>5.1f}% {sig_avg:>+5.2f}%  {bd_wr:>5.1f}% {bd_avg:>+5.2f}%  {base_wr:>5.1f}% {base_avg:>+5.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【五、对比：f''拐+f'>0 vs 60min背离 vs 组合】")
    print(sep)

    combos = {
        "f''拐+f'>0 (lb5)": (df["cross_lb5"]==1) & (df["f1"]>0),
        "60min底背离": df["bd60"]==1,
        "f''拐+f'>0+60背离": (df["cross_lb5"]==1) & (df["f1"]>0) & (df["bd60"]==1),
        "f''拐+f'>0+均线多头": (df["cross_lb5"]==1) & (df["f1"]>0) & (df["ma"]=="bull"),
        "60背离+均线多头": (df["bd60"]==1) & (df["ma"]=="bull"),
        "f''拐+f'>0+背离+多头": (df["cross_lb5"]==1) & (df["f1"]>0) & (df["bd60"]==1) & (df["ma"]=="bull"),
    }

    print(f"\n{'组合':>25s}  {'样本':>5s}  {'ret5_wr':>7s}  {'ret20_wr':>7s}  {'ret40_wr':>7s}  {'ret20_avg':>8s}  {'ret40_avg':>8s}")
    for label, mask in combos.items():
        sub = df[mask]
        if len(sub) < 10:
            print(f"{label:>25s}  {len(sub):>5,}  (不足)")
            continue
        print(f"{label:>25s}  {len(sub):>5,}  {(sub['ret5']>0).mean()*100:>6.1f}%  {(sub['ret20']>0).mean()*100:>6.1f}%  {(sub['ret40']>0).mean()*100:>6.1f}%  {sub['ret20'].mean():>+7.2f}%  {sub['ret40'].mean():>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【六、IC — 信号与未来收益的相关性】")
    print(sep)

    # f1 vs ret20
    ic_f1 = np.corrcoef(df["f1"], df["ret20"])[0, 1]
    ic_f2 = np.corrcoef(df["f2"], df["ret20"])[0, 1]

    # f''拐点作为二值信号 vs ret20
    mask_sig = (df["cross_lb5"]==1) & (df["f1"]>0)
    binary_signal = mask_sig.astype(int)
    ic_signal = np.corrcoef(binary_signal, df["ret20"])[0, 1]

    # bd60 二值 vs ret20
    ic_bd60 = np.corrcoef(df["bd60"], df["ret20"])[0, 1]

    print(f"  f1 vs ret20 IC: {ic_f1:+.4f}")
    print(f"  f2 vs ret20 IC: {ic_f2:+.4f}")
    print(f"  f''拐+f'>0(二值) vs ret20 IC: {ic_signal:+.4f}")
    print(f"  60min底背离(二值) vs ret20 IC: {ic_bd60:+.4f}")

    # ============================================================
    print(f"\n{sep}")
    print("【七、有趣发现 — f''拐+f'>0 后最大回撤 vs 最大收益】")
    print(sep)

    mask_sig = (df["cross_lb5"]==1) & (df["f1"]>0)
    sub = df[mask_sig]
    if len(sub) >= 20:
        # ret20分布
        ret20 = sub["ret20"].values
        print(f"  样本: {len(sub)}")
        print(f"  ret20分布:")
        for lo, hi in [(-999,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,999)]:
            cnt = ((ret20>=lo)&(ret20<hi)).sum()
            pct = cnt/len(sub)*100
            print(f"    [{lo:+4d}, {hi:+4d}): {cnt:>5} ({pct:>5.1f}%)")

        # 盈亏比
        wins = ret20[ret20>0]
        losses = ret20[ret20<0]
        if len(wins)>0 and len(losses)>0:
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            print(f"\n  平均盈利: {avg_win:.2f}%")
            print(f"  平均亏损: {avg_loss:.2f}%")
            print(f"  盈亏比: {avg_win/avg_loss:.2f}")
            print(f"  期望值: {ret20.mean():.2f}%")

    return df


def main():
    print("="*70)
    print("分钟级信号深挖 — f''拐点+f'>0")
    print("="*70)

    codes = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir() and (d/"60min.parquet").exists()])
    print(f"有60min数据的标的: {len(codes)}只")

    df = scan_deep(codes)
    if len(df) == 0:
        return
    analyze_deep(df)

    out = "/tmp/minute_deep_dive.parquet"
    df.to_parquet(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
