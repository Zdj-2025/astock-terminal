#!/usr/bin/env python3
"""分钟级信号回测 — 方案三验证。

验证目标：
  1. 30min+60min底背离共振是否真有效？
  2. 60min f''拐点是否提前预警？
  3. 组合信号 vs 单一信号 vs 无信号，胜率差多少？
  4. 有什么有趣的发现？

数据：1349只 × 1000条60min/30min
标签：未来N根K线的收益（5根=约1天，20根=约5天，40根=约10天）
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


# ── MACD 计算 ──────────────────────────────────────────────────────

def compute_macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准MACD: DIF=EMA12-EMA26, DEA=EMA9(DIF), MACD=2*(DIF-DEA)"""
    s = pd.Series(close)
    ema12 = s.ewm(span=12, adjust=False).mean().values
    ema26 = s.ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd = (dif - dea) * 2
    return dif, dea, macd


# ── 底背离检测 ──────────────────────────────────────────────────────

def detect_bottom_divergence(
    close: np.ndarray,
    low: np.ndarray,
    macd_hist: np.ndarray,
    lookback: int = 80,
    order: int = 3,
) -> tuple[bool, dict]:
    """底背离检测：LOW新低 + MACD柱不新低。

    Args:
        close: 收盘价（保留参数兼容，未使用）
        low: K线最低价（用LOW找价格低点）
        macd_hist: MACD柱值 = (DIF-DEA)*2
            全市场回测: MACD柱胜率54.3% > DIF 49.5% > DEA 47.6%
    """
    n = len(close)
    if n < lookback:
        return False, {"reason": "数据不足"}

    rec_low = low[-lookback:]
    rec_hist = macd_hist[-lookback:]

    # 找低点（用LOW）
    lows_idx = argrelextrema(rec_low, np.less, order=order)[0]

    if len(lows_idx) < 2:
        return False, {"reason": "低点不足2个"}

    i1, i2 = lows_idx[-2], lows_idx[-1]
    price_lower = rec_low[i2] < rec_low[i1]
    macd_higher = rec_hist[i2] > rec_hist[i1]

    details = {
        "low1": rec_low[i1],
        "low2": rec_low[i2],
        "hist1": rec_hist[i1],
        "hist2": rec_hist[i2],
        "price_lower": price_lower,
        "macd_higher": macd_higher,
    }

    if price_lower and macd_higher:
        return True, details
    return False, details


# ── 顶背离检测 ──────────────────────────────────────────────────────

def detect_top_divergence(
    close: np.ndarray,
    high: np.ndarray,
    macd_hist: np.ndarray,
    lookback: int = 80,
    order: int = 3,
) -> bool:
    """顶背离检测：HIGH新高 + MACD柱不新高。"""
    n = len(close)
    if n < lookback:
        return False

    rec_high = high[-lookback:]
    rec_hist = macd_hist[-lookback:]

    highs_idx = argrelextrema(rec_high, np.greater, order=order)[0]
    if len(highs_idx) < 2:
        return False

    i1, i2 = highs_idx[-2], highs_idx[-1]
    return rec_high[i2] > rec_high[i1] and rec_hist[i2] < rec_hist[i1]


# ── f'' 拐点检测 ────────────────────────────────────────────────────

def detect_f2_crossover(
    close: np.ndarray,
    sg_win: int = SG_WIN,
    sg_poly: int = SG_POLY,
    check_bars: int = 5,
) -> tuple[bool, float, float]:
    """检测 f'' 从负转正（拐点）。

    Returns:
        (has_crossover, f2_current, f2_previous_mean)
    """
    if len(close) < sg_win:
        return False, 0.0, 0.0

    ln_p = np.log(close.astype(float))
    f2 = savgol_filter(ln_p, sg_win, sg_poly, deriv=2) * 10000

    f2_now = f2[-1]
    f2_prev = np.mean(f2[-check_bars-1:-1])

    return f2_prev < 0 and f2_now > 0, round(f2_now, 4), round(f2_prev, 4)


# ── 均线排列 ──────────────────────────────────────────────────────

def check_ma_alignment(close: np.ndarray) -> tuple[bool, str]:
    """MA5 > MA10 > MA20 = 多头排列。"""
    if len(close) < 20:
        return False, "数据不足"
    s = pd.Series(close)
    ma5 = s.rolling(5).mean().values[-1]
    ma10 = s.rolling(10).mean().values[-1]
    ma20 = s.rolling(20).mean().values[-1]

    if ma5 > ma10 > ma20:
        return True, "多头"
    elif ma5 < ma10 < ma20:
        return False, "空头"
    else:
        return False, "交叉"


# ── 全量扫描 ──────────────────────────────────────────────────────

def scan_all(minute_codes: list[str]) -> pd.DataFrame:
    """对所有有分钟线的标的做信号扫描。"""
    all_rows = []
    t0 = time.time()

    for idx, code in enumerate(minute_codes):
        try:
            # 读取60min
            path60 = DATA_ROOT / code / "60min.parquet"
            path30 = DATA_ROOT / code / "30min.parquet"

            if not path60.exists():
                continue

            df60 = pd.read_parquet(path60)
            df60["datetime"] = pd.to_datetime(df60["datetime"])
            df60 = df60.sort_values("datetime").reset_index(drop=True)

            close60 = df60["close"].values.astype(float)
            low60 = df60["low"].values.astype(float)
            high60 = df60["high"].values.astype(float)
            n60 = len(close60)

            if n60 < 100:
                continue

            # MACD
            dif60, dea60, macd60 = compute_macd(close60)

            # f' f'' on 60min
            if n60 >= SG_WIN:
                ln_p60 = np.log(close60)
                f1_60 = savgol_filter(ln_p60, SG_WIN, SG_POLY, deriv=1) * 100
                f2_60 = savgol_filter(ln_p60, SG_WIN, SG_POLY, deriv=2) * 10000
            else:
                f1_60 = np.zeros(n60)
                f2_60 = np.zeros(n60)

            # 均线排列
            has_ma60, ma_label60 = check_ma_alignment(close60)

            # 30min 数据
            has_30min = False
            bottom_div_30 = False
            top_div_30 = False
            if path30.exists():
                df30 = pd.read_parquet(path30)
                df30["datetime"] = pd.to_datetime(df30["datetime"])
                df30 = df30.sort_values("datetime").reset_index(drop=True)
                if len(df30) >= 100:
                    has_30min = True
                    close30 = df30["close"].values.astype(float)
                    low30 = df30["low"].values.astype(float)
                    high30 = df30["high"].values.astype(float)
                    dif30, _, macd30 = compute_macd(close30)
                    bottom_div_30, _ = detect_bottom_divergence(close30, low30, macd30)
                    top_div_30 = detect_top_divergence(close30, high30, macd30)

            # 滑动窗口扫描（每20根取一个样本点）
            margin = max(80, SG_WIN)  # lookback for divergence + S-G
            sample_range = range(margin, n60 - 40, 20)  # 40根≈10天

            for i in sample_range:
                # 60min底背离
                seg_close = close60[:i+1]
                seg_low = low60[:i+1]
                seg_high = high60[:i+1]
                seg_macd = macd60[:i+1]

                bd60, _ = detect_bottom_divergence(seg_close, seg_low, seg_macd)
                td60 = detect_top_divergence(seg_close, seg_high, seg_macd)

                # f'' 拐点
                f2_cross, f2_val, f2_prev = detect_f2_crossover(seg_close)

                # f' 方向
                f1_val = f1_60[i] if i < len(f1_60) else 0
                f2_cur = f2_60[i] if i < len(f2_60) else 0

                # 均线排列
                ma_aligned, ma_label = check_ma_alignment(seg_close)

                # 标签：未来收益
                ret5 = (close60[min(i+5, n60-1)] / close60[i] - 1) * 100    # ~1天
                ret20 = (close60[min(i+20, n60-1)] / close60[i] - 1) * 100   # ~5天
                ret40 = (close60[min(i+40, n60-1)] / close60[i] - 1) * 100   # ~10天

                row = {
                    "code": code,
                    "datetime": df60.iloc[i]["datetime"],
                    "close": close60[i],
                    # 信号
                    "bd_60": int(bd60),         # 60min底背离
                    "td_60": int(td60),         # 60min顶背离
                    "f2_cross": int(f2_cross),  # f''拐点
                    "f1_60": round(f1_val, 4),
                    "f2_60": round(f2_cur, 4),
                    "ma_bull": int(ma_aligned), # 均线多头
                    "ma_label": ma_label,
                    # 30min
                    "has_30min": int(has_30min),
                    "bd_30": int(bottom_div_30 and has_30min),
                    "td_30": int(top_div_30 and has_30min),
                    # 组合
                    "resonance": int(bd60 and bottom_div_30 and has_30min),  # 底背离共振
                    # 标签
                    "ret5": round(ret5, 2),
                    "ret20": round(ret20, 2),
                    "ret40": round(ret40, 2),
                }
                all_rows.append(row)

        except Exception as e:
            continue

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(minute_codes)} 只, {len(all_rows):,}条, {elapsed:.0f}s")

    df = pd.DataFrame(all_rows)
    print(f"\n扫描完成: {len(df):,}条样本, {df['code'].nunique()}只")
    return df


# ── 分析 ──────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame):
    """全面分析分钟级信号效果。"""
    sep = "=" * 70

    # ============================================================
    print(f"\n{sep}")
    print("【一、单一信号胜率对比】")
    print(sep)

    signals = {
        "60min底背离": "bd_60",
        "30min底背离": "bd_30",
        "底背离共振": "resonance",
        "60min顶背离": "td_60",
        "f''拐点(负→正)": "f2_cross",
        "均线多头": "ma_bull",
    }

    print(f"\n{'信号':>16s}  {'样本':>6s}  {'ret5胜率':>8s}  {'ret20胜率':>8s}  {'ret40胜率':>8s}  {'ret5均值':>8s}  {'ret20均值':>8s}  {'ret40均值':>8s}")

    # 基准（全样本）
    n = len(df)
    print(f"{'[基准]':>16s}  {n:>6,}  {(df['ret5']>0).mean()*100:>7.1f}%  {(df['ret20']>0).mean()*100:>7.1f}%  {(df['ret40']>0).mean()*100:>7.1f}%  {df['ret5'].mean():>+7.2f}%  {df['ret20'].mean():>+7.2f}%  {df['ret40'].mean():>+7.2f}%")

    for label, col in signals.items():
        mask = df[col] == 1
        sub = df[mask]
        if len(sub) < 20:
            print(f"{label:>16s}  {len(sub):>6,}  (样本不足)")
            continue
        print(f"{label:>16s}  {len(sub):>6,}  {(sub['ret5']>0).mean()*100:>7.1f}%  {(sub['ret20']>0).mean()*100:>7.1f}%  {(sub['ret40']>0).mean()*100:>7.1f}%  {sub['ret5'].mean():>+7.2f}%  {sub['ret20'].mean():>+7.2f}%  {sub['ret40'].mean():>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【二、底背离共振 vs 单一背离 vs 无背离】")
    print(sep)

    groups = {
        "共振(30+60min)": df["resonance"] == 1,
        "仅60min背离": (df["bd_60"] == 1) & (df["resonance"] == 0),
        "仅30min背离": (df["bd_30"] == 1) & (df["resonance"] == 0),
        "均无背离": (df["bd_60"] == 0) & (df["bd_30"] == 0),
    }

    print(f"\n{'分组':>16s}  {'样本':>6s}  {'ret5胜率':>8s}  {'ret20胜率':>8s}  {'ret40胜率':>8s}  {'ret20均值':>8s}")
    for label, mask in groups.items():
        sub = df[mask]
        if len(sub) < 20:
            continue
        print(f"{label:>16s}  {len(sub):>6,}  {(sub['ret5']>0).mean()*100:>7.1f}%  {(sub['ret20']>0).mean()*100:>7.1f}%  {(sub['ret40']>0).mean()*100:>7.1f}%  {sub['ret20'].mean():>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【三、f''拐点效果 — 按 f' 方向拆分】")
    print(sep)

    cross = df[df["f2_cross"] == 1]
    if len(cross) >= 20:
        # f'>0 vs f'<0
        f1_up = cross[cross["f1_60"] > 0]
        f1_down = cross[cross["f1_60"] <= 0]
        f2_strong = cross[cross["f2_60"] > 0.05]
        f2_weak = cross[(cross["f2_60"] > 0) & (cross["f2_60"] <= 0.05)]

        print(f"\n全部f''拐点: {len(cross)}条")
        print(f"  ret20胜率: {(cross['ret20']>0).mean()*100:.1f}%  ret20均值: {cross['ret20'].mean():+.2f}%")

        for label, sub in [("f''拐+f'>0", f1_up), ("f''拐+f'<=0", f1_down)]:
            if len(sub) >= 20:
                print(f"  {label}: n={len(sub):,}  ret20胜率={(sub['ret20']>0).mean()*100:.1f}%  均值={sub['ret20'].mean():+.2f}%")

        for label, sub in [("f''强(>0.05)", f2_strong), ("f''弱(0~0.05)", f2_weak)]:
            if len(sub) >= 20:
                print(f"  {label}: n={len(sub):,}  ret20胜率={(sub['ret20']>0).mean()*100:.1f}%  均值={sub['ret20'].mean():+.2f}%")
    else:
        print(f"  样本不足: {len(cross)}")

    # ============================================================
    print(f"\n{sep}")
    print("【四、组合信号 — 背离 + f''拐点 + 均线】")
    print(sep)

    combos = {
        "共振+f''拐": (df["resonance"] == 1) & (df["f2_cross"] == 1),
        "共振+均线多头": (df["resonance"] == 1) & (df["ma_bull"] == 1),
        "60背离+均线多头": (df["bd_60"] == 1) & (df["ma_bull"] == 1),
        "f''拐+均线多头": (df["f2_cross"] == 1) & (df["ma_bull"] == 1),
        "60背离+f''拐": (df["bd_60"] == 1) & (df["f2_cross"] == 1),
        "顶背离+均线空头": (df["td_60"] == 1) & (df["ma_bull"] == 0),
    }

    print(f"\n{'组合':>20s}  {'样本':>6s}  {'ret5胜率':>8s}  {'ret20胜率':>8s}  {'ret40胜率':>8s}  {'ret20均值':>8s}")
    for label, mask in combos.items():
        sub = df[mask]
        if len(sub) < 10:
            print(f"{label:>20s}  {len(sub):>6,}  (样本不足)")
            continue
        print(f"{label:>20s}  {len(sub):>6,}  {(sub['ret5']>0).mean()*100:>7.1f}%  {(sub['ret20']>0).mean()*100:>7.1f}%  {(sub['ret40']>0).mean()*100:>7.1f}%  {sub['ret20'].mean():>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【五、有趣发现 — 顶背离的预测力】")
    print(sep)

    for label, mask in [("60min顶背离", df["td_60"] == 1), ("30min顶背离", df["td_30"] == 1)]:
        sub = df[mask]
        if len(sub) >= 20:
            print(f"\n{label}: {len(sub)}条")
            print(f"  ret5: 胜率={(sub['ret5']>0).mean()*100:.1f}%  均值={sub['ret5'].mean():+.2f}%")
            print(f"  ret20: 胜率={(sub['ret20']>0).mean()*100:.1f}%  均值={sub['ret20'].mean():+.2f}%")
            print(f"  ret40: 胜率={(sub['ret40']>0).mean()*100:.1f}%  均值={sub['ret40'].mean():+.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【六、背离信号衰减 — 信号后第N根K线的收益曲线】")
    print(sep)

    # 对背离共振的样本，看5/20/40根后的收益
    res_mask = df["resonance"] == 1
    if res_mask.sum() >= 20:
        print(f"\n底背离共振后收益衰减:")
        for n_bars, label in [(5, "5根(~1天)"), (20, "20根(~5天)"), (40, "40根(~10天)")]:
            col = f"ret{n_bars}"
            sub = df[res_mask]
            print(f"  {label}: 胜率={(sub[col]>0).mean()*100:.1f}%  均值={sub[col].mean():+.2f}%")

    # 无信号对比
    no_sig = (df["bd_60"] == 0) & (df["bd_30"] == 0) & (df["f2_cross"] == 0) & (df["td_60"] == 0)
    if no_sig.sum() >= 20:
        print(f"\n无信号基准:")
        for n_bars, label in [(5, "5根"), (20, "20根"), (40, "40根")]:
            col = f"ret{n_bars}"
            sub = df[no_sig]
            print(f"  {label}: 胜率={(sub[col]>0).mean()*100:.1f}%  均值={sub[col].mean():+.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【七、f''拐点作为预警 — 背离前几根就触发？】")
    print(sep)

    # 看f''拐点和底背离的时间关系
    bd_times = df[df["bd_60"] == 1][["code", "datetime"]].copy()
    cross_times = df[df["f2_cross"] == 1][["code", "datetime"]].copy()

    if len(bd_times) > 0 and len(cross_times) > 0:
        # 对每只股票，找背离前的最近一次拐点
        matched = 0
        gaps = []
        for code in bd_times["code"].unique():
            bd_dates = bd_times[bd_times["code"] == code]["datetime"].sort_values()
            cross_dates = cross_times[cross_times["code"] == code]["datetime"].sort_values()

            for bd_date in bd_dates:
                earlier = cross_dates[cross_dates < bd_date]
                if len(earlier) > 0:
                    last_cross = earlier.iloc[-1]
                    gap_hours = (bd_date - last_cross).total_seconds() / 3600
                    if gap_hours < 200:  # 200小时内
                        gaps.append(gap_hours)
                        matched += 1

        if gaps:
            print(f"\n背离前有f''拐点的比例: {matched}/{len(bd_times)} ({matched/len(bd_times)*100:.1f}%)")
            print(f"拐点到背离的时间间隔:")
            print(f"  中位数: {np.median(gaps):.1f}小时 ({np.median(gaps)/4:.1f}根60min)")
            print(f"  均值: {np.mean(gaps):.1f}小时")
            print(f"  <24小时: {sum(1 for g in gaps if g < 24)}/{len(gaps)} ({sum(1 for g in gaps if g < 24)/len(gaps)*100:.0f}%)")
            print(f"  <48小时: {sum(1 for g in gaps if g < 48)}/{len(gaps)} ({sum(1 for g in gaps if g < 48)/len(gaps)*100:.0f}%)")
        else:
            print("  没有匹配到拐点-背离对")

    return df


# ── 主入口 ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("分钟级信号回测")
    print("=" * 70)

    # 找有60min数据的标的
    codes = []
    for d in DATA_ROOT.iterdir():
        if d.is_dir() and (d / "60min.parquet").exists():
            codes.append(d.name)
    codes.sort()
    print(f"有60min数据的标的: {len(codes)}只")

    # 扫描
    print("\n开始扫描...")
    df = scan_all(codes)

    if len(df) == 0:
        print("无数据")
        return

    # 分析
    analyze(df)

    # 保存
    out = "/tmp/minute_signal_backtest.parquet"
    df.to_parquet(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
