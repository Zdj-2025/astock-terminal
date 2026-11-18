#!/usr/bin/env python3
"""Kalman vs Savgol 导数对比回测。

对比两种导数估计方法在V3评分框架下的选股质量：
  1. Savgol (window=21, poly=3, shift=10) — 现有方案
  2. Kalman (2-state, adaptive) — 无滞后方案

回测方法：
  - 选取多个历史快照日期
  - 分别用Savgol和Kalman计算导数，运行V3评分
  - 对Top K推荐股票计算未来20日收益
  - 对比两种方法的平均收益和命中率

用法：
  cd stock && PYTHONPATH=. python scripts/research/backtest_kalman_vs_savgol.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.v3_derivative_scorer import V3DerivativeScorer

SNAPSHOT_DATES = [
    "2025-10-27",
    "2025-11-24",
    "2025-12-22",
    "2026-01-19",
    "2026-02-23",
    "2026-03-23",
    "2026-04-20",
    "2026-05-18",
]

TOP_K = 20
FORWARD_DAYS = 20


def truncate_kline(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    cutoff = pd.to_datetime(date_str)
    return df[df[dt_col] <= cutoff].reset_index(drop=True)


def get_forward_return(code: str, snapshot_date: str, days: int, data_root: Path) -> float | None:
    path = data_root / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    snap = pd.to_datetime(snapshot_date)
    future = df[df[dt_col] > snap].head(days + 1)
    if len(future) < 2:
        return None
    entry = float(future.iloc[0]["open"])
    exit_price = float(future.iloc[-1]["close"])
    if entry <= 0:
        return None
    return (exit_price / entry - 1) * 100


def score_with_method(
    scorer: V3DerivativeScorer,
    codes: list,
    data_root: Path,
    snapshot_date: str,
) -> pd.DataFrame:
    """用指定方法评分，返回Top K。"""
    rows = []
    for code in codes:
        try:
            path = data_root / "kline" / code / "D.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            dt_col = "trade_date" if "trade_date" in df.columns else "date"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)
            df = truncate_kline(df, snapshot_date)
            if len(df) < 60:
                continue

            s = scorer.score_stock(code, df)
            if s.signal == "数据不足" or not np.isfinite(s.f1) or not np.isfinite(s.f2):
                continue

            rows.append({
                "code": code,
                "v3_raw": s.raw_total,
                "f1": s.f1,
                "f2": s.f2,
                "signal": s.signal,
                "above_ma20": s.details.get("above_ma20", False),
                "position_score": s.position_score,
                "trend_score": s.trend_score,
                "curvature_score": s.curvature_score,
                "minute_score": s.minute_score,
            })
        except Exception:
            continue

    batch = pd.DataFrame(rows)
    if len(batch) == 0:
        return batch

    # 趋势加速筛选（与daily_scan一致）
    mask = (
        (batch["f1"] > 0) &
        (batch["f2"] > 0) &
        (batch["above_ma20"] == True) &
        (batch["v3_raw"] >= 4.0)
    )
    return batch[mask].nlargest(TOP_K, "v3_raw")


def evaluate(top_df: pd.DataFrame, snapshot_date: str, data_root: Path) -> dict:
    if len(top_df) == 0:
        return {"n": 0, "avg_ret": 0, "hit_rate": 0, "max_ret": 0, "min_ret": 0, "median_ret": 0}

    returns = []
    for code in top_df["code"]:
        ret = get_forward_return(code, snapshot_date, FORWARD_DAYS, data_root)
        if ret is not None:
            returns.append(ret)

    if not returns:
        return {"n": 0, "avg_ret": 0, "hit_rate": 0, "max_ret": 0, "min_ret": 0, "median_ret": 0}

    returns = np.array(returns)
    return {
        "n": len(returns),
        "avg_ret": round(float(np.mean(returns)), 2),
        "hit_rate": round(float(np.mean(returns > 0)), 2),
        "max_ret": round(float(np.max(returns)), 2),
        "min_ret": round(float(np.min(returns)), 2),
        "median_ret": round(float(np.median(returns)), 2),
    }


def compare_derivatives_sample(
    scorer_sg: V3DerivativeScorer,
    scorer_km: V3DerivativeScorer,
    codes: list,
    data_root: Path,
    snapshot_date: str,
    n_sample: int = 10,
) -> dict:
    """对样本股票对比Kalman vs Savgol的导数值。"""
    f1_diffs = []
    f2_diffs = []
    f2_sign_agree = 0
    f1_sign_agree = 0
    total = 0

    for code in codes[:50]:
        path = data_root / "kline" / code / "D.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        dt_col = "trade_date" if "trade_date" in df.columns else "date"
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col).reset_index(drop=True)
        df = truncate_kline(df, snapshot_date)
        if len(df) < 60:
            continue

        close = df["close"].values.astype(float)
        if not np.all(np.isfinite(close)) or not np.all(close > 0):
            continue
        f1_sg, f2_sg = scorer_sg.compute_derivatives(close)
        f1_km, f2_km = scorer_km.compute_derivatives(close, use_kalman=True)

        # 只看稳定区间（跳过前30点）
        start = 30
        f1_diffs.append(float(np.mean(np.abs(f1_sg[start:] - f1_km[start:]))))
        f2_diffs.append(float(np.mean(np.abs(f2_sg[start:] - f2_km[start:]))))

        f1_agree = np.mean(np.sign(f1_sg[start:]) == np.sign(f1_km[start:]))
        f2_agree = np.mean(np.sign(f2_sg[start:]) == np.sign(f2_km[start:]))
        f1_sign_agree += float(f1_agree)
        f2_sign_agree += float(f2_agree)
        total += 1

        if total >= n_sample:
            break

    if total == 0:
        return {}

    return {
        "f1_avg_abs_diff": round(float(np.mean(f1_diffs)), 4),
        "f2_avg_abs_diff": round(float(np.mean(f2_diffs)), 4),
        "f1_sign_agree": round(f1_sign_agree / total, 3),
        "f2_sign_agree": round(f2_sign_agree / total, 3),
        "n_sampled": total,
    }


def main():
    data_root = Path("data/store")

    scorer_sg = V3DerivativeScorer(data_root=data_root, use_kalman=False)
    scorer_km = V3DerivativeScorer(data_root=data_root, use_kalman=True)

    codes = sorted([
        d.name for d in (data_root / "kline").iterdir()
        if d.is_dir() and (d / "D.parquet").exists()
    ])
    print(f"股票池: {len(codes)}只")
    print(f"快照日期: {SNAPSHOT_DATES}")
    print(f"Top K: {TOP_K}, 前瞻天数: {FORWARD_DAYS}")
    print("=" * 80)

    sg_results = []
    km_results = []
    overlap_counts = []

    # 先做一个导数对比
    print("\n--- 导数值对比（第一个快照）---")
    deriv_cmp = compare_derivatives_sample(scorer_sg, scorer_km, codes, data_root, SNAPSHOT_DATES[0])
    if deriv_cmp:
        print(f"  f' 平均绝对差: {deriv_cmp['f1_avg_abs_diff']:.4f}")
        print(f"  f'' 平均绝对差: {deriv_cmp['f2_avg_abs_diff']:.4f}")
        print(f"  f' 符号一致性: {deriv_cmp['f1_sign_agree']:.1%}")
        print(f"  f'' 符号一致性: {deriv_cmp['f2_sign_agree']:.1%}")

    print("\n--- 选股回测 ---")
    for sd in SNAPSHOT_DATES:
        print(f"\n--- {sd} ---")
        top_sg = pd.DataFrame()
        top_km = pd.DataFrame()

        # Savgol
        try:
            top_sg = score_with_method(scorer_sg, codes, data_root, sd)
            ev_sg = evaluate(top_sg, sd, data_root)
            sg_results.append(ev_sg)
            print(f"  Savgol   : n={ev_sg['n']:2d}  avg={ev_sg['avg_ret']:+6.2f}%  "
                  f"hit={ev_sg['hit_rate']:.0%}  med={ev_sg['median_ret']:+6.2f}%")
        except Exception as e:
            print(f"  Savgol   : ERROR {e}")
            sg_results.append({"n": 0, "avg_ret": 0, "hit_rate": 0, "max_ret": 0, "min_ret": 0, "median_ret": 0})

        # Kalman
        try:
            top_km = score_with_method(scorer_km, codes, data_root, sd)
            ev_km = evaluate(top_km, sd, data_root)
            km_results.append(ev_km)
            print(f"  Kalman   : n={ev_km['n']:2d}  avg={ev_km['avg_ret']:+6.2f}%  "
                  f"hit={ev_km['hit_rate']:.0%}  med={ev_km['median_ret']:+6.2f}%")
        except Exception as e:
            print(f"  Kalman   : ERROR {e}")
            km_results.append({"n": 0, "avg_ret": 0, "hit_rate": 0, "max_ret": 0, "min_ret": 0, "median_ret": 0})

        # 重叠率
        if len(top_sg) > 0 and len(top_km) > 0:
            overlap = len(set(top_sg["code"]) & set(top_km["code"]))
            overlap_counts.append(overlap)
            print(f"  重叠: {overlap}/{min(len(top_sg), len(top_km))}")

    # 汇总
    print("\n" + "=" * 80)
    print(f"{'方法':12s} | {'平均收益':>8s} | {'命中率':>6s} | {'中位收益':>8s} | {'样本数':>6s}")
    print("-" * 80)

    for name, results in [("Savgol", sg_results), ("Kalman", km_results)]:
        valid = [r for r in results if r["n"] > 0]
        if not valid:
            print(f"{name:12s} | {'N/A':>8s} | {'N/A':>6s} | {'N/A':>8s} | {0:>6d}")
            continue
        avg_rets = [r["avg_ret"] for r in valid]
        hit_rates = [r["hit_rate"] for r in valid]
        med_rets = [r["median_ret"] for r in valid]
        total_n = sum(r["n"] for r in valid)
        print(f"{name:12s} | {np.mean(avg_rets):+7.2f}% | {np.mean(hit_rates):5.0%} | "
              f"{np.mean(med_rets):+7.2f}% | {total_n:>6d}")

    if overlap_counts:
        print(f"\n平均重叠: {np.mean(overlap_counts):.1f}/{TOP_K}")


if __name__ == "__main__":
    main()
