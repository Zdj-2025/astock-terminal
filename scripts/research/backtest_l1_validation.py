#!/usr/bin/env python3
"""L1 vs Savgol vs Kalman 拐点检测三方对比。

用 L1 趋势滤波的拐点作为客观基准，评估 Savgol 和 Kalman 导数系统的拐点检测质量。

指标：
  1. 拐点对齐度：Savgol/Kalman 的 f'' 过零点与 L1 拐点的距离
  2. 选股一致性：L1 确认趋势向上的股票与 V3 推荐 Top K 的重叠率
  3. 选股质量：L1 确认的股票 vs 未确认的股票的未来收益对比

用法：
  cd stock && PYTHONPATH=. python scripts/research/backtest_l1_validation.py
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
from scripts.analyzers.l1_trend_filter import L1TrendFilter

SNAPSHOT_DATES = [
    "2025-12-22",
    "2026-02-23",
    "2026-04-20",
    "2026-05-18",
]

LAM = 20         # L1 正则化参数
WINDOW = 120     # L1 窗口（只用最近120天数据）
FORWARD_DAYS = 20
TOP_K = 20


def truncate_kline(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    return df[df[dt_col] <= pd.to_datetime(date_str)].reset_index(drop=True)


def get_forward_return(code: str, snapshot_date: str, data_root: Path) -> float | None:
    path = data_root / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    future = df[df[dt_col] > pd.to_datetime(snapshot_date)].head(FORWARD_DAYS + 1)
    if len(future) < 2:
        return None
    entry = float(future.iloc[0]["open"])
    exit_price = float(future.iloc[-1]["close"])
    if entry <= 0:
        return None
    return (exit_price / entry - 1) * 100


def l1_validate_stock(close: np.ndarray, l1: L1TrendFilter) -> dict:
    """用 L1 验证股票当前趋势状态。

    Returns:
        dict: {up_trend, slope, bars_since_kink, n_segments}
    """
    # 只用最近 WINDOW 天
    if len(close) > WINDOW:
        close_w = close[-WINDOW:]
    else:
        close_w = close

    if len(close_w) < 30:
        return {"up_trend": False, "slope": 0.0, "bars_since_kink": 999, "n_segments": 0}

    try:
        info = l1.last_kink_info(close_w)
    except Exception:
        return {"up_trend": False, "slope": 0.0, "bars_since_kink": 999, "n_segments": 0}

    # 当前趋势向上 = slope > 0 且最近拐点不遥远
    up_trend = info["slope_direction"] == "up" and info["bars_since_kink"] < 60

    return {
        "up_trend": up_trend,
        "slope": info["current_slope"],
        "bars_since_kink": info["bars_since_kink"],
        "n_segments": info["n_segments"],
        "slope_direction": info["slope_direction"],
    }


def main():
    data_root = Path("data/store")
    scorer = V3DerivativeScorer(data_root=data_root, use_kalman=False)
    l1 = L1TrendFilter(lam=LAM)

    codes = sorted([
        d.name for d in (data_root / "kline").iterdir()
        if d.is_dir() and (d / "D.parquet").exists() and (d / "60min.parquet").exists()
    ])
    print(f"股票池: {len(codes)}只（有分钟数据）")
    print(f"L1参数: lambda={LAM}, window={WINDOW}")
    print("=" * 80)

    for sd in SNAPSHOT_DATES:
        print(f"\n--- {sd} ---")

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
                df = truncate_kline(df, sd)
                if len(df) < 60:
                    continue

                close = df["close"].values.astype(float)
                if not np.all(np.isfinite(close)) or not np.all(close > 0):
                    continue

                # V3 评分
                s = scorer.score_stock(code, df)
                if s.signal == "数据不足" or not np.isfinite(s.f1):
                    continue

                # L1 验证
                l1_info = l1_validate_stock(close, l1)

                ma20 = float(np.mean(close[-20:])) if len(close) >= 20 else close[-1]

                rows.append({
                    "code": code,
                    "v3_raw": s.raw_total,
                    "f1": s.f1,
                    "f2": s.f2,
                    "signal": s.signal,
                    "above_ma20": close[-1] > ma20,
                    "l1_up": l1_info["up_trend"],
                    "l1_slope": l1_info["slope"],
                    "l1_bars": l1_info["bars_since_kink"],
                    "l1_n_seg": l1_info["n_segments"],
                    "l1_direction": l1_info["slope_direction"],
                })
            except Exception:
                continue

        batch = pd.DataFrame(rows)
        if len(batch) == 0:
            print("  无结果")
            continue

        # V3 Top K（趋势加速）
        mask = (
            (batch["f1"] > 0) & (batch["f2"] > 0) &
            (batch["above_ma20"] == True) & (batch["v3_raw"] >= 4.0)
        )
        top = batch[mask].nlargest(TOP_K, "v3_raw")

        if len(top) == 0:
            print("  无趋势加速股票")
            continue

        # L1 确认率
        l1_confirmed = top[top["l1_up"] == True]
        l1_rejected = top[top["l1_up"] == False]

        # 计算前瞻收益
        confirmed_returns = []
        for code in l1_confirmed["code"]:
            r = get_forward_return(code, sd, data_root)
            if r is not None:
                confirmed_returns.append(r)

        rejected_returns = []
        for code in l1_rejected["code"]:
            r = get_forward_return(code, sd, data_root)
            if r is not None:
                rejected_returns.append(r)

        cr = np.array(confirmed_returns) if confirmed_returns else np.array([0])
        rr = np.array(rejected_returns) if rejected_returns else np.array([0])

        print(f"  V3 Top{len(top)}: L1确认={len(l1_confirmed)}只, 未确认={len(l1_rejected)}只")
        print(f"  L1确认: n={len(confirmed_returns)} avg={np.mean(cr):+.2f}% hit={np.mean(cr>0):.0%}")
        print(f"  L1未确认: n={len(rejected_returns)} avg={np.mean(rr):+.2f}% hit={np.mean(rr>0):.0%}")

        # 全量 L1 方向分布
        up_count = len(batch[batch["l1_direction"] == "up"])
        down_count = len(batch[batch["l1_direction"] == "down"])
        flat_count = len(batch[batch["l1_direction"] == "flat"])
        print(f"  全市场L1方向: 上={up_count} 下={down_count} 平={flat_count}")

    # ── 汇总：L1 作为验证层的价值 ──
    print("\n" + "=" * 80)
    print("L1 验证层总结：")
    print("  如果 L1确认的收益 > L1未确认的收益 → L1 验证有价值")
    print("  如果两者差不多 → L1 没有额外信息增益")


if __name__ == "__main__":
    main()
