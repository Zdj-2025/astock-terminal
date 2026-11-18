#!/usr/bin/env python3
"""V3增强回测 v2 — 修正方法论问题。

修正项：
  1. 所有配置统一用 score_stock() 评分（口径一致）
  2. 分钟数据按 snapshot_date 截断（消除前瞻偏差）
  3. MTF公平对比：先从同候选池选Top K，再用MTF分"确认/未确认"两组比较收益
  4. 停牌过滤：入场日与snapshot间隔>3天则跳过
  5. Bootstrap 95%置信区间

用法：
  cd stock && PYTHONPATH=. python scripts/research/backtest_v3_enhanced.py
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
from scripts.screener.multi_timeframe_scorer import MultiTimeframeScorer

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


def get_forward_return(code: str, snapshot_date: str, data_root: Path) -> float | None:
    """计算前瞻收益，带停牌过滤。"""
    path = data_root / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    snap = pd.to_datetime(snapshot_date)
    future = df[df[dt_col] > snap].head(FORWARD_DAYS + 1)
    if len(future) < 2:
        return None

    # 停牌过滤：入场日与snapshot间隔>3天 → 可能停牌
    entry_date = pd.to_datetime(future.iloc[0][dt_col])
    if (entry_date - snap).days > 3:
        return None

    entry = float(future.iloc[0]["open"])
    exit_price = float(future.iloc[-1]["close"])
    if entry <= 0:
        return None
    return (exit_price / entry - 1) * 100


def bootstrap_ci(returns: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap 置信区间。"""
    if len(returns) < 3:
        return (np.nan, np.nan)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return (round(lo, 2), round(hi, 2))


def main():
    data_root = Path("data/store")
    scorer = V3DerivativeScorer(data_root=data_root)
    mtf = MultiTimeframeScorer(data_root=data_root)

    codes = sorted([
        d.name for d in (data_root / "kline").iterdir()
        if d.is_dir()
        and (d / "D.parquet").exists()
        and (d / "60min.parquet").exists()
    ])
    print(f"股票池: {len(codes)}只（有分钟数据）")
    print(f"快照日期: {SNAPSHOT_DATES}")
    print(f"Top K: {TOP_K}, 前瞻天数: {FORWARD_DAYS}")
    print("=" * 90)

    # ── 每个快照：统一评分 → 分两组对比 ──

    # 收集所有快照的结果
    baseline_returns_all = []       # V3 baseline Top K 收益
    mtf_confirmed_all = []          # MTF确认的收益
    mtf_rejected_all = []           # MTF未确认的收益

    for sd in SNAPSHOT_DATES:
        print(f"\n--- {sd} ---")

        # 1. 统一评分：用 score_stock()（已含分钟截断）
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
                df = df[df[dt_col] <= pd.to_datetime(sd)].reset_index(drop=True)
                if len(df) < 60:
                    continue

                s = scorer.score_stock(code, df)
                if s.signal == "数据不足" or not np.isfinite(s.f1):
                    continue

                rows.append({
                    "code": code,
                    "v3_raw": s.raw_total,
                    "v3_total": s.total,
                    "f1": s.f1,
                    "f2": s.f2,
                    "signal": s.signal,
                    "above_ma20": s.details.get("above_ma20", False),
                    "minute_score": s.minute_score,
                })
            except Exception:
                continue

        batch = pd.DataFrame(rows)
        if len(batch) == 0:
            print("  无结果")
            continue

        # 2. V3 baseline: 趋势加速 Top K
        mask = (
            (batch["f1"] > 0) & (batch["f2"] > 0) &
            (batch["above_ma20"] == True) & (batch["v3_raw"] >= 4.0)
        )
        top = batch[mask].nlargest(TOP_K, "v3_raw")
        if len(top) == 0:
            print("  无趋势加速股票")
            continue

        # 计算 baseline 收益
        baseline_rets = []
        for code in top["code"]:
            r = get_forward_return(code, sd, data_root)
            if r is not None:
                baseline_rets.append(r)
        baseline_rets = np.array(baseline_rets) if baseline_rets else np.array([])

        # 3. MTF 评分（截断到 snapshot_date）
        top_codes = top["code"].tolist()
        mtf_df = mtf.score_batch(top_codes, cutoff_date=sd)

        if len(mtf_df) > 0:
            top_mtf = top.merge(
                mtf_df[["code", "mtf_level", "mtf_confidence", "mtf_penalty"]],
                on="code", how="left"
            )
            top_mtf["mtf_level"] = top_mtf["mtf_level"].fillna("数据不足")

            confirmed = top_mtf[~top_mtf["mtf_level"].isin(["单周期信号", "无共识"])]
            rejected = top_mtf[top_mtf["mtf_level"].isin(["单周期信号", "无共识"])]

            conf_rets = []
            for code in confirmed["code"]:
                r = get_forward_return(code, sd, data_root)
                if r is not None:
                    conf_rets.append(r)

            rej_rets = []
            for code in rejected["code"]:
                r = get_forward_return(code, sd, data_root)
                if r is not None:
                    rej_rets.append(r)

            conf_arr = np.array(conf_rets) if conf_rets else np.array([])
            rej_arr = np.array(rej_rets) if rej_rets else np.array([])
        else:
            conf_arr = np.array([])
            rej_arr = np.array([])

        # 打印
        bl_str = f"avg={np.mean(baseline_rets):+.2f}% hit={np.mean(baseline_rets>0):.0%} n={len(baseline_rets)}" if len(baseline_rets) > 0 else "N/A"
        cf_str = f"avg={np.mean(conf_arr):+.2f}% hit={np.mean(conf_arr>0):.0%} n={len(conf_arr)}" if len(conf_arr) > 0 else "N/A"
        rj_str = f"avg={np.mean(rej_arr):+.2f}% hit={np.mean(rej_arr>0):.0%} n={len(rej_arr)}" if len(rej_arr) > 0 else "N/A"
        print(f"  V3 Top{len(top)}:     {bl_str}")
        print(f"  ├─ MTF确认:  {cf_str}")
        print(f"  └─ MTF未确认: {rj_str}")

        # 收集
        baseline_returns_all.extend(baseline_rets.tolist())
        mtf_confirmed_all.extend(conf_arr.tolist())
        mtf_rejected_all.extend(rej_arr.tolist())

    # ── 汇总 ──
    print("\n" + "=" * 90)
    print("汇总（所有快照合并）")
    print("-" * 90)

    bl = np.array(baseline_returns_all)
    cf = np.array(mtf_confirmed_all)
    rj = np.array(mtf_rejected_all)

    for name, arr in [("V3 Top K (全量)", bl), ("MTF确认", cf), ("MTF未确认", rj)]:
        if len(arr) < 3:
            print(f"  {name:20s}: n={len(arr)}, 样本不足")
            continue
        avg = np.mean(arr)
        hit = np.mean(arr > 0)
        med = np.median(arr)
        ci = bootstrap_ci(arr)
        print(f"  {name:20s}: n={len(arr):3d}  avg={avg:+6.2f}%  "
              f"hit={hit:.0%}  med={med:+6.2f}%  "
              f"95%CI=[{ci[0]:+.2f}%, {ci[1]:+.2f}%]")

    # 显著性检验：MTF确认 vs 未确认
    if len(cf) >= 5 and len(rj) >= 5:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(cf, rj, alternative='greater')
        print(f"\n  MTF确认 vs 未确认 Mann-Whitney U 检验: p={pval:.4f} "
              f"{'*显著*' if pval < 0.05 else '(不显著)'}")


if __name__ == "__main__":
    main()
