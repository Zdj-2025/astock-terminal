#!/usr/bin/env python3
"""波浪位置子模型回测 — 统计严谨版。

验证：下跌5浪结束后W2/W3买点策略是否有统计显著的超额收益。

方法论：
  1. 历史快照法：8个月度快照，每快照跑wave_position扫描
  2. 前瞻收益：快照后20日收益，含停牌过滤
  3. 对照组：
     - 实验组(A/B级) vs 随机选股 vs 全市场均值
     - W2_ending vs W3_early 内部对比
  4. 统计检验：Mann-Whitney U + Bootstrap 95%CI
  5. 前防偏差：as_of_date截断K线

用法：
  cd stock && PYTHONPATH=. python scripts/research/backtest_wave_position.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.wave_position_screener import WavePositionScreener

SNAPSHOT_DATES = [
    "2025-07-21",
    "2025-08-18",
    "2025-09-22",
    "2025-10-27",
    "2025-11-24",
    "2025-12-22",
    "2026-01-19",
    "2026-02-23",
    "2026-03-23",
    "2026-04-20",
]

FORWARD_DAYS = 20
DATA_ROOT = Path("data/store")


def get_forward_return(code: str, snapshot_date: str) -> float | None:
    """计算前瞻收益，带停牌过滤。"""
    path = DATA_ROOT / "kline" / code / "D.parquet"
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


def random_baseline(n_stocks: int, snapshot_date: str, n_random: int = 50) -> list[float]:
    """随机选股基线收益。"""
    all_codes = sorted([
        d.name for d in (DATA_ROOT / "kline").iterdir()
        if d.is_dir() and (d / "D.parquet").exists()
    ])
    seed = int(hashlib.md5(snapshot_date.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(seed)
    sampled = rng.choice(all_codes, size=min(n_random, len(all_codes)), replace=False)
    returns = []
    for code in sampled:
        r = get_forward_return(code, snapshot_date)
        if r is not None:
            returns.append(r)
    return returns


def main():
    codes = sorted([
        d.name for d in (DATA_ROOT / "kline").iterdir()
        if d.is_dir() and (d / "D.parquet").exists()
    ])
    print(f"股票池: {len(codes)}只")
    print(f"快照日期: {SNAPSHOT_DATES}")
    print(f"前瞻天数: {FORWARD_DAYS}")
    print("=" * 90)

    wp = WavePositionScreener(data_root=DATA_ROOT)
    from scripts.screener.multi_timeframe_scorer import MultiTimeframeScorer
    mtf = MultiTimeframeScorer(data_root=DATA_ROOT)

    # 收集所有快照结果
    ab_returns_all = []     # A/B级收益
    c_returns_all = []      # C级收益
    mtf_confirmed_all = []  # MTF确认的W3收益
    mtf_rejected_all = []   # MTF未确认的W3收益
    random_returns_all = [] # 随机基线

    for sd in SNAPSHOT_DATES:
        print(f"\n--- {sd} ---")

        results = wp.scan(codes, min_score=0, top=50, as_of_date=sd)

        if len(results) == 0:
            print("  无结果")
            continue

        # 分组计算收益
        ab_codes = results[results["grade"].isin(["A", "B"])]["code"].tolist()
        c_codes = results[results["grade"] == "C"]["code"].tolist()
        all_candidate_codes = results["code"].tolist()

        ab_rets = [r for r in (get_forward_return(c, sd) for c in ab_codes) if r is not None]
        c_rets = [r for r in (get_forward_return(c, sd) for c in c_codes) if r is not None]

        # MTF确认分组（公平对比：同一候选池，按MTF拆分）
        mtf_df = mtf.score_batch(all_candidate_codes, cutoff_date=sd)
        mtf_confirmed_codes = []
        mtf_rejected_codes = []
        if len(mtf_df) > 0:
            confirmed_mask = ~mtf_df["mtf_level"].isin(["单周期信号", "无共识", "数据不足"])
            mtf_confirmed_codes = mtf_df[confirmed_mask]["code"].tolist()
            mtf_rejected_codes = mtf_df[~confirmed_mask]["code"].tolist()

        mtf_conf_rets = [r for r in (get_forward_return(c, sd) for c in mtf_confirmed_codes) if r is not None]
        mtf_rej_rets = [r for r in (get_forward_return(c, sd) for c in mtf_rejected_codes) if r is not None]

        rand_rets = random_baseline(len(codes), sd, n_random=50)
        random_returns_all.extend(rand_rets)

        def fmt(returns):
            if not returns:
                return "N/A"
            arr = np.array(returns)
            return f"avg={np.mean(arr):+.2f}% hit={np.mean(arr>0):.0%} n={len(arr)}"

        all_rets = [r for r in (get_forward_return(c, sd) for c in all_candidate_codes) if r is not None]
        print(f"  W3全量:      {fmt(all_rets)}")
        print(f"  ├─ MTF确认:  {fmt(mtf_conf_rets)}")
        print(f"  ├─ MTF未确认:{fmt(mtf_rej_rets)}")
        print(f"  ├─ A/B级:    {fmt(ab_rets)}")
        print(f"  └─ 随机基线: {fmt(rand_rets)}")

        ab_returns_all.extend(ab_rets)
        c_returns_all.extend(c_rets)
        mtf_confirmed_all.extend(mtf_conf_rets)
        mtf_rejected_all.extend(mtf_rej_rets)

    # ── 汇总 ──
    print("\n" + "=" * 90)
    print("汇总（所有快照合并）")
    print("-" * 90)

    ab = np.array(ab_returns_all)
    c = np.array(c_returns_all)
    mtf_c = np.array(mtf_confirmed_all)
    mtf_r = np.array(mtf_rejected_all)
    rnd = np.array(random_returns_all)

    for name, arr in [
        ("W3+MTF确认", mtf_c),
        ("W3+MTF未确认", mtf_r),
        ("W3 A/B级", ab),
        ("随机基线", rnd),
    ]:
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

    # ── 显著性检验 ──
    print("\n" + "-" * 90)
    print("显著性检验")

    from scipy.stats import mannwhitneyu

    if len(mtf_c) >= 5 and len(rnd) >= 5:
        stat, pval = mannwhitneyu(mtf_c, rnd, alternative='greater')
        print(f"  MTF确认 vs 随机:   p={pval:.4f} {'*显著*' if pval < 0.05 else '(不显著)'}")

    if len(mtf_c) >= 5 and len(mtf_r) >= 5:
        stat, pval = mannwhitneyu(mtf_c, mtf_r, alternative='greater')
        print(f"  MTF确认 vs 未确认: p={pval:.4f} {'*显著*' if pval < 0.05 else '(不显著)'}")

    if len(mtf_c) >= 5 and len(rnd) >= 5:
        diff = np.mean(mtf_c) - np.mean(rnd)
        print(f"\n  MTF确认超额收益: {diff:+.2f}% (vs 随机基线)")
    if len(mtf_c) >= 5 and len(mtf_r) >= 5:
        diff = np.mean(mtf_c) - np.mean(mtf_r)
        print(f"  MTF确认 vs 未确认: {diff:+.2f}%")


if __name__ == "__main__":
    main()
