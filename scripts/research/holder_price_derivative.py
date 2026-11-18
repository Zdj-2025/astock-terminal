#!/usr/bin/env python3
"""股东人数一阶导+二阶导因子分析。

在 V2 研究基础上，新增：
  1. 一阶导(d1) = 股东人数变化率（已有，对比用）
  2. 二阶导(d2) = 变化率的变化率（加速度）
  3. 单因子IC/ICIR对比
  4. 四象限分组分析
  5. 极端信号分析
  6. d2分层回测 + d1×d2双因子5×5分层
  7. 因子衰减分析

运行方式：
  cd .
  PYTHONPATH=. python scripts/research/holder_price_derivative.py

输出：
  scripts/research/output/holder_price_derivative_report.md
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "holder_price_derivative_report.md"


# ══════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════

def load_cached_data() -> pd.DataFrame:
    """从缓存加载已处理好的指标数据。"""
    cache_path = OUTPUT_DIR / "_cache_metrics.parquet"
    if not cache_path.exists():
        print("❌ 未找到缓存数据 _cache_metrics.parquet，请先运行 V2 脚本")
        sys.exit(1)

    df = pd.read_parquet(cache_path)
    print(f"[数据] 加载缓存: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    print(f"  季度范围: {df['quarter'].min()} ~ {df['quarter'].max()}")
    return df


# ══════════════════════════════════════════════════════════════════════
#  因子计算
# ══════════════════════════════════════════════════════════════════════

def compute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """计算一阶导和二阶导因子。

    d1 = (当季人数 - 上季人数) / 上季人数 * 100  (已有 holder_chg_pct)
    d2_relative = (当季d1 - 上季d1) / |上季d1| * 100  (归一化二阶导)
    d2_absolute = 当季d1 - 上季d1  (绝对差分版)
    """
    print("\n[因子] 计算一阶导+二阶导...")
    df = df.sort_values(["ts_code", "quarter"]).copy()

    # d1 已经存在 (holder_chg_pct)，确认一下
    assert "holder_chg_pct" in df.columns, "缺少 holder_chg_pct 列"
    df["d1"] = df["holder_chg_pct"]

    # d2_absolute: 简单差分
    df["d2_abs"] = df.groupby("ts_code")["d1"].diff()

    # d2_relative: 归一化二阶导，避免除零
    prev_d1 = df.groupby("ts_code")["d1"].shift(1)
    df["d2_rel"] = np.where(
        prev_d1.abs() > 0.01,  # 避免 |上季d1| ≈ 0 时除零
        df["d2_abs"] / prev_d1.abs() * 100,
        0.0
    )

    # 裁剪极端值 (d2_rel 可能有极端异常)
    df["d2_rel"] = df["d2_rel"].clip(-500, 500)

    # 后续1季、2季收益率
    df["ret_1q"] = df.groupby("ts_code")["quarter_return_pct"].shift(-1)
    df["ret_2q"] = df.groupby("ts_code")["quarter_return_pct"].shift(-1) + \
                    df.groupby("ts_code")["quarter_return_pct"].shift(-2)

    # 需要 d2 非空，所以去掉前两行（第一个季度无 d1，第二个季度无 d2）
    df = df.dropna(subset=["d1", "d2_abs", "ret_1q"]).reset_index(drop=True)

    print(f"  有效记录（含d2）: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    print(f"  d1 统计: mean={df['d1'].mean():.2f}%, std={df['d1'].std():.2f}%")
    print(f"  d2_abs 统计: mean={df['d2_abs'].mean():.2f}%, std={df['d2_abs'].std():.2f}%")
    print(f"  d2_rel 统计: mean={df['d2_rel'].mean():.2f}%, std={df['d2_rel'].std():.2f}%")
    return df


# ══════════════════════════════════════════════════════════════════════
#  1. 单因子检验
# ══════════════════════════════════════════════════════════════════════

def single_factor_ic(df: pd.DataFrame, factor_col: str, return_col: str) -> dict:
    """计算单因子的 IC/ICIR。"""
    results = []
    for q in sorted(df["quarter"].unique()):
        sub = df[df["quarter"] == q].dropna(subset=[factor_col, return_col])
        if len(sub) < 30:
            continue
        ic, p = stats.spearmanr(sub[factor_col], sub[return_col])
        results.append({"quarter": q, "ic": ic, "p_value": p, "n": len(sub)})

    if not results:
        return {"ic_mean": np.nan, "icir": np.nan, "ic_positive_ratio": np.nan, "details": []}

    ic_series = [r["ic"] for r in results]
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    icir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_ratio": sum(1 for x in ic_series if x > 0) / len(ic_series),
        "details": results,
    }


def single_factor_analysis(df: pd.DataFrame) -> dict:
    """d1 vs d2 的单因子IC/ICIR对比。"""
    print("\n" + "=" * 60)
    print("1. 单因子检验（d1 vs d2）")
    print("=" * 60)

    results = {}
    for factor, label in [("d1", "一阶导(d1)"), ("d2_abs", "二阶导-绝对(d2_abs)"),
                           ("d2_rel", "二阶导-归一(d2_rel)")]:
        for ret, ret_label in [("ret_1q", "后1季"), ("ret_2q", "后2季")]:
            key = f"{factor}_{ret}"
            r = single_factor_ic(df, factor, ret)
            results[key] = r
            print(f"  {label} × {ret_label}: IC={r['ic_mean']:.4f}, "
                  f"ICIR={r['icir']:.4f}, IC>0占比={r['ic_positive_ratio']:.1%}")

    return results


# ══════════════════════════════════════════════════════════════════════
#  2. 组合因子
# ══════════════════════════════════════════════════════════════════════

def combo_factor_analysis(df: pd.DataFrame) -> dict:
    """d1和d2的相关性 + 四象限分组分析。"""
    print("\n" + "=" * 60)
    print("2. 组合因子分析")
    print("=" * 60)

    # Spearman 相关
    corr_abs, _ = stats.spearmanr(df["d1"], df["d2_abs"])
    corr_rel, _ = stats.spearmanr(df["d1"], df["d2_rel"])
    print(f"  d1 vs d2_abs Spearman: {corr_abs:.4f}")
    print(f"  d1 vs d2_rel Spearman: {corr_rel:.4f}")

    # 四象限分组
    # 用 d2_abs（更直观）
    def quadrant(row):
        if row["d1"] < 0 and row["d2_abs"] < 0:
            return "A:集中加速"  # d1<0, d2<0
        elif row["d1"] < 0 and row["d2_abs"] >= 0:
            return "B:集中减速"  # d1<0, d2>0
        elif row["d1"] >= 0 and row["d2_abs"] >= 0:
            return "C:分散加速"  # d1>0, d2>0
        else:
            return "D:分散减速"  # d1>0, d2<0

    df["quadrant"] = df.apply(quadrant, axis=1)

    quad_results = []
    for grp in ["A:集中加速", "B:集中减速", "C:分散加速", "D:分散减速"]:
        sub = df[df["quadrant"] == grp]
        if len(sub) < 10:
            continue
        quad_results.append({
            "group": grp,
            "count": len(sub),
            "count_pct": len(sub) / len(df) * 100,
            "ret_1q_mean": sub["ret_1q"].mean(),
            "ret_1q_median": sub["ret_1q"].median(),
            "ret_1q_winrate": (sub["ret_1q"] > 0).mean() * 100,
            "ret_2q_mean": sub["ret_2q"].mean(),
            "ret_2q_median": sub["ret_2q"].median(),
            "ret_2q_winrate": (sub["ret_2q"] > 0).mean() * 100,
        })

    quad_df = pd.DataFrame(quad_results)
    print(f"\n  四象限分组结果：")
    print(f"  {'组':>10s} {'占比':>6s} {'后1季收益':>10s} {'后1季胜率':>8s} {'后2季收益':>10s} {'后2季胜率':>8s}")
    for _, r in quad_df.iterrows():
        print(f"  {r['group']:>10s} {r['count_pct']:5.1f}% {r['ret_1q_mean']:+9.2f}% "
              f"{r['ret_1q_winrate']:7.1f}% {r['ret_2q_mean']:+9.2f}% {r['ret_2q_winrate']:7.1f}%")

    return {
        "corr_d1_d2_abs": corr_abs,
        "corr_d1_d2_rel": corr_rel,
        "quadrant_df": quad_df,
    }


# ══════════════════════════════════════════════════════════════════════
#  3. 极端信号分析
# ══════════════════════════════════════════════════════════════════════

def extreme_signal_analysis(df: pd.DataFrame) -> dict:
    """极端信号：d1<-10% 的子集中，d2<0 vs d2>0 的对比。"""
    print("\n" + "=" * 60)
    print("3. 极端信号分析")
    print("=" * 60)

    # d1 < -10% 的子集
    strong = df[df["d1"] < -10].copy()
    print(f"  d1<-10% 样本数: {len(strong)} ({len(strong)/len(df)*100:.1f}%)")

    results = []
    for label, cond in [
        ("强集中+加速(d1<-10%,d2<0)", (strong["d2_abs"] < 0)),
        ("强集中+减速(d1<-10%,d2>0)", (strong["d2_abs"] >= 0)),
        ("强分散+加速(d1>10%,d2>0)", None),  # 补充反向
        ("强分散+减速(d1>10%,d2<0)", None),
    ]:
        if "强集中" in label:
            sub = strong[cond]
        elif "强分散+加速" in label:
            sub = df[(df["d1"] > 10) & (df["d2_abs"] > 0)]
        else:
            sub = df[(df["d1"] > 10) & (df["d2_abs"] < 0)]

        if len(sub) < 10:
            results.append({"label": label, "count": len(sub)})
            print(f"  {label}: N={len(sub)} (不足)")
            continue

        results.append({
            "label": label,
            "count": len(sub),
            "ret_1q_mean": sub["ret_1q"].mean(),
            "ret_1q_median": sub["ret_1q"].median(),
            "ret_1q_winrate": (sub["ret_1q"] > 0).mean() * 100,
            "ret_2q_mean": sub["ret_2q"].mean(),
            "ret_2q_winrate": (sub["ret_2q"] > 0).mean() * 100,
        })
        r = results[-1]
        print(f"  {label}: N={r['count']}, 后1季={r['ret_1q_mean']:+.2f}%, "
              f"胜率={r['ret_1q_winrate']:.1f}%, 后2季={r['ret_2q_mean']:+.2f}%")

    return {"extreme_df": pd.DataFrame(results)}


# ══════════════════════════════════════════════════════════════════════
#  4. 分层回测
# ══════════════════════════════════════════════════════════════════════

def d2_quintile_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """按 d2 分5档回测。"""
    print("\n" + "=" * 60)
    print("4a. d2 五档分层回测")
    print("=" * 60)

    all_results = []
    for q in sorted(df["quarter"].unique()):
        sub = df[df["quarter"] == q].dropna(subset=["d2_abs", "ret_1q"])
        if len(sub) < 50:
            continue
        try:
            sub = sub.copy()
            sub["quintile"] = pd.qcut(sub["d2_abs"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                                       duplicates="drop")
        except ValueError:
            continue
        for qname in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            grp = sub[sub["quintile"] == qname]
            if len(grp) < 5:
                continue
            all_results.append({
                "quarter": q, "quintile": qname, "count": len(grp),
                "ret_1q_mean": grp["ret_1q"].mean(),
                "ret_1q_winrate": (grp["ret_1q"] > 0).mean() * 100,
                "ret_2q_mean": grp["ret_2q"].mean(),
            })

    if not all_results:
        print("  ❌ 分层回测无结果")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_results)
    summary = result_df.groupby("quintile").agg(
        avg_ret_1q=("ret_1q_mean", "mean"),
        avg_wr_1q=("ret_1q_winrate", "mean"),
        avg_ret_2q=("ret_2q_mean", "mean"),
        avg_n=("count", "mean"),
    ).reset_index()

    print(f"  d2 五档分层结果：")
    for _, r in summary.iterrows():
        print(f"    {r['quintile']}: 后1季={r['avg_ret_1q']:+.2f}%, "
              f"胜率={r['avg_wr_1q']:.1f}%, 后2季={r['avg_ret_2q']:+.2f}%")

    rets = summary.set_index("quintile")["avg_ret_1q"]
    mono = all(rets.iloc[i] >= rets.iloc[i+1] for i in range(len(rets)-1)) or \
           all(rets.iloc[i] <= rets.iloc[i+1] for i in range(len(rets)-1))
    print(f"  单调性: {'✅ 单调' if mono else '❌ 非单调'}")

    result_df.attrs["summary"] = summary
    result_df.attrs["monotonic"] = mono
    return result_df


def dual_5x5_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """d1 × d2 双因子 5×5 分层回测。"""
    print("\n" + "=" * 60)
    print("4b. d1×d2 双因子 5×5 分层")
    print("=" * 60)

    all_results = []
    for q in sorted(df["quarter"].unique()):
        sub = df[df["quarter"] == q].dropna(subset=["d1", "d2_abs", "ret_1q"]).copy()
        if len(sub) < 100:
            continue
        try:
            sub["d1_q"] = pd.qcut(sub["d1"], 5, labels=False, duplicates="drop")  # 0-4
            sub["d2_q"] = pd.qcut(sub["d2_abs"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue

        for i in range(5):
            for j in range(5):
                grp = sub[(sub["d1_q"] == i) & (sub["d2_q"] == j)]
                if len(grp) < 3:
                    continue
                all_results.append({
                    "quarter": q,
                    "d1_quintile": f"Q{i+1}",
                    "d2_quintile": f"Q{j+1}",
                    "count": len(grp),
                    "ret_1q_mean": grp["ret_1q"].mean(),
                    "ret_1q_winrate": (grp["ret_1q"] > 0).mean() * 100,
                })

    if not all_results:
        print("  ❌ 5×5 分层无结果")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_results)

    # 汇总为 5×5 矩阵
    matrix = result_df.groupby(["d1_quintile", "d2_quintile"]).agg(
        avg_ret=("ret_1q_mean", "mean"),
        avg_wr=("ret_1q_winrate", "mean"),
        avg_n=("count", "mean"),
    ).reset_index()

    # 打印矩阵
    print(f"\n  d1×d2 收益率矩阵（d1\\d2）：")
    print(f"  {'':>6s}", end="")
    for j in range(1, 6):
        print(f"  {'Q'+str(j):>8s}", end="")
    print()
    for i in range(1, 6):
        print(f"  {'Q'+str(i):>6s}", end="")
        for j in range(1, 6):
            val = matrix[(matrix["d1_quintile"] == f"Q{i}") & (matrix["d2_quintile"] == f"Q{j}")]
            if not val.empty:
                print(f"  {val.iloc[0]['avg_ret']:+7.2f}%", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        print()

    print(f"\n  说明：Q1=最小(最集中/减速最大) Q5=最大(最分散/加速最大)")
    result_df.attrs["matrix"] = matrix
    return result_df


# ══════════════════════════════════════════════════════════════════════
#  5. 因子衰减
# ══════════════════════════════════════════════════════════════════════

def factor_decay(df: pd.DataFrame) -> pd.DataFrame:
    """d2 的滞后1-4季 IC。"""
    print("\n" + "=" * 60)
    print("5. 因子衰减分析")
    print("=" * 60)

    results = []
    for lag in [1, 2, 3, 4]:
        # 计算滞后 lag 季度的累计收益
        df_shifted = df.sort_values(["ts_code", "quarter"]).copy()
        ret_parts = []
        for k in range(lag):
            ret_parts.append(df_shifted.groupby("ts_code")["quarter_return_pct"].shift(-(k+1)))
        df_shifted[f"ret_lag{lag}"] = sum(ret_parts)
        df_shifted = df_shifted.dropna(subset=["d2_abs", f"ret_lag{lag}"])

        if len(df_shifted) < 50:
            continue

        # 按季度算 IC
        ic_list = []
        for q in sorted(df_shifted["quarter"].unique()):
            sub = df_shifted[df_shifted["quarter"] == q]
            if len(sub) < 30:
                continue
            ic, _ = stats.spearmanr(sub["d2_abs"], sub[f"ret_lag{lag}"])
            ic_list.append(ic)

        if ic_list:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            results.append({
                "lag": lag, "ic_mean": ic_mean, "ic_std": ic_std,
                "icir": icir, "n_quarters": len(ic_list),
            })
            print(f"  滞后{lag}季: IC={ic_mean:.4f}, ICIR={icir:.4f}, N_quarters={len(ic_list)}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════
#  报告生成
# ══════════════════════════════════════════════════════════════════════

def generate_report(df, sf_results, combo_results, extreme_results,
                    d2_qt, dual_5x5, decay_df) -> str:
    print("\n" + "=" * 60)
    print("生成研究报告")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    def w(t=""):
        L.append(t)

    w("# 股东人数一阶导+二阶导因子分析报告")
    w()
    w(f"> 生成时间：{now}")
    w(f"> 数据范围：{df['quarter'].min()} ~ {df['quarter'].max()}")
    w(f"> 样本股票数：{df['ts_code'].nunique()}")
    w(f"> 有效记录数：{len(df)}")
    w()
    w("---")
    w()

    # ── 1. 因子定义 ──
    w("## 1. 因子定义")
    w()
    w("| 因子 | 公式 | 含义 |")
    w("|------|------|------|")
    w("| d1（一阶导） | (当季人数 - 上季人数) / 上季人数 × 100 | 股东人数变化率 |")
    w("| d2_abs（二阶导-绝对） | 当季d1 - 上季d1 | 变化率的绝对变化 |")
    w("| d2_rel（二阶导-归一） | (当季d1 - 上季d1) / \\|上季d1\\| × 100 | 变化率的相对加速度 |")
    w()
    w("> d2_abs 更直观，d2_rel 更适合不同基数股票的比较。本报告主要使用 d2_abs。")
    w()

    # ── 2. 单因子检验 ──
    w("## 2. 单因子检验（IC/ICIR对比）")
    w()
    w("| 因子 | 预测目标 | IC均值 | IC标准差 | ICIR | IC>0占比 | 判断 |")
    w("|------|----------|--------|----------|------|---------|------|")
    for key, r in sf_results.items():
        factor_label = key.split("_")[0]
        ret_label = "后1季" if "ret_1q" in key else "后2季"
        factor_names = {"d1": "d1(一阶导)", "d2_abs": "d2_abs", "d2_rel": "d2_rel"}
        icir = r["icir"]
        judge = "有效" if abs(icir) > 0.5 else "较弱" if abs(icir) > 0.2 else "无效"
        w(f"| {factor_names.get(factor_label, factor_label)} | {ret_label} | "
          f"{r['ic_mean']:.4f} | {r['ic_std']:.4f} | {icir:.4f} | "
          f"{r['ic_positive_ratio']:.1%} | {judge} |")
    w()

    # 找出最优因子
    best_key = max(sf_results, key=lambda k: abs(sf_results[k].get("icir", 0)))
    best = sf_results[best_key]
    w(f"**最优因子**: {best_key}，ICIR={best['icir']:.4f}")
    w()

    # ── 3. 组合因子 ──
    w("## 3. 组合因子分析")
    w()
    w("### 3.1 d1与d2的相关性")
    w()
    w(f"- d1 vs d2_abs Spearman: **{combo_results['corr_d1_d2_abs']:.4f}**")
    w(f"- d1 vs d2_rel Spearman: **{combo_results['corr_d1_d2_rel']:.4f}**")
    w()
    if abs(combo_results["corr_d1_d2_abs"]) < 0.3:
        w("> ✅ 相关性较低，d2 提供了增量信息，两个因子可以组合使用。")
    elif abs(combo_results["corr_d1_d2_abs"]) < 0.6:
        w("> ⚠️ 中等相关，d2 有一定增量信息，但存在共线性风险。")
    else:
        w("> ❌ 高度相关，d2 增量信息有限。")
    w()

    w("### 3.2 四象限分组")
    w()
    w("| 组 | 条件 | 含义 | 占比 | 后1季收益 | 后1季胜率 | 后2季收益 | 后2季胜率 |")
    w("|----|------|------|------|----------|----------|----------|----------|")
    quad_df = combo_results["quadrant_df"]
    for _, r in quad_df.iterrows():
        w(f"| {r['group']} | — | — | {r['count_pct']:.1f}% | "
          f"{r['ret_1q_mean']:+.2f}% | {r['ret_1q_winrate']:.1f}% | "
          f"{r['ret_2q_mean']:+.2f}% | {r['ret_2q_winrate']:.1f}% |")
    w()

    if not quad_df.empty:
        a_row = quad_df[quad_df["group"] == "A:集中加速"]
        d_row = quad_df[quad_df["group"] == "D:分散减速"]
        if not a_row.empty and not d_row.empty:
            spread = a_row.iloc[0]["ret_1q_mean"] - d_row.iloc[0]["ret_1q_mean"]
            w(f"**A组 vs D组后1季收益差**: {spread:+.2f}%")
            w()

    # ── 4. 极端信号 ──
    w("## 4. 极端信号分析")
    w()
    ext_df = extreme_results["extreme_df"]
    w("| 信号 | 样本量 | 后1季收益 | 后1季胜率 | 后2季收益 | 后2季胜率 |")
    w("|------|--------|----------|----------|----------|----------|")
    for _, r in ext_df.iterrows():
        ret1 = f"{r['ret_1q_mean']:+.2f}%" if "ret_1q_mean" in r and not pd.isna(r.get("ret_1q_mean")) else "N/A"
        wr1 = f"{r['ret_1q_winrate']:.1f}%" if "ret_1q_winrate" in r and not pd.isna(r.get("ret_1q_winrate")) else "N/A"
        ret2 = f"{r['ret_2q_mean']:+.2f}%" if "ret_2q_mean" in r and not pd.isna(r.get("ret_2q_mean")) else "N/A"
        wr2 = f"{r['ret_2q_winrate']:.1f}%" if "ret_2q_winrate" in r and not pd.isna(r.get("ret_2q_winrate")) else "N/A"
        w(f"| {r['label']} | {r['count']} | {ret1} | {wr1} | {ret2} | {wr2} |")
    w()

    # 对比加速 vs 减速
    strong_accel = ext_df[ext_df["label"].str.contains("强集中.*加速")]
    strong_decel = ext_df[ext_df["label"].str.contains("强集中.*减速")]
    if not strong_accel.empty and not strong_decel.empty:
        sa = strong_accel.iloc[0]
        sd = strong_decel.iloc[0]
        if "ret_1q_mean" in sa and not pd.isna(sa.get("ret_1q_mean")):
            diff = sa["ret_1q_mean"] - sd["ret_1q_mean"]
            w(f"**强集中加速 vs 强集中减速收益差**: {diff:+.2f}%")
            if diff > 0:
                w("> 加速集中（d2<0）的股票比集中减速（d2>0）表现更好，"
                  "说明'加速度'方向是重要信号。")
            else:
                w("> 减速集中的股票表现不差于甚至好于加速集中，"
                  "说明二阶导在极端情况下增量有限。")
            w()

    # ── 5. 分层回测 ──
    w("## 5. 分层回测")
    w()
    w("### 5.1 d2 五档分层")
    w()
    w("按 d2_abs 分5档，Q1=d2最小（加速集中最大）→ Q5=d2最大（加速分散最大）")
    w()

    if not d2_qt.empty and "summary" in d2_qt.attrs:
        summary = d2_qt.attrs["summary"]
        w("| 分档 | 后1季收益 | 后1季胜率 | 后2季收益 | 平均样本量 |")
        w("|------|----------|----------|----------|-----------|")
        for _, r in summary.iterrows():
            w(f"| {r['quintile']} | {r['avg_ret_1q']:+.2f}% | {r['avg_wr_1q']:.1f}% | "
              f"{r['avg_ret_2q']:+.2f}% | {r['avg_n']:.0f} |")
        w()
        w(f"**单调性**: {'✅ 单调' if d2_qt.attrs.get('monotonic') else '❌ 非单调'}")
        w()

    w("### 5.2 d1×d2 双因子 5×5 矩阵")
    w()
    w("行=d1分档(Q1=最集中→Q5=最分散)，列=d2分档(Q1=减速最大→Q5=加速最大)")
    w()

    if not dual_5x5.empty and "matrix" in dual_5x5.attrs:
        matrix = dual_5x5.attrs["matrix"]
        # 表头
        w("| d1\\d2 | Q1 | Q2 | Q3 | Q4 | Q5 |")
        w("|--------|------|------|------|------|------|")
        for i in range(1, 6):
            row_vals = []
            for j in range(1, 6):
                val = matrix[(matrix["d1_quintile"] == f"Q{i}") & (matrix["d2_quintile"] == f"Q{j}")]
                if not val.empty:
                    row_vals.append(f"{val.iloc[0]['avg_ret']:+.2f}%")
                else:
                    row_vals.append("N/A")
            w(f"| Q{i} | {' | '.join(row_vals)} |")
        w()

        # 最优和最差组合
        if not matrix.empty:
            best_combo = matrix.loc[matrix["avg_ret"].idxmax()]
            worst_combo = matrix.loc[matrix["avg_ret"].idxmin()]
            w(f"**最优组合**: d1={best_combo['d1_quintile']}, d2={best_combo['d2_quintile']}, "
              f"收益={best_combo['avg_ret']:+.2f}%")
            w(f"**最差组合**: d1={worst_combo['d1_quintile']}, d2={worst_combo['d2_quintile']}, "
              f"收益={worst_combo['avg_ret']:+.2f}%")
            w(f"**多空价差**: {best_combo['avg_ret'] - worst_combo['avg_ret']:+.2f}%")
            w()

    # ── 6. 因子衰减 ──
    w("## 6. 因子衰减（d2的滞后IC）")
    w()
    if not decay_df.empty:
        w("| 滞后期 | IC均值 | IC标准差 | ICIR | 季度数 |")
        w("|--------|--------|----------|------|--------|")
        for _, r in decay_df.iterrows():
            w(f"| {r['lag']}季后 | {r['ic_mean']:.4f} | {r['ic_std']:.4f} | "
              f"{r['icir']:.4f} | {r['n_quarters']} |")
        w()

        ic1 = decay_df.iloc[0]["ic_mean"]
        ic_last = decay_df.iloc[-1]["ic_mean"]
        if abs(ic_last) < abs(ic1) * 0.5:
            w(f"**衰减明显**: IC 从 {ic1:.4f} 衰减到 {ic_last:.4f}")
        else:
            w(f"**衰减缓慢**: IC 从 {ic1:.4f} 到 {ic_last:.4f}")
        w()
    else:
        w("衰减分析数据不足。")
        w()

    # ── 7. 综合结论 ──
    w("## 7. 综合结论")
    w()

    conclusions = []

    # d1 vs d2 预测力
    d1_icir = sf_results.get("d1_ret_1q", {}).get("icir", 0)
    d2_icir = sf_results.get("d2_abs_ret_1q", {}).get("icir", 0)
    if abs(d2_icir) > abs(d1_icir):
        conclusions.append(f"1. **d2预测力优于d1**：d2_abs的ICIR={d2_icir:.4f}，"
                          f"d1的ICIR={d1_icir:.4f}，二阶导提供了更强的预测信号。")
    else:
        conclusions.append(f"1. **d1预测力优于d2**：d1的ICIR={d1_icir:.4f}，"
                          f"d2_abs的ICIR={d2_icir:.4f}，二阶导的增量信息有限。")

    # 共线性
    corr = abs(combo_results["corr_d1_d2_abs"])
    if corr < 0.3:
        conclusions.append(f"2. **d1与d2低相关**({corr:.4f})，d2确实提供了独立增量信息。")
    elif corr < 0.6:
        conclusions.append(f"2. **d1与d2中等相关**({corr:.4f})，d2有一定增量，但需注意共线性。")
    else:
        conclusions.append(f"2. **d1与d2高度相关**({corr:.4f})，d2的增量信息有限。")

    # 四象限
    if not quad_df.empty:
        a_row = quad_df[quad_df["group"] == "A:集中加速"]
        c_row = quad_df[quad_df["group"] == "C:分散加速"]
        if not a_row.empty and not c_row.empty:
            conclusions.append(
                f"3. **四象限差异明显**：A组（集中加速）后1季收益{a_row.iloc[0]['ret_1q_mean']:+.2f}%，"
                f"C组（分散加速）{c_row.iloc[0]['ret_1q_mean']:+.2f}%，"
                f"价差{a_row.iloc[0]['ret_1q_mean'] - c_row.iloc[0]['ret_1q_mean']:+.2f}%。"
            )

    # 极端信号
    if not ext_df.empty:
        strong_a = ext_df[ext_df["label"].str.contains("强集中.*加速")]
        if not strong_a.empty and "ret_1q_mean" in strong_a.columns:
            sa = strong_a.iloc[0]
            if not pd.isna(sa.get("ret_1q_mean")):
                conclusions.append(
                    f"4. **极端信号有效**：强集中+加速(d1<-10%,d2<0)后1季收益"
                    f"{sa['ret_1q_mean']:+.2f}%，胜率{sa['ret_1q_winrate']:.1f}%。"
                )

    # 分层
    if not d2_qt.empty and d2_qt.attrs.get("monotonic"):
        conclusions.append("5. **d2分层收益单调**，作为独立选股因子有效。")
    elif not d2_qt.empty:
        conclusions.append("5. **d2分层收益非单调**，单独使用效果有限，建议与d1组合。")

    for c in conclusions:
        w(f"- {c}")
    w()

    # ── 8. 策略建议 ──
    w("## 8. 策略建议")
    w()
    w("1. **d1+d2组合优于单因子**：四象限分析显示组合效果更好")
    w("2. **重点关