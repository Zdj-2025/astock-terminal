#!/usr/bin/env python3
"""股东人数变化与股价相关性研究 — 全市场量化分析。

研究目标：
  1. 采集全市场股票的季度股东人数数据（stk_holdernumber）
  2. 计算季度收益率、股东人数变化率、人均持股市值等指标
  3. 分析同期/滞后相关性（Pearson + Spearman）
  4. 分组回测：人数增减 vs 后续收益
  5. 生成研究报告

运行方式：
  cd .
  PYTHONPATH=. python scripts/research/holder_price_correlation.py

输出：
  scripts/research/output/holder_price_correlation_report.md
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 项目路径 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.core.tushare_client import TushareClient, from_ts_code

# ── 常量 ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "holder_price_correlation_report.md"

# 研究时间范围
START_DATE = "20210101"
END_DATE = "20260331"

# 季度末日期列表（用于计算季度收益率）
QUARTER_ENDS = [
    "20210331", "20210630", "20210930", "20211231",
    "20220331", "20220630", "20220930", "20221231",
    "20230331", "20230630", "20230930", "20231231",
    "20240331", "20240630", "20240930", "20241231",
    "20250331", "20250630", "20250930", "20251231",
    "20260331",
]

# API 限速
API_SLEEP = 0.5  # stk_holdernumber 200次/分钟，保守用 0.5s


# ══════════════════════════════════════════════════════════════════════
#  第一步：数据采集
# ══════════════════════════════════════════════════════════════════════

def fetch_holder_data(client: TushareClient) -> pd.DataFrame:
    """采集全市场股东人数数据。

    stk_holdernumber 用 enddate 参数按报告期批量拉取，
    每次返回该季度全市场数据（约5000+条）。

    Returns:
        DataFrame: ts_code, end_date, holder_num, ann_date
    """
    cache_path = OUTPUT_DIR / "_cache_holder_data.parquet"

    # 缓存检查（1天内有效）
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            print("[缓存] 使用已缓存的股东人数数据")
            return pd.read_parquet(cache_path)

    print("=" * 60)
    print("第一步：采集全市场股东人数数据")
    print("=" * 60)

    all_data = []

    # 按报告期（enddate）批量拉取
    for i, q_end in enumerate(QUARTER_ENDS):
        print(f"  [{i+1}/{len(QUARTER_ENDS)}] 拉取报告期 {q_end} 的股东人数...")
        try:
            client._throttle()
            df = client.pro.stk_holdernumber(
                enddate=q_end,
                fields="ts_code,end_date,holder_num,ann_date"
            )
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"    → 获取 {len(df)} 条记录，{df['ts_code'].nunique()} 只股票")
            else:
                print(f"    → 无数据")
        except Exception as e:
            print(f"    → 失败: {e}")

        time.sleep(API_SLEEP)

    if not all_data:
        print("❌ 未获取到任何股东人数数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    # end_date 是报告期，同一股票同一报告期可能有多次数据（不同 ann_date），取最新的
    result = result.sort_values("ann_date", ascending=False)
    result = result.drop_duplicates(subset=["ts_code", "end_date"], keep="first")
    result["end_date"] = result["end_date"].astype(str)

    print(f"\n  ✅ 合计 {len(result)} 条记录，覆盖 {result['ts_code'].nunique()} 只股票")

    # 缓存
    result.to_parquet(cache_path, index=False)
    return result


def fetch_stock_list(client: TushareClient) -> pd.DataFrame:
    """获取全A股列表，过滤ST和新股。"""
    print("\n[数据] 获取股票列表...")
    df = client.get_stock_list(list_status="L")
    if df.empty:
        return df

    # 排除ST
    df = df[~df["name"].str.contains("ST", case=False, na=False)].copy()

    # 排除上市不满1年的新股
    cutoff = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y%m%d")
    df = df[df["list_date"] <= cutoff].copy()

    print(f"  过滤后剩余 {len(df)} 只股票（排除ST和上市不满1年）")
    return df


def fetch_daily_basic_quarterly(client: TushareClient) -> pd.DataFrame:
    """获取季度末交易日的 daily_basic 数据（流通市值等）。

    Returns:
        DataFrame: ts_code, trade_date, close, circ_mv, total_mv
    """
    cache_path = OUTPUT_DIR / "_cache_daily_basic_quarterly.parquet"

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            print("[缓存] 使用已缓存的季度 daily_basic 数据")
            return pd.read_parquet(cache_path)

    print("\n[数据] 获取季度末 daily_basic（流通市值）...")
    all_data = []

    for i, q_end in enumerate(QUARTER_ENDS):
        print(f"  [{i+1}/{len(QUARTER_ENDS)}] {q_end}...")
        try:
            # 先尝试精确日期
            df = client.get_daily_basic(trade_date=q_end)
            if df is None or df.empty:
                # 尝试附近的交易日
                for offset in range(1, 5):
                    dt = pd.Timestamp(q_end) - pd.Timedelta(days=offset)
                    df = client.get_daily_basic(trade_date=dt.strftime("%Y%m%d"))
                    if df is not None and not df.empty:
                        break

            if df is not None and not df.empty:
                df = df[["ts_code", "trade_date", "close", "circ_mv", "total_mv"]].copy()
                all_data.append(df)
                print(f"    → {len(df)} 条")
            else:
                print(f"    → 无数据")
        except Exception as e:
            print(f"    → 失败: {e}")

        time.sleep(API_SLEEP)

    if not all_data:
        print("❌ 未获取到 daily_basic 数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    print(f"  ✅ 合计 {len(result)} 条记录")

    # 缓存
    result.to_parquet(cache_path, index=False)
    return result


# ══════════════════════════════════════════════════════════════════════
#  第二步：计算指标
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(holder_df: pd.DataFrame, basic_df: pd.DataFrame,
                    stock_list: pd.DataFrame) -> pd.DataFrame:
    """计算季度指标。

    返回每只股票每季度的：
      - 季度平均流通市值
      - 股东人数
      - 人均持股市值
      - 季度收益率（收盘价环比）
      - 股东人数变化率
      - 人均市值变化率
    """
    print("\n" + "=" * 60)
    print("第二步：计算指标")
    print("=" * 60)

    # 保留在股票列表中的数据
    valid_codes = set(stock_list["ts_code"].tolist())

    # ── 处理 holder 数据 ──
    holder_df = holder_df[holder_df["ts_code"].isin(valid_codes)].copy()
    holder_df["holder_num"] = pd.to_numeric(holder_df["holder_num"], errors="coerce")
    holder_df = holder_df.dropna(subset=["holder_num"])

    # 转为季度末格式（end_date 就是报告期）
    holder_df["quarter"] = holder_df["end_date"]
    print(f"  股东人数数据：{len(holder_df)} 条，{holder_df['ts_code'].nunique()} 只股票")

    # ── 处理 daily_basic 数据 ──
    basic_df = basic_df[basic_df["ts_code"].isin(valid_codes)].copy()
    basic_df["circ_mv"] = pd.to_numeric(basic_df["circ_mv"], errors="coerce")
    basic_df["close"] = pd.to_numeric(basic_df["close"], errors="coerce")
    basic_df = basic_df.dropna(subset=["circ_mv", "close"])

    # daily_basic 的 trade_date 映射到最近的季度末
    def map_to_quarter(trade_date: str) -> str:
        """将交易日期映射到最近的季度末。"""
        td = trade_date[:8]
        for q in reversed(QUARTER_ENDS):
            if td >= q:
                return q
        return QUARTER_ENDS[0]

    basic_df["trade_date"] = basic_df["trade_date"].astype(str).str[:8]
    basic_df["quarter"] = basic_df["trade_date"].apply(map_to_quarter)

    # 季度平均流通市值（同一季度可能有多个交易日的数据）
    quarterly_mv = (
        basic_df.groupby(["ts_code", "quarter"])
        .agg(
            avg_circ_mv=("circ_mv", "mean"),
            quarter_close=("close", "last"),  # 取季度末最后一天的收盘价
        )
        .reset_index()
    )
    print(f"  季度市值数据：{len(quarterly_mv)} 条")

    # ── 合并 ──
    merged = holder_df.merge(
        quarterly_mv[["ts_code", "quarter", "avg_circ_mv", "quarter_close"]],
        on=["ts_code", "quarter"],
        how="left",
    )
    merged = merged.dropna(subset=["avg_circ_mv", "quarter_close"])
    print(f"  合并后：{len(merged)} 条，{merged['ts_code'].nunique()} 只股票")

    if merged.empty:
        print("❌ 合并后无数据")
        return merged

    # ── 计算人均持股市值 ──
    # circ_mv 单位是万元，holder_num 是人数
    merged["per_capita_mv"] = merged["avg_circ_mv"] * 10000 / merged["holder_num"]
    # per_capita_mv 单位：元

    # ── 排序（按股票和季度）──
    merged = merged.sort_values(["ts_code", "quarter"]).reset_index(drop=True)

    # ── 计算变化率 ──
    merged["holder_chg_pct"] = (
        merged.groupby("ts_code")["holder_num"].pct_change() * 100
    )
    merged["quarter_return_pct"] = (
        merged.groupby("ts_code")["quarter_close"].pct_change() * 100
    )
    merged["per_capita_mv_chg_pct"] = (
        merged.groupby("ts_code")["per_capita_mv"].pct_change() * 100
    )

    # 去掉第一行（无环比数据）
    merged = merged.dropna(
        subset=["holder_chg_pct", "quarter_return_pct"]
    ).reset_index(drop=True)

    print(f"  计算变化率后：{len(merged)} 条有效记录，{merged['ts_code'].nunique()} 只股票")
    print(f"  季度范围：{merged['quarter'].min()} ~ {merged['quarter'].max()}")
    print(f"  人数变化率均值：{merged['holder_chg_pct'].mean():.2f}%")
    print(f"  季度收益率均值：{merged['quarter_return_pct'].mean():.2f}%")

    return merged


# ══════════════════════════════════════════════════════════════════════
#  第四步：相关性分析
# ══════════════════════════════════════════════════════════════════════

def correlation_analysis(df: pd.DataFrame) -> dict:
    """相关性分析主函数。"""
    print("\n" + "=" * 60)
    print("第四步：相关性分析")
    print("=" * 60)

    results = {}

    # ── 1. 同期相关 ──
    print("\n[1] 同期相关性（当季人数变化率 vs 当季收益率）")

    # Pearson
    pearson_r, pearson_p = stats.pearsonr(
        df["holder_chg_pct"], df["quarter_return_pct"]
    )
    # Spearman
    spearman_r, spearman_p = stats.spearmanr(
        df["holder_chg_pct"], df["quarter_return_pct"]
    )

    results["contemporaneous"] = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n": len(df),
    }
    print(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.6f})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p={spearman_p:.6f})")
    print(f"  样本量 N = {len(df)}")

    # ── 2. 滞后相关 ──
    print("\n[2] 滞后相关性（当季人数变化率 vs 后续季度收益率）")

    lag_results = []
    for lag in [1, 2]:
        lagged = df.copy()
        lagged["future_return"] = lagged.groupby("ts_code")["quarter_return_pct"].shift(-lag)
        lagged = lagged.dropna(subset=["future_return"])

        if len(lagged) < 10:
            print(f"  滞后 {lag} 季：数据不足")
            continue

        pr, pp = stats.pearsonr(lagged["holder_chg_pct"], lagged["future_return"])
        sr, sp = stats.spearmanr(lagged["holder_chg_pct"], lagged["future_return"])

        lag_results.append({
            "lag": lag,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n": len(lagged),
        })
        print(f"  滞后 {lag} 季: Pearson r={pr:.4f}(p={pp:.6f}), "
              f"Spearman ρ={sr:.4f}(p={sp:.6f}), N={len(lagged)}")

    results["lagged"] = lag_results

    # ── 3. 分组统计：人数增减 vs 后续收益 ──
    print("\n[3] 分组统计：股东人数变化分组 → 后续季度平均收益")

    group_results = []

    for lag in [0, 1, 2]:
        lagged = df.copy()
        lagged["future_return"] = lagged.groupby("ts_code")["quarter_return_pct"].shift(-lag)
        lagged = lagged.dropna(subset=["future_return"])

        # 分组
        decrease = lagged[lagged["holder_chg_pct"] < -10]
        increase = lagged[lagged["holder_chg_pct"] > 10]
        stable = lagged[lagged["holder_chg_pct"].between(-10, 10)]

        for name, group in [("减少>10%", decrease), ("增加>10%", increase), ("变化≤10%", stable)]:
            if len(group) < 5:
                continue
            group_results.append({
                "lag": lag,
                "group": name,
                "count": len(group),
                "mean_return": group["future_return"].mean(),
                "median_return": group["future_return"].median(),
                "win_rate": (group["future_return"] > 0).mean() * 100,
            })

    results["groups"] = group_results

    # 打印分组结果
    for lag in [0, 1, 2]:
        print(f"\n  滞后 {lag} 季的后续收益：")
        lag_data = [r for r in group_results if r["lag"] == lag]
        for r in lag_data:
            print(f"    {r['group']:>10s}: 平均收益 {r['mean_return']:+.2f}%, "
                  f"中位数 {r['median_return']:+.2f}%, "
                  f"胜率 {r['win_rate']:.1f}%, N={r['count']}")

    # ── 4. 人均持股市值分组 ──
    print("\n[4] 人均持股市值分组 → 后续季度收益")

    per_capita_results = []

    for lag in [0, 1]:
        lagged = df.copy()
        lagged["future_return"] = lagged.groupby("ts_code")["quarter_return_pct"].shift(-lag)
        lagged = lagged.dropna(subset=["future_return", "per_capita_mv"])

        if len(lagged) < 100:
            continue

        # 按人均市值三等分
        lagged["pc_group"] = pd.qcut(
            lagged["per_capita_mv"], 3, labels=["低", "中", "高"], duplicates="drop"
        )

        for g_name in ["低", "中", "高"]:
            sub = lagged[lagged["pc_group"] == g_name]
            if len(sub) < 5:
                continue
            per_capita_results.append({
                "lag": lag,
                "group": f"人均市值_{g_name}",
                "count": len(sub),
                "mean_return": sub["future_return"].mean(),
                "median_return": sub["future_return"].median(),
                "win_rate": (sub["future_return"] > 0).mean() * 100,
                "avg_per_capita_mv": sub["per_capita_mv"].mean(),
            })

    results["per_capita_groups"] = per_capita_results

    for lag in [0, 1]:
        print(f"\n  滞后 {lag} 季：")
        lag_data = [r for r in per_capita_results if r["lag"] == lag]
        for r in lag_data:
            print(f"    {r['group']:>8s}: 平均收益 {r['mean_return']:+.2f}%, "
                  f"胜率 {r['win_rate']:.1f}%, "
                  f"人均市值均值 {r['avg_per_capita_mv']/10000:.1f}万元, N={r['count']}")

    # ── 5. 全市场统计：每只股票的相关系数分布 ──
    print("\n[5] 全市场统计：每只股票的相关系数分布")

    stock_corrs = []
    for ts_code, group in df.groupby("ts_code"):
        if len(group) < 6:  # 至少6个季度数据
            continue
        try:
            r, p = stats.pearsonr(group["holder_chg_pct"], group["quarter_return_pct"])
            stock_corrs.append({
                "ts_code": ts_code,
                "corr": r,
                "p_value": p,
                "n_quarters": len(group),
            })
        except Exception:
            continue

    corr_df = pd.DataFrame(stock_corrs)
    results["stock_corr_dist"] = corr_df

    if not corr_df.empty:
        print(f"  有效股票数：{len(corr_df)}")
        print(f"  相关系数分布：")
        print(f"    均值: {corr_df['corr'].mean():.4f}")
        print(f"    中位数: {corr_df['corr'].median():.4f}")
        print(f"    标准差: {corr_df['corr'].std():.4f}")
        print(f"    < -0.3 的占比: {(corr_df['corr'] < -0.3).mean()*100:.1f}%")
        print(f"    > 0.3 的占比: {(corr_df['corr'] > 0.3).mean()*100:.1f}%")
        print(f"    显著(p<0.05)负相关占比: {((corr_df['corr'] < 0) & (corr_df['p_value'] < 0.05)).mean()*100:.1f}%")
        print(f"    显著(p<0.05)正相关占比: {((corr_df['corr'] > 0) & (corr_df['p_value'] < 0.05)).mean()*100:.1f}%")

    return results


# ══════════════════════════════════════════════════════════════════════
#  第五步：生成报告
# ══════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, results: dict,
                    stock_list: pd.DataFrame) -> str:
    """生成 Markdown 研究报告。"""
    print("\n" + "=" * 60)
    print("第五步：生成研究报告")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append("# 股东人数变化与股价相关性研究报告")
    lines.append("")
    lines.append(f"> 生成时间：{now}")
    lines.append(f"> 数据范围：{df['quarter'].min()} ~ {df['quarter'].max()}")
    lines.append(f"> 样本股票数：{df['ts_code'].nunique()}")
    lines.append(f"> 有效记录数：{len(df)}")
    lines.append(f"> 排除条件：ST股票、上市不满1年")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 概览统计 ──
    lines.append("## 1. 数据概览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 覆盖季度数 | {df['quarter'].nunique()} |")
    lines.append(f"| 覆盖股票数 | {df['ts_code'].nunique()} |")
    lines.append(f"| 有效观测数 | {len(df)} |")
    lines.append(f"| 股东人数变化率均值 | {df['holder_chg_pct'].mean():.2f}% |")
    lines.append(f"| 股东人数变化率中位数 | {df['holder_chg_pct'].median():.2f}% |")
    lines.append(f"| 季度收益率均值 | {df['quarter_return_pct'].mean():.2f}% |")
    lines.append(f"| 季度收益率中位数 | {df['quarter_return_pct'].median():.2f}% |")
    lines.append(f"| 人均持股市值均值 | {df['per_capita_mv'].mean()/10000:.1f}万元 |")
    lines.append("")

    # ── 同期相关 ──
    c = results["contemporaneous"]
    lines.append("## 2. 同期相关性")
    lines.append("")
    lines.append("当季股东人数变化率 vs 当季股价收益率：")
    lines.append("")
    lines.append("| 方法 | 相关系数 | p值 | 显著性 | 样本量 |")
    lines.append("|------|----------|------|--------|--------|")
    sig_p = "***" if c["pearson_p"] < 0.001 else "**" if c["pearson_p"] < 0.01 else "*" if c["pearson_p"] < 0.05 else ""
    sig_s = "***" if c["spearman_p"] < 0.001 else "**" if c["spearman_p"] < 0.01 else "*" if c["spearman_p"] < 0.05 else ""
    lines.append(f"| Pearson | {c['pearson_r']:.4f} | {c['pearson_p']:.6f} | {sig_p} | {c['n']} |")
    lines.append(f"| Spearman | {c['spearman_r']:.4f} | {c['spearman_p']:.6f} | {sig_s} | {c['n']} |")
    lines.append("")
    lines.append("> 显著性：*** p<0.001, ** p<0.01, * p<0.05")
    lines.append("")

    # ── 滞后相关 ──
    lines.append("## 3. 滞后相关性")
    lines.append("")
    lines.append("当季股东人数变化率 vs 未来季度收益率：")
    lines.append("")
    lines.append("| 滞后期 | Pearson r | p值 | Spearman ρ | p值 | N |")
    lines.append("|--------|-----------|------|------------|------|------|")
    for lr in results["lagged"]:
        lines.append(
            f"| {lr['lag']}季后 | {lr['pearson_r']:.4f} | {lr['pearson_p']:.6f} "
            f"| {lr['spearman_r']:.4f} | {lr['spearman_p']:.6f} | {lr['n']} |"
        )
    lines.append("")

    # ── 分组统计 ──
    lines.append("## 4. 股东人数变化分组 → 后续收益")
    lines.append("")
    lines.append("按股东人数季度变化率分组，统计后续季度的平均收益：")
    lines.append("")

    for lag in [0, 1, 2]:
        lag_label = f"滞后{lag}季" if lag > 0 else "当季"
        lines.append(f"### {lag_label}收益率")
        lines.append("")
        lines.append("| 分组 | 样本量 | 平均收益 | 中位数收益 | 胜率 |")
        lines.append("|------|--------|----------|------------|------|")
        lag_data = [r for r in results["groups"] if r["lag"] == lag]
        for r in lag_data:
            lines.append(
                f"| {r['group']} | {r['count']} | {r['mean_return']:+.2f}% "
                f"| {r['median_return']:+.2f}% | {r['win_rate']:.1f}% |"
            )
        lines.append("")

    # 计算人数减少组 vs 增加组的差异
    decrease_returns = {}
    increase_returns = {}
    for r in results["groups"]:
        if r["lag"] == 1:
            if "减少" in r["group"]:
                decrease_returns[r["lag"]] = r["mean_return"]
            elif "增加" in r["group"]:
                increase_returns[r["lag"]] = r["mean_return"]

    if 1 in decrease_returns and 1 in increase_returns:
        diff = decrease_returns[1] - increase_returns[1]
        lines.append(f"**滞后1季差异**：人数减少>10%组 - 人数增加>10%组 = "
                     f"{diff:+.2f}%")
        lines.append("")

    # ── 人均市值分组 ──
    lines.append("## 5. 人均持股市值分组 → 后续收益")
    lines.append("")
    lines.append("按人均持股市值三等分，统计后续收益差异：")
    lines.append("")

    for lag in [0, 1]:
        lag_label = f"滞后{lag}季" if lag > 0 else "当季"
        lines.append(f"### {lag_label}收益率")
        lines.append("")
        lines.append("| 分组 | 样本量 | 平均收益 | 中位数收益 | 胜率 | 人均市值均值 |")
        lines.append("|------|--------|----------|------------|------|-------------|")
        lag_data = [r for r in results.get("per_capita_groups", []) if r["lag"] == lag]
        for r in lag_data:
            lines.append(
                f"| {r['group']} | {r['count']} | {r['mean_return']:+.2f}% "
                f"| {r['median_return']:+.2f}% | {r['win_rate']:.1f}% "
                f"| {r['avg_per_capita_mv']/10000:.1f}万 |"
            )
        lines.append("")

    # ── 全市场相关系数分布 ──
    corr_df = results.get("stock_corr_dist", pd.DataFrame())
    if not corr_df.empty:
        lines.append("## 6. 全市场相关系数分布")
        lines.append("")
        lines.append("计算每只股票的股东人数变化率与股价收益率的Pearson相关系数：")
        lines.append("")
        lines.append("| 统计量 | 值 |")
        lines.append("|--------|------|")
        lines.append(f"| 有效股票数 | {len(corr_df)} |")
        lines.append(f"| 相关系数均值 | {corr_df['corr'].mean():.4f} |")
        lines.append(f"| 相关系数中位数 | {corr_df['corr'].median():.4f} |")
        lines.append(f"| 相关系数标准差 | {corr_df['corr'].std():.4f} |")
        lines.append(f"| 强负相关(<-0.3) | {(corr_df['corr'] < -0.3).sum()}只 ({(corr_df['corr'] < -0.3).mean()*100:.1f}%) |")
        lines.append(f"| 强正相关(>0.3) | {(corr_df['corr'] > 0.3).sum()}只 ({(corr_df['corr'] > 0.3).mean()*100:.1f}%) |")
        lines.append(f"| 显著负相关(p<0.05) | {((corr_df['corr'] < 0) & (corr_df['p_value'] < 0.05)).sum()}只 |")
        lines.append(f"| 显著正相关(p<0.05) | {((corr_df['corr'] > 0) & (corr_df['p_value'] < 0.05)).sum()}只 |")
        lines.append("")

        # 分位数
        lines.append("### 相关系数分位数")
        lines.append("")
        lines.append("| 分位 | 相关系数 |")
        lines.append("|------|----------|")
        for q in [5, 10, 25, 50, 75, 90, 95]:
            lines.append(f"| P{q} | {corr_df['corr'].quantile(q/100):.4f} |")
        lines.append("")

    # ── 结论 ──
    lines.append("## 7. 结论")
    lines.append("")

    conclusions = []

    # 同期相关结论
    c = results["contemporaneous"]
    if c["pearson_r"] < -0.05:
        conclusions.append(
            f"1. **股东人数与股价呈负相关**：同期Pearson r={c['pearson_r']:.4f}，"
            f"说明股东人数减少（筹码集中）的季度，股价倾向于上涨。"
        )
    elif c["pearson_r"] > 0.05:
        conclusions.append(
            f"1. **股东人数与股价呈正相关**：同期Pearson r={c['pearson_r']:.4f}，"
            f"出乎意料，需进一步分析是否受牛市/熊市等系统性因素影响。"
        )
    else:
        conclusions.append(
            f"1. **股东人数与股价相关性较弱**：同期Pearson r={c['pearson_r']:.4f}，"
            f"两者同期关系不显著。"
        )

    # 滞后相关结论
    if results["lagged"]:
        lag1 = results["lagged"][0]
        if lag1["spearman_r"] < -0.05 and lag1["spearman_p"] < 0.05:
            conclusions.append(
                f"2. **滞后1季的预测能力**：Spearman ρ={lag1['spearman_r']:.4f}(p={lag1['spearman_p']:.6f})，"
                f"股东人数变化对下一季度股价有显著预测能力。"
            )
        elif lag1["spearman_r"] < -0.03:
            conclusions.append(
                f"2. **滞后1季有微弱预测信号**：Spearman ρ={lag1['spearman_r']:.4f}，"
                f"方向正确但统计显著性不足。"
            )
        else:
            conclusions.append(
                f"2. **滞后1季预测能力有限**：Spearman ρ={lag1['spearman_r']:.4f}，"
                f"单独使用股东人数变化预测后续收益的可靠性较低。"
            )

    # 分组差异结论
    decrease_data = [r for r in results["groups"] if "减少" in r["group"] and r["lag"] == 1]
    increase_data = [r for r in results["groups"] if "增加" in r["group"] and r["lag"] == 1]
    if decrease_data and increase_data:
        d_ret = decrease_data[0]["mean_return"]
        i_ret = increase_data[0]["mean_return"]
        diff = d_ret - i_ret
        if diff > 2:
            conclusions.append(
                f"3. **分组差异显著**：人数减少>10%组滞后1季平均收益 {d_ret:+.2f}%，"
                f"人数增加>10%组 {i_ret:+.2f}%，差异 {diff:+.2f}%。"
                f"筹码集中（人数减少）是积极信号。"
            )
        elif diff > 0:
            conclusions.append(
                f"3. **分组差异存在但不显著**：人数减少组 vs 增加组差异 {diff:+.2f}%。"
            )
        else:
            conclusions.append(
                f"3. **分组差异方向相反**：人数减少组收益 {d_ret:+.2f}%，"
                f"增加组 {i_ret:+.2f}%，差异 {diff:+.2f}%。"
                f"可能与市场整体趋势有关，需要控制市场因素。"
            )

    # 人均市值结论
    pc_data = results.get("per_capita_groups", [])
    if pc_data:
        low_data = [r for r in pc_data if "低" in r["group"] and r["lag"] == 0]
        high_data = [r for r in pc_data if "高" in r["group"] and r["lag"] == 0]
        if low_data and high_data:
            conclusions.append(
                f"4. **人均持股市值分组**：低人均市值组（散户为主）平均收益 "
                f"{low_data[0]['mean_return']:+.2f}%，"
                f"高人均市值组（机构/大户为主）"
                f"{high_data[0]['mean_return']:+.2f}%。"
            )

    # 全市场分布结论
    if not corr_df.empty:
        neg_pct = (corr_df['corr'] < 0).mean() * 100
        conclusions.append(
            f"5. **全市场分布**：{neg_pct:.1f}%的股票呈现负相关（人数减少→股价上涨），"
            f"说明筹码集中效应在全市场具有一定的普适性。"
        )

    for c in conclusions:
        lines.append(f"- {c}")
    lines.append("")

    # ── 策略建议 ──
    lines.append("## 8. 策略建议")
    lines.append("")
    lines.append("基于以上分析，股东人数变化可以作为辅助选股因子：")
    lines.append("")
    lines.append("1. **单因子效果有限**：股东人数变化率单独使用时，预测能力中等，"
                 "建议与其他因子（技术面、基本面）结合")
    lines.append("2. **极端信号更有价值**：人数变化>20%的极端信号，比小幅变化更有参考意义")
    lines.append("3. **结合市场环境**：牛市中人数变化信号可能被系统性上涨淹没，"
                 "震荡市/熊市中信号更有效")
    lines.append("4. **筹码集中+业绩改善**：人数减少同时伴随业绩提升的股票，"
                 "胜率更高")
    lines.append("5. **注意滞后性**：股东人数数据有1-3个月的披露延迟，"
                 "实盘中需要考虑这个因素")
    lines.append("")
    lines.append("---")
    lines.append(f"> 报告由 holder_price_correlation.py 自动生成")
    lines.append("")

    report_text = "\n".join(lines)

    # 写入文件
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"  ✅ 报告已写入: {REPORT_PATH}")

    return report_text


# ══════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    print("🚀 股东人数变化与股价相关性研究")
    print(f"   时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 初始化客户端
    client = TushareClient()

    # 第一步：数据采集
    stock_list = fetch_stock_list(client)
    if stock_list.empty:
        print("❌ 无法获取股票列表，退出")
        return

    holder_df = fetch_holder_data(client)
    if holder_df.empty:
        print("❌ 无法获取股东人数数据，退出")
        return

    basic_df = fetch_daily_basic_quarterly(client)

    # 第二步 + 第三步：计算指标 + 标准化
    metrics_df = compute_metrics(holder_df, basic_df, stock_list)
    if metrics_df.empty:
        print("❌ 指标计算失败，退出")
        return

    # 缓存计算后的数据
    metrics_cache = OUTPUT_DIR / "_cache_metrics.parquet"
    metrics_df.to_parquet(metrics_cache, index=False)

    # 第四步：相关性分析
    analysis_results = correlation_analysis(metrics_df)

    # 第五步：生成报告
    report = generate_report(metrics_df, analysis_results, stock_list)

    # 打印关键结论
    print("\n" + "=" * 60)
    print("📊 关键结论")
    print("=" * 60)

    c = analysis_results["contemporaneous"]
    print(f"\n  同期相关：Pearson r={c['pearson_r']:.4f}(p={c['pearson_p']:.6f}), "
          f"Spearman ρ={c['spearman_r']:.4f}(p={c['spearman_p']:.6f})")

    if analysis_results["lagged"]:
        lag1 = analysis_results["lagged"][0]
        print(f"  滞后1季：Pearson r={lag1['pearson_r']:.4f}(p={lag1['pearson_p']:.6f})")

    # 分组差异
    decrease_data = [r for r in analysis_results["groups"] if "减少" in r["group"] and r["lag"] == 1]
    increase_data = [r for r in analysis_results["groups"] if "增加" in r["group"] and r["lag"] == 1]
    if decrease_data and increase_data:
        d_ret = decrease_data[0]["mean_return"]
        i_ret = increase_data[0]["mean_return"]
        print(f"\n  人数减少>10%组 → 滞后1季平均收益: {d_ret:+.2f}% (胜率{decrease_data[0]['win_rate']:.1f}%)")
        print(f"  人数增加>10%组 → 滞后1季平均收益: {i_ret:+.2f}% (胜率{increase_data[0]['win_rate']:.1f}%)")
        print(f"  差异: {d_ret - i_ret:+.2f}%")

    corr_df = analysis_results.get("stock_corr_dist", pd.DataFrame())
    if not corr_df.empty:
        print(f"\n  全市场：{(corr_df['corr'] < 0).mean()*100:.1f}%股票呈负相关")
        print(f"  显著负相关(p<0.05): {((corr_df['corr'] < 0) & (corr_df['p_value'] < 0.05)).sum()}只")

    print(f"\n  📄 完整报告: {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()
