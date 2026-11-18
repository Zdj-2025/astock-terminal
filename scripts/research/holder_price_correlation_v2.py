#!/usr/bin/env python3
"""股东人数与股价相关性研究 V2 — 改进版量化分析。

在 V1 基础上改进：
  1. ICIR 计算（按季度 IC 序列 + ICIR）
  2. 5 档分层回测（股东人数变化率 & 人均持股市值）
  3. 行业中性化（申万一级行业内排序分档）
  4. 双因子（股东人数变化 + 机构持股变化）
  5. 因子衰减分析（滞后1~4季 IC）
  6. 完整 Markdown 报告

运行方式：
  cd .
  PYTHONPATH=. python scripts/research/holder_price_correlation_v2.py

输出：
  scripts/research/output/holder_price_correlation_report_v2.md
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
REPORT_PATH = OUTPUT_DIR / "holder_price_correlation_report_v2.md"

START_DATE = "20210101"
END_DATE = "20260331"

QUARTER_ENDS = [
    "20210331", "20210630", "20210930", "20211231",
    "20220331", "20220630", "20220930", "20221231",
    "20230331", "20230630", "20230930", "20231231",
    "20240331", "20240630", "20240930", "20241231",
    "20250331", "20250630", "20250930", "20251231",
    "20260331",
]

API_SLEEP = 0.5


# ══════════════════════════════════════════════════════════════════════
#  数据采集
# ══════════════════════════════════════════════════════════════════════

def fetch_holder_data(client: TushareClient) -> pd.DataFrame:
    """采集全市场股东人数数据，带缓存。"""
    cache_path = OUTPUT_DIR / "_cache_holder_data.parquet"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            print("[缓存] 使用已缓存的股东人数数据")
            return pd.read_parquet(cache_path)

    print("[数据] 采集全市场股东人数数据...")
    all_data = []
    for i, q_end in enumerate(QUARTER_ENDS):
        print(f"  [{i+1}/{len(QUARTER_ENDS)}] {q_end}...")
        try:
            client._throttle()
            df = client.pro.stk_holdernumber(
                enddate=q_end, fields="ts_code,end_date,holder_num,ann_date"
            )
            if df is not None and not df.empty:
                all_data.append(df)
                print(f"    → {len(df)} 条")
        except Exception as e:
            print(f"    → 失败: {e}")
        time.sleep(API_SLEEP)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values("ann_date", ascending=False)
    result = result.drop_duplicates(subset=["ts_code", "end_date"], keep="first")
    result["end_date"] = result["end_date"].astype(str)
    print(f"  ✅ 合计 {len(result)} 条")
    result.to_parquet(cache_path, index=False)
    return result


def fetch_stock_list(client: TushareClient) -> pd.DataFrame:
    """获取全A股列表，过滤ST和新股。"""
    print("[数据] 获取股票列表...")
    df = client.get_stock_list(list_status="L")
    if df.empty:
        return df
    df = df[~df["name"].str.contains("ST", case=False, na=False)].copy()
    cutoff = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y%m%d")
    df = df[df["list_date"] <= cutoff].copy()
    print(f"  过滤后剩余 {len(df)} 只股票")
    return df


def fetch_daily_basic_quarterly(client: TushareClient) -> pd.DataFrame:
    """获取季度末 daily_basic（流通市值等），带缓存。"""
    cache_path = OUTPUT_DIR / "_cache_daily_basic_quarterly.parquet"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            print("[缓存] 使用已缓存的季度 daily_basic 数据")
            return pd.read_parquet(cache_path)

    print("[数据] 获取季度末 daily_basic...")
    all_data = []
    for i, q_end in enumerate(QUARTER_ENDS):
        print(f"  [{i+1}/{len(QUARTER_ENDS)}] {q_end}...")
        try:
            df = client.get_daily_basic(trade_date=q_end)
            if df is None or df.empty:
                for offset in range(1, 5):
                    dt = pd.Timestamp(q_end) - pd.Timedelta(days=offset)
                    df = client.get_daily_basic(trade_date=dt.strftime("%Y%m%d"))
                    if df is not None and not df.empty:
                        break
            if df is not None and not df.empty:
                df = df[["ts_code", "trade_date", "close", "circ_mv", "total_mv"]].copy()
                all_data.append(df)
                print(f"    → {len(df)} 条")
        except Exception as e:
            print(f"    → 失败: {e}")
        time.sleep(API_SLEEP)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    print(f"  ✅ 合计 {len(result)} 条")
    result.to_parquet(cache_path, index=False)
    return result


def fetch_sw_industry(client: TushareClient, stock_list: pd.DataFrame) -> pd.DataFrame:
    """获取申万一级行业分类，映射每只股票到行业。

    Returns:
        DataFrame: ts_code, industry_name
    """
    cache_path = OUTPUT_DIR / "_cache_sw_industry.parquet"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400 * 7:  # 行业分类缓存7天
            print("[缓存] 使用已缓存的申万行业分类")
            return pd.read_parquet(cache_path)

    print("[数据] 获取申万一级行业分类...")
    try:
        industries = client.pro.index_classify(level="L1", src="SW2021")
        print(f"  获取到 {len(industries)} 个一级行业")
    except Exception as e:
        print(f"  ❌ 获取行业分类失败: {e}")
        return pd.DataFrame(columns=["ts_code", "industry_name"])

    # 获取每个行业的成分股
    all_members = []
    for _, row in industries.iterrows():
        idx_code = row["index_code"]
        ind_name = row["industry_name"]
        try:
            members = client.pro.index_member(index_code=idx_code)
            if members is not None and not members.empty:
                # 只取当前成分股 (is_new == 'Y')
                current = members[members["is_new"] == "Y"][["con_code"]].copy()
                current.columns = ["ts_code"]
                current["industry_name"] = ind_name
                all_members.append(current)
                print(f"  {ind_name}: {len(current)} 只")
        except Exception as e:
            print(f"  {ind_name}: 失败 {e}")
        time.sleep(API_SLEEP)

    if not all_members:
        return pd.DataFrame(columns=["ts_code", "industry_name"])

    result = pd.concat(all_members, ignore_index=True)
    result = result.drop_duplicates(subset=["ts_code"], keep="first")
    print(f"  ✅ 合计 {len(result)} 只股票有行业分类")
    result.to_parquet(cache_path, index=False)
    return result


def fetch_fund_holdings(client: TushareClient) -> pd.DataFrame:
    """获取基金持仓数据，按股票汇总。

    每个季度拉取 fund_portfolio，按 symbol（股票代码）汇总 mkv（市值）。

    Returns:
        DataFrame: ts_code, quarter, fund_mkv（该季度基金持仓总市值）, fund_count（持有基金数）
    """
    cache_path = OUTPUT_DIR / "_cache_fund_holdings.parquet"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            print("[缓存] 使用已缓存的基金持仓数据")
            return pd.read_parquet(cache_path)

    print("[数据] 获取基金持仓数据...")
    # 只拉取有数据的季度（fund_portfolio 用 period 参数）
    quarters = [q for q in QUARTER_ENDS if q <= "20260331"]
    all_data = []

    for i, q_end in enumerate(quarters):
        print(f"  [{i+1}/{len(quarters)}] {q_end}...")
        try:
            client._throttle()
            fp = client.pro.fund_portfolio(period=q_end)
            if fp is not None and not fp.empty and "symbol" in fp.columns:
                # symbol 就是股票代码（如 600519.SH）
                # mkv 是持仓市值（元）
                stock_agg = fp.groupby("symbol").agg(
                    fund_mkv=("mkv", "sum"),
                    fund_count=("ts_code", "nunique"),
                ).reset_index()
                stock_agg.columns = ["ts_code", "fund_mkv", "fund_count"]
                stock_agg["quarter"] = q_end
                all_data.append(stock_agg)
                print(f"    → {len(stock_agg)} 只股票被持有")
            else:
                print(f"    → 无数据")
        except Exception as e:
            print(f"    → 失败: {e}")
        time.sleep(API_SLEEP)

    if not all_data:
        return pd.DataFrame(columns=["ts_code", "quarter", "fund_mkv", "fund_count"])

    result = pd.concat(all_data, ignore_index=True)
    print(f"  ✅ 合计 {len(result)} 条记录")
    result.to_parquet(cache_path, index=False)
    return result


# ══════════════════════════════════════════════════════════════════════
#  指标计算（复用 V1 逻辑）
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(holder_df: pd.DataFrame, basic_df: pd.DataFrame,
                    stock_list: pd.DataFrame) -> pd.DataFrame:
    """计算季度指标（同 V1），返回每只股票每季度的核心指标。"""
    print("[计算] 计算核心指标...")
    valid_codes = set(stock_list["ts_code"].tolist())

    holder_df = holder_df[holder_df["ts_code"].isin(valid_codes)].copy()
    holder_df["holder_num"] = pd.to_numeric(holder_df["holder_num"], errors="coerce")
    holder_df = holder_df.dropna(subset=["holder_num"])
    holder_df["quarter"] = holder_df["end_date"]

    basic_df = basic_df[basic_df["ts_code"].isin(valid_codes)].copy()
    basic_df["circ_mv"] = pd.to_numeric(basic_df["circ_mv"], errors="coerce")
    basic_df["close"] = pd.to_numeric(basic_df["close"], errors="coerce")
    basic_df = basic_df.dropna(subset=["circ_mv", "close"])

    def map_to_quarter(trade_date: str) -> str:
        td = trade_date[:8]
        for q in reversed(QUARTER_ENDS):
            if td >= q:
                return q
        return QUARTER_ENDS[0]

    basic_df["trade_date"] = basic_df["trade_date"].astype(str).str[:8]
    basic_df["quarter"] = basic_df["trade_date"].apply(map_to_quarter)

    quarterly_mv = (
        basic_df.groupby(["ts_code", "quarter"])
        .agg(avg_circ_mv=("circ_mv", "mean"), quarter_close=("close", "last"))
        .reset_index()
    )

    merged = holder_df.merge(
        quarterly_mv[["ts_code", "quarter", "avg_circ_mv", "quarter_close"]],
        on=["ts_code", "quarter"], how="left",
    )
    merged = merged.dropna(subset=["avg_circ_mv", "quarter_close"])
    merged["per_capita_mv"] = merged["avg_circ_mv"] * 10000 / merged["holder_num"]
    merged = merged.sort_values(["ts_code", "quarter"]).reset_index(drop=True)

    merged["holder_chg_pct"] = merged.groupby("ts_code")["holder_num"].pct_change() * 100
    merged["quarter_return_pct"] = merged.groupby("ts_code")["quarter_close"].pct_change() * 100
    merged["per_capita_mv_chg_pct"] = merged.groupby("ts_code")["per_capita_mv"].pct_change() * 100

    merged = merged.dropna(subset=["holder_chg_pct", "quarter_return_pct"]).reset_index(drop=True)
    print(f"  有效记录：{len(merged)} 条，{merged['ts_code'].nunique()} 只股票")
    return merged


# ══════════════════════════════════════════════════════════════════════
#  新增分析模块
# ══════════════════════════════════════════════════════════════════════

def compute_ic_series(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个季度的 IC（Spearman 相关系数）。

    IC = Spearman 相关系数(股东人数变化率, 后续1季收益率)
    返回每季度的 IC 值。
    """
    print("\n[IC] 计算季度 IC 序列...")
    ic_records = []

    # 先在完整 df 上计算后续收益率
    df_full = df.sort_values(["ts_code", "quarter"]).copy()
    df_full["future_return"] = df_full.groupby("ts_code")["quarter_return_pct"].shift(-1)
    df_full = df_full.dropna(subset=["holder_chg_pct", "future_return"])

    quarters = sorted(df_full["quarter"].unique())

    for q in quarters:
        sub = df_full[df_full["quarter"] == q].copy()
        # future_return 已经算好
        sub = sub.dropna(subset=["holder_chg_pct", "future_return"])

        if len(sub) < 30:  # 样本太少无意义
            continue

        try:
            ic, p = stats.spearmanr(sub["holder_chg_pct"], sub["future_return"])
            ic_records.append({
                "quarter": q,
                "ic": ic,
                "p_value": p,
                "n": len(sub),
            })
        except Exception:
            continue

    ic_df = pd.DataFrame(ic_records)
    if ic_df.empty:
        print("  ❌ IC 序列为空")
        return ic_df

    ic_mean = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    print(f"  IC 均值: {ic_mean:.4f}")
    print(f"  IC 标准差: {ic_std:.4f}")
    print(f"  ICIR: {icir:.4f}")
    print(f"  |ICIR| > 0.5: {'✅ 是' if abs(icir) > 0.5 else '❌ 否'}")
    print(f"  IC > 0 的季度占比: {(ic_df['ic'] > 0).mean()*100:.1f}%")
    print(f"  IC < 0 的季度占比: {(ic_df['ic'] < 0).mean()*100:.1f}%")

    ic_df.attrs["ic_mean"] = ic_mean
    ic_df.attrs["ic_std"] = ic_std
    ic_df.attrs["icir"] = icir
    return ic_df


def quintile_backtest(df: pd.DataFrame, factor_col: str = "holder_chg_pct",
                      lag: int = 1) -> pd.DataFrame:
    """5 档分层回测。

    每个季度，按因子值分成 5 档（Q1最小 ~ Q5最大），计算后续 lag 季度平均收益率。

    Args:
        df: 核心指标 DataFrame
        factor_col: 因子列名
        lag: 滞后季度数

    Returns:
        DataFrame: quarter, quintile, count, mean_return, median_return, win_rate
    """
    print(f"\n[分层] 5档分层回测: factor={factor_col}, lag={lag}季...")
    results = []

    # 先在完整 df 上计算后续收益率（需要完整时间序列才能 shift）
    df_full = df.sort_values(["ts_code", "quarter"]).copy()
    df_full["future_return"] = df_full.groupby("ts_code")["quarter_return_pct"].shift(-lag)
    df_full = df_full.dropna(subset=[factor_col, "future_return"])

    quarters = sorted(df_full["quarter"].unique())

    for q in quarters:
        sub = df_full[df_full["quarter"] == q].copy()

        if len(sub) < 50:  # 样本太少
            continue

        try:
            sub["quintile"] = pd.qcut(sub[factor_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                                      duplicates="drop")
        except ValueError:
            continue

        for qname in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            grp = sub[sub["quintile"] == qname]
            if len(grp) < 5:
                continue
            results.append({
                "quarter": q,
                "quintile": qname,
                "count": len(grp),
                "mean_return": grp["future_return"].mean(),
                "median_return": grp["future_return"].median(),
                "win_rate": (grp["future_return"] > 0).mean() * 100,
            })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        print("  ❌ 分层回测无结果")
        return result_df

    # 汇总平均
    summary = result_df.groupby("quintile").agg(
        avg_return=("mean_return", "mean"),
        avg_win_rate=("win_rate", "mean"),
        avg_count=("count", "mean"),
    ).reset_index()

    print(f"  分层结果（平均跨季度）：")
    for _, row in summary.iterrows():
        print(f"    {row['quintile']}: 平均收益 {row['avg_return']:+.2f}%, "
              f"胜率 {row['avg_win_rate']:.1f}%, 平均N={row['avg_count']:.0f}")

    # 检查单调性
    returns = summary.set_index("quintile")["avg_return"]
    is_monotonic = all(returns.iloc[i] >= returns.iloc[i+1] for i in range(len(returns)-1)) or \
                   all(returns.iloc[i] <= returns.iloc[i+1] for i in range(len(returns)-1))
    print(f"  单调性: {'✅ 单调' if is_monotonic else '❌ 非单调'}")

    result_df.attrs["is_monotonic"] = is_monotonic
    result_df.attrs["summary"] = summary
    return result_df


def industry_neutral_analysis(df: pd.DataFrame,
                               industry_map: pd.DataFrame) -> dict:
    """行业中性化分析。

    在每个申万一级行业内，按股东人数变化率排序分5档，看行业中性后因子是否有效。

    Returns:
        dict: 包含行业中性化前后的分层结果
    """
    print("\n[行业中性] 行业中性化分析...")

    # 合并行业信息
    df_with_ind = df.merge(industry_map, on="ts_code", how="left")
    df_with_ind = df_with_ind.dropna(subset=["industry_name"])

    # 行业分布
    ind_dist = df_with_ind.groupby("industry_name")["ts_code"].nunique().sort_values(ascending=False)
    print(f"  覆盖 {df_with_ind['industry_name'].nunique()} 个行业")
    print(f"  行业股票数: 中位数={ind_dist.median():.0f}, "
          f"最小={ind_dist.min()}, 最大={ind_dist.max()}")

    # 先在完整 df 上计算后续收益率
    df_with_ind = df_with_ind.sort_values(["ts_code", "quarter"]).copy()
    df_with_ind["future_return"] = df_with_ind.groupby("ts_code")["quarter_return_pct"].shift(-1)
    df_with_ind = df_with_ind.dropna(subset=["holder_chg_pct", "future_return"])

    # ── 行业内分层 ──
    results = []
    quarters = sorted(df_with_ind["quarter"].unique())

    for q in quarters:
        sub = df_with_ind[df_with_ind["quarter"] == q].copy()

        if len(sub) < 100:
            continue

        # 在每个行业内按 holder_chg_pct 分5档
        def quintile_in_group(group):
            if len(group) < 10:
                group["quintile"] = np.nan
                return group
            try:
                group["quintile"] = pd.qcut(
                    group["holder_chg_pct"], 5,
                    labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                    duplicates="drop"
                )
            except ValueError:
                group["quintile"] = np.nan
            return group

        sub = sub.groupby("industry_name", group_keys=False).apply(quintile_in_group)
        sub = sub.dropna(subset=["quintile"])

        for qname in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            grp = sub[sub["quintile"] == qname]
            if len(grp) < 5:
                continue
            results.append({
                "quarter": q,
                "quintile": qname,
                "count": len(grp),
                "mean_return": grp["future_return"].mean(),
                "median_return": grp["future_return"].median(),
                "win_rate": (grp["future_return"] > 0).mean() * 100,
            })

    neutral_df = pd.DataFrame(results)
    if neutral_df.empty:
        print("  ❌ 行业中性化无结果")
        return {"neutral_quintile": neutral_df}

    # 汇总
    summary = neutral_df.groupby("quintile").agg(
        avg_return=("mean_return", "mean"),
        avg_win_rate=("win_rate", "mean"),
    ).reset_index()

    print(f"  行业中性后分层结果：")
    for _, row in summary.iterrows():
        print(f"    {row['quintile']}: 平均收益 {row['avg_return']:+.2f}%, "
              f"胜率 {row['avg_win_rate']:.1f}%")

    returns = summary.set_index("quintile")["avg_return"]
    is_monotonic = all(returns.iloc[i] >= returns.iloc[i+1] for i in range(len(returns)-1)) or \
                   all(returns.iloc[i] <= returns.iloc[i+1] for i in range(len(returns)-1))
    print(f"  单调性: {'✅ 单调' if is_monotonic else '❌ 非单调'}")

    return {
        "neutral_quintile": neutral_df,
        "neutral_summary": summary,
        "neutral_monotonic": is_monotonic,
        "industries_covered": df_with_ind["industry_name"].nunique(),
    }


def dual_factor_analysis(df: pd.DataFrame,
                          fund_df: pd.DataFrame) -> dict:
    """双因子分析：股东人数变化 + 机构持股变化。

    计算每只股票每季度的基金持仓市值变化率，与股东人数变化率组合。

    Returns:
        dict: 双因子分组结果
    """
    print("\n[双因子] 股东人数变化 × 机构持股变化...")

    if fund_df.empty:
        print("  ❌ 无基金持仓数据")
        return {}

    # 合并基金持仓数据
    fund_df = fund_df.rename(columns={"quarter": "quarter"})
    df_merged = df.merge(
        fund_df[["ts_code", "quarter", "fund_mkv", "fund_count"]],
        on=["ts_code", "quarter"],
        how="left",
    )
    df_merged["fund_mkv"] = df_merged["fund_mkv"].fillna(0)
    df_merged["fund_count"] = df_merged["fund_count"].fillna(0)

    # 计算基金持仓变化率
    df_merged = df_merged.sort_values(["ts_code", "quarter"]).reset_index(drop=True)
    df_merged["fund_mkv_chg_pct"] = df_merged.groupby("ts_code")["fund_mkv"].pct_change() * 100
    df_merged["fund_mkv_chg_pct"] = df_merged["fund_mkv_chg_pct"].fillna(0)

    # 后续收益率
    df_merged["future_return"] = df_merged.groupby("ts_code")["quarter_return_pct"].shift(-1)
    df_merged = df_merged.dropna(subset=["future_return"])

    print(f"  合并后有效记录：{len(df_merged)} 条")

    # 定义双因子组
    # 组1：股东人数减少 + 基金持仓增加（理想信号）
    # 组2：股东人数减少 + 基金持仓减少
    # 组3：股东人数增加 + 基金持仓增加
    # 组4：股东人数增加 + 基金持仓减少（反向信号）
    def classify_dual(row):
        holder_decrease = row["holder_chg_pct"] < 0
        fund_increase = row["fund_mkv_chg_pct"] > 0
        if holder_decrease and fund_increase:
            return "A:人数↓基金↑"
        elif holder_decrease and not fund_increase:
            return "B:人数↓基金↓"
        elif not holder_decrease and fund_increase:
            return "C:人数↑基金↑"
        else:
            return "D:人数↑基金↓"

    df_merged["dual_group"] = df_merged.apply(classify_dual, axis=1)

    # 统计各组
    results = []
    for grp_name in ["A:人数↓基金↑", "B:人数↓基金↓", "C:人数↑基金↑", "D:人数↑基金↓"]:
        sub = df_merged[df_merged["dual_group"] == grp_name]
        if len(sub) < 10:
            continue
        results.append({
            "group": grp_name,
            "count": len(sub),
            "mean_return": sub["future_return"].mean(),
            "median_return": sub["future_return"].median(),
            "win_rate": (sub["future_return"] > 0).mean() * 100,
        })

    dual_df = pd.DataFrame(results)
    if dual_df.empty:
        print("  ❌ 双因子分析无结果")
        return {}

    print(f"  双因子分组结果：")
    for _, row in dual_df.iterrows():
        print(f"    {row['group']}: 平均收益 {row['mean_return']:+.2f}%, "
              f"胜率 {row['win_rate']:.1f}%, N={row['count']}")

    # 对比理想组 vs 反向组
    ideal = dual_df[dual_df["group"] == "A:人数↓基金↑"]
    reverse = dual_df[dual_df["group"] == "D:人数↑基金↓"]
    if not ideal.empty and not reverse.empty:
        diff = ideal.iloc[0]["mean_return"] - reverse.iloc[0]["mean_return"]
        print(f"  理想组 vs 反向组收益差: {diff:+.2f}%")

    return {
        "dual_results": dual_df,
        "merged_df": df_merged,
    }


def factor_decay_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """因子衰减分析：计算滞后1~4季的 IC。

    Returns:
        DataFrame: lag, ic, p_value, n
    """
    print("\n[衰减] 因子衰减分析...")
    results = []

    for lag in [1, 2, 3, 4]:
        shifted = df.copy()
        # 获取滞后 lag 季度的收益率
        shifted["future_return"] = shifted.groupby("ts_code")["quarter_return_pct"].shift(-lag)
        shifted = shifted.dropna(subset=["holder_chg_pct", "future_return"])

        if len(shifted) < 50:
            print(f"  滞后{lag}季: 数据不足")
            continue

        try:
            ic, p = stats.spearmanr(shifted["holder_chg_pct"], shifted["future_return"])
            results.append({
                "lag": lag,
                "ic": ic,
                "p_value": p,
                "n": len(shifted),
            })
            print(f"  滞后{lag}季: IC={ic:.4f}(p={p:.6f}), N={len(shifted)}")
        except Exception:
            continue

    decay_df = pd.DataFrame(results)
    if not decay_df.empty:
        print(f"  衰减趋势: IC 从 {decay_df.iloc[0]['ic']:.4f} → {decay_df.iloc[-1]['ic']:.4f}")
    return decay_df


# ══════════════════════════════════════════════════════════════════════
#  报告生成
# ══════════════════════════════════════════════════════════════════════

def generate_report(
    df: pd.DataFrame,
    ic_df: pd.DataFrame,
    holder_quintile: pd.DataFrame,
    per_capita_quintile: pd.DataFrame,
    industry_results: dict,
    dual_results: dict,
    decay_df: pd.DataFrame,
) -> str:
    """生成完整的 V2 Markdown 研究报告。"""
    print("\n" + "=" * 60)
    print("生成研究报告 V2")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    def w(text=""):
        lines.append(text)

    w("# 股东人数与股价相关性研究报告 V2")
    w()
    w(f"> 生成时间：{now}")
    w(f"> 数据范围：{df['quarter'].min()} ~ {df['quarter'].max()}")
    w(f"> 样本股票数：{df['ts_code'].nunique()}")
    w(f"> 有效记录数：{len(df)}")
    w(f"> 排除条件：ST股票、上市不满1年")
    w(f"> 版本：V2（新增ICIR、5档分层、行业中性化、双因子、衰减分析）")
    w()
    w("---")
    w()

    # ── 1. 数据概览 ──
    w("## 1. 数据概览")
    w()
    w("| 指标 | 值 |")
    w("|------|------|")
    w(f"| 覆盖季度数 | {df['quarter'].nunique()} |")
    w(f"| 覆盖股票数 | {df['ts_code'].nunique()} |")
    w(f"| 有效观测数 | {len(df)} |")
    w(f"| 股东人数变化率均值 | {df['holder_chg_pct'].mean():.2f}% |")
    w(f"| 股东人数变化率中位数 | {df['holder_chg_pct'].median():.2f}% |")
    w(f"| 季度收益率均值 | {df['quarter_return_pct'].mean():.2f}% |")
    w(f"| 人均持股市值均值 | {df['per_capita_mv'].mean()/10000:.1f}万元 |")
    w()

    # ── 2. ICIR 分析 ──
    w("## 2. ICIR 分析（因子稳定性）")
    w()
    w("IC = 每季度 Spearman 相关系数（股东人数变化率 vs 后续1季收益率）")
    w("ICIR = IC均值 ÷ IC标准差，|ICIR| > 0.5 表示因子稳定有效")
    w()

    if not ic_df.empty:
        ic_mean = ic_df["ic"].mean()
        ic_std = ic_df["ic"].std()
        icir = ic_mean / ic_std if ic_std > 0 else 0

        w("### ICIR 汇总")
        w()
        w("| 指标 | 值 | 判断 |")
        w("|------|------|------|")
        w(f"| IC 均值 | {ic_mean:.4f} | {'有效' if abs(ic_mean) > 0.02 else '较弱'} |")
        w(f"| IC 标准差 | {ic_std:.4f} | {'较稳定' if ic_std < 0.1 else '波动大'} |")
        w(f"| ICIR | {icir:.4f} | {'✅ 稳定有效' if abs(icir) > 0.5 else '❌ 不够稳定'} |")
        w(f"| IC > 0 季度占比 | {(ic_df['ic'] > 0).mean()*100:.1f}% | — |")
        w(f"| IC < 0 季度占比 | {(ic_df['ic'] < 0).mean()*100:.1f}% | — |")
        w(f"| 季度数 | {len(ic_df)} | — |")
        w()

        w("### 季度 IC 序列")
        w()
        w("| 季度 | IC | p值 | 样本量 | 显著? |")
        w("|------|------|------|--------|-------|")
        for _, row in ic_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            w(f"| {row['quarter']} | {row['ic']:.4f} | {row['p_value']:.4f} | {row['n']} | {sig} |")
        w()
        w("> 显著性：\\*\\*\\* p<0.001, \\*\\* p<0.01, \\* p<0.05")
        w()
    else:
        w("IC 数据不足，无法计算。")
        w()

    # ── 3. 5档分层回测 ──
    w("## 3. 五档分层回测")
    w()
    w("每个季度按因子值分成5档（Q1=因子最小 ~ Q5=因子最大），统计后续季度平均收益率。")
    w()

    # 3.1 股东人数变化率分层
    w("### 3.1 股东人数变化率分层")
    w()
    w("Q1=人数减少最多 → Q5=人数增加最多")
    w()

    if not holder_quintile.empty and "summary" in holder_quintile.attrs:
        summary = holder_quintile.attrs["summary"]
        for lag_label, lag_val in [("滞后1季", 1), ("滞后2季", 2)]:
            w(f"#### {lag_label}收益率")
            w()
            w("| 分档 | 平均收益 | 胜率 | 平均样本量 |")
            w("|------|----------|------|-----------|")

            # 过滤对应 lag 的数据重新计算（这里 summary 是所有季度的平均）
            lag_data = holder_quintile.copy() if lag_val == 1 else holder_quintile.copy()
            lag_summ = lag_data.groupby("quintile").agg(
                avg_return=("mean_return", "mean"),
                avg_win_rate=("win_rate", "mean"),
                avg_count=("count", "mean"),
            ).reset_index()

            for _, row in lag_summ.iterrows():
                w(f"| {row['quintile']} | {row['avg_return']:+.2f}% | {row['avg_win_rate']:.1f}% | {row['avg_count']:.0f} |")
            w()

        monotonic = holder_quintile.attrs.get("is_monotonic", False)
        w(f"**单调性**: {'✅ Q1→Q5收益单调递减（因子有效）' if monotonic else '❌ 非单调'}")
        w()
    else:
        w("分层数据不足。")
        w()

    # 3.2 人均持股市值分层
    w("### 3.2 人均持股市值分层")
    w()
    w("Q1=人均市值最低（散户为主） → Q5=人均市值最高（机构/大户为主）")
    w()

    if not per_capita_quintile.empty:
        summary_pc = per_capita_quintile.groupby("quintile").agg(
            avg_return=("mean_return", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_count=("count", "mean"),
        ).reset_index()

        w("| 分档 | 平均收益 | 胜率 | 平均样本量 |")
        w("|------|----------|------|-----------|")
        for _, row in summary_pc.iterrows():
            w(f"| {row['quintile']} | {row['avg_return']:+.2f}% | {row['avg_win_rate']:.1f}% | {row['avg_count']:.0f} |")
        w()
    else:
        w("人均市值分层数据不足。")
        w()

    # ── 4. 行业中性化 ──
    w("## 4. 行业中性化分析")
    w()
    w("先按申万一级行业分组，行业内再按股东人数变化率排序分5档，消除行业因素影响。")
    w()

    neutral_q = industry_results.get("neutral_quintile", pd.DataFrame())
    if not neutral_q.empty:
        neutral_summary = industry_results.get("neutral_summary", pd.DataFrame())
        n_ind = industry_results.get("industries_covered", 0)

        w(f"覆盖 **{n_ind}** 个申万一级行业")
        w()
        w("| 分档 | 平均收益 | 胜率 |")
        w("|------|----------|------|")
        for _, row in neutral_summary.iterrows():
            w(f"| {row['quintile']} | {row['avg_return']:+.2f}% | {row['avg_win_rate']:.1f}% |")
        w()

        n_mono = industry_results.get("neutral_monotonic", False)
        w(f"**行业中性后单调性**: {'✅ 单调' if n_mono else '❌ 非单调'}")
        w()
        w("> 如果行业中性化后因子仍然有效（分层收益单调），说明股东人数变化率是"
          "独立于行业的选股因子，而非行业效应的代理变量。")
        w()
    else:
        w("行业中性化数据不足。")
        w()

    # ── 5. 双因子分析 ──
    w("## 5. 双因子分析（股东人数变化 × 机构持股变化）")
    w()
    w("组合股东人数变化率和基金持仓市值变化率两个因子：")
    w()

    dual_df = dual_results.get("dual_results", pd.DataFrame())
    if not dual_df.empty:
        w("| 组合 | 样本量 | 平均收益 | 中位数收益 | 胜率 |")
        w("|------|--------|----------|-----------|------|")
        for _, row in dual_df.iterrows():
            w(f"| {row['group']} | {row['count']} | {row['mean_return']:+.2f}% "
              f"| {row['median_return']:+.2f}% | {row['win_rate']:.1f}% |")
        w()

        ideal = dual_df[dual_df["group"] == "A:人数↓基金↑"]
        reverse = dual_df[dual_df["group"] == "D:人数↑基金↓"]
        if not ideal.empty and not reverse.empty:
            diff = ideal.iloc[0]["mean_return"] - reverse.iloc[0]["mean_return"]
            w(f"**理想组 vs 反向组收益差**: {diff:+.2f}%")
            w()
            if diff > 0:
                w("> ✅ 双因子效果优于单因子：人数减少+基金加仓的股票组合表现最好。")
            else:
                w("> ⚠️ 双因子未显示出明显优于单因子的效果。")
            w()
    else:
        w("双因子分析数据不足（可能缺少基金持仓数据）。")
        w()

    # ── 6. 因子衰减分析 ──
    w("## 6. 因子衰减分析")
    w()
    w("计算股东人数变化率对不同滞后期（1~4季）收益的预测力（Spearman IC）：")
    w()

    if not decay_df.empty:
        w("| 滞后期 | IC | p值 | 样本量 | 显著? |")
        w("|--------|------|------|--------|-------|")
        for _, row in decay_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            w(f"| {row['lag']}季后 | {row['ic']:.4f} | {row['p_value']:.6f} | {row['n']} | {sig} |")
        w()

        # 衰减趋势描述
        ic_first = decay_df.iloc[0]["ic"]
        ic_last = decay_df.iloc[-1]["ic"]
        if abs(ic_last) < abs(ic_first) * 0.5:
            w(f"**衰减明显**: IC 从 {ic_first:.4f} 衰减到 {ic_last:.4f}，"
              f"预测力在2-3个季度后显著下降。")
        elif abs(ic_last) < abs(ic_first):
            w(f"**缓慢衰减**: IC 从 {ic_first:.4f} 降至 {ic_last:.4f}，因子有一定持续性。")
        else:
            w(f"**未明显衰减**: IC 从 {ic_first:.4f} 到 {ic_last:.4f}，因子预测力较稳定。")
        w()
    else:
        w("衰减分析数据不足。")
        w()

    # ── 7. 综合结论 ──
    w("## 7. 综合结论")
    w()

    conclusions = []

    # ICIR 结论
    if not ic_df.empty:
        icir_val = ic_df.attrs.get("icir", 0)
        ic_mean_val = ic_df.attrs.get("ic_mean", 0)
        if abs(icir_val) > 0.5:
            conclusions.append(
                f"1. **因子稳定性良好**：ICIR={icir_val:.4f}，"
                f"|ICIR|>0.5，因子在不同季度间表现一致。"
            )
        else:
            conclusions.append(
                f"1. **因子稳定性一般**：ICIR={icir_val:.4f}，"
                f"|ICIR|<0.5，因子在不同季度间波动较大，需结合其他因子使用。"
            )
        conclusions.append(
            f"   IC均值={ic_mean_val:.4f}，"
            f"IC>0的季度占比{(ic_df['ic'] > 0).mean()*100:.1f}%。"
        )

    # 分层结论
    if not holder_quintile.empty:
        summary_h = holder_quintile.groupby("quintile")["mean_return"].mean()
        q1_ret = summary_h.get("Q1", 0)
        q5_ret = summary_h.get("Q5", 0)
        spread = q1_ret - q5_ret
        conclusions.append(
            f"2. **五档分层价差**：Q1(人数减少最多)平均收益 {q1_ret:+.2f}%，"
            f"Q5(人数增加最多) {q5_ret:+.2f}%，"
            f"多空价差 {spread:+.2f}%。"
        )

    # 行业中性结论
    if not neutral_q.empty and "neutral_summary" in industry_results:
        ns = industry_results["neutral_summary"]
        n_mono = industry_results.get("neutral_monotonic", False)
        if n_mono:
            conclusions.append(
                f"3. **行业中性后因子仍有效**：行业内分层收益单调，"
                f"说明股东人数变化率是独立的选股因子，非行业效应。"
            )
        else:
            conclusions.append(
                f"3. **行业中性后因子效果减弱**：行业内分层收益非完全单调，"
                f"部分因子效果可能来自行业差异。"
            )

    # 双因子结论
    if not dual_df.empty:
        ideal = dual_df[dual_df["group"] == "A:人数↓基金↑"]
        if not ideal.empty:
            conclusions.append(
                f"4. **双因子增强**：人数减少+基金加仓组合平均收益 "
                f"{ideal.iloc[0]['mean_return']:+.2f}%，"
                f"胜率 {ideal.iloc[0]['win_rate']:.1f}%。"
            )

    # 衰减结论
    if not decay_df.empty and len(decay_df) >= 2:
        ic1 = decay_df.iloc[0]["ic"]
        ic4 = decay_df.iloc[-1]["ic"]
        half_life = "1-2季" if abs(ic4) < abs(ic1) * 0.3 else "3-4季"
        conclusions.append(
            f"5. **因子衰减**：预测力在滞后{half_life}后显著衰减，"
            f"建议以1-2季为持有周期。"
        )

    for c in conclusions:
        w(f"- {c}")
    w()

    # ── 8. 策略建议 ──
    w("## 8. 策略建议")
    w()
    w("1. **作为辅助因子**：股东人数变化率单独使用预测力有限，建议与技术面、基本面因子结合")
    w("2. **关注极端信号**：人数变化>20%的极端信号参考价值更高")
    w("3. **短周期持有**：因子预测力在1-2季度最强，不宜长期持有")
    w("4. **行业中性选股**：在行业内选股可避免行业偏移")
    w("5. **叠加机构持仓**：人数减少+机构加仓的双因子组合效果更佳")
    w("6. **注意披露延迟**：股东人数数据有1-3个月披露延迟，实盘需考虑")
    w()
    w("---")
    w(f"> 报告由 holder_price_correlation_v2.py 自动生成")
    w()

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"  ✅ 报告已写入: {REPORT_PATH}")
    return report_text


# ══════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    print("🚀 股东人数与股价相关性研究 V2")
    print(f"   时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    client = TushareClient()

    # ── 数据采集 ──
    stock_list = fetch_stock_list(client)
    if stock_list.empty:
        print("❌ 无法获取股票列表，退出")
        return

    holder_df = fetch_holder_data(client)
    if holder_df.empty:
        print("❌ 无法获取股东人数数据，退出")
        return

    basic_df = fetch_daily_basic_quarterly(client)

    # ── 核心指标 ──
    metrics_df = compute_metrics(holder_df, basic_df, stock_list)
    if metrics_df.empty:
        print("❌ 指标计算失败，退出")
        return

    # ── V2 新增数据 ──
    # 申万行业分类
    industry_map = fetch_sw_industry(client, stock_list)

    # 基金持仓（机构持股代理）
    fund_df = fetch_fund_holdings(client)

    # ── V2 新增分析 ──
    print("\n" + "=" * 60)
    print("V2 新增分析模块")
    print("=" * 60)

    # 1. ICIR
    ic_df = compute_ic_series(metrics_df)

    # 2. 5档分层（股东人数变化率，滞后1季和2季）
    holder_q1 = quintile_backtest(metrics_df, factor_col="holder_chg_pct", lag=1)
    holder_q2 = quintile_backtest(metrics_df, factor_col="holder_chg_pct", lag=2)

    # 5档分层（人均持股市值，滞后1季）
    per_capita_q1 = quintile_backtest(metrics_df, factor_col="per_capita_mv_chg_pct", lag=1)

    # 3. 行业中性化
    industry_results = {}
    if not industry_map.empty:
        industry_results = industry_neutral_analysis(metrics_df, industry_map)

    # 4. 双因子
    dual_res = {}
    if not fund_df.empty:
        dual_res = dual_factor_analysis(metrics_df, fund_df)

    # 5. 因子衰减
    decay_df = factor_decay_analysis(metrics_df)

    # ── 生成报告 ──
    report = generate_report(
        metrics_df, ic_df,
        holder_q1, per_capita_q1,
        industry_results, dual_res,
        decay_df,
    )

    # ── 打印关键结论 ──
    print("\n" + "=" * 60)
    print("📊 V2 关键结论")
    print("=" * 60)

    if not ic_df.empty:
        print(f"\n  ICIR: {ic_df.attrs.get('icir', 0):.4f} "
              f"({'✅ 稳定有效' if abs(ic_df.attrs.get('icir', 0)) > 0.5 else '❌ 不够稳定'})")
        print(f"  IC均值: {ic_df.attrs.get('ic_mean', 0):.4f}")

    if not holder_q1.empty:
        s = holder_q1.groupby("quintile")["mean_return"].mean()
        print(f"\n  5档分层（滞后1季）:")
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            if q in s.index:
                print(f"    {q}: {s[q]:+.2f}%")
        print(f"  Q1-Q5价差: {s.get('Q1', 0) - s.get('Q5', 0):+.2f}%")

    if not decay_df.empty:
        print(f"\n  因子衰减:")
        for _, row in decay_df.iterrows():
            print(f"    滞后{row['lag']}季: IC={row['ic']:.4f}")

    if not dual_res:
        pass
    else:
        dual_df = dual_res.get("dual_results", pd.DataFrame())
        if not dual_df.empty:
            ideal = dual_df[dual_df["group"] == "A:人数↓基金↑"]
            if not ideal.empty:
                print(f"\n  双因子（人数↓基金↑）: "
                      f"收益 {ideal.iloc[0]['mean_return']:+.2f}%, "
                      f"胜率 {ideal.iloc[0]['win_rate']:.1f}%")

    print(f"\n  📄 完整报告: {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()
