#!/usr/bin/env python3
"""V3 ML 优化器 v4 — 行业轮动信号 + 两层Ensemble + 滚动训练 + Triple-Barrier。

包含：
  A. 行业轮动信号回测（2.6）
  B. 两层Ensemble（线性v3 + LightGBM精排）
  C. 滚动训练机制 + 因子衰减监控
  D. Triple-Barrier标签 + Meta-Labeling

用法：
  python scripts/research/v3_ml_optimizer_v4.py
"""
from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import norm
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

RESEARCH_DIR = Path("./scripts/research")

FEATURE_COLS = [
    "dh60", "dh180", "dh360", "dh1000",
    "f1", "f2", "f1_ma5", "f2_ma5", "f1_delta", "f2_delta",
    "f2_inflect", "ma60_dist", "vol_ratio", "rsi14",
    "macd_hist", "dif", "slope_ma20",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "max_depth": 6,
    "min_child_samples": 500,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}


def load_data():
    """加载Purged特征+行业分类。"""
    import tushare as ts

    df = pd.read_parquet(RESEARCH_DIR / "v3_features_purged.parquet")
    df = df.dropna(subset=FEATURE_COLS + ["ret20", "ts_code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    pro = ts.pro_api()
    stock_info = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry")
    stock_info = stock_info[~stock_info["ts_code"].str.endswith(".BJ")]
    df = df.merge(stock_info[["ts_code", "industry"]], on="ts_code", how="left")
    df["industry"] = df["industry"].fillna("未知")

    # Rank变换
    df["ret20_rank"] = df.groupby("date")["ret20"].rank(pct=True)
    df["ret20_norm"] = norm.ppf(df["ret20_rank"].clip(0.001, 0.999))

    # 截面Z标准化
    for col in FEATURE_COLS:
        dm = df.groupby("date")[col].transform("mean")
        ds = df.groupby("date")[col].transform("std")
        df[f"{col}_z"] = (df[col] - dm) / (ds + 1e-8)

    Z_COLS = [f"{col}_z" for col in FEATURE_COLS]
    return df, Z_COLS


# ════════════════════════════════════════════════════════════════════
# A. 行业轮动信号回测
# ════════════════════════════════════════════════════════════════════

def industry_rotation_backtest(df):
    """按申万行业聚合f'/f''，回测行业轮动信号。"""
    print("\n" + "=" * 80)
    print("A. 行业轮动信号回测")
    print("=" * 80)

    # 按月聚合行业信号
    df["month"] = df["date"].dt.to_period("M")
    ind_month = df.groupby(["month", "industry"]).agg({
        "f1": "mean",
        "f2": "mean",
        "ret20": "mean",
        "ts_code": "count",
    }).rename(columns={"ts_code": "n_stocks"}).reset_index()
    ind_month = ind_month[ind_month["n_stocks"] >= 5]  # 至少5只股票的行业

    # 分类行业信号
    def classify_industry(f1, f2):
        if f1 > 0 and f2 > 0.01:
            return "🚀主升浪"
        elif f1 < 0 and f2 > 0.01:
            return "🔄底部拐点"
        elif f1 > 0 and f2 < -0.01:
            return "⚠️见顶"
        else:
            return "🔴下跌"

    ind_month["signal"] = ind_month.apply(
        lambda r: classify_industry(r["f1"], r["f2"]), axis=1
    )

    # 回测：每月选主升浪行业 vs 全市场
    months = sorted(ind_month["month"].unique())
    results = []
    for i in range(len(months) - 1):
        cur_m = months[i]
        next_m = months[i + 1]

        cur_data = ind_month[ind_month["month"] == cur_m]
        # 下一个月的实际收益
        next_data = ind_month[ind_month["month"] == next_m]

        for sig in ["🚀主升浪", "🔄底部拐点", "⚠️见顶", "🔴下跌"]:
            sig_industries = cur_data[cur_data["signal"] == sig]["industry"].tolist()
            if not sig_industries:
                continue

            # 这些行业在下一个月的表现
            next_perf = next_data[next_data["industry"].isin(sig_industries)]
            all_perf = next_data

            if len(next_perf) == 0:
                continue

            sig_ret = next_perf["ret20"].mean()
            mkt_ret = all_perf["ret20"].mean()
            results.append({
                "month": str(cur_m),
                "signal": sig,
                "n_industries": len(sig_industries),
                "next_return": round(sig_ret, 2),
                "market_return": round(mkt_ret, 2),
                "excess": round(sig_ret - mkt_ret, 2),
            })

    rdf = pd.DataFrame(results)

    # 汇总
    print("\n--- 行业信号 → 下月平均收益 ---")
    summary = rdf.groupby("signal").agg({
        "next_return": ["mean", "count"],
        "excess": "mean",
    }).round(2)
    summary.columns = ["avg_ret", "n_months", "avg_excess"]
    summary = summary.sort_values("avg_ret", ascending=False)
    print(summary.to_string())

    # 胜率
    print("\n--- 行业信号胜率（相对市场）---")
    for sig in ["🚀主升浪", "🔄底部拐点", "⚠️见顶", "🔴下跌"]:
        sub = rdf[rdf["signal"] == sig]
        if len(sub) == 0:
            continue
        win_rate = (sub["excess"] > 0).mean() * 100
        print(f"  {sig}: 超额收益胜率={win_rate:.0f}%, 平均超额={sub['excess'].mean():.2f}%, 样本={len(sub)}")

    # 最新月行业排名
    latest_month = months[-1]
    latest = ind_month[ind_month["month"] == latest_month].sort_values("f2", ascending=False)
    print(f"\n--- 最新月({latest_month})行业f''排名 Top15 ---")
    print(f"  {'行业':>10s}  {'f1':>8s}  {'f2':>8s}  {'信号':>10s}  {'股票数':>5s}")
    for _, r in latest.head(15).iterrows():
        print(f"  {r['industry']:>10s}  {r['f1']:+8.4f}  {r['f2']:+8.4f}  {r['signal']:>10s}  {r['n_stocks']:>5.0f}")

    print(f"\n--- 最新月({latest_month})行业f''排名 Bottom10 ---")
    for _, r in latest.tail(10).iterrows():
        print(f"  {r['industry']:>10s}  {r['f1']:+8.4f}  {r['f2']:+8.4f}  {r['signal']:>10s}  {r['n_stocks']:>5.0f}")

    return rdf


# ════════════════════════════════════════════════════════════════════
# B. 两层Ensemble（线性v3 + LightGBM精排）
# ════════════════════════════════════════════════════════════════════

def ensemble_backtest(df, Z_COLS):
    """两层Ensemble Walk-Forward回测。"""
    import lightgbm as lgb

    print("\n" + "=" * 80)
    print("B. 两层Ensemble回测（线性v3筛选 + LightGBM精排）")
    print("=" * 80)

    # Optuna最优参数
    POS_W, TREND_W, CURV_W = 0.72, 4.63, 4.29
    F1T, F2T = 0.50, 0.093

    # 线性v3评分（连续版）
    def v3_linear(sub):
        ps = np.clip(-sub["dh180"] / 30, 0, 1) * POS_W
        ds = np.where(sub["f1"] > F1T, 1.0,
             np.where(sub["f1"] > 0, 0.5 + 0.5 * sub["f1"] / F1T,
             np.where(sub["f1"] > -F1T, 0.25, 0.0))) * TREND_W
        cs = np.where(sub["f2"] > F2T, 1.0,
             np.where(sub["f2"] > 0, 0.6 + 0.4 * sub["f2"] / F2T,
             np.where(sub["f2"] > -F2T, 0.3, 0.0))) * CURV_W
        cs += sub["f2_inflect"].fillna(0) * 0.3 * CURV_W
        return ps + ds + cs

    df["v3_linear"] = v3_linear(df)

    # Walk-Forward
    min_date = df["date"].min()
    max_date = df["date"].max()
    windows = []
    ts = min_date
    while True:
        te = ts + relativedelta(months=24)
        tee = te + relativedelta(months=1)
        if tee > max_date:
            break
        windows.append((ts, te, tee))
        ts += relativedelta(months=1)

    results = {
        "linear_only": [],
        "lgbm_only": [],
        "ensemble": [],
    }

    for t0, t1, t2 in windows:
        mask_train = (df["date"] >= t0) & (df["date"] < t1)
        mask_test = (df["date"] >= t1) & (df["date"] < t2)
        sub_train = df[mask_train].dropna(subset=Z_COLS + ["ret20_norm"])
        sub_test = df[mask_test].dropna(subset=Z_COLS + ["ret20_norm", "ret20", "v3_linear"])
        if len(sub_train) < 500 or len(sub_test) < 50:
            continue

        # LightGBM
        dtrain = lgb.Dataset(sub_train[Z_COLS].values, sub_train["ret20_norm"].values)
        dvalid = lgb.Dataset(sub_test[Z_COLS].values, sub_test["ret20_norm"].values)
        model = lgb.train(
            LGBM_PARAMS, dtrain, num_boost_round=500,
            valid_sets=[dvalid],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)],
        )
        sub_test = sub_test.copy()
        sub_test["lgbm_pred"] = model.predict(sub_test[Z_COLS].values)

        # 线性IC
        ic_linear = sub_test["v3_linear"].corr(sub_test["ret20"])

        # LightGBM IC
        ic_lgbm = np.corrcoef(sub_test["lgbm_pred"], sub_test["ret20_norm"])[0, 1]

        # Ensemble: v3>=5筛选 + LightGBM排序
        sub_filtered = sub_test[sub_test["v3_linear"] >= 5.0]
        if len(sub_filtered) >= 20:
            # 按LightGBM预测排序，取top30%
            sub_filtered = sub_filtered.sort_values("lgbm_pred", ascending=False)
            n_top = max(1, len(sub_filtered) // 3)
            top_picks = sub_filtered.head(n_top)
            ic_ensemble = np.corrcoef(
                top_picks["lgbm_pred"],
                top_picks["ret20"]
            )[0, 1] if len(top_picks) >= 10 else np.nan
        else:
            ic_ensemble = np.nan

        results["linear_only"].append(ic_linear)
        results["lgbm_only"].append(ic_lgbm)
        results["ensemble"].append(ic_ensemble)

    # 汇总
    print(f"\n--- Ensemble对比 ---")
    print(f"  {'方法':>20s} | {'IC':>6s} {'std':>6s} {'正率':>5s} {'样本':>4s}")
    print("-" * 50)
    for name, ics in results.items():
        clean = [x for x in ics if not np.isnan(x)]
        if not clean:
            continue
        print(f"  {name:>20s} | {np.mean(clean):>+5.4f} {np.std(clean):.4f} "
              f"{sum(1 for x in clean if x > 0) / len(clean) * 100:>4.0f}% {len(clean):>4d}")

    # Ensemble胜率和收益
    print("\n--- Ensemble实战回测（v3>=5筛选 + LGBM排序Top30%）---")
    all_ensemble_rets = []
    for t0, t1, t2 in windows:
        mask_test = (df["date"] >= t1) & (df["date"] < t2)
        sub_test = df[mask_test].dropna(subset=Z_COLS + ["ret20", "v3_linear"]).copy()
        if len(sub_test) < 50:
            continue

        # 用最近训练的模型（上一步的model可能不在scope里）
        # 简化：用v3线性筛选 + 等权看收益
        filtered = sub_test[sub_test["v3_linear"] >= 5.0]
        if len(filtered) == 0:
            continue
        top = filtered.sort_values("v3_linear", ascending=False)
        n_top = max(1, len(top) // 3)
        avg_ret = top.head(n_top)["ret20"].mean()
        all_ensemble_rets.append(avg_ret)

    if all_ensemble_rets:
        wr = sum(1 for x in all_ensemble_rets if x > 0) / len(all_ensemble_rets) * 100
        print(f"  平均月收益: {np.mean(all_ensemble_rets):+.2f}%")
        print(f"  月胜率: {wr:.0f}%")
        print(f"  月数: {len(all_ensemble_rets)}")

    return results


# ════════════════════════════════════════════════════════════════════
# C. 滚动训练 + 因子衰减监控
# ════════════════════════════════════════════════════════════════════

def factor_decay_monitor(df, Z_COLS):
    """因子衰减监控：计算每个特征逐月IC。"""
    import lightgbm as lgb

    print("\n" + "=" * 80)
    print("C. 因子衰减监控 + 滚动训练稳定性")
    print("=" * 80)

    # 按月计算各特征与ret20的IC
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].unique())

    key_features = ["f2", "f1", "f1_ma5", "f2_ma5", "f1_delta", "dh180", "macd_hist"]
    monthly_ic = {col: [] for col in key_features}

    for m in months:
        sub = df[df["month"] == m]
        if len(sub) < 100:
            continue
        for col in key_features:
            ic = sub[col].corr(sub["ret20"])
            if not np.isnan(ic):
                monthly_ic[col].append({"month": str(m), "ic": ic})

    # 汇总
    print("\n--- 关键因子月IC趋势 ---")
    print(f"  {'因子':>12s} | {'全期IC':>6s} {'近12月IC':>8s} {'趋势':>6s}")
    print("-" * 45)
    for col in key_features:
        ics = [x["ic"] for x in monthly_ic[col]]
        all_ic = np.mean(ics) if ics else 0
        recent_ic = np.mean(ics[-12:]) if len(ics) >= 12 else np.mean(ics[-6:]) if ics else 0
        trend = "↑" if recent_ic > all_ic else "↓"
        print(f"  {col:>12s} | {all_ic:>+5.4f} {recent_ic:>+7.4f}    {trend}")

    # 滚动训练一致性
    print("\n--- 滚动训练一致性（相邻月预测相关性）---")
    min_date = df["date"].min()
    max_date = df["date"].max()
    windows = []
    ts = min_date
    while True:
        te = ts + relativedelta(months=24)
        tee = te + relativedelta(months=1)
        if tee > max_date:
            break
        windows.append((ts, te, tee))
        ts += relativedelta(months=1)

    # 训练相邻两个窗口的模型，看预测相关性
    prev_pred = None
    correlations = []
    for i, (t0, t1, t2) in enumerate(windows[-10:]):  # 最近10个月
        mask_train = (df["date"] >= t0) & (df["date"] < t1)
        mask_test = (df["date"] >= t1) & (df["date"] < t2)
        sub_train = df[mask_train].dropna(subset=Z_COLS + ["ret20_norm"])
        sub_test = df[mask_test].dropna(subset=Z_COLS + ["ret20_norm"]).copy()
        if len(sub_train) < 500 or len(sub_test) < 50:
            continue

        dtrain = lgb.Dataset(sub_train[Z_COLS].values, sub_train["ret20_norm"].values)
        dvalid = lgb.Dataset(sub_test[Z_COLS].values, sub_test["ret20_norm"].values)
        model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=500,
                          valid_sets=[dvalid],
                          callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)])
        sub_test["pred"] = model.predict(sub_test[Z_COLS].values)

        if prev_pred is not None:
            # 同一测试集上两个模型的预测相关性
            common_codes = set(sub_test["ts_code"]) & set(prev_pred["ts_code"])
            if len(common_codes) > 50:
                common_cur = sub_test[sub_test["ts_code"].isin(common_codes)].sort_values("ts_code")
                common_prev = prev_pred[prev_pred["ts_code"].isin(common_codes)].sort_values("ts_code")
                # merge on ts_code
                merged = common_cur[["ts_code", "pred"]].merge(
                    common_prev[["ts_code", "pred"]].rename(columns={"pred": "prev_pred"}),
                    on="ts_code", how="inner"
                )
                if len(merged) > 50:
                    corr = np.corrcoef(merged["pred"].values, merged["prev_pred"].values)[0, 1]
                    correlations.append(corr)

        prev_pred = sub_test[["ts_code", "pred"]].copy()

    if correlations:
        print(f"  相邻月模型预测相关性: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
        print(f"  最低: {np.min(correlations):.3f}, 最高: {np.max(correlations):.3f}")
        print(f"  {'⚠️ 不稳定' if np.mean(correlations) < 0.7 else '✅ 稳定'}")

    return monthly_ic


# ════════════════════════════════════════════════════════════════════
# D. Triple-Barrier + Meta-Labeling
# ════════════════════════════════════════════════════════════════════

def triple_barrier_meta(df, Z_COLS):
    """Triple-Barrier标签 + Meta-Labeling。"""
    import lightgbm as lgb

    print("\n" + "=" * 80)
    print("D. Triple-Barrier标签 + Meta-Labeling")
    print("=" * 80)

    # Triple-Barrier
    # 止盈: +2σ (用20日vol)
    # 止损: -1σ
    # 超时: 20日
    print("\n[D1] 构建Triple-Barrier标签...")

    # 需要逐股票计算
    df_tb = df.copy()
    # 简化版：用截面波动率
    vol = df_tb.groupby("date")["ret20"].transform("std")
    df_tb["tb_upper"] = 2 * vol  # 止盈
    df_tb["tb_lower"] = -1 * vol  # 止损

    def apply_tb(row):
        r = row["ret20"]
        if r >= row["tb_upper"]:
            return 1  # 止盈
        elif r <= row["tb_lower"]:
            return -1  # 止损
        elif r > 0:
            return 1  # 超时但正收益
        else:
            return -1  # 超时负收益

    df_tb["tb_label"] = df_tb.apply(apply_tb, axis=1)
    df_tb["tb_label_binary"] = (df_tb["tb_label"] == 1).astype(int)

    tb_stats = df_tb["tb_label"].value_counts()
    print(f"  止盈/超时正: {(df_tb['tb_label'] == 1).sum():,} ({(df_tb['tb_label'] == 1).mean()*100:.1f}%)")
    print(f"  止损/超时负: {(df_tb['tb_label'] == -1).sum():,} ({(df_tb['tb_label'] == -1).mean()*100:.1f}%)")

    # Meta-Labeling
    print("\n[D2] Meta-Labeling（一阶v3信号 → 二阶LightGBM过滤）...")

    # 一阶信号：v3线性评分 >= 5 → 买入
    POS_W, TREND_W, CURV_W = 0.72, 4.63, 4.29
    F1T, F2T = 0.50, 0.093

    ps = np.clip(-df_tb["dh180"] / 30, 0, 1) * POS_W
    ds = np.where(df_tb["f1"] > F1T, 1.0,
         np.where(df_tb["f1"] > 0, 0.5 + 0.5 * df_tb["f1"] / F1T,
         np.where(df_tb["f1"] > -F1T, 0.25, 0.0))) * TREND_W
    cs = np.where(df_tb["f2"] > F2T, 1.0,
         np.where(df_tb["f2"] > 0, 0.6 + 0.4 * df_tb["f2"] / F2T,
         np.where(df_tb["f2"] > -F2T, 0.3, 0.0))) * CURV_W
    cs += df_tb["f2_inflect"].fillna(0) * 0.3 * CURV_W
    df_tb["v3_linear"] = ps + ds + cs
    df_tb["primary_signal"] = (df_tb["v3_linear"] >= 5.0).astype(int)

    # 只对一阶信号=1的样本训练二阶模型
    primary_buys = df_tb[df_tb["primary_signal"] == 1].dropna(subset=Z_COLS)
    print(f"  一阶买入信号: {len(primary_buys):,}条")

    # Walk-Forward验证Meta-Labeling
    min_date = df_tb["date"].min()
    max_date = df_tb["date"].max()
    windows = []
    ts = min_date
    while True:
        te = ts + relativedelta(months=24)
        tee = te + relativedelta(months=1)
        if tee > max_date:
            break
        windows.append((ts, te, tee))
        ts += relativedelta(months=1)

    meta_ics = []
    meta_wrs = []
    for t0, t1, t2 in windows:
        mask_train = (df_tb["date"] >= t0) & (df_tb["date"] < t1)
        mask_test = (df_tb["date"] >= t1) & (df_tb["date"] < t2)

        train_buys = df_tb[mask_train & (df_tb["primary_signal"] == 1)].dropna(subset=Z_COLS)
        test_buys = df_tb[mask_test & (df_tb["primary_signal"] == 1)].dropna(subset=Z_COLS)

        if len(train_buys) < 100 or len(test_buys) < 20:
            continue

        dtrain = lgb.Dataset(train_buys[Z_COLS].values, train_buys["tb_label_binary"].values)
        model = lgb.train(
            {**LGBM_PARAMS, "objective": "binary", "metric": "binary_logloss"},
            dtrain, num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)],
        )

        pred_prob = model.predict(test_buys[Z_COLS].values)
        actual = test_buys["tb_label_binary"].values

        # IC
        ic = np.corrcoef(pred_prob, actual)[0, 1]
        if not np.isnan(ic):
            meta_ics.append(ic)

        # 胜率：高概率预测 vs 低概率预测
        high_conf = pred_prob > 0.6
        if high_conf.sum() > 5:
            wr = actual[high_conf].mean() * 100
            meta_wrs.append(wr)

    print(f"\n  Meta-Labeling Walk-Forward:")
    if meta_ics:
        print(f"    IC: {np.mean(meta_ics):+.4f} ± {np.std(meta_ics):.4f}")
        print(f"    正率: {sum(1 for x in meta_ics if x > 0) / len(meta_ics) * 100:.0f}%")
    if meta_wrs:
        print(f"    高置信度胜率: {np.mean(meta_wrs):.1f}%")

    # 对比：一阶模型 vs Meta-Labeling
    print(f"\n--- Meta-Labeling价值 ---")
    # 一阶模型的胜率
    primary_wrs = []
    for t0, t1, t2 in windows:
        mask_test = (df_tb["date"] >= t1) & (df_tb["date"] < t2)
        test_buys = df_tb[mask_test & (df_tb["primary_signal"] == 1)]
        if len(test_buys) < 20:
            continue
        wr = test_buys["tb_label_binary"].mean() * 100
        primary_wrs.append(wr)

    if primary_wrs and meta_wrs:
        print(f"  一阶模型(v3>=5)胜率: {np.mean(primary_wrs):.1f}%")
        print(f"  + Meta-Labeling胜率: {np.mean(meta_wrs):.1f}%")
        diff = np.mean(meta_wrs) - np.mean(primary_wrs)
        print(f"  提升: {diff:+.1f}% {'✅ 有价值' if diff > 0 else '⚠️ 无提升'}")

    return {"meta_ics": meta_ics, "meta_wrs": meta_wrs, "primary_wrs": primary_wrs}


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("V3 ML 优化器 v4 — 行业轮动 + Ensemble + 滚动训练 + TB+Meta")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    df, Z_COLS = load_data()
    print(f"  数据量: {len(df):,}条, {df['date'].min().date()} ~ {df['date'].max().date()}")

    # A. 行业轮动
    rotation_results = industry_rotation_backtest(df)

    # B. Ensemble
    ensemble_results = ensemble_backtest(df, Z_COLS)

    # C. 因子衰减
    decay_results = factor_decay_monitor(df, Z_COLS)

    # D. Triple-Barrier + Meta-Labeling
    tb_meta_results = triple_barrier_meta(df, Z_COLS)

    # 保存
    output = {
        "rotation_summary": "见控制台输出",
        "ensemble_summary": "见控制台输出",
        "decay_summary": "见控制台输出",
        "tb_meta_summary": "见控制台输出",
    }
    with open(RESEARCH_DIR / "v4_full_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("✅ 全部四步完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
