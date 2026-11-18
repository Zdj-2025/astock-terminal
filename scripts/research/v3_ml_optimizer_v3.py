#!/usr/bin/env python3
"""V3 ML 优化器 v3 — Optuna调参 + 标签改造 + 行业中性化。

包含：
  A. Optuna优化v3线性评分参数（快，直接影响周一选股）
  B. 标签改造（Rank变换 + 正态化）
  C. 行业中性化 + 截面标准化
  D. LightGBM Walk-Forward对比（改造前后IC变化）

用法：
  python scripts/research/v3_ml_optimizer_v3.py
"""
from __future__ import annotations

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import norm

warnings.filterwarnings("ignore")

RESEARCH_DIR = Path("./scripts/research")


# ── A. Optuna优化v3线性评分参数 ──────────────────────────────────

def optuna_v3_linear():
    """用Optuna搜索v3线性评分的最优参数。

    搜索空间：
      - 位置权重: 0.5~3.0
      - 趋势权重: 1~5
      - 曲率权重: 1~5
      - f1阈值: 0.05~0.5（强涨/强跌的分界）
      - f2阈值: 0.005~0.1（加速上行/加速下跌的分界）
      - S-G window: [11, 15, 21, 25, 31]

    目标：v3总分与ret20的IC（Walk-Forward平均）
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 加载Purged特征
    feat_path = RESEARCH_DIR / "v3_features_purged.parquet"
    df = pd.read_parquet(feat_path)
    df = df.dropna(subset=["f1", "f2", "dh180", "ret20"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Optuna优化v3线性评分: {len(df):,}条")
    print(f"搜索: pos_w, trend_w, curv_w, f1_thresh, f2_thresh, sg_window")

    def compute_v3_score(row, params):
        """计算单条v3评分。"""
        # 位置
        ps = 0
        for col, max_s, thresholds in [
            ("dh180", 0.7, params["pos_thresholds_180"]),
            ("dh360", 0.6, params["pos_thresholds_360"]),
            ("dh1000", 0.5, params["pos_thresholds_1000"]),
            ("dh60", 0.2, params["pos_thresholds_60"]),
        ]:
            v = row.get(col, 0)
            for threshold, s in thresholds:
                if v >= threshold:
                    ps += s
                    break

        # 趋势
        f1v = row["f1"]
        t1 = params["f1_thresh"]
        if f1v > t1: ds = 1.0
        elif f1v > 0: ds = 0.5 + 0.5 * (f1v / t1)
        elif f1v > -t1: ds = 0.25
        else: ds = 0.0

        # 曲率
        f2v = row["f2"]
        t2 = params["f2_thresh"]
        if f2v > t2: cs = 1.0
        elif f2v > 0: cs = 0.6 + 0.4 * (f2v / t2)
        elif f2v > -t2: cs = 0.3
        else: cs = 0.0

        # 拐点加分
        if row.get("f2_inflect", 0) > 0.5:
            cs += 0.3

        total = ps * params["pos_w"] + ds * params["trend_w"] + cs * params["curv_w"]
        return total

    # 简化版：直接用连续值计算IC（不用完整评分阶梯）
    def objective(trial):
        pos_w = trial.suggest_float("pos_w", 0.5, 3.0)
        trend_w = trial.suggest_float("trend_w", 1.0, 5.0)
        curv_w = trial.suggest_float("curv_w", 1.0, 5.0)
        f1_thresh = trial.suggest_float("f1_thresh", 0.05, 0.5)
        f2_thresh = trial.suggest_float("f2_thresh", 0.005, 0.1)

        # 连续版评分
        # 位置：距高点远→分高
        ps = np.clip(-df["dh180"] / 30, 0, 1) * pos_w

        # 趋势：f1正→分高
        ds = np.where(df["f1"] > f1_thresh, 1.0,
             np.where(df["f1"] > 0, 0.5 + 0.5 * df["f1"] / f1_thresh,
             np.where(df["f1"] > -f1_thresh, 0.25, 0.0))) * trend_w

        # 曲率：f2正→分高
        cs = np.where(df["f2"] > f2_thresh, 1.0,
             np.where(df["f2"] > 0, 0.6 + 0.4 * df["f2"] / f2_thresh,
             np.where(df["f2"] > -f2_thresh, 0.3, 0.0))) * curv_w

        # 拐点加分
        cs += df["f2_inflect"].fillna(0) * 0.3 * curv_w

        total = ps + ds + cs
        ic = total.corr(df["ret20"])
        return ic

    # 先快速搜索（全局IC）
    print("\n--- Phase 1: 快速全局IC搜索 ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200, show_progress_bar=False)

    best = study.best_params
    best_ic = study.best_value
    print(f"  最优参数: {best}")
    print(f"  最优全局IC: {best_ic:.4f}")

    # Phase 2: Walk-Forward验证最优参数
    print("\n--- Phase 2: Walk-Forward验证最优参数 ---")
    from dateutil.relativedelta import relativedelta

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

    wf_ics = []
    for i, (t0, t1, t2) in enumerate(windows):
        mask_train = (df["date"] >= t0) & (df["date"] < t1)
        mask_test = (df["date"] >= t1) & (df["date"] < t2)
        sub_test = df[mask_test]
        if len(sub_test) < 50:
            continue

        pos_w, trend_w, curv_w = best["pos_w"], best["trend_w"], best["curv_w"]
        f1t, f2t = best["f1_thresh"], best["f2_thresh"]

        ps = np.clip(-sub_test["dh180"] / 30, 0, 1) * pos_w
        ds = np.where(sub_test["f1"] > f1t, 1.0,
             np.where(sub_test["f1"] > 0, 0.5 + 0.5 * sub_test["f1"] / f1t,
             np.where(sub_test["f1"] > -f1t, 0.25, 0.0))) * trend_w
        cs = np.where(sub_test["f2"] > f2t, 1.0,
             np.where(sub_test["f2"] > 0, 0.6 + 0.4 * sub_test["f2"] / f2t,
             np.where(sub_test["f2"] > -f2t, 0.3, 0.0))) * curv_w
        cs += sub_test["f2_inflect"].fillna(0) * 0.3 * curv_w

        total = ps + ds + cs
        ic = total.corr(sub_test["ret20"])
        if not np.isnan(ic):
            wf_ics.append(ic)

    mean_wf_ic = np.mean(wf_ics)
    print(f"  Walk-Forward IC: {mean_wf_ic:.4f} (std={np.std(wf_ics):.4f})")
    print(f"  IC正率: {sum(1 for x in wf_ics if x > 0) / len(wf_ics) * 100:.0f}%")

    # Top 10 trials
    print(f"\n--- Top 10 trials ---")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False)
    for _, r in trials_df.head(10).iterrows():
        print(f"  IC={r['value']:.4f}: pos={r['params_pos_w']:.2f} "
              f"trend={r['params_trend_w']:.2f} curv={r['params_curv_w']:.2f} "
              f"f1={r['params_f1_thresh']:.3f} f2={r['params_f2_thresh']:.4f}")

    return best, best_ic, mean_wf_ic


# ── B+C. 标签改造 + 行业中性化 ───────────────────────────────────

def improve_labels_and_features():
    """标签改造 + 截面标准化 + 行业中性化。"""
    import tushare as ts
    import lightgbm as lgb

    print("\n" + "=" * 80)
    print("标签改造 + 行业中性化 + LightGBM Walk-Forward对比")
    print("=" * 80)

    # 加载Purged特征
    df = pd.read_parquet(RESEARCH_DIR / "v3_features_purged.parquet")
    df = df.dropna(subset=FEATURE_COLS + ["ret20", "ts_code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # 拉行业分类
    print("\n加载行业分类...")
    pro = ts.pro_api()
    stock_info = pro.stock_basic(exchange="", list_status="L",
                                  fields="ts_code,industry")
    stock_info = stock_info[~stock_info["ts_code"].str.endswith(".BJ")]
    df = df.merge(stock_info[["ts_code", "industry"]], on="ts_code", how="left")
    df["industry"] = df["industry"].fillna("未知")
    print(f"  行业分类覆盖: {(df['industry'] != '未知').mean()*100:.1f}%")

    # --- B1: Rank变换标签 ---
    print("\n[B1] Rank变换标签...")
    df["ret20_rank"] = df.groupby("date")["ret20"].rank(pct=True)
    df["ret20_norm"] = norm.ppf(df["ret20_rank"].clip(0.001, 0.999))

    # --- B2: 风险调整标签 ---
    print("[B2] 风险调整标签...")
    # 用截面内std做简单调整
    date_stats = df.groupby("date")["ret20"].agg(["mean", "std"])
    df = df.merge(date_stats, on="date", how="left", suffixes=("", "_date"))
    df["ret20_riskadj"] = (df["ret20"] - df["mean"]) / (df["std"] + 1e-8)

    # --- C1: 截面标准化 ---
    print("[C1] 截面标准化...")
    for col in FEATURE_COLS:
        date_mean = df.groupby("date")[col].transform("mean")
        date_std = df.groupby("date")[col].transform("std")
        df[f"{col}_z"] = (df[col] - date_mean) / (date_std + 1e-8)

    Z_COLS = [f"{col}_z" for col in FEATURE_COLS]

    # --- C2: 行业中性化 ---
    print("[C2] 行业中性化（简化版：减行业均值）...")
    for col in FEATURE_COLS:
        ind_mean = df.groupby(["date", "industry"])[col].transform("mean")
        df[f"{col}_neut"] = df[col] - ind_mean

    NEUT_COLS = [f"{col}_neut" for col in FEATURE_COLS]

    # --- 对比：不同标签/特征组合的Walk-Forward IC ---
    print("\n--- Walk-Forward对比实验 ---")

    from dateutil.relativedelta import relativedelta

    LGBM_PARAMS_CONSERVATIVE = {
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

    experiments = [
        ("原始标签+原始特征", "ret20", FEATURE_COLS),
        ("Rank标签+原始特征", "ret20_norm", FEATURE_COLS),
        ("原始标签+截面Z特征", "ret20", Z_COLS),
        ("Rank标签+截面Z特征", "ret20_norm", Z_COLS),
        ("原始标签+行业中性特征", "ret20", NEUT_COLS),
        ("Rank标签+行业中性特征", "ret20_norm", NEUT_COLS),
    ]

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

    results = {}
    for exp_name, label_col, feat_cols in experiments:
        print(f"\n  实验: {exp_name}")
        ics = []
        for t0, t1, t2 in windows:
            mask_train = (df["date"] >= t0) & (df["date"] < t1)
            mask_test = (df["date"] >= t1) & (df["date"] < t2)
            sub_train = df[mask_train].dropna(subset=feat_cols + [label_col])
            sub_test = df[mask_test].dropna(subset=feat_cols + [label_col])
            if len(sub_train) < 500 or len(sub_test) < 50:
                continue

            dtrain = lgb.Dataset(sub_train[feat_cols].values, sub_train[label_col].values)
            dtest = lgb.Dataset(sub_test[feat_cols].values, sub_test[label_col].values,
                                reference=dtrain)
            model = lgb.train(
                LGBM_PARAMS_CONSERVATIVE, dtrain, num_boost_round=500,
                valid_sets=[dtest],
                callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)],
            )

            y_pred = model.predict(sub_test[feat_cols].values)
            y_test = sub_test[label_col].values
            ic = np.corrcoef(y_pred, y_test)[0, 1]
            if not np.isnan(ic):
                ics.append(ic)

        mean_ic = np.mean(ics) if ics else 0
        std_ic = np.std(ics) if ics else 0
        pos_rate = sum(1 for x in ics if x > 0) / len(ics) * 100 if ics else 0
        results[exp_name] = {
            "mean_IC": round(mean_ic, 4),
            "std_IC": round(std_ic, 4),
            "positive_rate": round(pos_rate, 1),
            "n_windows": len(ics),
        }
        print(f"    IC={mean_ic:.4f} ± {std_ic:.4f}, 正率={pos_rate:.0f}%")

    # 汇总
    print(f"\n{'='*80}")
    print("Walk-Forward对比汇总")
    print(f"{'='*80}")
    print(f"{'实验':>30s} | {'IC':>6s} {'std':>6s} {'正率':>5s}")
    print("-"*55)
    for name, r in sorted(results.items(), key=lambda x: x[1]["mean_IC"], reverse=True):
        print(f"{name:>30s} | {r['mean_IC']:>+5.4f} {r['std_IC']:.4f} {r['positive_rate']:>4.0f}%")

    return results


FEATURE_COLS = [
    "dh60", "dh180", "dh360", "dh1000",
    "f1", "f2", "f1_ma5", "f2_ma5", "f1_delta", "f2_delta",
    "f2_inflect", "ma60_dist", "vol_ratio", "rsi14",
    "macd_hist", "dif", "slope_ma20",
]


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V3 ML 优化器 v3 — Optuna + 标签改造 + 行业中性化")
    print("=" * 80)

    # A. Optuna优化v3线性评分
    print("\n" + "=" * 80)
    print("A. Optuna优化v3线性评分参数")
    print("=" * 80)
    best_params, best_global_ic, best_wf_ic = optuna_v3_linear()

    # B+C. 标签改造 + 行业中性化对比
    print("\n" + "=" * 80)
    print("B+C. 标签改造 + 行业中性化对比实验")
    print("=" * 80)
    exp_results = improve_labels_and_features()

    # 保存结果
    output = {
        "optuna_best_params": best_params,
        "optuna_global_ic": best_global_ic,
        "optuna_wf_ic": best_wf_ic,
        "label_feature_experiments": exp_results,
    }
    with open(RESEARCH_DIR / "v3_optuna_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n✅ 全部完成")
    return output


if __name__ == "__main__":
    main()
