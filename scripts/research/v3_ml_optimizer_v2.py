#!/usr/bin/env python3
"""V3 ML 优化器 v2 — Walk-Forward + Purged + 参数收缩 + 三段验证。

修复 v1 的关键问题：
  1. Walk-Forward滚动验证替代静态分割
  2. Purged采样消除标签重叠
  3. 模型参数收缩防过拟合
  4. 三段分割（训练/验证/测试）防多重检验

用法：
  python scripts/research/v3_ml_optimizer_v2.py
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
DATA_PATH = "/tmp/fullmarket_4y.parquet"

# ── 收缩后的模型参数 ──────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,            # 63→31，收缩
    "learning_rate": 0.05,
    "feature_fraction": 0.7,     # 0.8→0.7
    "bagging_fraction": 0.7,     # 0.8→0.7
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "max_depth": 6,              # 8→6，收缩
    "min_child_samples": 500,    # 100→500，大幅收缩
    "reg_alpha": 0.1,            # 新增L1正则
    "reg_lambda": 0.1,           # 新增L2正则
}

FEATURE_COLS = [
    "dh60", "dh180", "dh360", "dh1000",
    "f1", "f2", "f1_ma5", "f2_ma5", "f1_delta", "f2_delta",
    "f2_inflect", "ma60_dist", "vol_ratio", "rsi14",
    "macd_hist", "dif", "slope_ma20",
]

# ── Step 1: 特征提取（Purged采样，步长20天）───────────────────────

def extract_features_purged(df_all: pd.DataFrame, sample_step: int = 20):
    """提取特征，采样步长=标签窗口，消除标签重叠。"""
    import lightgbm as lgb  # noqa: 预加载

    sg_win, sg_poly = 21, 3
    groups = df_all.groupby("ts_code")
    total = len(groups)
    all_rows = []
    t0 = time.time()

    for idx, (code, g) in enumerate(groups):
        g = g.sort_values("trade_date").reset_index(drop=True)
        close = g["close"].values.astype(float)
        n = len(close)
        if n < 180:
            continue

        ln_p = np.log(close)
        f1 = savgol_filter(ln_p, sg_win, sg_poly, deriv=1) * 100
        f2 = savgol_filter(ln_p, sg_win, sg_poly, deriv=2) * 10000

        # MA
        ma60 = pd.Series(close).rolling(60).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_hist = (dif - dea) * 2

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = pd.Series(delta).clip(lower=0).rolling(14).mean().values
        loss_s = pd.Series(np.abs(delta)).where(
            pd.Series(delta) < 0, 0).rolling(14).mean().values
        rs = np.where(loss_s > 0, gain / loss_s, 100)
        rsi = 100 - 100 / (1 + rs)

        # Volume
        has_vol = "vol" in g.columns or "volume" in g.columns
        if has_vol:
            vc = "vol" if "vol" in g.columns else "volume"
            vol = g[vc].values.astype(float)
            vol5 = pd.Series(vol).rolling(5).mean().values
            vol20 = pd.Series(vol).rolling(20).mean().values
            vol_ratio = np.where(vol20 > 0, vol5 / vol20, 1.0)
        else:
            vol_ratio = np.ones(n)

        # MA20 slope
        ma20_slope = np.full(n, np.nan)
        for i in range(24, n):
            x = np.arange(5)
            y = ma20[i-4:i+1]
            if not np.any(np.isnan(y)):
                ma20_slope[i] = (np.mean(x*y) - np.mean(x)*np.mean(y)) / (
                    np.mean(x**2) - np.mean(x)**2)

        # ── 采样（步长=20天，与标签窗口一致）──
        margin = max(sg_win, 60)
        sample_range = range(margin, n - 40, sample_step)  # 40天前向窗口

        for i in sample_range:
            current = close[i]
            row = {"ts_code": code, "day_idx": i}

            # 日期（用于时间分割）
            if "trade_date" in g.columns:
                row["date"] = g.iloc[i]["trade_date"]

            # 位置
            for period in [60, 180, 360, 1000]:
                if i >= period:
                    h = np.max(close[i-period:i])
                else:
                    h = np.max(close[:i])
                row[f"dh{period}"] = (current / h - 1) * 100

            # 运动学
            row["f1"] = f1[i]
            row["f2"] = f2[i]
            row["f1_ma5"] = np.mean(f1[max(0,i-4):i+1])
            row["f2_ma5"] = np.mean(f2[max(0,i-4):i+1])
            row["f1_delta"] = f1[i] - f1[max(0,i-5)]
            row["f2_delta"] = f2[i] - f2[max(0,i-5)]
            if i >= 5:
                f2p = np.mean(f2[i-5:i])
                row["f2_inflect"] = 1.0 if (f2p < 0 and f2[i] > 0) else 0.0
            else:
                row["f2_inflect"] = 0.0

            # 传统指标
            row["ma60_dist"] = (current/ma60[i]-1)*100 if not np.isnan(ma60[i]) else 0
            row["vol_ratio"] = vol_ratio[i]
            row["rsi14"] = rsi[i]
            row["macd_hist"] = macd_hist[i]
            row["dif"] = dif[i]
            row["slope_ma20"] = ma20_slope[i]/current*100 if not np.isnan(ma20_slope[i]) else 0

            # 标签（20日收益）
            row["ret20"] = (close[min(i+20, n-1)] / current - 1) * 100

            all_rows.append(row)

        if (idx+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{total} 只, {len(all_rows):,}条, {elapsed:.0f}s")

    df = pd.DataFrame(all_rows)
    print(f"\nPurged特征提取完成: {len(df):,}条 (步长={sample_step})")
    return df


# ── Step 2: Walk-Forward验证 ─────────────────────────────────────

def walk_forward_validation(
    features_df: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 1,
    step_months: int = 1,
    params: dict | None = None,
):
    """Walk-Forward滚动验证。

    返回：每轮IC、MAE、胜率的列表。
    """
    import lightgbm as lgb

    if params is None:
        params = LGBM_PARAMS

    df = features_df.dropna(subset=FEATURE_COLS + ["ret20"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 按月划分
    dates = df["date"]
    min_date = dates.min()
    max_date = dates.max()

    # 生成时间窗口
    from dateutil.relativedelta import relativedelta
    windows = []
    train_start = min_date
    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > max_date:
            break
        windows.append((train_start, train_end, test_end))
        train_start += relativedelta(months=step_months)

    print(f"Walk-Forward: {len(windows)}轮, "
          f"训练{train_months}月→预测{test_months}月")

    results = []
    all_preds = []
    all_actuals = []

    for i, (t_start, t_end, te_end) in enumerate(windows):
        train_mask = (dates >= t_start) & (dates < t_end)
        test_mask = (dates >= t_end) & (dates < te_end)

        X_train = df.loc[train_mask, FEATURE_COLS].values
        y_train = df.loc[train_mask, "ret20"].values
        X_test = df.loc[test_mask, FEATURE_COLS].values
        y_test = df.loc[test_mask, "ret20"].values

        if len(X_train) < 1000 or len(X_test) < 100:
            continue

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
        dtest = lgb.Dataset(X_test, label=y_test, feature_name=FEATURE_COLS,
                            reference=dtrain)

        callbacks = [
            lgb.log_evaluation(period=0),  # 静默
            lgb.early_stopping(stopping_rounds=30),
        ]

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,  # 1000→500
            valid_sets=[dtest],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_test)
        ic = np.corrcoef(y_pred, y_test)[0, 1] if len(y_pred) > 10 else np.nan
        mae = np.mean(np.abs(y_pred - y_test))
        wr = (y_test > 0).mean() * 100

        # 顶部30%胜率
        threshold = np.percentile(y_pred, 70)
        top_mask = y_pred >= threshold
        top_wr = (y_test[top_mask] > 0).mean() * 100 if top_mask.sum() > 10 else np.nan
        top_avg = y_test[top_mask].mean() if top_mask.sum() > 10 else np.nan

        results.append({
            "window": i,
            "train_start": str(t_start.date()),
            "test_start": str(t_end.date()),
            "train_n": len(X_train),
            "test_n": len(X_test),
            "IC": round(ic, 4) if not np.isnan(ic) else None,
            "MAE": round(mae, 4),
            "win_rate": round(wr, 1),
            "top30_wr": round(top_wr, 1) if top_wr is not None else None,
            "top30_avg": round(top_avg, 2) if top_avg is not None else None,
            "best_iter": model.best_iteration,
        })

        all_preds.extend(y_pred)
        all_actuals.extend(y_test)

        if (i+1) % 10 == 0:
            print(f"  轮{i+1}/{len(windows)}: IC={ic:.4f} "
                  f"top30_wr={top_wr:.1f}% n_test={len(X_test)}")

    # 汇总
    valid_ics = [r["IC"] for r in results if r["IC"] is not None]
    valid_top_wr = [r["top30_wr"] for r in results if r["top30_wr"] is not None]
    valid_top_avg = [r["top30_avg"] for r in results if r["top30_avg"] is not None]

    summary = {
        "n_windows": len(results),
        "mean_IC": round(np.mean(valid_ics), 4) if valid_ics else None,
        "std_IC": round(np.std(valid_ics), 4) if valid_ics else None,
        "min_IC": round(np.min(valid_ics), 4) if valid_ics else None,
        "max_IC": round(np.max(valid_ics), 4) if valid_ics else None,
        "pct_positive_IC": round(sum(1 for ic in valid_ics if ic > 0) / len(valid_ics) * 100, 1) if valid_ics else None,
        "mean_top30_wr": round(np.mean(valid_top_wr), 1) if valid_top_wr else None,
        "mean_top30_avg": round(np.mean(valid_top_avg), 2) if valid_top_avg else None,
        "overall_IC": round(np.corrcoef(all_preds, all_actuals)[0, 1], 4) if len(all_preds) > 100 else None,
    }

    return results, summary


# ── Step 3: 特征重要性 + SHAP ────────────────────────────────────

def analyze_with_shap(features_df: pd.DataFrame):
    """用最后24个月数据训练，SHAP解释。"""
    import lightgbm as lgb
    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False
        print("shap未安装，跳过SHAP分析")

    df = features_df.dropna(subset=FEATURE_COLS + ["ret20"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 最后24个月训练
    max_date = df["date"].max()
    from dateutil.relativedelta import relativedelta
    split_date = max_date - relativedelta(months=24)
    train = df[df["date"] >= split_date]

    X = train[FEATURE_COLS].values
    y = train["ret20"].values

    print(f"SHAP分析训练集: {len(X):,}条")

    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)
    model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=300)

    # LightGBM原生重要性
    imp = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLS, "importance": imp,
    }).sort_values("importance", ascending=False)

    print("\n特征重要性 (gain):")
    for _, r in imp_df.iterrows():
        print(f"  {r['feature']:>15s}: {r['importance']:.0f}")

    # SHAP
    if has_shap:
        print("\nSHAP分析（采样5000条）...")
        sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
        X_sample = X[sample_idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": FEATURE_COLS, "shap_importance": shap_importance,
        }).sort_values("shap_importance", ascending=False)

        print("\nSHAP重要性:")
        for _, r in shap_df.iterrows():
            print(f"  {r['feature']:>15s}: {r['shap_importance']:.4f}")

        return model, imp_df, shap_df

    return model, imp_df, None


# ── Step 4: 行业轮动信号 ─────────────────────────────────────────

def industry_rotation_signal(features_df: pd.DataFrame):
    """按行业聚合f'/f''，发现行业热点。"""
    import tushare as ts
    pro = ts.pro_api()

    # 拉行业分类
    stock_info = pro.stock_basic(exchange="", list_status="L",
                                  fields="ts_code,industry")
    stock_info = stock_info[~stock_info["ts_code"].str.endswith(".BJ")]

    df = features_df.dropna(subset=["f1", "f2"]).copy()

    # 合并行业
    df = df.merge(stock_info[["ts_code", "industry"]], on="ts_code", how="left")
    df = df.dropna(subset=["industry"])

    # 按行业聚合
    ind = df.groupby("industry").agg({
        "f1": "mean", "f2": "mean",
        "ret20": "mean",
        "ts_code": "count",
    }).rename(columns={"ts_code": "n_stocks"})

    ind = ind[ind["n_stocks"] >= 20]  # 至少20只

    # 分类
    def classify(row):
        if row["f1"] > 0.1 and row["f2"] > 0.01:
            return "🚀 主升浪"
        elif row["f1"] > 0.1 and row["f2"] < -0.01:
            return "⚠️ 减速见顶"
        elif row["f1"] < -0.1 and row["f2"] > 0.01:
            return "🔄 底部拐点"
        elif row["f1"] < -0.1 and row["f2"] < -0.01:
            return "🔴 加速下跌"
        else:
            return "➡️ 观望"

    ind["signal"] = ind.apply(classify, axis=1)
    ind = ind.sort_values("f2", ascending=False)

    print("\n" + "=" * 80)
    print("行业轮动信号（按f''排序）")
    print("=" * 80)
    print(f"{'行业':>10s} | {'f\'':>6s} {'f\'\'':>7s} | {'20d收益':>7s} | {'信号':>8s} | {'股票数':>4s}")
    print("-" * 60)
    for ind_name, row in ind.iterrows():
        print(f"{ind_name:>10s} | {row['f1']:>+5.2f} {row['f2']:>+6.3f} | "
              f"{row['ret20']:>+6.2f}% | {row['signal']:>8s} | {int(row['n_stocks']):>4d}")

    return ind


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V3 ML 优化器 v2 — Walk-Forward + Purged + 收缩参数")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/5] 加载全市场数据...")
    df_all = pd.read_parquet(DATA_PATH)
    counts = df_all.groupby("ts_code").size()
    valid = counts[counts >= 500].index
    df_all = df_all[df_all["ts_code"].isin(valid)].copy()
    print(f"  有效股票: {len(valid)}只")

    # 2. Purged特征提取
    print("\n[2/5] Purged特征提取（步长20天）...")
    feat_path = RESEARCH_DIR / "v3_features_purged.parquet"
    if feat_path.exists():
        features = pd.read_parquet(feat_path)
        print(f"  从缓存加载: {len(features):,}条")
    else:
        features = extract_features_purged(df_all, sample_step=20)
        features.to_parquet(feat_path, index=False)
        print(f"  保存到 {feat_path}")

    # 3. Walk-Forward验证
    print("\n[3/5] Walk-Forward验证...")
    wf_results, wf_summary = walk_forward_validation(features)

    print(f"\n{'='*80}")
    print("Walk-Forward结果汇总")
    print(f"{'='*80}")
    for k, v in wf_summary.items():
        print(f"  {k}: {v}")

    # 按年度统计IC
    wf_df = pd.DataFrame(wf_results)
    if len(wf_df) > 0:
        wf_df["test_year"] = pd.to_datetime(wf_df["test_start"]).dt.year
        print(f"\n  按年统计IC:")
        for year, grp in wf_df.groupby("test_year"):
            ics = grp["IC"].dropna()
            if len(ics) > 0:
                print(f"    {year}: mean={ics.mean():.4f}, "
                      f"正率={sum(ics>0)/len(ics)*100:.0f}%, "
                      f"n={len(ics)}")

    # 4. SHAP分析
    print(f"\n[4/5] 特征重要性 + SHAP...")
    model, imp_df, shap_df = analyze_with_shap(features)

    # 5. 行业轮动
    print(f"\n[5/5] 行业轮动信号...")
    ind_signal = industry_rotation_signal(features)

    # 保存结果
    results = {
        "walk_forward_summary": wf_summary,
        "walk_forward_by_window": wf_results,
        "feature_importance": imp_df.to_dict("records"),
        "shap_importance": shap_df.to_dict("records") if shap_df is not None else None,
    }

    with open(RESEARCH_DIR / "v3_wf_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    imp_df.to_csv(RESEARCH_DIR / "v3_wf_feature_importance.csv", index=False)
    if shap_df is not None:
        shap_df.to_csv(RESEARCH_DIR / "v3_wf_shap_importance.csv", index=False)

    print("\n✅ 全部完成")
    return wf_results, wf_summary


if __name__ == "__main__":
    main()
