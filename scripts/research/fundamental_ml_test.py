#!/usr/bin/env python3
"""基本面特征加入 LightGBM 的 IC 变化测试。

测试目标：在原有 17 个技术特征基础上，加入 daily_basic 中的基本面截面特征，
比较 IC (Information Coefficient) 的变化。

基本面特征（截面 rank 标准化）：
  - pe_rank: PE 百分位
  - pb_rank: PB 百分位
  - mv_rank: 总市值百分位
  - turnover_rank: 换手率百分位
  - turnover_rate: 原始换手率

对比实验：
  A) 原始 17 特征
  B) 17 特征 + 5 基本面特征
  C) 仅 5 基本面特征

用法：
  cd .
  PYTHONPATH=. python scripts/research/fundamental_ml_test.py
"""
from __future__ import annotations

import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────

STOCK_ROOT = Path(".")
DATA_ROOT = STOCK_ROOT / "data" / "store"
RESEARCH_DIR = STOCK_ROOT / "scripts" / "research"
DAILY_BASIC_DIR = DATA_ROOT / "daily_basic"

# ── 原始 17 特征列 ────────────────────────────────────────────────────

FEATURE_COLS_17 = [
    "dh60", "dh180", "dh360", "dh1000",
    "f1", "f2", "f1_ma5", "f2_ma5", "f1_delta", "f2_delta",
    "f2_inflect", "ma60_dist", "vol_ratio", "rsi14",
    "macd_hist", "dif", "slope_ma20",
]

FUNDAMENTAL_COLS = [
    "pe_rank", "pb_rank", "mv_rank", "turnover_rank", "turnover_rate",
]


# ── Step 1: 加载技术特征 ──────────────────────────────────────────────

def load_tech_features() -> pd.DataFrame:
    """加载 purged 技术特征数据。"""
    print("=" * 60)
    print("Step 1: 加载技术特征")
    print("=" * 60)
    
    feat_path = RESEARCH_DIR / "v3_features_purged.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"技术特征文件不存在: {feat_path}")
    
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 过滤无效标签
    df = df.dropna(subset=FEATURE_COLS_17 + ["ret20", "ts_code", "date"]).copy()
    
    print(f"  技术特征: {len(df):,} 条, {df['ts_code'].nunique()} 只股票")
    print(f"  日期范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    return df


# ── Step 2: 加载 daily_basic 并计算截面 rank ──────────────────────────

def load_fundamental_features() -> pd.DataFrame:
    """加载 daily_basic 数据，计算截面 rank 特征。

    Returns:
        DataFrame with columns: ts_code, date, pe_rank, pb_rank, mv_rank, turnover_rank, turnover_rate
    """
    print("\n" + "=" * 60)
    print("Step 2: 加载 daily_basic 并计算截面 rank")
    print("=" * 60)
    
    # 扫描所有日期文件
    date_files = sorted([
        f for f in DAILY_BASIC_DIR.iterdir()
        if f.suffix == ".parquet" and f.stem.isdigit() and len(f.stem) == 8
    ])
    print(f"  找到 {len(date_files)} 个日期文件")
    
    if not date_files:
        raise FileNotFoundError(f"daily_basic 目录下没有日期文件: {DAILY_BASIC_DIR}")
    
    # 批量加载
    rows_list = []
    chunk_size = 200
    for i in range(0, len(date_files), chunk_size):
        chunk = date_files[i: i + chunk_size]
        for fp in chunk:
            try:
                df_day = pd.read_parquet(fp, columns=[
                    "ts_code", "trade_date", "pe", "pb", "total_mv", "turnover_rate",
                ])
                if len(df_day) == 0:
                    continue
                date_str = fp.stem
                df_day["date"] = pd.to_datetime(date_str, format="%Y%m%d")
                
                # 截面 rank
                n = len(df_day)
                for col in ["pe", "pb", "total_mv", "turnover_rate"]:
                    valid = df_day[col].dropna()
                    if len(valid) > 0:
                        df_day[f"{col}_rank"] = (valid.rank(pct=True)).reindex(df_day.index)
                    else:
                        df_day[f"{col}_rank"] = np.nan
                
                # 重命名
                df_day = df_day.rename(columns={
                    "pe_rank": "pe_rank",
                    "pb_rank": "pb_rank",
                    "total_mv_rank": "mv_rank",
                    "turnover_rate_rank": "turnover_rank",
                })
                
                rows_list.append(df_day[[
                    "ts_code", "date", "pe_rank", "pb_rank", "mv_rank",
                    "turnover_rank", "turnover_rate",
                ]])
            except Exception as e:
                continue
        
        if (i // chunk_size + 1) % 5 == 0:
            print(f"  已加载 {min(i + chunk_size, len(date_files))}/{len(date_files)} 个日期文件")
    
    if not rows_list:
        raise RuntimeError("未能加载任何 daily_basic 数据")
    
    df_fund = pd.concat(rows_list, ignore_index=True)
    print(f"  基本面特征: {len(df_fund):,} 条")
    print(f"  日期范围: {df_fund['date'].min().strftime('%Y-%m-%d')} ~ {df_fund['date'].max().strftime('%Y-%m-%d')}")
    print(f"  缺失率:")
    for col in FUNDAMENTAL_COLS:
        miss = df_fund[col].isna().mean() * 100
        print(f"    {col}: {miss:.1f}%")
    
    return df_fund


# ── Step 3: 合并特征 ──────────────────────────────────────────────────

def merge_features(df_tech: pd.DataFrame, df_fund: pd.DataFrame) -> pd.DataFrame:
    """合并技术特征和基本面特征。"""
    print("\n" + "=" * 60)
    print("Step 3: 合并技术特征 + 基本面特征")
    print("=" * 60)
    
    before = len(df_tech)
    df = df_tech.merge(
        df_fund,
        on=["ts_code", "date"],
        how="left",
    )
    after = len(df)
    
    print(f"  合并前: {before:,} 条")
    print(f"  合并后: {after:,} 条")
    
    # 基本面匹配率
    matched = df["pb_rank"].notna().sum()
    print(f"  基本面匹配: {matched:,} 条 ({matched/after*100:.1f}%)")
    
    # 填充未匹配的为 0.5（中性）
    for col in FUNDAMENTAL_COLS:
        if col == "turnover_rate":
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(0.5)
    
    return df


# ── Step 4: 滚动 IC 回测 ─────────────────────────────────────────────

def rolling_ic_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "ret20",
    train_days: int = 240,
    test_days: int = 60,
    ic_type: str = "spearman",
) -> pd.DataFrame:
    """滚动窗口 IC 回测。

    使用 LightGBM 训练，计算 spearman rank IC。

    Args:
        df: 数据
        feature_cols: 特征列
        label_col: 标签列
        train_days: 训练窗口（交易日）
        test_days: 测试窗口（交易日）
        ic_type: IC 类型

    Returns:
        各窗口的 IC 结果
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("  ⚠️ LightGBM 未安装，使用简单 Spearman IC 代替")
        return simple_ic_test(df, feature_cols, label_col, test_days)
    
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    
    if n_dates < train_days + test_days:
        print(f"  ⚠️ 日期数({n_dates})不足，减少窗口大小")
        train_days = max(n_dates // 2, 60)
        test_days = max(n_dates // 4, 20)
    
    results = []
    window_idx = 0
    
    # 滑动窗口
    start = 0
    while start + train_days + test_days <= n_dates:
        train_end = start + train_days
        test_end = train_end + test_days
        
        train_dates = dates[start:train_end]
        test_dates = dates[train_end:test_end]
        
        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        if len(df_train) < 1000 or len(df_test) < 100:
            start += test_days
            continue
        
        X_train = df_train[feature_cols].values
        y_train = df_train[label_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[label_col].values
        
        # 清理 NaN
        valid_train = np.isfinite(y_train)
        valid_test = np.isfinite(y_test)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        
        if len(X_train) < 500 or len(X_test) < 50:
            start += test_days
            continue
        
        # LightGBM 训练
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 200,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": 42,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
        )
        
        y_pred = model.predict(X_test)
        
        # Spearman IC
        ic, _ = spearmanr(y_pred, y_test)
        
        # 分层回测：按预测分5组，看各组平均收益
        df_test_valid = df_test[valid_test].copy()
        df_test_valid["pred"] = y_pred
        df_test_valid["pred_group"] = pd.qcut(df_test_valid["pred"], 5, labels=False, duplicates="drop")
        
        group_rets = {}
        for g in range(5):
            g_mask = df_test_valid["pred_group"] == g
            if g_mask.sum() > 0:
                group_rets[f"group_{g}_ret"] = round(df_test_valid.loc[g_mask, label_col].mean(), 4)
            else:
                group_rets[f"group_{g}_ret"] = np.nan
        
        # Top 组胜率
        top_mask = df_test_valid["pred_group"] == 4
        if top_mask.sum() > 0:
            win_rate = (df_test_valid.loc[top_mask, label_col] > 0).mean()
        else:
            win_rate = np.nan
        
        results.append({
            "window": window_idx,
            "train_start": train_dates[0].strftime("%Y-%m-%d"),
            "train_end": train_dates[-1].strftime("%Y-%m-%d"),
            "test_start": test_dates[0].strftime("%Y-%m-%d"),
            "test_end": test_dates[-1].strftime("%Y-%m-%d"),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "ic": round(ic, 4) if np.isfinite(ic) else np.nan,
            "top_win_rate": round(win_rate, 4) if np.isfinite(win_rate) else np.nan,
            **group_rets,
        })
        
        window_idx += 1
        start += test_days
    
    return pd.DataFrame(results)


def simple_ic_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "ret20",
    test_days: int = 60,
) -> pd.DataFrame:
    """简单 Spearman IC（无 LightGBM 时的备选方案）。"""
    dates = sorted(df["date"].unique())
    results = []
    
    for i in range(0, len(dates) - test_days, test_days):
        test_dates = dates[i: i + test_days]
        df_test = df[df["date"].isin(test_dates)].copy()
        
        if len(df_test) < 100:
            continue
        
        # 用第一个特征（通常是 f1）做简单 IC
        for feat in feature_cols[:3]:
            valid = df_test[[feat, label_col]].dropna()
            if len(valid) < 50:
                continue
            ic, _ = spearmanr(valid[feat], valid[label_col])
            if np.isfinite(ic):
                results.append({
                    "window": len(results),
                    "test_start": test_dates[0].strftime("%Y-%m-%d"),
                    "test_end": test_dates[-1].strftime("%Y-%m-%d"),
                    "feature": feat,
                    "ic": round(ic, 4),
                })
    
    return pd.DataFrame(results)


# ── Step 5: 特征重要性分析 ────────────────────────────────────────────

def feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "ret20",
) -> pd.DataFrame | None:
    """计算 LightGBM 特征重要性。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  ⚠️ LightGBM 未安装，跳过特征重要性")
        return None
    
    # 使用最后 60% 数据训练
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * 0.4)]
    df_train = df[df["date"] >= cutoff].copy()
    
    X = df_train[feature_cols].values
    y = df_train[label_col].values
    
    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]
    
    if len(X) < 1000:
        print("  ⚠️ 数据不足，跳过特征重要性")
        return None
    
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }
    
    model = lgb.train(params, train_data, num_boost_round=300)
    
    importance = model.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importance,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    
    fi_df["importance_pct"] = round(fi_df["importance_gain"] / fi_df["importance_gain"].sum() * 100, 2)
    
    return fi_df


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    
    print("🔬 基本面特征 + LightGBM IC 测试")
    print("=" * 60)
    
    # Step 1: 加载技术特征
    df_tech = load_tech_features()
    
    # Step 2: 加载基本面特征
    df_fund = load_fundamental_features()
    
    # Step 3: 合并
    df = merge_features(df_tech, df_fund)
    
    # Step 4: 三组对比实验
    print("\n" + "=" * 60)
    print("Step 4: 滚动 IC 回测")
    print("=" * 60)
    
    experiments = {
        "A_17tech": FEATURE_COLS_17,
        "B_17tech_5fund": FEATURE_COLS_17 + FUNDAMENTAL_COLS,
        "C_5fund": FUNDAMENTAL_COLS,
    }
    
    all_results = {}
    
    for exp_name, feats in experiments.items():
        print(f"\n  ── 实验 {exp_name}: {len(feats)} 特征 ──")
        ic_df = rolling_ic_test(df, feats)
        all_results[exp_name] = ic_df
        
        if len(ic_df) > 0 and "ic" in ic_df.columns:
            mean_ic = ic_df["ic"].mean()
            std_ic = ic_df["ic"].std()
            ic_ir = mean_ic / std_ic if std_ic > 0 else 0
            pos_rate = (ic_df["ic"] > 0).mean()
            
            print(f"    Mean IC: {mean_ic:.4f}")
            print(f"    IC IR:   {ic_ir:.4f}")
            print(f"    IC>0 率: {pos_rate:.2%}")
            
            if "top_win_rate" in ic_df.columns:
                avg_win = ic_df["top_win_rate"].mean()
                print(f"    Top组胜率: {avg_win:.2%}")
        else:
            print(f"    结果为空（可能 LightGBM 未安装）")
    
    # Step 5: 特征重要性（22 特征版本）
    print("\n" + "=" * 60)
    print("Step 5: 特征重要性分析（17 技术 + 5 基本面）")
    print("=" * 60)
    
    fi_df = feature_importance(df, FEATURE_COLS_17 + FUNDAMENTAL_COLS)
    if fi_df is not None:
        print("\n  特征重要性排名:")
        for i, row in fi_df.iterrows():
            marker = "🆕" if row["feature"] in FUNDAMENTAL_COLS else "  "
            print(f"    {marker} {i+1:2d}. {row['feature']:20s}  gain={row['importance_gain']:8.0f}  ({row['importance_pct']:.1f}%)")
    
    # Step 6: 输出汇总
    print("\n" + "=" * 60)
    print("📊 汇总")
    print("=" * 60)
    
    for exp_name, ic_df in all_results.items():
        if len(ic_df) > 0 and "ic" in ic_df.columns:
            mean_ic = ic_df["ic"].mean()
            ic_ir = mean_ic / ic_df["ic"].std() if ic_df["ic"].std() > 0 else 0
            print(f"  {exp_name:25s}  Mean IC={mean_ic:+.4f}  IR={ic_ir:+.4f}")
    
    # 保存结果
    output_dir = RESEARCH_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    for exp_name, ic_df in all_results.items():
        if len(ic_df) > 0:
            ic_df.to_csv(output_dir / f"fundamental_{exp_name}_ic.csv", index=False)
    
    if fi_df is not None:
        fi_df.to_csv(output_dir / "fundamental_feature_importance.csv", index=False)
    
    elapsed = time.time() - t0
    print(f"\n  结果保存至: {output_dir}")
    print(f"  耗时: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
