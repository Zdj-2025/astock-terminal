#!/usr/bin/env python3
"""V3 ML 优化器 v5 — 量能标准化 + 导数 + 量价配合回测。

选股思路（合作者提出）：
  成交量 → 标准化（20日均值归一化）→ ln → S-G求导
  量能f' = 放量/缩量的速度
  量能f'' = 放量的加速度
  
  配合价格导数：
  价格涨 + 量能涨 = 量价齐升（最强）
  价格涨 + 量能缩 = 量价背离（警惕）
  价格跌 + 量能缩 = 缩量下跌（可能见底）
  价格跌 + 量能涨 = 放量下跌（恐慌）

用法：
  python scripts/research/v3_ml_optimizer_v5.py
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


def load_fullmarket():
    """加载全市场数据，计算量能导数。"""
    print("加载全市场数据...")
    feat_path = RESEARCH_DIR / "v3_features_purged.parquet"
    df = pd.read_parquet(feat_path)
    df = df.dropna(subset=FEATURE_COLS + ["ret20", "ts_code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Purged数据: {len(df):,}条")
    return df


def compute_volume_derivatives(df_raw_path="/tmp/fullmarket_4y.parquet"):
    """从原始日线数据计算量能标准化+导数 — 向量化版。"""
    print("\n从原始日线计算量能导数（向量化）...")
    
    raw = pd.read_parquet(df_raw_path)
    print(f"  原始数据: {len(raw):,}条")
    
    # 列名标准化
    if "trade_date" in raw.columns:
        raw = raw.rename(columns={"trade_date": "date"})
    if "vol" in raw.columns and "volume" not in raw.columns:
        raw = raw.rename(columns={"vol": "volume"})
    
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[~raw["ts_code"].str.endswith(".BJ")].copy()
    raw = raw.sort_values(["ts_code", "date"]).reset_index(drop=True)
    print(f"  过滤后: {len(raw):,}条")
    
    # 按股票分组计算
    def calc_group(group):
        if len(group) < 60:
            return None
        
        close = group["close"].values.astype(float)
        volume = group["volume"].values.astype(float)
        
        # 量能标准化
        vol_ma20 = pd.Series(volume).rolling(20, min_periods=10).mean().values
        vol_ratio = volume / (vol_ma20 + 1e-8)
        
        # ln(vol_ratio) → 清理NaN/Inf
        ln_vr = np.log(np.clip(vol_ratio, 0.01, 100))
        # 用前值填充NaN（主要是前20天）
        mask = np.isfinite(ln_vr)
        if mask.sum() < 60:
            return None
        # 插值填充NaN
        ln_vr_series = pd.Series(ln_vr)
        ln_vr = ln_vr_series.interpolate().bfill().ffill().values
        
        ln_p = np.log(close)
        ln_p = pd.Series(ln_p).interpolate().bfill().ffill().values
        
        try:
            vf1 = savgol_filter(ln_vr, 21, 3, deriv=1) * 100
            vf2 = savgol_filter(ln_vr, 21, 3, deriv=2) * 10000
            pf1 = savgol_filter(ln_p, 21, 3, deriv=1) * 100
            pf2 = savgol_filter(ln_p, 21, 3, deriv=2) * 10000
        except Exception:
            return None
        
        # 检查NaN
        if np.any(~np.isfinite(vf1)) or np.any(~np.isfinite(vf2)):
            return None
        
        result = group[["ts_code", "date"]].copy()
        result["vol_ratio_raw"] = vol_ratio
        result["vf1"] = vf1
        result["vf2"] = vf2
        result["pf1"] = pf1
        result["pf2"] = pf2
        return result
    
    groups = raw.groupby("ts_code")
    total = len(groups)
    
    chunks = []
    for i, (_, group) in enumerate(groups):
        if i % 1000 == 0 and i > 0:
            print(f"    {i}/{total}...")
        r = calc_group(group)
        if r is not None:
            chunks.append(r)
    
    vol_df = pd.concat(chunks, ignore_index=True)
    print(f"  量能导数计算完成: {len(vol_df):,}条")
    
    vol_df.to_parquet(RESEARCH_DIR / "v5_volume_derivatives.parquet", index=False)
    print(f"  保存到 v5_volume_derivatives.parquet")
    
    return vol_df


def analyze_volume_price_synergy(vol_df):
    """量价配合回测。"""
    print("\n" + "=" * 80)
    print("量价配合分析")
    print("=" * 80)
    
    # 加载Purged特征
    df = pd.read_parquet(RESEARCH_DIR / "v3_features_purged.parquet")
    df = df.dropna(subset=FEATURE_COLS + ["ret20", "ts_code", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Merge量能导数
    df = df.merge(vol_df, on=["ts_code", "date"], how="inner")
    print(f"  Merge后: {len(df):,}条")
    
    # 量价配合分类
    def classify_vp(pf1, vf1):
        if pf1 > 0 and vf1 > 0:
            return "量价齐升"
        elif pf1 > 0 and vf1 <= 0:
            return "价涨量缩"
        elif pf1 <= 0 and vf1 <= 0:
            return "缩量下跌"
        else:
            return "放量下跌"
    
    df["vp_signal"] = df.apply(lambda r: classify_vp(r["pf1"], r["vf1"]), axis=1)
    
    # 各信号收益统计
    print("\n--- 量价配合信号 → 20日收益 ---")
    for sig in ["量价齐升", "价涨量缩", "缩量下跌", "放量下跌"]:
        sub = df[df["vp_signal"] == sig]
        if len(sub) == 0:
            continue
        avg_ret = sub["ret20"].mean()
        wr = (sub["ret20"] > 0).mean() * 100
        print(f"  {sig}: 平均{avg_ret:+.2f}%, 胜率{wr:.1f}%, 样本{len(sub):,}")
    
    # 量能导数IC
    print("\n--- 量能导数因子IC ---")
    for col in ["vf1", "vf2", "vol_ratio_raw"]:
        ic = df[col].corr(df["ret20"])
        print(f"  {col}: IC={ic:+.4f}")
    
    # 量能f'' IC
    vf2_ic = df["vf2"].corr(df["ret20"])
    pf2_ic = df["pf2"].corr(df["ret20"])
    print(f"\n  价格f'' IC: {pf2_ic:+.4f}")
    print(f"  量能f'' IC: {vf2_ic:+.4f}")
    
    # 量价配合 + 价格f'' 的组合矩阵
    print("\n--- 价格f'' × 量能f'' 组合矩阵 ---")
    df["pf2_pos"] = df["pf2"] > 0.01
    df["vf2_pos"] = df["vf2"] > 0
    
    for pf2_sign in [True, False]:
        for vf2_sign in [True, False]:
            sub = df[(df["pf2_pos"] == pf2_sign) & (df["vf2_pos"] == vf2_sign)]
            if len(sub) < 100:
                continue
            avg_ret = sub["ret20"].mean()
            wr = (sub["ret20"] > 0).mean() * 100
            label = f"价格f''{'正' if pf2_sign else '负'} + 量能f''{'正' if vf2_sign else '负'}"
            print(f"  {label}: 收益{avg_ret:+.2f}%, 胜率{wr:.1f}%, 样本{len(sub):,}")
    
    # 加入量能导数后的LightGBM回测
    print("\n--- 加入量能导数的LightGBM Walk-Forward ---")
    import lightgbm as lgb
    
    # 扩展特征
    EXTENDED_COLS = FEATURE_COLS + ["vf1", "vf2", "vol_ratio_raw"]
    
    # 截面Z标准化（所有特征+量能）
    all_features = FEATURE_COLS + ["vf1", "vf2", "vol_ratio_raw"]
    for col in all_features:
        dm = df.groupby("date")[col].transform("mean")
        ds = df.groupby("date")[col].transform("std")
        df[f"{col}_z"] = (df[col] - dm) / (ds + 1e-8)
    
    Z_BASE = [f"{col}_z" for col in FEATURE_COLS]
    Z_EXTENDED = Z_BASE + ["vf1_z", "vf2_z", "vol_ratio_raw_z"]
    
    # Rank标签
    df["ret20_rank"] = df.groupby("date")["ret20"].rank(pct=True)
    df["ret20_norm"] = norm.ppf(df["ret20_rank"].clip(0.001, 0.999))
    
    LGBM_PARAMS = {
        "objective": "regression", "metric": "mae",
        "boosting_type": "gbdt", "num_leaves": 31,
        "learning_rate": 0.05, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "verbose": -1, "n_jobs": -1, "max_depth": 6,
        "min_child_samples": 500, "reg_alpha": 0.1, "reg_lambda": 0.1,
    }
    
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
    
    # 对比：有量能 vs 无量能
    experiments = [
        ("无量能(基线)", Z_BASE),
        ("+量能导数", Z_EXTENDED),
    ]
    
    for exp_name, z_cols in experiments:
        ics = []
        for t0, t1, t2 in windows:
            mask_train = (df["date"] >= t0) & (df["date"] < t1)
            mask_test = (df["date"] >= t1) & (df["date"] < t2)
            sub_train = df[mask_train].dropna(subset=z_cols + ["ret20_norm"])
            sub_test = df[mask_test].dropna(subset=z_cols + ["ret20_norm"])
            if len(sub_train) < 500 or len(sub_test) < 50:
                continue
            
            dtrain = lgb.Dataset(sub_train[z_cols].values, sub_train["ret20_norm"].values)
            dtest = lgb.Dataset(sub_test[z_cols].values, sub_test["ret20_norm"].values)
            model = lgb.train(
                LGBM_PARAMS, dtrain, num_boost_round=500,
                valid_sets=[dtest],
                callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)],
            )
            
            y_pred = model.predict(sub_test[z_cols].values)
            ic = np.corrcoef(y_pred, sub_test["ret20_norm"].values)[0, 1]
            if not np.isnan(ic):
                ics.append(ic)
        
        mean_ic = np.mean(ics) if ics else 0
        std_ic = np.std(ics) if ics else 0
        pos_rate = sum(1 for x in ics if x > 0) / len(ics) * 100 if ics else 0
        print(f"  {exp_name}: IC={mean_ic:+.4f} ± {std_ic:.4f}, 正率={pos_rate:.0f}%")
        
        # 特征重要性
        if exp_name == "+量能导数":
            imp = pd.DataFrame({
                "feature": z_cols,
                "gain": model.feature_importance("gain"),
            }).sort_values("gain", ascending=False)
            print(f"\n--- 量能导数在特征重要性中的位置 ---")
            for _, r in imp.iterrows():
                print(f"    {r['feature']:>20s}: {r['gain']:>10.0f}")
    
    return df


def main():
    print("=" * 80)
    print("V3 ML 优化器 v5 — 量能标准化 + 导数 + 量价配合")
    print("=" * 80)
    
    # Step 1: 计算量能导数
    vol_deriv_path = RESEARCH_DIR / "v5_volume_derivatives.parquet"
    if vol_deriv_path.exists():
        print("加载已存在的量能导数数据...")
        vol_df = pd.read_parquet(vol_deriv_path)
    else:
        vol_df = compute_volume_derivatives()
    
    # Step 2: 量价配合分析 + LightGBM对比
    result_df = analyze_volume_price_synergy(vol_df)
    
    print("\n" + "=" * 80)
    print("✅ 量能导数分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
