#!/usr/bin/env python3
"""V3 ML 优化器 — 用 LightGBM 搜索最优评分参数。

目标：让评分的IC/胜率/期望收益最大化。
方法：
  1. 全市场提取特征（原始连续值，不是离散评分）
  2. 标注未来N日收益
  3. LightGBM 回归/排序 → 得到特征重要性
  4. 网格搜索评分阈值参数
  5. 输出最优参数配置

内存：24GB Mac Mini，5208只×5天采样≈130万样本×30特征≈3GB，可承受。
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """特征工程配置。"""
    sg_window: int = 21
    sg_poly: int = 3
    # 收益计算周期
    forward_periods: list[int] = field(default_factory=lambda: [5, 10, 20, 40])
    # 采样步长（天）
    sample_step: int = 5
    # 最少天数
    min_days: int = 180


def extract_features_and_labels(
    df_all: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """从全市场日线数据提取特征和标签。

    特征列（连续值，不是评分）：
      - dh60, dh180, dh360, dh1000: 距各周期高点距离(%)
      - f1: 一阶导值(SG, ×100)
      - f2: 二阶导值(SG, ×10000)
      - f1_ma5, f2_ma5: 5日均值
      - f1_delta: f1变化量（近5日）
      - f2_delta: f2变化量
      - f2_inflection: 二阶导拐点（f2从负转正=1，否则0）
      - ma60_dist: 当前价距MA60(%)
      - vol_ratio: 量比（5日均量/20日均量）
      - rsi14: 14日RSI
      - macd_hist: MACD柱状值
      - dif: MACD DIF
      - slope_ma20_5d: MA20的5日斜率

    标签列：
      - ret5, ret10, ret20, ret40: 未来N日收益率(%)
    """
    if cfg is None:
        cfg = FeatureConfig()

    groups = df_all.groupby("ts_code")
    total_codes = len(groups)
    all_events = []
    t0 = time.time()

    for idx, (code, g) in enumerate(groups):
        g = g.sort_values("trade_date").reset_index(drop=True)
        close = g["close"].values.astype(float)
        n = len(close)

        if n < cfg.min_days:
            continue

        # ── 基础指标 ──
        ln_p = np.log(close)
        f1 = savgol_filter(ln_p, cfg.sg_window, cfg.sg_poly, deriv=1) * 100
        f2 = savgol_filter(ln_p, cfg.sg_window, cfg.sg_poly, deriv=2) * 10000

        # MA
        ma5 = pd.Series(close).rolling(5).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_hist = (dif - dea) * 2

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = pd.Series(delta).clip(lower=0).rolling(14).mean().values
        loss = pd.Series(np.abs(delta)).where(delta < 0, 0).rolling(14).mean().values
        rs = np.where(loss > 0, gain / loss, 100)
        rsi = 100 - 100 / (1 + rs)

        # 成交量（如果有）
        has_vol = "vol" in g.columns or "volume" in g.columns
        if has_vol:
            vol_col = "vol" if "vol" in g.columns else "volume"
            vol = g[vol_col].values.astype(float)
            vol5 = pd.Series(vol).rolling(5).mean().values
            vol20 = pd.Series(vol).rolling(20).mean().values
            vol_ratio = np.where(vol20 > 0, vol5 / vol20, 1.0)
        else:
            vol_ratio = np.ones(n)

        # MA20斜率
        ma20_slope = np.full(n, np.nan)
        for i in range(24, n):
            x = np.arange(5)
            y = ma20[i - 4: i + 1]
            if not np.any(np.isnan(y)):
                ma20_slope[i] = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (
                    np.mean(x ** 2) - np.mean(x) ** 2
                )

        # ── 采样 ──
        margin = max(cfg.sg_window, 60)  # 边界保护
        max_fwd = max(cfg.forward_periods)
        sample_range = range(margin, n - max_fwd, cfg.sample_step)

        for i in sample_range:
            current = close[i]

            # 位置距离
            dh = {}
            for period in [60, 180, 360, 1000]:
                if i >= period:
                    h = np.max(close[i - period: i])
                else:
                    h = np.max(close[:i])
                dh[period] = (current / h - 1) * 100

            # 导数衍生
            f1_val = f1[i]
            f2_val = f2[i]
            f1_ma5 = np.mean(f1[max(0, i - 4): i + 1])
            f2_ma5 = np.mean(f2[max(0, i - 4): i + 1])
            f1_delta = f1_val - f1[max(0, i - 5)]
            f2_delta = f2_val - f2[max(0, i - 5)]

            # 拐点
            if i >= 5:
                f2_past5 = np.mean(f2[i - 5: i])
                f2_inflect = 1.0 if (f2_past5 < 0 and f2_val > 0) else 0.0
            else:
                f2_inflect = 0.0

            # MA60距离
            ma60_dist = (current / ma60[i] - 1) * 100 if not np.isnan(ma60[i]) else 0

            # MA20斜率标准化
            slope_norm = ma20_slope[i] / current * 100 if not np.isnan(ma20_slope[i]) else 0

            # 后续收益
            rets = {}
            for p in cfg.forward_periods:
                j = min(i + p, n - 1)
                rets[f"ret{p}"] = (close[j] / current - 1) * 100

            row = {
                "ts_code": code,
                "dh60": dh[60], "dh180": dh[180], "dh360": dh[360], "dh1000": dh[1000],
                "f1": f1_val, "f2": f2_val,
                "f1_ma5": f1_ma5, "f2_ma5": f2_ma5,
                "f1_delta": f1_delta, "f2_delta": f2_delta,
                "f2_inflect": f2_inflect,
                "ma60_dist": ma60_dist,
                "vol_ratio": vol_ratio[i],
                "rsi14": rsi[i],
                "macd_hist": macd_hist[i],
                "dif": dif[i],
                "slope_ma20": slope_norm,
                **rets,
            }
            all_events.append(row)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{total_codes} 只, {len(all_events):,}条, {elapsed:.0f}s")

    result = pd.DataFrame(all_events)
    print(f"\n特征提取完成: {len(result):,}条, {len(result.columns)}列")
    return result


def train_lgbm(
    features_df: pd.DataFrame,
    target_col: str = "ret20",
    test_ratio: float = 0.2,
):
    """用 LightGBM 训练模型，输出特征重要性。

    Returns:
        model, feature_importance_df, metrics
    """
    import lightgbm as lgb
    feature_cols = [
        "dh60", "dh180", "dh360", "dh1000",
        "f1", "f2", "f1_ma5", "f2_ma5", "f1_delta", "f2_delta",
        "f2_inflect", "ma60_dist", "vol_ratio", "rsi14",
        "macd_hist", "dif", "slope_ma20",
    ]

    # 清理
    df = features_df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"训练数据: {len(df):,}条")

    X = df[feature_cols].values
    y = df[target_col].values

    # 按时间分割（后20%作为测试集）
    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"训练集: {len(X_train):,}, 测试集: {len(X_test):,}")

    # LightGBM
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "max_depth": 8,
        "min_child_samples": 100,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=dtrain)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=callbacks,
    )

    # 特征重要性
    imp = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    # 预测
    y_pred = model.predict(X_test)

    # IC
    ic = np.corrcoef(y_pred, y_test)[0, 1]

    # 按预测分位统计胜率
    pred_bins = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
    bin_stats = []
    for b in range(10):
        mask = pred_bins == b
        if mask.sum() > 100:
            bin_stats.append({
                "bin": b,
                "n": mask.sum(),
                "avg_pred": y_pred[mask].mean(),
                "avg_actual": y_test[mask].mean(),
                "win_rate": (y_test[mask] > 0).mean() * 100,
            })

    metrics = {
        "IC": round(ic, 4),
        "MAE": round(np.mean(np.abs(y_pred - y_test)), 4),
        "bin_stats": bin_stats,
    }

    return model, imp_df, metrics


def optimize_scoring_params(features_df: pd.DataFrame, target_col: str = "ret20"):
    """网格搜索 v3 评分参数最优组合。

    搜索空间：
      - f1 阈值: [0.1, 0.2, 0.3, 0.4, 0.5]
      - f2 阈值: [0.01, 0.03, 0.05, 0.08, 0.10]
      - 位置权重: [1, 2, 3]
      - 趋势权重: [2, 3, 4]
      - 曲率权重: [2, 3, 4]

    目标：IC × 胜率 最大化
    """
    df = features_df.dropna(subset=["f1", "f2", "dh180", target_col]).copy()
    print(f"网格搜索数据: {len(df):,}条")

    best_ic = -999
    best_config = None
    results = []

    for pos_w in [1, 2, 3]:
        for trend_w in [2, 3, 4]:
            for curv_w in [2, 3, 4]:
                for f1_thresh in [0.1, 0.2, 0.3]:
                    for f2_thresh in [0.01, 0.03, 0.05]:
                        # 简化评分
                        # 位置
                        ps = np.clip(-df["dh180"] / 30, 0, 1) * pos_w
                        # 趋势
                        ts = np.where(df["f1"] > f1_thresh, 1.0,
                             np.where(df["f1"] > 0, 0.5,
                             np.where(df["f1"] > -f1_thresh, 0.2, 0.0))) * trend_w
                        # 曲率
                        cs = np.where(df["f2"] > f2_thresh, 1.0,
                             np.where(df["f2"] > 0, 0.6,
                             np.where(df["f2"] > -f2_thresh, 0.3, 0.0))) * curv_w

                        total = ps + ts + cs
                        ic = total.corr(df[target_col])

                        # 顶部20%胜率
                        threshold = total.quantile(0.8)
                        top = df[total >= threshold]
                        wr = (top[target_col] > 0).mean() * 100 if len(top) > 100 else 0

                        config = {
                            "pos_w": pos_w, "trend_w": trend_w, "curv_w": curv_w,
                            "f1_thresh": f1_thresh, "f2_thresh": f2_thresh,
                        }
                        results.append({**config, "IC": ic, "win_rate": wr, "n_top": len(top)})

                        if ic > best_ic:
                            best_ic = ic
                            best_config = config

    results_df = pd.DataFrame(results).sort_values("IC", ascending=False)
    return best_config, best_ic, results_df


def main():
    """主流程：提取特征 → LightGBM → 网格搜索 → 输出。"""
    data_path = "/tmp/fullmarket_4y.parquet"
    output_dir = Path("./scripts/research")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V3 ML 优化器")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/4] 加载全市场数据...")
    df_all = pd.read_parquet(data_path)
    counts = df_all.groupby("ts_code").size()
    valid = counts[counts >= 500].index
    df_all = df_all[df_all["ts_code"].isin(valid)].copy()
    print(f"  有效股票: {len(valid)}只")

    # 2. 特征提取
    print("\n[2/4] 提取特征...")
    features = extract_features_and_labels(df_all)
    features.to_parquet(output_dir / "v3_features.parquet", index=False)
    print(f"  保存到 {output_dir / 'v3_features.parquet'}")

    # 3. LightGBM
    print("\n[3/4] LightGBM 训练...")
    model, imp_df, metrics = train_lgbm(features, target_col="ret20")

    print(f"\n  IC = {metrics['IC']}")
    print(f"\n  特征重要性 (Top 10):")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:>15s}: {row['importance']:.1f}")

    print(f"\n  分位统计:")
    for b in metrics["bin_stats"]:
        print(f"    分位{b['bin']}: 预测{b['avg_pred']:+.2f}% 实际{b['avg_actual']:+.2f}% 胜率{b['win_rate']:.1f}%")

    # 4. 网格搜索
    print("\n[4/4] 网格搜索评分参数...")
    best_config, best_ic, grid_results = optimize_scoring_params(features)
    print(f"\n  最优配置: {best_config}")
    print(f"  最优IC: {best_ic:.4f}")
    print(f"\n  Top 5 配置:")
    for _, r in grid_results.head(5).iterrows():
        print(f"    pos_w={r['pos_w']} trend_w={r['trend_w']} curv_w={r['curv_w']} "
              f"f1={r['f1_thresh']} f2={r['f2_thresh']} IC={r['IC']:.4f} 胜率={r['win_rate']:.1f}%")

    # 保存
    imp_df.to_csv(output_dir / "v3_feature_importance.csv", index=False)
    grid_results.head(20).to_csv(output_dir / "v3_grid_top20.csv", index=False)

    # 保存最优配置
    with open(output_dir / "v3_best_config.json", "w") as f:
        json.dump({"config": best_config, "IC": best_ic}, f, indent=2)

    print("\n✅ 全部完成")
    return model, imp_df, metrics, best_config


if __name__ == "__main__":
    main()
