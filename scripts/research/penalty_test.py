#!/usr/bin/env python3
"""惩罚函数回测 — 方案一验证。

对比：
  1. 原始 v3_score（无惩罚）
  2. v3_score * penalty（Sigmoid + CI衰减）

验证指标：
  - IC（总分 vs ret20 的相关系数）
  - 分数段胜率（v3>=5, >=6, >=7, >=8 的 20日胜率）
  - 暴涨股处理（ret20_0 > 30% 的样本，惩罚前后胜率对比）
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

DATA_PATH = "/tmp/fullmarket_4y.parquet"
SG_WIN, SG_POLY = 21, 3


# ── 特征提取 ──────────────────────────────────────────────────────────

def extract_features(df_all: pd.DataFrame, sample_step: int = 20):
    """提取 v3 评分特征 + 惩罚函数所需的额外特征。"""
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
        f1 = savgol_filter(ln_p, SG_WIN, SG_POLY, deriv=1) * 100
        f2 = savgol_filter(ln_p, SG_WIN, SG_POLY, deriv=2) * 10000

        # Volume
        has_vol = "vol" in g.columns or "volume" in g.columns
        if has_vol:
            vc = "vol" if "vol" in g.columns else "volume"
            vol = g[vc].values.astype(float)
        else:
            vol = np.ones(n)

        # MA
        ma60 = pd.Series(close).rolling(60).mean().values

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_hist = (dif - dea) * 2

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain_s = pd.Series(delta).clip(lower=0).rolling(14).mean().values
        loss_s = pd.Series(np.abs(delta)).where(
            pd.Series(delta) < 0, 0).rolling(14).mean().values
        rs = np.where(loss_s > 0, gain_s / loss_s, 100)
        rsi = 100 - 100 / (1 + rs)

        # Volume features for CI
        ln_vol = np.log(vol + 1)
        vf1 = savgol_filter(ln_vol, SG_WIN, SG_POLY, deriv=1) * 100

        margin = max(SG_WIN, 60)
        sample_range = range(margin, n - 40, sample_step)

        for i in sample_range:
            current = close[i]
            row = {"ts_code": code, "day_idx": i}

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
            row["rsi14"] = rsi[i]
            row["macd_hist"] = macd_hist[i]
            row["dif"] = dif[i]

            # Volume ratio
            vol5 = pd.Series(vol).rolling(5).mean().values
            vol20 = pd.Series(vol).rolling(20).mean().values
            row["vol_ratio"] = vol5[i] / vol20[i] if vol20[i] > 0 else 1.0

            # CI 需要的额外特征
            row["vf1"] = vf1[i]  # 量能一阶导
            if i >= 5:
                row["f2_delta_5"] = f2[i] - np.mean(f2[i-5:i])
            else:
                row["f2_delta_5"] = 0.0

            # 历史20日收益（建仓时点已实现的涨幅）
            if i >= 20:
                row["ret20_0"] = (current / close[i-20] - 1) * 100
            else:
                row["ret20_0"] = 0.0

            # 标签
            row["ret20"] = (close[min(i+20, n-1)] / current - 1) * 100

            all_rows.append(row)

        if (idx+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{total} 只, {len(all_rows):,}条, {elapsed:.0f}s")

    df = pd.DataFrame(all_rows)
    print(f"\n特征提取完成: {len(df):,}条")
    return df


# ── V3 评分计算（复现 v3_derivative_scorer 的逻辑）─────────────────

def compute_v3_score(row: dict) -> float:
    """从特征计算 v3 总分。"""
    # 位置分 (0-2)
    ps = 0.0
    configs = [
        ("dh180", 0.7, [(10, 0), (3, 0.1), (0, 0.3), (-10, 0.5), (-20, 0.6), (-999, 0.7)]),
        ("dh360", 0.6, [(7, 0), (0, 0.2), (-20, 0.3), (-30, 0.5), (-999, 0.6)]),
        ("dh1000", 0.5, [(5, 0), (0, 0.1), (-30, 0.2), (-999, 0.5)]),
        ("dh60", 0.2, [(5, 0), (0, 0.1), (-999, 0.2)]),
    ]
    for col, max_s, thresholds in configs:
        pct = row.get(col, 0)
        for threshold, s in thresholds:
            if pct >= threshold:
                ps += s
                break

    # 趋势分 (0-3)
    f1_val = row["f1"]
    if f1_val > 0.3:
        ds = 3.0
    elif f1_val > 0.1:
        ds = 2.0
    elif f1_val > 0:
        ds = 1.0
    elif f1_val > -0.1:
        ds = 0.5
    else:
        ds = 0.0

    # 曲率分 (0-3) = 当前值(0-2) + 拐点(0-1)
    f2_val = row["f2"]
    if f2_val > 0.05:
        cur_score = 2.0
    elif f2_val > 0.01:
        cur_score = 1.5
    elif f2_val > -0.01:
        cur_score = 1.0
    elif f2_val > -0.05:
        cur_score = 0.5
    else:
        cur_score = 0.0

    inflect = row.get("f2_inflect", 0)
    cs = min(cur_score + inflect, 3.0)

    return round(ps + ds + cs, 1)  # 不含分钟分(0-2)，最高8分


# ── 涨势延续性指数 CI ──────────────────────────────────────────────

def compute_ci(row: dict) -> float:
    """计算涨势延续性指数 (Continuation Index)。

    维度：
      1. f''趋势 (0-1): 加速度是否在收敛
      2. 量能配合 (0-1): 价量关系
      3. 位置空间 (0-1): 距高点距离
    权重: 0.35 / 0.35 / 0.30
    """
    # 维度1: f''趋势
    f2_val = row["f2"]
    f2_delta = row.get("f2_delta_5", 0.0)
    if f2_delta > 0.01:
        ci_f2 = 1.0   # 加速度扩大
    elif f2_delta > -0.005:
        ci_f2 = 0.8   # 平稳
    elif f2_val > 0.01:
        ci_f2 = 0.5   # 还在加速但减弱
    else:
        ci_f2 = 0.2   # 加速度转负

    # 维度2: 量能配合
    f1_val = row["f1"]
    vf1_val = row.get("vf1", 0.0)
    vol_ratio = row.get("vol_ratio", 1.0)

    if f1_val > 0:
        if vf1_val > 0.01:
            ci_vol = 1.0    # 量价齐升
        elif vf1_val > -0.01:
            ci_vol = 0.9    # 缩量上涨（A股特征，筹码锁定）
        elif vol_ratio > 0.7:
            ci_vol = 0.6    # 价涨量平
        else:
            ci_vol = 0.3    # 量能衰竭
    else:
        ci_vol = 0.5  # 下跌中，量能维度中性

    # 维度3: 位置空间
    dh180 = row.get("dh180", 0.0)
    if dh180 < -40:
        ci_pos = 1.0
    elif dh180 < -20:
        ci_pos = 0.7
    elif dh180 < -10:
        ci_pos = 0.5
    elif dh180 < 0:
        ci_pos = 0.3
    else:
        ci_pos = 0.2

    ci = 0.35 * ci_f2 + 0.35 * ci_vol + 0.30 * ci_pos
    return round(ci, 3)


def compute_penalty(ret20_0: float, ci: float) -> float:
    """Sigmoid 平滑衰减惩罚。

    penalty = 1 / (1 + exp(5 * (ret20_0/30 - CI)))

    ret20_0=0%:   penalty≈1.0
    ret20_0=30% + CI高(0.8): penalty≈0.73
    ret20_0=30% + CI低(0.3): penalty≈0.27
    ret20_0=60% + CI高(0.8): penalty≈0.27
    ret20_0=60% + CI低(0.3): penalty≈0.047
    """
    x = ret20_0 / 30.0 - ci
    penalty = 1.0 / (1.0 + np.exp(5.0 * x))
    return round(penalty, 4)


# ── 回测 ──────────────────────────────────────────────────────────

def evaluate(scores: np.ndarray, ret20: np.ndarray, label: str) -> dict:
    """评估分数预测能力。"""
    mask = ~(np.isnan(scores) | np.isnan(ret20))
    s, r = scores[mask], ret20[mask]
    if len(s) < 100:
        return {"label": label, "n": len(s)}

    ic = np.corrcoef(s, r)[0, 1]

    result = {
        "label": label,
        "n": len(s),
        "IC": round(ic, 4),
    }

    # 分段胜率
    for threshold in [5, 6, 7, 8]:
        top = s >= threshold
        if top.sum() >= 50:
            wr = (r[top] > 0).mean() * 100
            avg_ret = r[top].mean()
            result[f"v3>={threshold}_wr"] = round(wr, 1)
            result[f"v3>={threshold}_avg"] = round(avg_ret, 2)
            result[f"v3>={threshold}_n"] = int(top.sum())
        else:
            result[f"v3>={threshold}_wr"] = None

    # 暴涨股分析
    surge = r  # 注意：这里ret20是标签（未来收益），我们看的是ret20_0（历史涨幅）
    # 下面单独处理

    return result


def run_backtest(df: pd.DataFrame):
    """跑回测。"""
    print("\n" + "="*70)
    print("计算 v3 评分...")
    t0 = time.time()

    df["v3_raw"] = df.apply(compute_v3_score, axis=1)
    print(f"  耗时: {time.time()-t0:.1f}s")

    print("\n计算 CI + 惩罚...")
    t0 = time.time()
    df["ci"] = df.apply(compute_ci, axis=1)
    df["penalty"] = df.apply(
        lambda r: compute_penalty(r["ret20_0"], r["ci"]), axis=1
    )
    df["v3_penalized"] = (df["v3_raw"] * df["penalty"]).round(2)
    print(f"  耗时: {time.time()-t0:.1f}s")

    ret20 = df["ret20"].values

    # 原始 v3
    print("\n" + "="*70)
    print("【原始 v3 评分】")
    res_raw = evaluate(df["v3_raw"].values, ret20, "v3_raw")
    print_summary(res_raw)

    # 惩罚后 v3
    print("\n" + "="*70)
    print("【惩罚后 v3 评分】")
    res_pen = evaluate(df["v3_penalized"].values, ret20, "v3_penalized")
    print_summary(res_pen)

    # 暴涨股对比
    print("\n" + "="*70)
    print("【暴涨股分析 (ret20_0 > 30%)】")
    surge_mask = df["ret20_0"] > 30
    if surge_mask.sum() >= 50:
        surge = df[surge_mask]
        s_ret = surge["ret20"].values
        s_v3_raw = surge["v3_raw"].values
        s_v3_pen = surge["v3_penalized"].values

        print(f"  暴涨样本数: {surge_mask.sum():,}")
        print(f"  暴涨后20日胜率(原始): {(s_ret > 0).mean()*100:.1f}%")
        print(f"  暴涨后20日均收益(原始): {s_ret.mean():.2f}%")
        print(f"  原始v3均值: {s_v3_raw.mean():.2f}")
        print(f"  惩罚后v3均值: {s_v3_pen.mean():.2f}")
        print(f"  平均惩罚系数: {surge['penalty'].mean():.3f}")

        # CI 高的暴涨股
        ci_high = surge[surge["ci"] > 0.6]
        ci_low = surge[surge["ci"] <= 0.4]
        if len(ci_high) >= 20:
            print(f"\n  CI高(>0.6)暴涨股: {len(ci_high)}只")
            print(f"    20日胜率: {(ci_high['ret20'].values > 0).mean()*100:.1f}%")
            print(f"    均收益: {ci_high['ret20'].values.mean():.2f}%")
            print(f"    平均惩罚: {ci_high['penalty'].mean():.3f}")
        if len(ci_low) >= 20:
            print(f"\n  CI低(<=0.4)暴涨股: {len(ci_low)}只")
            print(f"    20日胜率: {(ci_low['ret20'].values > 0).mean()*100:.1f}%")
            print(f"    均收益: {ci_low['ret20'].values.mean():.2f}%")
            print(f"    平均惩罚: {ci_low['penalty'].mean():.3f}")
    else:
        print(f"  暴涨样本不足: {surge_mask.sum()}")

    # Walk-Forward IC
    print("\n" + "="*70)
    print("【Walk-Forward IC 对比（按月）】")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    months = sorted(df["month"].unique())
    print(f"  {'月份':>8s}  {'IC_raw':>7s}  {'IC_pen':>7s}  {'改善':>6s}")

    ic_raw_list = []
    ic_pen_list = []
    for m in months[-12:]:  # 最近12个月
        sub = df[df["month"] == m]
        if len(sub) < 100:
            continue
        ic_raw = np.corrcoef(sub["v3_raw"].values, sub["ret20"].values)[0, 1]
        ic_pen = np.corrcoef(sub["v3_penalized"].values, sub["ret20"].values)[0, 1]
        diff = ic_pen - ic_raw
        print(f"  {str(m):>8s}  {ic_raw:>+7.4f}  {ic_pen:>+7.4f}  {diff:>+6.4f}")
        ic_raw_list.append(ic_raw)
        ic_pen_list.append(ic_pen)

    if ic_raw_list:
        print(f"  {'均值':>8s}  {np.mean(ic_raw_list):>+7.4f}  {np.mean(ic_pen_list):>+7.4f}  "
              f"{np.mean(ic_pen_list)-np.mean(ic_raw_list):>+6.4f}")

    # 惩罚分布
    print("\n" + "="*70)
    print("【惩罚系数分布】")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (df["penalty"] >= lo) & (df["penalty"] < hi)
        if mask.sum() > 0:
            sub = df[mask]
            wr = (sub["ret20"].values > 0).mean() * 100
            avg_ret = sub["ret20"].values.mean()
            print(f"  penalty [{lo:.1f}, {hi:.1f}): n={mask.sum():>8,}  "
                  f"胜率={wr:.1f}%  均收益={avg_ret:+.2f}%  "
                  f"v3_raw={sub['v3_raw'].mean():.2f}  v3_pen={sub['v3_penalized'].mean():.2f}")

    return df


def print_summary(res: dict):
    """打印评估摘要。"""
    print(f"  样本数: {res['n']:,}")
    print(f"  IC: {res['IC']:+.4f}")
    for t in [5, 6, 7, 8]:
        wr_key = f"v3>={t}_wr"
        if res.get(wr_key) is not None:
            avg_key = f"v3>={t}_avg"
            n_key = f"v3>={t}_n"
            print(f"  v3>={t}: 胜率={res[wr_key]:.1f}%  均收益={res[avg_key]:+.2f}%  n={res[n_key]:,}")


# ── 主入口 ──────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("V3 惩罚函数回测")
    print("="*70)

    # 加载数据
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {DATA_PATH}")
        print("请先运行数据准备步骤")
        return

    print(f"加载数据: {DATA_PATH}")
    t0 = time.time()
    df_all = pd.read_parquet(DATA_PATH)
    print(f"  原始数据: {len(df_all):,}条, {df_all['ts_code'].nunique()}只")
    print(f"  耗时: {time.time()-t0:.1f}s")

    # 特征提取
    print("\n提取特征（含CI所需特征）...")
    df = extract_features(df_all)

    # 回测
    result_df = run_backtest(df)

    # 保存结果
    out_path = "/tmp/penalty_test_result.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

    return result_df


if __name__ == "__main__":
    main()
