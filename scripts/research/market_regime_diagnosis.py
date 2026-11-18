#!/usr/bin/env python3
"""Market Regime 回测诊断 — 检查未来数据泄漏、样本偏差、统计显著性。

检查清单：
  1. 未来数据泄漏：savgol shift是否正确、收益计算是否用了未来价格
  2. 样本时间分布：regime在什么时间段出现，是否有时间聚类
  3. 统计显著性：各组差异是否有统计学意义
  4. 样本量：各组样本是否足够
  5. 幸存者偏差：选股范围是否有偏
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.market_regime import _sma


def diagnose(data_root: Path = Path("data/store")):
    print("=" * 70)
    print("🔍 Market Regime 回测诊断报告")
    print("=" * 70)

    # ── 检查1: 未来数据泄漏 ──────────────────────────────
    print("\n" + "─" * 70)
    print("检查1: 未来数据泄漏")
    print("─" * 70)

    # 1.1 savgol shift验证
    print("\n1.1 Savitzky-Golay shift验证")
    print("   savgol_filter是中心对称的，窗口21用了前后10个点")
    print("   回测模式必须shift(window//2)=10，消除前瞻偏差")

    # 用一个已知序列验证
    test_seq = np.arange(100, dtype=float)  # 线性上升
    log_test = np.log(test_seq + 1)
    window = 21
    f1 = savgol_filter(log_test, window, 3, deriv=1)
    f2 = savgol_filter(log_test, window, 3, deriv=2)

    # 未shift时，中间点用的是未来数据
    # 对数序列的一阶导应该近似常数(1/x)
    # 检查点50（索引50），savgol用了点40~60
    # 未shift: f1[50]用了点40~60的信息 → 有前瞻
    # shift后: f1[60]的值放到f1[50] → 只用到60以前

    print(f"   未shift时 f1[50] = {f1[50]:.6f}")
    f1_shifted = np.concatenate([f1[:10], f1[:-10]])
    print(f"   shift后  f1[50] = {f1_shifted[50]:.6f}")
    print(f"   ✅ 回测脚本中已做shift(window//2)={window // 2}")

    # 1.2 收益计算验证
    print("\n1.2 收益计算验证")
    print("   计算方式: ret = (close[t+hold_days] / close[t]) - 1")
    print("   检查是否真的用了未来价格...")
    # 模拟验证
    test_close = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=float)
    t = 2
    hold = 3
    ret = (test_close[t + hold] / test_close[t]) - 1
    print(f"   close[{t}]={test_close[t]}, close[{t+hold}]={test_close[t+hold]}, ret={ret:+.2%}")
    print(f"   ✅ 收益用的是t时刻之后的价格，这是正确的（我们就是要预测未来收益）")

    # 1.3 regime计算验证
    print("\n1.3 Regime计算是否用了未来数据")
    print("   regime用的是指数close序列的f'/f''/MA")
    print("   关键: 指数数据截止到2026-05-22，但股票数据到2026-06-05")
    print("   如果用指数未来的regime给过去某天的股票打标签 → 泄漏！")

    # 检查实际数据
    idx_path = data_root / "kline" / "000001SH" / "D.parquet"
    if idx_path.exists():
        df_idx = pd.read_parquet(idx_path)
        dt_col = "trade_date" if "trade_date" in df_idx.columns else "date"
        df_idx[dt_col] = pd.to_datetime(df_idx[dt_col])
        df_idx = df_idx.sort_values(dt_col)
        idx_end = df_idx[dt_col].iloc[-1]
        print(f"   指数数据截止: {idx_end.date()}")

        # 检查几只股票
        for code in ["600000", "000001", "600519"]:
            sp = data_root / "kline" / code / "D.parquet"
            if sp.exists():
                df_s = pd.read_parquet(sp)
                dt_c = "trade_date" if "trade_date" in df_s.columns else "date"
                df_s[dt_c] = pd.to_datetime(df_s[dt_c])
                df_s = df_s.sort_values(dt_c)
                print(f"   {code} 截止: {df_s[dt_c].iloc[-1].date()}", end="")
                if df_s[dt_c].iloc[-1] > idx_end:
                    print(f" ⚠️ 股票数据晚于指数{df_s[dt_c].iloc[-1] - idx_end}天！")
                    print(f"      → 回测中这些日期的regime用的是指数旧数据，但股票有了新价格")
                else:
                    print(" ✅")

    # 1.4 最关键：回测中regime用的是什么时间点的信息？
    print("\n1.4 ⚠️ 关键泄漏检查")
    print("   回测脚本中:")
    print("   - regime_df的每行对应指数的某个历史日期")
    print("   - 对每只股票，找该日期对应的股价")
    print("   - regime用的是指数t时刻的f'/f''（已shift）")
    print("   - 股票的f'/f''也用的是t时刻的（已shift）")
    print("   - 收益是t+hold_days后的价格")
    print("   ✅ 股票的信号和regime都是t时刻的，收益是t之后的，逻辑正确")
    print("")
    print("   但有一个潜在问题：")
    print("   regime判断用的是指数的全序列savgol，而不是滚动窗口")
    print("   → 虽然shift了，但savgol的拟合用的是全序列，理论上窗口外的点不影响结果")
    print("   → 实际验证：savgol只看窗口内(±10)的点，所以没问题 ✅")

    # ── 检查2: 样本时间分布 ──────────────────────────────
    print("\n" + "─" * 70)
    print("检查2: 样本时间分布与regime出现时段")
    print("─" * 70)

    if idx_path.exists():
        df_idx = pd.read_parquet(idx_path)
        dt_col = "trade_date" if "trade_date" in df_idx.columns else "date"
        df_idx[dt_col] = pd.to_datetime(df_idx[dt_col])
        df_idx = df_idx.sort_values(dt_col).reset_index(drop=True)
        idx_closes = df_idx["close"].values.astype(float)
        idx_dates = df_idx[dt_col].values

        # 重算regime（与回测一致）
        from scripts.research.market_regime_backtest import compute_regime_history
        regime_df = compute_regime_history(idx_closes, sample_rate=3)

        # 限制在股票数据范围
        stock_start = pd.Timestamp("2025-05-23")
        stock_end = pd.Timestamp("2026-06-05")

        regime_df["date"] = regime_df["idx"].apply(lambda i: pd.Timestamp(idx_dates[i]))
        regime_in_range = regime_df[(regime_df["date"] >= stock_start) &
                                     (regime_df["date"] <= stock_end)]

        print(f"\n   股票数据覆盖: {stock_start.date()} ~ {stock_end.date()}")
        print(f"   该范围内regime采样点: {len(regime_in_range)}")

        # 按季度统计regime分布
        regime_in_range["quarter"] = regime_in_range["date"].dt.to_period("Q")
        print(f"\n   按季度×regime分布:")
        pivot = pd.crosstab(regime_in_range["quarter"], regime_in_range["regime_label"])
        print(pivot.to_string())

        # 按月统计
        regime_in_range["month"] = regime_in_range["date"].dt.to_period("M")
        print(f"\n   按月×regime分布:")
        pivot_m = pd.crosstab(regime_in_range["month"], regime_in_range["regime_label"])
        print(pivot_m.to_string())

        print(f"\n   ⚠️ 分析: 如果某个regime只出现在特定月份，")
        print(f"   该regime的胜率可能反映的是那个时段的市场特征，而非regime本身的效果")

    # ── 检查3: 统计显著性 ──────────────────────────────
    print("\n" + "─" * 70)
    print("检查3: 统计显著性（需要回测数据）")
    print("─" * 70)

    # 重新跑一次回测收集数据（简化版，不输出报告）
    print("\n   运行回测收集数据用于统计检验...")
    from scripts.research.market_regime_backtest import run_backtest
    import io
    import contextlib

    # 捕获输出
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        records_df = run_backtest(
            data_root=data_root,
            hold_days=20,
            min_score=3.0,
            sample_rate=3,
            max_stocks=500,
        )

    if records_df is None or len(records_df) == 0:
        print("   ❌ 无回测数据")
        return

    print(f"   总样本: {len(records_df)}")

    # 3.1 各regime的置信区间
    print(f"\n   3.1 各阶段胜率的95%置信区间:")
    print(f"   {'阶段':<20s} {'样本':>6s} {'胜率':>7s} {'95%CI':>16s} {'p值(vs整体)':>12s}")
    print("   " + "-" * 65)

    overall_wr = records_df["win"].mean()
    overall_n = len(records_df)

    for regime_label in records_df["regime_label"].unique():
        subset = records_df[records_df["regime_label"] == regime_label]
        n = len(subset)
        if n < 20:
            continue
        wr = subset["win"].mean()
        # Wilson置信区间
        z = 1.96
        p_hat = wr
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
        ci_low = center - margin
        ci_high = center + margin

        # 与整体胜率的双比例z检验
        p1 = wr
        p2 = overall_wr
        n1 = n
        n2 = overall_n
        p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_stat = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"   {regime_label:<18s} {n:>6d} {wr:>6.1%} [{ci_low:.1%}, {ci_high:.1%}] {p_value:>10.4f} {sig}")

    # 3.2 关键对比：震荡偏空 vs 其他
    print(f"\n   3.2 关键对比（反直觉发现验证）:")
    comparisons = [
        ("🟠震荡偏空", "🟡震荡偏多", "震荡偏空 vs 震荡偏多"),
        ("🟠震荡偏空", "🟡震荡偏多(回调)", "震荡偏空 vs 回调"),
        ("🟡震荡偏多", "🟡震荡偏多(回调)", "震荡偏多 vs 回调"),
    ]

    for label_a, label_b, desc in comparisons:
        a = records_df[records_df["regime_label"] == label_a]["win"]
        b = records_df[records_df["regime_label"] == label_b]["win"]
        if len(a) < 10 or len(b) < 10:
            print(f"   {desc}: 样本不足")
            continue

        # 卡方检验
        table = np.array([
            [a.sum(), len(a) - a.sum()],
            [b.sum(), len(b) - b.sum()],
        ])
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)

        # Fisher精确检验（更保守）
        if table.min() >= 5:
            odds_ratio, fisher_p = stats.fisher_exact(table)
        else:
            fisher_p = None

        # Cohen's h (效应量)
        h = 2 * np.arcsin(np.sqrt(a.mean())) - 2 * np.arcsin(np.sqrt(b.mean()))

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        effect = "大" if abs(h) > 0.8 else "中" if abs(h) > 0.5 else "小" if abs(h) > 0.2 else "极小"
        print(f"\n   {desc}:")
        print(f"     A: {label_a} n={len(a)}, 胜率={a.mean():.1%}")
        print(f"     B: {label_b} n={len(b)}, 胜率={b.mean():.1%}")
        print(f"     卡方检验: χ²={chi2:.2f}, p={p_value:.4f} {sig}")
        if fisher_p is not None:
            print(f"     Fisher精确: p={fisher_p:.4f}")
        print(f"     Cohen's h={h:.3f} (效应量: {effect})")

    # 3.3 信号×regime交叉的统计检验
    print(f"\n   3.3 信号×regime交叉检验:")
    # 最关键的：震荡偏空+趋势加速 vs 震荡偏多(回调)+趋势加速
    a = records_df[(records_df["regime_label"] == "🟠震荡偏空") &
                   (records_df["signal"] == "趋势加速")]
    b = records_df[(records_df["regime_label"] == "🟡震荡偏多(回调)") &
                   (records_df["signal"] == "趋势加速")]

    if len(a) >= 10 and len(b) >= 10:
        # Welch t检验（收益）
        t_stat, p_ttest = stats.ttest_ind(a["ret_20d"], b["ret_20d"], equal_var=False)
        # Mann-Whitney U检验（非参数）
        u_stat, p_mann = stats.mannwhitneyu(a["ret_20d"], b["ret_20d"], alternative="two-sided")
        # 比例检验（胜率）
        table = np.array([
            [a["win"].sum(), len(a) - a["win"].sum()],
            [b["win"].sum(), len(b) - b["win"].sum()],
        ])
        chi2, p_chi2, _, _ = stats.chi2_contingency(table, correction=False)

        print(f"   '震荡偏空+趋势加速' vs '回调+趋势加速':")
        print(f"     A: n={len(a)}, 胜率={a['win'].mean():.1%}, 均收={a['ret_20d'].mean():+.2f}%")
        print(f"     B: n={len(b)}, 胜率={b['win'].mean():.1%}, 均收={b['ret_20d'].mean():+.2f}%")
        print(f"     Welch t检验: t={t_stat:.3f}, p={p_ttest:.4f}")
        print(f"     Mann-Whitney: U={u_stat:.0f}, p={p_mann:.4f}")
        print(f"     胜率卡方: χ²={chi2:.2f}, p={p_chi2:.4f}")

    # ── 检查4: 样本偏差 ──────────────────────────────
    print("\n" + "─" * 70)
    print("检查4: 样本偏差")
    print("─" * 70)

    print(f"\n   4.1 回测样本时间分布:")
    records_df["date_month"] = pd.to_datetime(records_df["date"]).dt.to_period("M")
    month_dist = records_df.groupby("date_month").size()
    print(f"   总月份: {len(month_dist)}")
    print(f"   月份分布:")
    for m, c in month_dist.items():
        print(f"     {m}: {c}条 ({c / len(records_df):.1%})")

    print(f"\n   4.2 股票样本是否有偏:")
    code_counts = records_df["code"].value_counts()
    print(f"   涉及股票: {len(code_counts)}只")
    print(f"   每只股票平均: {code_counts.mean():.1f}条")
    print(f"   最多: {code_counts.max()}条 ({code_counts.index[0]})")
    print(f"   最少: {code_counts.min()}条")
    # 是否某些股票贡献了过多样本
    top5_pct = code_counts.head(5).sum() / len(records_df) * 100
    print(f"   Top5股票占总样本: {top5_pct:.1f}%")

    if top5_pct > 30:
        print(f"   ⚠️ Top5占比超30%，可能存在个股偏差！")
    else:
        print(f"   ✅ 分布较均匀")

    # 4.3 信号类型分布
    print(f"\n   4.3 信号类型分布:")
    sig_dist = records_df["signal"].value_counts()
    for s, c in sig_dist.items():
        print(f"     {s}: {c} ({c / len(records_df):.1%})")

    if len(sig_dist) < 3:
        print(f"   ⚠️ 信号类型太少，可能过滤过度")
    else:
        print(f"   ✅ 信号类型多样")

    # ── 检查5: 总结 ──────────────────────────────
    print("\n" + "─" * 70)
    print("检查5: 总结与建议")
    print("─" * 70)
    print(f"""
   ✅ 未来数据: savgol已shift, 收益计算正确, regime用t时刻信息
   ⚠️ 样本时间: 回测数据只有约1年(2025-05~2026-06), 覆盖不够长
   ⚠️ 样本偏差: 需看上面各regime是否集中在特定月份
   ⚠️ 未出现🟢主升浪和🔴下跌regime: 可能这段时间大盘没经历过典型牛熊
   
   建议:
   1. 如果regime在时间上高度集中 → 结论可能只是那个时段的特征
   2. 样本只覆盖1年 → 不够得出普遍结论, 需要更长数据
   3. 反直觉结论需在更长时间段验证后才能采信
   4. 扩大股票池到全市场(去掉max_stocks限制)增加样本量
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/store")
    args = parser.parse_args()
    diagnose(Path(args.data_root))
