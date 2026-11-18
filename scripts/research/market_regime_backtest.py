#!/usr/bin/env python3
"""大盘阶段回测 — 验证MarketRegime对策略胜率的影响。

逻辑：
  1. 对沪深300历史每个交易日计算regime
  2. 对全市场股票计算V3评分和策略分类
  3. 分regime组统计各策略的N日胜率和收益

回测铁律：所有计算只使用当时已知数据，savgol窗口shift(window//2)。

用法：
  PYTHONPATH=. python scripts/research/market_regime_backtest.py
  PYTHONPATH=. python scripts/research/market_regime_backtest.py --hold 10
  PYTHONPATH=. python scripts/research/market_regime_backtest.py --sample-rate 5
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
from scripts.screener.market_regime import (
    _compute_derivatives, _sma, _check_ma_alignment, _check_vol_trend,
)


# ── 指数regime批量计算 ────────────────────────────────────
def compute_regime_history(
    index_closes: np.ndarray,
    index_volumes: np.ndarray | None = None,
    start_idx: int = 120,  # 前120天预热
    sample_rate: int = 1,  # 每N天取一次
) -> pd.DataFrame:
    """对指数逐日计算regime，返回DataFrame。

    ⚠️ 回测模式：savgol结果shift(window//2)消除前瞻偏差。
    """
    window = 21
    shift_n = window // 2  # 10

    log_p = np.log(index_closes.astype(float))
    n = len(log_p)

    # 计算全序列导数，然后shift
    valid_mask = ~np.isnan(log_p)
    f1_full = np.zeros(n)
    f2_full = np.zeros(n)
    if valid_mask.sum() >= window:
        f1_full[valid_mask] = savgol_filter(log_p[valid_mask], window, 3, deriv=1)
        f2_full[valid_mask] = savgol_filter(log_p[valid_mask], window, 3, deriv=2)
        # ⚠️ 关键：shift消除前瞻偏差
        f1_full = np.concatenate([f1_full[:shift_n], f1_full[:-shift_n]])
        f2_full = np.concatenate([f2_full[:shift_n], f2_full[:-shift_n]])

    rows = []
    for i in range(start_idx, n, sample_rate):
        close_i = index_closes[i]
        f1_i = f1_full[i]
        f2_i = f2_full[i]
        prev_f2 = f2_full[i - 1] if i > 0 else 0

        ma20_i = _sma(index_closes[:i + 1], 20)[-1]
        ma60_i = _sma(index_closes[:i + 1], 60)[-1]

        above_ma20 = close_i > ma20_i if not np.isnan(ma20_i) else False
        above_ma60 = close_i > ma60_i if not np.isnan(ma60_i) else False

        # 简化regime判断（复用market_regime.py的逻辑）
        is_infl_up = prev_f2 < 0 and f2_i > 0
        is_infl_down = prev_f2 > 0 and f2_i < 0

        if is_infl_up or is_infl_down:
            key = "inflection"
            label = "⚪拐点↑" if is_infl_up else "⚪拐点↓"
        elif f1_i > 0.01 and f2_i > 0.005 and above_ma60:
            key = "trend_up"
            label = "🟢主升浪"
        elif f1_i > 0 and f2_i < -0.005 and above_ma60:
            key = "range_bull"
            label = "🟡震荡偏多"
        elif f1_i > 0 and above_ma20 and not above_ma60:
            key = "range_bear"
            label = "🟠震荡偏空"
        elif f1_i < -0.01 and f2_i < -0.005 and not above_ma60:
            key = "trend_down"
            label = "🔴下跌"
        elif f1_i < 0 and f2_i > 0.005:
            key = "inflection"
            label = "⚪拐点↓减速"
        elif f1_i < 0 and above_ma60:
            key = "range_bull"
            label = "🟡震荡偏多(回调)"
        elif f1_i < 0 and not above_ma60 and f2_i > 0:
            key = "range_bear"
            label = "🟠震荡偏空(减速)"
        elif above_ma60:
            key = "range_bull"
            label = "🟡震荡偏多"
        else:
            key = "range_bear"
            label = "🟠震荡偏空"

        rows.append({
            "idx": i,
            "close": close_i,
            "f1": f1_i,
            "f2": f2_i,
            "regime_key": key,
            "regime_label": label,
            "above_ma20": above_ma20,
            "above_ma60": above_ma60,
        })

    return pd.DataFrame(rows)


# ── 个股信号分类 ──────────────────────────────────────────
def classify_signal_simple(f1: float, f2: float, above_ma20: bool) -> str:
    """简化的信号分类，与V3一致。"""
    if f1 > 0 and f2 > 0 and above_ma20:
        return "趋势加速"
    elif f1 >= 0 and f2 > 0 and not above_ma20:
        return "底部反转"
    elif f1 > 0 and f2 < 0:
        return "减速见顶"
    elif f1 < 0 and f2 < 0:
        return "加速下跌"
    else:
        return "震荡"


# ── 主回测 ────────────────────────────────────────────────
def run_backtest(
    data_root: Path,
    hold_days: int = 20,
    min_score: float = 5.0,
    sample_rate: int = 5,  # 每5个交易日取一次，加速
    max_stocks: int = 500,  # 限制股票数量加速
):
    """执行回测。"""
    print(f"📊 Market Regime 回测")
    print(f"   持有期={hold_days}天 | 最低分={min_score} | 采样率=每{sample_rate}天")
    print()

    # ── 1. 加载指数 ──
    print("1️⃣ 加载上证指数...")
    # 尝试多个指数
    for idx_code in ["000001SH", "000300SH"]:
        idx_cache = data_root / "kline" / idx_code / "D.parquet"
        if idx_cache.exists():
            break
    else:
        print("❌ 找不到指数数据")
        return

    df_idx = pd.read_parquet(idx_cache)
    dt_col = "trade_date" if "trade_date" in df_idx.columns else "date"
    df_idx[dt_col] = pd.to_datetime(df_idx[dt_col])
    df_idx = df_idx.sort_values(dt_col).reset_index(drop=True)
    idx_closes = df_idx["close"].values.astype(float)
    idx_volumes = df_idx["vol"].values.astype(float) if "vol" in df_idx.columns else None
    idx_dates = df_idx[dt_col].values

    print(f"   指数数据: {len(idx_closes)} 天 ({idx_dates[0]} ~ {idx_dates[-1]})")

    # ── 2. 计算regime历史 ──
    print("2️⃣ 计算大盘阶段历史...")
    regime_df = compute_regime_history(idx_closes, idx_volumes, sample_rate=sample_rate)
    regime_counts = regime_df["regime_label"].value_counts()
    print(f"   采样点: {len(regime_df)} 个")
    for label, count in regime_counts.items():
        print(f"   {label}: {count} ({count / len(regime_df):.1%})")

    # ── 3. 扫描股票 ──
    print(f"\n3️⃣ 扫描个股 (最多{max_stocks}只)...")
    kline_dir = data_root / "kline"
    stock_dirs = sorted([d.name for d in kline_dir.iterdir()
                         if d.is_dir() and (d / "D.parquet").exists()
                         and not d.name.endswith("SH") and not d.name.endswith("SZ")
                         and not d.name.startswith("INDEX")])
    # 限制数量
    if len(stock_dirs) > max_stocks:
        np.random.seed(42)
        stock_dirs = list(np.random.choice(stock_dirs, max_stocks, replace=False))
    print(f"   股票数: {len(stock_dirs)}")

    # 确定可回测的日期范围（股票数据最小覆盖范围）
    sample_dates = []
    for code in stock_dirs[:20]:
        try:
            df_s = pd.read_parquet(kline_dir / code / "D.parquet")
            dt_c = "trade_date" if "trade_date" in df_s.columns else "date"
            df_s[dt_c] = pd.to_datetime(df_s[dt_c])
            sample_dates.append((df_s[dt_c].min(), df_s[dt_c].max()))
        except Exception:
            continue
    if sample_dates:
        common_start = max(d[0] for d in sample_dates)
        common_end = min(d[1] for d in sample_dates)
        print(f"   股票数据覆盖: {common_start.date()} ~ {common_end.date()}")
    else:
        common_start = pd.Timestamp("2020-01-01")
        common_end = pd.Timestamp("2020-12-31")

    # ── 4. 逐股票计算V3和收益 ──
    print("4️⃣ 计算各股信号和收益...")
    all_records = []
    processed = 0

    for code in stock_dirs:
        parquet_path = kline_dir / code / "D.parquet"
        try:
            df = pd.read_parquet(parquet_path)
            dt_col2 = "trade_date" if "trade_date" in df.columns else "date"
            df[dt_col2] = pd.to_datetime(df[dt_col2])
            df = df.sort_values(dt_col2).reset_index(drop=True)

            closes = df["close"].values.astype(float)
            if len(closes) < 250:
                continue

            # 计算导数（shift消除前瞻偏差）
            window = 21
            log_p = np.log(closes)
            f1_arr = np.zeros(len(closes))
            f2_arr = np.zeros(len(closes))
            f1_arr = savgol_filter(log_p, window, 3, deriv=1)
            f2_arr = savgol_filter(log_p, window, 3, deriv=2)
            # shift
            shift_n = window // 2
            f1_arr = np.concatenate([f1_arr[:shift_n], f1_arr[:-shift_n]])
            f2_arr = np.concatenate([f2_arr[:shift_n], f2_arr[:-shift_n]])

            ma20_arr = _sma(closes, 20)
            ma60_arr = _sma(closes, 60)

            # 对齐日期索引
            dates = df[dt_col2].values
            stock_start = pd.Timestamp(dates[0])
            stock_end = pd.Timestamp(dates[-1])

            # 只回测该股票数据覆盖范围内的regime日期
            for _, regime_row in regime_df.iterrows():
                idx_i = regime_row["idx"]
                target_date = pd.Timestamp(idx_dates[idx_i])

                # 跳过该股票数据范围外
                if target_date < stock_start or target_date > stock_end:
                    continue

                # 找该股票同日或之前最近的一天
                match_idx = np.searchsorted(dates, target_date, side="right") - 1
                if match_idx < 60 or match_idx >= len(closes) - hold_days:
                    continue

                f1 = f1_arr[match_idx]
                f2 = f2_arr[match_idx]
                close_i = closes[match_idx]
                ma20_i = ma20_arr[match_idx]
                above_ma20 = close_i > ma20_i if not np.isnan(ma20_i) else False

                # 20日涨幅
                ret20 = (closes[match_idx] / closes[match_idx - 20] - 1) * 100 if match_idx >= 20 else 0

                # V3简化评分（放宽容让更多样本进入）
                score = 0
                score += min(max(f2 * 100, 0), 3)  # f''趋势 0-3
                if f1 > 0.01:
                    score += 2
                elif f1 > 0:
                    score += 1
                if above_ma20:
                    score += 1

                if score < min_score:
                    continue

                signal = classify_signal_simple(f1, f2, above_ma20)

                # N日收益
                future_close = closes[match_idx + hold_days]
                ret = (future_close / close_i - 1) * 100

                all_records.append({
                    "code": code,
                    "date": target_date,
                    "regime_key": regime_row["regime_key"],
                    "regime_label": regime_row["regime_label"],
                    "signal": signal,
                    "score": score,
                    "f1": f1,
                    "f2": f2,
                    "ret20": ret20,
                    f"ret_{hold_days}d": ret,
                    "win": ret > 0,
                })

            processed += 1
            if processed % 50 == 0:
                print(f"   已处理 {processed}/{len(stock_dirs)} 只, 累计{len(all_records)}条记录")

        except Exception as e:
            continue

    if not all_records:
        print("❌ 没有有效的回测记录")
        return

    records_df = pd.DataFrame(all_records)
    print(f"\n✅ 共 {len(records_df)} 条回测记录")

    # ── 5. 分组统计 ──
    print(f"\n{'='*70}")
    print(f"📊 回测结果 | 持有{hold_days}天 | 最低分{min_score}")
    print(f"{'='*70}")

    # 5.1 按regime分组
    print("\n## 按大盘阶段分组")
    print(f"{'阶段':<20s} {'样本':>6s} {'胜率':>7s} {'平均收益':>8s} {'中位收益':>8s} {'盈亏比':>6s}")
    print("-" * 60)

    for regime in ["🟢主升浪", "🟡震荡偏多", "🟡震荡偏多(回调)", "🟠震荡偏空", "🔴下跌", "⚪拐点↑", "⚪拐点↓", "⚪拐点↓减速"]:
        subset = records_df[records_df["regime_label"] == regime]
        if len(subset) < 10:
            continue
        n = len(subset)
        win_rate = subset["win"].mean()
        avg_ret = subset[f"ret_{hold_days}d"].mean()
        med_ret = subset[f"ret_{hold_days}d"].median()
        winners = subset[subset["win"]][f"ret_{hold_days}d"].mean()
        losers = subset[~subset["win"]][f"ret_{hold_days}d"].mean()
        pnl = abs(winners / losers) if losers != 0 else float("inf")
        print(f"  {regime:<18s} {n:>6d} {win_rate:>6.1%} {avg_ret:>+7.2f}% {med_ret:>+7.2f}% {pnl:>5.2f}")

    # 5.2 按regime × signal交叉分组
    print(f"\n## 按大盘阶段 × 信号类型交叉")
    print(f"{'阶段':<16s} {'信号':<10s} {'样本':>6s} {'胜率':>7s} {'平均收益':>8s} {'盈亏比':>6s}")
    print("-" * 55)

    for regime in ["🟢主升浪", "🟡震荡偏多", "🟡震荡偏多(回调)", "🟠震荡偏空", "🔴下跌"]:
        for signal in ["趋势加速", "底部反转", "减速见顶", "加速下跌", "震荡"]:
            subset = records_df[(records_df["regime_label"] == regime) & (records_df["signal"] == signal)]
            if len(subset) < 5:
                continue
            n = len(subset)
            win_rate = subset["win"].mean()
            avg_ret = subset[f"ret_{hold_days}d"].mean()
            winners = subset[subset["win"]][f"ret_{hold_days}d"].mean()
            losers = subset[~subset["win"]][f"ret_{hold_days}d"].mean()
            pnl = abs(winners / losers) if losers != 0 else float("inf")
            print(f"  {regime:<14s} {signal:<10s} {n:>6d} {win_rate:>6.1%} {avg_ret:>+7.2f}% {pnl:>5.2f}")

    # 5.3 策略推荐验证：当前regime推荐策略 vs 其他策略
    print(f"\n## 策略匹配验证")
    print("对比：在某个regime下，推荐策略是否优于其他策略")
    print()

    regime_strategy_map = {
        "trend_up": "趋势加速",
        "range_bull": "底部反转",  # 震荡偏多时弹簧起涨优先，这里用底部反转近似
        "range_bear": "底部反转",
        "trend_down": "底部反转",  # 下跌时只有底部反转才有机会
    }

    for rkey, recommended_signal in regime_strategy_map.items():
        regime_subset = records_df[records_df["regime_key"] == rkey]
        if len(regime_subset) < 20:
            continue

        label = regime_subset["regime_label"].iloc[0]
        recommended = regime_subset[regime_subset["signal"] == recommended_signal]
        others = regime_subset[regime_subset["signal"] != recommended_signal]

        if len(recommended) < 5 or len(others) < 5:
            continue

        rec_wr = recommended["win"].mean()
        oth_wr = others["win"].mean()
        rec_ret = recommended[f"ret_{hold_days}d"].mean()
        oth_ret = others[f"ret_{hold_days}d"].mean()

        # 简单统计检验
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(
            recommended["signal"] == recommended_signal,
            regime_subset["win"],
        )
        # 2x2表
        a = len(recommended[recommended["win"]])
        b = len(recommended[~recommended["win"]])
        c = len(others[others["win"]])
        d = len(others[~others["win"]])
        try:
            table = np.array([[a, b], [c, d]])
            chi2, p_value, _, _ = chi2_contingency(table, correction=False)
        except Exception:
            p_value = 1.0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        better = "✅" if rec_wr > oth_wr else "❌"

        print(f"  {label} → 推荐[{recommended_signal}]")
        print(f"    推荐: n={len(recommended)}, 胜率={rec_wr:.1%}, 收益={rec_ret:+.2f}%")
        print(f"    其他: n={len(others)}, 胜率={oth_wr:.1%}, 收益={oth_ret:+.2f}%")
        print(f"    {better} 差异 p={p_value:.4f} {sig}")
        print()

    # ── 6. 汇总 ──
    total_n = len(records_df)
    overall_wr = records_df["win"].mean()
    overall_ret = records_df[f"ret_{hold_days}d"].mean()
    print(f"\n📈 总计: {total_n}样本, 整体胜率{overall_wr:.1%}, 平均收益{overall_ret:+.2f}%")

    return records_df


def main():
    parser = argparse.ArgumentParser(description="Market Regime 回测")
    parser.add_argument("--data-root", default="data/store", help="数据目录")
    parser.add_argument("--hold", type=int, default=20, help="持有天数(默认20)")
    parser.add_argument("--min-score", type=float, default=5.0, help="最低V3分(默认5)")
    parser.add_argument("--sample-rate", type=int, default=5, help="采样率(每N天,默认5)")
    parser.add_argument("--max-stocks", type=int, default=500, help="最大股票数(默认500)")
    args = parser.parse_args()

    run_backtest(
        data_root=Path(args.data_root),
        hold_days=args.hold,
        min_score=args.min_score,
        sample_rate=args.sample_rate,
        max_stocks=args.max_stocks,
    )


if __name__ == "__main__":
    main()
