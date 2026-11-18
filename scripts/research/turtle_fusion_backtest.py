#!/usr/bin/env python3
"""海龟融合回测 — 对比4种配置在全市场的表现。

4种配置：
  1. 纯V3评分（基线）：V3≥7入场，V3<5退出，否则持有到20天
  2. V3 + ATR仓位管理：同上 + 仓位=ATR计算（1%风险规则）
  3. V3 + ATR仓位 + 2ATR止损：同上 + 硬止损
  4. V3 + ATR仓位 + 止损 + 10/20日移动止盈：全套海龟

回测规则：
  - 回测周期：最近2年（2024-06-14到2026-06-14）
  - 股票池：沪深300+中证500+中证1000（data/stock_pool.csv）
  - 入场条件：V3评分≥7.0
  - 持有期：最长20个交易日
  - 每只股票独立回测，最终汇总

⚠️ 未来数据铁律：savgol_filter必须shift(window//2)，ATR只用历史数据

用法:
  PYTHONPATH=. python scripts/research/turtle_fusion_backtest.py
  PYTHONPATH=. python scripts/research/turtle_fusion_backtest.py --limit 20
  PYTHONPATH=. python scripts/research/turtle_fusion_backtest.py --sample 300

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import random
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 抑制警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 导入模块 ──────────────────────────────────────────────
from scripts.core.data_store import DataStore
from scripts.core.atr_calculator import ATRCalculator
from scripts.screener.v3_derivative_scorer import V3DerivativeScorer

# ── 回测参数 ──────────────────────────────────────────────
BACKTEST_START = "2024-06-14"
BACKTEST_END = "2026-06-14"
MAX_HOLD_DAYS = 20
V3_ENTRY_THRESHOLD = 5.0   # raw_total >= 5 入场（对应约 total >= 3.5）
V3_EXIT_THRESHOLD = 3.0     # raw_total < 3 退出
ATR_PERIOD = 20
ATR_RISK_PERCENT = 0.01  # 1% 风险规则
HARD_STOP_ATR_MULT = 2.0  # 2ATR 硬止损
TRAIL_LOOKBACK_SHORT = 10  # 10日低点移动止盈（短期）
TRAIL_LOOKBACK_LONG = 20   # 20日低点移动止盈（长期）
SG_WINDOW = 21
SG_POLY = 3
MIN_DATA_DAYS = 60  # 最少需要60根日K

OUTPUT_CSV = PROJECT_ROOT / "data" / "turtle_fusion_backtest_result.csv"
STOCK_POOL_CSV = PROJECT_ROOT / "data" / "stock_pool.csv"


# ── 工具函数 ──────────────────────────────────────────────

# 全局 scorer 实例（复用，避免重复初始化）
_SCORER = None


def get_scorer(data_root: Path = None) -> V3DerivativeScorer:
    """获取全局 V3DerivativeScorer 实例。"""
    global _SCORER
    if _SCORER is None:
        _SCORER = V3DerivativeScorer(data_root=data_root)
    return _SCORER


def compute_v3_series(close: np.ndarray, end_idx: int,
                      df_full: pd.DataFrame = None,
                      code: str = "",
                      sg_window: int = SG_WINDOW,
                      sg_poly: int = SG_POLY) -> tuple[float, float]:
    """计算某历史时点的V3评分（用截止到 end_idx 的数据）。

    使用 V3DerivativeScorer 完整评分（含分钟信号），
    严格只用 end_idx 及之前的数据，防止未来泄漏。

    Returns:
        (raw_total, penalized_total)
        raw_total: 惩罚前原始分（入场信号用这个）
        penalized_total: 惩罚后总分
    """
    if end_idx + 1 < MIN_DATA_DAYS:
        return 0.0, 0.0

    if df_full is None:
        return 0.0, 0.0

    df_slice = df_full.iloc[:end_idx + 1].copy()

    scorer = get_scorer()
    try:
        result = scorer.score_stock(code, df_slice)
        return result.raw_total, result.total
    except Exception:
        return 0.0, 0.0


def calc_atr_at(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                end_idx: int, period: int = ATR_PERIOD) -> float:
    """计算 end_idx 时点的 ATR（只用历史数据）。"""
    if end_idx < period:
        return 0.0

    h = high[:end_idx + 1]
    l = low[:end_idx + 1]
    c = close[:end_idx + 1]

    prev_close = pd.Series(c).shift(1)
    tr1 = pd.Series(h) - pd.Series(l)
    tr2 = (pd.Series(h) - prev_close).abs()
    tr3 = (pd.Series(l) - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    if len(tr) < period + 1:
        return 0.0

    # Wilder平滑
    atr_val = tr.iloc[1:period + 1].mean()
    for i in range(period + 1, len(tr)):
        if not np.isnan(atr_val):
            atr_val = (atr_val * (period - 1) + tr.iloc[i]) / period
        else:
            atr_val = tr.iloc[i]

    return float(atr_val) if not np.isnan(atr_val) else 0.0


# ── 单股回测 ──────────────────────────────────────────────

def backtest_single_stock(
    code: str,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """对单只股票做4种配置的回测。

    在回测期间扫描所有V3≥7的入场信号，每个信号独立计算4种配置的收益。

    Returns:
        trade records list, 每条包含 config 1-4 的收益
    """
    # 确保列名规范
    dt_col = "date" if "date" in df.columns else "trade_date"
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    # 确保有 high/low/close
    for col in ["high", "low", "close"]:
        if col not in df.columns:
            return []

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    dates = df[dt_col].values

    n = len(close)
    if n < MIN_DATA_DAYS + MAX_HOLD_DAYS:
        return []

    # 定位回测区间
    start_ts = np.datetime64(start_date)
    end_ts = np.datetime64(end_date)
    bt_start_idx = np.searchsorted(dates, start_ts, side="left")
    bt_end_idx = np.searchsorted(dates, end_ts, side="right")

    # 确保有足够前置数据计算指标
    bt_start_idx = max(bt_start_idx, MIN_DATA_DAYS + SG_WINDOW)

    if bt_start_idx >= bt_end_idx:
        return []

    trades = []

    # 遍历每个交易日，检查入场信号
    i = bt_start_idx
    while i < bt_end_idx - MAX_HOLD_DAYS:
        # 计算当日V3评分（用raw_total作为入场信号）
        v3_raw, v3_score = compute_v3_series(close, i, df, code)

        if v3_raw < V3_ENTRY_THRESHOLD:
            i += 1
            continue

        # ── 入场！ ──
        entry_idx = i
        entry_price = close[entry_idx]
        entry_date = dates[entry_idx]

        # 计算入场时 ATR
        atr = calc_atr_at(high, low, close, entry_idx, ATR_PERIOD)
        if atr <= 0:
            i += 1
            continue

        # ATR仓位（1%风险规则）：仓位 = 总资金×1% / (ATR)
        # 满仓=1.0，ATR仓位可能>1（加杠杆）或<1（轻仓）
        # 这里记录仓位比例，但收益按满仓计算（方便对比）
        position_ratio = min(ATR_RISK_PERCENT * entry_price / atr, 1.0)  # 不加杠杆，上限100%

        # 硬止损价
        hard_stop_price = entry_price - HARD_STOP_ATR_MULT * atr

        # 预计算持有期内每天的V3和ATR
        hold_end = min(entry_idx + MAX_HOLD_DAYS, bt_end_idx - 1)

        # 4种配置的退出价
        exit_prices = [0.0, 0.0, 0.0, 0.0]
        exit_indices = [hold_end, hold_end, hold_end, hold_end]
        exit_reasons = ["到期", "到期", "到期", "到期"]

        # 跟踪持有期最高价（用于移动止损）
        peak_idx = entry_idx

        # 逐日检查退出条件
        for j in range(entry_idx + 1, hold_end + 1):
            day_high = high[j]
            day_low = low[j]
            day_close = close[j]

            # 更新最高价
            if day_high > high[peak_idx]:
                peak_idx = j

            # 计算当日V3
            v3_raw_j, v3_j = compute_v3_series(close, j, df, code)

            # ── 配置1: V3<5退出 or 到期 ──
            if exit_indices[0] == hold_end and v3_raw_j < V3_EXIT_THRESHOLD:
                exit_indices[0] = j
                exit_prices[0] = day_close
                exit_reasons[0] = f"V3raw={v3_raw_j:.1f}<{V3_EXIT_THRESHOLD}"

            # ── 配置2: 同配置1（仓位管理影响绝对收益但不影响退出逻辑）──
            if exit_indices[1] == hold_end and v3_raw_j < V3_EXIT_THRESHOLD:
                exit_indices[1] = j
                exit_prices[1] = day_close
                exit_reasons[1] = f"V3raw={v3_raw_j:.1f}<{V3_EXIT_THRESHOLD}"

            # ── 配置3: V3<5退出 + 2ATR硬止损 ──
            if exit_indices[2] == hold_end:
                if v3_raw_j < V3_EXIT_THRESHOLD:
                    exit_indices[2] = j
                    exit_prices[2] = day_close
                    exit_reasons[2] = f"V3raw={v3_raw_j:.1f}<{V3_EXIT_THRESHOLD}"
                elif day_low <= hard_stop_price:
                    exit_indices[2] = j
                    exit_prices[2] = hard_stop_price  # 假设精确止损
                    exit_reasons[2] = f"2ATR止损@{hard_stop_price:.2f}"

            # ── 配置4: V3<5退出 + 2ATR硬止损 + 10/20日低点移动止盈 ──
            if exit_indices[3] == hold_end:
                if v3_raw_j < V3_EXIT_THRESHOLD:
                    exit_indices[3] = j
                    exit_prices[3] = day_close
                    exit_reasons[3] = f"V3raw={v3_raw_j:.1f}<{V3_EXIT_THRESHOLD}"
                elif day_low <= hard_stop_price:
                    exit_indices[3] = j
                    exit_prices[3] = hard_stop_price
                    exit_reasons[3] = f"2ATR止损@{hard_stop_price:.2f}"
                else:
                    # 10/20日低点移动止盈（持有期超过10天后启用）
                    days_held = j - entry_idx
                    if days_held >= 10:
                        # 用最近10日和20日最低价的较高者
                        lookback_short = min(TRAIL_LOOKBACK_SHORT, days_held)
                        lookback_long = min(TRAIL_LOOKBACK_LONG, days_held)
                        low_10 = np.min(low[j - lookback_short + 1: j + 1])
                        low_20 = np.min(low[j - lookback_long + 1: j + 1])
                        # 取较高者作为止盈线
                        trail_stop = max(low_10, low_20)
                        if day_low <= trail_stop and j > entry_idx + 5:
                            exit_indices[3] = j
                            exit_prices[3] = trail_stop
                            exit_reasons[3] = f"移动止盈@{trail_stop:.2f}(10/20日低点)"

        # 填充到期退出的价格
        for cfg in range(4):
            if exit_prices[cfg] == 0.0:
                exit_prices[cfg] = close[exit_indices[cfg]]

        # 计算每种配置的收益率
        for cfg in range(4):
            if cfg == 1 or cfg == 2 or cfg == 3:
                # 配置2-4有ATR仓位管理
                ret_pct = (exit_prices[cfg] / entry_price - 1) * 100
                # 仓位调整后的收益（不影响百分比对比，但记录仓位）
                trades.append({
                    "code": code,
                    "entry_date": str(entry_date)[:10],
                    "entry_price": round(entry_price, 2),
                    "exit_date": str(dates[exit_indices[cfg]])[:10],
                    "exit_price": round(exit_prices[cfg], 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": exit_indices[cfg] - entry_idx,
                    "exit_reason": exit_reasons[cfg],
                    "config": cfg + 1,
                    "config_name": ["纯V3", "V3+ATR仓位", "V3+ATR+止损", "V3+ATR+止损+止盈"][cfg],
                    "v3_entry": round(v3_raw, 2),
                    "v3_total": round(v3_score, 2),
                    "atr": round(atr, 4),
                    "position_ratio": round(position_ratio, 4) if cfg >= 1 else 1.0,
                    "hard_stop_price": round(hard_stop_price, 2) if cfg >= 2 else 0,
                })
            else:
                ret_pct = (exit_prices[cfg] / entry_price - 1) * 100
                trades.append({
                    "code": code,
                    "entry_date": str(entry_date)[:10],
                    "entry_price": round(entry_price, 2),
                    "exit_date": str(dates[exit_indices[cfg]])[:10],
                    "exit_price": round(exit_prices[cfg], 2),
                    "return_pct": round(ret_pct, 2),
                    "hold_days": exit_indices[cfg] - entry_idx,
                    "exit_reason": exit_reasons[cfg],
                    "config": cfg + 1,
                    "config_name": ["纯V3", "V3+ATR仓位", "V3+ATR+止损", "V3+ATR+止损+止盈"][cfg],
                    "v3_entry": round(v3_raw, 2),
                    "v3_total": round(v3_score, 2),
                    "atr": round(atr, 4),
                    "position_ratio": 1.0,
                    "hard_stop_price": 0,
                })

        # 跳过持有期，避免重叠信号
        i = exit_indices[0] + 1  # 用配置1的退出点作为跳过点

    return trades


# ── 汇统统计 ──────────────────────────────────────────────

def compute_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    """计算每种配置的统计指标。"""
    stats = []
    for cfg in sorted(trades_df["config"].unique()):
        cfg_df = trades_df[trades_df["config"] == cfg]
        name = cfg_df["config_name"].iloc[0]

        n = len(cfg_df)
        if n == 0:
            stats.append({
                "config": cfg,
                "name": name,
                "samples": 0,
                "win_rate": 0,
                "avg_return": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_loss": 0,
                "total_return": 0,
            })
            continue

        wins = cfg_df[cfg_df["return_pct"] > 0]
        losses = cfg_df[cfg_df["return_pct"] <= 0]

        win_rate = len(wins) / n * 100
        avg_return = cfg_df["return_pct"].mean()
        avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["return_pct"].mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        max_loss = cfg_df["return_pct"].min()
        total_return = cfg_df["return_pct"].sum()

        stats.append({
            "config": cfg,
            "name": name,
            "samples": n,
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_return, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_loss": round(max_loss, 2),
            "total_return": round(total_return, 1),
        })

    return pd.DataFrame(stats)


# ── 打印对比表 ────────────────────────────────────────────

def print_comparison(stats_df: pd.DataFrame):
    """打印漂亮的对比表格。"""
    print("\n" + "=" * 100)
    print("🏆 海龟融合回测 — 4种配置对比")
    print("=" * 100)
    print(f"回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"入场条件: V3评分≥{V3_ENTRY_THRESHOLD}")
    print(f"最大持有: {MAX_HOLD_DAYS}个交易日")
    print()

    # 表头
    header = f"{'配置':<28} {'样本':>6} {'胜率':>7} {'平均收益':>8} {'平均盈利':>8} {'平均亏损':>8} {'盈亏比':>7} {'最大亏损':>8} {'总收益':>8}"
    print(header)
    print("-" * 100)

    for _, row in stats_df.iterrows():
        line = (
            f"{row['name']:<28} "
            f"{row['samples']:>6} "
            f"{row['win_rate']:>6.1f}% "
            f"{row['avg_return']:>+7.2f}% "
            f"{row['avg_win']:>+7.2f}% "
            f"{row['avg_loss']:>+7.2f}% "
            f"{row['profit_factor']:>7.2f} "
            f"{row['max_loss']:>+7.2f}% "
            f"{row['total_return']:>+7.1f}% "
        )
        print(line)

    print("-" * 100)
    print()

    # 找出最优配置
    best_win = stats_df.loc[stats_df["win_rate"].idxmax()]
    best_ret = stats_df.loc[stats_df["avg_return"].idxmax()]
    best_pf = stats_df.loc[stats_df["profit_factor"].idxmax()]

    print("📊 结论:")
    print(f"  最高胜率: {best_win['name']} ({best_win['win_rate']:.1f}%)")
    print(f"  最高平均收益: {best_ret['name']} ({best_ret['avg_return']:+.2f}%)")
    print(f"  最高盈亏比: {best_pf['name']} ({best_pf['profit_factor']:.2f})")
    print("=" * 100)


# ── 主流程 ────────────────────────────────────────────────

def load_stock_pool(limit: int = 0, sample: int = 0) -> list[str]:
    """加载股票池。"""
    df = pd.read_csv(STOCK_POOL_CSV, dtype={"code": str})
    codes = df["code"].tolist()

    if sample > 0 and sample < len(codes):
        random.seed(42)
        codes = random.sample(codes, sample)
        print(f"🎲 随机采样 {sample} 只股票")

    if limit > 0:
        codes = codes[:limit]

    return codes


def main():
    parser = argparse.ArgumentParser(description="海龟融合回测")
    parser.add_argument("--limit", type=int, default=0, help="限制股票数量（测试用）")
    parser.add_argument("--sample", type=int, default=0, help="随机采样N只股票")
    parser.add_argument("--start", type=str, default=BACKTEST_START, help="回测起始日期")
    parser.add_argument("--end", type=str, default=BACKTEST_END, help="回测结束日期")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 海龟融合回测启动")
    print(f"   回测区间: {args.start} ~ {args.end}")
    print(f"   入场条件: V3raw≥{V3_ENTRY_THRESHOLD}, 退出: V3raw<{V3_EXIT_THRESHOLD}")
    print(f"   最大持有: {MAX_HOLD_DAYS}天, ATR周期: {ATR_PERIOD}")
    print(f"   硬止损: {HARD_STOP_ATR_MULT}×ATR")
    print(f"   移动止盈: {TRAIL_LOOKBACK_SHORT}/{TRAIL_LOOKBACK_LONG}日低点")
    print("=" * 80)

    # 加载股票池
    codes = load_stock_pool(limit=args.limit, sample=args.sample)
    print(f"\n📋 股票池: {len(codes)} 只")

    # 初始化DataStore（只读模式，不触发下载）
    store = DataStore(lazy=True)
    data_root = PROJECT_ROOT / "data" / "store"
    get_scorer(data_root=data_root)

    all_trades = []
    skipped = 0
    errors = 0
    t0 = time.time()

    for i, code in enumerate(codes):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(codes) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(codes)}] 已完成, 耗时{elapsed:.0f}s, ETA {eta:.0f}s, "
                  f"信号{len(all_trades)//4}个, 跳过{skipped}, 错误{errors}")

        try:
            df = store.get_kline(code, period="D", auto_update=False)
            if df is None or df.empty:
                skipped += 1
                continue

            # 确保列名规范
            dt_col = "date" if "date" in df.columns else "trade_date"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)

            if len(df) < MIN_DATA_DAYS + MAX_HOLD_DAYS:
                skipped += 1
                continue

            trades = backtest_single_stock(code, df, args.start, args.end)
            all_trades.extend(trades)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ❌ {code}: {e}")

    elapsed = time.time() - t0
    print(f"\n⏱️ 总耗时: {elapsed:.1f}s")
    print(f"   股票总数: {len(codes)}, 跳过: {skipped}, 错误: {errors}")
    print(f"   信号总数: {len(all_trades)//4} (×4配置 = {len(all_trades)}条记录)")

    if len(all_trades) == 0:
        print("\n⚠️ 没有产生任何交易信号！请检查数据或参数。")
        return

    # 汇总统计
    trades_df = pd.DataFrame(all_trades)
    stats_df = compute_stats(trades_df)

    # 打印对比表
    print_comparison(stats_df)

    # 保存结果
    trades_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n💾 交易明细已保存: {OUTPUT_CSV}")

    # 同时保存统计摘要
    stats_path = OUTPUT_CSV.parent / "turtle_fusion_backtest_stats.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"💾 统计摘要已保存: {stats_path}")

    # 打印退出原因分布
    print("\n📊 各配置退出原因分布:")
    for cfg in sorted(trades_df["config"].unique()):
        cfg_df = trades_df[trades_df["config"] == cfg]
        reasons = cfg_df["exit_reason"].value_counts().head(5)
        name = cfg_df["config_name"].iloc[0]
        print(f"\n  配置{cfg}: {name}")
        for reason, count in reasons.items():
            print(f"    {reason}: {count} ({count/len(cfg_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
