#!/usr/bin/env python3
"""MACD动量选股器回测 — 验证MACDMomentumScreener的选股效果。

核心问题：
  - 选股器在历史日期选出的top 10股票，后续5/10/20日涨跌如何？
  - 对比上证指数同期涨跌幅，是否有超额收益？
  - 胜率（正收益比例）如何？

设计：
  1. 选3个历史日期（间隔约1个月，避免区间重叠）
  2. 对每个日期：
     a. 获取全市场股票列表（排除ST、停牌、次新）
     b. 获取每只股票截至该日期的120天日线数据
     c. 用 MACDMomentumScreener.score() 评分，过滤红柱>=5天
     d. 取top 10
     e. 获取这10只股票从该日期起5日/10日/20日后的收盘价，计算收益率
  3. 基线对比：上证指数同周期涨跌幅
  4. 输出报告

方案A（默认）：每个日期只扫前500只（快速验证）
方案B（--full）：直接跑全市场（约50分钟）

用法：
  cd .
  conda activate base
  PYTHONPATH=. python scripts/research/backtest_macd_screener.py             # 方案A: 前500只
  PYTHONPATH=. python scripts/research/backtest_macd_screener.py --full      # 方案B: 全市场
  PYTHONPATH=. python scripts/research/backtest_macd_screener.py --top 20    # 取top 20
  PYTHONPATH=. python scripts/research/backtest_macd_screener.py --sample 300  # 自定义抽样数
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 确保后台输出可见
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from scripts.core.tushare_client import TushareClient, from_ts_code, to_ts_code
from scripts.screener.macd_momentum_screener import MACDMomentumScreener, MACDScoreResult

# ── 常量 ──────────────────────────────────────────────────────────────

INDEX_CODE = "000001.SH"  # 上证指数
LOOKBACK_DAYS = 200       # 获取历史日线天数（略大于120，确保MA120可用）
FORWARD_PERIODS = [5, 10, 20]  # 前瞻天数

DEFAULT_DATES = ["20250415", "20250515", "20250616"]
DEFAULT_SAMPLE = 500
DEFAULT_TOP_N = 10


# ── 数据结构 ──────────────────────────────────────────────────────────

@dataclass
class StockPick:
    """单只选股结果"""
    code: str = ""
    name: str = ""
    ts_code: str = ""
    score: float = 0.0
    red_bars: int = 0
    buy_price: float = 0.0       # 回测日期收盘价
    forward_returns: dict = field(default_factory=dict)  # {5: pct, 10: pct, 20: pct}
    detail_desc: str = ""


@dataclass
class DateResult:
    """单个日期的回测结果"""
    date: str = ""
    picks: list[StockPick] = field(default_factory=list)
    index_returns: dict = field(default_factory=dict)  # {5: pct, 10: pct, 20: pct}
    avg_returns: dict = field(default_factory=dict)     # {5: avg_pct, 10: avg_pct, 20: avg_pct}
    win_rates: dict = field(default_factory=dict)       # {5: rate, 10: rate, 20: rate}
    total_scanned: int = 0
    total_passed: int = 0


# ── 主类 ──────────────────────────────────────────────────────────────

class MACDScreenerBacktester:
    """MACD动量选股器回测"""

    def __init__(self):
        self.ts = TushareClient()
        self.screener = MACDMomentumScreener(client=self.ts)
        self._trade_cal_cache: Optional[pd.DataFrame] = None

    def _get_trade_cal(self) -> pd.DataFrame:
        """获取交易日历（带缓存）"""
        if self._trade_cal_cache is None:
            start = "20250101"
            end = "20261231"
            self._trade_cal_cache = self.ts.get_trade_cal(start=start, end=end)
        return self._trade_cal_cache

    def _find_trade_day(self, date: str) -> str:
        """找到>=指定日期的最近交易日"""
        date = date.replace("-", "")
        cal = self._get_trade_cal()
        if cal.empty:
            return date
        open_days = cal[cal["is_open"] == 1].sort_values("cal_date")
        future = open_days[open_days["cal_date"] >= date]
        if future.empty:
            return date
        return future.iloc[0]["cal_date"]

    def _get_future_trade_day(self, base_date: str, n: int) -> Optional[str]:
        """获取base_date之后第n个交易日"""
        cal = self._get_trade_cal()
        if cal.empty:
            return None
        open_days = cal[cal["is_open"] == 1].sort_values("cal_date")
        future = open_days[open_days["cal_date"] > base_date]
        if len(future) < n:
            return None
        return future.iloc[n - 1]["cal_date"]

    def _get_stock_list(self, trade_date: str) -> pd.DataFrame:
        """获取过滤后的股票列表（排除ST、北交所、次新、停牌）"""
        print(f"  获取股票列表...", flush=True)
        stocks = self.ts.get_stock_list()
        if stocks.empty:
            return stocks

        before = len(stocks)

        # 排除ST
        stocks = stocks[~stocks["name"].str.contains("ST", na=False)]

        # 排除北交所
        stocks = stocks[~stocks["ts_code"].str.endswith(".BJ")]

        # 排除次新股（上市<60天）
        if "list_date" in stocks.columns:
            cutoff = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
            stocks = stocks[stocks["list_date"].notna()]
            stocks = stocks[stocks["list_date"] <= cutoff]

        # 排除停牌：通过daily_basic获取当日有数据的股票
        print(f"  获取{trade_date}行情数据过滤停牌...", flush=True)
        daily_basic = self.ts.get_daily_basic(trade_date=trade_date)
        if not daily_basic.empty:
            active_codes = set(daily_basic["ts_code"].tolist())
            stocks = stocks[stocks["ts_code"].isin(active_codes)]

        print(f"  过滤: {before} → {len(stocks)} 只", flush=True)
        return stocks.reset_index(drop=True)

    def _scan_date(self, trade_date: str, stocks: pd.DataFrame,
                   top_n: int = 10) -> tuple[list[MACDScoreResult], int, int]:
        """在指定日期用选股器扫描股票
        
        Returns:
            (top_results, passed_count, total_scanned)
        """
        start_dt = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

        results: list[MACDScoreResult] = []
        passed = 0
        filtered_out = 0
        total = len(stocks)
        t0 = time.time()

        for i, (_, row) in enumerate(stocks.iterrows()):
            ts_code = row["ts_code"]
            code = from_ts_code(ts_code)
            name = row.get("name", code)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"    [{trade_date}] 进度: {i+1}/{total} | "
                      f"通过: {passed} | 排除: {filtered_out} | "
                      f"耗时: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)

            try:
                df = self.ts.get_kline(code, freq="D", start=start_dt, end=trade_date, adj="qfq")
                if df is None or len(df) < 60:
                    filtered_out += 1
                    continue

                r = self.screener.score(code, df)
                r.name = name

                # 过滤条件：MACD金叉 + 红柱>=5天
                if r.red_bars < 5:
                    filtered_out += 1
                    continue

                if r.total_score > 0:
                    results.append(r)
                    passed += 1

            except Exception as e:
                filtered_out += 1
                continue

        elapsed = time.time() - t0
        print(f"    [{trade_date}] 扫描完成: 通过={passed}, 排除={filtered_out}, "
              f"耗时={elapsed:.0f}s", flush=True)

        # 按评分排序，取top N
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results[:top_n], passed, total

    def _compute_forward_returns(self, code: str, trade_date: str,
                                 buy_price: float) -> dict[int, Optional[float]]:
        """计算单只股票的前瞻收益率"""
        returns = {}
        for n_days in FORWARD_PERIODS:
            future_date = self._get_future_trade_day(trade_date, n_days)
            if future_date is None:
                returns[n_days] = None
                continue

            # 获取前瞻期间的日线（从trade_date到future_date）
            df = self.ts.get_kline(code, freq="D", start=trade_date, end=future_date, adj="qfq")
            if df is None or len(df) < 2:
                returns[n_days] = None
                continue

            # 最后一天收盘价
            future_close = float(df.iloc[-1]["close"])
            if buy_price > 0:
                returns[n_days] = round((future_close / buy_price - 1) * 100, 2)
            else:
                returns[n_days] = None

        return returns

    def _compute_index_returns(self, trade_date: str) -> dict[int, Optional[float]]:
        """计算上证指数的前瞻收益率"""
        # 获取buy日期的指数收盘价
        idx_df = self.ts.get_index_daily(ts_code=INDEX_CODE, start=trade_date, end=trade_date)
        if idx_df is None or idx_df.empty:
            return {n: None for n in FORWARD_PERIODS}

        # index_daily返回的列名: trade_date, close, ...
        buy_close = float(idx_df.iloc[0]["close"])

        returns = {}
        for n_days in FORWARD_PERIODS:
            future_date = self._get_future_trade_day(trade_date, n_days)
            if future_date is None:
                returns[n_days] = None
                continue

            idx_future = self.ts.get_index_daily(ts_code=INDEX_CODE, start=trade_date, end=future_date)
            if idx_future is None or idx_future.empty:
                returns[n_days] = None
                continue

            future_close = float(idx_future.iloc[-1]["close"])
            if buy_close > 0:
                returns[n_days] = round((future_close / buy_close - 1) * 100, 2)
            else:
                returns[n_days] = None

        return returns

    def backtest_single_date(self, date: str, top_n: int = 10,
                              sample_size: Optional[int] = None) -> DateResult:
        """在某个历史日期运行选股，追踪后续涨跌

        Args:
            date: 回测日期 YYYYMMDD
            top_n: 取评分最高的前N只
            sample_size: 抽样数量，None表示全市场
        """
        result = DateResult(date=date)

        # 确保是交易日
        trade_date = self._find_trade_day(date)
        if trade_date != date:
            print(f"\n{'='*70}", flush=True)
            print(f"回测日期 {date} 非交易日，调整为 {trade_date}", flush=True)
        result.date = trade_date

        print(f"\n{'='*70}", flush=True)
        print(f"回测日期: {trade_date}", flush=True)
        print(f"{'='*70}", flush=True)

        # 1. 获取股票列表
        stocks = self._get_stock_list(trade_date)
        if stocks.empty:
            print("  ✗ 股票列表为空", flush=True)
            return result

        # 2. 抽样（如果指定）
        if sample_size and sample_size < len(stocks):
            # 随机抽样，但固定种子确保可重复
            random.seed(42)
            indices = random.sample(range(len(stocks)), sample_size)
            stocks = stocks.iloc[indices].reset_index(drop=True)
            print(f"  随机抽样: {sample_size} 只", flush=True)

        # 3. 扫描评分
        print(f"\n  开始扫描 {len(stocks)} 只股票...", flush=True)
        scored_results, passed, total = self._scan_date(trade_date, stocks, top_n)
        result.total_scanned = total
        result.total_passed = passed

        if not scored_results:
            print(f"  ✗ 无符合条件的股票", flush=True)
            return result

        # 4. 获取每只top股票的买入价和前瞻收益
        print(f"\n  计算Top {len(scored_results)} 只的前瞻收益...", flush=True)
        start_dt = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")

        for i, sr in enumerate(scored_results):
            print(f"    [{i+1}/{len(scored_results)}] {sr.name}({sr.symbol}) "
                  f"评分={sr.total_score} 红柱={sr.red_bars}", flush=True)

            pick = StockPick(
                code=sr.symbol,
                name=sr.name,
                ts_code=to_ts_code(sr.symbol),
                score=sr.total_score,
                red_bars=sr.red_bars,
                detail_desc=sr.detail_desc,
            )

            # 获取买入价（回测日期收盘价）— 从之前扫描时获取的数据中取
            try:
                df = self.ts.get_kline(sr.symbol, freq="D", start=start_dt, end=trade_date, adj="qfq")
                if df is not None and len(df) > 0:
                    pick.buy_price = float(df.iloc[-1]["close"])
            except Exception:
                pass

            # 计算前瞻收益
            if pick.buy_price > 0:
                pick.forward_returns = self._compute_forward_returns(
                    sr.symbol, trade_date, pick.buy_price
                )

            ret_str = " | ".join(
                f"{n}日: {pick.forward_returns.get(n, 'N/A')}%"
                for n in FORWARD_PERIODS
            )
            print(f"        {ret_str}", flush=True)

            result.picks.append(pick)

        # 5. 计算上证指数基准收益
        print(f"\n  计算上证指数基准收益...", flush=True)
        result.index_returns = self._compute_index_returns(trade_date)
        idx_str = " | ".join(
            f"{n}日: {result.index_returns.get(n, 'N/A')}%"
            for n in FORWARD_PERIODS
        )
        print(f"  上证指数: {idx_str}", flush=True)

        # 6. 汇总统计
        for n in FORWARD_PERIODS:
            rets = [p.forward_returns.get(n) for p in result.picks
                    if p.forward_returns.get(n) is not None]
            if rets:
                result.avg_returns[n] = round(np.mean(rets), 2)
                result.win_rates[n] = round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1)
            else:
                result.avg_returns[n] = None
                result.win_rates[n] = None

        # 7. 立刻输出该日期的结果
        self._print_date_result(result)

        return result

    def _print_date_result(self, r: DateResult):
        """打印单个日期的回测结果"""
        print(f"\n{'─'*70}", flush=True)
        print(f"📊 回测结果: {r.date} | 扫描={r.total_scanned} | 通过={r.total_passed}", flush=True)
        print(f"{'─'*70}", flush=True)

        # 表头
        header = f"{'排名':>4} {'代码':<8} {'名称':<10} {'评分':>6} {'红柱':>4}"
        for n in FORWARD_PERIODS:
            header += f" {'%d日' % n:>8}"
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for i, p in enumerate(r.picks):
            line = f"{i+1:>4} {p.code:<8} {p.name:<10} {p.score:>6.1f} {p.red_bars:>4}"
            for n in FORWARD_PERIODS:
                ret = p.forward_returns.get(n)
                if ret is not None:
                    sign = "+" if ret > 0 else ""
                    line += f" {sign}{ret:>7.2f}%"
                else:
                    line += f" {'N/A':>8}"
            print(line, flush=True)

        # 汇总
        print(f"{'─'*70}", flush=True)
        avg_line = f"{'平均':>4} {'':8} {'':10} {'':6} {'':4}"
        for n in FORWARD_PERIODS:
            avg = r.avg_returns.get(n)
            if avg is not None:
                sign = "+" if avg > 0 else ""
                avg_line += f" {sign}{avg:>7.2f}%"
            else:
                avg_line += f" {'N/A':>8}"
        print(avg_line, flush=True)

        idx_line = f"{'上证':>4} {'':8} {'':10} {'':6} {'':4}"
        for n in FORWARD_PERIODS:
            idx_ret = r.index_returns.get(n)
            if idx_ret is not None:
                sign = "+" if idx_ret > 0 else ""
                idx_line += f" {sign}{idx_ret:>7.2f}%"
            else:
                idx_line += f" {'N/A':>8}"
        print(idx_line, flush=True)

        wr_line = f"{'胜率':>4} {'':8} {'':10} {'':6} {'':4}"
        for n in FORWARD_PERIODS:
            wr = r.win_rates.get(n)
            if wr is not None:
                wr_line += f" {wr:>7.1f}%"
            else:
                wr_line += f" {'N/A':>8}"
        print(wr_line, flush=True)
        print(f"{'─'*70}", flush=True)

        # 超额收益
        print(f"  超额收益:", flush=True)
        for n in FORWARD_PERIODS:
            avg = r.avg_returns.get(n)
            idx_ret = r.index_returns.get(n)
            if avg is not None and idx_ret is not None:
                excess = round(avg - idx_ret, 2)
                sign = "+" if excess > 0 else ""
                emoji = "✅" if excess > 0 else "❌"
                print(f"    {n}日: {sign}{excess:.2f}% {emoji}", flush=True)
            else:
                print(f"    {n}日: N/A", flush=True)

    def run(self, dates: list[str], top_n: int = 10,
            sample_size: Optional[int] = None) -> list[DateResult]:
        """多日期回测"""
        print("=" * 70, flush=True)
        print("MACD动量选股器 — 回测验证", flush=True)
        print("=" * 70, flush=True)
        print(f"回测日期: {dates}", flush=True)
        print(f"Top N: {top_n}", flush=True)
        print(f"抽样数量: {'全市场' if sample_size is None else sample_size}", flush=True)
        print(f"前瞻周期: {FORWARD_PERIODS} 日", flush=True)

        t0 = time.time()
        results = []

        for date in dates:
            r = self.backtest_single_date(date, top_n=top_n, sample_size=sample_size)
            results.append(r)

        elapsed = time.time() - t0
        print(f"\n{'='*70}", flush=True)
        print(f"回测完成！总耗时: {elapsed:.0f}s", flush=True)
        print(f"{'='*70}", flush=True)

        # 综合统计
        self._print_summary(results)

        return results

    def _print_summary(self, results: list[DateResult]):
        """打印综合统计"""
        print(f"\n{'='*70}", flush=True)
        print("📊 综合统计（3个日期汇总）", flush=True)
        print(f"{'='*70}", flush=True)

        for n in FORWARD_PERIODS:
            # 收集所有有效收益
            all_rets = []
            all_idx_rets = []
            for r in results:
                for p in r.picks:
                    ret = p.forward_returns.get(n)
                    if ret is not None:
                        all_rets.append(ret)
                idx_ret = r.index_returns.get(n)
                if idx_ret is not None:
                    all_idx_rets.append(idx_ret)

            if not all_rets:
                print(f"\n  {n}日: 无有效数据", flush=True)
                continue

            avg_ret = np.mean(all_rets)
            med_ret = np.median(all_rets)
            win_rate = sum(1 for r in all_rets if r > 0) / len(all_rets) * 100
            avg_idx = np.mean(all_idx_rets) if all_idx_rets else 0
            excess = avg_ret - avg_idx
            max_ret = max(all_rets)
            min_ret = min(all_rets)

            print(f"\n  ── {n}日收益 ──", flush=True)
            print(f"    样本数: {len(all_rets)}", flush=True)
            print(f"    平均收益: {avg_ret:+.2f}% (中位数: {med_ret:+.2f}%)", flush=True)
            print(f"    上证指数: {avg_idx:+.2f}%", flush=True)
            print(f"    超额收益: {excess:+.2f}% {'✅' if excess > 0 else '❌'}", flush=True)
            print(f"    胜率: {win_rate:.1f}% ({sum(1 for r in all_rets if r > 0)}/{len(all_rets)})", flush=True)
            print(f"    最大: {max_ret:+.2f}% | 最小: {min_ret:+.2f}%", flush=True)

        # 总览表
        print(f"\n{'─'*70}", flush=True)
        print("总览: 选股平均收益 vs 上证指数", flush=True)
        print(f"{'─'*70}", flush=True)
        header = f"{'周期':>6}"
        for r in results:
            header += f" | {r.date:>10}"
        header += " | 综合"
        print(header, flush=True)

        for n in FORWARD_PERIODS:
            line = f"{n:>4}日"
            all_rets = []
            for r in results:
                avg = r.avg_returns.get(n)
                idx = r.index_returns.get(n)
                if avg is not None and idx is not None:
                    excess = avg - idx
                    sign = "+" if excess > 0 else ""
                    line += f" | {sign}{excess:>8.2f}%"
                    all_rets.append(excess)
                else:
                    line += f" | {'N/A':>10}"
            if all_rets:
                overall = np.mean(all_rets)
                sign = "+" if overall > 0 else ""
                line += f" | {sign}{overall:>6.2f}%"
            else:
                line += f" | {'N/A':>8}"
            print(line, flush=True)

        print(f"{'─'*70}", flush=True)

        # 胜率总览
        header2 = f"{'胜率':>6}"
        for r in results:
            header2 += f" | {r.date:>10}"
        header2 += " | 综合"
        print(header2, flush=True)

        for n in FORWARD_PERIODS:
            line = f"{n:>4}日"
            all_wins = []
            all_total = 0
            for r in results:
                wr = r.win_rates.get(n)
                if wr is not None:
                    line += f" | {wr:>9.1f}%"
                    cnt = sum(1 for p in r.picks if p.forward_returns.get(n) is not None)
                    wins = sum(1 for p in r.picks if (p.forward_returns.get(n) or 0) > 0)
                    all_wins.append(wins)
                    all_total += cnt
                else:
                    line += f" | {'N/A':>10}"
            if all_wins:
                total_wins = sum(all_wins)
                overall_wr = total_wins / all_total * 100 if all_total > 0 else 0
                line += f" | {overall_wr:>6.1f}%"
            else:
                line += f" | {'N/A':>8}"
            print(line, flush=True)

        print(f"{'─'*70}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MACD动量选股器回测")
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES,
                        help="回测日期 YYYYMMDD，默认3个日期")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help=f"取Top N只 (默认{DEFAULT_TOP_N})")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help=f"每个日期抽样数量 (默认{DEFAULT_SAMPLE})")
    parser.add_argument("--full", action="store_true",
                        help="全市场扫描（忽略--sample）")
    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    bt = MACDScreenerBacktester()
    bt.run(dates=args.dates, top_n=args.top, sample_size=sample_size)


if __name__ == "__main__":
    main()
