#!/usr/bin/env python3
"""多标的强弱对比分析器。

对比持仓股票与同板块/同类标的的强弱关系，识别补跌风险和领涨标的。

核心思路：
  - 跌得多的可能先企稳（超卖反弹）
  - 跌得少的要防补跌
  - 涨幅差异反映资金偏好

用法：
  from scripts.analyzers.relative_strength import RelativeStrength

  rs = RelativeStrength(store)
  result = rs.compare(symbols=["603881", "002250", "603617"], period=20)
  result = rs.sector_compare("603881")  # 自动找同板块对比
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import pandas as pd
import numpy as np

from scripts.core.data_store import DataStore


@dataclass
class StrengthRanking:
    """单个标的的强弱排名。"""
    symbol: str
    name: str = ""
    change_pct: float = 0.0       # 区间涨跌幅%
    max_drawdown: float = 0.0     # 最大回撤%
    volatility: float = 0.0       # 波动率%
    relative_strength: float = 0.0  # 相对强度（相对平均）
    rank: int = 0                  # 排名（1=最强）
    risk_flag: str = ""            # 风险标记


@dataclass
class CompareResult:
    """对比结果。"""
    symbols_count: int = 0
    avg_change: float = 0.0
    rankings: List[StrengthRanking] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""


class RelativeStrength:
    """多标的强弱对比分析器。"""

    def __init__(self, store: DataStore = None):
        self._store = store or DataStore()

    def compare(self, symbols: List[str], period: int = 20,
                benchmark: str = "000001") -> CompareResult:
        """对比多个标的的近期强弱。

        Args:
            symbols: 股票代码列表
            period: 对比天数（默认20日）
            benchmark: 基准指数（默认上证）

        Returns:
            CompareResult
        """
        result = CompareResult()
        rankings = []

        # 获取基准数据
        bench_df = self._store.get_kline(benchmark, period="D", auto_update=False)
        bench_change = 0.0
        if bench_df is not None and len(bench_df) >= period:
            bench_change = (bench_df["close"].iloc[-1] / bench_df["close"].iloc[-period] - 1) * 100

        all_changes = []

        for symbol in symbols:
            df = self._store.get_kline(symbol, period="D", auto_update=True)
            if df is None or len(df) < period:
                rankings.append(StrengthRanking(
                    symbol=symbol, name="", change_pct=0,
                    risk_flag="数据不足",
                ))
                continue

            recent = df.tail(period)
            close = recent["close"].values

            # 区间涨跌幅
            change_pct = (close[-1] / close[0] - 1) * 100
            all_changes.append(change_pct)

            # 最大回撤
            cummax = np.maximum.accumulate(close)
            drawdown = (close - cummax) / cummax * 100
            max_drawdown = drawdown.min()

            # 波动率（日收益率标准差 * sqrt(252)）
            daily_returns = np.diff(close) / close[:-1]
            volatility = np.std(daily_returns) * np.sqrt(252) * 100

            # 相对强度（相对基准）
            relative_strength = change_pct - bench_change

            # 获取名称（尝试从数据中获取）
            name = ""

            rankings.append(StrengthRanking(
                symbol=symbol,
                name=name,
                change_pct=round(change_pct, 2),
                max_drawdown=round(max_drawdown, 2),
                volatility=round(volatility, 1),
                relative_strength=round(relative_strength, 2),
            ))

        # 按涨跌幅排名
        rankings.sort(key=lambda x: x.change_pct, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1

        # 计算平均值
        if all_changes:
            result.avg_change = round(np.mean(all_changes), 2)

        result.symbols_count = len(symbols)
        result.rankings = rankings

        # ── 风险提示 ──
        warnings = []
        for r in rankings:
            if r.change_pct > result.avg_change * 2 and r.change_pct > 5:
                warnings.append(
                    f"⚠️ {r.symbol} 涨幅{r.change_pct:+.1f}% 远超均值{result.avg_change:+.1f}%，"
                    f"注意回调风险"
                )
            elif r.change_pct < result.avg_change - 10 and r.change_pct < -5:
                warnings.append(
                    f"⚠️ {r.symbol} 跌幅{r.change_pct:+.1f}% 远弱于均值{result.avg_change:+.1f}%，"
                    f"{'可能超跌反弹' if r.max_drawdown < -15 else '趋势偏弱'}"
                )

            # 补跌风险检测
            if r.rank > len(rankings) * 0.7 and abs(r.change_pct) < abs(result.avg_change) * 0.3:
                if result.avg_change < 0:
                    warnings.append(
                        f"🔴 {r.symbol} 跌幅偏小({r.change_pct:+.1f}%)，"
                        f"大盘跌了{result.avg_change:+.1f}%，注意补跌风险"
                    )

        result.warnings = warnings

        # 生成总结
        lines = [f"📊 {period}日强弱对比 (均值{result.avg_change:+.1f}%)\n"]
        for r in rankings:
            bar_len = int(abs(r.change_pct) / 2)
            if r.change_pct >= 0:
                bar = "█" * bar_len
                line = f"  {r.rank}. {r.symbol:>6s} {r.change_pct:+6.1f}% {bar}"
            else:
                bar = "░" * bar_len
                line = f"  {r.rank}. {r.symbol:>6s} {r.change_pct:+6.1f}% {bar}"
            line += f"  (vs基准{r.relative_strength:+.1f}%)"
            if r.max_drawdown < -10:
                line += f"  回撤{r.max_drawdown:.1f}%"
            lines.append(line)

        if warnings:
            lines.append("")
            for w in warnings:
                lines.append(w)

        result.summary = "\n".join(lines)
        return result

    def sector_compare(self, symbol: str, period: int = 20) -> CompareResult:
        """自动找同板块标的进行对比。

        Args:
            symbol: 目标股票
            period: 对比天数

        Returns:
            CompareResult（包含同板块标的）
        """
        # 先获取行业信息
        try:
            basic = self._store.tushare.get_stock_info(symbol)
            industry = basic.get("industry", "")
        except Exception:
            industry = ""

        if not industry:
            # 找不到行业，只对比自身
            return self.compare([symbol], period=period)

        # 获取同行业股票
        try:
            industry_stocks = self._store.tushare.get_industry_stocks(industry)
            if industry_stocks is not None and not industry_stocks.empty:
                # 取前10只（按代码）
                peers = industry_stocks["ts_code"].head(10).tolist()
                # 转换代码格式
                peer_symbols = [c.split(".")[0] for c in peers]
                # 确保目标在里面
                if symbol not in peer_symbols:
                    peer_symbols.insert(0, symbol)
                return self.compare(peer_symbols[:10], period=period)
        except Exception:
            pass

        return self.compare([symbol], period=period)

    def format_report(self, result: CompareResult, title: str = "") -> str:
        """格式化输出。"""
        lines = [f"{'='*55}"]
        lines.append(f"📊 多标的强弱对比: {title}")
        lines.append(f"{'='*55}")
        lines.append(result.summary)
        lines.append(f"{'='*55}")
        return "\n".join(lines)
