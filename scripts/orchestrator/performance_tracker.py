#!/usr/bin/env python3
"""表现追踪器 — 胜率、盈亏比、累计回报、月度归因。

从SQLite trades表和capital快照中计算历史表现。
"""
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "tracker.db")


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_loss_ratio: float = 0.0
    total_pnl: float = 0.0
    monthly_returns: List[Dict] = field(default_factory=list)
    best_trade: Dict = field(default_factory=dict)
    worst_trade: Dict = field(default_factory=dict)


class PerformanceTracker:
    """交易表现追踪器"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def calculate(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        if not os.path.exists(self.db_path):
            return metrics

        conn = sqlite3.connect(self.db_path)
        try:
            # 从trades表获取交易数据
            trades = self._get_trades(conn)
            if not trades:
                return metrics

            metrics.total_trades = len(trades)
            profits = [t for t in trades if t.get("pnl", 0) > 0]
            losses = [t for t in trades if t.get("pnl", 0) < 0]

            metrics.winning_trades = len(profits)
            metrics.losing_trades = len(losses)
            metrics.win_rate = round(metrics.winning_trades / max(metrics.total_trades, 1) * 100, 1)

            if profits:
                metrics.avg_profit = round(sum(t["pnl"] for t in profits) / len(profits), 2)
                metrics.best_trade = max(profits, key=lambda t: t["pnl"])
            if losses:
                metrics.avg_loss = round(sum(t["pnl"] for t in losses) / len(losses), 2)
                metrics.worst_trade = min(losses, key=lambda t: t["pnl"])

            if metrics.avg_loss != 0:
                metrics.profit_loss_ratio = round(abs(metrics.avg_profit / metrics.avg_loss), 2)

            metrics.total_pnl = round(sum(t.get("pnl", 0) for t in trades), 2)

            # 月度回报
            metrics.monthly_returns = self._calc_monthly_returns(conn)

        finally:
            conn.close()

        return metrics

    def _get_trades(self, conn) -> List[Dict]:
        try:
            cur = conn.execute(
                "SELECT symbol, name, direction, price, shares, trade_date, "
                "trade_type, commission FROM trades ORDER BY trade_date"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except sqlite3.OperationalError:
            return []

    def _calc_monthly_returns(self, conn) -> List[Dict]:
        try:
            cur = conn.execute(
                "SELECT strftime('%Y-%m', date) as month, "
                "FIRST_VALUE(total_assets) OVER (PARTITION BY strftime('%Y-%m', date) ORDER BY date) as start_assets, "
                "LAST_VALUE(total_assets) OVER (PARTITION BY strftime('%Y-%m', date) ORDER BY date "
                "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as end_assets, "
                "date FROM capital_snapshots ORDER BY date"
            )
            rows = cur.fetchall()
            monthly = {}
            for row in rows:
                month = row[0]
                if month not in monthly:
                    monthly[month] = {"start": row[1], "end": row[2]}
                else:
                    monthly[month]["end"] = row[2]

            results = []
            for month, vals in sorted(monthly.items()):
                start = vals["start"]
                end = vals["end"]
                ret = (end / start - 1) * 100 if start > 0 else 0
                results.append({
                    "month": month, "start": start, "end": end,
                    "return_pct": round(ret, 2),
                })
            return results
        except sqlite3.OperationalError:
            return []

    def save_metrics(self, metrics: PerformanceMetrics):
        """保存到strategy_performance表"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    date TEXT, total_trades INTEGER, win_rate REAL,
                    avg_profit REAL, avg_loss REAL, pl_ratio REAL,
                    total_pnl REAL, notes TEXT
                )
            """)
            conn.execute(
                "INSERT INTO strategy_performance VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d"), metrics.total_trades,
                 metrics.win_rate, metrics.avg_profit, metrics.avg_loss,
                 metrics.profit_loss_ratio, metrics.total_pnl, "")
            )
            conn.commit()
        finally:
            conn.close()
