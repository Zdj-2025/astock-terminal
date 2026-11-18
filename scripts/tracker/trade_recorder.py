"""Trade recording and querying.

Usage:
    recorder = TradeRecorder("data/tracker.db")
    recorder.record_trade("2026-04-22", "603881", "数据港", "buy", 38.10, 500, ...)
    trades = recorder.get_trades(days=7)
    t_profit = recorder.calc_t_profit(days=30)
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

_shared_libs = str(Path.home() / "shared-libs")
if _shared_libs not in sys.path:
    sys.path.insert(0, _shared_libs)

from sqlite_utils.connection import get_connection


class TradeRecorder:
    """Records and queries trade history."""

    def __init__(self, db_path: str = "data/tracker.db"):
        self.db_path = db_path

    @property
    def conn(self):
        return get_connection(self.db_path)

    def record_trade(
        self,
        trade_date: str,
        stock_code: str,
        stock_name: str,
        direction: str,
        price: float,
        shares: int,
        amount: Optional[float] = None,
        trade_type: str = "normal",
        commission: Optional[float] = None,
        notes: str = "",
        detected_from: str = "manual",
    ) -> int:
        """Record a single trade.

        Args:
            trade_date: YYYY-MM-DD.
            direction: 'buy' or 'sell'.
            trade_type: 'normal', 't_buy', 't_sell', 't_buyback'.
            commission: Override auto-calc if provided.
            detected_from: 'manual', 'snapshot_diff', 'jqktrader'.

        Returns:
            The inserted row id.
        """
        if amount is None:
            amount = price * shares

        if commission is None:
            commission = self._calc_commission(amount)

        cur = self.conn.execute("""
            INSERT INTO trades
            (trade_date, stock_code, stock_name, direction, price, shares,
             amount, trade_type, commission, notes, detected_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_date, stock_code, stock_name, direction, price, shares,
            amount, trade_type, commission, notes, detected_from,
        ))
        self.conn.commit()
        return cur.lastrowid

    def _calc_commission(self, amount: float, rate: float = 0.00025,
                         min_commission: float = 5.0) -> float:
        """Calculate commission (default: 万2.5, min ¥5)."""
        return max(amount * rate, min_commission)

    # ---- Queries ----

    def get_trades(self, days: Optional[int] = None,
                   stock_code: Optional[str] = None,
                   direction: Optional[str] = None,
                   trade_type: Optional[str] = None,
                   detected_from: Optional[str] = None) -> list[dict]:
        """Query trades with optional filters."""
        conditions = []
        params = []

        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            conditions.append("trade_date >= ?")
            params.append(cutoff)

        if stock_code:
            conditions.append("stock_code = ?")
            params.append(stock_code)

        if direction:
            conditions.append("direction = ?")
            params.append(direction)

        if trade_type:
            conditions.append("trade_type = ?")
            params.append(trade_type)

        if detected_from:
            conditions.append("detected_from = ?")
            params.append(detected_from)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM trades WHERE {where} ORDER BY trade_date DESC, id DESC"

        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def get_t_trades(self, days: int = 30) -> list[dict]:
        """Get all T-trade records (t_buy, t_sell, t_buyback)."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT * FROM trades WHERE trade_date >= ? AND trade_type != 'normal' "
            "ORDER BY trade_date DESC",
            (cutoff,)
        )
        return [dict(r) for r in cur.fetchall()]

    def calc_t_profit(self, days: int = 30) -> dict:
        """Calculate T-trade profit over a period.

        Matches t_buy + t_sell (or t_sell + t_buyback) pairs by stock and date.
        """
        t_trades = self.get_t_trades(days)
        if not t_trades:
            return {"total_profit": 0, "pair_count": 0, "details": []}

        # Group by stock_code + trade_date
        pairs = {}
        for t in t_trades:
            key = (t["stock_code"], t["trade_date"])
            pairs.setdefault(key, []).append(t)

        results = []
        total_profit = 0

        for (code, date), trades in pairs.items():
            buys = [t for t in trades if t["direction"] == "buy"]
            sells = [t for t in trades if t["direction"] == "sell"]

            for buy, sell in zip(buys, sells):
                profit = sell["amount"] - buy["amount"] - buy["commission"] - sell["commission"]
                total_profit += profit
                results.append({
                    "stock_code": code,
                    "stock_name": buy["stock_name"],
                    "date": date,
                    "buy_price": buy["price"],
                    "sell_price": sell["price"],
                    "shares": min(buy["shares"], sell["shares"]),
                    "profit": profit,
                    "commission_total": buy["commission"] + sell["commission"],
                })

        return {
            "total_profit": total_profit,
            "pair_count": len(results),
            "details": results,
        }

    def get_trade_summary(self, days: int = 30) -> dict:
        """Get trade statistics for a period."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT direction, COUNT(*) as cnt, SUM(amount) as total_amount "
            "FROM trades WHERE trade_date >= ? GROUP BY direction",
            (cutoff,)
        )
        stats = {r["direction"]: {"count": r["cnt"], "amount": r["total_amount"]} for r in cur.fetchall()}

        return {
            "period_days": days,
            "buy_count": stats.get("buy", {}).get("count", 0),
            "sell_count": stats.get("sell", {}).get("count", 0),
            "buy_amount": stats.get("buy", {}).get("amount", 0),
            "sell_amount": stats.get("sell", {}).get("amount", 0),
        }

    def format_trades_table(self, trades: list[dict]) -> str:
        """Format trades as a readable table."""
        if not trades:
            return "无交易记录。"

        lines = []
        for t in trades:
            arrow = "买入" if t["direction"] == "buy" else "卖出"
            type_tag = ""
            if t["trade_type"] != "normal":
                type_map = {"t_buy": "[T买]", "t_sell": "[T卖]", "t_buyback": "[T回买]"}
                type_tag = type_map.get(t["trade_type"], "")

            source_tag = ""
            if t["detected_from"] == "snapshot_diff":
                source_tag = "(自动检测)"
            elif t["detected_from"] == "jqktrader":
                source_tag = "(同花顺)"

            lines.append(
                f"  {t['trade_date']} {t['stock_name']}({t['stock_code']}) "
                f"{arrow}{type_tag} {t['shares']}股 × ¥{t['price']:.3f} "
                f"= ¥{t['amount']:,.2f} {source_tag}"
            )

        return "\n".join(lines)
