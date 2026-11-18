"""Core portfolio tracker: SQLite schema, snapshot CRUD, history queries.

Usage:
    tracker = PortfolioTracker("data/tracker.db")
    tracker.save_snapshot("2026-04-22", "screenshot", holdings=[...], capital={...})
    latest = tracker.get_latest_snapshot()
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add shared-libs to path
_shared_libs = str(Path.home() / "shared-libs")
if _shared_libs not in sys.path:
    sys.path.insert(0, _shared_libs)

from sqlite_utils.connection import get_connection, init_db

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS holdings_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    source TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    shares INTEGER NOT NULL,
    cost_price REAL NOT NULL,
    current_price REAL,
    market_value REAL,
    pnl_amount REAL,
    pnl_pct REAL,
    UNIQUE(snapshot_date, source, stock_code)
);

CREATE TABLE IF NOT EXISTS capital_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    source TEXT NOT NULL,
    total_assets REAL,
    available_cash REAL,
    market_value REAL,
    total_pnl REAL,
    position_ratio REAL,
    UNIQUE(snapshot_date, source)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('buy', 'sell')),
    price REAL NOT NULL,
    shares INTEGER NOT NULL,
    amount REAL NOT NULL,
    trade_type TEXT DEFAULT 'normal' CHECK(trade_type IN ('normal', 't_buy', 't_sell', 't_buyback')),
    commission REAL DEFAULT 0,
    notes TEXT DEFAULT '',
    detected_from TEXT DEFAULT 'manual' CHECK(detected_from IN ('manual', 'snapshot_diff', 'jqktrader'))
);

CREATE TABLE IF NOT EXISTS daily_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL UNIQUE,
    total_assets REAL,
    available_cash REAL,
    market_value REAL,
    total_pnl REAL,
    position_count INTEGER,
    top_gainer TEXT,
    top_loser TEXT,
    notes TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_holdings_date ON holdings_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_holdings_code ON holdings_snapshots(stock_code);
CREATE INDEX IF NOT EXISTS idx_capital_date ON capital_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_code ON trades(stock_code);
CREATE INDEX IF NOT EXISTS idx_summary_date ON daily_summary(trade_date);
"""


class PortfolioTracker:
    """Manages portfolio snapshots, trade detection, and history queries."""

    def __init__(self, db_path: str = "data/tracker.db"):
        self.db_path = db_path
        self._conn = init_db(db_path, SCHEMA_SQL)

    @property
    def conn(self):
        return get_connection(self.db_path)

    # ---- Snapshot Save ----

    def save_snapshot(self, date: str, source: str,
                      holdings: list[dict], capital: Optional[dict] = None) -> dict:
        """Save a complete portfolio snapshot.

        Args:
            date: Snapshot datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).
            source: 'screenshot', 'jqktrader', or 'manual'.
            holdings: List of dicts with keys: stock_code, stock_name, shares,
                      cost_price, current_price, market_value, pnl_amount, pnl_pct.
            capital: Dict with keys: total_assets, available_cash, market_value,
                     total_pnl, position_ratio.

        Returns:
            Dict with 'saved_count', 'changes' (from diff), 'trades_detected'.
        """
        saved_count = 0
        for h in holdings:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO holdings_snapshots
                    (snapshot_date, source, stock_code, stock_name, shares,
                     cost_price, current_price, market_value, pnl_amount, pnl_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, source,
                    h["stock_code"], h["stock_name"],
                    h["shares"], h["cost_price"],
                    h.get("current_price"), h.get("market_value"),
                    h.get("pnl_amount"), h.get("pnl_pct"),
                ))
                saved_count += 1
            except Exception as e:
                print(f"  ⚠️ 保存 {h.get('stock_code')} 失败: {e}")

        if capital:
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO capital_snapshots
                    (snapshot_date, source, total_assets, available_cash,
                     market_value, total_pnl, position_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, source,
                    capital.get("total_assets"), capital.get("available_cash"),
                    capital.get("market_value"), capital.get("total_pnl"),
                    capital.get("position_ratio"),
                ))
            except Exception as e:
                print(f"  ⚠️ 保存资金快照失败: {e}")

        self.conn.commit()

        # Diff with previous snapshot
        changes = self._diff_with_previous(date, source)
        trades_detected = self._record_detected_trades(date, changes)

        return {
            "saved_count": saved_count,
            "changes": changes,
            "trades_detected": trades_detected,
        }

    # ---- Snapshot Queries ----

    def get_latest_snapshot(self, source: Optional[str] = None) -> list[dict]:
        """Get the most recent holdings snapshot."""
        if source:
            cur = self.conn.execute(
                "SELECT MAX(snapshot_date) as max_date FROM holdings_snapshots WHERE source = ?",
                (source,)
            )
        else:
            cur = self.conn.execute(
                "SELECT MAX(snapshot_date) as max_date FROM holdings_snapshots"
            )
        row = cur.fetchone()
        if not row or not row["max_date"]:
            return []

        max_date = row["max_date"]
        if source:
            cur = self.conn.execute(
                "SELECT * FROM holdings_snapshots WHERE snapshot_date = ? AND source = ?",
                (max_date, source)
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM holdings_snapshots WHERE snapshot_date = ?",
                (max_date,)
            )
        return [dict(r) for r in cur.fetchall()]

    def get_snapshot_at(self, date: str, source: Optional[str] = None) -> list[dict]:
        """Get holdings snapshot for a specific date."""
        if source:
            cur = self.conn.execute(
                "SELECT * FROM holdings_snapshots WHERE snapshot_date LIKE ? AND source = ?",
                (f"{date}%", source)
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM holdings_snapshots WHERE snapshot_date LIKE ?",
                (f"{date}%",)
            )
        return [dict(r) for r in cur.fetchall()]

    def get_snapshot_dates(self, limit: int = 30) -> list[str]:
        """Get list of distinct snapshot dates."""
        cur = self.conn.execute(
            "SELECT DISTINCT date(snapshot_date) as d FROM holdings_snapshots "
            "ORDER BY d DESC LIMIT ?", (limit,)
        )
        return [r["d"] for r in cur.fetchall()]

    def get_snapshot_history(self, days: int = 30) -> dict[str, list[dict]]:
        """Get snapshots grouped by date for the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT * FROM holdings_snapshots WHERE snapshot_date >= ? ORDER BY snapshot_date DESC",
            (cutoff,)
        )
        result = {}
        for r in cur.fetchall():
            d = r["snapshot_date"][:10]
            result.setdefault(d, []).append(dict(r))
        return result

    def get_capital_at(self, date: str) -> Optional[dict]:
        """Get capital snapshot for a specific date."""
        cur = self.conn.execute(
            "SELECT * FROM capital_snapshots WHERE snapshot_date LIKE ? ORDER BY snapshot_date DESC LIMIT 1",
            (f"{date}%",)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_capital(self) -> Optional[dict]:
        """Get the most recent capital snapshot."""
        cur = self.conn.execute(
            "SELECT * FROM capital_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # ---- Holding Queries ----

    def get_current_holding(self, stock_code: str) -> Optional[dict]:
        """Get the latest holding info for a specific stock."""
        cur = self.conn.execute(
            "SELECT * FROM holdings_snapshots WHERE stock_code = ? "
            "ORDER BY snapshot_date DESC LIMIT 1",
            (stock_code,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_cost_basis(self, stock_code: str) -> Optional[dict]:
        """Get current shares and cost price for a stock (for T-trade analysis)."""
        h = self.get_current_holding(stock_code)
        if not h:
            return None
        return {
            "stock_code": h["stock_code"],
            "stock_name": h["stock_name"],
            "shares": h["shares"],
            "cost_price": h["cost_price"],
            "current_price": h["current_price"],
        }

    # ---- Diff Logic ----

    def _diff_with_previous(self, current_date: str, source: str) -> dict:
        """Compare current snapshot with the previous one."""
        from .snapshot_diff import diff_snapshots

        current = self.get_snapshot_at(current_date, source)
        if not current:
            return {"new_positions": [], "closed": [], "increased": [], "decreased": [], "unchanged": current}

        # Find previous snapshot date
        cur = self.conn.execute(
            "SELECT DISTINCT date(snapshot_date) as d FROM holdings_snapshots "
            "WHERE date(snapshot_date) < date(?) ORDER BY d DESC LIMIT 1",
            (current_date,)
        )
        prev_row = cur.fetchone()
        if not prev_row:
            return {
                "new_positions": current,
                "closed": [],
                "increased": [],
                "decreased": [],
                "unchanged": [],
                "message": "首次快照，所有持仓视为新建仓",
            }

        prev_date = prev_row["d"]
        previous = self.get_snapshot_at(prev_date)
        return diff_snapshots(previous, current)

    def _record_detected_trades(self, date: str, changes: dict) -> list[dict]:
        """Record trades detected from snapshot diff."""
        from .trade_recorder import TradeRecorder

        recorder = TradeRecorder(self.db_path)
        detected = []

        for h in changes.get("new_positions", []):
            if h.get("shares", 0) > 0:
                tid = recorder.record_trade(
                    trade_date=date[:10],
                    stock_code=h["stock_code"],
                    stock_name=h["stock_name"],
                    direction="buy",
                    price=h.get("cost_price", 0),
                    shares=h["shares"],
                    amount=h.get("market_value", h.get("cost_price", 0) * h.get("shares", 0)),
                    detected_from="snapshot_diff",
                    notes="快照自动检测：新建仓",
                )
                detected.append({"id": tid, "action": "new", "stock": h["stock_name"]})

        for h in changes.get("closed", []):
            if h.get("shares", 0) > 0:
                tid = recorder.record_trade(
                    trade_date=date[:10],
                    stock_code=h["stock_code"],
                    stock_name=h["stock_name"],
                    direction="sell",
                    price=h.get("current_price", h.get("cost_price", 0)),
                    shares=h["shares"],
                    amount=h.get("market_value", h.get("current_price", 0) * h.get("shares", 0)),
                    detected_from="snapshot_diff",
                    notes="快照自动检测：清仓",
                )
                detected.append({"id": tid, "action": "closed", "stock": h["stock_name"]})

        for h in changes.get("increased", []):
            diff_shares = h.get("diff_shares", 0)
            if diff_shares > 0:
                tid = recorder.record_trade(
                    trade_date=date[:10],
                    stock_code=h["stock_code"],
                    stock_name=h["stock_name"],
                    direction="buy",
                    price=h.get("current_price", h.get("cost_price", 0)),
                    shares=diff_shares,
                    amount=diff_shares * h.get("current_price", h.get("cost_price", 0)),
                    detected_from="snapshot_diff",
                    notes=f"快照自动检测：加仓{diff_shares}股",
                )
                detected.append({"id": tid, "action": "increased", "stock": h["stock_name"], "shares": diff_shares})

        for h in changes.get("decreased", []):
            diff_shares = abs(h.get("diff_shares", 0))
            if diff_shares > 0:
                tid = recorder.record_trade(
                    trade_date=date[:10],
                    stock_code=h["stock_code"],
                    stock_name=h["stock_name"],
                    direction="sell",
                    price=h.get("current_price", h.get("cost_price", 0)),
                    shares=diff_shares,
                    amount=diff_shares * h.get("current_price", h.get("cost_price", 0)),
                    detected_from="snapshot_diff",
                    notes=f"快照自动检测：减仓{diff_shares}股",
                )
                detected.append({"id": tid, "action": "decreased", "stock": h["stock_name"], "shares": diff_shares})

        return detected
