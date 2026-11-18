#!/usr/bin/env python3
"""股票雷达编排器 — 系统性发现新机会，维护关注列表。

串联: StockScreener(8模板) + AIStockMiner
输出: new_candidates + watchlist更新 + 板块轮动信号
"""
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.orchestrator.data_backbone import DataBackbone

WATCHLIST_PATH = os.path.join(PROJECT_ROOT, "config", "watchlist.json")


@dataclass
class RadarCandidate:
    code: str
    name: str
    score: float
    template: str
    reason: str = ""
    sector: str = ""


@dataclass
class RadarReport:
    trade_date: str
    new_candidates: List[RadarCandidate] = field(default_factory=list)
    watchlist_updates: List[Dict] = field(default_factory=list)
    sector_rotation: Dict[str, str] = field(default_factory=dict)
    scan_mode: str = "daily"  # daily / weekly


class StockRadar:
    """股票雷达编排器"""

    WEEKLY_TEMPLATES = ["wave3", "wave3_startup", "bottom_reversal", "momentum"]

    def __init__(self, backbone: Optional[DataBackbone] = None):
        self.bb = backbone or DataBackbone.get()

    def run_weekly(self, trade_date: str = None) -> RadarReport:
        """每周全量扫描"""
        if not trade_date:
            trade_date = self.bb.get_latest_trade_date()

        report = RadarReport(trade_date=trade_date, scan_mode="weekly")

        for template in self.WEEKLY_TEMPLATES:
            candidates = self._run_screener(template)
            for c in candidates:
                report.new_candidates.append(RadarCandidate(
                    code=c.get("symbol", c.get("ts_code", "")),
                    name=c.get("name", ""),
                    score=float(c.get("total_score", c.get("score", 0))),
                    template=template,
                    reason=c.get("reason", ""),
                    sector=c.get("industry", ""),
                ))

        # 按分数排序去重
        seen = set()
        unique = []
        for c in sorted(report.new_candidates, key=lambda x: x.score, reverse=True):
            if c.code not in seen:
                seen.add(c.code)
                unique.append(c)
        report.new_candidates = unique[:30]

        # 更新关注列表
        report.watchlist_updates = self._update_watchlist(report.new_candidates)

        return report

    def run_daily(self, trade_date: str = None) -> RadarReport:
        """每日快速扫描关注列表"""
        if not trade_date:
            trade_date = self.bb.get_latest_trade_date()

        report = RadarReport(trade_date=trade_date, scan_mode="daily")
        watchlist = self._load_watchlist()

        for item in watchlist[:15]:
            code = item.get("code", "")
            name = item.get("name", "")
            price = 0
            try:
                ts_code = self.bb.ts._to_ts_code(code) if hasattr(self.bb.ts, '_to_ts_code') else code
                df = self.bb.ts.get_kline(code, start=trade_date, end=trade_date)
                if df is not None and not df.empty:
                    price = float(df.iloc[-1]["close"])
            except Exception:
                pass

            target = item.get("target_price", 0)
            if target > 0 and price > 0:
                gap = (target - price) / price * 100
                if gap < 5:
                    report.watchlist_updates.append({
                        "code": code, "name": name, "price": price,
                        "target": target, "gap_pct": round(gap, 1),
                        "status": "接近目标价",
                    })

        return report

    def _run_screener(self, template: str) -> List[Dict]:
        try:
            from scripts.screener.screener import StockScreener
            screener = StockScreener()
            results = screener.run(template=template, top_n=10)
            if results is not None and not results.empty:
                return results.to_dict(orient="records")
        except Exception:
            pass
        return []

    def _load_watchlist(self) -> List[Dict]:
        if not os.path.exists(WATCHLIST_PATH):
            return []
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _update_watchlist(self, candidates: List[RadarCandidate]) -> List[Dict]:
        watchlist = self._load_watchlist()
        existing_codes = {item["code"] for item in watchlist}
        updates = []

        for c in candidates[:10]:
            if c.code not in existing_codes:
                entry = {
                    "code": c.code, "name": c.name,
                    "added_date": datetime.now().strftime("%Y-%m-%d"),
                    "source_template": c.template,
                    "target_price": 0,
                    "score": c.score,
                }
                watchlist.append(entry)
                updates.append({"action": "added", **entry})

        if updates:
            os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
            with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
                json.dump(watchlist, f, ensure_ascii=False, indent=2)

        return updates
