#!/usr/bin/env python3
"""市场脉搏编排器 — 每日市场环境判断。

串联: MarketRegimeDetector + market_sentiment + 北向资金
输出: MarketPulseReport (regime, 热点板块, 成交龙头, 仓位建议)
"""
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


@dataclass
class MarketPulseReport:
    trade_date: str
    regime: str = "unknown"          # risk_on / neutral / risk_off
    regime_score: float = 0.0        # -1.0 ~ +1.0
    regime_confidence: float = 0.0
    hot_sectors: List[Dict] = field(default_factory=list)
    volume_leaders: List[Dict] = field(default_factory=list)
    northbound_net: float = 0.0
    position_advice: str = "normal"  # aggressive / normal / defensive
    summary: str = ""


class MarketPulse:
    """市场脉搏编排器"""

    def __init__(self, backbone: Optional[DataBackbone] = None):
        self.bb = backbone or DataBackbone.get()

    def run(self, trade_date: str = None) -> MarketPulseReport:
        if not trade_date:
            trade_date = self.bb.get_latest_trade_date()

        report = MarketPulseReport(trade_date=trade_date)

        # 1. 市场环境检测
        regime_result = self._detect_regime()
        if regime_result:
            report.regime = regime_result.get("regime", "unknown")
            report.regime_score = regime_result.get("score", 0.0)
            report.regime_confidence = regime_result.get("confidence", 0.0)

        # 2. 成交额TOP20
        report.volume_leaders = self._get_volume_leaders(trade_date)

        # 3. 板块涨幅TOP20
        report.hot_sectors = self._get_hot_sectors(trade_date)

        # 4. 北向资金
        report.northbound_net = self._get_northbound(trade_date)

        # 5. 仓位建议
        report.position_advice = self._calc_position_advice(report.regime_score)
        report.summary = self._generate_summary(report)

        return report

    def _detect_regime(self) -> Optional[Dict]:
        try:
            from scripts.market_regime.regime_detector import MarketRegimeDetector
            detector = MarketRegimeDetector()
            result = detector.detect()
            if result:
                return {
                    "regime": result.regime if hasattr(result, "regime") else "unknown",
                    "score": result.score if hasattr(result, "score") else 0.0,
                    "confidence": result.confidence if hasattr(result, "confidence") else 0.0,
                }
        except Exception:
            pass
        return None

    def _get_volume_leaders(self, trade_date: str, top_n: int = 5) -> List[Dict]:
        df = self.bb.get_market_data(trade_date)
        if df is None or df.empty:
            return []
        top = df.sort_values("amount", ascending=False).head(top_n)

        stock_list = self.bb.ts.get_stock_list()
        results = []
        for _, row in top.iterrows():
            name = ""
            industry = ""
            if stock_list is not None and not stock_list.empty:
                match = stock_list[stock_list["ts_code"] == row["ts_code"]]
                if not match.empty:
                    name = match.iloc[0].get("name", "")
                    industry = match.iloc[0].get("industry", "")
            results.append({
                "code": row["ts_code"], "name": name,
                "pct_chg": round(float(row["pct_chg"]), 2),
                "amount": float(row["amount"]),
                "industry": industry,
            })
        return results

    def _get_hot_sectors(self, trade_date: str, top_n: int = 5) -> List[Dict]:
        cached = self.bb.get_cached(f"hot_sectors_{trade_date}", 3600)
        if cached is not None:
            return cached[:top_n]

        concepts = self.bb.ts._call(self.bb.ts._pro.ths_index, src="ts", type="N")
        if concepts is None or concepts.empty:
            return []

        results = []
        for _, row in concepts.head(80).iterrows():
            try:
                df = self.bb.ts._call(self.bb.ts._pro.ths_daily,
                                      ts_code=row["ts_code"],
                                      start_date=trade_date, end_date=trade_date,
                                      fields="ts_code,pct_change")
                if df is not None and not df.empty:
                    results.append({
                        "code": row["ts_code"], "name": row["name"],
                        "pct": round(float(df.iloc[0]["pct_change"]), 2),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: x["pct"], reverse=True)
        self.bb.set_cached(f"hot_sectors_{trade_date}", results)
        return results[:top_n]

    def _get_northbound(self, trade_date: str) -> float:
        try:
            from scripts.market_regime.data_fetcher import MarketDataFetcher
            fetcher = MarketDataFetcher()
            nb = fetcher.get_northbound_flow()
            if nb is not None and not nb.empty:
                latest = nb.sort_index(ascending=False).iloc[0]
                return float(latest.get("north_net", 0))
        except Exception:
            pass
        return 0.0

    def _calc_position_advice(self, score: float) -> str:
        if score >= 0.3:
            return "aggressive"
        elif score >= -0.2:
            return "normal"
        else:
            return "defensive"

    def _generate_summary(self, report: MarketPulseReport) -> str:
        parts = []
        regime_cn = {"risk_on": "risk on", "neutral": "中性",
                     "risk_off": "risk off"}.get(report.regime, "未知")
        parts.append(f"市场{regime_cn}(score={report.regime_score:+.2f})")

        if report.hot_sectors:
            top3 = ", ".join(s["name"] for s in report.hot_sectors[:3])
            parts.append(f"热点: {top3}")

        if report.volume_leaders:
            up = sum(1 for v in report.volume_leaders if v["pct_chg"] > 0)
            parts.append(f"龙头涨跌: {up}/{len(report.volume_leaders)}")

        advice_cn = {"aggressive": "可加仓", "normal": "正常仓位",
                     "defensive": "缩减仓位"}.get(report.position_advice, "")
        parts.append(f"建议: {advice_cn}")

        return " | ".join(parts)
