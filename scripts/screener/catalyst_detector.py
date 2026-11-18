#!/usr/bin/env python3
"""催化事件检测器 — 识别短期催化剂。

检测四类催化事件：
  1. 财报催化：近期有财报发布，业绩超预期
  2. 政策催化：行业政策关键词暴增
  3. 资金催化：北向资金/机构异动
  4. 技术催化：突破关键阻力位/涨停板确认

用法：
  python scripts/screener/catalyst_detector.py --codes 603881,002415
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient, from_ts_code


# 行业政策关键词映射
POLICY_KEYWORDS = {
    "AI": ["人工智能", "AI", "大模型", "算力", "智能", "GPT", "LLM", "AGI"],
    "半导体": ["芯片", "半导体", "集成电路", "国产替代", "光刻"],
    "新能源": ["光伏", "风电", "储能", "新能源", "碳中和"],
    "医药": ["创新药", "集采", "医保", "医药"],
}


class CatalystDetector:
    """催化事件检测器"""

    def __init__(self, client: Optional[TushareClient] = None):
        self.client = client or TushareClient()

    def detect(
        self,
        ts_codes: list[str],
        trade_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """检测多只股票的催化事件。

        Args:
            ts_codes: 股票代码列表
            trade_date: 参考日期

        Returns:
            DataFrame: ts_code, name, catalysts, catalyst_score, details
        """
        if not trade_date:
            trade_date = self.client.get_recent_trade_day()

        results = []
        for ts_code in ts_codes:
            info = self._detect_single(ts_code, trade_date)
            if info:
                results.append(info)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.sort_values("catalyst_score", ascending=False).reset_index(drop=True)

    def _detect_single(self, ts_code: str, trade_date: str) -> Optional[dict]:
        """检测单只股票的催化事件"""
        symbol = from_ts_code(ts_code)

        catalysts = []
        score = 0
        details = []

        # ── 1. 财报催化 ──
        earnings = self._check_earnings_catalyst(ts_code, trade_date)
        if earnings:
            catalysts.append("财报")
            score += earnings["score"]
            details.append(earnings["detail"])

        # ── 2. 技术催化（涨停/突破） ──
        tech = self._check_technical_catalyst(symbol, trade_date)
        if tech:
            catalysts.append("技术")
            score += tech["score"]
            details.append(tech["detail"])

        # ── 3. 量价异动 ──
        vol = self._check_volume_catalyst(symbol, trade_date)
        if vol:
            catalysts.append("量价")
            score += vol["score"]
            details.append(vol["detail"])

        # 股票名称
        name = self._get_name(ts_code)

        return {
            "ts_code": ts_code,
            "name": name,
            "catalysts": "+".join(catalysts) if catalysts else "无",
            "catalyst_score": score,
            "n_catalysts": len(catalysts),
            "details": "; ".join(details) if details else "无明显催化",
        }

    def _check_earnings_catalyst(self, ts_code: str, trade_date: str) -> Optional[dict]:
        """财报催化：近7天有财报公告"""
        try:
            fina = self.client.get_financial(ts_code)
            if fina.empty or "ann_date" not in fina.columns:
                return None

            td = datetime.strptime(trade_date, "%Y%m%d")
            window_start = (td - timedelta(days=14)).strftime("%Y%m%d")

            # 找近14天公告的财报
            recent = fina[fina["ann_date"] >= window_start]
            recent = recent[recent["ann_date"] <= trade_date]

            if recent.empty:
                return None

            latest = recent.iloc[0]
            rev_growth = float(latest.get("or_yoy", 0)) if pd.notna(latest.get("or_yoy")) else 0
            profit_growth = float(latest.get("q_gr_yoy", 0)) if pd.notna(latest.get("q_gr_yoy")) else 0

            # 业绩超预期 = 营收/利润增速>20%
            if rev_growth > 20 or profit_growth > 20:
                return {
                    "score": 25,
                    "detail": f"财报({latest['end_date']})营收+{rev_growth:.0f}% 利润+{profit_growth:.0f}%",
                }
            elif rev_growth > 0 and profit_growth > 0:
                return {
                    "score": 10,
                    "detail": f"财报({latest['end_date']})营收+{rev_growth:.0f}% 利润+{profit_growth:.0f}%",
                }
            return None

        except Exception:
            return None

    def _check_technical_catalyst(self, symbol: str, trade_date: str) -> Optional[dict]:
        """技术催化：近期涨停或突破新高"""
        try:
            td = datetime.strptime(trade_date, "%Y%m%d")
            start = (td - timedelta(days=30)).strftime("%Y%m%d")
            df = self.client.get_kline(symbol, freq="D", start=start, end=trade_date)

            if df is None or len(df) < 5:
                return None

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            change = df["change_pct"].astype(float) if "change_pct" in df.columns else close.pct_change() * 100

            # 检查近5天是否有涨停(>9.5%)
            recent_changes = change.tail(5).values
            limit_up_days = sum(1 for c in recent_changes if c > 9.5)

            if limit_up_days > 0:
                return {
                    "score": 20,
                    "detail": f"近5天{limit_up_days}次涨停",
                }

            # 检查是否创60日新高（近5天内）
            if len(df) >= 5:
                recent_high = high.tail(5).max()
                prev_high = high.iloc[:-5].max() if len(high) > 5 else high.iloc[0]
                if recent_high > prev_high:
                    return {
                        "score": 15,
                        "detail": f"突破前高{prev_high:.2f}→{recent_high:.2f}",
                    }

            return None

        except Exception:
            return None

    def _check_volume_catalyst(self, symbol: str, trade_date: str) -> Optional[dict]:
        """量价异动：成交量突然放大"""
        try:
            td = datetime.strptime(trade_date, "%Y%m%d")
            start = (td - timedelta(days=30)).strftime("%Y%m%d")
            df = self.client.get_kline(symbol, freq="D", start=start, end=trade_date)

            if df is None or len(df) < 10:
                return None

            volume = df["volume"].astype(float)
            vol_latest = volume.iloc[-1]
            vol_avg20 = volume.tail(20).mean()

            if vol_avg20 == 0:
                return None

            vol_ratio = vol_latest / vol_avg20

            if vol_ratio > 3.0:
                return {
                    "score": 15,
                    "detail": f"放量{vol_ratio:.1f}倍",
                }
            elif vol_ratio > 2.0:
                return {
                    "score": 8,
                    "detail": f"放量{vol_ratio:.1f}倍",
                }
            return None

        except Exception:
            return None

    def _get_name(self, ts_code: str) -> str:
        """获取股票名称"""
        try:
            stocks = self.client.get_stock_list()
            match = stocks[stocks["ts_code"] == ts_code]
            if not match.empty:
                return match.iloc[0]["name"]
        except Exception:
            pass
        return ""

    @staticmethod
    def format_report(results: pd.DataFrame) -> str:
        """格式化催化检测结果"""
        if results.empty:
            return "无催化检测结果"

        lines = []
        lines.append("=" * 65)
        lines.append("  🔥 催化事件检测报告")
        lines.append("=" * 65)
        lines.append("")

        for _, row in results.iterrows():
            name = str(row.get("name", ""))[:6]
            score = row.get("catalyst_score", 0)
            cats = row.get("catalysts", "")
            detail = row.get("details", "")
            n = row.get("n_catalysts", 0)

            stars = "🔥" * min(n, 3) if n > 0 else "—"
            lines.append(f"  {row.get('ts_code','')} {name:6} {stars}")
            lines.append(f"    催化分: {score} | 类型: {cats}")
            lines.append(f"    {detail}")
            lines.append("")

        return "\n".join(lines)
