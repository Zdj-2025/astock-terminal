#!/usr/bin/env python3
"""轻量级三态市场环境判断。

基于上证指数（000001.SH）：
  - 多头：周线 MA5 > MA10 > MA20
  - 空头：周线 MA5 < MA10 < MA20
  - 震荡：其他

结合日线 MACD 方向微调置信度。

用法:
    from scripts.market_regime.simple_regime import SimpleRegime
    regime = SimpleRegime()
    result = regime.judge()
    print(result.regime, result.confidence, result.advice)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient


@dataclass
class RegimeResult:
    """市场环境判断结果"""
    regime: str           # "bull" / "sideways" / "bear"
    confidence: float     # 0-100
    advice: str           # 操作建议
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    macd_signal: str = ""  # "bullish" / "bearish" / "neutral"
    detail: str = ""


class SimpleRegime:
    """三态市场环境：多头 / 震荡 / 空头"""

    ADVICE = {
        "bull": "可积极操作，仓位可到70%",
        "sideways": "精选个股，仓位50%以下",
        "bear": "谨慎观望，仓位30%以下，不做T",
    }

    def __init__(self, client: Optional[TushareClient] = None):
        self.client = client or TushareClient()

    def _get_weekly(self) -> pd.DataFrame:
        """取上证指数周线数据"""
        try:
            # 用日线数据重采样为周线
            df = self.client.get_index_daily(
                "000001.SH",
                start=(pd.Timestamp.now() - pd.Timedelta(days=400))
                    .strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date")
                weekly = df.resample("W").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "vol": "sum",
                }).dropna()
                weekly = weekly.reset_index()
                return weekly
        except Exception as e:
            print(f"获取周线失败: {e}")
        return pd.DataFrame()

    def _get_daily(self) -> pd.DataFrame:
        """取上证指数日线数据（用于MACD微调）"""
        try:
            df = self.client.get_index_daily(
                "000001.SH",
                start=(pd.Timestamp.now() - pd.Timedelta(days=120))
                    .strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                df = df.sort_values("trade_date").reset_index(drop=True)
                df["close"] = df["close"].astype(float)
                return df
        except Exception:
            pass
        return pd.DataFrame()

    @staticmethod
    def _calc_macd(close: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9):
        """计算 MACD"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = (dif - dea) * 2
        return dif, dea, hist

    def judge(self) -> RegimeResult:
        """判断当前市场环境"""
        # ── 周线均线判断 ──
        weekly = self._get_weekly()
        if weekly.empty or len(weekly) < 25:
            return RegimeResult(
                regime="sideways", confidence=30,
                advice=self.ADVICE["sideways"],
                detail="周线数据不足，默认震荡",
            )

        close = weekly["close"]
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]

        # 判断三态
        if ma5 > ma10 > ma20:
            regime = "bull"
        elif ma5 < ma10 < ma20:
            regime = "bear"
        else:
            regime = "sideways"

        # ── 日线 MACD 微调 ──
        macd_signal = "neutral"
        daily = self._get_daily()
        if not daily.empty and len(daily) >= 35:
            dif, dea, hist = self._calc_macd(daily["close"])
            last_hist = hist.iloc[-1]
            prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0
            if last_hist > prev_hist and last_hist > 0:
                macd_signal = "bullish"
            elif last_hist < prev_hist and last_hist < 0:
                macd_signal = "bearish"
            # else neutral

        # ── 置信度计算 ──
        base_conf = 50

        # 均线间距越大置信度越高
        spread = abs(ma5 - ma20) / ma20 * 100  # 百分比
        base_conf += min(spread * 5, 20)  # 最多+20

        # MACD 同向加成
        macd_aligned = False
        if regime == "bull" and macd_signal == "bullish":
            base_conf += 15
            macd_aligned = True
        elif regime == "bear" and macd_signal == "bearish":
            base_conf += 15
            macd_aligned = True
        elif regime == "bull" and macd_signal == "bearish":
            base_conf -= 10
        elif regime == "bear" and macd_signal == "bullish":
            base_conf -= 10
        elif regime == "sideways":
            if macd_signal == "bullish":
                base_conf += 5
            elif macd_signal == "bearish":
                base_conf += 5
            # 震荡态 MACD 方向明确反而增加判断信心

        confidence = max(10, min(95, base_conf))

        detail_parts = [
            f"MA5={ma5:.2f} MA10={ma10:.2f} MA20={ma20:.2f}",
        ]
        if macd_signal != "neutral":
            detail_parts.append(f"日线MACD{macd_signal}")
        if macd_aligned:
            detail_parts.append("周线+日线共振")

        return RegimeResult(
            regime=regime,
            confidence=round(confidence, 0),
            advice=self.ADVICE[regime],
            ma5=round(ma5, 2),
            ma10=round(ma10, 2),
            ma20=round(ma20, 2),
            macd_signal=macd_signal,
            detail=" | ".join(detail_parts),
        )

    def format_report(self, result: RegimeResult) -> str:
        """格式化输出"""
        labels = {"bull": "🟢 多头", "sideways": "🟡 震荡", "bear": "🔴 空头"}
        lines = [
            "=" * 50,
            "  📊 市场环境判断（上证指数）",
            "=" * 50,
            f"  环境: {labels.get(result.regime, result.regime)}",
            f"  置信度: {result.confidence:.0f}%",
            f"  建议: {result.advice}",
            "",
            f"  周线均线:",
            f"    MA5  = {result.ma5:.2f}",
            f"    MA10 = {result.ma10:.2f}",
            f"    MA20 = {result.ma20:.2f}",
            f"  日线MACD: {result.macd_signal or 'N/A'}",
            f"  {result.detail}",
            "=" * 50,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    regime = SimpleRegime()
    result = regime.judge()
    print(regime.format_report(result))
