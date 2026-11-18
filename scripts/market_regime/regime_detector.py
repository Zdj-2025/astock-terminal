#!/usr/bin/env python3
"""大盘环境检测器 — 多因子阈值对齐法。

4层因子加权汇总，输出环境分(-1.0~+1.0)和regime标签。
定位：软参考，不硬性拦截选股，用于报告提示和仓位建议。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from scripts.market_regime.data_fetcher import MarketDataFetcher


class MarketRegimeDetector:
    """大盘环境检测器"""

    # 因子权重
    WEIGHTS = {
        "domestic_indices": 0.35,
        "breadth": 0.25,
        "external": 0.25,
        "capital_flow": 0.15,
    }

    # A股指数配置
    INDICES = {
        "sh": {"ts_code": "000001.SH", "name": "上证指数"},
        "sz": {"ts_code": "399001.SZ", "name": "深证成指"},
        "cyb": {"ts_code": "399006.SZ", "name": "创业板指"},
    }

    def __init__(self):
        self.fetcher = MarketDataFetcher()

    def detect(self, as_of_date: Optional[str] = None) -> dict:
        """执行大盘环境检测。

        Args:
            as_of_date: 参考日期YYYYMMDD，None=今天

        Returns:
            {
                "regime": "risk_on" / "neutral" / "risk_off",
                "score": float (-1.0 ~ +1.0),
                "confidence": float (0~1),
                "factors": {...},
                "as_of_date": str,
            }
        """
        factors = {}

        # Layer 1: 国内指数趋势
        factors["domestic_indices"] = self._score_domestic_indices()

        # Layer 2: 市场宽度
        factors["breadth"] = self._score_breadth()

        # Layer 3: 外部环境
        factors["external"] = self._score_external()

        # Layer 4: 资金面
        factors["capital_flow"] = self._score_capital_flow()

        # 加权汇总
        total_score = 0.0
        factor_count = 0
        for layer, weight in self.WEIGHTS.items():
            s = factors[layer].get("score", 0)
            total_score += s * weight
            factor_count += 1

        total_score = round(total_score, 3)

        # 置信度：因子方向一致性
        scores = [factors[k].get("score", 0) for k in self.WEIGHTS]
        non_zero = [s for s in scores if s != 0]
        if len(non_zero) >= 2:
            positive = sum(1 for s in non_zero if s > 0)
            negative = sum(1 for s in non_zero if s < 0)
            confidence = round(max(positive, negative) / len(non_zero), 2)
        elif len(non_zero) == 1:
            confidence = 0.5
        else:
            confidence = 0.3

        # Regime标签
        if total_score >= 0.5:
            regime = "risk_on"
        elif total_score <= -0.5:
            regime = "risk_off"
        else:
            regime = "neutral"

        return {
            "regime": regime,
            "score": total_score,
            "confidence": confidence,
            "factors": factors,
            "as_of_date": as_of_date or pd.Timestamp.now().strftime("%Y%m%d"),
        }

    # ── Layer 1: 国内指数趋势 ──

    def _score_domestic_indices(self) -> dict:
        """三大指数MA20/MA60方向打分"""
        scores = []
        details = []

        for key, cfg in self.INDICES.items():
            df = self.fetcher.get_index_kline(cfg["ts_code"], days=150)
            if df is None or len(df) < 60:
                details.append(f"{cfg['name']}: 数据不足")
                continue

            df = df.sort_values("trade_date").reset_index(drop=True)
            close = df["close"].astype(float)
            score = 0

            # MA20方向
            ma20 = close.rolling(20).mean()
            if len(ma20) >= 5:
                ma20_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100
                if ma20_slope > 0.5:
                    score += 1
                elif ma20_slope < -0.5:
                    score -= 1

            # MA60方向
            ma60 = close.rolling(60).mean()
            if len(ma60) >= 5 and not ma60.iloc[-5:].isna().any():
                ma60_slope = (ma60.iloc[-1] - ma60.iloc[-5]) / ma60.iloc[-5] * 100
                if ma60_slope > 0.3:
                    score += 1
                elif ma60_slope < -0.3:
                    score -= 1

            # 价格站上MA20
            if close.iloc[-1] > ma20.iloc[-1]:
                score += 1
            else:
                score -= 1

            # 归一化到 -1/0/+1
            norm_score = 1 if score >= 2 else (-1 if score <= -2 else 0)
            scores.append(norm_score)
            direction = "↑" if norm_score > 0 else ("↓" if norm_score < 0 else "→")
            details.append(f"{cfg['name']}{direction}")

        avg = round(np.mean(scores), 2) if scores else 0
        return {
            "score": avg,
            "detail": " | ".join(details),
        }

    # ── Layer 2: 市场宽度 ──

    def _score_breadth(self) -> dict:
        """涨跌家数比 + 全市场MA20站上率"""
        score = 0
        details = []

        # 涨跌家数
        ad = self.fetcher.get_advance_decline()
        if ad:
            up = ad.get("上涨", 0)
            down = ad.get("下跌", 0)
            if down > 0:
                ratio = up / down
                if ratio >= 2.0:
                    score += 1
                elif ratio <= 0.5:
                    score -= 1
                details.append(f"涨跌比={ratio:.1f}")
            else:
                details.append(f"涨{up}/跌{down}")

        # 全市场MA20站上率（用指数成分近似）
        ma20_above_rates = []
        for key, cfg in self.INDICES.items():
            df = self.fetcher.get_index_kline(cfg["ts_code"], days=150)
            if df is None or len(df) < 25:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            close = df["close"].astype(float)
            ma20 = close.rolling(20).mean()
            if close.iloc[-1] > ma20.iloc[-1]:
                ma20_above_rates.append(1)
            else:
                ma20_above_rates.append(0)

        if ma20_above_rates:
            above_pct = np.mean(ma20_above_rates)
            if above_pct >= 0.8:
                score += 1
            elif above_pct <= 0.2:
                score -= 1
            details.append(f"指数站上MA20={above_pct:.0%}")

        norm = 1 if score >= 1 else (-1 if score <= -1 else 0)
        return {
            "score": norm,
            "detail": " | ".join(details) if details else "数据不足",
        }

    # ── Layer 3: 外部环境 ──

    def _score_external(self) -> dict:
        """美股趋势 + 美债收益率变化"""
        score = 0
        details = []

        # S&P500趋势
        sp500 = self.fetcher.get_sp500(days=60)
        if sp500 is not None and len(sp500) >= 25:
            close = sp500["close"].astype(float)
            ma20 = close.rolling(20).mean()
            if close.iloc[-1] > ma20.iloc[-1]:
                score += 1
                details.append("S&P500>MA20↑")
            else:
                score -= 1
                details.append("S&P500<MA20↓")

        # 美国10Y国债收益率变化
        bond = self.fetcher.get_us_bond_10y(days=30)
        if bond is not None and len(bond) >= 10:
            us_10y = bond["us_10y"].dropna()
            if len(us_10y) >= 5:
                recent = us_10y.iloc[-1]
                prev = us_10y.iloc[-5]
                change = recent - prev
                # 美债收益率上升→利空A股（资金外流）
                if change > 0.1:
                    score -= 1
                    details.append(f"美债10Y↑{change:.2f}%")
                elif change < -0.1:
                    score += 1
                    details.append(f"美债10Y↓{change:.2f}%")
                else:
                    details.append(f"美债10Y持平")

        norm = 1 if score >= 1 else (-1 if score <= -1 else 0)
        return {
            "score": norm,
            "detail": " | ".join(details) if details else "数据不足",
        }

    # ── Layer 4: 资金面 ──

    def _score_capital_flow(self) -> dict:
        """北向资金 + 融资余额"""
        score = 0
        details = []

        # 北向资金5日净流入
        nb = self.fetcher.get_northbound_flow(days=15)
        if nb is not None and len(nb) >= 5:
            flow_col = "flow" if "flow" in nb.columns else "net_buy"
            if flow_col in nb.columns:
                vals = nb[flow_col].dropna()
                if len(vals) >= 5:
                    recent_5d = vals.tail(5)
                    total_flow = recent_5d.sum()
                    # 简单阈值：5日累计>50亿=积极，<-50亿=消极
                    if total_flow > 50e8:
                        score += 1
                    elif total_flow < -50e8:
                        score -= 1
                    flow_display = total_flow / 1e8
                    details.append(f"北向5日={flow_display:+.0f}亿")

        # 融资余额5日变化
        margin = self.fetcher.get_margin_balance(days=15)
        if margin is not None and len(margin) >= 5:
            bal = margin["margin_balance"].astype(float).dropna()
            if len(bal) >= 5:
                change = (bal.iloc[-1] - bal.iloc[-5]) / bal.iloc[-5] * 100
                if change > 0.5:
                    score += 1
                elif change < -0.5:
                    score -= 1
                details.append(f"融资5日={change:+.2f}%")

        norm = 1 if score >= 1 else (-1 if score <= -1 else 0)
        return {
            "score": norm,
            "detail": " | ".join(details) if details else "数据不足",
        }

    # ── 格式化 ──

    @staticmethod
    def format_report(result: dict) -> str:
        """格式化输出报告"""
        lines = []
        regime = result["regime"]
        score = result["score"]
        confidence = result["confidence"]
        factors = result.get("factors", {})
        date = result.get("as_of_date", "")

        regime_label = {
            "risk_on": "多头环境 🟢",
            "neutral": "震荡环境 🟡",
            "risk_off": "空头环境 🔴",
        }.get(regime, regime)

        lines.append("=" * 50)
        lines.append("  大盘环境检测报告")
        lines.append(f"  日期: {date}")
        lines.append("=" * 50)
        lines.append(f"")
        lines.append(f"  环境: {regime_label}")
        lines.append(f"  环境分: {score:+.2f}  (范围 -1.0 ~ +1.0)")
        lines.append(f"  置信度: {confidence:.0%}")
        lines.append(f"")

        pos_hint = {
            "risk_on": "选股积极，可适当放大仓位",
            "neutral": "正常选股，标准仓位",
            "risk_off": "防守优先，降低仓位或观望",
        }.get(regime, "")
        lines.append(f"  操作建议: {pos_hint}")
        lines.append(f"")

        lines.append("  ── 因子明细 ──")
        layer_names = {
            "domestic_indices": "国内指数",
            "breadth": "市场宽度",
            "external": "外部环境",
            "capital_flow": "资金面",
        }
        for layer, weight in MarketRegimeDetector.WEIGHTS.items():
            info = factors.get(layer, {})
            s = info.get("score", 0)
            d = info.get("detail", "")
            sign = "+" if s > 0 else ("-" if s < 0 else " ")
            bar = "█" * int(abs(s) * 3) if s != 0 else "─"
            lines.append(
                f"  {layer_names.get(layer, layer):6} "
                f"[权重{weight:.0%}] "
                f"{sign}{abs(s):.1f} {bar}  {d}"
            )

        lines.append(f"")
        lines.append("=" * 50)
        return "\n".join(lines)
