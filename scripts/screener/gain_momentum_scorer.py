#!/usr/bin/env python3
"""增幅/动量多维评分系统

三维模型：弹性(40%) + 动量(35%) + 回撤(25%)
每个维度内部用 4 个周期（5d/10d/20d/60d）加权融合

设计文档: memory/trend-algorithm/05-gain-momentum-scoring.md
实施文档: memory/trend-algorithm/07-gain-momentum-implementation.md

用法:
    from scripts.screener.gain_momentum_scorer import GainMomentumScorer
    scorer = GainMomentumScorer(client)
    result = scorer.score("603881")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class GainMomentumResult:
    """增幅/动量评分结果"""
    symbol: str
    name: str = ""

    # 总分
    total_score: float = 0.0       # [0, 100]
    grade: str = ""                 # A+/A/B+/B/B-/C

    # 三维分数
    elasticity_score: float = 0.0   # [0, 100]
    momentum_score: float = 0.0     # [0, 100]
    drawdown_score: float = 0.0     # [0, 100]

    # 分维度详情
    elasticity_detail: dict = field(default_factory=dict)
    momentum_detail: dict = field(default_factory=dict)
    drawdown_detail: dict = field(default_factory=dict)

    # 交叉判断
    cross_signal: str = ""

    # 原始数据
    raw_returns: dict = field(default_factory=dict)
    raw_drawdowns: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "gm_total": round(self.total_score, 1),
            "gm_grade": self.grade,
            "gm_elasticity": round(self.elasticity_score, 1),
            "gm_momentum": round(self.momentum_score, 1),
            "gm_drawdown": round(self.drawdown_score, 1),
            "gm_cross": self.cross_signal,
            "gm_elasticity_detail": self.elasticity_detail,
            "gm_momentum_detail": self.momentum_detail,
            "gm_drawdown_detail": self.drawdown_detail,
        }


class GainMomentumScorer:
    """增幅/动量多维评分系统"""

    # 三维权重
    W_ELASTICITY = 0.40
    W_MOMENTUM = 0.35
    W_DRAWDOWN = 0.25

    # 弹性子权重（短→长，中期更重）
    ELASTICITY_WEIGHTS = {5: 0.20, 10: 0.25, 20: 0.30, 60: 0.25}

    # 动量子权重（长期更重，学术验证中期动量最持续）
    MOMENTUM_WEIGHTS = {5: 0.15, 10: 0.20, 20: 0.30, 60: 0.35}

    # 回撤子权重（短期更重，反映近期风险）
    DRAWDOWN_WEIGHTS = {5: 0.35, 10: 0.25, 20: 0.25, 60: 0.15}

    # 标准化参数
    ELASTICITY_DECAY = 30.0     # sigmoid 衰减系数
    DRAWDOWN_DECAY = 20.0       # exp 衰减系数
    VOL_FLOOR = 5.0             # 波动率下限（防除零）

    def __init__(self, client=None):
        self.client = client

    def score(
        self,
        symbol: str,
        trade_date: Optional[str] = None,
        lookback_days: int = 150,
    ) -> GainMomentumResult:
        """主评分入口

        Args:
            symbol: 6位股票代码
            trade_date: 截止日期 YYYYMMDD
            lookback_days: 回溯天数（默认150，保证60日计算有余量）

        Returns:
            GainMomentumResult
        """
        result = GainMomentumResult(symbol=symbol)

        if self.client is None:
            result.cross_signal = "错误: 无数据客户端"
            return result

        # 确定日期范围
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")
        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)
        ).strftime("%Y%m%d")

        # 获取日线数据
        try:
            df = self.client.get_kline(
                symbol, freq="D", start=start_date, end=trade_date, adj="qfq"
            )
        except Exception as e:
            result.cross_signal = f"数据获取失败: {e}"
            return result

        if df is None or len(df) < 61:
            result.cross_signal = "数据不足（需至少61个交易日）"
            return result

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n = len(close)

        # ── 三维评分 ──
        e_score, e_detail = self._score_elasticity(close, low, n)
        m_score, m_detail = self._score_momentum(close, n)
        d_score, d_detail = self._score_drawdown(close, high, n)

        # ── 基础总分 ──
        base_score = (
            e_score * self.W_ELASTICITY
            + m_score * self.W_MOMENTUM
            + d_score * self.W_DRAWDOWN
        )

        # ── 交叉验证调整 ──
        cross_signal = ""

        # 高弹性 + 低动量 = 暴跌后反弹 → 打折
        if e_score > 60 and m_score < 30:
            base_score *= 0.75
            cross_signal = "暴跌反弹（弹性高但动量差）"

        # 三维共振强势 → 加分
        elif e_score > 60 and m_score > 60 and d_score > 70:
            base_score = min(100, base_score * 1.10)
            cross_signal = "强势共振"

        # 回撤过大 → 额外扣分
        if d_score < 30:
            base_score *= 0.85
            if cross_signal:
                cross_signal += " + "
            cross_signal += "回撤过大"

        if not cross_signal:
            cross_signal = "正常"

        # 最终评分
        final_score = max(0, min(100, base_score))

        # ── 原始数据 ──
        raw_returns = {}
        raw_drawdowns = {}
        for period in [5, 10, 20, 60]:
            idx = -(period + 1)
            if n > period:
                raw_returns[f"{period}d"] = round(
                    (close[-1] / close[idx] - 1) * 100, 2
                )
                raw_drawdowns[f"{period}d"] = round(
                    (1 - close[-1] / high[idx + 1:].max()) * 100, 2
                )

        # ── 填充结果 ──
        result.total_score = final_score
        result.grade = self._to_grade(final_score)
        result.elasticity_score = e_score
        result.momentum_score = m_score
        result.drawdown_score = d_score
        result.elasticity_detail = e_detail
        result.momentum_detail = m_detail
        result.drawdown_detail = d_detail
        result.cross_signal = cross_signal
        result.raw_returns = raw_returns
        result.raw_drawdowns = raw_drawdowns

        return result

    # ── 三维评分方法 ──

    def _score_elasticity(
        self, close: np.ndarray, low: np.ndarray, n: int
    ) -> tuple[float, dict]:
        """弹性评分：从区间低点反弹的能力

        Returns: (score 0-100, detail dict)
        """
        c = float(close[-1])
        detail = {}
        weighted_sum = 0.0

        for period, weight in self.ELASTICITY_WEIGHTS.items():
            if n > period:
                low_in_period = float(low[-(period + 1):].min())
                if low_in_period > 0:
                    elasticity_pct = (c / low_in_period - 1) * 100
                else:
                    elasticity_pct = 0.0
                detail[f"{period}d"] = round(elasticity_pct, 2)
                weighted_sum += elasticity_pct * weight

        # sigmoid 映射到 [0, 100]
        # 30% 弹性→~63分, 50%→~82分, 100%→~96分
        score = 100 * (1 - math.exp(-weighted_sum / self.ELASTICITY_DECAY))

        return max(0, min(100, score)), detail

    def _score_momentum(
        self, close: np.ndarray, n: int
    ) -> tuple[float, dict]:
        """动量评分：风险调整后的趋势方向和速度

        Returns: (score 0-100, detail dict)
        """
        c = float(close[-1])
        detail = {}
        weighted_sum = 0.0

        for period, weight in self.MOMENTUM_WEIGHTS.items():
            if n > period:
                idx = -(period + 1)
                # 原始收益
                raw_return = (c / float(close[idx]) - 1) * 100

                # 区间日收益波动率
                period_close = close[idx + 1:]
                daily_returns = np.diff(period_close) / period_close[:-1]
                volatility = float(np.std(daily_returns)) * math.sqrt(252) * 100
                volatility = max(volatility, self.VOL_FLOOR)

                # 风险调整后动量
                risk_adj = raw_return / volatility

                detail[f"{period}d"] = {
                    "return": round(raw_return, 2),
                    "volatility": round(volatility, 2),
                    "risk_adj": round(risk_adj, 3),
                }
                weighted_sum += risk_adj * weight

        # tanh 映射到 [0, 100]
        # risk_adj 典型范围 [-3, +3]
        normalized = (math.tanh(weighted_sum) + 1) / 2 * 100

        return max(0, min(100, normalized)), detail

    def _score_drawdown(
        self, close: np.ndarray, high: np.ndarray, n: int
    ) -> tuple[float, dict]:
        """回撤评分：距区间高点的距离（越少回撤越好）

        Returns: (score 0-100, detail dict)
        """
        c = float(close[-1])
        detail = {}
        weighted_sum = 0.0

        for period, weight in self.DRAWDOWN_WEIGHTS.items():
            if n > period:
                idx = -(period + 1)
                high_in_period = float(high[idx + 1:].max())
                if high_in_period > 0:
                    drawdown_pct = (1 - c / high_in_period) * 100
                else:
                    drawdown_pct = 0.0
                detail[f"{period}d"] = round(drawdown_pct, 2)
                weighted_sum += drawdown_pct * weight

        # exp 映射：回撤越小分越高
        # 0%→100, 10%→61, 20%→37, 30%→22
        score = 100 * math.exp(-weighted_sum / self.DRAWDOWN_DECAY)

        return max(0, min(100, score)), detail

    # ── 工具方法 ──

    @staticmethod
    def _to_grade(score: float) -> str:
        """分数 → 评级"""
        if score >= 85:
            return "A+"
        elif score >= 70:
            return "A"
        elif score >= 55:
            return "B+"
        elif score >= 40:
            return "B"
        elif score >= 25:
            return "B-"
        else:
            return "C"
