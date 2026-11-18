#!/usr/bin/env python3
"""策略评分引擎 — 按目的组合V3各维度，数据实证版。

基于 2026-05-26 全市场回测(22,927样本, 1800只)：
  - f''(二阶导) IC=+0.436，所有场景的王牌
  - 放量在底部反转和主升浪都优于缩量
  - Alpha042缩量上涨因子无效(IC=-0.003)，不加

策略：
  1. 底部反转: f''拐点 + f'>=0 + 放量 + dh180低位 → 77.3%胜率
  2. 主升浪:   f'>0 + f''>0 + >MA20 + 放量 → 80.9%胜率
  3. 综合评估: 全维度独立输出，不做组合判断

不改动 V3DerivativeScorer 原有接口，本模块是上层策略层。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.screener.v3_derivative_scorer import V3Score, V3DerivativeScorer


@dataclass
class StrategyScore:
    """策略评分结果。"""
    strategy: str = ""           # 策略名
    total: float = 0.0           # 综合分(0-10)
    signal: str = ""             # 信号标签
    match: bool = False          # 是否命中该策略
    dimensions: dict = field(default_factory=dict)  # 各维度原始值
    advice: str = ""             # 操作建议


class StrategyEngine:
    """策略评分引擎 — 按目的组合维度。"""

    def __init__(self, scorer: V3DerivativeScorer):
        self.scorer = scorer

    def score_bottom_reversal(self, v3: V3Score) -> StrategyScore:
        """底部反转策略评分。

        最优组合(回测验证): f''>0 + f'>=0 + dh180<-20 + 放量
        → 77.3%胜率, +10.08%, 盈亏比2.41

        评分:
          f''拐点(0-3) + f'方向(0-2) + 放量加分(0-2) + 位置加分(0-2) + 分钟确认(0-1)
          满分10
        """
        result = StrategyScore(strategy="底部反转")

        f1, f2 = v3.f1, v3.f2
        dh180 = v3.details.get("dh180", 0)
        vol_ratio = v3.details.get("vol_ratio", 1.0)
        min_det = v3.details.get("minute_details", {})

        dims = {
            "f1": f1, "f2": f2, "dh180": dh180,
            "vol_ratio": vol_ratio,
            "accel_confirm": min_det.get("accel_confirm", False) if isinstance(min_det, dict) else False,
        }
        result.dimensions = dims

        score = 0.0

        # 1. f''拐点 (0-3)
        if f2 > 0.05:
            f2_score = 3.0
        elif f2 > 0.02:
            f2_score = 2.5
        elif f2 > 0.01:
            f2_score = 2.0
        elif f2 > 0:
            f2_score = 1.0
        else:
            f2_score = 0.0
        score += f2_score

        # 2. f'方向 (0-2)
        if f1 >= 0.1:
            f1_score = 2.0
        elif f1 >= 0:
            f1_score = 1.5
        elif f1 >= -0.1:
            f1_score = 0.5
        else:
            f1_score = 0.0
        score += f1_score

        # 3. 放量加分 (0-2) — 回测验证放量优于缩量
        if vol_ratio > 1.3:
            vol_score = 2.0
        elif vol_ratio > 1.0:
            vol_score = 1.5
        elif vol_ratio > 0.8:
            vol_score = 0.5
        else:
            vol_score = 0.0  # 缩量不加分
        score += vol_score

        # 4. 位置加分 (0-2) — 越低越好
        if dh180 < -30:
            pos_score = 2.0
        elif dh180 < -20:
            pos_score = 1.5
        elif dh180 < -15:
            pos_score = 1.0
        elif dh180 < -10:
            pos_score = 0.5
        else:
            pos_score = 0.0
        score += pos_score

        # 5. 分钟确认 (0-1)
        min_score = 0.0
        if isinstance(min_det, dict) and min_det.get("accel_confirm"):
            min_score = 1.0
        score += min_score

        result.total = round(score, 1)

        # 判断是否命中
        result.match = (f2 > 0.01) and (f1 >= 0) and (dh180 < -15)

        # 信号和建议
        if score >= 7.5:
            result.signal = "🟢 强反转"
            result.advice = "底部反转信号强烈，f'已转正+放量+低位"
        elif score >= 5.5:
            result.signal = "🟡 反转关注"
            result.advice = "底部拐点出现，等放量确认"
        elif score >= 3.5:
            result.signal = "🟠 早期拐点"
            result.advice = "f''转正但f'仍负或量能不足，继续观察"
        else:
            result.signal = "⚪ 无信号"
            result.advice = "不满足底部反转条件"

        return result

    def score_uptrend(self, v3: V3Score) -> StrategyScore:
        """主升浪策略评分 — 方案F(IC加权f''×1.5)。

        回测(2521主升浪样本, 22927全市场):
          ≥8.0分: 1305样本, 81.8%胜率, +13.31%, 盈亏比3.41
          分组胜率完美单调递增(60.7%→68.1%→74.4%→81.8%)

        评分(IC加权):
          f'趋势(0-3) + f''加速度×1.5(0-3) + 放量(0-2) + MA20(0-0.5) + 位置(0-0.5)
          满分9
        """
        result = StrategyScore(strategy="主升浪")

        f1, f2 = v3.f1, v3.f2
        dh180 = v3.details.get("dh180", 0)
        vol_ratio = v3.details.get("vol_ratio", 1.0)
        above_ma20 = v3.details.get("above_ma20", False)
        min_det = v3.details.get("minute_details", {})

        dims = {
            "f1": f1, "f2": f2, "dh180": dh180,
            "vol_ratio": vol_ratio, "above_ma20": above_ma20,
            "accel_bull": min_det.get("accel_bull", False) if isinstance(min_det, dict) else False,
        }
        result.dimensions = dims

        score = 0.0

        # 1. f'趋势 (0-3)
        if f1 > 0.3:
            f1_score = 3.0
        elif f1 > 0.15:
            f1_score = 2.5
        elif f1 > 0.05:
            f1_score = 2.0
        elif f1 > 0:
            f1_score = 1.0
        else:
            f1_score = 0.0
        score += f1_score

        # 2. f''加速度 ×1.5 (IC加权，f'' IC最高) (0-3)
        if f2 > 0.05:
            f2_score = 3.0  # 2.0 × 1.5
        elif f2 > 0.02:
            f2_score = 2.25  # 1.5 × 1.5
        elif f2 > 0.01:
            f2_score = 1.5  # 1.0 × 1.5
        elif f2 > 0:
            f2_score = 0.75  # 0.5 × 1.5
        else:
            f2_score = 0.0
        score += f2_score

        # 3. 放量 (0-2)
        if vol_ratio > 1.3:
            vol_score = 2.0
        elif vol_ratio > 1.0:
            vol_score = 1.5
        elif vol_ratio > 0.8:
            vol_score = 0.5
        else:
            vol_score = 0.0
        score += vol_score

        # 4. MA20之上 (0-0.5)
        score += 0.5 if above_ma20 else 0.0

        # 5. 位置空间 (0-0.5)
        if dh180 < -10:
            score += 0.5
        elif dh180 < 0:
            score += 0.25

        result.total = round(score, 1)

        # 判断是否命中
        result.match = (f1 > 0) and (f2 > 0.01) and above_ma20

        # 信号和建议
        if score >= 8.0:
            result.signal = "🚀 强主升浪"
            result.advice = "趋势加速+放量+IC加权高分，强势持有"
        elif score >= 6.0:
            result.signal = "📈 主升浪"
            result.advice = "趋势向上，关注量能持续"
        elif score >= 4.0:
            result.signal = "📊 趋势萌芽"
            result.advice = "方向偏多但力度不够，等确认"
        else:
            result.signal = "➡️ 非主升浪"
            result.advice = "不满足主升浪条件"

        return result

    def evaluate_risk(self, v3: V3Score) -> StrategyScore:
        """风险评估（已知股票，多角度诊断）。

        不做综合评分，返回各维度独立判断。
        """
        result = StrategyScore(strategy="风险评估")

        f1, f2 = v3.f1, v3.f2
        dh180 = v3.details.get("dh180", 0)
        ret20 = v3.details.get("ret20_0", 0)
        min_det = v3.details.get("minute_details", {})

        # 趋势健康度
        if f1 > 0 and f2 > 0:
            trend_health = "✅ 趋势加速中"
        elif f1 > 0 and f2 < -0.01:
            trend_health = "⚠️ 涨速放缓"
        elif f1 > 0:
            trend_health = "🟡 上涨但动力减弱"
        elif f1 < -0.1 and f2 < -0.01:
            trend_health = "🔴 加速下跌"
        else:
            trend_health = "⚪ 震荡"

        # 量价配合
        vol_ratio = v3.details.get("vol_ratio", 1.0)
        if f1 > 0 and vol_ratio > 1.0:
            vol_health = "✅ 量价齐升"
        elif f1 > 0 and vol_ratio < 0.8:
            vol_health = "⚠️ 缩量上涨，持续性存疑"
        elif f1 < 0 and vol_ratio > 1.3:
            vol_health = "⚠️ 放量下跌"
        else:
            vol_health = "⚪ 量能正常"

        # 位置安全度
        if dh180 < -20:
            pos_health = "✅ 低位安全"
        elif dh180 < -5:
            pos_health = "🟡 中等位置"
        elif dh180 < 5:
            pos_health = "🟠 接近高位"
        else:
            pos_health = "🔴 高位风险"

        # 卖出信号
        sell_signals = []
        if isinstance(min_det, dict):
            if min_det.get("top_div"):
                sell_signals.append("60min顶背离")
        if f1 > 0 and f2 < -0.01 and ret20 > 5:
            sell_signals.append("减速+高位")
        if f1 < 0 and dh180 > 5:
            sell_signals.append("趋势破位+高位")

        risk_score = 0.0  # 0=最安全, 10=最危险
        if sell_signals:
            risk_score += 4.0
        if dh180 > 10:
            risk_score += 3.0
        if f2 < -0.03:
            risk_score += 2.0
        if vol_ratio < 0.6 and f1 > 0:
            risk_score += 1.0
        risk_score = min(risk_score, 10.0)

        result.total = round(risk_score, 1)
        result.dimensions = {
            "trend_health": trend_health,
            "vol_health": vol_health,
            "pos_health": pos_health,
            "sell_signals": sell_signals,
            "f1": f1, "f2": f2, "dh180": dh180,
        }

        if risk_score >= 7:
            result.signal = "🔴 高风险"
            result.advice = f"多重卖出信号: {', '.join(sell_signals)}，考虑减仓"
        elif risk_score >= 4:
            result.signal = "🟠 风险上升"
            result.advice = "趋势或位置出现隐患，密切关注"
        else:
            result.signal = "🟢 相对安全"
            result.advice = "暂无明确风险信号"

        return result

    def full_evaluation(self, v3: V3Score) -> dict:
        """全维度评估（已知股票的综合诊断）。

        Returns:
            dict with 所有策略评分 + 原始V3维度
        """
        return {
            "v3_original": {
                "total": v3.total,
                "raw": v3.raw_total,
                "penalty": v3.penalty,
                "f1": v3.f1,
                "f2": v3.f2,
                "signal": v3.signal,
                "position": v3.position_score,
                "trend": v3.trend_score,
                "curvature": v3.curvature_score,
                "minute": v3.minute_score,
            },
            "bottom_reversal": self.score_bottom_reversal(v3),
            "uptrend": self.score_uptrend(v3),
            "risk": self.evaluate_risk(v3),
        }
