#!/usr/bin/env python3
"""环境自适应策略顾问。

根据大盘环境（多头/震荡/空头）自动推荐：
  - 最适合的策略模板
  - 因子权重调整
  - 仓位上限
  - 持有周期建议
"""
from __future__ import annotations

from typing import Optional


# ── 环境对应的策略配置 ──

REGIME_CONFIGS = {
    "risk_on": {
        "label": "多头🟢",
        "description": "大盘处于上升趋势，进攻为主",
        "recommended_templates": ["trend_scan", "wave3_startup", "fusion"],
        "factor_weights": {
            "momentum": 0.40, "growth": 0.25, "quality": 0.10,
            "value": 0.10, "theme": 0.15,
        },
        "max_position_per_stock": 0.20,   # 单仓上限20%
        "max_total_exposure": 0.85,       # 总仓位上限85%
        "preferred_hold_days": "5-15",    # 短中线
        "stop_loss_pct": -0.08,           # 止损-8%
        "tips": [
            "进攻为主，趋势股优先",
            "可以适当提高单仓到20%",
            "关注突破新高的标的",
        ],
    },
    "neutral": {
        "label": "震荡🟡",
        "description": "大盘区间震荡，精选个股",
        "recommended_templates": ["chanlun_t3", "bottom_reversal", "fusion"],
        "factor_weights": {
            "value": 0.25, "quality": 0.25, "growth": 0.20,
            "momentum": 0.15, "theme": 0.15,
        },
        "max_position_per_stock": 0.15,
        "max_total_exposure": 0.70,
        "preferred_hold_days": "10-20",
        "stop_loss_pct": -0.05,
        "tips": [
            "精选个股，不追涨",
            "三买点和底部反转更适合",
            "单仓控制在15%以内",
        ],
    },
    "risk_off": {
        "label": "空头🔴",
        "description": "大盘下跌趋势，防守为主",
        "recommended_templates": ["bottom_reversal"],
        "factor_weights": {
            "quality": 0.35, "value": 0.30, "growth": 0.10,
            "momentum": 0.10, "theme": 0.15,
        },
        "max_position_per_stock": 0.10,
        "max_total_exposure": 0.50,
        "preferred_hold_days": "3-5",
        "stop_loss_pct": -0.03,
        "tips": [
            "⚠️ 防守为主，控制仓位≤50%",
            "只做超跌反弹，快进快出",
            "止损收紧到-3%",
            "保持现金≥50%",
        ],
    },
}


class RegimeAdvisor:
    """环境自适应策略顾问"""

    @staticmethod
    def get_config(regime: str) -> dict:
        """根据环境获取策略配置"""
        return REGIME_CONFIGS.get(regime, REGIME_CONFIGS["neutral"])

    @staticmethod
    def advise(regime_info: dict) -> str:
        """生成环境自适应建议报告"""
        regime = regime_info.get("regime", "neutral")
        score = regime_info.get("score", 0)
        confidence = regime_info.get("confidence", 0)
        config = REGIME_CONFIGS.get(regime, REGIME_CONFIGS["neutral"])

        lines = []
        lines.append("─" * 50)
        lines.append(f"  🎯 环境自适应策略建议")
        lines.append(f"  环境: {config['label']} ({regime}) | "
                      f"环境分{score:+.2f} | 置信度{confidence:.0%}")
        lines.append(f"  {config['description']}")
        lines.append("─" * 50)
        lines.append("")
        lines.append(f"  推荐模板: {', '.join(config['recommended_templates'])}")
        lines.append(f"  持有周期: {config['preferred_hold_days']}天")
        lines.append(f"  单仓上限: {config['max_position_per_stock']*100:.0f}%")
        lines.append(f"  总仓上限: {config['max_total_exposure']*100:.0f}%")
        lines.append(f"  止损线: {config['stop_loss_pct']*100:.0f}%")
        lines.append("")
        lines.append("  因子权重:")
        for factor, weight in config["factor_weights"].items():
            bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
            lines.append(f"    {factor:10} {bar} {weight*100:.0f}%")
        lines.append("")
        lines.append("  操作建议:")
        for tip in config["tips"]:
            lines.append(f"    • {tip}")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def get_weights_for_regime(regime: str) -> dict:
        """获取当前环境下的因子权重（用于多因子评分）"""
        config = REGIME_CONFIGS.get(regime, REGIME_CONFIGS["neutral"])
        return config["factor_weights"]
