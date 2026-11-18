#!/usr/bin/env python3
"""铁律v2 执行模块 — 盈利保护 + 趋势加仓

核心功能：
1. 盈利保护：根据盈利幅度动态调整止损线，绝不让盈利变亏损
2. 趋势加仓：在底仓盈利且趋势不变时，分阶段加仓
3. 每日检查：对组合内所有持仓做盈利保护检查 + 加仓机会检测

用法:
  from scripts.analyzers.profit_guard import ProfitGuard
  guard = ProfitGuard()
  result = guard.check_position(holding, tech_data)
  advice = guard.check_portfolio(positions, tech_map, total_capital)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProtectionResult:
    """盈利保护检查结果"""
    code: str
    name: str
    # 当前状态
    cost: float
    price: float
    pnl_pct: float
    position_pct: float  # 占总资金池比例
    # 保护线
    current_sl: float      # 当前止损价
    new_sl: float          # 建议止损价
    sl_moved: bool         # 是否需要移动止损
    sl_reason: str         # 移动原因
    # 保护级别
    protection_level: str  # none/standard/strong/max
    # 加仓建议
    can_add: bool          # 是否可以加仓
    add_max_pct: float     # 可加仓到多少（占总资金池）
    add_reason: str        # 加仓理由


# ── 盈利保护规则（铁律v2） ──
PROFIT_PROTECTION_RULES = [
    {"threshold": 1.00, "sl_above_cost": 0.50, "level": "max",     "label": "成本+50%"},
    {"threshold": 0.50, "sl_above_cost": 0.25, "level": "strong",  "label": "成本+25%"},
    {"threshold": 0.30, "sl_above_cost": 0.10, "level": "standard","label": "成本+10%"},
    {"threshold": 0.15, "sl_above_cost": 0.00, "level": "basic",   "label": "成本价"},
]

# ── 趋势加仓规则（铁律v2） ──
ADD_POSITION_RULES = [
    {"pnl_threshold": 0.20, "max_position": 0.25, "label": "盈利>20%+行业催化+机构流入"},
    {"pnl_threshold": 0.10, "max_position": 0.20, "label": "盈利>10%+放量突破前高"},
    {"pnl_threshold": 0.05, "max_position": 0.15, "label": "盈利>5%+趋势不变"},
]

# ── 止损分档（铁律v2） ──
STOP_LOSS_TIERS = {
    "light":  -0.10,  # 仓位≤10%
    "normal": -0.08,  # 仓位10-20%
    "heavy":  -0.05,  # 仓位>20%
}


class ProfitGuard:
    """铁律v2 盈利保护 + 趋势加仓执行器"""

    def check_position(
        self,
        holding: Dict,
        tech: Optional[Dict] = None,
        total_capital: float = 0,
    ) -> ProtectionResult:
        """对单只持仓做盈利保护检查。

        Args:
            holding: {code, name, cost, price, shares, market_value,
                      buy_date, pnl_pct, position_pct}
            tech: {weekly_trend, daily_macd_hist, daily_ma20_slope,
                   volume_ratio, has_catalyst}
            total_capital: 总资金池金额（用于计算仓位限制）
        """
        code = holding.get("code", "")
        name = holding.get("name", "")
        cost = holding.get("cost", 0)
        price = holding.get("price", 0)
        pnl_pct = holding.get("pnl_pct", 0)
        position_pct = holding.get("position_pct", 0)  # 占总资金池

        # ── 1. 盈利保护止损线 ──
        new_sl = self._calc_profit_protection_sl(cost, pnl_pct)

        # 基础止损（按仓位分档）
        if position_pct > 0.20:
            base_sl_pct = STOP_LOSS_TIERS["heavy"]
        elif position_pct > 0.10:
            base_sl_pct = STOP_LOSS_TIERS["normal"]
        else:
            base_sl_pct = STOP_LOSS_TIERS["light"]
        base_sl = cost * (1 + base_sl_pct)

        # 取保护止损和基础止损的较高者
        new_sl = max(new_sl, base_sl)

        # 当前持仓的止损价
        current_sl = holding.get("stop_loss", base_sl)

        sl_moved = new_sl > current_sl
        protection_level = "none"
        sl_reason = f"基础止损{base_sl_pct*100:.0f}%"

        for rule in PROFIT_PROTECTION_RULES:
            if pnl_pct / 100 >= rule["threshold"]:
                protected_sl = cost * (1 + rule["sl_above_cost"])
                if protected_sl >= new_sl:
                    protection_level = rule["level"]
                    sl_reason = f"盈利{pnl_pct:+.1f}%，保护止损→{rule['label']}"
                    break

        # ── 2. 趋势加仓判断 ──
        can_add = False
        add_max_pct = position_pct  # 默认不加
        add_reason = ""

        if pnl_pct > 0 and tech:
            for rule in ADD_POSITION_RULES:
                if pnl_pct / 100 < rule["pnl_threshold"]:
                    continue
                if position_pct >= rule["max_position"]:
                    continue

                # 趋势确认
                weekly_ok = tech.get("weekly_trend") in ("bullish",)
                macd_ok = tech.get("daily_macd_hist", 0) > 0
                ma20_up = tech.get("daily_ma20_slope", 0) > 0

                if weekly_ok and (macd_ok or ma20_up):
                    can_add = True
                    add_max_pct = rule["max_position"]
                    add_reason = rule["label"]
                    break

        return ProtectionResult(
            code=code,
            name=name,
            cost=cost,
            price=price,
            pnl_pct=pnl_pct,
            position_pct=position_pct,
            current_sl=current_sl,
            new_sl=new_sl,
            sl_moved=sl_moved,
            sl_reason=sl_reason,
            protection_level=protection_level,
            can_add=can_add,
            add_max_pct=add_max_pct,
            add_reason=add_reason,
        )

    def check_portfolio(
        self,
        positions: List[Dict],
        tech_map: Optional[Dict] = None,
        total_capital: float = 0,
    ) -> Dict:
        """批量检查组合盈利保护 + 加仓机会。

        Returns:
            {total, sl_updates, add_opportunities, alerts, details}
        """
        tech_map = tech_map or {}
        results = []
        sl_updates = []
        add_opportunities = []
        alerts = []

        for h in positions:
            code = h.get("code", "")
            tech = tech_map.get(code)
            r = self.check_position(h, tech, total_capital)
            results.append(r)

            # 需要更新止损
            if r.sl_moved:
                sl_updates.append({
                    "code": r.code,
                    "name": r.name,
                    "old_sl": r.current_sl,
                    "new_sl": r.new_sl,
                    "reason": r.sl_reason,
                    "action": f"更新止损价: {r.current_sl:.2f} → {r.new_sl:.2f}",
                })

            # 加仓机会
            if r.can_add:
                add_pct = r.add_max_pct - r.position_pct
                add_amount = add_pct * total_capital if total_capital > 0 else 0
                add_opportunities.append({
                    "code": r.code,
                    "name": r.name,
                    "current_pct": r.position_pct,
                    "target_pct": r.add_max_pct,
                    "add_pct": round(add_pct, 3),
                    "add_amount": round(add_amount, 0),
                    "reason": r.add_reason,
                })

            # 预警
            if r.pnl_pct > 30 and r.protection_level in ("none", "basic"):
                alerts.append({
                    "type": "protection_gap",
                    "code": r.code,
                    "name": r.name,
                    "message": f"盈利{r.pnl_pct:+.1f}%但保护级别仅{r.protection_level}，检查止损线",
                })

            if r.price <= r.new_sl * 1.02 and r.protection_level in ("strong", "max"):
                alerts.append({
                    "type": "near_stop",
                    "code": r.code,
                    "name": r.name,
                    "message": f"距保护止损仅{(r.price/r.new_sl-1)*100:.1f}%，注意执行",
                })

        return {
            "total": len(results),
            "sl_updates": sl_updates,
            "add_opportunities": add_opportunities,
            "alerts": alerts,
            "details": results,
        }

    def format_report(self, portfolio_result: Dict) -> str:
        """格式化盈利保护报告"""
        lines = [
            "🛡️ 铁律v2 盈利保护检查",
            "─" * 50,
        ]

        # 止损更新
        updates = portfolio_result.get("sl_updates", [])
        if updates:
            lines.append(f"\n📐 需要更新止损 ({len(updates)}只):")
            for u in updates:
                lines.append(f"  📌 {u['name']}({u['code']}): {u['action']}")
                lines.append(f"     原因: {u['reason']}")
        else:
            lines.append("\n✅ 所有止损线正常，无需调整")

        # 加仓机会
        adds = portfolio_result.get("add_opportunities", [])
        if adds:
            lines.append(f"\n📈 趋势加仓机会 ({len(adds)}只):")
            for a in adds:
                lines.append(
                    f"  🔼 {a['name']}({a['code']}): "
                    f"{a['current_pct']*100:.1f}% → {a['target_pct']*100:.1f}% "
                    f"(加{a['add_amount']:,.0f}元) | {a['reason']}"
                )

        # 预警
        alerts = portfolio_result.get("alerts", [])
        if alerts:
            lines.append(f"\n⚠️ 预警 ({len(alerts)}条):")
            for a in alerts:
                lines.append(f"  {a['message']}")

        # 每只持仓的保护状态
        lines.append(f"\n📋 保护状态明细:")
        for r in portfolio_result.get("details", []):
            level_icon = {"none": "⬜", "basic": "🟡", "standard": "🟠", "strong": "🟢", "max": "🟢🟢"}
            icon = level_icon.get(r.protection_level, "⬜")
            lines.append(
                f"  {icon} {r.name}({r.code}): "
                f"盈亏{r.pnl_pct:+.1f}% | 止损{r.new_sl:.2f} | "
                f"保护{r.protection_level}"
            )

        return "\n".join(lines)

    # ── 内部方法 ──

    @staticmethod
    def _calc_profit_protection_sl(cost: float, pnl_pct: float) -> float:
        """根据盈利幅度计算保护止损线。"""
        if cost <= 0:
            return 0

        for rule in PROFIT_PROTECTION_RULES:
            if pnl_pct / 100 >= rule["threshold"]:
                return cost * (1 + rule["sl_above_cost"])

        # 盈利<15%，用仓位分档止损
        return 0  # 返回0表示使用基础止损
