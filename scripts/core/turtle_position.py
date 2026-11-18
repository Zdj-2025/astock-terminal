#!/usr/bin/env python3
"""
海龟交易法 — ATR仓位管理模块

核心思想：
  - 价格波动1个ATR，总资金波动1%（1个单位头寸）
  - 单笔最大止损 = 2ATR = 总资金的2%
  - 金字塔加仓：每涨0.5ATR加1单位，最多4单位
  - 4单位全部止损 = 总资金亏损8%

与我们的v3.5系统整合：
  - v3评分决定信号强度 → 映射到1-4个单位
  - ATR决定每单位买多少股
  - 输出：建议买入股数 + 止损价 + 加仓点位

用法:
  from scripts.core.turtle_position import TurtlePosition
  tp = TurtlePosition(total_capital=467100)
  result = tp.calculate(code='600894', price=37.35, atr=1.2, v3_score=7.3, signal_type='uptrend')

作者: Elon (OpenClaw)
日期: 2026-06-07
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurtleResult:
    """ATR仓位计算结果。"""
    # 基础参数
    code: str = ''
    price: float = 0.0           # 当前价格
    atr: float = 0.0             # 20日ATR
    v3_score: float = 0.0        # v3评分

    # 仓位计算
    unit_shares: int = 0         # 1单位 = 多少股
    unit_value: float = 0.0      # 1单位金额
    suggested_units: int = 0     # 建议单位数(1-4)
    suggested_shares: int = 0    # 建议总股数
    suggested_value: float = 0.0 # 建议总金额
    position_pct: float = 0.0    # 占总资金比例

    # 风控
    stop_loss_price: float = 0.0 # 止损价(最新买入价-2ATR)
    max_loss_pct: float = 0.0    # 最大亏损占总资金%
    max_loss_amount: float = 0.0 # 最大亏损金额

    # 加仓
    add_positions: list = field(default_factory=list)  # [加仓价, 加仓股数]

    # 说明
    explanation: str = ''

    def summary(self) -> str:
        """返回可读摘要。"""
        lines = [
            f"🐢 ATR仓位计算 — {self.code}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"当前价格: {self.price:.2f}  |  ATR(20): {self.atr:.3f}",
            f"v3评分: {self.v3_score:.1f}  |  建议单位: {self.suggested_units}个",
            f"",
            f"📊 仓位建议:",
            f"  1单位 = {self.unit_shares}股 ({self.unit_value:,.0f}元)",
            f"  建议买入 = {self.suggested_shares}股 ({self.suggested_value:,.0f}元)",
            f"  占总资金 = {self.position_pct*100:.1f}%",
            f"",
            f"🛡️ 风控:",
            f"  止损价 = {self.stop_loss_price:.2f} (价格 - 2ATR)",
            f"  最大亏损 = {self.max_loss_amount:,.0f}元 ({self.max_loss_pct:.1f}%)",
        ]
        if self.add_positions:
            lines.append(f"")
            lines.append(f"📈 加仓计划(金字塔):")
            for i, (px, shares) in enumerate(self.add_positions, 1):
                lines.append(f"  第{i}次加仓: 价格≥{px:.2f} 时加{shares}股")
        lines.append(f"")
        lines.append(f"💡 {self.explanation}")
        return "\n".join(lines)


class TurtlePosition:
    """
    海龟交易法ATR仓位管理器。

    参数:
        total_capital: 总资金（元）
        risk_per_unit: 每单位风险占总资金比例，默认1%
        max_units: 单品种最大单位数，默认4
        max_position_pct: 单品种最大仓位占总资金比例，默认25%
        lot_size: 最小交易单位（A股=100股）
    """

    def __init__(
        self,
        total_capital: float = 467100,
        risk_per_unit: float = 0.01,
        max_units: int = 4,
        max_position_pct: float = 0.25,
        lot_size: int = 100,
    ):
        self.total_capital = total_capital
        self.risk_per_unit = risk_per_unit
        self.max_units = max_units
        self.max_position_pct = max_position_pct
        self.lot_size = lot_size

    def calculate(
        self,
        code: str,
        price: float,
        atr: float,
        v3_score: float,
        signal_type: str = 'uptrend',
        existing_shares: int = 0,
        existing_avg_cost: float = 0.0,
        mtf_confirm: str = '',
        spring_squeeze: bool = False,
        profit_cushion: float = 0.0,
    ) -> TurtleResult:
        """
        计算ATR仓位建议。

        参数:
            code: 股票代码
            price: 当前价格
            atr: 20日ATR
            v3_score: v3.5评分(0-10)
            signal_type: 信号类型 uptrend/uptrend_confirmed/bottom_reversal
            existing_shares: 已有持仓股数(0=新建仓)
            existing_avg_cost: 已有持仓成本价
            mtf_confirm: MTF确认级别 'strong'/'partial'/'weak'/''
            spring_squeeze: 是否弹簧起涨信号
            profit_cushion: 浮盈比例(0.0=无浮盈, 0.20=浮盈20%)

        返回:
            TurtleResult
        """
        result = TurtleResult(
            code=code, price=price, atr=atr, v3_score=v3_score
        )

        if atr <= 0 or price <= 0:
            result.explanation = "ATR或价格无效"
            return result

        # ── 1. 计算1个单位 = 多少股 ──
        # 海龟公式: unit_shares = (total × risk_per_unit) / ATR
        # 即：持有这么多股，价格波动1ATR，盈亏 = total × 1%
        unit_value = self.total_capital * self.risk_per_unit / atr * price
        raw_shares = unit_value / price  # = total * 1% / ATR
        unit_shares = max(1, int(raw_shares / self.lot_size)) * self.lot_size
        unit_value_actual = unit_shares * price

        # ── 1b. 如果1单位金额就超单仓上限，按上限重算 ──
        max_value = self.total_capital * self.max_position_pct
        if unit_value_actual > max_value:
            capped_shares = int(max_value / price / self.lot_size) * self.lot_size
            if capped_shares < unit_shares:
                unit_shares = max(self.lot_size, capped_shares)
                unit_value_actual = unit_shares * price

        result.unit_shares = unit_shares
        result.unit_value = unit_value_actual

        # ── 2. 根据v3评分和信号类型决定单位数 ──
        # 主升浪/底部反转：评分越高，单位越多
        units = self._score_to_units(v3_score, signal_type)

        # ── 2b. 信号加成 ──
        bonuses = []
        if mtf_confirm == 'strong':
            units += 0.5
            bonuses.append('MTF强确认+0.5')
        if spring_squeeze:
            units += 0.5
            bonuses.append('弹簧起涨+0.5')

        # ── 2c. 利润垫限制 ──
        # 没有利润垫时最多2单位（保守），利润垫>20%才允许满仓
        if profit_cushion < 0.20:
            units = min(units, 2)
            if units == 2:
                bonuses.append('无利润垫限制≤2单位')
        else:
            bonuses.append(f'利润垫{profit_cushion*100:.0f}%放开上限')

        units = int(units)  # 取整

        # ── 3. 已有持仓时减少单位 ──
        if existing_shares > 0:
            existing_units = existing_shares / max(unit_shares, 1)
            remaining = max(0, units - int(existing_units))
            units = min(units, int(existing_units) + remaining)
            # 如果已满仓就不加
            if remaining <= 0:
                units = 0
                result.explanation = f"已有{existing_shares}股({existing_units:.1f}单位)，已达目标仓位"

        result.suggested_units = units

        # ── 4. 计算总仓位 ──
        # 高价股边界：如果1单位就超单仓上限，强制1单位=1手，并放宽上限到30%
        high_price_adjusted = False
        if unit_value_actual > max_value and units >= 1:
            units = 1
            unit_shares = self.lot_size
            unit_value_actual = unit_shares * price
            result.unit_shares = unit_shares
            result.unit_value = unit_value_actual
            max_value = self.total_capital * 0.30  # 高价股放宽到30%
            high_price_adjusted = True
            bonuses.append('高价股限1手(上限放宽30%)')

        new_shares = units * unit_shares
        total_shares = existing_shares + new_shares
        total_value = total_shares * price
        position_pct = total_value / self.total_capital

        # 应用单仓上限（二次保护）
        capped_by_limit = False
        if total_value > max_value:
            allowed_shares = int(max_value / price / self.lot_size) * self.lot_size
            new_shares = max(0, allowed_shares - existing_shares)
            total_shares = existing_shares + new_shares
            total_value = total_shares * price
            position_pct = total_value / self.total_capital
            # 重新计算实际单位数
            actual_units = max(0, new_shares // max(unit_shares, 1)) if unit_shares > 0 else 0
            if actual_units < units:
                capped_by_limit = True
                units = actual_units if actual_units > 0 else 1  # 至少显示1
                result.suggested_units = units

        result.suggested_shares = new_shares
        result.suggested_value = new_shares * price
        result.position_pct = position_pct

        # ── 5. 止损价(2ATR硬止损) ──
        # 如果有加仓，以最新买入价为准
        stop_price = price - 2 * atr
        # 如果已有持仓，取已有成本的更低止损
        if existing_avg_cost > 0:
            stop_from_existing = existing_avg_cost - 2 * atr
            # 用更宽松的（更高的）止损价
            stop_price = max(stop_price, stop_from_existing)

        result.stop_loss_price = round(stop_price, 2)

        # 最大亏损 = 全部头寸 × 2ATR
        total_position_value = total_shares * price
        max_loss = total_shares * 2 * atr
        result.max_loss_amount = round(max_loss, 0)
        result.max_loss_pct = max_loss / self.total_capital * 100

        # ── 6. 金字塔加仓计划 ──
        if units >= 2:
            add_positions = []
            for i in range(1, units):
                add_price = price + i * 0.5 * atr
                add_positions.append((round(add_price, 2), unit_shares))
            result.add_positions = add_positions

        # ── 7. 说明 ──
        if units > 0:
            signal_desc = {
                'uptrend': '🚀主升浪',
                'uptrend_confirmed': '🚀主升浪(强确认)',
                'bottom_reversal': '🔄底部反转',
            }.get(signal_type, signal_type)

            bonus_str = ' | '.join(bonuses) if bonuses else '无加成'
            result.explanation = (
                f"{signal_desc}信号，v3={v3_score:.1f}分，"
                f"建议{units}个单位共{new_shares}股({new_shares * price:,.0f}元)。"
                f"止损{result.stop_loss_price:.2f}，"
                f"最大风险{result.max_loss_pct:.1f}%。"
                f"[{bonus_str}]"
            )
        elif not result.explanation:
            result.explanation = f"v3={v3_score:.1f}分，信号不足，不建议建仓"

        return result

    def _score_to_units(self, v3_score: float, signal_type: str) -> int:
        """
        v3评分 → 建议单位数映射。

        保守策略：底部反转少买，主升浪多买
        """
        # 底部反转：信号不确定性更高，保守
        if signal_type == 'bottom_reversal':
            if v3_score >= 8.0:
                return 3
            elif v3_score >= 7.5:
                return 2
            elif v3_score >= 6.5:
                return 1
            else:
                return 0

        # 主升浪：信号更确定，积极
        if v3_score >= 8.5:
            return 4
        elif v3_score >= 7.5:
            return 3
        elif v3_score >= 6.5:
            return 2
        elif v3_score >= 5.5:
            return 1
        else:
            return 0

    def batch_calculate(
        self,
        candidates: list[dict],
    ) -> list[TurtleResult]:
        """
        批量计算多只股票的仓位建议。

        参数:
            candidates: [{'code':..., 'price':..., 'atr':..., 'v3_score':..., 'signal_type':...,
                          'mtf_confirm':..., 'spring_squeeze':..., 'profit_cushion':...}]

        返回:
            按v3评分降序排列的TurtleResult列表
        """
        results = []

        for c in candidates:
            r = self.calculate(
                code=c['code'],
                price=c['price'],
                atr=c['atr'],
                v3_score=c['v3_score'],
                signal_type=c.get('signal_type', 'uptrend'),
                existing_shares=c.get('existing_shares', 0),
                existing_avg_cost=c.get('existing_avg_cost', 0),
                mtf_confirm=c.get('mtf_confirm', ''),
                spring_squeeze=c.get('spring_squeeze', False),
                profit_cushion=c.get('profit_cushion', 0.0),
            )
            results.append(r)

        # 按v3评分降序
        results.sort(key=lambda x: x.v3_score, reverse=True)
        return results


# ── CLI入口 ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    # 演示：用庄哥的资金计算
    tp = TurtlePosition(total_capital=467100)

    print("=" * 50)
    print("🐢 海龟ATR仓位管理 — 演示")
    print(f"总资金: {tp.total_capital:,.0f}元")
    print(f"单笔风险: {tp.risk_per_unit*100:.0f}%/单位")
    print(f"最大止损: {tp.risk_per_unit*2*100:.0f}%/单位")
    print(f"单仓上限: {tp.max_position_pct*100:.0f}%")
    print("=" * 50)

    # 模拟几只股票
    demo_stocks = [
        {'code': '600894', 'name': '广日股份', 'price': 18.5, 'atr': 0.65, 'v3_score': 7.3, 'signal_type': 'uptrend'},
        {'code': '601686', 'name': '友发集团', 'price': 6.80, 'atr': 0.22, 'v3_score': 8.0, 'signal_type': 'bottom_reversal'},
        {'code': '000727', 'name': '冠捷科技', 'price': 3.50, 'atr': 0.12, 'v3_score': 7.2, 'signal_type': 'uptrend'},
    ]

    for s in demo_stocks:
        print()
        r = tp.calculate(
            code=f"{s['code']} {s['name']}",
            price=s['price'],
            atr=s['atr'],
            v3_score=s['v3_score'],
            signal_type=s['signal_type'],
        )
        print(r.summary())
