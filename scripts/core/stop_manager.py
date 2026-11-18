#!/usr/bin/env python3
"""
三重止损管理器 — 海龟2ATR硬止损 + 移动止损 + 信号止损。

取代固定百分比止损，适配不同波动率的个股。

三层止损（取最严格的）：
  1. 硬止损: 最新买入价 - 2×ATR（海龟标准）
  2. 移动止损: 持有期间最高价 - N×ATR（trailing stop）
  3. 信号止损: V3评分<5 或 f''连续<0（逻辑走坏）

用法:
    from scripts.core.stop_manager import StopManager
    sm = StopManager()
    result = sm.check_stop(code='600894', entry_price=18.5, atr=0.65,
                           current_price=19.2, peak_price=20.1,
                           v3_score=6.5, f2_history=[0.01, -0.02, -0.03])

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StopType(Enum):
    """触发止损的类型。"""
    NONE = "正常持有"
    HARD_STOP = "硬止损(2ATR)"        # 海龟硬止损
    TRAILING_STOP = "移动止损"          # trailing stop
    SIGNAL_STOP = "信号止损"            # V3/f''走坏
    PROFIT_TAKE = "移动止盈"            # 跌破N日低点


class Action(Enum):
    """建议操作。"""
    HOLD = "持有"
    WATCH = "关注"        # 接近止损位，提高警惕
    REDUCE = "减仓"        # 部分止盈/止损
    EXIT = "清仓"          # 全部退出


@dataclass
class StopResult:
    """止损检查结果。"""
    action: Action = Action.HOLD
    stop_type: StopType = StopType.NONE
    stop_price: float = 0.0          # 当前应触发的止损价
    current_stop: float = 0.0        # 当前持仓的止损线
    distance_pct: float = 0.0        # 当前价距止损线的百分比
    reason: str = ""                 # 触发原因
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        emoji = {
            Action.HOLD: "✅",
            Action.WATCH: "👀",
            Action.REDUCE: "🟡",
            Action.EXIT: "🔴",
        }.get(self.action, "?")
        lines = [
            f"{emoji} {self.action.value} — {self.stop_type.value}",
            f"  止损线: {self.current_stop:.2f}  |  距离: {self.distance_pct:+.1f}%",
            f"  {self.reason}" if self.reason else "",
        ]
        return "\n".join(l for l in lines if l)


class StopManager:
    """
    三重止损管理器。

    参数:
        hard_atr_mult: 硬止损ATR倍数，默认2.0（海龟标准）
        trail_atr_mult: 移动止损ATR倍数，默认2.0
        signal_threshold: 信号止损的V3评分阈值，默认5.0
        f2_consecutive_days: f''连续负的天数阈值，默认3
        watch_threshold_pct: 距止损线多少%开始预警，默认2%
    """

    def __init__(
        self,
        hard_atr_mult: float = 2.0,
        trail_atr_mult: float = 2.0,
        signal_threshold: float = 5.0,
        f2_consecutive_days: int = 3,
        watch_threshold_pct: float = 2.0,
    ):
        self.hard_atr_mult = hard_atr_mult
        self.trail_atr_mult = trail_atr_mult
        self.signal_threshold = signal_threshold
        self.f2_consecutive_days = f2_consecutive_days
        self.watch_threshold_pct = watch_threshold_pct

    def calculate_stops(
        self,
        entry_price: float,
        atr: float,
        peak_price: float,
        last_add_price: float = 0.0,
        profit_cushion: float = 0.0,
    ) -> dict:
        """计算各层止损价。

        Args:
            entry_price: 首次买入价
            atr: 当前ATR
            peak_price: 持有期间最高价
            last_add_price: 最后一次加仓价（0=没加过仓）
            profit_cushion: 浮盈比例

        Returns:
            {
                'hard_stop': 硬止损价,
                'trailing_stop': 移动止损价,
                'effective_stop': 实际生效的止损价（取最高）,
            }
        """
        if atr <= 0:
            return {'hard_stop': 0, 'trailing_stop': 0, 'effective_stop': 0}

        # 硬止损：以最后一次买入价为准（海龟规则）
        base_price = last_add_price if last_add_price > 0 else entry_price
        hard_stop = base_price - self.hard_atr_mult * atr

        # 移动止损：从最高点回撤 trail_atr_mult × ATR
        trailing_stop = peak_price - self.trail_atr_mult * atr

        # 利润垫分层trailing（改进版：回测验证后的最优配置）
        # 亏损中(<0%)：不用trailing，只用硬止损（避免建仓初期被洗出）
        # 微利(0-10%)：trailing用3ATR（给足空间，等于不启用）
        # 小利(10-30%)：trailing用2.5ATR
        # 厚利(>=30%)：trailing用2ATR（锁利润）
        if profit_cushion < 0:
            # 亏损中：只用硬止损
            trailing_stop = 0  # 0=不生效
        elif profit_cushion < 0.10:
            trailing_stop = peak_price - 3.0 * atr
        elif profit_cushion < 0.30:
            trailing_stop = peak_price - 2.5 * atr
        else:
            trailing_stop = peak_price - 2.0 * atr

        # 生效止损 = 取最高（最严格），但trailing=0时不参与
        if trailing_stop > 0:
            effective = max(hard_stop, trailing_stop)
        else:
            effective = hard_stop

        return {
            'hard_stop': round(hard_stop, 2),
            'trailing_stop': round(trailing_stop, 2),
            'effective_stop': round(effective, 2),
        }

    def check_stop(
        self,
        code: str,
        entry_price: float,
        atr: float,
        current_price: float,
        peak_price: float,
        v3_score: float = 10.0,
        f2_history: list[float] = None,
        f2_current: float = 0.0,
        last_add_price: float = 0.0,
        profit_cushion: float = 0.0,
    ) -> StopResult:
        """
        综合检查止损/止盈状态。

        Args:
            code: 股票代码
            entry_price: 买入价
            atr: 当前ATR
            current_price: 当前价
            peak_price: 持有期间最高价
            v3_score: 当前V3评分
            f2_history: 最近N日的f''值列表（最新在后）
            f2_current: 当前f''值（快捷传入，不用history时用）
            last_add_price: 最后加仓价
            profit_cushion: 浮盈比例

        Returns:
            StopResult
        """
        result = StopResult()
        result.details['code'] = code

        if atr <= 0 or entry_price <= 0:
            result.action = Action.HOLD
            result.reason = "数据不足"
            return result

        # ── 1. 计算各层止损价 ──
        stops = self.calculate_stops(
            entry_price, atr, peak_price,
            last_add_price, profit_cushion
        )
        effective_stop = stops['effective_stop']
        result.current_stop = effective_stop
        result.details.update(stops)

        # 距离百分比
        if current_price > 0:
            result.distance_pct = round(
                (current_price - effective_stop) / current_price * 100, 1
            )

        # ── 2. 检查信号止损（逻辑走坏）──
        # V3评分过低
        if v3_score < self.signal_threshold:
            result.action = Action.REDUCE
            result.stop_type = StopType.SIGNAL_STOP
            result.reason = f"V3评分={v3_score:.1f}<{self.signal_threshold}，逻辑走坏"
            result.stop_price = current_price
            return result

        # f''连续为负
        if f2_history and len(f2_history) >= self.f2_consecutive_days:
            recent = f2_history[-self.f2_consecutive_days:]
            if all(x < 0 for x in recent):
                result.action = Action.REDUCE
                result.stop_type = StopType.SIGNAL_STOP
                result.reason = f"f''连续{self.f2_consecutive_days}日<0，拐点消失"
                result.stop_price = current_price
                return result
        elif f2_current < 0 and v3_score < 6.0:
            result.action = Action.WATCH
            result.stop_type = StopType.SIGNAL_STOP
            result.reason = f"f''<0且V3={v3_score:.1f}偏弱，密切关注"

        # ── 3. 检查硬止损/移动止损 ──
        if current_price <= effective_stop:
            # 判断是硬止损还是移动止损触发
            if current_price <= stops['hard_stop']:
                result.stop_type = StopType.HARD_STOP
                result.reason = f"跌破硬止损{stops['hard_stop']:.2f}(买入价-{self.hard_atr_mult}ATR)"
            else:
                result.stop_type = StopType.TRAILING_STOP
                result.reason = f"跌破移动止损{stops['trailing_stop']:.2f}(最高价-{self.trail_atr_mult}ATR)"

            # 有利润垫时触发移动止损 = 止盈（减仓而非清仓）
            if profit_cushion > 0.15 and result.stop_type == StopType.TRAILING_STOP:
                result.action = Action.REDUCE
                result.reason += f"，有{profit_cushion*100:.0f}%利润垫，减仓保利润"
            else:
                result.action = Action.EXIT
                if profit_cushion < 0:
                    result.reason += f"，亏损{abs(profit_cushion)*100:.0f}%，严格执行"

            result.stop_price = effective_stop
            return result

        # ── 4. 预警区 ──
        if result.distance_pct <= self.watch_threshold_pct:
            result.action = Action.WATCH
            result.reason = (
                f"距止损线仅{result.distance_pct:.1f}%，"
                f"止损价{effective_stop:.2f}，提高警惕"
            )
            return result

        # ── 5. 正常持有 ──
        result.action = Action.HOLD
        result.reason = (
            f"安全余量{result.distance_pct:.1f}%，"
            f"止损线{effective_stop:.2f}，继续持有"
        )
        return result

    def batch_check(
        self,
        positions: list[dict],
    ) -> list[StopResult]:
        """
        批量检查持仓止损状态。

        Args:
            positions: [{'code':..., 'entry_price':..., 'atr':..., 'current_price':...,
                         'peak_price':..., 'v3_score':..., 'profit_cushion':..., ...}]

        Returns:
            按紧急程度排序的StopResult列表（EXIT > REDUCE > WATCH > HOLD）
        """
        results = []
        for p in positions:
            r = self.check_stop(
                code=p['code'],
                entry_price=p['entry_price'],
                atr=p['atr'],
                current_price=p['current_price'],
                peak_price=p['peak_price'],
                v3_score=p.get('v3_score', 10.0),
                f2_history=p.get('f2_history'),
                f2_current=p.get('f2_current', 0.0),
                last_add_price=p.get('last_add_price', 0.0),
                profit_cushion=p.get('profit_cushion', 0.0),
            )
            results.append(r)

        # 按紧急程度排序
        priority = {Action.EXIT: 0, Action.REDUCE: 1, Action.WATCH: 2, Action.HOLD: 3}
        results.sort(key=lambda x: priority.get(x.action, 99))
        return results


# ── CLI入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("🛡️ 三重止损管理器 — 演示")
    print("=" * 55)

    sm = StopManager()

    # 场景1: 正常持有
    print("\n── 场景1: 正常持有 ──")
    r = sm.check_stop(
        code='600894', entry_price=18.5, atr=0.65,
        current_price=19.2, peak_price=20.1,
        v3_score=7.5, profit_cushion=0.04
    )
    print(r.summary())

    # 场景2: 接近止损
    print("\n── 场景2: 接近止损 ──")
    r = sm.check_stop(
        code='600894', entry_price=18.5, atr=0.65,
        current_price=17.5, peak_price=20.1,
        v3_score=6.0, profit_cushion=-0.05
    )
    print(r.summary())

    # 场景3: 触发硬止损
    print("\n── 场景3: 跌破硬止损 ──")
    r = sm.check_stop(
        code='600894', entry_price=18.5, atr=0.65,
        current_price=17.0, peak_price=19.0,
        v3_score=5.5, profit_cushion=-0.08
    )
    print(r.summary())

    # 场景4: 移动止盈（有利润垫，从高点回撤）
    print("\n── 场景4: 移动止盈触发 ──")
    r = sm.check_stop(
        code='600894', entry_price=18.5, atr=0.65,
        current_price=21.0, peak_price=23.5,
        v3_score=7.0, profit_cushion=0.135
    )
    print(r.summary())

    # 场景5: 信号止损（V3走坏）
    print("\n── 场景5: V3评分走坏 ──")
    r = sm.check_stop(
        code='600894', entry_price=18.5, atr=0.65,
        current_price=19.5, peak_price=20.5,
        v3_score=4.0, profit_cushion=0.054
    )
    print(r.summary())

    # 场景6: f''连续负
    print("\n── 场景6: f''连续3日负 ──")
    r = sm.check_stop(
        code='600894', entry_price=18.5, atr=0.65,
        current_price=19.5, peak_price=20.5,
        v3_score=6.5, f2_history=[0.02, -0.01, -0.02, -0.03, -0.04],
        profit_cushion=0.054
    )
    print(r.summary())
