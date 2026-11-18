#!/usr/bin/env python3
"""
趋势退出模块 — 海龟移动止盈 + V3信号退出。

解决"什么时候卖"的核心问题。

退出信号（满足任一即触发）：
  1. 价格退出：收盘价跌破N日最低价（唐奇安通道下轨）
     - V3≥8的强势股用20日低点（给趋势更大空间）
     - V3<8用10日低点（较早锁利）
  2. 信号退出：V3评分<5.0 或 f''连续3日<0
  3. 减仓信号：V3在5-6之间 或 价格在N日低点附近(<1ATR)

与stop_manager的区别：
  - stop_manager: 防守型，"价格到线就执行"
  - trend_exit: 进攻型，"趋势结束就退出"

用法:
    from scripts.core.trend_exit import TrendExit
    te = TrendExit()
    result = te.check_exit(code='600894', kline_df=df,
                           v3_score=7.5, entry_price=18.5,
                           current_price=19.2, atr=0.65)

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExitAction(Enum):
    HOLD = "持有"
    WATCH = "关注"       # 接近退出条件
    REDUCE = "减仓"       # 部分退出
    EXIT = "退出"         # 全部退出


@dataclass
class ExitResult:
    """退出信号检查结果。"""
    action: ExitAction = ExitAction.HOLD
    reason: str = ""
    channel_low: float = 0.0       # N日最低价（退出线）
    channel_period: int = 0        # 使用的通道周期
    distance_to_low: float = 0.0   # 当前价距退出线的ATR倍数
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        emoji = {
            ExitAction.HOLD: "✅",
            ExitAction.WATCH: "👀",
            ExitAction.REDUCE: "🟡",
            ExitAction.EXIT: "🔴",
        }.get(self.action, "?")
        lines = [
            f"{emoji} {self.action.value} — {self.reason}" if self.reason else f"{emoji} {self.action.value}",
        ]
        if self.channel_low > 0:
            lines.append(f"  {self.channel_period}日最低: {self.channel_low:.2f}  |  距离: {self.distance_to_low:.1f}ATR")
        return "\n".join(lines)


class TrendExit:
    """
    趋势退出管理器。

    参数:
        strong_channel: 强势股通道周期，默认20
        normal_channel: 普通股通道周期，默认10
        strong_threshold: 强势股V3阈值，默认8.0
        signal_exit_threshold: 信号退出V3阈值，默认5.0
        f2_negative_days: f''连续负天数，默认3
    """

    def __init__(
        self,
        strong_channel: int = 20,
        normal_channel: int = 10,
        strong_threshold: float = 8.0,
        signal_exit_threshold: float = 5.0,
        f2_negative_days: int = 3,
    ):
        self.strong_channel = strong_channel
        self.normal_channel = normal_channel
        self.strong_threshold = strong_threshold
        self.signal_exit_threshold = signal_exit_threshold
        self.f2_negative_days = f2_negative_days

    def get_donchian_low(
        self,
        kline_df: pd.DataFrame,
        period: int,
        low_col: str = "low",
    ) -> float:
        """获取N日最低价（唐奇安通道下轨）。

        注意：不包含当日，用过去N日的最低价。
        """
        if kline_df is None or len(kline_df) < period + 1:
            return 0.0

        lows = kline_df[low_col].astype(float)
        # 取最近 period+1 根，排除最后一根（当日）
        lookback = lows.iloc[-(period + 1):-1]
        if len(lookback) < period:
            return 0.0
        return round(lookback.min(), 2)

    def check_exit(
        self,
        code: str,
        kline_df: pd.DataFrame,
        v3_score: float,
        atr: float,
        current_price: float = 0.0,
        entry_price: float = 0.0,
        f2_history: list[float] = None,
        profit_cushion: float = 0.0,
        low_col: str = "low",
        close_col: str = "close",
    ) -> ExitResult:
        """
        检查退出信号。

        Args:
            code: 股票代码
            kline_df: 日线K线数据
            v3_score: 当前V3评分
            atr: 当前ATR
            current_price: 当前价（0=用K线最新收盘）
            entry_price: 买入价
            f2_history: 最近f''值列表
            profit_cushion: 浮盈比例
            low_col/close_col: K线列名

        Returns:
            ExitResult
        """
        result = ExitResult()
        result.details['code'] = code

        if kline_df is None or kline_df.empty:
            result.reason = "无K线数据"
            return result

        # 确定当前价
        if current_price <= 0:
            current_price = float(kline_df[close_col].iloc[-1])

        # ── 1. 确定使用哪个通道周期 ──
        # 利润垫分层选择通道周期（改进版）
        # 亏损中(<0%)：用20日低点（给足反弹空间，不因正常波动退出）
        # 微利(0-10%)：用20日低点
        # 小利(10-30%)：用10日低点
        # 厚利(>=30%)：用10日低点（积极锁利）
        # V3<6 且 利润垫>10%：也用10日（弱势锁利）
        if profit_cushion < 0.10:
            channel_period = self.strong_channel  # 20日
        elif profit_cushion < 0.30 and v3_score >= self.strong_threshold:
            channel_period = self.strong_channel  # 20日（强势股给空间）
        else:
            channel_period = self.normal_channel  # 10日
        result.channel_period = channel_period

        # ── 2. 获取通道下轨 ──
        channel_low = self.get_donchian_low(kline_df, channel_period, low_col)
        result.channel_low = channel_low

        if channel_low <= 0:
            result.reason = f"数据不足{channel_period}日"
            return result

        # 距退出线的ATR倍数
        if atr > 0:
            result.distance_to_low = round(
                (current_price - channel_low) / atr, 1
            )

        # ── 3. 检查信号退出 ──
        if v3_score < self.signal_exit_threshold:
            result.action = ExitAction.EXIT
            result.reason = f"V3={v3_score:.1f}<{self.signal_exit_threshold}，趋势逻辑走坏"
            return result

        if f2_history and len(f2_history) >= self.f2_negative_days:
            recent = f2_history[-self.f2_negative_days:]
            if all(x < 0 for x in recent):
                result.action = ExitAction.REDUCE
                result.reason = f"f''连续{self.f2_negative_days}日<0，拐点确认消失"
                return result

        # ── 4. 检查价格退出 ──
        if current_price <= channel_low:
            result.action = ExitAction.EXIT
            result.reason = (
                f"收盘价{current_price:.2f}跌破{channel_period}日最低{channel_low:.2f}，"
                f"趋势结束"
            )
            if profit_cushion > 0:
                result.reason += f"（{profit_cushion*100:.1f}%利润垫，移动止盈）"
            return result

        # ── 5. 减仓预警 ──
        # 距退出线 < 1ATR
        if atr > 0 and (current_price - channel_low) < atr:
            # V3在5-6之间 + 接近退出线 = 减仓
            if v3_score < 6.0:
                result.action = ExitAction.REDUCE
                result.reason = (
                    f"V3={v3_score:.1f}偏弱 + 距{channel_period}日低仅"
                    f"{result.distance_to_low:.1f}ATR，减仓避险"
                )
            else:
                result.action = ExitAction.WATCH
                result.reason = (
                    f"距{channel_period}日低{channel_low:.2f}仅"
                    f"{result.distance_to_low:.1f}ATR，密切关注"
                )
            return result

        # ── 6. V3中间区域 ──
        if 5.0 <= v3_score < 6.0:
            result.action = ExitAction.WATCH
            result.reason = (
                f"V3={v3_score:.1f}处于弱势区域，"
                f"退出线{channel_low:.2f}({channel_period}日低)"
            )
            return result

        # ── 7. 正常持有 ──
        result.action = ExitAction.HOLD
        result.reason = (
            f"V3={v3_score:.1f}，退出线{channel_low:.2f}"
            f"({channel_period}日低)，安全距离{result.distance_to_low:.1f}ATR"
        )
        return result

    def batch_check(
        self,
        positions: list[dict],
        kline_map: dict[str, pd.DataFrame],
    ) -> list[ExitResult]:
        """
        批量检查持仓退出信号。

        Args:
            positions: [{'code':..., 'v3_score':..., 'atr':..., ...}]
            kline_map: {code: kline_df}

        Returns:
            按紧急程度排序
        """
        results = []
        for p in positions:
            code = p['code']
            df = kline_map.get(code)
            if df is None:
                results.append(ExitResult(reason=f"{code}无K线数据"))
                continue

            r = self.check_exit(
                code=code,
                kline_df=df,
                v3_score=p.get('v3_score', 10.0),
                atr=p.get('atr', 0),
                current_price=p.get('current_price', 0),
                entry_price=p.get('entry_price', 0),
                f2_history=p.get('f2_history'),
                profit_cushion=p.get('profit_cushion', 0),
            )
            results.append(r)

        priority = {ExitAction.EXIT: 0, ExitAction.REDUCE: 1,
                    ExitAction.WATCH: 2, ExitAction.HOLD: 3}
        results.sort(key=lambda x: priority.get(x.action, 99))
        return results


# ── CLI入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("📤 趋势退出模块 — 演示")
    print("=" * 55)

    # 模拟K线数据
    np.random.seed(42)
    n = 60

    # 场景A: 上升趋势中的正常回调
    prices_a = np.cumsum(np.random.normal(0.1, 0.5, n)) + 18
    df_a = pd.DataFrame({
        'high': prices_a + np.abs(np.random.normal(0, 0.3, n)),
        'low': prices_a - np.abs(np.random.normal(0, 0.3, n)),
        'close': prices_a,
    })

    te = TrendExit()

    print("\n── 场景A: 强势股正常持有 ──")
    r = te.check_exit(code='A', kline_df=df_a, v3_score=8.0, atr=0.65,
                      current_price=prices_a[-1], profit_cushion=0.15)
    print(r.summary())

    print("\n── 场景B: 弱势股减仓预警 ──")
    r = te.check_exit(code='B', kline_df=df_a, v3_score=5.5, atr=0.65,
                      current_price=prices_a[-1], profit_cushion=0.02)
    print(r.summary())

    # 场景C: 跌破10日低点
    df_c = df_a.copy()
    df_c.loc[df_c.index[-1], 'close'] = df_c['low'].iloc[-11:-1].min() - 0.1
    print("\n── 场景C: 跌破10日低点 ──")
    r = te.check_exit(code='C', kline_df=df_c, v3_score=6.5, atr=0.65,
                      current_price=df_c['close'].iloc[-1], profit_cushion=0.03)
    print(r.summary())

    # 场景D: V3走坏
    print("\n── 场景D: V3评分走坏 ──")
    r = te.check_exit(code='D', kline_df=df_a, v3_score=4.0, atr=0.65,
                      current_price=prices_a[-1], profit_cushion=0.01)
    print(r.summary())

    # 场景E: f''连续负
    print("\n── 场景E: f''连续3日负 ──")
    r = te.check_exit(code='E', kline_df=df_a, v3_score=6.5, atr=0.65,
                      current_price=prices_a[-1],
                      f2_history=[0.02, 0.01, -0.01, -0.02, -0.03],
                      profit_cushion=0.05)
    print(r.summary())

    # 场景F: 用真实数据测试
    print("\n── 场景F: 真实数据(平安银行) ──")
    try:
        import sys
        sys.path.insert(0, '.')
        from scripts.core.data_store import DataStore
        store = DataStore()
        df = store.get_kline('000001', period='D', auto_update=False)
        if df is not None and not df.empty:
            last_close = float(df['close'].iloc[-1])
            from scripts.core.atr_calculator import ATRCalculator
            atr = ATRCalculator.calculate(df)
            r = te.check_exit(code='000001', kline_df=df, v3_score=7.0,
                              atr=atr, profit_cushion=0.05)
            print(r.summary())
            print(f"  (最新收盘: {last_close:.2f}, ATR: {atr:.3f})")
    except Exception as e:
        print(f"  需要本地数据: {e}")
