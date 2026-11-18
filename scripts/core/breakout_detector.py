#!/usr/bin/env python3
"""
唐奇安通道突破检测器 — 海龟交易法的入场信号组件。

唐奇安通道（Donchian Channel）：
  上轨 = 过去N日最高价（不含当日）
  下轨 = 过去N日最低价（不含当日）

突破信号：
  系统1: 价格突破20日高点 → 短期趋势启动
  系统2: 价格突破55日高点 → 大趋势确认

与V3的融合：
  - V3≥7 + 突破20日高点 → 强买入信号（双确认）
  - V3≥6 + 突破20日高点 → 正常买入
  - V3<6 + 突破 → 仅关注，需更多确认

用法:
    from scripts.core.breakout_detector import BreakoutDetector
    bd = BreakoutDetector()
    result = bd.detect(df, v3_score=7.5, atr=0.65)

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BreakoutType(Enum):
    NONE = "无突破"
    BREAKOUT_20 = "20日突破"       # 系统1
    BREAKOUT_55 = "55日突破"       # 系统2
    BOTH = "20+55双突破"            # 强趋势


class SignalStrength(Enum):
    STRONG_BUY = "强买"     # V3≥7 + 突破
    BUY = "买入"            # V3≥6 + 突破
    WATCH = "关注"          # V3<6 + 突破
    NONE = "无信号"


@dataclass
class BreakoutResult:
    """突破检测结果。"""
    breakout_type: BreakoutType = BreakoutType.NONE
    signal_strength: SignalStrength = SignalStrength.NONE
    upper_20: float = 0.0        # 20日高点
    upper_55: float = 0.0        # 55日高点
    lower_20: float = 0.0        # 20日低点
    current_price: float = 0.0
    atr_at_breakout: float = 0.0 # 突破时的ATR（用于止损）
    distance_to_high: float = 0.0  # 距高点的百分比（负=已突破）
    channel_width_pct: float = 0.0 # 通道宽度占比（波动性参考）
    is_new_high_20: bool = False
    is_new_high_55: bool = False
    reason: str = ""

    def summary(self) -> str:
        strength_emoji = {
            SignalStrength.STRONG_BUY: "🚀",
            SignalStrength.BUY: "📈",
            SignalStrength.WATCH: "👀",
            SignalStrength.NONE: "➖",
        }.get(self.signal_strength, "?")
        lines = [
            f"{strength_emoji} {self.signal_strength.value} — {self.breakout_type.value}",
            f"  20日高/低: {self.upper_20:.2f} / {self.lower_20:.2f}",
            f"  当前价: {self.current_price:.2f}  |  距高点: {self.distance_to_high:+.1f}%",
        ]
        if self.upper_55 > 0:
            lines.append(f"  55日高: {self.upper_55:.2f}  |  通道宽度: {self.channel_width_pct:.1f}%")
        if self.reason:
            lines.append(f"  {self.reason}")
        return "\n".join(lines)


class BreakoutDetector:
    """
    唐奇安通道突破检测器。

    参数:
        short_period: 短期通道周期，默认20
        long_period: 长期通道周期，默认55
        min_volume_ratio: 突破日量比要求，默认1.2
    """

    def __init__(
        self,
        short_period: int = 20,
        long_period: int = 55,
        min_volume_ratio: float = 1.2,
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.min_volume_ratio = min_volume_ratio

    def _get_channel(
        self,
        series: pd.Series,
        period: int,
        exclude_last: bool = True,
    ) -> float:
        """获取N日极值（不含当日）。"""
        if len(series) < period + 1:
            return 0.0
        if exclude_last:
            lookback = series.iloc[-(period + 1):-1]
        else:
            lookback = series.iloc[-period:]
        return round(lookback.max(), 2) if not lookback.empty else 0.0

    def detect(
        self,
        kline_df: pd.DataFrame,
        v3_score: float = 0.0,
        atr: float = 0.0,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> BreakoutResult:
        """
        检测突破信号。

        Args:
            kline_df: K线数据
            v3_score: V3评分
            atr: 当前ATR
            high_col/low_col/close_col/volume_col: 列名

        Returns:
            BreakoutResult
        """
        result = BreakoutResult()

        if kline_df is None or kline_df.empty:
            result.reason = "无K线数据"
            return result

        if len(kline_df) < self.short_period + 1:
            result.reason = f"数据不足{self.short_period}日"
            return result

        highs = kline_df[high_col].astype(float)
        lows = kline_df[low_col].astype(float)
        closes = kline_df[close_col].astype(float)
        current_price = float(closes.iloc[-1])
        result.current_price = current_price

        # ── 1. 计算通道 ──
        upper_20 = self._get_channel(highs, self.short_period)
        lower_20 = self._get_channel(lows, self.short_period)
        result.upper_20 = upper_20
        result.lower_20 = lower_20

        # 55日通道
        if len(kline_df) >= self.long_period + 1:
            upper_55 = self._get_channel(highs, self.long_period)
            result.upper_55 = upper_55
        else:
            upper_55 = 0.0

        # 距高点百分比
        if upper_20 > 0:
            result.distance_to_high = round(
                (current_price - upper_20) / upper_20 * 100, 1
            )

        # 通道宽度
        if upper_20 > 0 and lower_20 > 0:
            result.channel_width_pct = round(
                (upper_20 - lower_20) / current_price * 100, 1
            )

        # ── 2. 判断突破 ──
        is_new_high_20 = current_price > upper_20 and upper_20 > 0
        is_new_high_55 = upper_55 > 0 and current_price > upper_55
        result.is_new_high_20 = is_new_high_20
        result.is_new_high_55 = is_new_high_55

        if is_new_high_20 and is_new_high_55:
            result.breakout_type = BreakoutType.BOTH
        elif is_new_high_55:
            result.breakout_type = BreakoutType.BREAKOUT_55
        elif is_new_high_20:
            result.breakout_type = BreakoutType.BREAKOUT_20
        else:
            result.breakout_type = BreakoutType.NONE

        # ── 3. 量能确认 ──
        vol_confirmed = True
        if volume_col in kline_df.columns and len(kline_df) >= 21:
            vol = kline_df[volume_col].astype(float)
            today_vol = vol.iloc[-1]
            avg_vol = vol.iloc[-21:-1].mean()
            if avg_vol > 0:
                vol_ratio = today_vol / avg_vol
                vol_confirmed = vol_ratio >= self.min_volume_ratio
                result.details = {'vol_ratio': round(vol_ratio, 2)}

        # ── 4. 信号强度 ──
        if result.breakout_type == BreakoutType.NONE:
            result.signal_strength = SignalStrength.NONE
            # 接近突破？
            if upper_20 > 0 and 0 < (upper_20 - current_price) / upper_20 < 0.02:
                result.reason = f"距20日高点{upper_20:.2f}仅{result.distance_to_high:.1f}%，接近突破"
            else:
                result.reason = f"未突破20日高点{upper_20:.2f}"

        elif not vol_confirmed:
            result.signal_strength = SignalStrength.WATCH
            result.reason = (
                f"突破{result.breakout_type.value}但量能不足"
                f"（需>{self.min_volume_ratio}倍），观望"
            )

        elif v3_score >= 7.0:
            result.signal_strength = SignalStrength.STRONG_BUY
            result.reason = (
                f"V3={v3_score:.1f} + {result.breakout_type.value} + 放量 = 强买入信号"
            )
            result.atr_at_breakout = atr

        elif v3_score >= 6.0:
            result.signal_strength = SignalStrength.BUY
            result.reason = (
                f"V3={v3_score:.1f} + {result.breakout_type.value} + 放量 = 买入信号"
            )
            result.atr_at_breakout = atr

        else:
            result.signal_strength = SignalStrength.WATCH
            result.reason = (
                f"{result.breakout_type.value}但V3={v3_score:.1f}偏低，"
                f"需更多确认"
            )

        return result

    def batch_detect(
        self,
        kline_map: dict[str, pd.DataFrame],
        scores: dict[str, dict],
    ) -> list[BreakoutResult]:
        """
        批量检测突破信号。

        Args:
            kline_map: {code: kline_df}
            scores: {code: {'v3_score':..., 'atr':...}}

        Returns:
            按信号强度排序（STRONG_BUY > BUY > WATCH > NONE）
        """
        results = []
        for code, df in kline_map.items():
            info = scores.get(code, {})
            r = self.detect(
                df,
                v3_score=info.get('v3_score', 0),
                atr=info.get('atr', 0),
            )
            r.details['code'] = code
            results.append(r)

        priority = {
            SignalStrength.STRONG_BUY: 0,
            SignalStrength.BUY: 1,
            SignalStrength.WATCH: 2,
            SignalStrength.NONE: 3,
        }
        results.sort(key=lambda x: priority.get(x.signal_strength, 99))
        return results


# ── CLI入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    print("=" * 55)
    print("📊 唐奇安通道突破检测器 — 演示")
    print("=" * 55)

    bd = BreakoutDetector()

    # 模拟数据
    np.random.seed(42)
    n = 80

    # 场景1: 突破20日高点
    prices1 = list(np.concatenate([
        np.random.normal(10, 0.3, 30),  # 横盘
        np.random.normal(10, 0.2, 20),  # 继续横盘
        [10.5, 10.8, 11.2, 11.5, 11.8]  # 突破
    ]))
    df1 = pd.DataFrame({
        'high': [p + 0.2 for p in prices1],
        'low': [p - 0.2 for p in prices1],
        'close': prices1,
        'volume': [1000]*50 + [1500]*4 + [2000],
    })

    print("\n── 场景1: V3=7.5 + 突破20日高点 ──")
    r = bd.detect(df1, v3_score=7.5, atr=0.3)
    print(r.summary())

    print("\n── 场景2: V3=5.5 + 同样突破 ──")
    r = bd.detect(df1, v3_score=5.5, atr=0.3)
    print(r.summary())

    # 场景3: 无突破
    prices3 = list(np.random.normal(10, 0.3, 60))
    df3 = pd.DataFrame({
        'high': [p + 0.2 for p in prices3],
        'low': [p - 0.2 for p in prices3],
        'close': prices3,
        'volume': [1000]*60,
    })
    print("\n── 场景3: V3=7.5 但无突破 ──")
    r = bd.detect(df3, v3_score=7.5, atr=0.3)
    print(r.summary())

    # 场景4: 真实数据
    print("\n── 场景4: 真实数据 ──")
    try:
        from scripts.core.data_store import DataStore
        from scripts.core.atr_calculator import ATRCalculator
        store = DataStore()

        for code in ['002594', '688981', '000001']:
            df = store.get_kline(code, period='D', auto_update=False)
            if df is not None and len(df) >= 21:
                atr = ATRCalculator.calculate(df)
                r = bd.detect(df, v3_score=7.0, atr=atr)
                print(f"\n{code}:")
                print(r.summary())
    except Exception as e:
        print(f"需要本地数据: {e}")
