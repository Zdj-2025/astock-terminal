#!/usr/bin/env python3
"""斐波那契回撤/扩展计算器。

根据最近一段行情的极值点，计算关键回撤位和扩展位。
用于判断调整/反弹的目标价。

用法：
  from scripts.analyzers.fibonacci_calculator import FibonacciCalculator

  calc = FibonacciCalculator()
  levels = calc.calc_retracement(df_daily)
  ext = calc.calc_extension(df_daily)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FibLevel:
    """斐波那契价位。"""
    ratio: float          # 回撤比例（如 0.382）
    price: float          # 对应价格
    label: str            # 说明
    direction: str        # "support" / "resistance"


@dataclass
class FibResult:
    """斐波那契计算结果。"""
    trend: str            # "up" / "down"
    high: float           # 行情高点
    low: float            # 行情低点
    amplitude: float      # 振幅百分比
    retracement: List[FibLevel]  # 回撤位
    extension: List[FibLevel]    # 扩展位
    summary: str          # 文字总结


class FibonacciCalculator:
    """斐波那契回撤/扩展计算器。"""

    # 标准回撤比例
    RETRACEMENT_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]

    # 标准扩展比例
    EXTENSION_RATIOS = [1.0, 1.272, 1.618, 2.0, 2.618]

    def __init__(self, lookback: int = 120, swing_order: int = 5):
        """
        Args:
            lookback: 回溯K线数
            swing_order: 找极值点的窗口大小
        """
        self.lookback = lookback
        self.swing_order = swing_order

    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[float, float, int, int]:
        """找最近一段行情的高点和低点。

        Returns:
            (high_price, low_price, high_idx, low_idx)
        """
        if len(df) < 20:
            # 数据太少，用全局极值
            high_idx = df["high"].idxmax()
            low_idx = df["low"].idxmin()
            return df.loc[high_idx, "high"], df.loc[low_idx, "low"], high_idx, low_idx

        # 找最近的显著波峰和波谷
        close = df["close"].values
        high_col = df["high"].values if "high" in df.columns else close
        low_col = df["low"].values if "low" in df.columns else close

        # 局部极值
        order = min(self.swing_order, len(df) // 4)

        # 波峰
        peaks = []
        for i in range(order, len(high_col) - order):
            is_peak = True
            for j in range(max(0, i - order), min(len(high_col), i + order + 1)):
                if j != i and high_col[j] > high_col[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append((i, high_col[i]))

        # 波谷
        troughs = []
        for i in range(order, len(low_col) - order):
            is_trough = True
            for j in range(max(0, i - order), min(len(low_col), i + order + 1)):
                if j != i and low_col[j] < low_col[i]:
                    is_trough = False
                    break
            if is_trough:
                troughs.append((i, low_col[i]))

        if not peaks:
            high_idx = np.argmax(high_col)
            high_price = high_col[high_idx]
        else:
            # 取最近的高点（后半段中最高）
            recent_peaks = peaks[len(peaks) // 2:] if len(peaks) > 2 else peaks
            best = max(recent_peaks, key=lambda x: x[1])
            high_idx, high_price = best

        if not troughs:
            low_idx = np.argmin(low_col)
            low_price = low_col[low_idx]
        else:
            # 取最近的低点
            recent_troughs = troughs[len(troughs) // 2:] if len(troughs) > 2 else troughs
            best = min(recent_troughs, key=lambda x: x[1])
            low_idx, low_price = best

        return high_price, low_price, high_idx, low_idx

    def calc_retracement(self, df: pd.DataFrame) -> FibResult:
        """计算斐波那契回撤位。

        上升趋势：从低到高，回撤位是支撑
        下降趋势：从高到低，回撤位是压力
        """
        if df is None or len(df) < 10:
            return FibResult("unknown", 0, 0, 0, [], [], "数据不足")

        # 截取回溯范围
        data = df.tail(self.lookback).reset_index(drop=True)
        current_price = data["close"].iloc[-1]

        high_price, low_price, _, _ = self._find_swing_points(data)

        if high_price == low_price:
            return FibResult("unknown", high_price, low_price, 0, [], [], "高低点相同")

        amplitude = (high_price - low_price) / low_price * 100
        trend = "up" if data["close"].iloc[-1] > data["close"].iloc[0] else "down"

        retracement = []
        for ratio in self.RETRACEMENT_RATIOS:
            if trend == "up":
                # 上升趋势：回撤 = 高点 - 振幅 * 比例（支撑位）
                price = high_price - (high_price - low_price) * ratio
                direction = "support"
            else:
                # 下降趋势：反弹 = 低点 + 振幅 * 比例（压力位）
                price = low_price + (high_price - low_price) * ratio
                direction = "resistance"

            # 标注当前位置关系
            if trend == "up":
                if price > current_price:
                    label = f"上方支撑 {ratio:.1%}"
                else:
                    label = f"已跌破 {ratio:.1%}"
            else:
                if price < current_price:
                    label = f"下方压力 {ratio:.1%}"
                else:
                    label = f"已突破 {ratio:.1%}"

            retracement.append(FibLevel(
                ratio=ratio, price=round(price, 3),
                label=label, direction=direction,
            ))

        # 计算扩展位
        extension = []
        for ratio in self.EXTENSION_RATIOS:
            if trend == "up":
                price = high_price + (high_price - low_price) * (ratio - 1)
                label = f"扩展 {ratio:.1%}"
                direction = "resistance"
            else:
                price = low_price - (high_price - low_price) * (ratio - 1)
                label = f"扩展 {ratio:.1%}"
                direction = "support"

            extension.append(FibLevel(
                ratio=ratio, price=round(price, 3),
                label=label, direction=direction,
            ))

        # 生成总结
        trend_label = "上升" if trend == "up" else "下降"
        summary_lines = [
            f"{trend_label}趋势 | 高{high_price:.2f} 低{low_price:.2f} 振幅{amplitude:.1f}%",
            f"现价 {current_price:.2f}",
            "回撤位:",
        ]
        for lv in retracement:
            marker = "⬅️" if abs(lv.price - current_price) / current_price < 0.01 else "  "
            summary_lines.append(f"  {marker} {lv.ratio:.1%} = {lv.price:.2f} ({lv.label})")

        summary = "\n".join(summary_lines)

        return FibResult(
            trend=trend,
            high=round(high_price, 3),
            low=round(low_price, 3),
            amplitude=round(amplitude, 2),
            retracement=retracement,
            extension=extension,
            summary=summary,
        )

    def format_report(self, result: FibResult, symbol: str = "",
                       period: str = "") -> str:
        """格式化输出。"""
        lines = []
        lines.append(f"📐 斐波那契回撤: {symbol} {period}")
        lines.append(f"  {result.summary}")
        if result.extension:
            lines.append("扩展位:")
            for lv in result.extension:
                lines.append(f"    {lv.ratio:.1%} = {lv.price:.2f} ({lv.direction})")
        return "\n".join(lines)
