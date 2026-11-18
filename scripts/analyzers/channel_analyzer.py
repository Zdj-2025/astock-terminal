#!/usr/bin/env python3
"""通道分析器 — 上升/下降通道识别与交易信号。

从任意起始点开始，自动识别价格通道：
  1. 用 scipy.signal 找关键高低点
  2. 线性回归拟合上轨和下轨
  3. 计算通道位置百分比（0%=下轨, 100%=上轨）
  4. 历史触碰下轨/上轨后的涨跌统计
  5. 预测未来N天通道位置
  6. 综合信号：通道内做多/做空/观望

用法:
    from scripts.analyzers.channel_analyzer import ChannelAnalyzer
    analyzer = ChannelAnalyzer()
    result = analyzer.analyze('603881', start_date='2025-04-09')
    print(result)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.signal import argrelextrema, savgol_filter
from typing import Optional


@dataclass
class ChannelResult:
    """通道分析结果。"""
    has_channel: bool = False
    direction: str = "none"        # "up" / "down" / "flat"
    lower_slope: float = 0.0       # 下轨斜率（元/天）
    lower_intercept: float = 0.0   # 下轨截距
    upper_slope: float = 0.0       # 上轨斜率
    upper_intercept: float = 0.0   # 上轨截距
    lower_now: float = 0.0         # 当前下轨值
    upper_now: float = 0.0         # 当前上轨值
    channel_width: float = 0.0     # 通道宽度
    channel_pct: float = 0.0       # 通道位置 0-100%
    price: float = 0.0             # 当前价格
    distance_to_lower: float = 0.0 # 距下轨(%)
    distance_to_upper: float = 0.0 # 距上轨(%)
    touch_count: int = 0           # 触碰下轨次数
    avg_return_5d: float = 0.0     # 触碰下轨后5日平均涨幅
    avg_return_10d: float = 0.0    # 触碰下轨后10日平均涨幅
    avg_return_20d: float = 0.0    # 触碰下轨后20日平均涨幅
    win_rate_20d: float = 0.0      # 触碰下轨后20日胜率
    # 下降通道上轨触碰统计
    upper_touch_count: int = 0    # 触碰上轨次数
    upper_avg_return_5d: float = 0.0  # 触碰上轨后5日平均涨幅（负=下跌）
    upper_avg_return_10d: float = 0.0
    upper_avg_return_20d: float = 0.0
    upper_win_rate_20d: float = 0.0   # 触碰上轨后20日上涨胜率（低=上轨压力大）
    upper_touch_history: list = field(default_factory=list)
    #
    future_lower: dict = field(default_factory=dict)  # 未来N天下轨预测
    future_upper: dict = field(default_factory=dict)  # 未来N天上轨预测
    lower_points: list = field(default_factory=list)   # 下轨关键点 [(date, price), ...]
    upper_points: list = field(default_factory=list)   # 上轨关键点 [(date, price), ...]
    touch_history: list = field(default_factory=list)   # 触碰下轨历史
    signal: str = "观望"           # 信号
    score: float = 0.0             # 通道信号分 -10~+10
    details: str = ""
    start_date: str = ""           # 通道起点日期
    start_price: float = 0.0       # 通道起点价格
    trend_method: str = ""         # 趋势判断方法


class ChannelAnalyzer:
    """上升/下降通道分析器。"""

    # 通道参数
    DEFAULT_EXTREMA_ORDER = 12      # 找高低点的order参数
    TOUCH_TOLERANCE = 0.015         # 触碰容差1.5%
    MIN_LOW_POINTS = 2              # 至少2个低点才能画下轨
    MIN_HIGH_POINTS = 2             # 至少2个高点才能画上轨
    MIN_R2 = 0.70                   # R²最低要求（通道有效性）
    MIN_SLOPE_PCT = 0.0005          # 最小斜率(%)才认定为上升/下降通道

    def _ensure_ascending(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数据按日期升序排列（从旧到新）。"""
        date_col = 'trade_date' if 'trade_date' in df.columns else (
            'date' if 'date' in df.columns else None)
        if date_col and len(df) > 1:
            first = str(df[date_col].iloc[0])
            last = str(df[date_col].iloc[-1])
            if first > last:  # 降序 → 翻转
                df = df.iloc[::-1].reset_index(drop=True)
        return df

    def analyze(
        self,
        df: pd.DataFrame,
        start_idx: int = 0,
        current_price: float | None = None,
        extrema_order: int = 0,
    ) -> ChannelResult:
        """分析通道。

        Args:
            df: K线数据，需有 high/low/close 列（自动处理升/降序）
            start_idx: 起始索引（如某重要低点）
            current_price: 当前实时价格（None则用收盘价）
            extrema_order: 找极值的order（0则用默认值）
        """
        result = ChannelResult()

        df = self._ensure_ascending(df)
        high = df['high'].values[start_idx:]
        low = df['low'].values
        close = df['close'].values[start_idx:]
        n = len(high)

        if n < 30:
            result.details = "数据不足30根K线"
            return result

        order = extrema_order or self.DEFAULT_EXTREMA_ORDER
        order = max(5, min(order, n // 4))  # 自适应

        # 找关键高低点
        lows_idx = argrelextrema(low[start_idx:], np.less, order=order)[0]
        highs_idx = argrelextrema(high, np.greater, order=order)[0]

        if len(lows_idx) < self.MIN_LOW_POINTS:
            result.details = f"低点不足{self.MIN_LOW_POINTS}个"
            return result
        if len(highs_idx) < self.MIN_HIGH_POINTS:
            result.details = f"高点不足{self.MIN_HIGH_POINTS}个"
            return result

        # 线性回归拟合下轨
        low_xs = lows_idx.astype(float)
        low_ys = np.array([low[start_idx + i] for i in lows_idx])
        coeffs_lower = np.polyfit(low_xs, low_ys, 1)
        lower_slope, lower_intercept = coeffs_lower

        # 线性回归拟合上轨
        high_xs = highs_idx.astype(float)
        high_ys = np.array([high[i] for i in highs_idx])
        coeffs_upper = np.polyfit(high_xs, high_ys, 1)
        upper_slope, upper_intercept = coeffs_upper

        # 计算R²验证通道有效性
        lower_predicted = np.polyval(coeffs_lower, low_xs)
        ss_res = np.sum((low_ys - lower_predicted) ** 2)
        ss_tot = np.sum((low_ys - np.mean(low_ys)) ** 2)
        r2_lower = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        upper_predicted = np.polyval(coeffs_upper, high_xs)
        ss_res_u = np.sum((high_ys - upper_predicted) ** 2)
        ss_tot_u = np.sum((high_ys - np.mean(high_ys)) ** 2)
        r2_upper = 1 - ss_res_u / ss_tot_u if ss_tot_u > 0 else 0

        avg_r2 = (r2_lower + r2_upper) / 2

        # 当前通道值
        lower_now = lower_slope * (n - 1) + lower_intercept
        upper_now = upper_slope * (n - 1) + upper_intercept
        channel_width = upper_now - lower_now

        if channel_width <= 0:
            result.details = "通道宽度<=0，无效"
            return result

        # 方向判断
        slope_pct = lower_slope / (lower_now if lower_now > 0 else 1)
        if slope_pct > self.MIN_SLOPE_PCT and lower_slope > 0:
            direction = "up"
        elif slope_pct < -self.MIN_SLOPE_PCT and lower_slope < 0:
            direction = "down"
        else:
            direction = "flat"

        # 当前价格
        price = current_price if current_price is not None else close[-1]
        channel_pct = (price - lower_now) / channel_width * 100
        channel_pct = max(0, min(100, channel_pct))

        # 填充结果
        result.has_channel = True
        result.direction = direction
        result.lower_slope = lower_slope
        result.lower_intercept = lower_intercept
        result.upper_slope = upper_slope
        result.upper_intercept = upper_intercept
        result.lower_now = lower_now
        result.upper_now = upper_now
        result.channel_width = channel_width
        result.channel_pct = channel_pct
        result.price = price
        result.distance_to_lower = (price / lower_now - 1) * 100
        result.distance_to_upper = (price / upper_now - 1) * 100

        # 关键点记录
        dates = df['trade_date'].values if 'trade_date' in df.columns else df['date'].values if 'date' in df.columns else None
        if dates is not None:
            for i in lows_idx:
                d = dates[start_idx + i]
                result.lower_points.append((str(d)[:10], float(low[start_idx + i])))
            for i in highs_idx:
                d = dates[i]
                result.upper_points.append((str(d)[:10], float(high[i])))

        # 触碰下轨统计
        tolerance = self.TOUCH_TOLERANCE
        touch_returns_5d = []
        touch_returns_10d = []
        touch_returns_20d = []
        touch_count = 0

        for i in range(n):
            ch_val = lower_slope * i + lower_intercept
            if low[start_idx + i] <= ch_val * (1 + tolerance):
                touch_count += 1
                ret5 = (close[min(i + 5, n - 1)] / close[i] - 1) * 100 if i + 5 < n else None
                ret10 = (close[min(i + 10, n - 1)] / close[i] - 1) * 100 if i + 10 < n else None
                ret20 = (close[min(i + 20, n - 1)] / close[i] - 1) * 100 if i + 20 < n else None

                if ret5 is not None:
                    touch_returns_5d.append(ret5)
                if ret10 is not None:
                    touch_returns_10d.append(ret10)
                if ret20 is not None:
                    touch_returns_20d.append(ret20)

                touch_entry = {
                    'price': float(low[start_idx + i]),
                    'lower_val': float(ch_val),
                    'ret5': ret5,
                    'ret10': ret10,
                    'ret20': ret20,
                }
                if dates is not None:
                    touch_entry['date'] = str(dates[start_idx + i])[:10]
                result.touch_history.append(touch_entry)

        result.touch_count = touch_count

        if touch_returns_5d:
            result.avg_return_5d = np.mean(touch_returns_5d)
        if touch_returns_10d:
            result.avg_return_10d = np.mean(touch_returns_10d)
        if touch_returns_20d:
            result.avg_return_20d = np.mean(touch_returns_20d)
            result.win_rate_20d = sum(1 for r in touch_returns_20d if r > 0) / len(touch_returns_20d) * 100

        # 未来预测
        for days in [1, 3, 5, 10, 20]:
            result.future_lower[days] = lower_slope * (n - 1 + days) + lower_intercept
            result.future_upper[days] = upper_slope * (n - 1 + days) + upper_intercept

        # ====== 上轨触碰统计（下降通道关键信号） ======
        upper_touch_returns_5d = []
        upper_touch_returns_10d = []
        upper_touch_returns_20d = []
        upper_touch_count = 0

        for i in range(n):
            ch_upper = upper_slope * i + upper_intercept
            if high[i] >= ch_upper * (1 - tolerance):
                upper_touch_count += 1
                ret5 = (close[min(i + 5, n - 1)] / close[i] - 1) * 100 if i + 5 < n else None
                ret10 = (close[min(i + 10, n - 1)] / close[i] - 1) * 100 if i + 10 < n else None
                ret20 = (close[min(i + 20, n - 1)] / close[i] - 1) * 100 if i + 20 < n else None

                if ret5 is not None:
                    upper_touch_returns_5d.append(ret5)
                if ret10 is not None:
                    upper_touch_returns_10d.append(ret10)
                if ret20 is not None:
                    upper_touch_returns_20d.append(ret20)

                upper_entry = {
                    'price': float(high[i]),
                    'upper_val': float(ch_upper),
                    'ret5': ret5,
                    'ret10': ret10,
                    'ret20': ret20,
                }
                if dates is not None:
                    upper_entry['date'] = str(dates[start_idx + i])[:10]
                result.upper_touch_history.append(upper_entry)

        result.upper_touch_count = upper_touch_count
        if upper_touch_returns_5d:
            result.upper_avg_return_5d = np.mean(upper_touch_returns_5d)
        if upper_touch_returns_10d:
            result.upper_avg_return_10d = np.mean(upper_touch_returns_10d)
        if upper_touch_returns_20d:
            result.upper_avg_return_20d = np.mean(upper_touch_returns_20d)
            result.upper_win_rate_20d = sum(1 for r in upper_touch_returns_20d if r > 0) / len(upper_touch_returns_20d) * 100

        # ====== 信号判断 ======
        score = 0.0
        signal_parts = []

        # 通道方向加分
        if direction == "up":
            score += 2.0
            signal_parts.append("上升通道")
        elif direction == "down":
            score -= 2.0
            signal_parts.append("下降通道")
        else:
            signal_parts.append("横盘通道")

        # R²加分（通道有效性）
        if avg_r2 >= 0.90:
            score += 2.0
            signal_parts.append(f"高有效性(R²={avg_r2:.2f})")
        elif avg_r2 >= 0.80:
            score += 1.0
            signal_parts.append(f"中有效性(R²={avg_r2:.2f})")
        elif avg_r2 >= self.MIN_R2:
            score += 0.5
        else:
            score -= 1.0
            signal_parts.append(f"低有效性(R²={avg_r2:.2f})")

        # 通道位置信号（上升通道）
        if direction == "up":
            if channel_pct <= 10:
                score += 5.0
                signal_parts.append("贴近下轨（强买入区）")
            elif channel_pct <= 25:
                score += 3.0
                signal_parts.append("下轨附近（买入区）")
            elif channel_pct <= 50:
                score += 1.0
                signal_parts.append("通道下半区")
            elif channel_pct >= 90:
                score -= 3.0
                signal_parts.append("贴近上轨（卖出区）")
            elif channel_pct >= 75:
                score -= 1.0
                signal_parts.append("上轨附近（减仓区）")
            else:
                signal_parts.append("通道上半区")

        # 通道位置信号（下降通道）
        elif direction == "down":
            if channel_pct >= 90:
                score -= 5.0
                signal_parts.append("贴近上轨（强卖出/做空区）")
            elif channel_pct >= 75:
                score -= 3.0
                signal_parts.append("上轨附近（卖出区）")
            elif channel_pct >= 50:
                score -= 1.0
                signal_parts.append("通道上半区（弱势）")
            elif channel_pct <= 10:
                score += 3.0
                signal_parts.append("贴近下轨（可能反转/抄底区）")
            elif channel_pct <= 25:
                score += 1.5
                signal_parts.append("下轨附近（关注反转）")
            else:
                signal_parts.append("通道下半区")

        # 历史下轨胜率加分（上升通道用）
        if direction == "up":
            if result.win_rate_20d >= 80 and touch_count >= 3:
                score += 2.0
                signal_parts.append(f"下轨20日胜率{result.win_rate_20d:.0f}%")
            elif result.win_rate_20d >= 60 and touch_count >= 3:
                score += 1.0

        # 历史上轨胜率加分（下降通道用 — 上涨胜率低=上轨压力强=做空信号）
        if direction == "down":
            if result.upper_win_rate_20d <= 30 and upper_touch_count >= 3:
                score -= 2.0
                signal_parts.append(f"上轨20日上涨胜率仅{result.upper_win_rate_20d:.0f}%（压力强）")
            elif result.upper_win_rate_20d <= 50 and upper_touch_count >= 3:
                score -= 1.0
                signal_parts.append(f"上轨20日上涨胜率{result.upper_win_rate_20d:.0f}%")

        # 通道完整性检查
        penetrations = 0
        for i in range(n):
            ch_lower = lower_slope * i + lower_intercept
            if low[start_idx + i] < ch_lower * 0.97:
                penetrations += 1
        if penetrations > n * 0.05:
            score -= 1.5
            signal_parts.append(f"穿透{penetrations}次")

        result.score = round(max(-10, min(10, score)), 1)

        if result.score >= 6:
            result.signal = "🟢 强买入（通道下轨支撑+历史高胜率）"
        elif result.score >= 3:
            result.signal = "🟢 买入（通道下轨附近）"
        elif result.score >= 1:
            result.signal = "🟡 关注（通道支撑区）"
        elif result.score <= -6:
            result.signal = "🔴 强卖出（通道上轨压力+历史低胜率）"
        elif result.score <= -3:
            result.signal = "🔴 卖出（通道上轨附近）"
        elif result.score <= -1:
            result.signal = "🟡 警惕（通道压力区）"
        else:
            result.signal = "⚪ 观望（通道中间）"

        result.details = " | ".join(signal_parts)
        return result

    def find_channel_starts(self, df, lookback=500):
        """三级通道起点检测。

        Level 1: 三种方法独立生成候选
          A. 结构底 — 重要低点（反弹>15%）
          B. 趋势转向 — MA20从下转上
          C. 动量反转 — f''从负转正（大级别）
        Level 2: 区域聚合（±30根合并，按共识度排序）
        Level 3: 通道R²验证

        Returns:
            list of {idx, methods, consensus, importance, date, price}
        """
        df = self._ensure_ascending(df)
        close = df['close'].values
        low = df['low'].values
        n = len(close)
        dates = df['trade_date'].values if 'trade_date' in df.columns else (
            df['date'].values if 'date' in df.columns else None)

        candidates = {}  # idx -> {methods, importance}

        # ====== 方法A: 结构底 ======
        for order in [20, 30, 40]:
            lows_idx = argrelextrema(low, np.less, order=order)[0]
            for i in lows_idx:
                if i + 30 < n:
                    bounce = (close[min(i + 30, n - 1)] / low[i] - 1) * 100
                    if bounce > 15:
                        if i not in candidates:
                            candidates[i] = {'methods': set(), 'importance': 0}
                        candidates[i]['methods'].add('结构底')
                        candidates[i]['importance'] = max(
                            candidates[i]['importance'], bounce)

        # ====== 方法B: MA20趋势转向 ======
        if n > 60:
            ma20 = np.convolve(close, np.ones(20) / 20, mode='valid')
            for i in range(5, len(ma20)):
                if (ma20[i] > ma20[i - 1] and ma20[i - 1] <= ma20[i - 2] and
                        (ma20[i] - ma20[i - 2]) / ma20[i - 2] * 100 > 1):
                    s = max(0, i - 10)
                    e = min(n - 20, i + 20)
                    min_idx = s + np.argmin(low[s:e])
                    if min_idx not in candidates:
                        candidates[min_idx] = {'methods': set(), 'importance': 0}
                    candidates[min_idx]['methods'].add('MA20转向')
                    candidates[min_idx]['importance'] = max(
                        candidates[min_idx]['importance'],
                        (ma20[i] - ma20[i - 2]) / ma20[i - 2] * 100 * 10)

        # ====== 方法C: f''拐点（只取大级别） ======
        ln_p = np.log(close)
        window = min(21, n // 4 * 2 + 1)
        if window >= 5:
            f2 = savgol_filter(ln_p, window, 3, deriv=2)
            # 向后平移：savgol输出滞后 window//2 根K线，补偿对齐
            shift = window // 2
            f2_safe = np.empty_like(f2)
            f2_safe[:shift] = f2[:shift]
            f2_safe[shift:] = f2[:-shift]
            f2_smooth = savgol_filter(f2_safe, min(15, len(f2_safe) // 2 * 2 + 1), 2)
            for i in range(5, len(f2_smooth)):
                if f2_smooth[i] > 0 and f2_smooth[i - 1] <= 0:
                    if f2_smooth[i] - f2_smooth[i - 1] > 0.001:
                        s = max(0, i - 10)
                        e = min(n, i + 10)
                        min_idx = s + np.argmin(low[s:e])
                        if min_idx not in candidates:
                            candidates[min_idx] = {'methods': set(), 'importance': 0}
                        candidates[min_idx]['methods'].add("f''拐点")
                        candidates[min_idx]['importance'] = max(
                            candidates[min_idx]['importance'],
                            (f2_smooth[i] - f2_smooth[i - 1]) * 1000)

        # ====== Level 2: 区域聚合 ======
        all_indices = sorted(candidates.keys())
        merged = []
        used = set()

        for idx in all_indices:
            if idx in used:
                continue
            cluster = [idx]
            for other in all_indices:
                if other != idx and other not in used and abs(other - idx) <= 30:
                    cluster.append(other)
                    used.add(other)
            used.add(idx)

            methods = set()
            max_imp = 0
            for c in cluster:
                methods.update(candidates[c]['methods'])
                max_imp = max(max_imp, candidates[c]['importance'])

            entry = {
                'idx': cluster[0],
                'methods': methods,
                'consensus': len(methods),
                'importance': max_imp,
            }
            if dates is not None:
                entry['date'] = str(dates[cluster[0]])[:10]
                entry['price'] = float(low[cluster[0]])
            merged.append(entry)

        merged.sort(key=lambda x: (x['consensus'], x['importance']), reverse=True)
        return merged

    def find_best_channel(
        self,
        df: pd.DataFrame,
        current_price: float | None = None,
        min_days: int = 60,
        lookback: int = 500,
    ) -> ChannelResult:
        """智能寻找最佳通道 — 极值定方向+起点。

        庄哥的方法（简洁直接）：
        1. 在给定范围内找全局最低和全局最高
        2. 用线性回归斜率判断整体趋势方向
        3. 上升通道 → 起点=最低点（从底部画起）
        4. 下降通道 → 起点=最高点（从顶部画起）
        5. 起点确定后画通道，验证R²有效性

        如果自动找：逐步扩大范围（60天→120→250→500），
        取R²最高且方向明确的通道。
        """
        df = self._ensure_ascending(df)
        n_total = len(df)
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        dates = df['trade_date'].values if 'trade_date' in df.columns else (
            df['date'].values if 'date' in df.columns else None)

        best_result = ChannelResult()
        best_score = -999

        # 多个时间范围逐一尝试，取最佳
        for span in [60, 120, 250, 500, n_total]:
            if span < min_days:
                continue
            start_bound = max(0, n_total - span)
            end_bound = n_total

            # 只看范围内的数据
            sub_low = low[start_bound:end_bound]
            sub_high = high[start_bound:end_bound]
            sub_close = close[start_bound:end_bound]
            n_sub = len(sub_close)
            if n_sub < min_days:
                continue

            # 1) 找全局最低和最高
            min_idx = np.argmin(sub_low)  # 范围内最低价位置
            max_idx = np.argmax(sub_high)  # 范围内最高价位置
            min_price = sub_low[min_idx]
            max_price = sub_high[max_idx]

            # 2) 判断趋势方向 — 用线性回归斜率
            xs = np.arange(n_sub, dtype=float)
            reg = np.polyfit(xs, sub_close, 1)
            slope = reg[0]
            slope_pct = slope / np.mean(sub_close)  # 归一化斜率

            if slope_pct > 0.0005:
                direction = "up"
                # 上升通道：起点=最低点
                start_offset = min_idx
            elif slope_pct < -0.0005:
                direction = "down"
                # 下降通道：起点=最高点
                start_offset = max_idx
            else:
                direction = "flat"
                start_offset = min_idx  # 横盘也用最低点

            start_idx = start_bound + start_offset

            # 确保起点后有足够数据
            if n_total - start_idx < min_days:
                continue

            try:
                result = self.analyze(
                    df, start_idx=start_idx, current_price=current_price)
                if not result.has_channel:
                    continue

                # 记录趋势判断信息
                result.trend_method = f"回归斜率{slope_pct:+.5f} | 极值定起点({direction})"
                if dates is not None:
                    result.start_date = str(dates[start_idx])[:10]
                    result.start_price = float(
                        low[start_idx] if direction == "up" else high[start_idx])

                # 综合评分
                score = 0

                # 1) 方向和回归斜率一致 → 加分
                if result.direction == direction:
                    score += 3.0

                # 2) R²有效性
                avg_r2 = self._calc_r2(df, start_idx, result)
                score += avg_r2 * 4

                # 3) 触碰次数（被反复验证）
                score += min(result.touch_count + result.upper_touch_count, 20) * 0.2

                # 4) 上下轨方向一致
                if result.lower_slope * result.upper_slope > 0:
                    score += 1.0

                # 5) 当前位置靠近边界 → 更有实战价值
                if result.channel_pct < 20 or result.channel_pct > 80:
                    score += 1.5

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception:
                continue

        return best_result

    def _calc_r2(self, df, start_idx, result):
        """计算通道的平均R²。"""
        n = len(df) - start_idx
        low = df['low'].values[start_idx:]
        high = df['high'].values[start_idx:]

        xs = np.arange(n, dtype=float)

        # 下轨R²
        lower_pred = result.lower_slope * xs + result.lower_intercept
        lows_idx = argrelextrema(low, np.less, order=self.DEFAULT_EXTREMA_ORDER)[0]
        if len(lows_idx) >= 2:
            actual = low[lows_idx]
            predicted = lower_pred[lows_idx]
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2_l = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r2_l = 0

        # 上轨R²
        upper_pred = result.upper_slope * xs + result.upper_intercept
        highs_idx = argrelextrema(high, np.greater, order=self.DEFAULT_EXTREMA_ORDER)[0]
        if len(highs_idx) >= 2:
            actual = high[highs_idx]
            predicted = upper_pred[highs_idx]
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2_u = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r2_u = 0

        return (r2_l + r2_u) / 2

    def format_report(self, r: ChannelResult, symbol: str = "", name: str = "") -> str:
        """生成可读报告。"""
        if not r.has_channel:
            return f"{name}({symbol}): 无法识别通道 — {r.details}"

        lines = [
            f"📊 {name}({symbol}) 通道分析",
            f"起点: {r.start_date} ({r.start_price:.2f}) | {r.trend_method}",
            f"方向: {'📈上升' if r.direction=='up' else '📉下降' if r.direction=='down' else '➡️横盘'}通道",
            f"通道宽度: {r.channel_width:.2f}元",
            f"",
            f"下轨: {r.lower_now:.2f} (斜率+{r.lower_slope:.4f}元/天)",
            f"上轨: {r.upper_now:.2f} (斜率+{r.upper_slope:.4f}元/天)",
            f"",
            f"当前价: {r.price:.2f}",
            f"通道位置: {r.channel_pct:.0f}%",
            f"距下轨: {r.distance_to_lower:+.1f}%",
            f"距上轨: {r.distance_to_upper:+.1f}%",
            f"",
        ]

        if r.touch_count > 0:
            lines.append(f"历史触碰下轨{r.touch_count}次:")
            if r.avg_return_5d:
                lines.append(f"  5日平均: {r.avg_return_5d:+.1f}%")
            if r.avg_return_10d:
                lines.append(f"  10日平均: {r.avg_return_10d:+.1f}%")
            if r.avg_return_20d:
                lines.append(f"  20日平均: {r.avg_return_20d:+.1f}% (胜率{r.win_rate_20d:.0f}%)")

        if r.upper_touch_count > 0:
            lines.append("")
            lines.append(f"历史触碰上轨{r.upper_touch_count}次:")
            if r.upper_avg_return_5d:
                lines.append(f"  5日平均: {r.upper_avg_return_5d:+.1f}%")
            if r.upper_avg_return_10d:
                lines.append(f"  10日平均: {r.upper_avg_return_10d:+.1f}%")
            if r.upper_avg_return_20d:
                lines.append(f"  20日平均: {r.upper_avg_return_20d:+.1f}% (上涨胜率{r.upper_win_rate_20d:.0f}%)")

        lines.append("")
        lines.append("未来通道预测:")
        for days in [5, 10, 20]:
            lines.append(f"  +{days}天: 下轨{r.future_lower[days]:.2f} 上轨{r.future_upper[days]:.2f}")

        lines.append("")
        lines.append(f"信号: {r.signal} (评分{r.score:+.1f})")
        lines.append(f"详情: {r.details}")

        return "\n".join(lines)


# ====== CLI 入口 ======
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="通道分析器")
    parser.add_argument("symbol", help="股票代码 如 603881")
    parser.add_argument("--start", help="通道起始日 YYYY-MM-DD")
    parser.add_argument("--period", default="D", help="K线周期 D/W")
    parser.add_argument("--auto", action="store_true", help="自动找最佳起点")
    parser.add_argument("--price", type=float, help="当前实时价格")
    args = parser.parse_args()

    from scripts.core.data_store import DataStore
    store = DataStore()
    df = store.get_kline(args.symbol, args.period)
    if df is None:
        print(f"无法获取{args.symbol}数据")
        sys.exit(1)

    analyzer = ChannelAnalyzer()

    if args.auto:
        result = analyzer.find_best_channel(df, current_price=args.price)
    else:
        start_idx = 0
        if args.start:
            dates = df['trade_date'].values if 'trade_date' in df.columns else df['date'].values
            for i, d in enumerate(dates):
                if str(d) >= args.start:
                    start_idx = i
                    break
        result = analyzer.analyze(df, start_idx=start_idx, current_price=args.price)

    print(analyzer.format_report(result, symbol=args.symbol))
