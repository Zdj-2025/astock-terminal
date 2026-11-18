#!/usr/bin/env python3
"""CUSUM 变化检测器 — V3 二阶导数评分器辅助模块。

核心功能：
  1. 对 f'(趋势速度) 做 CUSUM，检测结构性断点
  2. 自适应阈值（EWMA 波动率归一化）
  3. 与 V3 评分结果融合，过滤假拐点信号

设计原则：
  - O(n) 计算复杂度，支持实时更新
  - 不依赖重型库（仅需 numpy）
  - 完全消除 look-ahead bias

数学原理：
  S_t^+ = max(0, S_{t-1}^+ + (x_t - mu_t - k))    上界CUSUM
  S_t^- = max(0, S_{t-1}^- + (mu_t - x_t - k))    下界CUSUM
  触发: S_t^+ > h_t 或 S_t^- > h_t

用法：
  from scripts.analyzers.cusum_detector import CUSUMDetector

  # 方式1: 单只股票检测
  detector = CUSUMDetector()
  result = detector.detect(f1_array)

  # 方式2: 与V3融合
  fused = detector.fuse_with_v3(v3_score, f1_array)

  # 方式3: 批量扫描
  results = detector.detect_batch(f1_dict)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── 默认参数（A股日线调优）────────────────────────────────────────

# 阈值参数
DEFAULT_H_BASE = 4.0     # 基准阈值（单位: sigma）
DEFAULT_K = 0.5           # 参考值（最小可检测偏移的一半，单位: sigma）
DEFAULT_EWMA_SPAN = 20    # 波动率EWMA窗口（约4周）

# 基准值参数
DEFAULT_BASELINE_SPAN = 60  # 基准值(f'滑动均值)窗口（约3个月）

# 触发后冷却
DEFAULT_COOLDOWN = 10     # 触发后多少根K线内不再触发（避免连续误报）


@dataclass
class CUSUMResult:
    """CUSUM 检测结果。"""

    # 当前状态
    s_plus: float = 0.0       # 上界CUSUM当前值
    s_minus: float = 0.0      # 下界CUSUM当前值
    h_current: float = 0.0    # 当前自适应阈值

    # 触发状态
    up_crossing: bool = False    # 是否检测到向上突变
    down_crossing: bool = False  # 是否检测到向下突变

    # 最近一次触发的详细信息
    last_up_change_idx: int = -1    # 最近向上触发的位置
    last_down_change_idx: int = -1  # 最近向下触发的位置
    change_magnitude: float = 0.0   # 变化幅度估计

    # 全序列统计
    n_points: int = 0
    n_up_events: int = 0     # 历史向上触发次数
    n_down_events: int = 0   # 历史向下触发次数

    # 置信度
    signal_strength: float = 0.0   # 信号强度 (0-1)
    confidence: str = "无信号"      # 高/中/低/无信号


class CUSUMDetector:
    """CUSUM 序贯变化检测器。

    特点：
      1. 自适应阈值 h_t = h_base * sigma_t，sigma_t 用 EWMA 估计
      2. 基准值 mu_t 用滑动均值，避免趋势性干扰
      3. 触发后冷却机制，防止连续误报
      4. O(n) 复杂度，支持增量更新
    """

    def __init__(
        self,
        h_base: float = DEFAULT_H_BASE,
        k: float = DEFAULT_K,
        ewma_span: int = DEFAULT_EWMA_SPAN,
        baseline_span: int = DEFAULT_BASELINE_SPAN,
        cooldown: int = DEFAULT_COOLDOWN,
    ):
        """初始化 CUSUM 检测器。

        Args:
            h_base: 基准阈值（sigma 的倍数）。A股日线推荐 3.0~6.0。
                    - 3.0: 灵敏，多信号，适合底部拐点预警
                    - 4.0: 平衡（推荐）
                    - 6.0: 保守，少信号但高置信度
            k: 参考值/允许偏差带（sigma 的倍数）。
               通常取 delta_min / 2，delta_min 是你想检测的最小变化。
               k=0.5 对应检测 1 sigma 的偏移。
            ewma_span: 波动率EWMA窗口（K线数）。
            baseline_span: 基准值滑动窗口（K线数）。
            cooldown: 触发后冷却期（K线数）。
        """
        self.h_base = h_base
        self.k = k
        self.ewma_span = ewma_span
        self.baseline_span = baseline_span
        self.cooldown = cooldown

        # EWMA 衰减系数
        self._ewma_alpha = 2.0 / (ewma_span + 1)

    def detect(self, x: np.ndarray) -> CUSUMResult:
        """对完整序列做 CUSUM 检测。

        Args:
            x: 输入序列。通常是 f'（一阶导）序列。
               应该已经做了 shift 消除 look-ahead bias。

        Returns:
            CUSUMResult 包含检测结果。
        """
        n = len(x)
        result = CUSUMResult(n_points=n)

        if n < 20:
            return result

        # ── 计算自适应参数 ──────────────────────────────
        baselines, sigmas = self._compute_adaptive_params(x)

        # ── 序贯 CUSUM 计算 ────────────────────────────
        s_plus = 0.0
        s_minus = 0.0
        last_up_idx = -self.cooldown  # 初始化为冷却期外
        last_down_idx = -self.cooldown

        up_events = 0
        down_events = 0

        # 只从有足够历史数据的位置开始
        start_idx = min(self.baseline_span, n)

        for i in range(start_idx, n):
            mu_t = baselines[i]
            sigma_t = sigmas[i]

            # 避免除零
            if sigma_t < 1e-10:
                sigma_t = 1e-10

            # 标准化偏差
            k_t = self.k * sigma_t
            h_t = self.h_base * sigma_t

            # CUSUM 递推
            s_plus = max(0.0, s_plus + (x[i] - mu_t - k_t))
            s_minus = max(0.0, s_minus + (mu_t - x[i] - k_t))

            # 检测触发
            if s_plus > h_t and (i - last_up_idx) >= self.cooldown:
                up_events += 1
                last_up_idx = i
                s_plus = 0.0  # 重置

            if s_minus > h_t and (i - last_down_idx) >= self.cooldown:
                down_events += 1
                last_down_idx = i
                s_minus = 0.0  # 重置

        # ── 填充结果 ──────────────────────────────────
        result.s_plus = s_plus
        result.s_minus = s_minus
        result.h_current = self.h_base * sigmas[-1] if sigmas[-1] > 0 else 0.0
        result.last_up_change_idx = last_up_idx
        result.last_down_change_idx = last_down_idx
        result.n_up_events = up_events
        result.n_down_events = down_events

        # 触发状态：最近 N 根内发生的变化
        recent_window = min(15, n - start_idx)
        result.up_crossing = (n - 1 - last_up_idx) < recent_window and up_events > 0
        result.down_crossing = (n - 1 - last_down_idx) < recent_window and down_events > 0

        # 信号强度
        if result.up_crossing or result.down_crossing:
            # 基于 CUSUM 累积量与阈值的比值
            max_stat = max(result.s_plus, result.s_minus)
            result.signal_strength = min(max_stat / max(result.h_current, 1e-10), 1.0)

            # 变化幅度估计
            if result.up_crossing:
                result.change_magnitude = float(x[-1] - baselines[-1])
            else:
                result.change_magnitude = float(baselines[-1] - x[-1])

        # 置信度标签
        if result.up_crossing or result.down_crossing:
            if result.signal_strength > 0.7:
                result.confidence = "高"
            elif result.signal_strength > 0.3:
                result.confidence = "中"
            else:
                result.confidence = "低"
        else:
            result.confidence = "无信号"

        return result

    def detect_incremental(
        self,
        x_new: float,
        state: dict,
    ) -> tuple[CUSUMResult, dict]:
        """增量更新（实时模式）。

        每来一个新的 f' 值，只做一次 O(1) 计算。

        Args:
            x_new: 新的观测值（最新 f'）
            state: 上一次的内部状态（首次调用传空dict）

        Returns:
            (result, new_state) — 结果和更新后的状态
        """
        # 初始化状态
        if not state:
            state = {
                "s_plus": 0.0,
                "s_minus": 0.0,
                "mu_ewma": None,
                "sigma_ewma": None,
                "last_up_idx": -self.cooldown,
                "last_down_idx": -self.cooldown,
                "idx": 0,
                "history": [],  # 保留 baseline_span 个历史值
            }

        state["idx"] += 1
        idx = state["idx"]
        state["history"].append(x_new)

        # 维护滑动窗口
        if len(state["history"]) > self.baseline_span:
            state["history"] = state["history"][-self.baseline_span:]

        # 滑动均值作为基准值
        hist = np.array(state["history"])
        mu_t = float(np.mean(hist))

        # EWMA 波动率
        if state["sigma_ewma"] is None:
            # 首次用历史标准差初始化
            if len(hist) >= 5:
                state["sigma_ewma"] = float(np.std(hist))
                state["mu_ewma"] = mu_t
            result = CUSUMResult(n_points=idx)
            return result, state

        # 更新 EWMA
        sigma_ewma = state["sigma_ewma"]
        mu_ewma = state["mu_ewma"]
        alpha = self._ewma_alpha

        dev = abs(x_new - mu_ewma)
        sigma_ewma = alpha * dev + (1 - alpha) * sigma_ewma
        mu_ewma = alpha * x_new + (1 - alpha) * mu_ewma
        state["sigma_ewma"] = sigma_ewma
        state["mu_ewma"] = mu_ewma

        sigma_t = max(sigma_ewma, 1e-10)
        k_t = self.k * sigma_t
        h_t = self.h_base * sigma_t

        # CUSUM 递推
        s_plus = max(0.0, state["s_plus"] + (x_new - mu_t - k_t))
        s_minus = max(0.0, state["s_minus"] + (mu_t - x_new - k_t))

        # 触发检测
        up_crossing = False
        down_crossing = False
        last_up_idx = state["last_up_idx"]
        last_down_idx = state["last_down_idx"]

        if s_plus > h_t and (idx - last_up_idx) >= self.cooldown:
            up_crossing = True
            last_up_idx = idx
            s_plus = 0.0

        if s_minus > h_t and (idx - last_down_idx) >= self.cooldown:
            down_crossing = True
            last_down_idx = idx
            s_minus = 0.0

        # 更新状态
        state["s_plus"] = s_plus
        state["s_minus"] = s_minus
        state["last_up_idx"] = last_up_idx
        state["last_down_idx"] = last_down_idx

        # 构造结果
        result = CUSUMResult(
            s_plus=s_plus,
            s_minus=s_minus,
            h_current=h_t,
            up_crossing=up_crossing,
            down_crossing=down_crossing,
            last_up_change_idx=last_up_idx if up_crossing else -1,
            last_down_change_idx=last_down_idx if down_crossing else -1,
            n_points=idx,
            signal_strength=0.0,
            confidence="无信号",
        )

        if up_crossing or down_crossing:
            stat = max(s_plus, s_minus)
            result.signal_strength = min(stat / max(h_t, 1e-10), 1.0)
            if up_crossing:
                result.change_magnitude = x_new - mu_t
            else:
                result.change_magnitude = mu_t - x_new
            if result.signal_strength > 0.7:
                result.confidence = "高"
            elif result.signal_strength > 0.3:
                result.confidence = "中"
            else:
                result.confidence = "低"

        return result, state

    def fuse_with_v3(
        self,
        v3_score,  # V3Score dataclass
        f1_array: np.ndarray,
        f2_val: float = 0.0,
    ) -> dict:
        """将 CUSUM 结果与 V3 评分融合。

        这是核心集成接口。

        Args:
            v3_score: V3DerivativeScorer.score_stock() 返回的 V3Score
            f1_array: f' 序列（已 shift）
            f2_val: 当前 f'' 值

        Returns:
            dict 包含融合建议：
              - cusum_result: CUSUMResult
              - curvature_adjustment: 曲率分调整值 (-1.0 ~ +0.5)
              - trend_adjustment: 趋势分调整值 (0.0 ~ 1.0)
              - signal_override: 是否覆盖 V3 信号
              - new_signal: 覆盖后的信号
              - reason: 调整原因
        """
        cusum_r = self.detect(f1_array)

        # 默认不调整
        curvature_adj = 0.0
        trend_adj = 0.0
        signal_override = False
        new_signal = v3_score.signal
        reason = "CUSUM未触发，维持V3原始评分"

        signal = v3_score.signal

        # ── 情况1：V3报底部拐点(f'<0, f''>0) ──
        if signal in ("底部拐点", "🔄 底部拐点"):
            if cusum_r.up_crossing and cusum_r.confidence in ("高", "中"):
                # CUSUM确认：f' 正在发生结构性向上变化
                curvature_adj = 0.5
                reason = f"CUSUM确认向上突破(强度{cusum_r.signal_strength:.2f})," \
                         f"拐点置信度提升"
            elif cusum_r.up_crossing and cusum_r.confidence == "低":
                # CUSUM弱确认
                curvature_adj = 0.2
                reason = "CUSUM弱确认向上突破，小幅加分"
            else:
                # CUSUM未确认：可能是f''的噪声（你的核心痛点）
                curvature_adj = -0.3
                reason = "CUSUM未确认向上变化，f''正可能仅是噪声(暴跌后横盘)，降分"

        # ── 情况2：V3报主升浪(f'>0, f''>0) ──
        elif signal in ("主升浪", "🚀 主升浪"):
            if cusum_r.down_crossing:
                # CUSUM检测到f'正在发生结构性向下变化
                trend_adj = -0.5
                reason = "CUSUM检测到趋势减速，主升浪可能结束"
                # 如果减速明显，降级信号
                if cusum_r.confidence in ("高", "中"):
                    signal_override = True
                    new_signal = "⚠️ 趋势减速(CUSUM确认)"
            elif cusum_r.up_crossing and cusum_r.confidence in ("高", "中"):
                # CUSUM确认加速
                curvature_adj = 0.3
                reason = "CUSUM确认趋势加速中，主升浪可信度高"

        # ── 情况3：V3报减速见顶(f'>0, f''<0) ──
        elif signal in ("减速见顶", "⚠️ 减速见顶"):
            if cusum_r.down_crossing and cusum_r.confidence in ("高", "中"):
                # CUSUM确认：趋势确实在减速
                trend_adj = -0.3
                reason = "CUSUM确认趋势减速，见顶概率增大"

        # ── 情况4：V3报加速下跌(f'<0, f''<0) ──
        elif signal in ("加速下跌", "🔴 加速下跌"):
            if cusum_r.up_crossing:
                # CUSUM检测到f'可能正在止跌
                curvature_adj = 0.3
                reason = "CUSUM检测到跌势可能放缓，关注止跌信号"

        return {
            "cusum_result": cusum_r,
            "curvature_adjustment": curvature_adj,
            "trend_adjustment": trend_adj,
            "signal_override": signal_override,
            "new_signal": new_signal,
            "reason": reason,
        }

    def detect_batch(
        self,
        f1_dict: dict[str, np.ndarray],
    ) -> dict[str, CUSUMResult]:
        """批量 CUSUM 检测。

        Args:
            f1_dict: {code: f1_array} 字典

        Returns:
            {code: CUSUMResult} 字典
        """
        results = {}
        for code, f1 in f1_dict.items():
            try:
                results[code] = self.detect(f1)
            except Exception:
                continue
        return results

    # ── 内部方法 ──────────────────────────────────────────

    def _compute_adaptive_params(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算自适应基准值和波动率序列。

        Returns:
            (baselines, sigmas) 两个与 x 等长的数组
        """
        n = len(x)
        baselines = np.zeros(n)
        sigmas = np.zeros(n)

        # 初始化基准值和波动率
        baselines[0] = x[0]
        sigmas[0] = max(abs(x[0]), 1e-8)

        alpha = self._ewma_alpha

        for i in range(1, n):
            # 基准值：滑动均值
            start = max(0, i - self.baseline_span)
            baselines[i] = np.mean(x[start:i + 1])

            # 波动率：EWMA of |x - baseline|
            dev = abs(x[i] - baselines[i])
            sigmas[i] = alpha * dev + (1 - alpha) * sigmas[i - 1]

            # 下界保护
            if sigmas[i] < 1e-10:
                # 用局部标准差兜底
                local_start = max(0, i - 20)
                local_std = np.std(x[local_start:i + 1])
                sigmas[i] = max(local_std, 1e-10)

        return baselines, sigmas


# ── 便捷函数 ──────────────────────────────────────────────

def quick_cusum(
    f1: np.ndarray,
    h_base: float = DEFAULT_H_BASE,
    k: float = DEFAULT_K,
) -> CUSUMResult:
    """快速 CUSUM 检测（单行调用）。

    Args:
        f1: f' 序列（已 shift 消除 look-ahead bias）
        h_base: 阈值
        k: 参考值

    Returns:
        CUSUMResult

    Example:
        from scripts.analyzers.cusum_detector import quick_cusum
        result = quick_cusum(f1_array, h_base=4.0)
        if result.up_crossing:
            print("检测到向上突变!")
    """
    detector = CUSUMDetector(h_base=h_base, k=k)
    return detector.detect(f1)


def cusum_series(
    f1: np.ndarray,
    h_base: float = DEFAULT_H_BASE,
    k: float = DEFAULT_K,
) -> dict:
    """返回完整 CUSUM 序列（用于可视化/调试）。

    Returns:
        dict with:
          s_plus_arr, s_minus_arr, h_arr, baselines, sigmas
          events: list of (idx, direction, magnitude)
    """
    n = len(f1)
    if n < 20:
        return {
            "s_plus_arr": np.zeros(n),
            "s_minus_arr": np.zeros(n),
            "h_arr": np.zeros(n),
            "events": [],
        }

    detector = CUSUMDetector(h_base=h_base, k=k)
    baselines, sigmas = detector._compute_adaptive_params(f1)

    s_plus_arr = np.zeros(n)
    s_minus_arr = np.zeros(n)
    h_arr = np.zeros(n)
    events = []

    s_plus = 0.0
    s_minus = 0.0
    last_up = -detector.cooldown
    last_down = -detector.cooldown

    start_idx = min(detector.baseline_span, n)

    for i in range(start_idx, n):
        mu_t = baselines[i]
        sigma_t = sigmas[i]
        if sigma_t < 1e-10:
            sigma_t = 1e-10

        k_t = k * sigma_t
        h_t = h_base * sigma_t
        h_arr[i] = h_t

        s_plus = max(0.0, s_plus + (f1[i] - mu_t - k_t))
        s_minus = max(0.0, s_minus + (mu_t - f1[i] - k_t))

        s_plus_arr[i] = s_plus
        s_minus_arr[i] = s_minus

        if s_plus > h_t and (i - last_up) >= detector.cooldown:
            events.append((i, "up", float(f1[i] - mu_t)))
            last_up = i
            s_plus = 0.0

        if s_minus > h_t and (i - last_down) >= detector.cooldown:
            events.append((i, "down", float(mu_t - f1[i])))
            last_down = i
            s_minus = 0.0

    return {
        "s_plus_arr": s_plus_arr,
        "s_minus_arr": s_minus_arr,
        "h_arr": h_arr,
        "baselines": baselines,
        "sigmas": sigmas,
        "events": events,
    }
