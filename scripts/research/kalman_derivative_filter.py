#!/usr/bin/env python3
"""2-State Adaptive Kalman Filter — 趋势导数估计器。

替代 Savitzky-Golay 滤波器，用于 V3 二阶导数评分系统。

核心优势：
  - 天然因果（无 look-ahead bias，无需 shift）
  - 直接输出 f'(速度) + f''(加速度) + 不确定性(P矩阵)
  - O(1) 每步更新（无窗口卷积）
  - 自适应噪声估计（Myers-Tapley 协方差匹配法）

状态空间模型：
  X = [位置, 速度] = [ln(P), f']
  状态转移：位置 += 速度*dt + 0.5*a*dt², 速度 += a*dt
  观测：z = ln(P) + noise

自适应方法：
  Myers-Tapley 协方差匹配 — 滑动窗口内用新息(innovation)和
  状态修正量估计 Q 和 R，无需 EM 算法。

参考文献：
  [1] Myers & Tapley (1976), "Adaptive sequential estimation"
  [2] Stengel (1994), "Optimal Control and Estimation"
  [3] Alpha Architect, "Trend-Following Filters Part 4 & 5"

Usage:
  kf = AdaptiveKalmanFilter2S()
  kf.initialize(ln_price[0])
  for p in ln_price[1:]:
      state = kf.update(p)
  # state.velocity -> f', state.acceleration -> f''
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════

@dataclass
class KalmanState:
    """卡尔曼滤波器单步输出。"""
    position: float = 0.0        # 滤波后 ln(P)
    velocity: float = 0.0        # f'(t) 一阶导 (原始单位: 1/day)
    acceleration: float = 0.0    # f''(t) 二阶导 (从速度差分)
    var_position: float = 0.0    # 位置估计方差
    var_velocity: float = 0.0    # 速度估计方差
    gain_position: float = 0.0   # 位置滤波增益 k1
    gain_velocity: float = 0.0   # 速度滤波增益 k2
    Q: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    R: float = 0.0
    innovation: float = 0.0      # 新息 z - z_pred
    step: int = 0


# ═══════════════════════════════════════════════════════════════════
#  2-State Adaptive Kalman Filter
# ═══════════════════════════════════════════════════════════════════

class AdaptiveKalmanFilter2S:
    """2状态自适应卡尔曼滤波器 — 位置+速度模型。

    状态向量:
        X = [x1, x2]^T = [位置 ln(P), 速度 f']

    状态转移 (匀速+随机加速度模型):
        x1(t) = x1(t-1) + x2(t-1)*dt + g1*w(t)
        x2(t) = x2(t-1)               + g2*w(t)
    其中 w(t) ~ N(0, q(t)) 是过程噪声（随机加速度），
    g1=0.5, g2=1.0 基于「速度受随机加速度扰动」物理模型。

    观测模型:
        z(t) = x1(t) + v(t),  v(t) ~ N(0, r(t))

    自适应:
        Myers-Tapley 协方差匹配法，滑动窗口长度 N。
        窗口内用新息和过程噪声样本估计 Q 和 R。

    加速度 f'' 的估计:
        f''(t) = [x2+(t) - x2+(t-1)] / dt
        这是速度的后验差分，不是直接的状态变量，
        但在 dt=1 时等价于二阶导数。
    """

    def __init__(
        self,
        dt: float = 1.0,
        q_scale: float = 1e-5,
        r_init: float = 1e-4,
        g1: float = 0.5,
        g2: float = 1.0,
        window: int = 10,
        p0_diag: tuple = (1e-2, 1e-4),
        q_floor: float = 1e-10,
        r_floor: float = 1e-8,
    ):
        """初始化。

        Args:
            dt: 时间步长（日线=1.0）
            q_scale: 初始过程噪声标度
            r_init: 初始观测噪声方差
            g1: 位置过程噪声输入系数（0.5=随机加速度模型）
            g2: 速度过程噪声输入系数（1.0）
            window: 协方差匹配滑动窗口长度
            p0_diag: 初始P矩阵对角线 (位置方差, 速度方差)
            q_floor: Q矩阵元素下界（防止自适应为0）
            r_floor: R下界
        """
        self.dt = dt
        self.g1 = g1
        self.g2 = g2
        self.window = window
        self.q_floor = q_floor
        self.r_floor = r_floor

        # ── 状态转移矩阵 F ──
        # [x1]   [1  dt] [x1]
        # [x2] = [0   1] [x2]
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        # ── 观测矩阵 H ──
        # z = [1  0] [x1 x2]^T
        self.H = np.array([[1.0, 0.0]])

        # ── 过程噪声输入向量 G ──
        self.G = np.array([[g1], [g2]])

        # ── 初始噪声 ──
        self._q_scalar = q_scale
        self.Q = self._build_Q(q_scale)
        self.R = np.array([[r_init]])

        # ── 状态 & 协方差 ──
        self.X = np.zeros((2, 1))
        self.P = np.diag([p0_diag[0], p0_diag[1]])

        # ── 自适应缓存 ──
        self._innovations = []     # 新息历史
        self._q1_samples = []      # 位置过程噪声样本
        self._q2_samples = []      # 速度过程噪声样本
        self._prev_X_post = None   # 上一步后验状态

        # ── 上一步速度（用于计算加速度）──
        self._prev_velocity = None

        self._step = 0
        self._initialized = False

    def _build_Q(self, q_scalar: float) -> np.ndarray:
        """从标量构建 Q 矩阵。Q = G * q * G^T"""
        return self.G @ np.array([[q_scalar]]) @ self.G.T

    def initialize(self, z0: float, velocity0: float = 0.0) -> None:
        """用第一个观测值初始化滤波器。

        Args:
            z0: 第一个 ln(price)
            velocity0: 初始速度（默认0，表示不知道趋势方向）
        """
        self.X = np.array([[z0], [velocity0]])
        self._prev_X_post = self.X.copy()
        self._prev_velocity = velocity0
        self._initialized = True
        self._step = 1

    def update(self, z: float) -> KalmanState:
        """单步更新（核心循环）。

        Args:
            z: 当前时刻的 ln(price)

        Returns:
            KalmanState 包含所有估计结果
        """
        if not self._initialized:
            self.initialize(z)
            return self._make_state(acceleration=0.0)

        # ══════════════════════════════════════════════
        # 1. 预测 (Time Update / a priori)
        # ══════════════════════════════════════════════
        X_pred = self.F @ self.X                          # (2,1)
        P_pred = self.F @ self.P @ self.F.T + self.Q     # (2,2)

        # ══════════════════════════════════════════════
        # 2. 新息 (Innovation)
        # ══════════════════════════════════════════════
        z_pred = (self.H @ X_pred)[0, 0]
        innovation = z - z_pred

        # ══════════════════════════════════════════════
        # 3. 增益 (Gain Update)
        # ══════════════════════════════════════════════
        S = self.H @ P_pred @ self.H.T + self.R           # (1,1) 新息协方差
        S_inv = 1.0 / S[0, 0]
        K = P_pred @ self.H.T @ np.array([[S_inv]])       # (2,1) 增益

        k1 = K[0, 0]
        k2 = K[1, 0]

        # ══════════════════════════════════════════════
        # 4. 修正 (Measurement Update / a posteriori)
        # ══════════════════════════════════════════════
        y = np.array([[innovation]])
        X_post = X_pred + K * y

        # Joseph 稳定化形式（减少数值误差）
        I_KH = np.eye(2) - K @ self.H
        P_post = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        # 保证 P 对称
        P_post = 0.5 * (P_post + P_post.T)

        # ══════════════════════════════════════════════
        # 5. 自适应噪声估计 (Myers-Tapley)
        # ══════════════════════════════════════════════
        self._adapt_noise(innovation, X_post)

        # ══════════════════════════════════════════════
        # 6. 加速度估计
        # ══════════════════════════════════════════════
        current_velocity = X_post[1, 0]
        if self._prev_velocity is not None:
            acceleration = (current_velocity - self._prev_velocity) / self.dt
        else:
            acceleration = 0.0
        self._prev_velocity = current_velocity

        # ══════════════════════════════════════════════
        # 7. 状态保存
        # ══════════════════════════════════════════════
        self.X = X_post
        self.P = P_post
        self._step += 1

        return self._make_state(
            acceleration=acceleration,
            innovation=innovation,
            k1=k1, k2=k2,
        )

    def _adapt_noise(self, innovation: float, X_post: np.ndarray) -> None:
        """Myers-Tapley 协方差匹配法。

        用滑动窗口内的:
          - 新息 (innovation) 样本估计 R
          - 过程噪声样本估计 Q

        前 window-1 步用扩展窗口（逐步增长），之后用固定窗口。
        """
        # ── 收集过程噪声样本 ──
        if self._prev_X_post is not None:
            # q1: 位置过程噪声 = [x1+(t) - (x1+(t-1) + x2+(t-1)*dt)] / g1
            expected_pos = self._prev_X_post[0, 0] + self._prev_X_post[1, 0] * self.dt
            q1 = (X_post[0, 0] - expected_pos) / self.g1 if abs(self.g1) > 1e-12 else 0.0
            # q2: 速度过程噪声 = [x2+(t) - x2+(t-1)] / g2
            q2 = (X_post[1, 0] - self._prev_X_post[1, 0]) / self.g2 if abs(self.g2) > 1e-12 else 0.0

            self._q1_samples.append(q1)
            self._q2_samples.append(q2)

        self._innovations.append(innovation)
        self._prev_X_post = X_post.copy()

        n_samples = len(self._innovations)
        if n_samples < 2:
            return

        # ── 滑动窗口 vs 扩展窗口 ──
        w = min(n_samples, self.window)

        # ── R 估计（观测噪声方差）──
        innov_window = self._innovations[-w:]
        innov_mean = np.mean(innov_window)
        r_est = np.sum((np.array(innov_window) - innov_mean) ** 2) / (w - 1)
        # 减去 P_pred 在观测方向的投影（去偏）
        # 简化：直接用新息方差作为 R 上界
        r_est = max(abs(r_est), self.r_floor)
        self.R = np.array([[r_est]])

        # ── Q 估计（过程噪声协方差）──
        q1_win = self._q1_samples[-w:]
        q2_win = self._q2_samples[-w:]
        q1_mean = np.mean(q1_win)
        q2_mean = np.mean(q2_win)

        q11 = np.sum((np.array(q1_win) - q1_mean) ** 2) / (w - 1)
        q22 = np.sum((np.array(q2_win) - q2_mean) ** 2) / (w - 1)
        q12 = np.sum(
            (np.array(q1_win) - q1_mean) * (np.array(q2_win) - q2_mean)
        ) / (w - 1)

        # 绝对值防止负定
        q11 = max(abs(q11), self.q_floor)
        q22 = max(abs(q22), self.q_floor)
        q12 = np.clip(q12, -np.sqrt(q11 * q22), np.sqrt(q11 * q22))

        Q_raw = np.array([[q11, q12], [q12, q22]])
        # 映射到 Q = G * q * G^T 结构（保持模型一致性）
        # G = [g1; g2], Q = [[g1^2*q, g1*g2*q], [g1*g2*q, g2^2*q]]
        # q_scalar = trace(Q_raw) / (g1^2 + g2^2)
        q_scalar = max(
            (Q_raw[0, 0] + Q_raw[1, 1]) / (self.g1 ** 2 + self.g2 ** 2),
            self.q_floor,
        )
        self.Q = self._build_Q(q_scalar)

    def _make_state(
        self,
        acceleration: float = 0.0,
        innovation: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
    ) -> KalmanState:
        return KalmanState(
            position=self.X[0, 0],
            velocity=self.X[1, 0],
            acceleration=acceleration,
            var_position=self.P[0, 0],
            var_velocity=self.P[1, 1],
            gain_position=k1,
            gain_velocity=k2,
            Q=self.Q.copy(),
            R=float(self.R[0, 0]),
            innovation=innovation,
            step=self._step,
        )


# ═══════════════════════════════════════════════════════════════════
#  批量接口：兼容 V3 系统
# ═══════════════════════════════════════════════════════════════════

def kalman_derivatives(
    close: np.ndarray,
    dt: float = 1.0,
    q_scale: float = 1e-5,
    r_init: float = 1e-4,
    window: int = 10,
    g1: float = 0.5,
    g2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """对价格序列计算卡尔曼 f' 和 f''，接口兼容 V3 compute_derivatives。

    与 V3 Savgol 接口完全对齐：
      - 输入：close 价格数组
      - 输出：(f1, f2) 数组
        f1 × 100 ≈ 日收益率%   （与V3 Savgol一致）
        f2 × 10000 ≈ 无量纲加速度（与V3 Savgol一致）

    Args:
        close: 收盘价序列（按时间升序）
        dt: 时间步长
        q_scale: 初始过程噪声标度
        r_init: 初始观测噪声
        window: 自适应窗口
        g1: 位置噪声系数
        g2: 速度噪声系数

    Returns:
        (f1, f2): 一阶导(×100), 二阶导(×10000) 数组，长度=len(close)
    """
    n = len(close)
    if n == 0:
        return np.array([]), np.array([])

    kf = AdaptiveKalmanFilter2S(
        dt=dt, q_scale=q_scale, r_init=r_init,
        window=window, g1=g1, g2=g2,
    )

    ln_p = np.log(close.astype(float))

    f1_raw = np.zeros(n)   # 原始速度
    f2_raw = np.zeros(n)   # 原始加速度

    for i in range(n):
        state = kf.update(ln_p[i])
        f1_raw[i] = state.velocity
        f2_raw[i] = state.acceleration

    # 与 V3 对齐的缩放
    f1 = f1_raw * 100      # ≈ 日收益率%
    f2 = f2_raw * 10000    # 无量纲加速度

    return f1, f2


def kalman_derivatives_with_uncertainty(
    close: np.ndarray,
    dt: float = 1.0,
    q_scale: float = 1e-5,
    r_init: float = 1e-4,
    window: int = 10,
    g1: float = 0.5,
    g2: float = 1.0,
) -> dict:
    """带不确定性的卡尔曼导数估计。

    Returns:
        dict with:
          f1, f2: 导数（×100, ×10000 缩放后）
          var_f1, var_f2: 导数方差（同缩放）
          position, var_position: 滤波位置 + 方差
          gains: (k1, k2) 增益历史
          Q_trace, R_history: 噪声历史（调试用）
    """
    n = len(close)
    if n == 0:
        return {}

    kf = AdaptiveKalmanFilter2S(
        dt=dt, q_scale=q_scale, r_init=r_init,
        window=window, g1=g1, g2=g2,
    )

    ln_p = np.log(close.astype(float))

    f1_raw = np.zeros(n)
    f2_raw = np.zeros(n)
    var_f1 = np.zeros(n)
    var_f2 = np.zeros(n)
    position = np.zeros(n)
    var_position = np.zeros(n)
    k1_hist = np.zeros(n)
    k2_hist = np.zeros(n)
    q_trace = np.zeros(n)
    r_hist = np.zeros(n)

    for i in range(n):
        state = kf.update(ln_p[i])
        f1_raw[i] = state.velocity
        f2_raw[i] = state.acceleration
        var_f1[i] = state.var_velocity
        var_f2[i] = state.var_velocity  # 加速度方差 ≈ 速度方差/dt² (近似)
        position[i] = state.position
        var_position[i] = state.var_position
        k1_hist[i] = state.gain_position
        k2_hist[i] = state.gain_velocity
        q_trace[i] = np.trace(state.Q)
        r_hist[i] = state.R

    return {
        'f1': f1_raw * 100,
        'f2': f2_raw * 10000,
        'var_f1': var_f1 * 100 ** 2,
        'var_f2': var_f2 * 10000 ** 2,
        'position': position,
        'var_position': var_position,
        'gains': np.column_stack([k1_hist, k2_hist]),
        'Q_trace': q_trace,
        'R_history': r_hist,
    }


# ═══════════════════════════════════════════════════════════════════
#  退化处理 & 边界情况
# ═══════════════════════════════════════════════════════════════════

def kalman_derivatives_safe(
    close: np.ndarray,
    min_length: int = 5,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """安全的卡尔曼导数计算，处理边界情况。

    处理：
      - 空数组 / 长度不足
      - NaN / Inf
      - 零价格 / 负价格
      - 停牌（价格不变）

    Returns:
        (f1, f2) 长度=len(close)，无效位置填 NaN
    """
    n = len(close)
    nan_arr = np.full(n, np.nan)

    if n < min_length:
        return nan_arr.copy(), nan_arr.copy()

    # 检查价格有效性
    valid_mask = np.isfinite(close) & (close > 0)
    if not np.any(valid_mask):
        return nan_arr.copy(), nan_arr.copy()

    # 对无效价格做前向填充
    close_clean = close.astype(float).copy()
    last_valid = None
    for i in range(n):
        if valid_mask[i]:
            last_valid = close_clean[i]
        elif last_valid is not None:
            close_clean[i] = last_valid
        else:
            # 开头就没有有效值，用第一个有效值回填
            first_valid_idx = np.argmax(valid_mask)
            close_clean[i] = close_clean[first_valid_idx]

    f1, f2 = kalman_derivatives(close_clean, **kwargs)

    # 标记无效位置
    f1[~valid_mask] = np.nan
    f2[~valid_mask] = np.nan

    return f1, f2


# ═══════════════════════════════════════════════════════════════════
#  与 V3 Savgol 的对比工具
# ═══════════════════════════════════════════════════════════════════

def compare_kalman_vs_savgol(
    close: np.ndarray,
    sg_window: int = 21,
    sg_poly: int = 3,
    kalman_window: int = 10,
    q_scale: float = 1e-5,
) -> dict:
    """对比卡尔曼滤波器 vs Savitzky-Golay 的导数估计。

    Returns:
        dict with f1_savgol, f2_savgol, f1_kalman, f2_kalman, metrics
    """
    from scipy.signal import savgol_filter

    n = len(close)
    ln_p = np.log(close.astype(float))

    # ── Savgol ──
    f1_sg = savgol_filter(ln_p, sg_window, sg_poly, deriv=1) * 100
    f2_sg = savgol_filter(ln_p, sg_window, sg_poly, deriv=2) * 10000

    # shift 消除前瞻偏差
    shift = sg_window // 2
    f1_sg_safe = np.empty_like(f1_sg)
    f2_sg_safe = np.empty_like(f2_sg)
    f1_sg_safe[:shift] = f1_sg[:shift]
    f2_sg_safe[:shift] = f2_sg[:shift]
    f1_sg_safe[shift:] = f1_sg[:-shift]
    f2_sg_safe[shift:] = f2_sg[:-shift]

    # ── Kalman ──
    f1_km, f2_km = kalman_derivatives(
        close, window=kalman_window, q_scale=q_scale,
    )

    # ── 统计对比（取稳定区间，跳过前30个点）──
    start = min(30, n - 1)
    metrics = {}
    if n > start + 10:
        # 滞后分析：卡尔曼导数与N步前的Savgol导数的相关性
        for lag in range(0, 15):
            corr = np.corrcoef(
                f1_km[start:],
                f1_sg_safe[start - lag: n - lag] if lag < n - start else f1_sg_safe[start:]
            )[0, 1] if lag < n - start else 0.0
            metrics[f'f1_corr_lag{lag}'] = corr

        # 噪声比：导数的标准差比（越小越平滑）
        metrics['f1_noise_ratio'] = float(np.std(f1_km[start:]) / max(np.std(f1_sg_safe[start:]), 1e-10))
        metrics['f2_noise_ratio'] = float(np.std(f2_km[start:]) / max(np.std(f2_sg_safe[start:]), 1e-10))

        # 符号一致性
        same_sign_f1 = np.mean(np.sign(f1_km[start:]) == np.sign(f1_sg_safe[start:]))
        same_sign_f2 = np.mean(np.sign(f2_km[start:]) == np.sign(f2_sg_safe[start:]))
        metrics['f1_sign_agreement'] = float(same_sign_f1)
        metrics['f2_sign_agreement'] = float(same_sign_f2)

        # 拐点检测一致性
        sg_turn = np.diff(np.sign(f2_sg_safe[start:]))
        km_turn = np.diff(np.sign(f2_km[start:]))
        sg_turn_points = np.where(sg_turn != 0)[0]
        km_turn_points = np.where(km_turn != 0)[0]
        metrics['sg_turn_count'] = len(sg_turn_points)
        metrics['km_turn_count'] = len(km_turn_points)

    return {
        'f1_savgol': f1_sg_safe,
        'f2_savgol': f2_sg_safe,
        'f1_kalman': f1_km,
        'f2_kalman': f2_km,
        'metrics': metrics,
    }


# ═══════════════════════════════════════════════════════════════════
#  自包含测试
# ═══════════════════════════════════════════════════════════════════

def _self_test():
    """自包含测试：验证滤波器基本功能。"""
    print("=" * 60)
    print("2-State Adaptive Kalman Filter — 自测试")
    print("=" * 60)

    np.random.seed(42)

    # ── 测试1: 线性趋势 ──
    print("\n[测试1] 线性趋势 y = 0.01*t + noise")
    n = 200
    t = np.arange(n, dtype=float)
    true_slope = 0.01
    ln_p = true_slope * t + np.random.normal(0, 0.02, n)
    close = np.exp(ln_p)

    f1_km, f2_km = kalman_derivatives(close)
    f1_last = f1_km[-1]
    f2_last = f2_km[-1]
    f1_mean = np.mean(f1_km[50:]) / 100  # 去掉缩放

    print(f"  真实斜率: {true_slope:.4f}")
    print(f"  卡尔曼f'(最后): {f1_last:.4f} (×100后), 恢复={f1_last/100:.4f}")
    print(f"  卡尔曼f'(均值50+): {f1_mean:.4f}")
    print(f"  误差: {abs(f1_mean - true_slope):.6f}")
    assert abs(f1_mean - true_slope) < 0.005, f"斜率误差过大: {abs(f1_mean - true_slope)}"
    print("  PASS")

    # ── 测试2: 正弦曲线（测试加速度跟踪）──
    print("\n[测试2] 正弦曲线 y = sin(t/20)")
    t = np.arange(300, dtype=float)
    ln_p = np.sin(t / 20.0) + np.random.normal(0, 0.01, 300)
    close = np.exp(ln_p)

    f1_km, f2_km = kalman_derivatives(close)

    # 真实导数
    true_f1 = np.cos(t / 20.0) / 20.0 * 100
    true_f2 = -np.sin(t / 20.0) / 400.0 * 10000

    # 相关系数
    corr_f1 = np.corrcoef(f1_km[50:], true_f1[50:])[0, 1]
    corr_f2 = np.corrcoef(f2_km[50:], true_f2[50:])[0, 1]
    print(f"  f' 相关系数: {corr_f1:.4f}")
    print(f"  f'' 相关系数: {corr_f2:.4f}")
    assert corr_f1 > 0.8, f"f'相关过低: {corr_f1}"
    print("  PASS")

    # ── 测试3: 阶跃变化（暴跌后横盘，V3痛点场景）──
    print("\n[测试3] 阶跃变化（暴跌后横盘）")
    ln_p = np.zeros(200)
    ln_p[:100] = np.log(10.0) + np.random.normal(0, 0.01, 100)
    ln_p[100:] = np.log(7.0) + np.random.normal(0, 0.01, 100)  # -30% 跳水
    close = np.exp(ln_p)

    f1_km, f2_km = kalman_derivatives(close)

    # 暴跌后30天，f' 应该回到接近0
    post_crash_f1 = np.mean(f1_km[130:150])
    post_crash_f2 = np.mean(f2_km[130:150])
    print(f"  暴跌后30天 f': {post_crash_f1:.4f} (应接近0)")
    print(f"  暴跌后30天 f'': {post_crash_f2:.4f} (应接近0)")
    assert abs(post_crash_f1) < 0.5, f"暴跌后f'未恢复: {post_crash_f1}"
    print("  PASS")

    # ── 测试4: 对比 Savgol ──
    print("\n[测试4] 卡尔曼 vs Savgol 对比")
    t = np.arange(500, dtype=float)
    ln_p = 0.005 * t + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.015, 500)
    close = np.exp(ln_p)

    result = compare_kalman_vs_savgol(close)
    m = result['metrics']
    print(f"  零滞后相关性: {m.get('f1_corr_lag0', 0):.4f}")
    print(f"  f'噪声比(KM/SG): {m.get('f1_noise_ratio', 0):.4f}")
    print(f"  f'符号一致性: {m.get('f1_sign_agreement', 0):.4f}")
    print(f"  f''符号一致性: {m.get('f2_sign_agreement', 0):.4f}")
    print(f"  Savgol拐点数: {m.get('sg_turn_count', 0)}, 卡尔曼拐点数: {m.get('km_turn_count', 0)}")

    # ── 测试5: 边界情况 ──
    print("\n[测试5] 边界情况")
    # 空数组
    f1, f2 = kalman_derivatives_safe(np.array([]))
    assert len(f1) == 0
    print("  空数组: PASS")

    # 单元素
    f1, f2 = kalman_derivatives_safe(np.array([10.0]))
    assert len(f1) == 1
    print("  单元素: PASS")

    # 含NaN
    close_nan = np.array([10.0, 11.0, np.nan, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    f1, f2 = kalman_derivatives_safe(close_nan, min_length=3)
    assert np.isnan(f1[2])  # NaN位置应标记
    assert not np.isnan(f1[3])  # 有效位置不应NaN
    print("  含NaN: PASS")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  性能基准
# ═══════════════════════════════════════════════════════════════════

def benchmark(n_stocks: int = 1800, n_days: int = 500):
    """性能基准测试。

    模拟 1800 只股票 × 500 天日线数据的处理时间。
    """
    import time

    print(f"\n{'=' * 60}")
    print(f"性能基准: {n_stocks} 只股票 × {n_days} 天")
    print(f"{'=' * 60}")

    np.random.seed(123)
    total_points = n_stocks * n_days
    print(f"总数据点: {total_points:,}")

    # 生成模拟数据
    t0 = time.perf_counter()
    all_close = []
    for _ in range(n_stocks):
        ln_p = np.cumsum(np.random.normal(0.0003, 0.02, n_days))
        all_close.append(np.exp(ln_p))
    gen_time = time.perf_counter() - t0
    print(f"数据生成: {gen_time:.2f}s")

    # 卡尔曼滤波
    t0 = time.perf_counter()
    for close in all_close:
        kalman_derivatives(close)
    kf_time = time.perf_counter() - t0
    print(f"卡尔曼滤波: {kf_time:.2f}s ({total_points / kf_time:,.0f} points/s)")

    # 对比: Savgol
    t0 = time.perf_counter()
    from scipy.signal import savgol_filter
    for close in all_close:
        ln_p = np.log(close)
        savgol_filter(ln_p, 21, 3, deriv=1)
        savgol_filter(ln_p, 21, 3, deriv=2)
    sg_time = time.perf_counter() - t0
    print(f"Savgol:     {sg_time:.2f}s ({total_points / sg_time:,.0f} points/s)")

    print(f"\n速度比: 卡尔曼/Savgol = {kf_time / sg_time:.2f}x")

    # 内存估算
    # 每只股票: 2×1 state + 2×2 P + 2×2 Q + 窗口缓存 ~200 floats
    # 1800只: ~1.4M floats = ~11MB
    mem_per_stock = (2 + 4 + 4 + 10 * 3) * 8  # bytes
    total_mem = mem_per_stock * n_stocks
    print(f"内存估算: {total_mem / 1024 / 1024:.1f} MB (状态+缓存)")


if __name__ == "__main__":
    _self_test()
    benchmark()
