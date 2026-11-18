#!/usr/bin/env python3
"""L1 趋势滤波 — 客观拐点基准。

基于 Kim/Koh/Boyd (2009) 的 L1 趋势滤波，用凸优化自动检测价格趋势拐点。
作为 Savgol/Kalman 导数系统的独立验证基准。

核心优势：
  - 自动检测拐点位置（无需阈值调参）
  - 客观分段：每段内趋势是线性的，拐点处斜率突变
  - 不依赖窗口参数（仅lambda控制稀疏度）

与 V3 的关系：
  - V3 的 Savgol f'' 是连续估计，L1 是离散拐点检测
  - L1 检测到的拐点可作为"ground truth"评估导数系统的准确性
  - 不替代 V3 评分，仅作为验证层

用法：
  from scripts.analyzers.l1_trend_filter import L1TrendFilter
  l1 = L1TrendFilter(lam=20)
  result = l1.filter(close_prices)
  print(result.kinks)  # 拐点位置
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse.linalg import splu

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


@dataclass
class L1Result:
    """L1 趋势滤波结果。"""
    trend: np.ndarray       # 滤波后趋势
    Dx: np.ndarray          # 二阶差分（非零处 = 拐点）
    kinks: np.ndarray       # 拐点位置索引
    n_segments: int         # 分段数
    slopes: list[float]     # 每段斜率
    segments: list[tuple]   # [(start, end, slope), ...]


def _build_D(n: int) -> sparse.csc_matrix:
    """二阶差分矩阵 (n-2) x n, CSC 格式。"""
    e = np.ones(n)
    D = sparse.diags([e[1:], -2 * e, e[1:]], [1, 0, -1], shape=(n, n), format='csc')
    return D[1:-1]


def _l1_tf_cvxpy(y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray] | None:
    """cvxpy 精确求解（CLARABEL solver）。"""
    if not HAS_CVXPY:
        return None
    n = len(y)
    y = np.asarray(y, dtype=float)
    x = cp.Variable(n)
    D = _build_D(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x) + lam * cp.norm1(D @ x))
    prob = cp.Problem(obj)
    try:
        prob.solve(solver='CLARABEL')
    except Exception:
        try:
            prob.solve()
        except Exception:
            return None
    if x.value is None:
        return None
    trend = x.value
    Dx = np.asarray(D @ trend).flatten()
    return trend, Dx


def _l1_tf_admm(y: np.ndarray, lam: float, max_iter: int = 2000,
                 eps_abs: float = 1e-6, eps_rel: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """ADMM 求解（cvxpy 不可用时的回退方案）。"""
    n = len(y)
    y = np.asarray(y, dtype=float)
    D = _build_D(n)

    rho = 1.0
    DtD = D.T @ D

    def factor(rho_val):
        A = sparse.eye(n, format='csc') + rho_val * DtD
        return splu(A.tocsc())

    A_lu = factor(rho)
    z = np.zeros(n - 2)
    u = np.zeros(n - 2)
    x = y.copy()

    mu, tau = 10, 2.0

    for _ in range(max_iter):
        rhs = y + rho * D.T.dot(z - u)
        x_new = A_lu.solve(rhs)
        Dx = D.dot(x_new)
        z_new = np.sign(Dx + u) * np.maximum(np.abs(Dx + u) - lam / rho, 0)
        u_new = u + Dx - z_new

        r_norm = np.linalg.norm(Dx - z_new)
        s_norm = np.linalg.norm(-rho * D.T.dot(z_new - z))

        eps_pri = np.sqrt(n - 2) * eps_abs + eps_rel * max(np.linalg.norm(Dx), np.linalg.norm(z_new))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(rho * D.T.dot(u_new))

        if r_norm > mu * s_norm:
            rho *= tau
            u_new /= tau
            A_lu = factor(rho)
        elif s_norm > mu * r_norm:
            rho /= tau
            u_new *= tau
            A_lu = factor(rho)

        x, z, u = x_new, z_new, u_new

        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return x, D.dot(x)


class L1TrendFilter:
    """L1 趋势滤波器。"""

    def __init__(self, lam: float = 20.0, kink_threshold: float = 1e-4):
        """
        Args:
            lam: 正则化参数，越大拐点越少。推荐：蓝筹20，中小盘20，题材股10-20
            kink_threshold: Dx 绝对值超过此阈值判定为拐点
        """
        self.lam = lam
        self.kink_threshold = kink_threshold

    def filter(self, close: np.ndarray) -> L1Result:
        """对价格序列做 L1 趋势滤波。

        Args:
            close: 收盘价序列（按时间升序）

        Returns:
            L1Result 包含趋势、拐点、分段信息
        """
        ln_p = np.log(close.astype(float))
        n = len(ln_p)

        # 优先用 cvxpy（精确），回退到 ADMM
        result = _l1_tf_cvxpy(ln_p, self.lam)
        if result is not None:
            trend, Dx = result
        else:
            trend, Dx = _l1_tf_admm(ln_p, self.lam)

        # 检测拐点
        kink_mask = np.abs(Dx) > self.kink_threshold
        kinks = np.where(kink_mask)[0] + 1  # +1 因为 Dx 长度是 n-2
        kinks = kinks[kinks < n]  # 边界保护

        # 分段斜率
        breaks = np.concatenate([[0], kinks, [n - 1]])
        segments = []
        slopes = []
        for i in range(len(breaks) - 1):
            s, e = int(breaks[i]), int(breaks[i + 1])
            if e <= s:
                continue
            seg = trend[s:e + 1]
            slope = float((seg[-1] - seg[0]) / max(e - s, 1))
            segments.append((s, e, slope))
            slopes.append(slope)

        return L1Result(
            trend=trend,
            Dx=Dx,
            kinks=kinks,
            n_segments=len(segments),
            slopes=slopes,
            segments=segments,
        )

    def last_kink_info(self, close: np.ndarray) -> dict:
        """获取最近拐点信息（用于实时判断）。

        Returns:
            dict: {last_kink_bar, bars_since_kink, current_slope, slope_direction}
        """
        result = self.filter(close)
        n = len(close)

        if len(result.kinks) == 0:
            return {
                "last_kink_bar": None,
                "bars_since_kink": n,
                "current_slope": result.slopes[-1] if result.slopes else 0.0,
                "slope_direction": "flat",
                "n_segments": result.n_segments,
            }

        last_kink = int(result.kinks[-1])
        # 当前所在分段
        for s, e, slope in result.segments:
            if s <= last_kink <= e or s <= n - 1 <= e:
                current_slope = slope
                break
        else:
            current_slope = result.slopes[-1] if result.slopes else 0.0

        direction = "up" if current_slope > 0.001 else ("down" if current_slope < -0.001 else "flat")

        return {
            "last_kink_bar": last_kink,
            "bars_since_kink": n - 1 - last_kink,
            "current_slope": round(current_slope, 5),
            "slope_direction": direction,
            "n_segments": result.n_segments,
        }
