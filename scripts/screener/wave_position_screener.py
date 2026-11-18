# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

#!/usr/bin/env python3
"""W3主升浪启动检测器 — 纯数学特征，无ElliottWaveAnalyzer。

核心思路：W3是波浪理论中最强的主升浪。不依赖完整5浪识别，
而是通过swing结构+数学特征直接定位W3启动点。

数学定义：
  W3候选 = swing结构(Low→High→Low→突破High)
         + W2回撤在Fibonacci [0.382, 0.618]区间
         + f'>0（趋势向上）且 f''>0（加速）
         + 放量（vol5 > vol20 × 1.2）
         + 前期有显著下跌（距250日高点>25%）

架构：
  5000只 → L0 swing结构预筛选(~200只, 5秒)
         → L1 Fibonacci+导数+量能评分(~50只, 1秒)
         → 排序输出

用法：
  from scripts.screener.wave_position_screener import WavePositionScreener
  wp = WavePositionScreener(data_root=Path("data/store"))
  results = wp.scan(codes)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from scipy.signal import argrelextrema


@dataclass
class WavePositionResult:
    """单只股票的W3分析结果。"""
    code: str = ""
    position: str = ""              # w3_breakout / w2_bottom / w1_up
    composite_score: float = 0.0    # 0-100
    grade: str = ""                 # A/B/C
    details: dict = field(default_factory=dict)


class WavePositionScreener:
    """W3主升浪启动检测器（纯数学特征版）。"""

    def __init__(self, data_root: str | Path | None = None):
        self.data_root = Path(data_root) if data_root else None

    # ── 批量扫描 ──

    def scan(
        self,
        codes: list[str],
        min_score: float = 50.0,
        top: int = 30,
        as_of_date: str | None = None,
    ) -> pd.DataFrame:
        if self.data_root is None:
            return pd.DataFrame()

        results = []
        for code in codes:
            try:
                r = self._detect(code, as_of_date)
                if r is not None and r.composite_score >= min_score:
                    results.append(r)
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        results.sort(key=lambda r: r.composite_score, reverse=True)
        results = results[:top]

        print(f"  [W3检测] {len(codes)} → {len(results)}只通过(≥{min_score}分)")

        rows = []
        for r in results:
            rows.append({
                "code": r.code,
                "position": r.position,
                "composite_score": round(r.composite_score, 1),
                "grade": r.grade,
                **r.details,
            })
        return pd.DataFrame(rows)

    # ── 单只检测 ──

    def _detect(self, code: str, as_of_date: str | None = None) -> WavePositionResult | None:
        path = self.data_root / "kline" / code / "D.parquet"
        if not path.exists():
            return None

        df = pd.read_parquet(path)
        dt_col = "trade_date" if "trade_date" in df.columns else "date"
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col).reset_index(drop=True)
        if as_of_date:
            df = df[df[dt_col] <= pd.to_datetime(as_of_date)].reset_index(drop=True)
        if len(df) < 60:
            return None

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float) if "high" in df.columns else close
        low = df["low"].values.astype(float) if "low" in df.columns else close
        n = len(close)

        # ── 1. Swing结构检测 ──
        swing = self._find_w1_w2_structure(close, high, low)
        if swing is None:
            return None

        w1_low_idx = swing["w1_low_idx"]
        w1_low = swing["w1_low"]
        w1_high_idx = swing["w1_high_idx"]
        w1_high = swing["w1_high"]
        w2_low_idx = swing["w2_low_idx"]
        w2_low = swing["w2_low"]
        w2_retrace = swing["w2_retrace"]

        # ── 2. 前期下跌验证 ──
        lookback = min(250, n)
        prior_high = float(np.max(high[max(0, w1_low_idx - lookback):w1_low_idx + 1]))
        if prior_high <= 0:
            return None
        prior_decline = (prior_high - w1_low) / prior_high
        if prior_decline < 0.25:
            return None  # 前期跌幅不够，不是真正的底部反转

        # ── 3. 趋势方向验证 ──
        # W2底部/W3突破意味着趋势应该向上，不能还在持续下跌
        current_price = close[-1]
        if current_price < w2_low:
            return None  # 当前价低于W2低点，下跌趋势未变

        # 近10日趋势：必须整体向上或至少持平
        if n >= 15:
            recent_trend = close[-1] - close[-10]
            if recent_trend < -0.02 * close[-10]:
                return None  # 近10日跌幅>2%，还在跌

        # ── 4. 位置分类 ──
        w1_range = w1_high - w1_low
        if w1_range <= 0:
            return None

        breakout_pct = (current_price / w1_high - 1) * 100
        if breakout_pct > 15:
            return None  # 已涨太多，不是"W3刚启动"
        if current_price > w1_high * 1.01:
            position = "w3_breakout"
        elif current_price > w1_high * 0.97:
            position = "w3_breakout"  # 接近突破也算
        elif abs(current_price - w2_low) / max(current_price, 0.01) < 0.10:
            position = "w2_bottom"
        else:
            return None  # 不在关键位置

        # ── 4. 数学特征评分 ──
        score = self._score_math_features(
            close, high, low, n, position,
            w1_low, w1_high, w1_low_idx, w1_high_idx,
            w2_low, w2_low_idx, w2_retrace, prior_decline, code,
        )

        grade = "A" if score >= 70 else ("B" if score >= 55 else "C")

        # Fibonacci位标签
        fib_levels = {0.382: "浅回调", 0.5: "标准", 0.618: "深回调", 0.786: "极深"}
        fib_tag = ""
        for fv, fn in fib_levels.items():
            if abs(w2_retrace - fv) < 0.08:
                fib_tag = f"{fv:.3f}({fn})"
                break

        return WavePositionResult(
            code=code,
            position=position,
            composite_score=score,
            grade=grade,
            details={
                "w1_high": round(w1_high, 2),
                "w1_high_date": str(df.iloc[w1_high_idx][dt_col])[:10],
                "w2_low": round(w2_low, 2),
                "w2_retrace": round(w2_retrace, 3),
                "w2_at_fib": fib_tag,
                "prior_decline": round(prior_decline * 100, 1),
                "breakout_pct": round((current_price / w1_high - 1) * 100, 1),
                "w1_gain_pct": round((w1_high / w1_low - 1) * 100, 1),
            },
        )

    # ── Swing结构检测 ──

    def _find_w1_w2_structure(self, close, high, low):
        """找最近的 W1(上升)-W2(回调) swing结构。

        严格约束（防止假阳性）：
          1. W1涨幅: 5%~50%（超过50%不是底部反转，是趋势延续）
          2. W1持续时间: ≥10天（排除几天的脉冲噪声）
          3. W2持续时间: ≥5天（排除瞬时回撤）
          4. W2回撤: Fibonacci 23.6%~78.6%
          5. W1低点必须是前期下跌的真正底部（近120日最低点附近）
        """
        n = len(close)
        if n < 60:
            return None

        # order=8 约等于1.5周，过滤掉噪声swing
        local_max_idx = argrelextrema(high, np.greater, order=8)[0]
        local_min_idx = argrelextrema(low, np.less, order=8)[0]

        if len(local_max_idx) < 2 or len(local_min_idx) < 2:
            return None

        # 前期下跌底部参考：近120日最低价及其位置
        lookback = min(120, n - 1)
        recent_low = float(np.min(low[-lookback:]))
        # 找到最低点的索引
        recent_low_region = low[-lookback:]
        recent_low_idx = len(close) - lookback + int(np.argmin(recent_low_region))

        # 从后往前找第一个合格的 W1-W2 结构
        for hi_pos in range(len(local_max_idx) - 1, -1, -1):
            w1_high_idx = local_max_idx[hi_pos]
            w1_high = float(high[w1_high_idx])

            # 找W1_high之后的低点(W2)
            later_mins = local_min_idx[local_min_idx > w1_high_idx]
            if len(later_mins) == 0:
                continue
            w2_low_idx = int(later_mins[0])
            w2_low = float(low[w2_low_idx])

            # 找W1_high之前的低点(W1起点)
            earlier_mins = local_min_idx[local_min_idx < w1_high_idx]
            if len(earlier_mins) == 0:
                continue
            w1_low_idx = int(earlier_mins[-1])
            w1_low = float(low[w1_low_idx])

            if w1_low <= 0:
                continue

            # ── 约束1: W1涨幅 5%~40% ──
            w1_gain = (w1_high / w1_low - 1)
            if w1_gain < 0.05 or w1_gain > 0.40:
                continue

            # ── 约束2: W1持续时间 ≥10天 ──
            w1_duration = w1_high_idx - w1_low_idx
            if w1_duration < 10:
                continue

            # ── 约束3: W2持续时间 ≥5天 ──
            w2_duration = w2_low_idx - w1_high_idx
            if w2_duration < 5:
                continue

            # W2必须在当前价格之前（不能是未来数据）
            if w2_low_idx >= len(close) - 1:
                continue

            # ── 约束4: W2回撤在Fibonacci区间 ──
            w1_range = w1_high - w1_low
            if w1_range <= 0:
                continue
            w2_retrace = (w1_high - w2_low) / w1_range
            if w2_retrace < 0.236 or w2_retrace > 0.786:
                continue

            # ── 约束5: W1低点必须靠近前期下跌底部 ──
            # W1低点距近120日最低点不超过15%
            if recent_low > 0:
                dist_to_bottom = (w1_low - recent_low) / recent_low
                if dist_to_bottom > 0.15:
                    continue
            # W1低点必须在recent_low_idx附近（前后30天内）
            if abs(w1_low_idx - recent_low_idx) > 30:
                continue

            return {
                "w1_low_idx": w1_low_idx,
                "w1_low": w1_low,
                "w1_high_idx": w1_high_idx,
                "w1_high": w1_high,
                "w2_low_idx": w2_low_idx,
                "w2_low": w2_low,
                "w2_retrace": w2_retrace,
            }

        return None

    # ── 数学特征评分 ──

    def _score_math_features(
        self, close, high, low, n, position,
        w1_low, w1_high, w1_low_idx, w1_high_idx,
        w2_low, w2_low_idx, w2_retrace, prior_decline, code,
    ) -> float:
        """纯数学特征评分。总分0-100。

        四维度：Fibonacci(0-20) + 导数(0-25) + 突破位置(0-25) + 浪结构(0-30)
        """

        # ── 1. Fibonacci回撤质量 (0-20) ──
        fib_score = 0
        for target, pts in [(0.618, 20), (0.5, 18), (0.382, 16), (0.786, 10)]:
            if abs(w2_retrace - target) < 0.05:
                fib_score = pts
                break
        if fib_score == 0:
            fib_score = 3

        # ── 2. 导数特征 (0-25) ──
        from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
        f1, f2 = V3DerivativeScorer.compute_derivatives(close)
        f1_val = float(f1[-1])
        f2_val = float(f2[-1])

        deriv_score = 0
        if f1_val > 0 and f2_val > 0:
            deriv_score = 25  # 主升浪特征: 加速上涨
        elif f1_val > 0 and f2_val <= 0:
            deriv_score = 15  # 上涨但减速
        elif f1_val <= 0 and f2_val > 0:
            deriv_score = 10  # 拐点: 还在跌但加速度转正

        # ── 3. 突破/位置得分 (0-25) ──
        # W3刚启动：突破0-5%最理想，>10%太远
        if position == "w3_breakout":
            bp = (close[-1] / w1_high - 1) * 100
            if 0 <= bp <= 3:
                breakout_score = 25  # 刚突破，最佳时机
            elif 3 < bp <= 5:
                breakout_score = 20
            elif 5 < bp <= 8:
                breakout_score = 12
            elif bp < 0:
                breakout_score = 8   # 接近但未突破
            else:
                breakout_score = 5   # >8%，已经涨远了
        else:  # w2_bottom
            dist_to_w2 = abs(close[-1] - w2_low) / max(close[-1], 0.01) * 100
            if dist_to_w2 < 3:
                breakout_score = 22
            elif dist_to_w2 < 8:
                breakout_score = 14
            else:
                breakout_score = 5

        # ── 4. 浪结构质量 (0-30) ──
        # 4a. W1强度 (0-15): 15-30%最理想(典型W1), 太小无意义
        w1_gain = (w1_high / w1_low - 1) * 100
        if 15 <= w1_gain <= 30:
            w1_score = 15
        elif 10 <= w1_gain < 15 or 30 < w1_gain <= 40:
            w1_score = 10
        else:
            w1_score = 4

        # 4b. 前期跌幅 (0-10): >40%最好（真正的底部反转）
        if prior_decline > 0.40:
            decline_score = 10
        elif prior_decline > 0.30:
            decline_score = 7
        else:
            decline_score = 3

        # 4c. W1持续时间 (0-5): 15-40天最典型
        w1_days = w1_high_idx - w1_low_idx
        if 15 <= w1_days <= 40:
            time_score = 5
        elif 10 <= w1_days < 15 or 40 < w1_days <= 60:
            time_score = 3
        else:
            time_score = 1

        structure_score = w1_score + decline_score + time_score

        total = fib_score + deriv_score + breakout_score + structure_score

        return min(total, 100)

    # ── 格式化输出 ──

    def format_report(self, results_df: pd.DataFrame) -> str:
        if len(results_df) == 0:
            return "未找到W3主升浪启动候选股。"

        lines = []
        lines.append(f"\U0001f30a **W3主升浪启动检测** | {len(results_df)}只")
        lines.append("─" * 50)

        for i, (_, r) in enumerate(results_df.iterrows()):
            pos_label = "W3突破" if r["position"] == "w3_breakout" else "W2底部"
            grade_icon = "⭐" if r["grade"] == "A" else ("✅" if r["grade"] == "B" else "\U0001f4ca")
            fib_tag = f" Fib={r.get('w2_at_fib', '')}" if r.get("w2_at_fib") else ""
            lines.append(
                f"  {i+1}. {grade_icon} {r['code']} [{r['grade']}级] "
                f"{pos_label} 综合={r['composite_score']:.0f} "
                f"W1涨幅={r.get('w1_gain_pct', 0):.0f}% "
                f"W2回撤={r.get('w2_retrace', 0):.1%}"
                f"{fib_tag} "
                f"突破={r.get('breakout_pct', 0):+.1f}% "
                f"前期跌={r.get('prior_decline', 0):.0f}%"
            )

        return "\n".join(lines)
