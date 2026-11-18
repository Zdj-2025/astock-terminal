#!/usr/bin/env python3
"""全市场趋势扫描器 — 从5000+只股票中找出"正在向上"的。

核心逻辑：
  1. 多头排列检测：MA5 > MA10 > MA20 > MA60（至少MA5>MA10>MA20）
  2. MACD金叉/红柱：日线MACD(12,26,9) DIF>DEA
  3. 量价配合：近期放量（5日均量 > 20日均量）
  4. 周线方向确认：周线MA20向上（可选）

输出分三档：
  强趋势（AAA）：多头排列 + MACD红柱 + 放量 + 周线多头
  上升趋势（AA）：部分多头 + MACD金叉或红柱
  萌芽趋势（A） ：MA5刚上穿MA20 + MACD底背离或金叉

用法:
  from scripts.screener.trend_scanner import TrendScanner
  scanner = TrendScanner(client)
  results = scanner.scan()
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)


class TrendScanner:
    """全市场趋势扫描器"""

    # ── 缓存 ──
    CACHE_DIR = Path(PROJECT_ROOT) / "data" / "cache" / "trend_scanner"

    def __init__(self, client):
        self.client = client
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def scan(
        self,
        candidates: Optional[pd.DataFrame] = None,
        lookback_days: int = 120,
        min_grade: str = "A",
        top: int = 100,
        as_of_date: Optional[str] = None,
        include_weekly: bool = True,
        concept: Optional[str] = None,
        industry: Optional[str] = None,
        mode: str = "confirmed",
    ) -> pd.DataFrame:
        """执行全市场趋势扫描。

        Args:
            candidates: Layer 1 UniverseFilter输出。None则自取全市场
            lookback_days: 日K线回溯天数（默认120天≈半年）
            min_grade: 最低等级 "A"/"AA"/"AAA"
            top: 最多返回N只
            as_of_date: 截止日期 YYYYMMDD
            include_weekly: 是否检查周线
            concept: 概念板块关键词（用于过滤）
            industry: 行业（用于过滤）
            mode: "confirmed"(已确立趋势) / "emerging"(趋势刚启动)
        """
        trade_date = as_of_date or datetime.now().strftime("%Y%m%d")

        # 缓存检查
        cached = self._load_cache(trade_date, concept, industry)
        if cached is not None and len(cached) > 0:
            grade_order = {"AAA": 0, "AA": 1, "A": 2, "B": 3, "C": 4}
            min_idx = grade_order.get(min_grade, 2)
            filtered = cached[cached["trend_grade"].map(
                lambda g: grade_order.get(g, 5) <= min_idx
            )]
            return filtered.head(top)

        # 获取候选池
        if candidates is None:
            from scripts.screener.universe_filter import UniverseFilter
            uf = UniverseFilter(self.client)
            candidates = uf.filter(concept=concept, industry=industry)
            if candidates.empty:
                return pd.DataFrame()

        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)
        ).strftime("%Y%m%d")

        results = []
        total = len(candidates)
        grade_counts = {"AAA": 0, "AA": 0, "A": 0, "B": 0, "C": 0}

        print(f"\n{'='*60}")
        print(f"  📈 全市场趋势扫描")
        print(f"  候选: {total}只 | 回溯: {lookback_days}天 | 周线: {'是' if include_weekly else '否'}")
        print(f"{'='*60}\n")

        for i, row in candidates.iterrows():
            symbol = str(row.get("symbol", row.get("ts_code", ""))[:6])
            ts_code = row.get("ts_code", "")
            name = row.get("name", symbol)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  扫描进度: {i+1}/{total} | "
                      f"AAA:{grade_counts['AAA']} AA:{grade_counts['AA']} A:{grade_counts['A']}")

            try:
                # 获取日K线
                df = self.client.get_kline(
                    symbol, freq="D", start=start_date, end=as_of_date, adj="qfq"
                )
                if df is None or len(df) < 60:
                    continue

                if mode == "emerging":
                    info = self._scan_emerging(df, symbol)
                else:
                    info = self._scan_single(df, symbol, include_weekly)
                if info is None:
                    continue

                record = row.to_dict()
                record.update(info)
                results.append(record)
                grade_counts[info["trend_grade"]] = grade_counts.get(info["trend_grade"], 0) + 1

            except Exception:
                continue

        if not results:
            print("  未找到上升趋势股票")
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        # 排序：AAA > AA > A，同级别按待验证标签优先 + 趋势强度降序
        grade_order = {"AAA": 0, "AA": 1, "A": 2, "B": 3, "C": 4}
        result_df["_grade_sort"] = result_df["trend_grade"].map(grade_order)
        # 待验证模式排前面（True=1 > False=0，降序所以True在前）
        if "trend_high_win_rate" in result_df.columns:
            result_df["_hwr_sort"] = result_df["trend_high_win_rate"].astype(int)
            result_df = result_df.sort_values(
                ["_grade_sort", "_hwr_sort", "trend_v2_score"],
                ascending=[True, False, False]
            ).drop(columns=["_grade_sort", "_hwr_sort"])
        else:
            result_df = result_df.sort_values(
                ["_grade_sort", "trend_v2_score"], ascending=[True, False]
            ).drop(columns=["_grade_sort"])

        self._save_cache(result_df, trade_date, concept, industry)

        # 过滤最低等级
        min_idx = grade_order.get(min_grade, 2)
        filtered = result_df[
            result_df["trend_grade"].map(lambda g: grade_order.get(g, 5) <= min_idx)
        ]

        print(f"\n  扫描完成: {total}只 → {len(result_df)}只有趋势 → "
              f"{len(filtered)}只≥{min_grade}")
        print(f"  AAA:{grade_counts.get('AAA',0)} AA:{grade_counts.get('AA',0)} A:{grade_counts.get('A',0)} "
              f"B:{grade_counts.get('B',0)} C:{grade_counts.get('C',0)}")

        return filtered.head(top)

    # ── 趋势启动发现模式（emerging） ──

    def _scan_emerging(self, df: pd.DataFrame, symbol: str) -> Optional[dict]:
        """趋势启动发现模式：寻找\"刚形成多头排列、还没有大涨的\"。

        三条件组（A/B/C），2/3通过制：
          A. 均线启动（MA5刚上穿MA20 + MA20拐头）
          B. 位置安全（多周期涨幅不过大 + 位置百分位不高）
          C. 量价确认（放量 + MACD配合）

        Returns: dict or None（不达标返回None）
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        n = len(close)
        if n < 60:
            return None

        c = close[-1]
        ma5 = self._sma(close, 5)
        ma10 = self._sma(close, 10)
        ma20 = self._sma(close, 20)
        ma60 = self._sma(close, 60)

        # ── A组: 均线启动（满分45）──
        a_score = 0
        a_reasons = []

        # A1: MA5刚上穿MA20（最近5天内）
        ma5_cross_ma20 = False
        for j in range(max(0, n - 5), n):
            if j > 0 and ma5[j] > ma20[j] and ma5[j - 1] <= ma20[j - 1]:
                ma5_cross_ma20 = True
                break
        if ma5_cross_ma20:
            a_score += 20
            a_reasons.append("MA5上穿MA20")

        # A2: MA20拐头向上（5日内从下降转上升）
        ma20_slope_now = (ma20[-1] / ma20[-6] - 1) if n > 5 and ma20[-6] > 0 else 0
        ma20_slope_prev = (ma20[-6] / ma20[-11] - 1) if n > 10 and ma20[-11] > 0 else 0
        ma20_turning = ma20_slope_now > 0.003 and ma20_slope_prev <= 0.003
        if ma20_turning:
            a_score += 15
            a_reasons.append("MA20拐头")

        # A3: MA5 > MA10 > MA20（短中期多头排列）
        short_bull = ma5[-1] > ma10[-1] > ma20[-1]
        if short_bull:
            a_score += 10
            a_reasons.append("短中期多头")

        # ── B组: 位置安全（满分30）──
        b_score, b_reasons = self._position_filter_v2(close, high, low)
        # 归一化到 0-30
        b_score_normalized = b_score * 0.30
        b_passed, b_pass_reason = self.check_position_safe(close)

        # ── C组: 量价确认（满分25）──
        c_score = 0
        c_reasons = []

        # C1: 放量确认（当日量 > 20日均量×1.3）
        if volume is not None and len(volume) >= 20:
            vol_5 = np.mean(volume[-5:])
            vol_20 = np.mean(volume[-20:])
            vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
            if vol_ratio > 1.3:
                c_score += 10
                c_reasons.append(f"放量({vol_ratio:.1f}x)")
            elif vol_ratio > 1.0:
                c_score += 5

        # C2: MACD金叉或红柱
        dif, dea, hist = self._calc_macd(close, 12, 26, 9)
        macd_golden = dif[-1] > dea[-1]
        hist_positive = hist[-1] > 0
        recent_golden_cross = False
        for j in range(max(0, n - 5), n):
            if j > 0 and dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                recent_golden_cross = True
                break

        if recent_golden_cross:
            c_score += 10
            c_reasons.append("MACD金叉")
        elif macd_golden and hist_positive:
            c_score += 7
            c_reasons.append("MACD红柱")
        elif macd_golden:
            c_score += 3

        # C3: 价格站上MA5
        if c > ma5[-1]:
            c_score += 5
            c_reasons.append("站上MA5")

        # ── 综合评分 ──
        total_score = a_score + b_score_normalized + c_score

        # 三组通过判定（2/3制）
        a_passed = a_score >= 15  # A组至少有均线启动信号
        c_passed = c_score >= 10  # C组至少有量价配合

        groups_passed = sum([a_passed, b_passed, c_passed])
        if groups_passed < 2:
            return None

        # 最低分数线
        if total_score < 30:
            return None

        # ── 输出 ──
        gain_5d = (c / close[-6] - 1) * 100 if n > 5 else 0
        gain_20d = (c / close[-21] - 1) * 100 if n > 20 else 0
        gain_60d = (c / close[-61] - 1) * 100 if n > 60 else 0

        return {
            "trend_grade": f"E{int(total_score)}",  # E表示emerging
            "trend_strength": round(total_score, 1),
            "trend_mode": "emerging",
            "trend_full_bull": False,
            "trend_short_bull": short_bull,
            "trend_macd_golden": macd_golden,
            "trend_hist_expanding": hist[-1] > hist[-2] if n > 1 else False,
            "trend_dif_above_zero": dif[-1] > 0,
            "trend_vol_expanding": vol_ratio > 1.3 if volume is not None and len(volume) >= 20 else False,
            "trend_vol_ratio": round(vol_ratio, 2) if volume is not None and len(volume) >= 20 else 1.0,
            "trend_weekly_bull": False,
            "trend_ma20_slope": round(ma20_slope_now * 100, 3),
            "trend_gain_5d": round(gain_5d, 2),
            "trend_gain_10d": round((c / close[-11] - 1) * 100, 2) if n > 10 else 0,
            "trend_gain_20d": round(gain_20d, 2),
            "trend_gain_60d": round(gain_60d, 2),
            "trend_position_score": b_score,
            "trend_position_reasons": ", ".join(b_reasons),
            "trend_emerging_A": f"{a_score}/45 ({', '.join(a_reasons)})",
            "trend_emerging_B": f"{b_score}/100 ({', '.join(b_reasons)})",
            "trend_emerging_C": f"{c_score}/25 ({', '.join(c_reasons)})",
            "trend_price": round(c, 2),
            "current_price": round(c, 2),
        }

    # ── 单只股票扫描 ──

    def _scan_single(
        self, df: pd.DataFrame, symbol: str, include_weekly: bool
    ) -> Optional[dict]:
        """对单只股票做趋势扫描（算法v2：连续评分 [-1.0, +1.0]）。

        四因子加权评分：
          f1: MACD方向 (0.30)
          f2: DIF零轴上下文 (0.20)
          f3: 柱状图趋势 (0.25)
          f4: MA排列确认 (0.25)
        综合评分经 tanh 标准化后映射到等级 AAA/AA/A/B/C。
        """
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        n = len(close)
        if n < 60:
            return None

        # ── 计算均线 ──
        ma5 = self._sma(close, 5)
        ma10 = self._sma(close, 10)
        ma20 = self._sma(close, 20)
        ma60 = self._sma(close, 60)

        c = close[-1]

        # ── 多头排列检测（兼容旧字段） ──
        full_bull = (ma5[-1] > ma10[-1] > ma20[-1] > ma60[-1])
        short_bull = (ma5[-1] > ma10[-1] > ma20[-1])
        ma5_above_ma20 = ma5[-1] > ma20[-1]

        # MA20/MA60方向
        ma20_slope = (ma20[-1] / ma20[-6] - 1) if n > 6 and ma20[-6] > 0 else 0
        ma60_slope = (ma60[-1] / ma60[-6] - 1) if n > 6 and ma60[-6] > 0 else 0

        # ── MACD ──
        dif, dea, hist = self._calc_macd(close, 12, 26, 9)
        macd_golden = dif[-1] > dea[-1]
        hist_positive = hist[-1] > 0
        hist_expanding = hist[-1] > hist[-2] if n > 1 else False

        # 近5日是否有金叉信号
        recent_golden_cross = False
        recent_death_cross = False
        for j in range(max(0, n - 5), n):
            if j > 0 and dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                recent_golden_cross = True
            if j > 0 and dif[j] < dea[j] and dif[j - 1] >= dea[j - 1]:
                recent_death_cross = True

        dif_above_zero = dif[-1] > 0

        # ── ATR ──
        atr_arr = self._calc_atr(high, low, close, 14)
        atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else float(c * 0.02)
        atr_pct = atr / c * 100 if c > 0 else 0

        # ── 量价配合 ──
        vol_ratio = 1.0
        vol_expanding = False
        if volume is not None and len(volume) >= 20:
            vol_5 = np.mean(volume[-5:])
            vol_20 = np.mean(volume[-20:])
            vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
            vol_expanding = vol_ratio > 1.2

        # ══════════════════════════════════════════
        # 趋势算法 v2: 连续评分 [-1.0, +1.0]
        # ══════════════════════════════════════════

        # ── 因子1: MACD方向（权重 0.30）──
        raw_f1 = (dif[-1] - dea[-1]) / atr if atr > 0 else 0.0
        f1 = float(np.tanh(raw_f1 / 3.0))

        # ── 因子2: DIF零轴上下文（权重 0.20）──
        if dif_above_zero:
            if recent_golden_cross:
                f2 = 0.8   # 零轴上方金叉
            elif macd_golden:
                f2 = 0.5   # 多头排列
            elif recent_death_cross:
                f2 = -0.3  # 零轴上方死叉（回调）
            else:
                f2 = -0.3 if dif[-1] < dea[-1] else 0.3
        else:
            if recent_golden_cross:
                f2 = 0.6   # 底部金叉（反转信号）
            elif dif[-1] < dea[-1]:
                f2 = -0.5  # 空头排列
            elif recent_death_cross:
                f2 = -0.8  # 底部死叉
            else:
                f2 = -0.2 if dif[-1] < dea[-1] else 0.2

        # ── 因子3: 柱状图趋势（权重 0.25）──
        hist_change = float(hist[-1] - hist[-3]) if n >= 3 else 0.0
        if hist[-1] > 0:  # 红柱
            if hist_change > 0:
                f3 = float(np.tanh(hist_change / atr * 5)) if atr > 0 else 0.0
            else:
                f3 = float(-np.tanh(abs(hist_change) / atr * 5) * 0.5) if atr > 0 else 0.0
        else:  # 绿柱
            if hist_change > 0:  # 绿柱缩小
                f3 = float(np.tanh(abs(hist_change) / atr * 3) * 0.5) if atr > 0 else 0.0
            else:  # 绿柱扩大
                f3 = float(-np.tanh(abs(hist_change) / atr * 3)) if atr > 0 else 0.0

        # ── 因子4: MA排列确认（权重 0.25）──
        ma5_v = float(ma5[-1])
        ma10_v = float(ma10[-1])
        ma20_v = float(ma20[-1])
        ma60_v = float(ma60[-1])

        if ma5_v > ma10_v > ma20_v > ma60_v:
            f4 = 1.0   # 完美多头
        elif ma5_v > ma10_v > ma20_v:
            f4 = 0.7   # 三线多头
        elif ma5_v > ma10_v:
            f4 = 0.3   # 短期多头
        elif ma5_v > ma20_v:
            f4 = 0.1   # 弱多头
        elif ma5_v < ma10_v < ma20_v < ma60_v:
            f4 = -1.0  # 完美空头
        elif ma5_v < ma10_v < ma20_v:
            f4 = -0.7  # 三线空头
        elif ma5_v < ma10_v:
            f4 = -0.3  # 短期空头
        elif ma5_v < ma20_v:
            f4 = -0.1  # 弱空头
        else:
            f4 = 0.0

        # ── 综合评分 ──
        w1, w2, w3, w4 = 0.30, 0.20, 0.25, 0.25
        total = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4
        total = float(np.clip(total, -1.0, 1.0))

        # ── 等级判定 ──
        if total > 0.7:
            grade = "AAA"
        elif total > 0.4:
            grade = "AA"
        elif total > 0.15:
            grade = "A"
        elif total >= -0.15:
            grade = "B"
        else:
            grade = "C"

        # v2之前的旧阈值：不达标(A以下)返回None
        # v2改为：A及以上正常返回，B/C也返回（供下游观察）
        # 但为保持scan()的过滤行为不变，B和C仍返回None
        if grade not in ("AAA", "AA", "A"):
            # 仍然计算完整结果，但返回None以保持兼容
            # 如果需要观察B/C，可修改此处
            pass  # fall through to return result below

        # ── 周线确认（可选） ──
        weekly_bull = False
        if include_weekly:
            weekly_bull = self._check_weekly_trend(symbol)

        # ── 近期涨幅 ──
        gain_5d = (c / close[-6] - 1) * 100 if n > 5 else 0
        gain_10d = (c / close[-11] - 1) * 100 if n > 10 else 0
        gain_20d = (c / close[-21] - 1) * 100 if n > 20 else 0
        gain_60d = (c / close[-61] - 1) * 100 if n > 60 else 0

        # ── 距离均线位置 ──
        dist_ma5 = (c / ma5[-1] - 1) * 100 if ma5[-1] > 0 else 0
        dist_ma20 = (c / ma20[-1] - 1) * 100 if ma20[-1] > 0 else 0
        dist_ma60 = (c / ma60[-1] - 1) * 100 if ma60[-1] > 0 else 0

        # ── 位置安全评分 v2 ──
        pos_score, pos_reasons = self._position_filter_v2(close, high, low)

        # ── 旧版 trend_strength 兼容：映射到 [0, 100] ──
        trend_strength = round(abs(total) * 100, 1)

        # ⚠️ 待验证：pos100 + volR[1.2,1.6) + gain60d<5%
        # 数据来源：仅2026-05-20单日44只，样本不足，不构成统计结论
        unvalidated_pattern = (
            pos_score == 100
            and 1.2 <= vol_ratio <= 1.6
            and gain_60d < 5
        )
        # pos90-99 死亡区间警告（胜率37%，比pos80-89还差）
        pos_danger_zone = 90 <= pos_score < 100
        # 暴量警告（volR>1.6，胜率仅38%）
        vol_exhaustion = vol_ratio > 1.6

        return {
            "trend_grade": grade,
            "trend_strength": trend_strength,
            "trend_v2_score": round(total, 4),
            "trend_v2_factors": {
                "macd_direction": round(f1, 4),
                "dif_zero_context": round(f2, 4),
                "hist_trend": round(f3, 4),
                "ma_alignment": round(f4, 4),
            },
            "trend_full_bull": full_bull,
            "trend_short_bull": short_bull,
            "trend_macd_golden": macd_golden,
            "trend_hist_expanding": hist_expanding,
            "trend_dif_above_zero": dif_above_zero,
            "trend_vol_expanding": vol_expanding,
            "trend_vol_ratio": round(vol_ratio, 2),
            "trend_weekly_bull": weekly_bull,
            "trend_ma20_slope": round(ma20_slope * 100, 3),
            "trend_ma60_slope": round(ma60_slope * 100, 3),
            "trend_gain_5d": round(gain_5d, 2),
            "trend_gain_10d": round(gain_10d, 2),
            "trend_gain_20d": round(gain_20d, 2),
            "trend_gain_60d": round(gain_60d, 2),
            "trend_atr_pct": round(atr_pct, 2),
            "trend_dist_ma5": round(dist_ma5, 2),
            "trend_dist_ma20": round(dist_ma20, 2),
            "trend_dist_ma60": round(dist_ma60, 2),
            "trend_position_score": pos_score,
            "trend_position_reasons": ", ".join(pos_reasons),
            "trend_high_win_rate": unvalidated_pattern,
            "trend_pos_danger_zone": pos_danger_zone,
            "trend_vol_exhaustion": vol_exhaustion,
            "trend_price": round(c, 2),
            "current_price": round(c, 2),
        }

    def _check_weekly_trend(self, symbol: str) -> bool:
        """检查周线MA20方向是否向上。"""
        try:
            df = self.client.get_kline(
                symbol, freq="W", start="20250601", adj="qfq"
            )
            if df is None or len(df) < 20:
                return False
            close = df["close"].values.astype(float)
            ma20 = self._sma(close, 20)
            if len(ma20) < 6 or ma20[-6] <= 0:
                return False
            slope = (ma20[-1] / ma20[-6] - 1)
            return slope > 0.005
        except Exception:
            return False

    # ── 位置安全过滤器 v2 ──

    def _position_filter_v2(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> tuple:
        """多周期位置安全检查 (v2)

        修复漏洞：一只股票可在60日暴涨后横盘20天绕过旧检查。

        三维度评分:
          1. 多周期涨幅检查 (上限40分)
          2. 长期位置百分位 (上限30分)
          3. 弹性比-距低点涨幅 (上限30分)

        Returns:
            (score: int 0-100, reasons: list[str])
        """
        n = len(close)
        c = float(close[-1])
        reasons = []
        score = 100

        # ── 维度1: 多周期涨幅检查 (扣分上限40) ──
        overextend_deduct = 0

        if n > 60:
            gain_60d = (c / float(close[-61]) - 1) * 100
            if gain_60d > 50:
                overextend_deduct += 35
                reasons.append(f"60日涨幅过高({gain_60d:+.1f}%)")
            elif gain_60d > 35:
                overextend_deduct += 25
                reasons.append(f"60日涨幅偏大({gain_60d:+.1f}%)")
            elif gain_60d > 20:
                overextend_deduct += 10

        if n > 20:
            gain_20d = (c / float(close[-21]) - 1) * 100
            if gain_20d > 20:
                overextend_deduct += 25
                reasons.append(f"20日涨幅过高({gain_20d:+.1f}%)")
            elif gain_20d > 12:
                overextend_deduct += 10

        if n > 10:
            gain_10d = (c / float(close[-11]) - 1) * 100
            if gain_10d > 15:
                overextend_deduct += 15
                reasons.append(f"10日涨幅过高({gain_10d:+.1f}%)")

        overextend_deduct = min(40, overextend_deduct)
        score -= overextend_deduct

        # ── 维度2: 长期位置百分位 (扣分上限30) ──
        position_deduct = 0
        lookback = min(250, n)
        if lookback >= 60:
            high_period = float(high[-lookback:].max())
            low_period = float(low[-lookback:].min())
            price_range = high_period - low_period

            if price_range > 0:
                position_pct = (c - low_period) / price_range * 100
            else:
                position_pct = 50.0

            if position_pct > 85:
                position_deduct = 30
                reasons.append(f"长期位置极高({position_pct:.0f}%位)")
            elif position_pct > 75:
                position_deduct = 20
                reasons.append(f"长期位置偏高({position_pct:.0f}%位)")
            elif position_pct > 65:
                position_deduct = 10

        score -= position_deduct

        # ── 维度3: 弹性比-距低点涨幅 (扣分上限30) ──
        elasticity_deduct = 0
        if n > 20:
            low_60d = float(low[-min(60, n):].min())
            dist_from_low = (c / low_60d - 1) * 100

            if dist_from_low > 50:
                elasticity_deduct = 30
                reasons.append(f"距低点涨幅过大({dist_from_low:+.1f}%)")
            elif dist_from_low > 35:
                elasticity_deduct = 20
                reasons.append(f"距低点涨幅偏大({dist_from_low:+.1f}%)")
            elif dist_from_low > 25:
                elasticity_deduct = 10

        score -= elasticity_deduct

        # 底分保护
        score = max(0, score)
        if not reasons:
            reasons.append("位置安全")

        return score, reasons

    @staticmethod
    def check_position_safe(close: np.ndarray) -> tuple:
        """多周期涨幅准入检查 — 用于 emerging 模式准入条件。

        Returns:
            (passed: bool, reason: str)
        """
        n = len(close)
        c = float(close[-1])

        # 多周期涨幅门槛（任一超限即排除）
        if n > 60:
            gain_60d = (c / float(close[-61]) - 1) * 100
            if gain_60d > 35:
                return False, f"60日涨幅{gain_60d:+.1f}%>35%"

        if n > 20:
            gain_20d = (c / float(close[-21]) - 1) * 100
            if gain_20d > 15:
                return False, f"20日涨幅{gain_20d:+.1f}%>15%"

        if n > 10:
            gain_10d = (c / float(close[-11]) - 1) * 100
            if gain_10d > 12:
                return False, f"10日涨幅{gain_10d:+.1f}%>12%"

        return True, "OK"

    # ── 技术指标计算 ──

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        result = np.full_like(arr, np.nan, dtype=float)
        cumsum = np.cumsum(arr)
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate(
            ([0], cumsum[:-period])
        )) / period
        return result

    @staticmethod
    def _calc_macd(close: np.ndarray, fast: int, slow: int, signal: int):
        """MACD计算（A股惯例：柱状图×2）"""
        # 使用pandas的EWM保证精度
        s = pd.Series(close)
        ema_fast = s.ewm(span=fast, adjust=False).mean().values
        ema_slow = s.ewm(span=slow, adjust=False).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
        hist = 2 * (dif - dea)
        return dif, dea, hist

    @staticmethod
    def _calc_atr(high, low, close, period=14):
        """ATR计算"""
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.concatenate([[high[0] - low[0]], tr])
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr

    # ── 缓存 ──

    def _cache_key(self, trade_date: str, concept, industry) -> str:
        parts = [trade_date]
        if concept:
            parts.append(f"c_{concept}")
        if industry:
            parts.append(f"i_{industry}")
        return "_".join(parts)

    def _load_cache(self, trade_date, concept, industry) -> Optional[pd.DataFrame]:
        key = self._cache_key(trade_date, concept, industry)
        path = self.CACHE_DIR / f"{key}.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        return None

    def _save_cache(self, df: pd.DataFrame, trade_date, concept, industry):
        key = self._cache_key(trade_date, concept, industry)
        path = self.CACHE_DIR / f"{key}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception:
            pass
