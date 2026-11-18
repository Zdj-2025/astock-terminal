#!/usr/bin/env python3
"""背离共振检测器 — 多周期MACD背离共振分析。

核心逻辑：
  - 单一周期背离不可靠，多周期同时背离 = 共振 = 强信号
  - 底背离：价格新低但MACD没新低 → 买入信号
  - 顶背离：价格新高但MACD没新高 → 卖出信号
  - 30min + 60min 同时背离 = 做T/短中期判断的核心指标

用法：
  from scripts.analyzers.divergence_detector import DivergenceDetector

  store = DataStore()
  df_60 = store.get_kline("603881", "60min")
  df_30 = store.get_kline("603881", "30min")

  detector = DivergenceDetector()
  result = detector.detect_resonance(df_60, df_30)
  # result = {"resonance": True, "type": "bottom", "strength": "strong", ...}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DivergenceSignal:
    """单周期背离信号。"""
    period: str               # "60min" / "30min" / "D" / "W"
    div_type: str             # "bottom" / "top"
    detected: bool            # 是否检测到
    strength: float = 0.0     # 0-1，背离强度
    price_high: float = 0.0   # 价格极值点
    macd_high: float = 0.0    # MACD极值点
    bar_index: int = 0        # 背离发生的K线位置
    confirmed: bool = False   # 量能K线确认(谷后放量阳线/峰后放量阴线);可实时复现,消除centered未来函数倾向
    description: str = ""


@dataclass
class ResonanceResult:
    """多周期背离共振结果。"""
    resonance: bool = False           # 是否共振
    div_type: str = "none"            # "bottom" / "top" / "none"
    strength: str = "none"            # "strong" / "medium" / "weak" / "none"
    score: float = 0.0               # 0-100共振分数
    signals: List[DivergenceSignal] = field(default_factory=list)
    recommendation: str = ""
    details: str = ""


class DivergenceDetector:
    """背离共振检测器。

    检测逻辑：
    1. 在给定周期内找价格的局部极值点（波峰/波谷）
    2. 找对应位置的MACD极值点
    3. 对比最近两个极值：价格创新低/高但MACD没有 → 背离
    4. 多个周期同时出现 → 共振
    """

    def __init__(self, macd_fast: int = 12, macd_slow: int = 26,
                 macd_signal: int = 9, fast_macd_fast: int = 5,
                 fast_macd_slow: int = 34, fast_macd_signal: int = 5):
        """
        Args:
            macd_fast/slow/signal: 标准MACD参数
            fast_macd_fast/slow/signal: 快速MACD参数（灵敏度更高）
        """
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.fast_macd_fast = fast_macd_fast
        self.fast_macd_slow = fast_macd_slow
        self.fast_macd_signal = fast_macd_signal

    def _calc_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标（标准 + 快速双参数）。"""
        close = df["close"].values

        # 标准MACD
        ema_fast = self._ema(close, self.macd_fast)
        ema_slow = self._ema(close, self.macd_slow)
        dif = ema_fast - ema_slow
        dea = self._ema(dif, self.macd_signal)
        macd_hist = 2 * (dif - dea)

        # 快速MACD
        ema_fast2 = self._ema(close, self.fast_macd_fast)
        ema_slow2 = self._ema(close, self.fast_macd_slow)
        dif2 = ema_fast2 - ema_slow2
        dea2 = self._ema(dif2, self.fast_macd_signal)
        macd_hist2 = 2 * (dif2 - dea2)

        result = df.copy()
        result["dif"] = dif
        result["dea"] = dea
        result["macd_hist"] = macd_hist
        result["dif_fast"] = dif2
        result["dea_fast"] = dea2
        result["macd_hist_fast"] = macd_hist2

        return result

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """计算EMA。修复: 跳过开头NaN再算种子(否则dif前缀NaN污染种子,整个DEA/HIST变NaN,
        导致背离检测全部失效)。"""
        data = np.asarray(data, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        valid = ~np.isnan(data)
        if valid.sum() < period:
            return result
        first_valid = int(np.argmax(valid))   # 第一个非NaN位置
        seg = data[first_valid:]
        if len(seg) < period:
            return result
        result[first_valid + period - 1] = float(np.mean(seg[:period]))
        multiplier = 2.0 / (period + 1)
        for i in range(first_valid + period, len(data)):
            result[i] = data[i] * multiplier + result[i - 1] * (1 - multiplier)
        return result

    def _find_peaks(self, data: np.ndarray, order: int = 5) -> np.ndarray:
        """找局部峰值（波峰）。"""
        peaks = []
        for i in range(order, len(data) - order):
            if np.isnan(data[i]):
                continue
            is_peak = True
            for j in range(max(0, i - order), min(len(data), i + order + 1)):
                if j != i and not np.isnan(data[j]) and data[j] > data[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        return np.array(peaks)

    def _find_troughs(self, data: np.ndarray, order: int = 5) -> np.ndarray:
        """找局部谷值（波谷）。"""
        troughs = []
        for i in range(order, len(data) - order):
            if np.isnan(data[i]):
                continue
            is_trough = True
            for j in range(max(0, i - order), min(len(data), i + order + 1)):
                if j != i and not np.isnan(data[j]) and data[j] < data[i]:
                    is_trough = False
                    break
            if is_trough:
                troughs.append(i)
        return np.array(troughs)

    def detect_divergence(self, df: pd.DataFrame, period: str = "",
                          lookback: int = 60) -> DivergenceSignal:
        """检测单周期MACD背离。

        Args:
            df: 包含 close 列的K线数据(建议传全量,内部先算MACD再切窗——EMA用前史预热)
            period: 周期标识（如 "60min"）
            lookback: 回溯K线数(默认60=日线季度窗,与回测口径一致;分钟级建议120)。
                      2026-08-19: 默认None→60——原None(全历史)口径下谷集完全不同且
                      外部tail(60)传入会导致窗口头部EMA未预热hist=NaN(百亚6/25谷漏检案例)

        Returns:
            DivergenceSignal
        """
        signal = DivergenceSignal(period=period, div_type="none", detected=False)

        if df is None or len(df) < 30:
            signal.description = "数据不足（需≥30条）"
            return signal

        # 2026-08-19 bugfix: 先在全历史上算MACD再切窗口——原"先切后算"导致窗口头部26日EMA未预热
        # hist=NaN(百亚6/25谷hist=NaN,谷对比较恒False,背离漏检)。先算后切=EMA用前史收敛,无NaN。
        df = self._calc_macd(df)

        if lookback and len(df) > lookback:
            df = df.tail(lookback).reset_index(drop=True)

        close = df["close"].values
        dif = df["dif"].values
        macd_hist = df["macd_hist"].values
        dif_fast = df["dif_fast"].values
        macd_hist_fast = df["macd_hist_fast"].values

        # 底背离用LOW（最低价），顶背离用HIGH（最高价）
        # 人眼看K线图对比的是K线低点/高点连线，不是收盘价
        low = df["low"].values if "low" in df.columns else close
        high = df["high"].values if "high" in df.columns else close

        # 动态调整 order（根据数据量）；上限5(原8对日线过严,全市场几乎检测不到谷)
        order = max(3, min(5, len(df) // 15))

        # ── 底背离检测 ──
        # 用LOW（最低价）找价格的波谷
        price_troughs = self._find_troughs(low, order)
        # 找MACD柱的波谷
        hist_troughs = self._find_troughs(macd_hist, order)

        if len(price_troughs) >= 2:
            # 注: 原还有 len(hist_troughs)>=2 冗余约束, 但背离只比较price谷位置的hist值,
            # 不需hist本身有谷; 该约束使全市场几乎检测不到背离, 已移除
            # 2026-08-19: 新谷出现会把旧有效谷对顶掉(百亚案例:8/17新谷掩盖6/25→7/14有效背离)
            # 先试最新两谷,不构成hist背离则回退前一对
            p1_idx, p2_idx = int(price_troughs[-2]), int(price_troughs[-1])
            if not (low[p2_idx] < low[p1_idx] and macd_hist[p2_idx] > macd_hist[p1_idx]) \
                    and len(price_troughs) >= 3:
                p1_idx, p2_idx = int(price_troughs[-3]), int(price_troughs[-2])
            p1_price = low[p1_idx]  # 用LOW做价格对比
            p2_price = low[p2_idx]

            # 找对应位置的MACD柱值
            # 全市场回测: MACD柱胜率54.3% > DIF 49.5% > DEA 47.6%
            d1_val = macd_hist[p1_idx]
            d2_val = macd_hist[p2_idx]

            # 底背离：LOW新低，MACD柱没有新低
            if p2_price < p1_price and d2_val > d1_val:
                # 强度：价格跌幅 vs MACD柱反向幅度
                price_drop = (p1_price - p2_price) / p1_price
                hist_rise = (d2_val - d1_val) / (abs(d1_val) + 0.001)
                strength = min(1.0, (price_drop + hist_rise) / 0.1)

                # 额外检查：快速MACD柱是否也背离（双参数确认）
                d1_fast = macd_hist_fast[p1_idx]
                d2_fast = macd_hist_fast[p2_idx]
                fast_confirm = d2_fast > d1_fast

                if fast_confirm:
                    strength = min(1.0, strength * 1.3)

                # 量能阳线确认 + E组合过滤(2026-08-19落地,回测验证: B+C+D全加超额+1.56/胜率58%,
                # vs基线+0.89/53%,样本3215事件。回测脚本/tmp/div_filter_backtest.py逻辑:
                #   B谷后再跌>10%=结构失效(南华案例:背离后暴跌26%被误报确认)
                #   C确认窗口=谷后10日内(技源案例:谷后28天的小阳线与谷脱节,不是启动确认)
                #   D DIF双口径抬升(技源案例:hist微抬0.14但DIF新低=弱背离陷阱))
                vol_confirmed = False
                e_reject = None   # E组合过滤拒绝原因(拒绝则不计信号,落入顶背离检测)
                if p2_idx + 1 < len(low):
                    after_low = float(np.min(low[p2_idx + 1:]))
                    if after_low < p2_price * 0.90:   # B过滤
                        e_reject = f"谷后再跌{(after_low / p2_price - 1):.0%}(结构失效·南华规则)"
                if e_reject is None and dif[p2_idx] <= dif[p1_idx]:   # D过滤
                    e_reject = f"DIF也新低({dif[p1_idx]:.3f}→{dif[p2_idx]:.3f},弱背离·技源规则)"
                if e_reject is None and "open" in df.columns and "volume" in df.columns:
                    last_idx = len(df) - 1
                    if last_idx > p2_idx:
                        op_v = df["open"].values; cl_v = df["close"].values; vl_v = df["volume"].values
                        vol_ma20 = pd.Series(vl_v).rolling(20).mean().values
                        # C过滤: 谷后10日内的放量阳线=确认, 且确认日须在近10日内(信号新鲜度——
                        # 否则历史确认的旧信号也报,60只暴增案例2026-08-19)
                        for _j in range(p2_idx + 1, min(p2_idx + 11, len(cl_v))):
                            if (cl_v[_j] > op_v[_j]
                                    and not np.isnan(vol_ma20[_j])
                                    and vl_v[_j] > vol_ma20[_j]
                                    and _j >= len(cl_v) - 10):
                                vol_confirmed = True
                                strength = min(1.0, strength * 1.2)
                                break
                elif e_reject is not None:
                    signal.description = (
                        f"底背离形态(p2@{p2_price:.2f}) 但{e_reject} → 不计见底信号; 继续查顶背离"
                    )

                # [bugfix 2026-07-26] 排除"已兑现/高位"过期底背离
                # (睿创微纳案例: 底背离在7/20, 价已反弹+19%到接近历史高, 误报"见底买入"误导追高)
                last_close = close[-1]
                bounce = last_close / p2_price - 1            # 谷后反弹幅度
                lb_high = float(np.max(high))                 # 回溯区间最高价
                drawdown = last_close / lb_high - 1           # 距高回撤
                div_stale = (e_reject is not None) or (bounce > 0.12) or (drawdown > -0.05)  # 2026-09-02 8%→12%(华胜案例:双底二底+8.7%被旧线拦截)
                if div_stale:
                    why = f"谷后已反弹{bounce:+.0%}" if bounce > 0.12 else f"在高位(距高{drawdown:.0%})"
                    signal.description = (
                        f"底背离形态(p2@{p2_price:.2f}, MACD柱{d1_val:.4f}→{d2_val:.4f}) "
                        f"但{why} → 已兑现/高位, 过期不计为见底信号; 继续查顶背离"
                    )
                    # 不设 detected, 落入下方顶背离检测
                else:
                    signal.detected = True
                    signal.div_type = "bottom"
                    signal.strength = round(strength, 3)
                    signal.confirmed = vol_confirmed
                    signal.price_high = p2_price
                    signal.macd_high = d2_val
                    signal.bar_index = p2_idx
                    vc_tag = " [+量能阳线确认·可复现]" if vol_confirmed else " [未确认·等谷后放量阳线]"
                    signal.description = (
                        f"底背离: LOW{p1_price:.2f}→{p2_price:.2f}(新低), "
                        f"MACD柱{d1_val:.4f}→{d2_val:.4f}(未新低)"
                        f"{' [快速MACD确认]' if fast_confirm else ''}{vc_tag}"
                    )
                    return signal

        # ── 顶背离检测 ──
        # 用HIGH（最高价）找价格的波峰
        price_peaks = self._find_peaks(high, order)
        # 找MACD柱的波峰
        hist_peaks = self._find_peaks(macd_hist, order)

        if len(price_peaks) >= 2:
            # 注: 同底背离, 移除 len(hist_peaks)>=2 冗余约束
            last_two_price_peaks = price_peaks[-2:]
            p1_idx, p2_idx = last_two_price_peaks
            p1_price = high[p1_idx]  # 用HIGH做价格对比
            p2_price = high[p2_idx]

            d1_val = macd_hist[p1_idx]
            d2_val = macd_hist[p2_idx]

            # 顶背离：HIGH新高，MACD柱没有新高
            if p2_price > p1_price and d2_val < d1_val:
                price_rise = (p2_price - p1_price) / p1_price
                hist_drop = (d1_val - d2_val) / (abs(d1_val) + 0.001)
                strength = min(1.0, (price_rise + hist_drop) / 0.1)

                # 快速MACD确认
                d1_fast = macd_hist_fast[p1_idx]
                d2_fast = macd_hist_fast[p2_idx]
                fast_confirm = d2_fast < d1_fast

                if fast_confirm:
                    strength = min(1.0, strength * 1.3)

                # 量能阴线确认(顶部,可实时复现): 峰后当前bar放量阴线=下跌确认
                vol_confirmed = False
                if "open" in df.columns and "volume" in df.columns:
                    last_idx = len(df) - 1
                    if last_idx > p2_idx:
                        op_v = df["open"].values; cl_v = df["close"].values; vl_v = df["volume"].values
                        vol_ma20 = pd.Series(vl_v).rolling(20).mean().values
                        if (cl_v[last_idx] < op_v[last_idx]
                                and not np.isnan(vol_ma20[last_idx])
                                and vl_v[last_idx] > vol_ma20[last_idx]):
                            vol_confirmed = True
                            strength = min(1.0, strength * 1.2)

                signal.detected = True
                signal.div_type = "top"
                signal.strength = round(strength, 3)
                signal.confirmed = vol_confirmed
                signal.price_high = p2_price
                signal.macd_high = d2_val
                signal.bar_index = p2_idx
                vc_tag = " [+量能阴线确认·可复现]" if vol_confirmed else " [未确认·等峰后放量阴线]"
                signal.description = (
                    f"顶背离: HIGH{p1_price:.2f}→{p2_price:.2f}(新高), "
                    f"MACD柱{d1_val:.4f}→{d2_val:.4f}(未新高)"
                    f"{' [快速MACD确认]' if fast_confirm else ''}{vc_tag}"
                )
                return signal

        signal.description = "未检测到背离"
        return signal

    def detect_resonance(self, df_60: pd.DataFrame, df_30: pd.DataFrame = None,
                         df_daily: pd.DataFrame = None,
                         df_15: pd.DataFrame = None) -> ResonanceResult:
        """检测多周期背离共振。

        核心逻辑：30min + 60min 同时背离 = 做T级别的强信号
        再加上日线 = 中期趋势信号

        Args:
            df_60: 60分钟K线
            df_30: 30分钟K线（可选）
            df_daily: 日线K线（可选）
            df_15: 15分钟K线（可选）

        Returns:
            ResonanceResult
        """
        result = ResonanceResult()
        signals = []

        # 60分钟背离（核心周期）
        sig_60 = self.detect_divergence(df_60, period="60min")
        signals.append(sig_60)

        # 30分钟背离
        if df_30 is not None:
            sig_30 = self.detect_divergence(df_30, period="30min")
            signals.append(sig_30)

        # 日线背离（加分项）
        if df_daily is not None:
            sig_d = self.detect_divergence(df_daily, period="D")
            signals.append(sig_d)

        # 15分钟背离（加分项）
        if df_15 is not None:
            sig_15 = self.detect_divergence(df_15, period="15min")
            signals.append(sig_15)

        result.signals = signals

        # ── 共振判断 ──
        # 统计各类型背离
        bottom_count = sum(1 for s in signals if s.detected and s.div_type == "bottom")
        top_count = sum(1 for s in signals if s.detected and s.div_type == "top")
        total_detected = bottom_count + top_count

        # 关键：30min + 60min 的背离是否同类型
        sig_30_data = next((s for s in signals if s.period == "30min"), None)

        # 60min+30min共振（核心判断）
        core_resonance = False
        core_type = "none"

        if sig_60.detected and sig_30_data and sig_30_data.detected:
            if sig_60.div_type == sig_30_data.div_type:
                core_resonance = True
                core_type = sig_60.div_type

        # 判断共振强度
        if core_resonance:
            result.resonance = True
            result.div_type = core_type

            # 计算共振分数
            score = 0
            # 核心共振（60+30）= 60分
            avg_strength = (sig_60.strength + sig_30_data.strength) / 2
            score += 60 * avg_strength

            # 日线也同向 = +25分
            sig_d = next((s for s in signals if s.period == "D"), None)
            if sig_d and sig_d.detected and sig_d.div_type == core_type:
                score += 25 * sig_d.strength

            # 15min也同向 = +15分
            sig_15_data = next((s for s in signals if s.period == "15min"), None)
            if sig_15_data and sig_15_data.detected and sig_15_data.div_type == core_type:
                score += 15 * sig_15_data.strength

            result.score = round(min(100, score), 1)

            if result.score >= 70:
                result.strength = "strong"
            elif result.score >= 40:
                result.strength = "medium"
            else:
                result.strength = "weak"

            # 生成建议
            if core_type == "bottom":
                result.recommendation = f"🟢 底背离共振({result.strength})，做T/短线买入信号"
                if sig_d and sig_d.detected and sig_d.div_type == "bottom":
                    result.recommendation += "，日线也确认 → 中期看多"
            else:
                result.recommendation = f"🔴 顶背离共振({result.strength})，做T/短线卖出信号"
                if sig_d and sig_d.detected and sig_d.div_type == "top":
                    result.recommendation += "，日线也确认 → 中期看空"

        else:
            # 无核心共振，看单个周期
            if total_detected > 0:
                # 找最强的单周期信号
                best = max((s for s in signals if s.detected),
                           key=lambda s: s.strength, default=None)
                if best:
                    result.div_type = best.div_type
                    result.score = round(best.strength * 30, 1)  # 单周期最多30分
                    result.strength = "weak"
                    result.recommendation = (
                        f"⚠️ {best.period}单周期{best.div_type}背离（弱信号），"
                        f"需等待其他周期确认"
                    )
            else:
                result.recommendation = "无背离信号，趋势正常"

        # 生成详情
        details = []
        for s in signals:
            status = "✅" if s.detected else "➡️"
            details.append(f"  {status} {s.period:>5s}: {s.description}")
        result.details = "\n".join(details)

        return result

    def format_report(self, result: ResonanceResult, symbol: str = "") -> str:
        """格式化输出报告。"""
        lines = []
        lines.append(f"{'='*55}")
        lines.append(f"📊 背离共振分析: {symbol}")
        lines.append(f"{'='*55}")

        if result.resonance:
            icon = "🟢" if result.div_type == "bottom" else "🔴"
            type_label = "底背离" if result.div_type == "bottom" else "顶背离"
            lines.append(f"\n{icon} {type_label}共振！ 强度: {result.strength} "
                         f"分数: {result.score}/100")
        else:
            lines.append(f"\n⚪ 无背离共振")

        lines.append(f"\n{result.details}")
        lines.append(f"\n💡 {result.recommendation}")
        lines.append(f"{'='*55}")

        return "\n".join(lines)
