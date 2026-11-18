#!/usr/bin/env python3
"""多级别买入信号检测器 — 五级信号体系。

核心思想：大级别定方向，小级别定时机。
- 高级别买点（日线三买、通道下轨、均线回踩）本身不需要底背离
- 但回踩过程在低级别图上可能形成背离 → 精确择时
- 没有背离时用f''拐点/企稳K线/量能变化替代

五级信号：
  S级: 30min+60min底背离共振（抄底）
  A级: 大级别买点 + 60min/30min底背离（趋势上车）
  B级: 通道下轨(<20%) + f''拐点 + 企稳K线（通道回调）
  C级: 60min f''从负转正 + f'回升（动量反转）
  D级: 突破回踩不破 + 量缩（右侧确认）

回测铁律（⚠️ 无未来数据）：
  - Savgol滤波器中心对称 → shift(window//2)
  - 信号检测只用当天及之前的数据
  - 收益计算从信号日次日开盘价开始
  - MACD/均线等指标用expanding/rolling，不用future

用法:
    from scripts.analyzers.buy_signal_detector import BuySignalDetector
    detector = BuySignalDetector()
    signals = detector.detect('603881')
    backtest = detector.backtest('603881')
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.signal import argrelextrema, savgol_filter
from typing import Optional


@dataclass
class Signal:
    """买入信号。"""
    level: str = ""           # S/A/B/C/D
    level_name: str = ""      # 中文描述
    date: str = ""            # 信号日期
    price: float = 0.0        # 信号时价格
    confidence: float = 0.0   # 置信度 0-1
    details: str = ""         # 详细描述
    ret5: float | None = None  # 5日收益(回测用)
    ret10: float | None = None
    ret20: float | None = None


@dataclass
class BacktestResult:
    """回测结果。"""
    level: str = ""
    total_signals: int = 0
    win_rate_5d: float = 0.0
    win_rate_10d: float = 0.0
    win_rate_20d: float = 0.0
    avg_ret_5d: float = 0.0
    avg_ret_10d: float = 0.0
    avg_ret_20d: float = 0.0
    signals: list = field(default_factory=list)


class BuySignalDetector:
    """多级别买入信号检测器。"""

    SG_WINDOW = 21
    SG_POLY = 3

    # ── 底背离检测 ──────────────────────────────────────────────

    @staticmethod
    def _detect_divergence(close, low, dif, lookback=80, order=5):
        """检测底背离（单周期）。

        Returns:
            list of (idx1, idx2) 底背离对
        """
        n = len(close)
        if n < lookback:
            return []

        lows_idx = argrelextrema(low, np.less, order=order)[0]
        recent = lows_idx[lows_idx >= n - lookback]

        pairs = []
        for j in range(len(recent) - 1):
            l1, l2 = recent[j], recent[j + 1]
            if low[l2] < low[l1] and dif[l2] > dif[l1]:
                pairs.append((l1, l2))
        return pairs

    def _compute_macd(self, close):
        """计算MACD，返回numpy数组。"""
        s = pd.Series(close)
        ema12 = s.ewm(span=12).mean().values
        ema26 = s.ewm(span=26).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9).mean().values
        hist = 2 * (dif - dea)
        return dif, dea, hist

    # ── 二阶导（防未来数据版本） ──────────────────────────────

    def _compute_f2_safe(self, close):
        """计算因果 f'/f''。

        2026-09-10: 原实现为中心化 savgol + shift(window//2)。该法因果、无未来泄漏,
        但整条序列滞后 shift 根(末值=f''发生在 shift 根之前的导数), 且与权威 causal_savgol 定义漂移。
        改用 core.derivatives.causal_savgol(pos=0=窗口右端): 每点即"当根"导数, 前 window-1 位为NaN。
        """
        ln_p = np.log(close)
        window = min(self.SG_WINDOW, len(close) // 2 * 2 + 1)
        if window < 5:
            return np.zeros(len(close)), np.zeros(len(close))
        from scripts.core.derivatives import causal_savgol
        return (causal_savgol(ln_p, window, self.SG_POLY, deriv=1),
                causal_savgol(ln_p, window, self.SG_POLY, deriv=2))

    # ── 企稳K线检测 ──────────────────────────────────────────

    @staticmethod
    def _detect_stabilization(close, low, high, volume, window=5):
        """检测企稳信号（最近window根K线）。

        企稳条件（满足任一）：
        1. 十字星：|C-O|/C < 0.005 且下影线较长
        2. 锤子线：下影线 > 实体*2 且上影线短
        3. 放量阳线：阳线且量能 > 20日均量*1.2
        4. 止跌阳线：连跌后出现阳线
        """
        if len(close) < window + 1:
            return False, ""

        recent_c = close[-window:]
        recent_l = low[-window:]
        recent_h = high[-window:]
        recent_v = volume[-window:] if volume is not None else None

        last_c = recent_c[-1]
        last_l = recent_l[-1]
        last_h = recent_h[-1]

        # 阳线?
        last_open = recent_c[-2] if len(recent_c) > 1 else last_c
        is_yang = last_c > last_open
        body = abs(last_c - last_open)
        upper_shadow = last_h - max(last_c, last_open)
        lower_shadow = min(last_c, last_open) - last_l

        # 十字星
        if body / last_c < 0.005 and lower_shadow > body * 1.5:
            return True, "十字星企稳"

        # 锤子线
        if lower_shadow > body * 2 and upper_shadow < body:
            return True, "锤子线企稳"

        # 放量阳线
        if is_yang and recent_v is not None:
            avg_vol = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            if recent_v[-1] > avg_vol * 1.2:
                return True, "放量阳线企稳"

        # 止跌（连跌后阳线）
        if is_yang and len(recent_c) >= 3:
            drops = sum(1 for i in range(-min(3, len(recent_c)-1), 0)
                       if recent_c[i] < recent_c[i-1])
            if drops >= 2:
                return True, "止跌阳线"

        return False, ""

    # ── S级：底背离共振 ──────────────────────────────────────

    def detect_s_level(self, close_60, low_60, close_30, low_30):
        """S级信号：30min+60min底背离共振。

        返回: (bool, str) 是否触发, 描述
        """
        dif_60, _, _ = self._compute_macd(close_60)
        dif_30, _, _ = self._compute_macd(close_30)

        div_60 = self._detect_divergence(close_60, low_60, dif_60, lookback=60, order=5)
        div_30 = self._detect_divergence(close_30, low_30, dif_30, lookback=80, order=5)

        if div_60 and div_30:
            # 最近一对的背离
            _, l60_2 = div_60[-1]
            _, l30_2 = div_30[-1]
            return True, f"30min+60min底背离共振(60min@{low_60[l60_2]:.2f}, 30min@{low_30[l30_2]:.2f})"

        return False, ""

    # ── B级：通道下轨+企稳 ──────────────────────────────────

    def detect_b_level(self, close_d, low_d, high_d, volume_d, f1_d, f2_d):
        """B级信号：通道下轨支撑 + f''拐点 + 企稳K线。

        不依赖底背离，而是看：
        1. 价格在上升通道下半区（用简单的均线通道判断）
        2. f''从负转正（动量拐点）
        3. 企稳K线
        """
        n = len(close_d)
        if n < 60:
            return False, ""

        # 简单通道：MA20±2std
        ma20 = np.convolve(close_d, np.ones(20)/20, mode='valid')
        if len(ma20) < 5:
            return False, ""

        # 当前位置
        cur_price = close_d[-1]
        cur_ma20 = ma20[-1]
        std20 = np.std(close_d[-20:])
        lower_band = cur_ma20 - 2 * std20
        upper_band = cur_ma20 + 2 * std20

        if upper_band == lower_band:
            return False, ""

        pct = (cur_price - lower_band) / (upper_band - lower_band) * 100

        # 条件1：在下半区
        if pct > 30:
            return False, ""

        # 条件2：f''拐点（最近3天内从负转正）
        recent_f2 = f2_d[-5:]
        has_turn = False
        for i in range(1, len(recent_f2)):
            if recent_f2[i] > 0 and recent_f2[i-1] <= 0:
                has_turn = True
                break

        # 条件3：企稳K线
        stabilized, stab_type = self._detect_stabilization(
            close_d, low_d, high_d, volume_d)

        # B级需要至少2/3条件
        score = (pct <= 30) + has_turn + stabilized
        if score >= 2:
            parts = [f"通道位置{pct:.0f}%"]
            if has_turn:
                parts.append("f''拐点")
            if stabilized:
                parts.append(stab_type)
            return True, " + ".join(parts)

        return False, ""

    # ── C级：二阶导拐点 ──────────────────────────────────────

    def detect_c_level(self, f1, f2):
        """C级信号：f''从负转正（加速度反转）+ f'开始回升。

        最简单的动量反转信号，不需要底背离。
        """
        if len(f1) < 5 or len(f2) < 5:
            return False, ""

        # f''从负转正
        f2_turn = f2[-1] > 0 and f2[-2] <= 0
        # 或最近3天内转正
        f2_turn_recent = any(
            f2[-i] > 0 and f2[-i-1] <= 0 for i in range(1, min(4, len(f2)))
        )

        # f'回升（从下降转上升）
        f1_rising = f1[-1] > f1[-3] if len(f1) >= 4 else False

        if (f2_turn or f2_turn_recent) and f1_rising:
            return True, f"f''拐点(f''={f2[-1]:.5f}) + f'回升(f'={f1[-1]:.5f})"

        # 弱化版：只要f''转正就行
        if f2_turn or f2_turn_recent:
            return True, f"f''拐点(f''={f2[-1]:.5f})"

        return False, ""

    # ── 综合检测 ──────────────────────────────────────────────

    def detect(self, df_daily, df_60min=None, df_30min=None,
               current_price=None):
        """综合检测所有级别的买入信号。

        Args:
            df_daily: 日线数据
            df_60min: 60分钟数据（可选）
            df_30min: 30分钟数据（可选）
            current_price: 当前实时价格
        """
        signals = []

        # 日线数据准备
        close_d = df_daily['close'].values
        low_d = df_daily['low'].values
        high_d = df_daily['high'].values
        vol_d = df_daily['vol'].values if 'vol' in df_daily.columns else (
            df_daily['volume'].values if 'volume' in df_daily.columns else None)
        dates_d = df_daily['trade_date'].values if 'trade_date' in df_daily.columns else (
            df_daily['date'].values if 'date' in df_daily.columns else None)

        # 确保升序
        if dates_d is not None and str(dates_d[0]) > str(dates_d[-1]):
            close_d = close_d[::-1]; low_d = low_d[::-1]; high_d = high_d[::-1]
            if vol_d is not None: vol_d = vol_d[::-1]
            dates_d = dates_d[::-1]

        price = current_price if current_price is not None else close_d[-1]

        # 二阶导（防未来数据）
        f1_d, f2_d = self._compute_f2_safe(close_d)

        # ── S级：底背离共振 ──
        if df_60min is not None and df_30min is not None:
            # 60min数据
            c60 = df_60min['close'].values; l60 = df_60min['low'].values
            d60 = df_60min['datetime'].values if 'datetime' in df_60min.columns else df_60min['trade_date'].values
            if str(d60[0]) > str(d60[-1]):
                c60 = c60[::-1]; l60 = l60[::-1]

            # 30min数据
            c30 = df_30min['close'].values; l30 = df_30min['low'].values
            d30 = df_30min['datetime'].values if 'datetime' in df_30min.columns else df_30min['trade_date'].values
            if str(d30[0]) > str(d30[-1]):
                c30 = c30[::-1]; l30 = l30[::-1]

            triggered, desc = self.detect_s_level(c60, l60, c30, l30)
            if triggered:
                signals.append(Signal(
                    level='S', level_name='底背离共振',
                    date=str(dates_d[-1])[:10] if dates_d is not None else '',
                    price=price, confidence=0.9, details=desc
                ))

        # ── B级：通道下轨+企稳 ──
        triggered, desc = self.detect_b_level(close_d, low_d, high_d, vol_d, f1_d, f2_d)
        if triggered:
            signals.append(Signal(
                level='B', level_name='通道支撑+企稳',
                date=str(dates_d[-1])[:10] if dates_d is not None else '',
                price=price, confidence=0.7, details=desc
            ))

        # ── C级：二阶导拐点 ──
        triggered, desc = self.detect_c_level(f1_d, f2_d)
        if triggered:
            signals.append(Signal(
                level='C', level_name='动量反转',
                date=str(dates_d[-1])[:10] if dates_d is not None else '',
                price=price, confidence=0.5, details=desc
            ))

        # ── 60min二阶导拐点（C级补充）──
        if df_60min is not None:
            c60 = df_60min['close'].values
            d60 = df_60min['datetime'].values if 'datetime' in df_60min.columns else df_60min['trade_date'].values
            if str(d60[0]) > str(d60[-1]): c60 = c60[::-1]
            f1_60, f2_60 = self._compute_f2_safe(c60)
            triggered, desc = self.detect_c_level(f1_60, f2_60)
            if triggered:
                signals.append(Signal(
                    level='C', level_name='60min动量反转',
                    date=str(dates_d[-1])[:10] if dates_d is not None else '',
                    price=price, confidence=0.6, details=f'60min {desc}'
                ))

        return signals

    # ── 回测（严格防未来数据） ────────────────────────────────

    def backtest(self, df_daily, df_60min=None, df_30min=None,
                 min_days=120) -> dict[str, BacktestResult]:
        """逐日回测所有级别信号。

        ⚠️ 防未来数据措施：
        1. 导数用 core.derivatives.causal_savgol(pos=0右端, 每点只用<=当根)  # 2026-09-10 弃中心化+shift滞后
        2. 信号检测只用 close[:i+1] / close_60[:cut](按日截断到当天, 不含未来)
        3. 收益从次日开盘价计算
        4. MACD/均线只用历史数据
        """
        close_d = df_daily['close'].values
        low_d = df_daily['low'].values
        high_d = df_daily['high'].values
        open_d = df_daily['open'].values
        vol_d = df_daily['vol'].values if 'vol' in df_daily.columns else (
            df_daily['volume'].values if 'volume' in df_daily.columns else None)
        dates_d = df_daily['trade_date'].values if 'trade_date' in df_daily.columns else (
            df_daily['date'].values if 'date' in df_daily.columns else None)

        if dates_d is not None and str(dates_d[0]) > str(dates_d[-1]):
            close_d = close_d[::-1]; low_d = low_d[::-1]; high_d = high_d[::-1]
            open_d = open_d[::-1]
            if vol_d is not None: vol_d = vol_d[::-1]
            dates_d = dates_d[::-1]

        n = len(close_d)
        if n < min_days:
            return {}

        # 预计算日线MACD（expanding，不含未来数据）
        dif_d, dea_d, hist_d = self._compute_macd(close_d)

        # 预计算二阶导（已shift防泄漏）
        f1_d, f2_d = self._compute_f2_safe(close_d)

        # 60min MACD
        dif_60 = dea_60 = low_60 = close_60 = None
        if df_60min is not None:
            close_60 = df_60min['close'].values
            low_60 = df_60min['low'].values
            d60 = df_60min['datetime'].values if 'datetime' in df_60min.columns else df_60min['trade_date'].values
            if str(d60[0]) > str(d60[-1]):
                close_60 = close_60[::-1]; low_60 = low_60[::-1]
            dif_60, dea_60, _ = self._compute_macd(close_60)

        # 30min MACD
        dif_30 = low_30 = close_30 = None
        if df_30min is not None:
            close_30 = df_30min['close'].values
            low_30 = df_30min['low'].values
            d30 = df_30min['datetime'].values if 'datetime' in df_30min.columns else df_30min['trade_date'].values
            if str(d30[0]) > str(d30[-1]):
                close_30 = close_30[::-1]; low_30 = low_30[::-1]
            dif_30, _, _ = self._compute_macd(close_30)

        # 逐日检测
        all_signals = {lvl: [] for lvl in ['S', 'B', 'C', 'C60']}

        for i in range(min_days, n - 20):  # 留20天算收益
            date_str = str(dates_d[i])[:10] if dates_d is not None else str(i)
            price = close_d[i]

            # ── S级：30min+60min底背离 ──
            if dif_60 is not None and dif_30 is not None:
                # ⚠️ 防未来数据：只用到当天及之前的分钟数据
                # 根据日线第 i 天的日期截断分钟级序列
                day_ts = str(dates_d[i])[:10]
                # 60min 截断
                mask_60 = np.array([str(t)[:10] <= day_ts for t in d60])
                cut_60 = int(mask_60.sum())
                if cut_60 < 60:
                    cut_60 = None  # 数据不足，跳过本次
                # 30min 截断
                mask_30 = np.array([str(t)[:10] <= day_ts for t in d30])
                cut_30 = int(mask_30.sum())
                if cut_30 < 80:
                    cut_30 = None

                div_60 = None
                div_30 = None
                if cut_60 is not None:
                    div_60 = self._detect_divergence(
                        close_60[:cut_60], low_60[:cut_60], dif_60[:cut_60],
                        lookback=60, order=5)
                if cut_30 is not None:
                    div_30 = self._detect_divergence(
                        close_30[:cut_30], low_30[:cut_30], dif_30[:cut_30],
                        lookback=80, order=5)

                # 检查背离是否在最近1-2天发生
                if div_60 and div_30:
                    s_triggered = True
                    all_signals['S'].append(i)

            # ── B级：通道支撑+企稳 ──
            sub_close = close_d[:i+1]
            sub_low = low_d[:i+1]
            sub_high = high_d[:i+1]
            sub_vol = vol_d[:i+1] if vol_d is not None else None
            sub_f1 = f1_d[:i+1]
            sub_f2 = f2_d[:i+1]

            triggered, desc = self.detect_b_level(
                sub_close, sub_low, sub_high, sub_vol, sub_f1, sub_f2)
            if triggered:
                all_signals['B'].append(i)

            # ── C级：f''拐点（日线） ──
            # 检测第i天的f''是否从负转正
            if i >= 2:
                if f2_d[i] > 0 and f2_d[i-1] <= 0:
                    all_signals['C'].append(i)

            # ── C60级：60min f''拐点 ──
            if close_60 is not None and len(close_60) > 60:
                # 修复(2026-09-10): 原 sub_c60 = close_60[:len(close_60)] 用"全样本最后两根60min"
                # 的 f'' gate 每一天i(条件不引用i) → 一旦末bar f''转正, C60 会在整个样本的每一天
                # 同时触发 = 真·未来泄漏, 该回测结果无效。改为与S级同法: 只用到当天及之前的60min
                # 前缀, 检测"前缀末根"的f''是否刚从负转正(因果, 触发日收敛到真实拐点所在日)。
                day_ts = str(dates_d[i])[:10]
                mask_60 = np.array([str(t)[:10] <= day_ts for t in d60])
                cut_c60 = int(mask_60.sum())
                if cut_c60 >= 60:
                    _, f2_60_sub = self._compute_f2_safe(close_60[:cut_c60])
                    if len(f2_60_sub) >= 2 and f2_60_sub[-1] > 0 and f2_60_sub[-2] <= 0:
                        all_signals['C60'].append(i)

        # 统计收益
        results = {}
        for lvl, indices in all_signals.items():
            if not indices:
                continue

            rets_5d, rets_10d, rets_20d = [], [], []
            for idx in indices:
                if idx + 20 < n:
                    entry = open_d[idx + 1]  # 次日开盘价入场
                    ret5 = (close_d[idx + 5] / entry - 1) * 100
                    ret10 = (close_d[idx + 10] / entry - 1) * 100
                    ret20 = (close_d[idx + 20] / entry - 1) * 100
                    rets_5d.append(ret5)
                    rets_10d.append(ret10)
                    rets_20d.append(ret20)

            if rets_20d:
                results[lvl] = BacktestResult(
                    level=lvl,
                    total_signals=len(indices),
                    win_rate_5d=sum(1 for r in rets_5d if r > 0) / len(rets_5d) * 100,
                    win_rate_10d=sum(1 for r in rets_10d if r > 0) / len(rets_10d) * 100,
                    win_rate_20d=sum(1 for r in rets_20d if r > 0) / len(rets_20d) * 100,
                    avg_ret_5d=np.mean(rets_5d),
                    avg_ret_10d=np.mean(rets_10d),
                    avg_ret_20d=np.mean(rets_20d),
                    signals=[str(dates_d[i])[:10] if dates_d is not None else str(i) for i in indices[-10:]],
                )

        return results

    def format_backtest(self, results: dict, symbol: str = '', name: str = '') -> str:
        """格式化回测报告。"""
        lines = [f"📊 {name}({symbol}) 多级别买入信号回测"]

        level_names = {
            'S': 'S级·底背离共振',
            'B': 'B级·通道支撑+企稳',
            'C': 'C级·日线f\'\'拐点',
            'C60': 'C级·60min f\'\'拐点',
        }

        for lvl in ['S', 'B', 'C', 'C60']:
            r = results.get(lvl)
            if r is None:
                lines.append(f"\n{level_names.get(lvl, lvl)}: 无信号")
                continue
            lines.append(f"\n{level_names.get(lvl, lvl)}")
            lines.append(f"  信号次数: {r.total_signals}")
            lines.append(f"  5日胜率: {r.win_rate_5d:.0f}% | 平均{r.avg_ret_5d:+.1f}%")
            lines.append(f"  10日胜率: {r.win_rate_10d:.0f}% | 平均{r.avg_ret_10d:+.1f}%")
            lines.append(f"  20日胜率: {r.win_rate_20d:.0f}% | 平均{r.avg_ret_20d:+.1f}%")
            if r.signals:
                lines.append(f"  最近信号: {', '.join(r.signals)}")

        return "\n".join(lines)


# ====== CLI ======
if __name__ == '__main__':
    import argparse, sys
    from pathlib import Path

    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description='多级别买入信号检测器')
    parser.add_argument('symbol', help='股票代码')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--price', type=float, help='当前实时价格')
    args = parser.parse_args()

    from scripts.core.data_store import DataStore
    store = DataStore()

    df_d = store.get_kline(args.symbol, 'D')
    df_60 = store.get_kline(args.symbol, '60min')
    df_30 = store.get_kline(args.symbol, '30min')

    detector = BuySignalDetector()

    if args.backtest:
        results = detector.backtest(df_d, df_60, df_30)
        print(detector.format_backtest(results, symbol=args.symbol))
    else:
        signals = detector.detect(df_d, df_60, df_30, current_price=args.price)
        if signals:
            for s in signals:
                print(f"[{s.level}] {s.level_name} | {s.date} | {s.price:.2f} | 置信度{ s.confidence:.0%}")
                print(f"  {s.details}")
        else:
            print("无买入信号")
