#!/usr/bin/env python3
"""成交量分析器 — 量价确认型。

定位：不单独产生买卖信号，而是确认或质疑其他维度的信号。
量在价先，但不孤立判断。成交量用于加权或减权 MACD/MA/RSI 等信号。

核心能力：
  1. 量价关系8种模式分类（量增价升、量缩价跌等）
  2. OBV（资金趋势 + 背离检测）
  3. MFI（量加权RSI）
  4. CMF（Chaikin资金流）
  5. VWAP（日内机构成本线，仅分钟级）
  6. 成交量分类（天量/放量/正常/缩量/地量）
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.core.tushare_client import TushareClient

# ── 阈值常量 ──────────────────────────────────────────────────────────

_VOL_THRESHOLDS = {
    "天量": 3.0,
    "放量": 1.5,
    "缩量": 0.7,
    "地量": 0.33,
}

_MFI_THRESHOLDS = {
    "超买": 80,
    "超卖": 20,
}

_CMF_PERIOD = 20
_MFI_PERIOD = 14


# ── 纯计算函数 ────────────────────────────────────────────────────────

def _calc_volume_class(vol_ratio: float) -> str:
    """量比 → 成交量分类"""
    if vol_ratio >= _VOL_THRESHOLDS["天量"]:
        return "天量"
    if vol_ratio >= _VOL_THRESHOLDS["放量"]:
        return "放量"
    if vol_ratio <= _VOL_THRESHOLDS["地量"]:
        return "地量"
    if vol_ratio < _VOL_THRESHOLDS["缩量"]:
        return "缩量"
    return "正常"


def _calc_vp_pattern(vol_ratio: float, price_chg: float,
                     is_high: bool = False, is_low: bool = False) -> str:
    """量比+价变 → 量价关系模式"""
    vol_up = vol_ratio >= _VOL_THRESHOLDS["放量"]
    vol_down = vol_ratio < _VOL_THRESHOLDS["缩量"]
    price_up = price_chg > 0.3
    price_down = price_chg < -0.3
    price_flat = abs(price_chg) <= 0.3

    # 特殊模式
    if vol_ratio >= _VOL_THRESHOLDS["天量"] and is_high:
        return "天量天价"
    if vol_ratio <= _VOL_THRESHOLDS["地量"] and is_low:
        return "地量地价"

    if vol_up and price_up:
        return "量增价升"
    if vol_down and price_up:
        return "量缩价升"
    if vol_up and price_down:
        return "量增价跌"
    if vol_down and price_down:
        return "量缩价跌"
    if vol_up and price_flat:
        return "量增价平"
    if vol_down and price_flat:
        return "量缩价平"
    # 量稳（正常量）时仍按价变方向描述
    if price_up:
        return "量平价升"
    if price_down:
        return "量平价跌"
    return "量平价平"


def _calc_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    return (direction * df["volume"]).cumsum()


def _calc_mfi(df: pd.DataFrame, period: int = _MFI_PERIOD) -> pd.Series:
    """Money Flow Index (0-100)"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    direction = tp.diff()
    pos_mf = pd.Series(np.where(direction > 0, mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(direction < 0, mf, 0), index=df.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mf_ratio)


def _calc_cmf(df: pd.DataFrame, period: int = _CMF_PERIOD) -> pd.Series:
    """Chaikin Money Flow (-1 ~ +1)"""
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, np.nan)
    mfv = clv * df["volume"]
    return mfv.rolling(period).sum() / df["volume"].rolling(period).sum()


def _calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price（日内累计，按交易日重置）"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = tp * df["volume"]
    # 按交易日分组重置累计，避免跨日累积
    if 'date' in df.columns:
        df_copy = df.copy()
        df_copy['_day'] = pd.to_datetime(df_copy['date']).dt.date
        df_copy['_tp_vol'] = tp_vol.values
        df_copy['_vol'] = df['volume'].values
        vwap = df_copy.groupby('_day').apply(
            lambda g: g['_tp_vol'].cumsum() / g['_vol'].cumsum(),
            include_groups=False
        ).droplevel(0)
        return vwap
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)


def _detect_obv_divergence(df: pd.DataFrame, obv: pd.Series,
                           lookback: int = 30) -> Optional[str]:
    """OBV背离检测"""
    if len(df) < lookback + 5:
        return None
    recent = df.tail(lookback)
    recent_obv = obv.tail(lookback)

    # 顶背离：价格新高但OBV未新高
    if recent["high"].iloc[-1] >= recent["high"].iloc[-5]:
        high_idx = recent["high"].idxmax()
        if recent_obv.iloc[-1] < recent_obv.loc[high_idx]:
            return "顶背离"

    # 底背离：价格新低但OBV未新低
    if recent["low"].iloc[-1] <= recent["low"].iloc[-5]:
        low_idx = recent["low"].idxmin()
        if recent_obv.iloc[-1] > recent_obv.loc[low_idx]:
            return "底背离"

    return None


# ── 主类 ──────────────────────────────────────────────────────────────

class VolumeAnalyzer:
    """成交量分析器 — 量价确认型"""

    def __init__(self, symbol: Optional[str] = None):
        self.symbol = symbol
        self._ts = TushareClient() if symbol else None

    # ── 单级别分析（纯计算，不获取数据）──────────────────────────────

    def analyze_level(self, df: pd.DataFrame,
                      is_minute: bool = False) -> Dict:
        """对单个时间级别的K线DataFrame计算全部量价指标。

        Args:
            df: K线数据，需含列 date, open, high, low, close, volume
            is_minute: 是否为分钟线（影响VWAP计算）

        Returns:
            量价指标字典
        """
        if df is None or len(df) < 5:
            return {}
        price = float(df["close"].iloc[-1])
        result = {}

        # ── 基础量价 ──
        vol_ma5 = df["volume"].rolling(5).mean()
        vol_ma20 = df["volume"].rolling(20).mean()
        vol_ratio = float(df["volume"].iloc[-1] / vol_ma5.iloc[-1]) \
            if not np.isnan(vol_ma5.iloc[-1]) and vol_ma5.iloc[-1] > 0 else 1.0
        price_chg = float((df["close"].iloc[-1] - df["close"].iloc[-2])
                          / df["close"].iloc[-2] * 100) if len(df) > 1 else 0.0

        # 判断高位/低位（用布林位置近似）
        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std(ddof=0)
        is_high = price > float(mid.iloc[-1]) + 1.5 * float(std.iloc[-1]) \
            if not np.isnan(mid.iloc[-1]) else False
        is_low = price < float(mid.iloc[-1]) - 1.5 * float(std.iloc[-1]) \
            if not np.isnan(mid.iloc[-1]) else False

        result["vol_ratio"] = round(vol_ratio, 2)
        result["vol_ma5"] = float(vol_ma5.iloc[-1]) if not np.isnan(vol_ma5.iloc[-1]) else 0
        result["vol_ma20"] = float(vol_ma20.iloc[-1]) if not np.isnan(vol_ma20.iloc[-1]) else 0
        result["vol_class"] = _calc_volume_class(vol_ratio)
        result["vp_pattern"] = _calc_vp_pattern(vol_ratio, price_chg, is_high, is_low)
        result["price_chg"] = round(price_chg, 2)

        # ── OBV ──
        obv = _calc_obv(df)
        result["obv"] = float(obv.iloc[-1])
        obv_ma5 = obv.rolling(5).mean()
        result["obv_trend"] = "上升" if obv.iloc[-1] > obv_ma5.iloc[-1] else \
                              "下降" if obv.iloc[-1] < obv_ma5.iloc[-1] else "平"
        result["obv_divergence"] = _detect_obv_divergence(df, obv)

        # ── MFI ──
        mfi = _calc_mfi(df)
        mfi_val = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50.0
        result["mfi"] = round(mfi_val, 1)

        # ── CMF ──
        cmf = _calc_cmf(df)
        cmf_val = float(cmf.iloc[-1]) if not np.isnan(cmf.iloc[-1]) else 0.0
        result["cmf"] = round(cmf_val, 4)

        # ── VWAP（仅分钟线）──
        if is_minute:
            vwap = _calc_vwap(df)
            vwap_val = float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else 0.0
            result["vwap"] = round(vwap_val, 2)
            result["vwap_deviation"] = round((price - vwap_val) / vwap_val * 100, 2) \
                if vwap_val > 0 else 0.0
        else:
            result["vwap"] = None
            result["vwap_deviation"] = None

        # ── 量价信号摘要 ──
        signals = self._collect_signals(result, price_chg)
        result["volume_signals"] = signals

        return result

    # ── 信号收集 ──────────────────────────────────────────────────────

    @staticmethod
    def _collect_signals(r: Dict, price_chg: float) -> List[str]:
        signals = []
        vp = r.get("vp_pattern", "")
        vc = r.get("vol_class", "")

        # 量价关系信号
        confirm_map = {
            "量增价升": "放量上涨，多头确认",
            "量缩价升": "缩量上涨，上涨动力不足",
            "量增价跌": "放量下跌，抛压沉重",
            "量缩价跌": "缩量下跌，卖盘枯竭",
            "天量天价": "天量天价，见顶风险",
            "地量地价": "地量地价，底部信号",
            "量增价平": "放量滞涨，变盘预警",
            "量平价升": "温和上涨",
            "量平价跌": "温和下跌",
        }
        if vp in confirm_map:
            signals.append(confirm_map[vp])

        # OBV背离
        div = r.get("obv_divergence")
        if div == "底背离":
            signals.append("OBV底背离，资金面支撑")
        elif div == "顶背离":
            signals.append("OBV顶背离，资金面走弱")

        # MFI
        mfi = r.get("mfi", 50)
        if mfi > 80:
            signals.append(f"MFI超买({mfi:.0f})")
        elif mfi < 20:
            signals.append(f"MFI超卖({mfi:.0f})")

        # CMF
        cmf = r.get("cmf", 0)
        if cmf > 0.1:
            signals.append("CMF资金流入")
        elif cmf < -0.1:
            signals.append("CMF资金流出")

        # 地量/天量
        if vc == "地量":
            signals.append("地量水平，变盘窗口")
        elif vc == "天量":
            signals.append("天量成交，关注方向")

        return signals

    # ── 确认器 ────────────────────────────────────────────────────────

    def confirm_signal(self, signal_type: str, direction: str,
                       level_result: Dict) -> Dict:
        """给其他维度的买卖信号加权或减权。

        Args:
            signal_type: 'macd_cross', 'ma_break', 'rsi_ob', 'divergence' 等
            direction: 'buy' 或 'sell'
            level_result: analyze_level() 返回的量价结果

        Returns:
            {'modifier': 0.7~1.3, 'confirmation': str, 'reason': str}
        """
        if not level_result:
            return {"modifier": 1.0, "confirmation": "neutral", "reason": "无量价数据"}

        modifier = 1.0
        reasons = []
        vp = level_result.get("vp_pattern", "")
        obv_div = level_result.get("obv_divergence")
        mfi = level_result.get("mfi", 50)
        cmf = level_result.get("cmf", 0)
        vc = level_result.get("vol_class", "正常")

        if direction == "buy":
            # 确认多头信号
            if vp == "量增价升":
                modifier += 0.15
                reasons.append("量增价升确认")
            if vp == "地量地价":
                modifier += 0.2
                reasons.append("地量地价强底部")
            if vc == "地量":
                modifier += 0.1
                reasons.append("地量见底")
            if obv_div == "底背离":
                modifier += 0.15
                reasons.append("OBV底背离")
            if mfi < 20:
                modifier += 0.1
                reasons.append("MFI超卖")
            if cmf > 0.05:
                modifier += 0.05
                reasons.append("CMF资金流入")

            # 质疑多头信号
            if vp == "量缩价升":
                modifier -= 0.1
                reasons.append("缩量上涨动力不足")
            if vp == "量增价跌":
                modifier -= 0.15
                reasons.append("放量下跌")
            if vp == "天量天价":
                modifier -= 0.2
                reasons.append("天量天价见顶风险")
            if obv_div == "顶背离":
                modifier -= 0.15
                reasons.append("OBV顶背离")
            if mfi > 80:
                modifier -= 0.1
                reasons.append("MFI超买")

        elif direction == "sell":
            # 确认空头信号
            if vp == "量增价跌":
                modifier += 0.15
                reasons.append("放量下跌确认")
            if vp == "天量天价":
                modifier += 0.2
                reasons.append("天量天价")
            if obv_div == "顶背离":
                modifier += 0.15
                reasons.append("OBV顶背离")
            if cmf < -0.05:
                modifier += 0.05
                reasons.append("CMF资金流出")

            # 质疑空头信号
            if vp == "量缩价跌":
                modifier -= 0.1
                reasons.append("缩量下跌卖盘枯竭")
            if vp == "地量地价":
                modifier -= 0.2
                reasons.append("地量地价底部")
            if vc == "地量":
                modifier -= 0.1
                reasons.append("地量水平")
            if obv_div == "底背离":
                modifier -= 0.15
                reasons.append("OBV底背离")
            if mfi < 20:
                modifier -= 0.1
                reasons.append("MFI超卖")

        modifier = max(0.7, min(1.3, modifier))
        if not reasons:
            return {"modifier": 1.0, "confirmation": "neutral", "reason": "量价中性"}
        confirmation = "confirmed" if modifier > 1.0 else \
                       "questioned" if modifier < 1.0 else "neutral"
        return {
            "modifier": round(modifier, 2),
            "confirmation": confirmation,
            "reason": "；".join(reasons),
        }

    # ── 完整分析（获取数据+计算）──────────────────────────────────────

    def analyze(self, timeframes: Optional[List[str]] = None) -> Dict:
        """完整多时间尺度成交量分析"""
        if not self.symbol:
            return {"error": "未指定股票代码"}

        tfs = timeframes or ["daily", "60min"]
        levels = {}
        for tf in tfs:
            try:
                is_minute = tf in ("60min", "15min", "5min", "1min")
                freq = {"daily": "D", "weekly": "W", "60min": "60",
                        "15min": "15", "5min": "5"}.get(tf, "D")
                start = "20250101" if not is_minute else None
                df = self._ts.get_kline(self.symbol, freq=freq, start=start, adj="qfq")
                if df is not None and not df.empty:
                    levels[tf] = self.analyze_level(df, is_minute=is_minute)
            except Exception as e:
                levels[tf] = {"error": str(e)}

        return {
            "symbol": self.symbol,
            "timeframes": levels,
            "signals": self._generate_signals(levels),
        }

    def _generate_signals(self, levels: Dict) -> Dict:
        """汇总买卖信号"""
        buy_signals = []
        sell_signals = []

        for tf, r in levels.items():
            if "error" in r:
                continue
            tf_label = {"daily": "日线", "60min": "60分钟",
                        "15min": "15分钟", "weekly": "周线"}.get(tf, tf)
            for sig in r.get("volume_signals", []):
                # 分类买卖
                buy_kw = ["确认", "底部", "超卖", "流入", "地量", "枯竭"]
                sell_kw = ["不足", "抛压", "见顶", "超买", "流出", "走弱"]
                if any(kw in sig for kw in buy_kw):
                    buy_signals.append(f"{tf_label}{sig}")
                if any(kw in sig for kw in sell_kw):
                    sell_signals.append(f"{tf_label}{sig}")

        return {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
        }

    # ── 格式化 ───────────────────────────────────────────────────────

    @staticmethod
    def format_report(result: Dict) -> str:
        """格式化分析报告"""
        if "error" in result:
            return f"分析失败: {result['error']}"

        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"📊 成交量分析: {result.get('symbol', '?')}")
        lines.append(f"{'='*60}")

        tf_labels = {"daily": "日线", "60min": "60分钟",
                     "15min": "15分钟", "weekly": "周线"}

        for tf, r in result.get("timeframes", {}).items():
            if "error" in r:
                lines.append(f"\n── {tf_labels.get(tf, tf)} ── 获取失败: {r['error']}")
                continue
            label = tf_labels.get(tf, tf)
            lines.append(f"\n── {label} ──")

            # 量价关系
            lines.append(f"  量价: {r.get('vp_pattern', '?')}  "
                         f"量比{r.get('vol_ratio', 0):.2f}x ({r.get('vol_class', '?')})")
            lines.append(f"  价变: {r.get('price_chg', 0):+.2f}%")

            # OBV
            lines.append(f"  OBV: {r.get('obv_trend', '?')}  "
                         f"背离: {r.get('obv_divergence') or '无'}")

            # MFI
            mfi = r.get("mfi", 50)
            mfi_state = "超买" if mfi > 80 else "超卖" if mfi < 20 else \
                        "偏强" if mfi > 50 else "偏弱"
            lines.append(f"  MFI: {mfi:.1f} ({mfi_state})")

            # CMF
            cmf = r.get("cmf", 0)
            cmf_state = "流入" if cmf > 0.05 else "流出" if cmf < -0.05 else "平衡"
            lines.append(f"  CMF: {cmf:+.4f} ({cmf_state})")

            # VWAP
            vwap = r.get("vwap")
            if vwap:
                lines.append(f"  VWAP: {vwap:.2f}  偏离{r.get('vwap_deviation', 0):+.2f}%")

            # 信号
            for sig in r.get("volume_signals", []):
                lines.append(f"  → {sig}")

        # 汇总
        signals = result.get("signals", {})
        if signals.get("buy_signals"):
            lines.append(f"\n📈 量价多头信号:")
            for s in signals["buy_signals"]:
                lines.append(f"  {s}")
        if signals.get("sell_signals"):
            lines.append(f"\n📉 量价空头信号:")
            for s in signals["sell_signals"]:
                lines.append(f"  {s}")

        return "\n".join(lines)
