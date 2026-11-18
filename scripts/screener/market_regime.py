#!/usr/bin/env python3
"""大盘阶段判断器 — 基于f'/f''/均线体系，自上而下驱动策略选择。

核心理念（来自大浪主"战略>战役>战术"思维）：
  1. 先判断大盘处于什么阶段
  2. 根据阶段决定策略权重（弹簧起涨/底部反转/趋势加速/空仓）
  3. 选股在合适的策略下进行

阶段划分（基于沪深300）：
  🟢 主升浪  (trend_up)     — f'>0, f''>0, close>MA60, 均线多头
  🟡 震荡偏多 (range_bull)   — f'≈0 或 f'>0但f''<0, close>MA60
  🟠 震荡偏空 (range_bear)   — f'≈0 或 f'<0但f''>0, close<MA60
  🔴 下跌    (trend_down)    — f'<0, f''<0, close<MA60
  ⚪ 拐点    (inflection)    — f''刚从负转正或正转负，关键转折

用法：
  PYTHONPATH=. python scripts/screener/market_regime.py
  PYTHONPATH=. python scripts/screener/market_regime.py --index 000300SH
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class MarketRegime:
    """大盘阶段判断结果。"""
    regime_key: str = "unknown"        # trend_up / range_bull / range_bear / trend_down / inflection
    label: str = "⚪未知"
    confidence: float = 0.0            # 0-1，判断置信度
    f1: float = 0.0                    # 一阶导（趋势方向）
    f2: float = 0.0                    # 二阶导（加速度）
    above_ma20: bool = False
    above_ma60: bool = False
    ma_alignment: str = "无"           # 多头排列/空头排列/交叉
    vol_trend: str = "正常"            # 放量/缩量/正常
    strategy_weights: dict = field(default_factory=dict)  # 各策略推荐权重
    advice: str = ""


# ── 均线计算 ──────────────────────────────────────────────
def _sma(series: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均，末尾NaN填充。"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    return pd.Series(series).rolling(period, min_periods=1).mean().values


# ── 导数计算（与V3体系一致）──────────────────────────────
def _compute_derivatives(closes: np.ndarray, window: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """对ln(close)用savgol求一阶/二阶导。与V3DerivativeScorer一致。"""
    log_p = np.log(closes.astype(float))
    # 填充NaN
    valid = ~np.isnan(log_p)
    if valid.sum() < window:
        return np.zeros(len(closes)), np.zeros(len(closes))

    f1 = np.zeros(len(closes))
    f2 = np.zeros(len(closes))
    if valid.sum() >= window:
        from scripts.core.derivatives import causal_savgol   # 2026-09-10: 权威因果(pos=0右端), 弃中心化
        sg1 = causal_savgol(log_p[valid], window, 3, deriv=1)
        sg2 = causal_savgol(log_p[valid], window, 3, deriv=2)
        f1[valid] = sg1
        f2[valid] = sg2

    return f1, f2


# ── 均线排列判断 ──────────────────────────────────────────
def _check_ma_alignment(closes: np.ndarray) -> tuple[str, float]:
    """检查5/10/20/30/60/120/250日均线排列。

    Returns:
        (状态, 发散度)
        状态: "多头排列" / "空头排列" / "交叉"
        发散度: 0-1, 越大越整齐
    """
    periods = [5, 10, 20, 30, 60, 120, 250]
    mas = {}
    for p in periods:
        ma = _sma(closes, p)
        mas[p] = ma[-1] if not np.isnan(ma[-1]) else None

    valid_mas = {k: v for k, v in mas.items() if v is not None}
    if len(valid_mas) < 4:
        return "交叉", 0.0

    # 按周期排序，短的应该在前面（多头）
    sorted_periods = sorted(valid_mas.keys())
    values = [valid_mas[p] for p in sorted_periods]

    # 多头排列：所有短周期 > 长周期
    bullish = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    # 空头排列：所有短周期 < 长周期
    bearish = all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    # 发散度：间距的一致性
    if len(values) >= 2:
        diffs = [values[i] - values[i + 1] for i in range(len(values) - 1)]
        avg_diff = np.mean([abs(d) for d in diffs])
        price = closes[-1]
        divergence = min(avg_diff / price * 100, 1.0)  # 归一化
    else:
        divergence = 0.0

    if bullish:
        return "多头排列", divergence
    elif bearish:
        return "空头排列", divergence
    else:
        return "交叉", 0.0


# ── 量能趋势 ──────────────────────────────────────────────
def _check_vol_trend(volumes: np.ndarray) -> str:
    """近5日均量 vs 近20日均量。"""
    if len(volumes) < 20:
        return "正常"
    vol5 = np.mean(volumes[-5:])
    vol20 = np.mean(volumes[-20:])
    ratio = vol5 / vol20 if vol20 > 0 else 1.0
    if ratio > 1.3:
        return "放量"
    elif ratio < 0.7:
        return "缩量"
    return "正常"


class MarketRegimeDetector:
    """大盘阶段判断器。"""

    def __init__(self, data_root: str | Path = "data/store"):
        self.data_root = Path(data_root)

    def _load_index(self, ts_code: str = "000300SH") -> pd.DataFrame | None:
        """加载指数日线数据。优先DataStore，回退Tushare。"""
        # 尝试列表：优先请求的代码，回退上证指数
        candidates = [ts_code]
        if ts_code != "000001SH":
            candidates.append("000001SH")

        for code in candidates:
            cache_path = self.data_root / "kline" / code / "D.parquet"
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    dt_col = "trade_date" if "trade_date" in df.columns else "date"
                    df[dt_col] = pd.to_datetime(df[dt_col])
                    df = df.sort_values(dt_col).reset_index(drop=True)
                    if len(df) > 120:
                        return df
                except Exception:
                    pass

        # 方法2: DataStore API
        try:
            from scripts.core.data_store import DataStore
            ds = DataStore(self.data_root)
            df = ds.get_index_kline(ts_code, auto_update=True)
            if df is not None and len(df) > 120:
                return df
        except Exception:
            pass

        # 方法3: Tushare在线
        try:
            import os
            token = os.environ.get("TUSHARE_TOKEN")
            if not token:
                from scripts.core.tushare_client import _DEFAULT_TOKEN
                token = _DEFAULT_TOKEN
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()
            # 上证: 000001.SH, 沪深300: 000300.SH
            api_code = ts_code.replace("SH", ".SH").replace("SZ", ".SZ")
            if "." not in api_code:
                api_code = api_code[:-2] + "." + api_code[-2:]
            df = pro.index_daily(ts_code=api_code, start_date="20240101")
            if df is not None and len(df) > 120:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.sort_values("trade_date").reset_index(drop=True)
                return df
        except Exception:
            pass

        return None

    def detect(self, ts_code: str = "000300SH") -> MarketRegime:
        """判断当前大盘阶段。

        Args:
            ts_code: 指数代码，默认沪深300。可选 000001SH(上证)

        Returns:
            MarketRegime 判断结果
        """
        result = MarketRegime()
        df = self._load_index(ts_code)
        if df is None or len(df) < 120:
            result.advice = "⚠️ 无法加载指数数据，请检查数据源"
            return result

        closes = df["close"].values.astype(float)
        volumes = df["vol"].values.astype(float) if "vol" in df.columns else np.ones(len(df))

        # ── 核心指标计算 ──
        f1, f2 = _compute_derivatives(closes)
        current_f1 = f1[-1]
        current_f2 = f2[-1]

        # 上一个f2用于判断拐点
        prev_f2 = f2[-2] if len(f2) > 1 else 0

        # 均线
        ma5 = _sma(closes, 5)[-1]
        ma20 = _sma(closes, 20)[-1]
        ma60 = _sma(closes, 60)[-1]
        close = closes[-1]

        above_ma20 = close > ma20 if not np.isnan(ma20) else False
        above_ma60 = close > ma60 if not np.isnan(ma60) else False

        # 均线排列
        ma_align, ma_div = _check_ma_alignment(closes)

        # 量能趋势
        vol_trend = _check_vol_trend(volumes)

        # ── 填充结果 ──
        result.f1 = round(float(current_f1), 4)
        result.f2 = round(float(current_f2), 4)
        result.above_ma20 = above_ma20
        result.above_ma60 = above_ma60
        result.ma_alignment = ma_align
        result.vol_trend = vol_trend

        # ── 阶段判断 ──
        # 拐点检测：f''刚从负转正 或 正转负
        is_inflection_up = prev_f2 < 0 and current_f2 > 0
        is_inflection_down = prev_f2 > 0 and current_f2 < 0

        if is_inflection_up or is_inflection_down:
            result.regime_key = "inflection"
            if is_inflection_up:
                result.label = "⚪拐点(↑加速)"
                result.confidence = 0.7
                result.advice = "f''从负转正，市场可能即将反转向上。关注弹簧起涨和底部反转信号。"
                result.strategy_weights = {
                    "弹簧起涨": 0.4,
                    "底部反转": 0.35,
                    "趋势加速": 0.15,
                    "空仓等待": 0.1,
                }
            else:
                result.label = "⚪拐点(↓减速)"
                result.confidence = 0.7
                result.advice = "f''从正转负，上涨动能衰减。已有持仓注意止盈，新开仓需谨慎。"
                result.strategy_weights = {
                    "弹簧起涨": 0.1,
                    "底部反转": 0.1,
                    "趋势加速": 0.1,
                    "空仓等待": 0.7,
                }

        elif current_f1 > 0.01 and current_f2 > 0.005 and above_ma60:
            result.regime_key = "trend_up"
            result.label = "🟢主升浪"
            result.confidence = 0.85
            result.advice = "大盘主升浪阶段，趋势加速策略优先。持股为主，回调就是机会。"
            result.strategy_weights = {
                "弹簧起涨": 0.15,
                "底部反转": 0.1,
                "趋势加速": 0.65,
                "空仓等待": 0.1,
            }

        elif current_f1 > 0 and current_f2 < -0.005 and above_ma60:
            result.regime_key = "range_bull"
            result.label = "🟡震荡偏多"
            result.confidence = 0.6
            result.advice = "大盘高位震荡，趋势在但动能衰减。弹簧起涨和精选底部反转为主。"
            result.strategy_weights = {
                "弹簧起涨": 0.45,
                "底部反转": 0.25,
                "趋势加速": 0.2,
                "空仓等待": 0.1,
            }

        elif current_f1 > 0 and above_ma20 and not above_ma60:
            # f'>0 但还在MA60下面 — 弱反弹
            result.regime_key = "range_bear"
            result.label = "🟠震荡偏空"
            result.confidence = 0.55
            result.advice = "大盘弱势反弹中，趋势不明。控制仓位，弹簧起涨为主。"
            result.strategy_weights = {
                "弹簧起涨": 0.5,
                "底部反转": 0.25,
                "趋势加速": 0.05,
                "空仓等待": 0.2,
            }
            result.notes = "回测验证: 此阶段数据不足(仅2个月)，暂不加仓"

        elif current_f1 < -0.01 and current_f2 < -0.005 and not above_ma60:
            result.regime_key = "trend_down"
            result.label = "🔴下跌"
            result.confidence = 0.8
            result.advice = "大盘下跌趋势，防守为主。只做超跌反弹，严格控制仓位和止损。"
            result.strategy_weights = {
                "弹簧起涨": 0.2,
                "底部反转": 0.3,
                "趋势加速": 0.0,
                "空仓等待": 0.5,
            }

        elif current_f1 < 0 and current_f2 > 0.005:
            # f'<0 但 f''>0 — 下跌减速，可能见底
            result.regime_key = "inflection"
            result.label = "⚪拐点(↓减速)"
            result.confidence = 0.65
            result.advice = "大盘下跌减速，关注底部反转信号。小仓位试探，不急着重仓。"
            result.strategy_weights = {
                "弹簧起涨": 0.3,
                "底部反转": 0.4,
                "趋势加速": 0.05,
                "空仓等待": 0.25,
            }

        elif current_f1 < 0 and above_ma60:
            # f'<0 但还在MA60上面 — 回调中
            # ⚠️ 回测验证: 回调期趋势加速胜率仅29%(p<0.001)，必须严控
            result.regime_key = "range_bull"
            result.label = "🟡震荡偏多(回调)"
            result.confidence = 0.6
            result.advice = "大盘回调中(回测验证:趋势加速胜率29%)，严控仓位，弹簧起涨为主，禁止追趋势。"
            result.strategy_weights = {
                "弹簧起涨": 0.55,
                "底部反转": 0.3,
                "趋势加速": 0.0,
                "空仓等待": 0.15,
            }

        elif current_f1 < 0 and not above_ma60 and current_f2 > 0:
            # f'<0, f''>0, 低于MA60 — 下跌但减速
            result.regime_key = "range_bear"
            result.label = "🟠震荡偏空(减速)"
            result.confidence = 0.5
            result.advice = "大盘弱势但有企稳迹象，关注底部反转信号。小仓位试探。"
            result.strategy_weights = {
                "弹簧起涨": 0.3,
                "底部反转": 0.35,
                "趋势加速": 0.0,
                "空仓等待": 0.35,
            }

        else:
            # 默认：方向不明
            if above_ma60:
                result.regime_key = "range_bull"
                result.label = "🟡震荡偏多"
                result.confidence = 0.35
                result.advice = "大盘方向不明，偏多震荡。精选个股，控制仓位。"
                result.strategy_weights = {
                    "弹簧起涨": 0.35,
                    "底部反转": 0.2,
                    "趋势加速": 0.15,
                    "空仓等待": 0.3,
                }
            else:
                result.regime_key = "range_bear"
                result.label = "🟠震荡偏空"
                result.confidence = 0.35
                result.advice = "大盘方向不明，偏弱。控制仓位，等明确信号。"
                result.strategy_weights = {
                    "弹簧起涨": 0.2,
                    "底部反转": 0.2,
                    "趋势加速": 0.05,
                    "空仓等待": 0.55,
                }

        # ── 均线排列加成 ──
        if ma_align == "多头排列" and result.regime_key in ("trend_up", "range_bull"):
            result.confidence = min(result.confidence + 0.1, 1.0)
            result.advice += " 均线多头排列，信号更可靠。"
        elif ma_align == "空头排列" and result.regime_key in ("trend_down", "range_bear"):
            result.confidence = min(result.confidence + 0.1, 1.0)
            result.advice += " 均线空头排列，谨慎。"

        # ── 量能加成 ──
        if vol_trend == "放量" and result.regime_key in ("trend_up", "inflection"):
            result.confidence = min(result.confidence + 0.05, 1.0)

        return result

    def print_report(self, regime: MarketRegime, ts_code: str = "000300SH"):
        """打印格式化的阶段报告。"""
        print("=" * 60)
        print(f"📊 大盘阶段判断 — {ts_code}")
        print("=" * 60)
        print(f"  阶段:     {regime.label}")
        print(f"  置信度:   {regime.confidence:.0%}")
        print(f"  f'(趋势): {regime.f1:+.4f}")
        print(f"  f''(加速): {regime.f2:+.4f}")
        print(f"  MA20:     {'✅ 站上' if regime.above_ma20 else '❌ 跌破'}")
        print(f"  MA60:     {'✅ 站上' if regime.above_ma60 else '❌ 跌破'}")
        print(f"  均线排列: {regime.ma_alignment}")
        print(f"  量能:     {regime.vol_trend}")
        print()
        print("📋 策略推荐权重:")
        for strategy, weight in sorted(regime.strategy_weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
            print(f"  {strategy:8s} {bar} {weight:.0%}")
        print()
        print(f"💡 {regime.advice}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="大盘阶段判断器")
    parser.add_argument("--index", default="000300SH", help="指数代码(默认沪深300)")
    parser.add_argument("--data-root", default="data/store", help="数据目录")
    args = parser.parse_args()

    detector = MarketRegimeDetector(data_root=args.data_root)
    regime = detector.detect(ts_code=args.index)
    detector.print_report(regime, ts_code=args.index)


if __name__ == "__main__":
    main()
