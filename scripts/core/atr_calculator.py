#!/usr/bin/env python3
"""
ATR（Average True Range）计算模块 — 海龟交易法的基础度量衡。

Wilder平滑法：
  TR = max(high - low, |high - prev_close|, |low - prev_close|)
  第一ATR = SMA(TR, period)
  后续ATR = (prev_ATR × (period-1) + TR) / period

用法:
    from scripts.core.atr_calculator import ATRCalculator
    atr = ATRCalculator.calculate(df)       # 返回最新ATR值
    series = ATRCalculator.calculate_series(df)  # 返回ATR序列

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class ATRCalculator:
    """ATR计算器，支持日线和分钟线。"""

    # ── 核心计算 ──────────────────────────────────────────

    @staticmethod
    def calculate_true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """计算True Range（真实波幅）。

        TR = max(
            当日最高 - 当日最低,
            |当日最高 - 昨收|,
            |当日最低 - 昨收|
        )
        第一根K线无昨收时，TR = high - low。
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # 合并：第一根用 tr1
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    @staticmethod
    def calculate_series(
        df: pd.DataFrame,
        period: int = 20,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """计算完整ATR序列（Wilder平滑法）。

        Args:
            df: K线DataFrame，必须包含 high/low/close 列
            period: ATR周期，默认20（海龟标准）
            high_col/low_col/close_col: 列名

        Returns:
            ATR Series，索引与df对齐，前period-1个为NaN
        """
        if len(df) < period + 1:
            return pd.Series(np.nan, index=df.index)

        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        close = df[close_col].astype(float)

        tr = ATRCalculator.calculate_true_range(high, low, close)

        # Wilder平滑：先SMA初始化，再EMA-like递推
        atr = pd.Series(np.nan, index=df.index, dtype=float)

        # 第一个有效ATR = 前 period 个 TR 的简单平均
        if len(tr) >= period:
            atr.iloc[period] = tr.iloc[1:period + 1].mean()

            # 后续用 Wilder 递推
            for i in range(period + 1, len(df)):
                atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

        return atr

    @staticmethod
    def calculate(
        df: pd.DataFrame,
        period: int = 20,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> float:
        """返回最新的ATR值。

        Args:
            df: K线DataFrame
            period: 周期，默认20

        Returns:
            最新ATR值，数据不足返回0.0
        """
        atr_series = ATRCalculator.calculate_series(
            df, period, high_col, low_col, close_col
        )
        val = atr_series.dropna()
        if val.empty:
            return 0.0
        return round(val.iloc[-1], 4)

    @staticmethod
    def calculate_batch(
        df: pd.DataFrame,
        periods: list[int] = None,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> dict[int, float]:
        """批量计算多个周期的ATR。

        Args:
            df: K线DataFrame
            periods: 周期列表，默认 [14, 20]

        Returns:
            {period: atr_value}
        """
        if periods is None:
            periods = [14, 20]
        return {
            p: ATRCalculator.calculate(df, p, high_col, low_col, close_col)
            for p in periods
        }

    @staticmethod
    def get_atr_pct(
        df: pd.DataFrame,
        period: int = 20,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> float:
        """ATR占股价的百分比（波动率指标）。

        用途：快速判断个股波动性大小
        - <2%: 低波动（大盘蓝筹）
        - 2-4%: 正常波动
        - 4-6%: 高波动（中小盘）
        - >6%: 极端波动

        Returns:
            ATR / 最新收盘价 × 100
        """
        atr = ATRCalculator.calculate(df, period, high_col, low_col, close_col)
        if atr <= 0 or df.empty:
            return 0.0
        last_close = float(df[close_col].iloc[-1])
        if last_close <= 0:
            return 0.0
        return round(atr / last_close * 100, 2)


# ── 便捷接口：直接从DataStore取数计算 ──────────────────────

def get_atr_from_store(
    symbol: str,
    period: int = 20,
    kline_period: str = "D",
    store=None,
) -> tuple[float, float]:
    """从DataStore取K线并计算ATR。

    Args:
        symbol: 6位股票代码
        period: ATR周期
        kline_period: K线周期 (D/60min/15min)
        store: DataStore实例，None则自动创建

    Returns:
        (atr_value, last_close)
    """
    if store is None:
        import sys
        sys.path.insert(0, '.')
        from scripts.core.data_store import DataStore
        store = DataStore()

    df = store.get_kline(symbol, period=kline_period, auto_update=False)
    if df is None or df.empty or len(df) < period + 1:
        return 0.0, 0.0

    # 清理列名
    cols = {c.lower(): c for c in df.columns}
    high_col = cols.get('high', 'high')
    low_col = cols.get('low', 'low')
    close_col = cols.get('close', 'close')

    atr = ATRCalculator.calculate(df, period, high_col, low_col, close_col)
    last_close = float(df[close_col].iloc[-1])
    return atr, last_close


# ── CLI入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    # 演示
    print("=" * 55)
    print("📊 ATR计算模块 — 演示")
    print("=" * 55)

    test_codes = ['600519', '000001', '002594', '688981']
    try:
        from scripts.core.data_store import DataStore
        store = DataStore()

        for code in test_codes:
            atr, close = get_atr_from_store(code, period=20, store=store)
            if atr > 0 and close > 0:
                atr_pct = atr / close * 100
                print(f"\n{code}:")
                print(f"  收盘价: {close:.2f}")
                print(f"  ATR(20): {atr:.3f}")
                print(f"  波动率: {atr_pct:.2f}%")
                print(f"  2ATR止损幅度: {2 * atr_pct:.1f}%")
                print(f"  0.5ATR加仓间距: {close + 0.5 * atr:.2f}")
            else:
                print(f"\n{code}: 数据不足")
    except Exception as e:
        print(f"演示需要本地数据: {e}")

    # 纯算法演示（不依赖数据源）
    print("\n" + "=" * 55)
    print("🧪 纯算法验证（模拟数据）")
    print("=" * 55)

    np.random.seed(42)
    n = 60
    base_price = 10.0
    returns = np.random.normal(0, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'close': prices,
    })

    for p in [14, 20]:
        atr = ATRCalculator.calculate(df, period=p)
        atr_pct = ATRCalculator.get_atr_pct(df, period=p)
        print(f"  ATR({p:2d}): {atr:.4f}  ({atr_pct:.2f}% of price)")

    # 验证：ATR序列的连续性
    atr_series = ATRCalculator.calculate_series(df, period=20)
    valid = atr_series.dropna()
    print(f"\n  ATR(20)序列: 共{len(valid)}个有效值")
    print(f"  首/末: {valid.iloc[0]:.4f} → {valid.iloc[-1]:.4f}")
    print(f"  均值: {valid.mean():.4f}, 标准差: {valid.std():.4f}")
