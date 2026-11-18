#!/usr/bin/env python3
"""波动率收敛回踩模式回测 — 用ATR/量能刻画"蓄力-突破-回踩"(2026-06-17)。

思路(MACD红柱是果, 波动率收敛是因):
- 蓄力 = 波动率收敛(ATR5/ATR20<1, 能量压缩) + 缩量(vol5/vol20<1)
- 突破 = 曾创新高(60日高 > MA20×1.05)
- 回踩 = 价格回MA20附近 + 波动收敛
- 均价不高 = dh60 中等

严格无未来: rolling/ewm无center; 信号用T及之前; 收益shift(-hold)仅衡量。
用法: PYTHONPATH=. python scripts/research/backtest_vol_pullback.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scripts.screener.tcl_pattern import load_codes, STORE
from scripts.screener.v3_derivative_scorer import V3DerivativeScorer


def compute_vol_indicators(df):
    """波动率+量能指标(无未来)。"""
    df = df.sort_values('date').reset_index(drop=True).copy()
    h, l, c = df['high'], df['low'], df['close']
    # ATR(True Range)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['ATR5'] = tr.rolling(5).mean()
    df['ATR20'] = tr.rolling(20).mean()
    df['atr_ratio'] = df['ATR5'] / df['ATR20']        # <1=波动收敛(蓄力)
    df['vol_ratio'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()  # <1=缩量
    df['MA5'] = c.rolling(5).mean()
    df['MA20'] = c.rolling(20).mean()
    df['hi60'] = df['high'].rolling(60).max()
    df['dh60'] = c / df['hi60'] - 1
    df['f1'], _ = V3DerivativeScorer.compute_derivatives(c.values)   # V3趋势(无未来)
    return df


def vol_pullback_mask(df, atr_max=0.85, vol_max=0.9, f1_min=0.2):
    """波动收敛回踩买点(替代MACD红柱)。"""
    c = df['close']
    return (
        (df['f1'] > f1_min) &                              # 趋势在(温和)
        (df['hi60'] > df['MA20'] * 1.05) &                 # 曾突破(60日高>MA20+5%)
        (c / df['MA20'] >= 0.97) & (c / df['MA20'] <= 1.04) &  # 回踩MA20附近
        (df['atr_ratio'] < atr_max) &                      # 波动收敛(蓄力核心)
        (df['vol_ratio'] < vol_max) &                      # 缩量
        (df['dh60'] >= -0.25) & (df['dh60'] <= -0.05)      # 均价不高
    )


def metrics(s):
    s = s.dropna()
    if len(s) < 5:
        return None
    w, l = s[s > 0], s[s < 0]
    pf = w.mean() / abs(l.mean()) if len(l) and l.mean() != 0 else 0
    sh = s.mean() / s.std() * np.sqrt(252 / 10) if s.std() > 0 else 0
    return {'n': len(s), 'mean': s.mean() * 100, 'win': (s > 0).mean() * 100,
            'PF': pf, 'sh': sh, 'med': s.median() * 100}


def main():
    codes = load_codes()
    print(f"波动收敛回踩回测 | {len(codes)}只 | 持有10日 | 严格无未来")
    for label, atr, vol, f1m in [
        ('A. 标准(波动<0.85+缩量<0.9)', 0.85, 0.9, 0.2),
        ('B. 更严(波动<0.75+缩量<0.8)', 0.75, 0.8, 0.2),
        ('C. 仅波动收敛(无量能)', 0.80, 99, 0.2),
        ('D. 仅缩量(无波动)', 99, 0.85, 0.2),
    ]:
        rows = []
        for i, code in enumerate(codes):
            try:
                df = pd.read_parquet(STORE / code / 'D.parquet')
                df['date'] = pd.to_datetime(df['date'])
                df = compute_vol_indicators(df)
                mask = vol_pullback_mask(df, atr, vol, f1m)
                df['sig'] = mask
                df['ret'] = df['close'].shift(-10) / df['close'] - 1
                for _, r in df[df['sig'] & df['ret'].notna()].iterrows():
                    rows.append(r['ret'])
            except Exception:
                pass
        s = pd.Series(rows)
        m = metrics(s)
        if m:
            print(f"\n{label}")
            print(f"  n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win']:5.1f}% "
                  f"PF{m['PF']:.2f} 夏普{m['sh']:.2f} 中位{m['med']:+.2f}%")
    print("\n(对比: TCL原模式 均值+0.66% 胜率47.9% 夏普0.32)")


if __name__ == "__main__":
    main()
