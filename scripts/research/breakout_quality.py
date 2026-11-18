#!/usr/bin/env python3
"""突破股性验证 — 对当前突破股, 算其历史突破信号的ret10表现(股性)。

逻辑: 有的股突破后常涨(股性好), 有的常假突破(股性差)。用历史突破胜率筛真假。
这是AI验证维度之一(散户只看当下一根突破K线, 看不出股性; 我们回算历史)。
数据: 同步量价, 无未来。
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

STORE = PROJECT_ROOT / 'data/store/kline'


def breakout_quality(code):
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet', columns=['date', 'close', 'volume'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 140:
            return None
        close = df['close'].astype(float)
        vol = df['volume'].astype(float)
        p20_hi = close.rolling(20).max().shift(1)
        p20_lo = close.rolling(20).min().shift(1)
        platform = (p20_hi / p20_lo - 1) < 0.15
        brk = close > p20_hi
        vol_ma = vol.rolling(20).mean().shift(1)
        surge = vol > vol_ma * 1.5
        sig = (platform & brk & surge).fillna(False)
        ret10 = (close.shift(-10) / close - 1)
        hits = ret10[sig].dropna()
        if len(hits) < 3:
            return None
        return {'n': len(hits), 'mean': hits.mean() * 100, 'win': (hits > 0).mean() * 100}
    except Exception:
        return None


def main():
    codes = "600876 002907 300321 300320 603072 300109 000795 603358 002145 603087 300402 300224 300449 300239 300860 002191".split()
    print(f"突破股性验证(历史平台+突破+放量后ret10):\n{'代码':<7}{'历史N':>7}{'均值':>8}{'胜率':>7}")
    rows = []
    for code in codes:
        q = breakout_quality(code)
        if q:
            rows.append((code, q))
    rows.sort(key=lambda x: -x[1]['win'])
    for code, q in rows:
        star = '★' if q['win'] >= 55 and q['mean'] > 1 else ('⚠️' if q['win'] < 40 else '')
        print(f"{code:<7}{q['n']:>7}{q['mean']:>+7.1f}%{q['win']:>6.0f}% {star}")
    print("\n★=股性好(历史突破胜率≥55%且均值>1%), ⚠️=股性差(胜率<40%, 常假突破)")


if __name__ == '__main__':
    main()
