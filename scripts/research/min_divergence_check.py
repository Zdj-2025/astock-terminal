#!/usr/bin/env python3
"""分钟级顶/底背离检查：现拉 tushare 数据(绕过可能stale的本地parquet)，算MACD背离。
用法: python min_divergence_check.py 002250 60   # code freq(1/5/15/30/60)"""
import sys, pandas as pd, numpy as np
from scipy.signal import argrelextrema
sys.path.insert(0, '.')
from scripts.core.tushare_client import TushareClient

code = sys.argv[1] if len(sys.argv) > 1 else '002250'
freq = sys.argv[2] if len(sys.argv) > 2 else '60'

cli = TushareClient()
df = cli.get_kline(code, freq=freq, start='20260601', end='20260723')
tcol = 'trade_time' if 'trade_time' in df.columns else ('datetime' if 'datetime' in df.columns else df.columns[0])
df = df.sort_values(tcol).reset_index(drop=True).dropna(subset=[tcol])
close = df['close'].values.astype(float)
high = df['high'].values.astype(float) if 'high' in df.columns else close
low = df['low'].values.astype(float) if 'low' in df.columns else close

ef = pd.Series(close).ewm(span=12, adjust=False).mean()
es = pd.Series(close).ewm(span=26, adjust=False).mean()
dif = (ef - es).values
dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
macd = (dif - dea) * 2

print(f"=== {code} {freq}min | 最新收 {close[-1]:.3f} @ {df[tcol].iloc[-1]} | {len(df)}根 ===")
print(f"最新 DIF {dif[-1]:+.4f} DEA {dea[-1]:+.4f} MACD柱 {macd[-1]:+.4f} | DIF近5根斜率 {dif[-1]-dif[-5]:+.4f}")

peaks = argrelextrema(dif, np.greater, order=5)[0]
troughs = argrelextrema(dif, np.less, order=5)[0]
print(f"\nDIF峰(近5):")
for p in peaks[-5:]:
    print(f"  {df[tcol].iloc[p]}  高{high[p]:.3f}  DIF{dif[p]:+.4f}")

if len(peaks) >= 2:
    p1, p2 = peaks[-2], peaks[-1]
    pu = high[p2] > high[p1]
    dd = dif[p2] < dif[p1]
    print(f"\n顶背离(最近两峰): {df[tcol].iloc[p1]} 高{high[p1]:.3f}/DIF{dif[p1]:+.4f}  →  {df[tcol].iloc[p2]} 高{high[p2]:.3f}/DIF{dif[p2]:+.4f}")
    if pu and dd:
        st = (dif[p1] - dif[p2]) / (abs(dif[p1]) + 1e-9)
        chg = (close[-1] / close[p2] - 1) * 100
        print(f"  → ✅顶背离成立 strength={st:.2f} | 后峰后最新价 {chg:+.1f}% ({'已部分兑现' if chg < -1 else '尚未兑现'})")
    else:
        print(f"  → ❌无顶背离 (价↑={pu} DIF↓={dd})")
