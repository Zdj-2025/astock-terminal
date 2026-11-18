#!/usr/bin/env python3
"""ETF持仓分析补全 - 2025/05/20"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from scripts.core.tushare_client import TushareClient

client = TushareClient()
pro = client.pro

TOTAL_CAPITAL = 487686

etfs = [
    {'name': '半导体ETF', 'code': '512480.SH', 'shares': 20000, 'cost': 1.589, 'price': 2.227},
    {'name': '芯片ETF', 'code': '512760.SH', 'shares': 36000, 'cost': 0.297, 'price': 1.201},
    {'name': 'AI智能ETF', 'code': '159819.SZ', 'shares': 10000, 'cost': 0.210, 'price': 2.002},
]

def calc_ma(arr, period):
    if len(arr) < period:
        return None
    return float(np.mean(arr[-period:]))

for etf in etfs:
    print(f"\n{'='*60}")
    print(f"分析: {etf['name']} ({etf['code']})")
    print(f"{'='*60}")
    
    # 获取日线
    df = pro.fund_daily(ts_code=etf['code'], start_date='20250101', end_date='20260520')
    if df is None or len(df) == 0:
        print("  ⚠️ 无数据")
        continue
    df = df.sort_values('trade_date').reset_index(drop=True)
    print(f"  日线: {len(df)}条, {df.iloc[0]['trade_date']} ~ {df.iloc[-1]['trade_date']}")
    
    close = df['close'].values.astype(float)
    current = etf['price']
    
    # 均线
    for p in [5, 10, 20, 60, 120, 250]:
        ma = calc_ma(close, p)
        if ma:
            diff = (current - ma) / ma * 100
            print(f"  MA{p}: {ma:.3f} (偏离{diff:+.1f}%)")
    
    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100
    print(f"  RSI(14): {rsi:.1f}")
    
    # MACD
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12).mean()
    ema26 = close_s.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    hist = (dif - dea) * 2
    print(f"  MACD: DIF={dif.iloc[-1]:.4f}, DEA={dea.iloc[-1]:.4f}, 柱={hist.iloc[-1]:.4f}")
    
    prev_h = hist.iloc[-2] if len(hist) > 1 else 0
    h_last = hist.iloc[-1]
    if h_last > 0 and prev_h <= 0: mt = '金叉'
    elif h_last < 0 and prev_h >= 0: mt = '死叉'
    elif h_last > prev_h > 0: mt = '红柱扩大'
    elif 0 < h_last < prev_h: mt = '红柱缩小'
    elif h_last < prev_h < 0: mt = '绿柱扩大'
    else: mt = '绿柱缩小'
    print(f"  MACD趋势: {mt}")
    
    # 周线
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df2 = df.set_index('trade_date')
    weekly = df2.resample('W').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum'}).dropna()
    if len(weekly) > 20:
        wc = weekly['close'].values
        w5 = calc_ma(wc, 5)
        w10 = calc_ma(wc, 10)
        w20 = calc_ma(wc, 20)
        last_c = wc[-1]
        if w5 and w10 and w20:
            if last_c > w5 > w10 > w20: wt = '多头排列'
            elif last_c > w5 > w10: wt = '偏多'
            elif last_c < w5 < w10: wt = '偏空'
            elif last_c < w5 < w10 < w20: wt = '空头排列'
            else: wt = '震荡'
            print(f"  周线趋势: {wt}")
            print(f"  周线MA5={w5:.3f} MA10={w10:.3f} MA20={w20:.3f}")
    
    # 量比
    vol = df2['vol'].values if 'vol' in df2.columns else None
    if vol is not None:
        vol = vol.astype(float)
        vma5 = calc_ma(vol, 5)
        vma20 = calc_ma(vol, 20)
        vr = vol[-1] / vma5 if vma5 and vma5 > 0 else 0
        print(f"  量比(vs MA5): {vr:.2f}")
    
    # 涨跌幅
    if len(close) > 5:
        c5 = (close[-1] - close[-6]) / close[-6] * 100
        print(f"  近5日涨跌: {c5:+.2f}%", end='')
    if len(close) > 20:
        c20 = (close[-1] - close[-21]) / close[-21] * 100
        print(f"  近20日涨跌: {c20:+.2f}%")
    else:
        print()
    
    # 仓位分析
    mv = etf['shares'] * etf['price']
    pct = mv / TOTAL_CAPITAL * 100
    pnl_pct = (etf['price'] - etf['cost']) / etf['cost'] * 100
    pnl_amt = (etf['price'] - etf['cost']) * etf['shares']
    print(f"  市值: {mv:,.0f} 占比: {pct:.1f}% 浮盈: {pnl_amt:+,.0f}({pnl_pct:+.1f}%)")

print("\n✅ ETF分析完成")
