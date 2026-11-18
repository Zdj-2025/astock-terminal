#!/usr/bin/env python3
"""前低支撑位计算：510500/512100 前复权，找近期谷低/30日最低/YTD最低，给再评估线。
跌破前低+放量 = 趋势恶化再评估（不是机械砍，宽基仍结合背离+政策底判断）。"""
import pandas as pd, numpy as np
from scipy.signal import argrelextrema

for code in ['510500', '512100']:
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet').sort_values('date').reset_index(drop=True)
    close = df['close'].values.astype(float).copy()
    low = df['low'].values.astype(float).copy()
    rets = df['close'].pct_change().values
    for i in range(len(close) - 2, -1, -1):          # qfq：除权日及之前 ×比例
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            r = close[i + 1] / close[i]
            close[:i + 1] *= r
            low[:i + 1] *= r

    last_close = float(close[-1])
    last_date = df['date'].iloc[-1]
    print(f"\n=== {code} | 最新收 {last_close:.3f} ({last_date}) ===")

    look = 90
    rl = low[-look:]
    tr = argrelextrema(rl, np.less, order=5)[0]
    if len(tr):
        print("  近期谷低(argrelextrema, 近90日):")
        for t in tr[-3:]:
            di = df['date'].iloc[-(look - t)]
            print(f"    {di}  低 {rl[t]:.3f}  (现价距 {(last_close / rl[t] - 1) * 100:+.1f}%)")
    min30 = float(low[-30:].min())
    yr = str(last_date)[:4]
    ytd_mask = df['date'].astype(str).str.startswith(yr)
    ytd_low = float(low[ytd_mask.values].min()) if ytd_mask.any() else np.nan
    print(f"  近30日最低: {min30:.3f}  (现价距 {(last_close / min30 - 1) * 100:+.1f}%)")
    print(f"  {yr}年初至今最低: {ytd_low:.3f}  (现价距 {(last_close / ytd_low - 1) * 100:+.1f}%)")
    # 当前下行腿中是否已缩量（最近5日 vol vs 之前20日）
    vol_col = 'vol' if 'vol' in df.columns else 'volume'
    vol = df[vol_col].values.astype(float)
    v_recent = vol[-5:].mean()
    v_prior = vol[-25:-5].mean()
    print(f"  量能: 近5日均量 {v_recent:.0f} vs 前20日均量 {v_prior:.0f} ({v_recent/v_prior:.2f}x)")
