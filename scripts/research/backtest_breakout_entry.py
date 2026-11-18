#!/usr/bin/env python3
"""突破族入场框架回测 — 浅入场 vs 等回踩MA5 (2026-09-11立).

背景: 9/11族计分卡发现——突破族纸面alpha强(招商南油+16.5%)但"等回踩"100%踏空.
回测问题: 突破信号(平台+破20日高+非高位)后,
  A 浅入场: 次日开盘直接买
  B 等回踩: 10日内触及MA5则以MA5价成交, 否则踏空
哪个净期望好? 踏空率多高? 踏空的代价多大(踏空组的事后表现)?

判定(R1): 若B的成交组超额>A且踏空组事后超额>0 → 框架矛盾(成交的都是弱的,
  跑掉的都是强的) → 突破族改浅入场; 若A>B则维持等回踩.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import pandas as pd
import numpy as np
from scipy import stats as st

START, END = '20240101', '20260820'
COST = 0.0013


def main():
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    ic = idx.set_index('d')['close'].astype(float)
    dates = idx['d'].tolist()
    dpos = {d: i for i, d in enumerate(dates)}

    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])
    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))

    a10, a20, b10, b20, missed21 = [], [], [], [], []
    n_sig = n_miss = 0
    for code in codes:
        try:
            df = pd.read_parquet(f'data/store/kline/{code}/D.parquet',
                                 columns=['date', 'close', 'low', 'open'])
        except Exception:
            continue
        for col in ['datetime', 'day', 'trade_date']:
            pass
        if 'date' not in df.columns:
            for c2 in ['datetime', 'day', 'trade_date']:
                if c2 in df.columns:
                    df = df.rename(columns={c2: 'date'})
        df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        df = df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)
        if len(df) < 200 or 'ST' in nm.get(code, '') or float(df['close'].iloc[-1]) < 2:
            continue
        c = df['close'].astype(float)
        prev20max = c.shift(1).rolling(20).max()
        prev20min = c.shift(1).rolling(20).min()
        hi120 = c.shift(1).rolling(120).max()
        ma5 = c.rolling(5).mean()
        sig = (prev20max / prev20min - 1 < 0.15) & (c > prev20max) & (c < 0.95 * hi120)
        sig_dates = df['d'][sig.fillna(False)]
        if sig_dates.empty:
            continue
        # 30日去重
        keep, last_i = [], -99
        for i in df.index[sig.fillna(False)]:
            if i - last_i >= 30:
                keep.append(i)
                last_i = i
        lows = df['low'].astype(float).values
        opens = df['open'].astype(float).values
        cl = c.values
        ma5v = ma5.values
        for t in keep:
            if t + 21 >= len(df) or not (START <= df.d.iloc[t] <= END):
                continue
            ed = df.d.iloc[t + 1]
            if ed not in dpos:
                continue
            # A 浅入场: t+1开盘
            ea = opens[t + 1]
            pc = cl[t]
            lim = 0.195 if code[0] in ('3', '68') else 0.095
            if ea / pc - 1 > lim or ea <= 0:
                continue
            n_sig += 1
            ii = dpos[ed]
            for H, store in ((10, a10), (20, a20)):
                px = cl[min(t + 1 + H, len(df) - 1)]
                jj = min(ii + H, len(dates) - 1)
                store.append((px / ea - 1 - COST - (ic[dates[jj]] / ic[ed] - 1)) * 100)
            # B 等回踩: 10日内low<=MA5 → 以MA5成交
            fill = None
            for k in range(t + 1, min(t + 11, len(df))):
                if lows[k] <= ma5v[k] and ma5v[k] > 0:
                    fill = (k, ma5v[k])
                    break
            if fill is None:
                n_miss += 1
                px = cl[min(t + 21, len(df) - 1)]
                jj = min(ii + 20, len(dates) - 1)
                missed21.append((px / ea - 1 - COST - (ic[dates[jj]] / ic[ed] - 1)) * 100)
            else:
                k, eb = fill
                fdate = df.d.iloc[k]
                if fdate in dpos:
                    fii = dpos[fdate]
                    for H, store in ((10, b10), (20, b20)):
                        px = cl[min(k + H, len(df) - 1)]
                        jj = min(fii + H, len(dates) - 1)
                        store.append((px / eb - 1 - COST - (ic[dates[jj]] / ic[fdate] - 1)) * 100)

    def rpt(label, xs):
        if not xs:
            print(f'{label:<24} N=0'); return
        x = np.array(xs)
        print(f'{label:<24} N={len(x):>4} | 10d均{np.mean(x[:0]) if False else "":s}', end='')
        print()
        return x

    for label, h10, h20 in (('A 浅入场(次日开盘)', a10, a20), ('B 等回踩MA5(10日内成交)', b10, b20), ('踏空组(事后20d)', None, missed21)):
        if h10 is None:
            x20 = np.array(h20)
            print(f'{label:<26} N={len(x20):>4} | 20d超额均{x20.mean():+.2f}%/中位{np.median(x20):+.2f}%/胜{(x20>0).mean()*100:.0f}%')
        else:
            x10, x20 = np.array(h10), np.array(h20)
            print(f'{label:<26} N={len(x20):>4} | 10d超额均{x10.mean():+.2f}%/胜{(x10>0).mean()*100:.0f}% | 20d超额均{x20.mean():+.2f}%/胜{(x20>0).mean()*100:.0f}%')
    print(f'信号总数{n_sig} | 等回踩踏空率 {n_miss/max(n_sig,1)*100:.0f}% (N={n_miss})')
    if len(a20) >= 10 and len(b20) >= 10:
        t, p = st.ttest_ind(a20, b20, equal_var=False)
        print(f'A vs B 20d超额差 {np.mean(a20)-np.mean(b20):+.2f}% (t={t:.2f}, p={p:.4f})')
    if len(missed21) >= 10 and len(b20) >= 10:
        t, p = st.ttest_ind(missed21, b20, equal_var=False)
        print(f'踏空组 vs 成交组 20d差 {np.mean(missed21)-np.mean(b20):+.2f}% (t={t:.2f}, p={p:.4f})')


if __name__ == '__main__':
    main()
