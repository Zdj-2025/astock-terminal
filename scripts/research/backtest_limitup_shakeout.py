#!/usr/bin/env python3
"""Sequoia-X limit_up_shakeout 回测(2026-09-10, 外来算法过R1闸).
原样规格: 昨涨停(>=昨昨收*1.095) + 今收阴 + 今量>昨量*2 + 今low>=昨收.
同时测板别感知变体(20cm用1.195). 前瞻5/10/20日."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)
import numpy as np, pandas as pd

STORE = PROJECT_ROOT / 'data/store/kline'
codes = sorted([d.name for d in STORE.iterdir()
                if (d/'D.parquet').exists() and d.name.isdigit() and len(d.name)==6
                and d.name[:2] not in ('51','15','56','58')])
rows = []
for code in codes:
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet')
        for col in ('datetime','day','trade_date'):
            if col in df.columns and col != 'date': df = df.rename(columns={col:'date'})
        df['date'] = df['date'].astype(str).str.replace('-','').str[:8]
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 30: continue
        c = df['close'].astype(float).values
        o = df['open'].astype(float).values
        l = df['low'].astype(float).values
        v = df['volume'].astype(float).values
        dates = df['date'].values
        limit_ratio = 1.195 if code[:2] in ('30','68') else 1.095
        for i in range(2, len(df)-21):
            if dates[i] < '20251101': continue
            if not (c[i-1] >= c[i-2]*1.095): continue
            strict = c[i-1] >= c[i-2]*limit_ratio
            if not (c[i] < o[i] and v[i] > v[i-1]*2.0 and l[i] >= c[i-1]): continue
            rows.append({'code':code,'date':dates[i],
                         'board':'20cm' if code[:2] in ('30','68') else '10cm',
                         'strict_limit':strict,
                         'f5':(c[min(i+5,len(c)-1)]/c[i]-1)*100,
                         'f10':(c[min(i+10,len(c)-1)]/c[i]-1)*100,
                         'f20':(c[min(i+20,len(c)-1)]/c[i]-1)*100})
    except Exception: continue

t = pd.DataFrame(rows)
print(f'事件数: {len(t)}')
if len(t):
    t.to_parquet(PROJECT_ROOT / 'data/store/shakeout_events.parquet', index=False)
    def rpt(name, sub):
        n = len(sub)
        if n < 30: print(f'{name}: n={n}不足'); return
        tt = sub['f10'].mean()/(sub['f10'].std()/np.sqrt(n))
        print(f"{name}: n={n} +5日{sub['f5'].mean():+.2f}%/+10日{sub['f10'].mean():+.2f}%(t={tt:+.1f})/+20日{sub['f20'].mean():+.2f}% 胜率10日{(sub['f10']>0).mean()*100:.0f}%")
    rpt('原样(统一1.095)', t)
    rpt('板别感知真涨停', t[t['strict_limit']])
    rpt('  └10cm', t[(t['strict_limit']) & (t['board']=='10cm')])
    rpt('  └20cm', t[(t['strict_limit']) & (t['board']=='20cm')])
