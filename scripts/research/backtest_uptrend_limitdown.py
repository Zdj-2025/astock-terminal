#!/usr/bin/env python3
"""Sequoia-X uptrend_limit_down 策略回测(2026-09-10, 外来算法过R1闸).
原样规格: 昨日MA20>MA60 + 今收<=昨收*0.905 + 今量>20日均量*2.0.
前瞻5/10/20日(信号日收盘入场,保守口径). 分板块检验0.905硬编码问题."""
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
print(f'扫描 {len(codes)}只...')
rows = []
for code in codes:
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet')
        for col in ('datetime','day','trade_date'):
            if col in df.columns and col != 'date': df = df.rename(columns={col:'date'})
        df['date'] = df['date'].astype(str).str.replace('-','').str[:8]
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 90: continue
        c = df['close'].astype(float).values
        v = df['volume'].astype(float).values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        vma20 = pd.Series(v).rolling(20).mean().values
        dates = df['date'].values
        for i in range(60, len(df)-21):
            if dates[i] < '20251101': continue
            if np.isnan(ma60[i-1]) or np.isnan(vma20[i]): continue
            if ma20[i-1] > ma60[i-1] and c[i] <= c[i-1]*0.905 and v[i] > vma20[i]*2.0:
                rows.append({'code':code,'date':dates[i],
                             'board':'20cm' if code[:2] in ('30','68') else '10cm',
                             'f5':(c[min(i+5,len(c)-1)]/c[i]-1)*100,
                             'f10':(c[min(i+10,len(c)-1)]/c[i]-1)*100,
                             'f20':(c[min(i+20,len(c)-1)]/c[i]-1)*100})
    except Exception: continue

t = pd.DataFrame(rows)
print(f'事件数: {len(t)}')
if len(t):
    t.to_parquet(PROJECT_ROOT / 'data/store/limitdown_events.parquet', index=False)
    for h in ['f5','f10','f20']:
        m, s = t[h].mean(), t[h].std()
        tt = m/(s/np.sqrt(len(t)))
        print(f'+{h[1:]}日: 均值{m:+.2f}% (t={tt:+.1f}) 胜率{(t[h]>0).mean()*100:.0f}% 中位{t[h].median():+.2f}%')
    print('\n分板块(检验0.905硬编码问题):')
    for b, sub in t.groupby('board'):
        tt = sub['f10'].mean()/(sub['f10'].std()/np.sqrt(len(sub)))
        print(f'{b}: n={len(sub)} +10日{sub["f10"].mean():+.2f}% (t={tt:+.1f}) 胜率{(sub["f10"]>0).mean()*100:.0f}%')
