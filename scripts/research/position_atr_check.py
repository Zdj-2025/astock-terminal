#!/usr/bin/env python3
"""海龟增量①② — ATR仓位单位 + 持仓相关性分组 (2026-10-13上线, 维克海龟法则落地)
①: 每只持仓的1ATR冲击 vs 账户(海龟单位=账户1%/ATR) — 看偏离与总风险
②: 90日收益相关性>0.5的分组 — 高相关组共享风险上限管理
用法: PYTHONPATH=. python scripts/research/position_atr_check.py
"""
import sys; sys.path.insert(0, '.')
import urllib.request, json, pandas as pd

def run(eq=506134):
    POS = [
        ('sz000100','TCL科技',4000), ('sz002250','联化科技',2200),
        ('sh600522','中天科技',500), ('sh688122','西部超导',400),
        ('sh601888','中国中免',400), ('sh601006','大秦铁路',4000),
        ('sh688615','合合信息',200),
        ('sh512480','半导体ETF',30000), ('sh512100','1000ETF',30000),
        ('sz159841','证券ETF',60000), ('sh515120','创新药ETF',50000),
    ]
    import pandas as pd
    df_p = pd.read_excel('config/portfolio.xlsx')
    R1 = eq * 0.01
    rets, rows = {}, []
    for code, name, n in POS:
        try:
            i = df_p.index[df_p['code']==int(code[2:])][0]
            n = int(df_p.loc[i,'shares'])
        except Exception: pass
        url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,2026-04-01,,140,qfq'
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'}), timeout=15))
        r = d['data'][code].get('qfqday') or d['data'][code].get('day')
        df = pd.DataFrame([x[:6] for x in r], columns=['date','open','close','high','low','vol'])
        for c in ['open','close','high','low']: df[c] = df[c].astype(float)
        pc = df['close'].shift(1)
        tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
        atr = tr.rolling(20).mean().iloc[-1]
        unit = int(R1/atr)
        risk = n*atr/eq*100
        rows.append((name, atr, unit, n, n/unit, risk))
        rets[name] = df.set_index('date')['close'].pct_change().iloc[-100:]
    print(f'{"名称":<9}{"ATR20":>7}{"标准股":>8}{"实际":>7}{"偏离":>6}{"1ATR冲击":>8}')
    for r_ in rows: print(f'{r_[0]:<9}{r_[1]:>7.3f}{r_[2]:>8}{r_[3]:>7}{r_[4]:>5.1f}x{r_[5]:>7.1f}%')
    print(f'总1ATR冲击: {sum(r[5] for r in rows):.1f}% (海龟2ATR口径上限15-20%=>1ATR 7.5-10%)')
    R = pd.DataFrame(rets).corr()
    pairs = [(R.columns[i], R.columns[j], R.iloc[i,j]) for i in range(len(R.columns)) for j in range(i+1,len(R.columns)) if abs(R.iloc[i,j])>0.5]
    print('高相关组(|ρ|>0.5):')
    for a,b,rho in sorted(pairs, key=lambda x:-abs(x[2])): print(f'  {a}×{b}: {rho:+.2f}')

if __name__ == '__main__':
    run()
