#!/usr/bin/env python3
"""周频风格轮动检验 — 2026-10-01立(P2-8)。

日频跷跷板已证伪(半导体→次日创新药p=0.5);但防御极×科技极60日相关-0.57在加深,
机构调仓以周为单位——周频上风格跷跷板是否可交易?

三假设(科技=512480半导体, 防御=512890红利低波, 备选防御=159611电力):
  H1 周频动量: 本周风格差(科技-防御)跑赢→下周延续?
  H2 周频反转: 本周跑赢→下周反转?
  H3 价差极值回归: 风格比值z-score极端→未来2/4周回归?
口径: W-FRI周收益;滞后1周(周五收盘信号→下周一可执行近似);成本0.1%/切换;
分期稳定性(2024/2025/2026);净化=剔除|风格差|<1%的噪声周。
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import requests
from scipy import stats

def load(code, n=640):
    sym = ('sh' if code[0]=='5' else 'sz')+code
    d = requests.get(f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{n},qfq',
                     headers={'User-Agent':'Mozilla/5.0'}, timeout=20).json()['data'][sym]
    rows = d.get('qfqday') or d['day']
    return pd.Series([float(r[2]) for r in rows], index=pd.to_datetime([r[0] for r in rows]))

def weekly_test(tech_c, def_c, names, cost=0.001):
    tech = load(tech_c); df = load(def_c)
    px = pd.concat([tech, df], axis=1, keys=['T','D']).dropna()
    wk = px.resample('W-FRI').last().pct_change().dropna()
    spread = wk['T'] - wk['D']          # 风格差周收益
    ratio = (px['T']/px['D']).resample('W-FRI').last()
    z = (ratio - ratio.rolling(12).mean()) / ratio.rolling(12).std()
    print(f'\n═══ {names} | {wk.index[0].date()}→{wk.index[-1].date()} n={len(wk)}周 ═══')

    # H1/H2: 滞后自相关
    r_all = stats.pearsonr(spread.iloc[:-1], spread.shift(-1).dropna())
    print(f'H1/H2 滞后1周自相关: r={r_all[0]:+.3f} (p={r_all[1]:.3f}) '
          f'{"→动量" if r_all[0]>0.15 else "→反转" if r_all[0]<-0.15 else "→无"}')

    # 三分位分析(净:剔除|spread|<1%周)
    m = spread.abs() > 0.01
    sp = spread[m]; nxt = spread.shift(-1)[m]
    q1 = sp <= sp.quantile(1/3); q3 = sp >= sp.quantile(2/3)
    print(f'净化后n={m.sum()}: 本周科技大胜(上1/3)→下周风格差均值{next_mean if False else nxt[q3].mean():+.2%} | '
          f'本周科技大败(下1/3)→下周{next_mean if False else nxt[q1].mean():+.2%}')

    # H3: 价差z极值回归
    for thr in (1.5, -1.5):
        hit = z.shift(1) > thr if thr > 0 else z.shift(1) < thr
        fwd2 = spread.rolling(2).sum().shift(-2)
        sel = fwd2[hit.fillna(False)].dropna()
        if len(sel) >= 8:
            t, p = stats.ttest_1samp(sel, 0)
            print(f'H3 z{"＞" if thr>0 else "＜"}{abs(thr)}(科技{"相对高估" if thr>0 else "相对低估"})后2周风格差: '
                  f'n={len(sel)} 均值{sel.mean():+.2%} p={p:.3f}')

    # 分期自相关稳定性
    for lab, seg in [('2024', wk[wk.index.year==2024]), ('2025', wk[wk.index.year==2025]),
                     ('2026', wk[wk.index.year==2026])]:
        if len(seg) < 15: continue
        s = spread.reindex(seg.index).dropna()
        if len(s) > 15:
            r = stats.pearsonr(s.iloc[:-1], s.shift(-1).dropna())
            print(f'  分期{lab}: n={len(s)} 滞后自相关{r[0]:+.3f}(p={r[1]:.2f})')

    # 可交易模拟: 周五信号下周全仓一侧(动量版/反转版) vs 50/50
    sig_mom = (spread.shift(1) > 0)     # 上周科技赢→下周持科技
    sig_rev = ~sig_mom.fillna(True)
    r5050 = 0.5*wk['T'] + 0.5*wk['D']
    out = {'50/50静态': r5050}
    for nm, sig in [('动量轮动', sig_mom), ('反转轮动', sig_rev)]:
        pos = sig.fillna(True).astype(int)  # 1=科技 0=防御
        r = np.where(pos.values, wk['T'].values, wk['D'].values)
        r = pd.Series(r, index=wk.index)
        switch = pos.diff().abs().fillna(0) > 0
        r = r - switch * cost * 2           # 切换双边成本
        out[nm] = r
    print('  可交易模拟(含0.1%×双边切换成本):')
    for k, r in out.items():
        r = r.dropna(); cum=(1+r).prod()-1; vol=r.std()*np.sqrt(52)
        eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
        print(f'    {k:8s} 累计{cum:+7.1%} 年化σ{vol:5.1%} 回撤{dd:6.1%}')
    return r_all[0], r_all[1]

if __name__ == '__main__':
    print('周频风格轮动检验(P2-8)')
    weekly_test('512480', '512890', '半导体×红利低波(主检验)')
    weekly_test('512480', '159611', '半导体×电力(防御代理稳健性)')
    weekly_test('512100', '512890', '中证1000×红利低波(宽基科技极代理)')
