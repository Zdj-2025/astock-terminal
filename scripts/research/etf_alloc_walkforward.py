#!/usr/bin/env python3
"""ETF配置walk-forward验证 — 2026-10-01立(P2-7)。

问题: v3报告的ERC锚(四轴14/16/48/22)与512890稀释建议来自全样本优化——有后视嫌疑。
方法: 训练窗252日算ERC权重 → 执行窗21日持有(线性组合近似) → 滚动至样本尾。
对照: WF-ERC四轴 / WF-ERC13只 / 等权四轴 / 当前袖珍实际权重 / 沪深300。
通过标准: WF-ERC的σ/回撤显著低于当前袖珍,且权重跨折稳定(锚可用)。

用法: PYTHONPATH=. python scripts/research/etf_alloc_walkforward.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import numpy as np
import pandas as pd
import requests

NAMES = {'512480':'半导体','515120':'创新药','512100':'中证1000','159841':'证券',
         '562500':'机器人','159611':'电力','518880':'黄金','510300':'沪深300','512690':'酒',
         '512890':'红利低波','159928':'消费','512800':'银行','510880':'红利'}
AXIS = ['512480','515120','512890','159611']
CUR = {'512480':0.11,'512100':0.42,'159841':0.32,'515120':0.15}  # 当前袖珍实际权重
N_BARS = 640

def load():
    closes = {}
    for code in NAMES:
        sym = ('sh' if code[0]=='5' else 'sz')+code
        d = requests.get(f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{N_BARS},qfq',
                         headers={'User-Agent':'Mozilla/5.0'}, timeout=20).json()['data'][sym]
        rows = d.get('qfqday') or d['day']
        closes[code] = pd.Series([float(r[2]) for r in rows],
                                 index=pd.to_datetime([r[0] for r in rows]))
    return pd.DataFrame(closes).sort_index()

def erc_fp(S, iters=20000, tol=1e-12):
    n = S.shape[0]; w = np.ones(n)/n
    for _ in range(iters):
        mrc = S@w
        w_new = np.sqrt(w/mrc); w_new /= w_new.sum()
        if np.max(np.abs(w_new-w)) < tol: w = w_new; break
        w = w_new
    return w

def main():
    px = load()
    ret = px.pct_change().dropna()
    print(f'样本: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}日)\n')
    cols13 = list(NAMES)
    R13 = ret[cols13].values
    i_axis = [cols13.index(c) for c in AXIS]

    folds = []
    start = 252
    while start + 21 <= len(ret):
        tr = ret.iloc[start-252:start]
        # 四轴ERC(训练期协方差,样本/LW)
        Sa = tr[AXIS].cov().values
        wa = erc_fp(Sa)
        # 13只ERC
        S13 = tr[cols13].cov().values
        w13 = erc_fp(S13)
        folds.append((ret.index[start], wa, w13))
        start += 21
    print(f'walk-forward折数: {len(folds)} (训练252日/执行21日月滚)\n')

    # 执行期收益(线性组合近似=日再平衡)
    res = {k: [] for k in ['WF-ERC四轴','WF-ERC13','等权四轴','当前袖珍','沪深300']}
    w_track = []
    for i, (dt, wa, w13) in enumerate(folds):
        seg = ret.iloc[252+i*21 : 252+i*21+21]
        if len(seg) == 0: break
        ra = seg[AXIS].values
        r13 = seg[cols13].values
        res['WF-ERC四轴'].extend(ra@wa)
        res['WF-ERC13'].extend(r13@w13)
        res['等权四轴'].extend(ra@(np.ones(4)/4))
        res['当前袖珍'].extend(r13@np.array([CUR.get(c,0) for c in cols13]))
        res['沪深300'].extend(seg['510300'].values)
        w_track.append((dt, *wa))

    print(f'{"组合":12s} {"累计":>8s} {"年化σ":>7s} {"夏普":>6s} {"最大回撤":>8s}')
    eq_best = None
    for k, v in res.items():
        v = np.array(v); cum = (1+v).prod()-1
        vol = v.std()*np.sqrt(252); shp = v.mean()*252/vol if vol>0 else 0
        eq = np.cumprod(1+v); dd = (eq/np.maximum.accumulate(eq)-1).min()
        print(f'{k:12s} {cum:+8.1%} {vol:7.1%} {shp:6.2f} {dd:8.1%}')
        if k=='WF-ERC四轴': eq_best = eq

    # 权重跨折稳定性
    wt = pd.DataFrame(w_track, columns=['date','半导体','创新药','红利低波','电力']).set_index('date')
    print('\n四轴ERC权重跨折演变(每3折采样):')
    print(wt.iloc[::3].round(2).to_string())
    print(f'\n权重稳定性: 红利低波 min={wt["红利低波"].min():.2f} max={wt["红利低波"].max():.2f} '
          f'| 半导体 min={wt["半导体"].min():.2f} max={wt["半导体"].max():.2f}')
    # 均值权重的样本外检验: 全期用平均权重跑一遍
    w_avg = wt.mean().values
    r_all = ret.iloc[252:][AXIS].values
    v = r_all@w_avg
    cum=(1+v).prod()-1; vol=v.std()*np.sqrt(252); eq=np.cumprod(1+v); dd=(eq/np.maximum.accumulate(eq)-1).min()
    print(f'\n[后验] 用跨折均值权重{dict(zip(AXIS,w_avg.round(2)))}全期持有: '
          f'累计{cum:+.1%} σ{vol:.1%} 回撤{dd:.1%}')

if __name__ == '__main__':
    main()
