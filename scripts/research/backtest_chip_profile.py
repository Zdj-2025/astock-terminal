#!/usr/bin/env python3
"""筹码分布因子回测(2026-09-10, R1验证门槛: 先p值后信号化).

因子(全部因果计算,窗口[T-89,T]):
  winner90: 90日成交额中价格下方占比(获利盘代理)
  va_pos:   收盘价相对Value Area位置(<VAL / VA内 / >VAH)
前瞻: T+20交易日收益. 对照: 同期全样本等权均值(剔市场漂移).
样本: 随机800只(种子42, 剔ETF), 月末快照 2025-07~2026-09.
已知局限: 无退市股(幸存者偏差,方向已知:高估底部策略);绝对收益含市场beta.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import numpy as np
import pandas as pd

STORE = PROJECT_ROOT / 'data/store/kline'
rng = np.random.default_rng(42)

codes = sorted([d.name for d in STORE.iterdir()
                if (d / 'D.parquet').exists() and d.name.isdigit() and len(d.name) == 6
                and d.name[:2] not in ('51', '15', '56', '58')])
sample = list(rng.choice(codes, size=min(800, len(codes)), replace=False))
print(f'样本: {len(sample)}只 / 库存{len(codes)}只')

idx = pd.read_parquet('data/store/index/sh000001_long.parquet')
idx['d'] = pd.to_datetime(idx['d'])
idx['ym'] = idx['d'].dt.strftime('%Y-%m')
snap_dates = idx.groupby('ym')['d'].max().tail(15)
snap_str = [d.strftime('%Y%m%d') for d in snap_dates]

rows = []
for code in sample:
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet')
        for col in ('datetime', 'day', 'trade_date'):
            if col in df.columns and col != 'date':
                df = df.rename(columns={col: 'date'})
        df['date'] = df['date'].astype(str).str.replace('-', '').str[:8]
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 160:
            continue
        c = df['close'].astype(float).values
        amt = df['amount'].astype(float).values if 'amount' in df.columns else df['volume'].astype(float).values
        dates = df['date'].values
        for T in snap_str:
            i = np.searchsorted(dates, T, side='right') - 1
            if i < 90 or i + 20 >= len(c):
                continue
            w0 = max(0, i - 89)
            lo = c[w0:i + 1].min(); hi = c[w0:i + 1].max()
            if hi <= lo:
                continue
            bins = np.linspace(lo, hi, 41)
            mid = (bins[:-1] + bins[1:]) / 2
            w = amt[w0:i + 1]
            bidx = np.clip(np.digitize(c[w0:i + 1], bins) - 1, 0, 39)
            vol = np.bincount(bidx, weights=w, minlength=40)
            vol = vol / vol.sum()
            px = c[i]
            winner = float(vol[mid <= px].sum())
            order = np.argsort(-vol)
            lo_i = hi_i = order[0]; cum = vol[order[0]]
            while cum < 0.70 and (lo_i > 0 or hi_i < 39):
                dn = vol[lo_i - 1] if lo_i > 0 else -1
                up = vol[hi_i + 1] if hi_i < 39 else -1
                if up >= dn: hi_i += 1; cum += up
                else: lo_i -= 1; cum += dn
            val, vah = bins[lo_i], bins[hi_i + 1]
            fwd = c[i + 20] / px - 1
            rows.append({'code': code, 'date': T, 'winner': winner,
                         'va_pos': 'below' if px < val else ('above' if px > vah else 'in'),
                         'fwd20': fwd * 100})
    except Exception:
        continue

t = pd.DataFrame(rows)
t.to_parquet('data/store/chip_factor_panel.parquet', index=False)
print(f'面板: {len(t)}条 (股×快照)\n')

t['adj20'] = t['fwd20'] - t.groupby('date')['fwd20'].transform('mean')

print('=== 分组: winner90五分位 → 20日超额收益(剔市场) ===')
t['q'] = pd.qcut(t['winner'], 5, labels=['Q1极低', 'Q2', 'Q3', 'Q4', 'Q5极高'])
g = t.groupby('q', observed=True)['adj20'].agg(['count', 'mean', 'median'])
g['t'] = t.groupby('q', observed=True)['adj20'].apply(lambda x: x.mean() / (x.std() / np.sqrt(len(x))))
print(g.round(2).to_string())

print('\n=== 极端事件 ===')
for name, cond in [('winner<15%(割肉盘出尽)', t['winner'] < 0.15),
                   ('winner>85%(兑现压力)', t['winner'] > 0.85),
                   ('价格<VAL(拒绝价值区)', t['va_pos'] == 'below'),
                   ('价格>VAH(接受上方)', t['va_pos'] == 'above')]:
    sub = t[cond]
    if len(sub) < 30:
        print(f'{name}: n={len(sub)} 样本不足'); continue
    tt = sub['adj20'].mean() / (sub['adj20'].std() / np.sqrt(len(sub)))
    win = (sub['fwd20'] > 0).mean() * 100
    print(f'{name}: n={len(sub)} 20日超额均值{sub["adj20"].mean():+.2f}% (t={tt:+.1f}) 胜率{win:.0f}%')

from scipy.stats import spearmanr
ics = t.groupby('date').apply(lambda x: spearmanr(x['winner'], x['fwd20'])[0], include_groups=False)
print(f'\nIC(winner vs fwd20, RankIC按期): 均值{ics.mean():+.4f} t={ics.mean()/ics.std()*np.sqrt(len(ics)):+.1f}')
