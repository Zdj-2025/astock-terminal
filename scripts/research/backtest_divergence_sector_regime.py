#!/usr/bin/env python3
"""背离×防御板块×regime交互回测 — 2026-09-11立.

假设来源: 9/11族计分卡——浙能电力(防御)背离+4.6%超额 vs 松霖(消费)背离-6%超额,
与7/2"大盘坏背离更强"(130万样本)叠加 → 三问:
  ①防御板块的背离在坏regime里是否显著优于非防御背离?
  ②板块分裂本身是否独立于regime成立?
  ③若①②都不显著, 浙能只是n=1运气.

引擎对齐reverify_divergence_recency(2026-08-17 Tier A复测):
  底背离=两DIF谷(间隔≤40日)价新低DIF抬升; 确认=10日内量能阳线; 入场=确认日收盘.
板块: stock_list.industry 含 电力/水务/燃气/银行/保险/高速/铁路/港口 → 防御.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy import stats as st

START, END = '20240101', '20260820'
DEF_KW = ('电力', '水务', '燃气', '银行', '保险', '高速', '铁路', '港口')


def main():
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    ic = idx.set_index('d')['close'].astype(float)
    dates = idx['d'].tolist()
    dpos = {d: i for i, d in enumerate(dates)}
    r20 = ic.pct_change(20).shift(1)

    sl = pd.read_parquet('data/store/stock_list.parquet')
    sym = sl['symbol'].astype(str).str.zfill(6)
    nm = dict(zip(sym, sl['name'].astype(str)))
    ind = dict(zip(sym, sl['industry'].astype(str)))
    is_def = {s: any(k in str(ind.get(s) or '') for k in DEF_KW) for s in sym}

    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])

    res = []  # (defensive, regime_bad, ex10, ex20)
    for code in codes:
        if 'ST' in nm.get(code, ''):
            continue
        try:
            df = pd.read_parquet(f'data/store/kline/{code}/D.parquet',
                                 columns=['date', 'close', 'open', 'volume'])
        except Exception:
            continue
        for c2 in ['datetime', 'day', 'trade_date']:
            if c2 in df.columns and 'date' not in df.columns:
                df = df.rename(columns={c2: 'date'})
        if 'date' not in df.columns:
            continue
        df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        df = df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)
        if len(df) < 220 or float(df['close'].iloc[-1]) < 2:
            continue
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        v = df['volume'].astype(float)
        ef = c.ewm(span=12, adjust=False).mean()
        es = c.ewm(span=26, adjust=False).mean()
        dif = ef - es
        hi120 = c.shift(1).rolling(120).max()
        vol5 = v.shift(1).rolling(5).mean()
        dv = dif.values; cv = c.values; ov = o.values; vv = v.values
        hv = hi120.values; v5 = vol5.values
        n = len(df)
        try:
            troughs = argrelextrema(dv, np.less, order=5)[0]
        except Exception:
            continue
        for a_i in range(len(troughs) - 1):
            t1, t2 = troughs[a_i], troughs[a_i + 1]
            if t2 - t1 > 40 or t2 + 11 >= n:
                continue
            if not (cv[t2] < cv[t1] and dv[t2] > dv[t1] and cv[t2] < 0.85 * (hv[t2] if not np.isnan(hv[t2]) else 1e9)):
                continue
            # 确认: t2后10日内量能阳线
            conf = None
            for k in range(t2 + 1, min(t2 + 11, n)):
                if ov[k] < cv[k] and cv[k] > cv[k - 1] and vv[k] > 1.2 * (v5[k] if not np.isnan(v5[k]) else 0):
                    conf = k
                    break
            if conf is None or not (START <= df.d.iloc[conf] <= END):
                continue
            ed = df.d.iloc[conf]
            if ed not in dpos:
                continue
            entry = cv[conf]
            ii = dpos[ed]
            bad = r20.get(ed, 0) < -0.03
            row_ok = True
            vals = {}
            for H in (10, 20):
                ci = min(conf + H, n - 1)
                jj = min(ii + H, len(dates) - 1)
                vals[H] = (cv[ci] / entry - 1 - 0.0013 - (ic[dates[jj]] / ic[ed] - 1)) * 100
                if np.isnan(vals[H]):
                    row_ok = False
            if row_ok:
                res.append((is_def.get(code, False), bad, vals[10], vals[20]))

    t = pd.DataFrame(res, columns=['defensive', 'bad', 'ex10', 'ex20'])
    if t.empty:
        print('无信号'); return
    print(f'背离+量能确认 × 板块 × regime | 信号N={len(t)} | {START}~{END} | 20d超额(个股-上证)')
    print('═' * 78)
    print(f"{'分组':<24}{'N':>5}{'10d均':>8}{'20d均':>8}{'20d胜率':>8}")
    for label, m in [('防御板块·坏regime', (t.defensive) & (t.bad)),
                     ('防御板块·好/震荡', (t.defensive) & (~t.bad)),
                     ('非防御·坏regime', (~t.defensive) & (t.bad)),
                     ('非防御·好/震荡', (~t.defensive) & (~t.bad))]:
        sub = t[m]
        if sub.empty:
            print(f'{label:<24}{0:>5}'); continue
        print(f"{label:<24}{len(sub):>5}{sub['ex10'].mean():>+7.2f}%{sub['ex20'].mean():>+7.2f}%{(sub['ex20']>0).mean()*100:>7.0f}%")
    # 交互检验: 2×2 ANOVA式 — 防御效应在坏regime内的简单斜率
    d_bad = t[(t.defensive) & (t.bad)]['ex20']
    nd_bad = t[(~t.defensive) & (t.bad)]['ex20']
    d_good = t[(t.defensive) & (~t.bad)]['ex20']
    nd_good = t[(~t.defensive) & (~t.bad)]['ex20']
    if len(d_bad) >= 15 and len(nd_bad) >= 15:
        t1_, p1_ = st.ttest_ind(d_bad, nd_bad, equal_var=False)
        print(f"\n坏regime内 防御vs非防御 20d差 {d_bad.mean()-nd_bad.mean():+.2f}% (t={t1_:.2f}, p={p1_:.4f})")
    if len(d_bad) >= 15 and len(d_good) >= 15:
        t2_, p2_ = st.ttest_ind(d_bad, d_good, equal_var=False)
        print(f"防御板块内 坏vs好 20d差 {d_bad.mean()-d_good.mean():+.2f}% (t={t2_:.2f}, p={p2_:.4f})")


if __name__ == '__main__':
    main()
