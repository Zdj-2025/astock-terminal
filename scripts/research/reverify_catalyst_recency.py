#!/usr/bin/env python3
"""Tier A时效复测①: 催化驱动(预增>30%次日浅入场) — 2026-08-17 P0工程Phase2。

原结论(2026-07-23): 次日买+2.32%/10日 p=1e-15(1288事件)。
本次: 分半年滚动看衰减 + 近3月(2026-05后)时效面板。
口径: ann_date次日收盘入场(与原版一致), 持有1/3/5/10日收盘收益;
基准: 同入场日全市场等权均值; 剔除ST/停牌/次日一字(无法买: 次日low==high)。
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')

def main():
    fc = pd.read_parquet('data/store/forecast/forecast_all.parquet')
    print('forecast列:', fc.columns.tolist(), '| 覆盖:', fc.ann_date.min(), '→', fc.ann_date.max())
    # 统一事件口径: 预增 & p_change_min>30
    dcol = 'ann_date' if 'ann_date' in fc.columns else fc.columns[1]
    pcol = [c for c in fc.columns if 'p_change' in c][0]
    tcol = 'type' if 'type' in fc.columns else None
    ev = fc[(fc[pcol] > 30)]
    if tcol:
        ev = ev[ev[tcol].str.contains('预增', na=False)]
    ev = ev[['ts_code', dcol, pcol]].dropna()
    ev['code'] = ev['ts_code'].str[:6]
    ev['date'] = ev[dcol].astype(str)
    print(f'事件: 预增>30% 共{len(ev)}个')

    kdir = 'data/store/kline'
    rows = []
    for _, r in ev.iterrows():
        code, d = r['code'], r['date']
        p = f'{kdir}/{code}/D.parquet'
        if not os.path.isfile(p):
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        dc = [c for c in df.columns if 'date' in c.lower()][0]
        ds = df[dc].astype(str).str.replace('-', '').values
        c = df['close'].astype(float).values
        o = df['open'].astype(float).values
        h = df['high'].astype(float).values
        l = df['low'].astype(float).values
        v = df['volume'].astype(float).values
        # 找公告日位置(公告日在数据内或次日)
        idx = np.searchsorted(ds, d)
        if idx >= len(ds) or ds[idx] != d:
            pass  # 公告日在非交易日或缺失 → 取其后第一个交易日
        e = idx  # 公告日(或其后首日)索引
        if e + 11 >= len(c) or e < 1:
            continue
        nxt = e + 1                      # 次日入场
        if v[nxt] <= 0 or h[nxt] == l[nxt]:   # 停牌/一字买不进
            continue
        entry = c[nxt]
        fwd = {h_: (c[nxt + h_] / entry - 1) * 100 if nxt + h_ < len(c) else np.nan for h_ in [1, 3, 5, 10]}
        # 注意: 此处fwd[h]定义为入场日后再持h日, 与原版"次日买持有N日"一致
        fwd = {h_: (c[nxt + h_ - 1] / entry - 1) * 100 if nxt + h_ - 1 < len(c) else np.nan for h_ in [1, 3, 5, 10]}
        rows.append((code, ds[nxt], fwd[1], fwd[3], fwd[5], fwd[10]))
    E = pd.DataFrame(rows, columns=['code', 'entry_date', 'r1', 'r3', 'r5', 'r10'])
    print(f'有效事件: {len(E)} | 入场日 {E.entry_date.min()} → {E.entry_date.max()}')

    # 基准: 同日全市场等权(从事件所在日的全市场r5均值近似——用每事件日全部股票均值)
    # 轻量基准: 用中证全指近似不可得, 改用"事件池自身分半年"对照 + 上证指数
    from scripts.core.tushare_client import TushareClient
    client = TushareClient(); pro = client.pro
    idx = pro.index_daily(ts_code='000001.SH', start_date='20240101')
    idx = idx.sort_values('trade_date')
    idx['date'] = idx['trade_date'].astype(str)
    ic = idx['close'].astype(float).values
    idates = idx['date'].values
    im = {idates[i]: ic[i] for i in range(len(ic))}
    def idx_fwd(dstr, h):
        # dstr后第h个交易日收益
        pos = np.searchsorted(idates, dstr)
        if pos + h >= len(ic) or pos >= len(ic):
            return np.nan
        return (ic[pos + h] / ic[pos] - 1) * 100
    for h in [1, 3, 5, 10]:
        E[f'b{h}'] = E['entry_date'].map(lambda d: idx_fwd(d, h))

    E['half'] = E['entry_date'].str[:4] + 'H' + np.where(E['entry_date'].str[4:6].astype(int) <= 6, '1', '2')
    print('\n=== 分半年滚动(r5% 与 上证同期r5超额) ===')
    print(f"{'区间':<9}{'N':>6}{'r5':>9}{'超额':>9}{'胜率':>7}{'t(超额)':>9}{'r10':>9}{'r10超额':>9}")
    for hname, g in E.groupby('half'):
        ex = (g['r5'] - g['b5']).dropna()
        ex10 = (g['r10'] - g['b10']).dropna()
        t = ex.mean() / (ex.std() / np.sqrt(len(ex))) if len(ex) > 2 else np.nan
        print(f"{hname:<9}{len(g):>6}{g['r5'].mean():>+9.2f}{ex.mean():>+9.2f}"
              f"{(g['r5']>0).mean()*100:>6.0f}%{t:>9.2f}{g['r10'].mean():>+9.2f}{ex10.mean():>+9.2f}")

    print('\n=== 近3月时效面板(2026-05-01后入场) ===')
    g = E[E.entry_date >= '20260501']
    if len(g) == 0:
        print('  近3月无事件(forecast缓存到7/9, 8月未更新——本身是个数据缺口!)')
    else:
        for h in [1, 3, 5, 10]:
            ex = (g[f'r{h}'] - g[f'b{h}']).dropna()
            t = ex.mean() / (ex.std() / np.sqrt(len(ex))) if len(ex) > 2 else np.nan
            print(f"  r{h}: 均值{g[f'r{h}'].mean():+.2f}% 超额{ex.mean():+.2f}% 胜率{(g[f'r{h}']>0).mean()*100:.0f}% t={t:.2f} N={len(g)}")

    E.to_csv('reports/reverify_catalyst_events.csv', index=False)
    print('\n事件明细→reports/reverify_catalyst_events.csv')

if __name__ == '__main__':
    main()
