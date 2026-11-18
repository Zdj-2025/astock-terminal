#!/usr/bin/env python3
"""Tier A时效复测⑤: divergence底背离+量能阳线确认 — 2026-08-17晚 P0工程Phase2。

对齐divergence_scanner核心逻辑(全因果右侧确认):
  底背离: 两个DIF谷(t1<t2,间隔≤40日), close[t2]<close[t1](价新低) 且 dif[t2]>dif[t1](DIF抬升)
  确认: t2后10日内出现量能阳线(close>open且close>prev_close且vol>vol5×1.2)→事件=确认日
  低位: close<hi120×0.85
  入场: 确认日收盘, 持1/5/10日 vs 上证
原结论(memory strategy_divergence_fix_ema_volconfirm): 右侧确认夏普0.33-0.60 p=0.0016。
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

os.chdir('.')
sys.path.insert(0, os.getcwd())


def main():
    from scripts.core.tushare_client import TushareClient
    pro = TushareClient().pro
    idx = pro.index_daily(ts_code='000001.SH', start_date='20230101').sort_values('trade_date')
    ic = idx['close'].astype(float).values
    idates = idx['trade_date'].astype(str).values

    def idx_f(d, h):
        p = np.searchsorted(idates, d)
        return (ic[p + h] / ic[p] - 1) * 100 if p + h < len(ic) else np.nan

    sl = pd.read_parquet('data/store/stock_list.parquet').set_index('symbol')
    st = {c for c in sl.index if 'ST' in str(sl.loc[c, 'name']).upper()}
    kdir = 'data/store/kline'
    codes = [d for d in os.listdir(kdir) if d.isdigit() and len(d) == 6
             and os.path.isfile(f'{kdir}/{d}/D.parquet') and d not in st]
    rows = []
    for k, code in enumerate(codes):
        try:
            df = pd.read_parquet(f'{kdir}/{code}/D.parquet')
        except Exception:
            continue
        dc = [c for c in df.columns if 'date' in c.lower()]
        if not dc:
            continue
        df = df.sort_values(dc[0])
        if len(df) < 140:
            continue
        ds = df[dc[0]].astype(str).str.replace('-', '').values
        c = df['close'].astype(float).values
        o = df['open'].astype(float).values
        v = df['volume'].astype(float).values
        ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        v5 = pd.Series(v).rolling(5).mean().values
        hi120 = pd.Series(c).rolling(120).max().values
        try:
            tr = argrelextrema(dif, np.less, order=5)[0]
        except Exception:
            continue
        used_t2 = -1
        for i in range(1, len(tr)):
            t1, t2 = int(tr[i - 1]), int(tr[i])
            if t2 <= used_t2 or (t2 - t1) > 40 or t2 < 130:
                continue
            if c[t2] < c[t1] and dif[t2] > dif[t1]:
                # 背离成立,找t2后10日内量能阳线确认
                for t in range(t2 + 1, min(t2 + 11, len(c) - 11)):
                    if (c[t] > o[t] and c[t] > c[t - 1] and v[t] > v5[t] * 1.2
                            and not np.isnan(hi120[t]) and c[t] < hi120[t] * 0.85):
                        d = str(ds[t])
                        if d >= '20240101':
                            fwd = tuple((c[t + h] / c[t] - 1) * 100 for h in (1, 5, 10))
                            rows.append((d, *fwd))
                        used_t2 = t2
                        break
        if (k + 1) % 1500 == 0:
            print(f'  进度{k+1}/{len(codes)} 事件{len(rows)}', flush=True)

    E = pd.DataFrame(rows, columns=['date', 'r1', 'r5', 'r10'])
    E['seg'] = E.date.str[:4] + 'H' + np.where(E.date.str[4:6].astype(int) <= 6, '1', '2')
    print(f'\n{"="*68}\n【5】底背离+量能阳线确认(右侧) 确认日收盘入场\n{"="*68}')
    print(f"{'时段':<9}{'N':>8}{'r5':>8}{'r5超额':>8}{'胜率':>7}{'r10超额':>9}")
    for seg, g in E.groupby('seg'):
        b5 = g['date'].map(lambda d: idx_f(d, 5)).values
        b10 = g['date'].map(lambda d: idx_f(d, 10)).values
        ex5, ex10 = g['r5'].values - b5, g['r10'].values - b10
        print(f"{seg:<9}{len(g):>8,}{g['r5'].mean():>+8.2f}{np.nanmean(ex5):>+8.2f}"
              f"{(g['r5']>0).mean()*100:>6.0f}%{np.nanmean(ex10):>+9.2f}")
    E.to_csv('reports/reverify_divergence_events.csv', index=False)
    print('\n明细→reports/reverify_divergence_events.csv')


if __name__ == '__main__':
    main()
