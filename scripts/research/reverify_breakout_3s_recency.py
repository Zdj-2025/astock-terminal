#!/usr/bin/env python3
"""Tier A时效复测③④: breakout突破信号 + 3S卖出信号 — 2026-08-17晚 P0工程Phase2。

③ breakout(对齐breakout_scanner逻辑): 平台(前20日振幅<15%)+突破前20日高+放量1.5x+非高位(距120高<-5%)
   事件后次日买入(实际可执行口径)的1/3/5/10日收益 vs 上证。分时段看衰减。
④ 3S(对齐sell_check): MA5死叉MA10 + MACD柱转负 + 收盘破MA20 三条件同日满 → 次日/5日收益
   (续跌=卖对)。原结论: 上升期有效-0.72%,下降期无效+4.18%(卖错)。分时段复核。
"""
import os, sys
import numpy as np
import pandas as pd

os.chdir('.')
sys.path.insert(0, os.getcwd())


def main():
    from scripts.core.tushare_client import TushareClient
    pro = TushareClient().pro
    idx = pro.index_daily(ts_code='000001.SH', start_date='20230101').sort_values('trade_date')
    ic = idx['close'].astype(float).values
    idates = idx['trade_date'].astype(str).values
    idmap = {d: ic[i] for i, d in enumerate(idates)}

    def idx_f(d, h):
        p = np.searchsorted(idates, d)
        return (ic[p + h] / ic[p] - 1) * 100 if p + h < len(ic) else np.nan

    sl = pd.read_parquet('data/store/stock_list.parquet').set_index('symbol')
    st = {c for c in sl.index if 'ST' in str(sl.loc[c, 'name']).upper()}
    kdir = 'data/store/kline'
    codes = [d for d in os.listdir(kdir) if d.isdigit() and len(d) == 6
             and os.path.isfile(f'{kdir}/{d}/D.parquet') and d not in st]
    bo_rows, s3_rows = [], []
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
        v = df['volume'].astype(float).values
        ma5 = pd.Series(c).rolling(5).mean().values
        ma10 = pd.Series(c).rolling(10).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        hist = dif - dea
        v5 = pd.Series(v).rolling(5).mean().values
        hi20 = pd.Series(c).rolling(20).max().values
        hi120 = pd.Series(c).rolling(120).max().values
        lo20 = pd.Series(c).rolling(20).min().values
        amp20 = (hi20 - lo20) / lo20            # 平台振幅
        # 前一日窗口(信号在t-1收盘确认,t日开盘/收盘可执行 → 用t日收盘入场近似"次日买")
        for t in range(130, len(c) - 11):
            p = t - 1
            if v[t] <= 0 or np.isnan(ma20[p]) or np.isnan(hi120[p]):
                continue
            d = str(ds[t])
            if d < '20240101':
                continue
            # breakout: 昨日及之前20日平台(amp<15%), 今日突破前20日高, 放量, 非高位
            if amp20[p] < 0.15 and c[t] > hi20[p] and v[t] > v5[p] * 1.5 and c[t] < hi120[p] * 0.95:
                fwd = tuple((c[t + h] / c[t] - 1) * 100 if t + h < len(c) else np.nan for h in (1, 3, 5, 10))
                bo_rows.append((d, *fwd))
            # 3S: 昨日三条件满(MA5<MA10 且 hist<0 且 c[p]<ma20[p]), 今日为次日
            dead = ma5[p] < ma10[p]
            neg = hist[p] < 0
            below = c[p] < ma20[p]
            if dead and neg and below:
                f1 = (c[t] / c[p] - 1) * 100
                f5 = (c[t + 4] / c[p] - 1) * 100 if t + 4 < len(c) else np.nan
                s3_rows.append((d, f1, f5))
        if (k + 1) % 1500 == 0:
            print(f'  进度{k+1}/{len(codes)} bo={len(bo_rows)} s3={len(s3_rows)}', flush=True)

    BO = pd.DataFrame(bo_rows, columns=['date', 'r1', 'r3', 'r5', 'r10'])
    S3 = pd.DataFrame(s3_rows, columns=['date', 'r1', 'r5'])
    BO['seg'] = BO.date.str[:4] + 'H' + np.where(BO.date.str[4:6].astype(int) <= 6, '1', '2')
    S3['seg'] = S3.date.str[:4] + 'H' + np.where(S3.date.str[4:6].astype(int) <= 6, '1', '2')

    print(f'\n{"="*70}\n【3】breakout信号(平台+突破+放量1.5x+非高位) 次日收盘入场\n{"="*70}')
    print(f"{'时段':<9}{'N':>8}{'r5':>8}{'r5超额':>8}{'胜率':>7}{'r10超额':>9}")
    for seg, g in BO.groupby('seg'):
        b5 = g['date'].map(lambda d: idx_f(d, 5)).values
        b10 = g['date'].map(lambda d: idx_f(d, 10)).values
        ex5 = g['r5'].values - b5
        ex10 = g['r10'].values - b10
        print(f"{seg:<9}{len(g):>8,}{g['r5'].mean():>+8.2f}{np.nanmean(ex5):>+8.2f}"
              f"{(g['r5']>0).mean()*100:>6.0f}%{np.nanmean(ex10):>+9.2f}")

    print(f'\n{"="*70}\n【4】3S满次日收益(负=卖对继续跌; 正=卖错反弹)\n{"="*70}')
    print(f"{'时段':<9}{'N':>8}{'次日均值':>9}{'次日为跌概率':>10}{'5日均值':>9}")
    for seg, g in S3.groupby('seg'):
        print(f"{seg:<9}{len(g):>8,}{g['r1'].mean():>+9.2f}{(g['r1']<0).mean()*100:>9.0f}%{g['r5'].mean():>+9.2f}")

    BO.to_csv('reports/reverify_breakout_events.csv', index=False)
    S3.to_csv('reports/reverify_3s_events.csv', index=False)
    print('\n明细→reports/reverify_breakout_events.csv / reverify_3s_events.csv')


if __name__ == '__main__':
    main()
