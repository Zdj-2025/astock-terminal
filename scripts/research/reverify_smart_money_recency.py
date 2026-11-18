#!/usr/bin/env python3
"""Tier A时效复测②: smart_money score≥4 → 未来收益 (2026-08-17 P0工程Phase2)。

与原回测的区别(关键盲区): 原回测(p=2.7e-91)只证明"预增公告前score≥3触发率高"(预知力),
从未证明"全市场score≥4名单的未来收益为正"(实战力)——scanner每天输出~169只,实战看的是这个。
本测试: score_5≥4(5特征向量化,近似生产6特征去掉底背离15.9%低触发项)的未来5/10日收益
vs 同日全市场等权基准, 分2025H2/2026H1/2026H2滚动看衰减。
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')

START = '20250701'   # 13.5个月
FWD = [5, 10]


def score5_series(close, vol):
    """向量化5特征逐日score(截至该日,无未来): 金叉/底部/放量异动/量价配合/稳步抬升。"""
    n = len(close)
    s = np.zeros(n, dtype=int)
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    vma20 = pd.Series(vol).rolling(20).mean().values
    chg = pd.Series(close).pct_change().values
    # 放量异动: 近5日 any(vol>1.3*vma20 & chg>0.04)
    blow = (vol > vma20 * 1.3) & (chg > 0.04)
    blow_any5 = pd.Series(blow.astype(int)).rolling(5).max().values
    # 量价配合: 近10日涨日均量/跌日均量>1.3 (逐日循环,纯数值)
    vp = np.zeros(n, dtype=bool)
    d10 = np.diff(close, prepend=close[0])
    for t in range(10, n):
        w = slice(t - 9, t + 1)
        up = vol[w][d10[w] > 0]
        dn = vol[w][d10[w] < 0]
        if len(up) and len(dn) and dn.mean() > 0:
            vp[t] = up.mean() / dn.mean() > 1.3
    vp_any = pd.Series(vp.astype(int)).rolling(1).max().values
    for t in range(21, n):
        sc = 0
        if ma5[t] > ma20[t]: sc += 1                    # 金叉
        if close[t] / ma20[t] < 0.92: sc += 1           # 底部
        if blow_any5[t]: sc += 1                        # 放量异动
        if vp[t]: sc += 1                               # 量价配合
        if close[t] / close[t - 10] - 1 > 0.03: sc += 1 # 稳步抬升
        s[t] = sc
    return s


def main():
    sl = pd.read_parquet('data/store/stock_list.parquet').set_index('symbol')
    st = {c for c in sl.index if 'ST' in str(sl.loc[c, 'name']).upper()}
    kdir = 'data/store/kline'
    codes = [d for d in os.listdir(kdir) if d.isdigit() and len(d) == 6
             and os.path.isfile(f'{kdir}/{d}/D.parquet')]
    print(f'样本{len(codes)}只 | 窗口{START}→ | score_5≥4 vs 全市场基准')

    all_rows = []          # (date, score, r5, r10) — 只存score>=3与抽样score<=1对照,控内存
    rng = np.random.default_rng(7)
    for k, code in enumerate(codes):
        if code in st:
            continue
        try:
            df = pd.read_parquet(f'{kdir}/{code}/D.parquet')
        except Exception:
            continue
        dc = [c for c in df.columns if 'date' in c.lower()]
        if not dc:
            continue
        df = df.sort_values(dc[0])
        ds = df[dc[0]].astype(str).str.replace('-', '').values
        m = ds >= START
        if m.sum() < 40:
            continue
        ds, c, v = ds[m], df['close'].astype(float).values[m], df['volume'].astype(float).values[m]
        # 导前历史给rolling(需21天预热): 取全序列算score再mask
        full_c = df['close'].astype(float).values
        full_v = df['volume'].astype(float).values
        full_d = df[dc[0]].astype(str).str.replace('-', '').values
        if len(full_c) < 60:
            continue
        s = score5_series(full_c, full_v)
        fwd = {}
        for h in FWD:
            f = np.full(len(full_c), np.nan)
            f[:-h] = (full_c[h:] / full_c[:-h] - 1) * 100
            fwd[h] = f
        for i in np.where(full_d >= START)[0]:
            if full_v[i] <= 0 or np.isnan(s[i]):
                continue
            keep = s[i] >= 4 or (s[i] <= 1 and rng.random() < 0.02)   # 高分组全存+低分组2%抽样
            if keep:
                all_rows.append((str(full_d[i]), int(s[i]), fwd[5][i], fwd[10][i]))
        if (k + 1) % 1000 == 0:
            print(f'  进度{k+1}/{len(codes)}', flush=True)

    E = pd.DataFrame(all_rows, columns=['date', 'score', 'r5', 'r10'])
    print(f'\n记录{len(E):,} (score≥4全量 + score≤1抽样2%)')

    # 基准: 每日组内均值自基准不行(组有偏), 用上证
    from scripts.core.tushare_client import TushareClient
    pro = TushareClient().pro
    idx = pro.index_daily(ts_code='000001.SH', start_date=START).sort_values('trade_date')
    ic = idx['close'].astype(float).values
    idates = idx['trade_date'].astype(str).values
    def idx_f(d, h):
        p = np.searchsorted(idates, d)
        return (ic[p + h] / ic[p] - 1) * 100 if p + h < len(ic) else np.nan
    E['half'] = E.date.str[:4] + 'H' + np.where(E.date.str[4:6].astype(int) <= 6, '1', '2')

    print('\n=== score_5≥4组: 未来收益 vs 上证 (分时段) ===')
    print(f"{'时段':<9}{'N':>9}{'r5':>9}{'r5超额':>9}{'胜率':>7}{'r10超额':>9}{'t(r5超额)':>10}")
    for hname, g in E[E.score >= 4].groupby('half'):
        b5 = g['date'].map(lambda d: idx_f(d, 5))
        b10 = g['date'].map(lambda d: idx_f(d, 10))
        ex5 = (g['r5'].values - b5.values)
        ex10 = (g['r10'].values - b10.values)
        ex5c = ex5[~np.isnan(ex5)]
        t = ex5c.mean() / (ex5c.std() / np.sqrt(len(ex5c))) if len(ex5c) > 2 else np.nan
        print(f"{hname:<9}{len(g):>9,}{g['r5'].mean():>+9.2f}{np.nanmean(ex5):>+9.2f}"
              f"{(g['r5']>0).mean()*100:>6.0f}%{np.nanmean(ex10):>+9.2f}{t:>10.2f}")

    print('\n=== 对照组(score_5≤1, 2%抽样): 同口径 ===')
    for hname, g in E[E.score <= 1].groupby('half'):
        b5 = g['date'].map(lambda d: idx_f(d, 5))
        ex5 = (g['r5'].values - b5.values)
        print(f"{hname:<9}{len(g):>9,}{g['r5'].mean():>+9.2f}{np.nanmean(ex5):>+9.2f}"
              f"{(g['r5']>0).mean()*100:>6.0f}%")

    E.to_parquet('reports/reverify_smart_money_events.parquet', index=False)
    print('\n明细→reports/reverify_smart_money_events.parquet')


if __name__ == '__main__':
    main()
