#!/usr/bin/env python3
"""回测计算方法审计 — 2026-09-12(用户令: 保证收益率/超额/胜率计算的科学性).

对象: 已采纳的"金叉×smart背书"结论(+0.66% vs -0.48%, p<0.0001).
六项实验:
  T1 复权正确性: 抽样20个事件, 本地qfq收益 vs tushare raw×adj_factor重算, 对账
  T2 涨停可成交性: 入场日开盘涨幅≥限价98%视为不可成交, 剔除后结论是否存活
  T3 基准敏感性: 上证 vs 沪深300 vs 中证1000 三种基准下的结论
  T4 成本敏感性: 单边0.13% / 0.25% / 0.4%(含滑点)
  T5 对齐完整性: 个股入场→期末交易日数 vs 指数交易日数错位分布
  T6 生存者: 2025-2026退市股是否缺席本地库(tushare list_status=D计数)
判定: 结论在"涨停过滤+300/1000基准+0.25%成本"最严口径下仍p<0.05 → 存活;
  否则撤回采纳并重估.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import numpy as np
import pandas as pd
from scipy import stats as st
from scripts.research.reverify_smart_money_recency import score5_series

START, END = '20250701', '20260828'


def load(code):
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    for c in ['datetime', 'day', 'trade_date']:
        if c in df.columns and 'date' not in df.columns:
            df = df.rename(columns={c: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def bench_series(f, col='000001_SH'):
    idx = pd.read_parquet(f'data/store/index/{f if f else col}.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    s = pd.to_numeric(idx['close'], errors='coerce')
    s.index = idx['d']
    return s.dropna().sort_index()


def main():
    ic_sh = bench_series('000001_SH')
    ic_300 = bench_series('000300_SH')
    ic_1000 = bench_series('000852_SH')
    dpos = {d: i for i, d in enumerate(ic_sh.index)}

    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))
    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])

    rows = []
    for code in codes:
        if 'ST' in nm.get(code, ''):
            continue
        try:
            df = load(code)
        except Exception:
            continue
        if len(df) < 300 or float(df['close'].iloc[-1]) < 2:
            continue
        c = df['close'].astype(float).values
        o = df['open'].astype(float).values if 'open' in df.columns else c
        v = df['volume'].astype(float).values
        n = len(df); dd = df['d'].values
        scores = score5_series(c, v)
        sm = np.zeros(n, dtype=bool)
        for i in np.where(scores >= 4)[0]:
            sm[i + 1:i + 61] = True
        ma5 = pd.Series(c).rolling(5).mean().values
        ma10 = pd.Series(c).rolling(10).mean().values
        last = -999
        for t in range(130, n - 21):
            if dd[t] < START or dd[t] > END or t - last < 15:
                continue
            if ma5[t - 1] <= ma10[t - 1] and ma5[t] > ma10[t] and c[t] > o[t]:
                last = t
            else:
                continue
            if dd[t] not in dpos:
                continue
            ii = dpos[dd[t]]
            ed = dd[t]
            j20 = min(ii + 20, len(ic_sh.index) - 1)
            d_end = ic_sh.index[j20]
            # 个股期末: 对齐到指数期末日期(而非min(t+20,n-1)) — 同时记录错位
            stock_end_i = t + np.searchsorted(dd, d_end, side='right') - 1 - 0
            # 简化: 找≤d_end的最后一个bar
            sei = np.searchsorted(dd, d_end, side='right') - 1
            misalign = (sei - t) - 20
            if sei <= t:
                continue
            px_end, px_entry = c[sei], c[t]
            r_stock = px_end / px_entry - 1
            r_sh = ic_sh.iloc[j20] / ic_sh.iloc[ii] - 1
            j300 = min(np.searchsorted(ic_300.index, ed, 'right') - 1 + 20, len(ic_300) - 1)
            i300 = np.searchsorted(ic_300.index, ed, 'right') - 1
            r_300 = ic_300.iloc[j300] / ic_300.iloc[i300] - 1 if i300 >= 0 else np.nan
            j1000 = min(np.searchsorted(ic_1000.index, ed, 'right') - 1 + 20, len(ic_1000) - 1)
            i1000 = np.searchsorted(ic_1000.index, ed, 'right') - 1
            r_1000 = ic_1000.iloc[j1000] / ic_1000.iloc[i1000] - 1 if i1000 >= 0 else np.nan
            prev_c = c[t - 1]
            open_gap = o[t] / prev_c - 1 if prev_c > 0 else 0
            lim = 0.195 if code[0] in ('3', '68') else 0.095
            unfillable = open_gap >= lim * 0.98
            rows.append(dict(code=code, d=ed, backed=bool(sm[t]), r=r_stock,
                             r_sh=r_sh, r_300=r_300, r_1000=r_1000,
                             unfillable=unfillable, misalign=misalign))

    t = pd.DataFrame(rows)
    t.to_csv('reports/methodology_audit_events.csv', index=False)
    print(f'金叉事件N={len(t)} | 不可成交(开盘即涨停)占比 {t.unfillable.mean()*100:.1f}% | 对齐错位>3天占比 {(t.misalign.abs()>3).mean()*100:.1f}%')

    def arm(sub, bench, cost):
        e = (sub['r'] - sub[bench] - cost) * 100
        return e.dropna()

    print('\n═══ T2+T3+T4 联合敏感性(背书组 vs 无背组, 20日) ═══')
    scenarios = [
        ('原口径: 上证, 成本0.13%, 不滤涨停', 'r_sh', 0.0013, False),
        ('+滤涨停(可成交)', 'r_sh', 0.0013, True),
        ('+滤涨停+成本0.25%', 'r_sh', 0.0025, True),
        ('+滤涨停+成本0.4%(含滑点)', 'r_sh', 0.004, True),
        ('+滤涨停+基准=沪深300', 'r_300', 0.0025, True),
        ('+滤涨停+基准=中证1000', 'r_1000', 0.0025, True),
        ('最严: 滤涨停+中证1000+成本0.4%', 'r_1000', 0.004, True),
    ]
    for label, bench, cost, filt in scenarios:
        sub = t[~t.unfillable] if filt else t
        a = arm(sub[sub.backed], bench, cost)
        b = arm(sub[~sub.backed], bench, cost)
        if not len(a) or not len(b):
            print(f'{label:<32} 样本缺'); continue
        ts, p = st.ttest_ind(a, b, equal_var=False)
        print(f"{label:<34} 背书{a.mean():>+6.2f}% vs 无背{b.mean():>+6.2f}% | 差{a.mean()-b.mean():>+5.2f} p={p:.4f} {'✅存活' if p<0.05 and a.mean()>b.mean() else '❌不存活'} N={len(a)}/{len(b)}")

    # T1 复权对账(抽样)
    print('\n═══ T1 复权对账(抽15个事件 vs tushare raw×adj_factor) ═══')
    import time as _t
    from scripts.core.tushare_client import TushareClient
    import tushare as tspro
    import re as _re
    tok = _re.search(r'["\']([0-9a-f]{32,64})["\']', open('scripts/core/tushare_client.py').read()).group(1)
    tspro.set_token(tok)
    sample = t.dropna(subset=['r']).sample(15, random_state=42)
    errs = []
    for _, r0 in sample.iterrows():
        try:
            tc = r0['code'] + ('.SH' if r0['code'][0] in '69' else '.SZ')
            raw = tspro.pro.daily(ts_code=tc, start_date=r0['d'], end_date=r0['d'])
            af = tspro.pro.adj_factor(ts_code=tc, start_date=r0['d'], end_date=r0['d'])
            _t.sleep(0.15)
            if raw is None or af is None or not len(raw) or not len(af):
                continue
            px_raw = float(raw.iloc[0]['close']); f = float(af.iloc[0]['adj_factor'])
            adjpx = px_raw * f
            # 本地qfq与"raw×因子"差一个常数比例(基准日), 用比值一致性校验: 本地价/adjpx应全窗口恒定
            loc = load(r0['code'])
            i0 = loc.index[loc['d'] == r0['d']][0]
            ratio_a = float(loc['close'].iloc[i0]) / adjpx
            ratio_b = float(loc['close'].iloc[min(i0 + 20, len(loc) - 1)]) / (float(tspro.pro.daily(ts_code=tc, start_date=loc['d'].iloc[min(i0 + 20, len(loc) - 1)], end_date=loc['d'].iloc[min(i0 + 20, len(loc) - 1)]).iloc[0]['close']) * float(tspro.pro.adj_factor(ts_code=tc, start_date=loc['d'].iloc[min(i0 + 20, len(loc) - 1)], end_date=loc['d'].iloc[min(i0 + 20, len(loc) - 1)]).iloc[0]['adj_factor']))
            errs.append(abs(ratio_a / ratio_b - 1))
            _t.sleep(0.15)
        except Exception:
            continue
    if errs:
        verdict = '✅复权一致' if max(errs) < 0.005 else '❌存在断裂'
        print(f'抽样{len(errs)}个: 本地qfq与raw×adj_factor比值漂移 最大{max(errs)*100:.3f}% 平均{np.mean(errs)*100:.3f}% ({verdict})')

    # T6 生存者
    print('\n═══ T6 生存者偏差 ═══')
    try:
        import tushare as ts2
        dead = ts2.pro.stock_basic(exchange='', list_status='D', fields='ts_code,list_date,delist_date')
        if dead is not None and len(dead):
            recent = dead[(dead['delist_date'] >= '20250101')]
            in_store = sum(1 for _, r in recent.iterrows() if Path(f'data/store/kline/{r.ts_code[:6]}/D.parquet').exists())
            print(f'2025年以来退市股: {len(recent)}只 | 其中在本地库: {in_store}只 → 缺失{len(recent)-in_store}只(幸存者偏差源)')
    except Exception as e:
        print('T6失败:', str(e)[:60])


if __name__ == '__main__':
    main()
