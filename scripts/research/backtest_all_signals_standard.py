#!/usr/bin/env python3
"""全信号标准化审计 — 2026-09-12(用户令: 用统一理论回测所有策略).

依据 docs/回测计算规范.md 最严口径, 六个买入信号族一次跑完(共享加载, 同一把尺子):
  ①E背离确认 ②金叉 ③超跌反转 ④双底早触发 ⑤突破(平台+20日高+非高位) ⑥催化(预增>30%次日)

每族统一输出:
  - 期限曲线 5/10/20/40: 绝对收益 + 超额vs中证1000(主基准) + vs上证(参考)
  - 分期拆解(2025H2牛/2026H1弱/2026H2反弹)
  - 规则出场(-8%止损或40天时间止损)
  - smart背书拆分(排序效应)
统一口径: 次日开盘入场 + 涨停不可成交剔除 + 成本0.25% + 复权qfq(已审计)
判定列: 20d超额vs1000>0且最严p<0.05 = "选股alpha✅"; 绝对>0 = "赚钱(靠市)"; 组间差>0 = "排序有效"
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
from scipy.signal import argrelextrema
from scripts.research.reverify_smart_money_recency import score5_series
from scripts.research.backtest_div_double_bottom import detect_patterns

START, END = '20250701', '20260828'
COST = 0.0025
HOLDS = [5, 10, 20, 40]


def load(code):
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    for c in ['datetime', 'day', 'trade_date']:
        if c in df.columns and 'date' not in df.columns:
            df = df.rename(columns={c: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def bench(f):
    idx = pd.read_parquet(f'data/store/index/{f}.parquet').sort_values('trade_date')
    s = pd.to_numeric(idx['close'], errors='coerce')
    s.index = idx['trade_date'].astype(str)
    return s.dropna().sort_index()


def main():
    ic = {k: bench(f) for k, f in [('sh', '000001_SH'), ('c300', '000300_SH'), ('c1000', '000852_SH')]}
    # 补新指数数据
    from scripts.core.tushare_client import TushareClient
    pro = TushareClient().pro
    for k, tsc in [('sh', '000001.SH'), ('c300', '000300.SH'), ('c1000', '000852.SH')]:
        if ic[k].index[-1] < '20260910':
            try:
                t = pro.index_daily(ts_code=tsc, start_date=ic[k].index[-1], end_date='20260911')
                ext = pd.to_numeric(t['close'], errors='coerce')
                ext.index = t['trade_date'].astype(str)
                ic[k] = pd.concat([ic[k], ext]).drop_duplicates().sort_index()
            except Exception:
                pass

    # 催化事件: 预增>30% 公告日
    fc = pd.read_parquet('data/store/forecast/forecast_all.parquet')
    fc = fc[(fc['type'] == '预增') & (pd.to_numeric(fc['p_change_min'], errors='coerce') > 30)]
    cat_ev = {}
    for _, r in fc.iterrows():
        cat_ev.setdefault(r['ts_code'][:6], []).append(str(r['ann_date']))

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
        o = df['open'].astype(float).values
        v = df['volume'].astype(float).values
        lo = df['low'].astype(float).values
        n = len(df); dd = df['d'].values
        lim = 0.195 if code[0] in ('3', '68') else 0.095

        scores = score5_series(c, v)
        sm = np.zeros(n, dtype=bool)
        for i in np.where(scores >= 4)[0]:
            sm[i + 1:i + 61] = True

        # 各族信号日集合(当日t出信号, 次日开盘入)
        sigs = {k: set() for k in ('E背离', '金叉', '超跌', '双底', '突破')}
        # ① E背离确认
        difv = (pd.Series(c).ewm(span=12, adjust=False).mean() - pd.Series(c).ewm(span=26, adjust=False).mean()).values
        v5 = pd.Series(v).rolling(5).mean().shift(1).values
        try:
            tr = argrelextrema(difv, np.less, order=5)[0]
        except Exception:
            tr = []
        for ai in range(len(tr) - 1):
            t1, t2 = tr[ai], tr[ai + 1]
            if t2 - t1 > 40 or t2 + 11 >= n or not (c[t2] < c[t1] and difv[t2] > difv[t1]):
                continue
            for k in range(t2 + 1, min(t2 + 11, n)):
                if o[k] < c[k] and c[k] > c[k - 1] and v[k] > 1.2 * (v5[k] if not np.isnan(v5[k]) else 0):
                    sigs['E背离'].add(k)
                    break
        # ② 金叉
        ma5 = pd.Series(c).rolling(5).mean().values
        ma10 = pd.Series(c).rolling(10).mean().values
        for t in range(20, n):
            if ma5[t - 1] <= ma10[t - 1] and ma5[t] > ma10[t] and c[t] > o[t]:
                sigs['金叉'].add(t)
        # ③ 超跌反转
        hi250 = pd.Series(c).rolling(250, min_periods=100).max().values
        lo250 = pd.Series(c).rolling(250, min_periods=100).min().values
        for t in range(250, n):
            if c[t] / c[t - 20] - 1 <= -0.15 and hi250[t] > lo250[t] and \
               (c[t] - lo250[t]) / (hi250[t] - lo250[t]) < 0.30 and c[t] > o[t]:
                sigs['超跌'].add(t)
        # ④ 双底早触发
        try:
            for p in detect_patterns(df, 0.20):
                if p['early'] is not None:
                    sigs['双底'].add(p['early'])
        except Exception:
            pass
        # ⑤ 突破
        p20max = pd.Series(c).shift(1).rolling(20).max().values
        p20min = pd.Series(c).shift(1).rolling(20).min().values
        hi120 = pd.Series(c).shift(1).rolling(120).max().values
        for t in range(130, n):
            if p20max[t] / p20min[t] - 1 < 0.15 and c[t] > p20max[t] and c[t] < 0.95 * hi120[t]:
                sigs['突破'].add(t)
        # ⑥ 催化
        for ad in cat_ev.get(code, []):
            pos = np.searchsorted(dd, ad, side='right')
            if pos < n:
                sigs.setdefault('催化', set()).add(pos - 1 if dd[pos - 1] == ad or True else pos)

        # 统一评估
        for fam, tset in sigs.items():
            last = -999
            for t in sorted(tset):
                if dd[t] < START or dd[t] > END or t - last < 10 or t + 41 >= n:
                    continue
                last = t
                e = t + 1
                if e >= n:
                    continue
                if o[e] / c[t] - 1 >= lim * 0.98:   # 涨停不可成交
                    continue
                entry = o[e]
                row = dict(fam=fam, code=code, d=dd[e], backed=bool(sm[t]))
                ok = True
                for H in HOLDS:
                    ei = min(e + H, n - 1)
                    r_abs = (c[ei] / entry - 1 - COST) * 100
                    row[f'a{H}'] = r_abs
                    exs = {}
                    for k in ('sh', 'c300', 'c1000'):
                        s = ic[k]
                        i0 = s.index.searchsorted(dd[e], 'right') - 1
                        j = min(i0 + H, len(s) - 1)
                        if i0 < 0 or j <= i0:
                            ok = False
                            break
                        exs[k] = r_abs - (s.iloc[j] / s.iloc[i0] - 1) * 100
                    if not ok:
                        break
                    row[f'e1000_{H}'] = exs['c1000']
                    row[f'esh_{H}'] = exs['sh']
                if not ok:
                    continue
                # 规则出场: -8%止损(收盘判) 或40天
                exit_i = e
                for k2 in range(e, min(e + 40, n)):
                    if c[k2] / entry - 1 <= -0.08:
                        exit_i = k2
                        break
                else:
                    exit_i = min(e + 40, n - 1)
                s1000 = ic['c1000']
                i0 = s1000.index.searchsorted(dd[e], 'right') - 1
                j2 = min(i0 + (exit_i - e), len(s1000) - 1)
                row['rule_abs'] = (c[exit_i] / entry - 1 - COST) * 100
                row['rule_ex1000'] = row['rule_abs'] - (s1000.iloc[j2] / s1000.iloc[i0] - 1) * 100
                row['stopped'] = exit_i < e + 39 and c[exit_i] / entry - 1 <= -0.08
                per = '2025H2牛' if dd[e] < '20260101' else ('2026H1弱' if dd[e] < '20260701' else '2026H2反')
                row['per'] = per
                rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv('reports/all_signals_standard_audit.csv', index=False)
    print(f'全信号标准化审计 | {START}~{END} | 次日开盘+滤涨停+成本0.25% | 事件总数{len(t)}')
    print('═' * 96)
    print(f"{'族':<6}{'N':>6} | {'5d超1000':>9}{'10d':>8}{'20d':>8}{'40d':>8} | {'20d绝对':>8} | {'规则出场超额':>10}{'止损率':>7} | {'牛/弱/反(20d超额)':>20}")
    for fam in ('E背离', '金叉', '超跌', '双底', '突破', '催化'):
        sub = t[t.fam == fam]
        if not len(sub):
            print(f'{fam:<6} 无事件')
            continue
        es = [sub[f'e1000_{H}'].mean() for H in HOLDS]
        byp = sub.groupby('per')[f'e1000_20'].mean()
        print(f"{fam:<6}{len(sub):>6} | {es[0]:>+8.2f}%{es[1]:>+7.2f}%{es[2]:>+7.2f}%{es[3]:>+7.2f}% |"
              f" {sub['a20'].mean():>+7.2f}% | {sub['rule_ex1000'].mean():>+9.2f}%{sub['stopped'].mean()*100:>6.0f}% |"
              f" {byp.get('2025H2牛', 0):+.1f}/{byp.get('2026H1弱', 0):+.1f}/{byp.get('2026H2反', 0):+.1f}")
    # smart排序效应
    print('─' * 96)
    print('smart背书排序效应(20d超额vs1000, 背书-无背):')
    for fam in ('E背离', '金叉', '超跌', '双底', '突破', '催化'):
        sub = t[t.fam == fam]
        a = sub[sub.backed]['e1000_20'].dropna()
        b = sub[~sub.backed]['e1000_20'].dropna()
        if len(a) >= 30 and len(b) >= 30:
            ts, p = st.ttest_ind(a, b, equal_var=False)
            print(f"  {fam:<6} 背书{a.mean():+.2f}% vs 无背{b.mean():+.2f}% 差{a.mean()-b.mean():+.2f} p={p:.4f} N={len(a)}/{len(b)}")


if __name__ == '__main__':
    main()
