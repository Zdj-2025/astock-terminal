#!/usr/bin/env python3
"""smart望远镜用法回测 — 2026-09-12(用户令: 模拟新用法).

假设(我提出的用法): smart名单=观察池地板——独立买点信号触发时,
  若该股近60日内出现过smart score≥4(资金背书), 信号应比无背书版更可靠.

三组独立信号(全信号审计的幸存者):
  ① 底背离+量能阳线确认(E组合): 价新低DIF抬升+10日内量能阳线, 确认日入场
  ② 超跌反转: ret20≤-15% + 距250日高<30%分位 + 当日阳线
  ③ 金叉: MA5上穿MA10(金叉scanner定义)
对照: 每组信号拆 smart背书(60日内score≥4) vs 无背书, 比20日超额.
判定(R1): 背书组显著优于无背书组(p<0.05且N≥30) → smart升格为scanner标注列;
  不显著 → smart纯观察池, 连加分项都不是(诚实终点).
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

START, END = '20250701', '20260828'
COST = 0.0013
BACK_DAYS = 60


def load(code):
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    for c in ['datetime', 'day', 'trade_date']:
        if c in df.columns and 'date' not in df.columns:
            df = df.rename(columns={c: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def main():
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    ic = idx.set_index('d')['close'].astype(float)
    dates = idx['d'].tolist()
    dpos = {d: i for i, d in enumerate(dates)}

    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))
    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])

    rows = []  # (signal, smart_backed, ex20)
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
        n = len(df)
        dd = df['d'].values
        scores = score5_series(c, v)
        # smart背书: 近60日内score>=4
        sm = np.zeros(n, dtype=bool)
        idx4 = np.where(scores >= 4)[0]
        for i in idx4:
            sm[i + 1:i + 1 + BACK_DAYS] = True

        def ex(entry):
            if entry is None or entry + 21 >= n or dd[entry] not in dpos:
                return None
            ed = dd[entry]
            ii = dpos[ed]
            px = c[min(entry + 20, n - 1)]
            jj = min(ii + 20, len(dates) - 1)
            return (px / c[entry] - 1 - COST - (ic[dates[jj]] / ic[ed] - 1)) * 100

        # ① 底背离+量能阳线(E组合) — 引擎对齐divergence_sector_regime
        ef = pd.Series(c).ewm(span=12, adjust=False).mean()
        es = pd.Series(c).ewm(span=26, adjust=False).mean()
        dif = ef - es
        difv = dif.values
        v5 = pd.Series(v).rolling(5).mean().shift(1).values
        try:
            troughs = argrelextrema(difv, np.less, order=5)[0]
        except Exception:
            troughs = []
        last_conf = -999
        for ai in range(len(troughs) - 1):
            t1, t2 = troughs[ai], troughs[ai + 1]
            if t2 - t1 > 40 or t2 + 11 >= n:
                continue
            if not (c[t2] < c[t1] and difv[t2] > difv[t1]):
                continue
            for k in range(t2 + 1, min(t2 + 11, n)):
                if o[k] < c[k] and c[k] > c[k - 1] and v[k] > 1.2 * (v5[k] if not np.isnan(v5[k]) else 0):
                    if dd[k] < START or dd[k] > END or k - last_conf < 15:
                        break
                    last_conf = k
                    r = ex(k)
                    if r is not None:
                        rows.append(('E背离确认', bool(sm[k]), r))
                    break

        # ② 超跌反转
        ma250max = pd.Series(c).rolling(250, min_periods=100).max().values
        last_os = -999
        for t in range(260, n - 21):
            if dd[t] < START or dd[t] > END or t - last_os < 15:
                continue
            ret20 = c[t] / c[t - 20] - 1
            pos = (c[t] - (c[t - 250:t].min() if t >= 250 else c[max(0, t-250):t].min())) / \
                  ((ma250max[t] - (c[t - 250:t].min() if t >= 250 else 0)) or 1)
            if ret20 <= -0.15 and pos < 0.30 and c[t] > o[t]:
                last_os = t
                r = ex(t)
                if r is not None:
                    rows.append(('超跌反转', bool(sm[t]), r))

        # ③ 金叉
        ma5 = pd.Series(c).rolling(5).mean().values
        ma10 = pd.Series(c).rolling(10).mean().values
        last_gc = -999
        for t in range(260, n - 21):
            if dd[t] < START or dd[t] > END or t - last_gc < 15:
                continue
            if ma5[t - 1] <= ma10[t - 1] and ma5[t] > ma10[t]:
                last_gc = t
                r = ex(t)
                if r is not None:
                    rows.append(('金叉', bool(sm[t]), r))

    t = pd.DataFrame(rows, columns=['signal', 'backed', 'ex20'])
    print(f'smart望远镜用法×三信号 | {START}~{END} | smart背书=60日内score≥4 | 20日超额')
    print('═' * 66)
    print(f"{'信号':<10}{'背书N':>7}{'背书均':>8}{'胜率':>6}{'无背N':>7}{'无背均':>8}{'胜率':>6}{'差':>7}{'p':>8}")
    for sig in ('E背离确认', '超跌反转', '金叉'):
        sub = t[t['signal'] == sig]
        a = sub[sub['backed']]['ex20'].dropna()
        b = sub[~sub['backed']]['ex20'].dropna()
        if not len(a) or not len(b):
            continue
        ts, p = st.ttest_ind(a, b, equal_var=False)
        print(f"{sig:<10}{len(a):>7}{a.mean():>+7.2f}%{(a>0).mean()*100:>5.0f}%{len(b):>7}{b.mean():>+7.2f}%{(b>0).mean()*100:>5.0f}%{a.mean()-b.mean():>+6.2f}{p:>8.4f}")


if __name__ == '__main__':
    main()
