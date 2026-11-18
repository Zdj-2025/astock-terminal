#!/usr/bin/env python3
"""smart×回踩组合回测 — 2026-09-12(用户令: 修改用法发挥smart真正实力).

假设: smart预知力真实(p=2.7e-91)但直接买已死(2026H2 -1.87%/t=-14.6).
     改用法 = smart选股(谁有资金背书) × 回踩/超跌择时(何时买) → 应优于直接买.

三臂(同一smart触发日t, score5≥4):
  A 直接买: t+1开盘 (对照=已证死的用法)
  B 等回踩: 30日内 close≤MA20×0.95 首日收盘买
  C 等超跌: 30日内 ret10≤-8% 且当日阳线 收盘买
全部: 10/20日超额 vs 上证(入场日→期末), 分期看, p值+t检验.
判定(R1): B或C在2026H2显著>A(p<0.05) → 组合用法上线smart_pullback_scanner;
      否则smart彻底退居观察池标注.
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
COST = 0.0013


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

    recA, recB, recC = [], [], []  # (entry_date, ex10, ex20, period)
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
        scores = score5_series(c, v)
        ma20 = pd.Series(c).rolling(20).mean().values
        dd = df['d'].values
        n = len(df)
        trig_last = -999
        for t in range(60, n - 31):
            if scores[t] < 4 or dd[t] < START or dd[t] > END:
                continue
            if t - trig_last < 20:      # 20日去重
                continue
            trig_last = t
            period = '2025H2' if dd[t] < '20260101' else ('2026H1' if dd[t] < '20260701' else '2026H2')

            def ex(entry_i):
                if entry_i is None or entry_i + 21 >= n or dd[entry_i] not in dpos:
                    return None
                ed = dd[entry_i]
                ii = dpos[ed]
                out = []
                for H in (10, 20):
                    px = c[min(entry_i + H, n - 1)]
                    jj = min(ii + H, len(dates) - 1)
                    out.append((px / c[entry_i] - 1 - COST - (ic[dates[jj]] / ic[ed] - 1)) * 100)
                return out

            # A 直接买
            r = ex(t + 1)
            if r:
                recA.append((dd[t + 1], r[0], r[1], period))
            # B 等回踩: 30日内 close<=MA20*0.95
            b = None
            for k in range(t + 1, min(t + 31, n)):
                if not np.isnan(ma20[k]) and c[k] <= ma20[k] * 0.95:
                    b = k
                    break
            r = ex(b) if b else None
            if r:
                recB.append((dd[b], r[0], r[1], period))
            # C 等超跌阳线: 30日内 ret10<=-8% 且当日阳线
            cc_i = None
            for k in range(t + 11, min(t + 31, n)):
                ret10 = c[k] / c[k - 10] - 1
                if ret10 <= -0.08 and c[k] > o[k]:
                    cc_i = k
                    break
            r = ex(cc_i) if cc_i else None
            if r:
                recC.append((dd[cc_i], r[0], r[1], period))

    A, B, C = pd.DataFrame(recA, columns=['d', 'ex10', 'ex20', 'p']), \
              pd.DataFrame(recB, columns=['d', 'ex10', 'ex20', 'p']), \
              pd.DataFrame(recC, columns=['d', 'ex10', 'ex20', 'p'])
    print(f'smart触发(20日去重): A直接买N={len(A)} | B回踩成交N={len(B)} | C超跌阳线成交N={len(C)}')
    print('═' * 78)
    print(f"{'臂':<12}{'期':>8}{'N':>6}{'10d均':>8}{'20d均':>8}{'20d胜率':>8}")
    for label, t in (('A直接买', A), ('B等回踩MA20×0.95', B), ('C等超跌阳线', C)):
        for p in ('2025H2', '2026H1', '2026H2', 'ALL'):
            sub = t if p == 'ALL' else t[t['p'] == p]
            if not len(sub):
                continue
            print(f"{label:<12}{p:>8}{len(sub):>6}{sub['ex10'].mean():>+7.2f}%{sub['ex20'].mean():>+7.2f}%{(sub['ex20']>0).mean()*100:>7.0f}%")
    # 关键检验: 2026H2 B/C vs A
    for label, t in (('B', B), ('C', C)):
        a2 = A[A['p'] == '2026H2']['ex20'].dropna()
        t2 = t[t['p'] == '2026H2']['ex20'].dropna()
        if len(a2) >= 20 and len(t2) >= 20:
            ts, p = st.ttest_ind(t2, a2, equal_var=False)
            print(f"\n2026H2 {label} vs A: 差{t2.mean()-a2.mean():+.2f}% (t={ts:.2f}, p={p:.4f})")


if __name__ == '__main__':
    main()
