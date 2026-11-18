#!/usr/bin/env python3
"""超跌信号分批执行优化 — 2026-10-01立(P2-9)。

问题: 超跌是最强真edge(40d超额+10.94%)但67%止损率——赚的是忍耐钱,浮亏深。
检验: 分批入场能否降"痛感"而不吃收益?

信号: ret20≤-15% 且 距120日高<-25% 且 当日阳线(与oversold_reversal_scanner口径一致)
基线: 次日开盘100%入场,持有40d,-8%止损(逐日low触及即止损),成本0.25%
变体(每tranche独立-8%止损,公平不起摊平幻觉):
  A越跌越买: 首笔50%@次日开盘;10日内触及首笔价-5%则补50%@该价;否则只有半仓
  B时间分散: 首笔50%@次日开盘;第5日开盘补50%
  C确认补仓: 首笔50%@次日开盘;收盘价站回信号日收盘价以上当日收盘补50%(≤10日窗)
指标: 资金加权收益 / 止损率(任一tranche触-8%) / 平均最大浮亏 / t检验vs基线
用法: PYTHONPATH=. python scripts/research/oversold_stagger_entry.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import numpy as np
import pandas as pd
from scipy import stats

KDIR = 'data/store/kline'
COST = 0.0025
HOLD = 40
STOP = 0.08


def load(code):
    df = pd.read_parquet(f'{KDIR}/{code}/D.parquet')
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    df = df.dropna(subset=['close']).sort_values(dcol).reset_index(drop=True)
    for c in ('open', 'close', 'high', 'low'):
        df[c] = df[c].astype(float)
    return df


def signals(df):
    c = df['close']
    ret20 = c / c.shift(20) - 1
    high120 = c.shift(1).rolling(120).max()
    pos = c / high120 - 1
    yang = df['close'] > df['open']
    sig = (ret20 <= -0.15) & (pos <= -0.25) & yang
    return sig


def sim(df, i_sig):
    """返回4策略的(资金加权收益率, 是否止损, 最大浮亏比例)"""
    o, h, l, c = (df[k].values for k in ('open', 'high', 'low', 'close'))
    n = len(c)
    e0 = min(i_sig + 1, n - 1)          # 次日
    if e0 + 5 >= n:
        return None
    end = min(e0 + HOLD, n - 1)

    def leg(entry_i, entry_px, size):
        """单tranche: -8%止损(逐日low检查)或到期收盘;worst统一为比率口径"""
        worst = 0.0
        for j in range(entry_i, end + 1):
            worst = min(worst, l[j] / entry_px - 1)
            if l[j] <= entry_px * (1 - STOP):
                return ((1 - STOP) - 1 - COST), True, worst
        return (c[end] / entry_px - 1 - COST), False, worst

    base_r, base_s, base_dd = leg(e0, o[e0], 1.0)

    # A 越跌越买: 10日内触及e价-5%
    add_i, add_px = None, None
    for j in range(e0, min(e0 + 10, end + 1)):
        if l[j] <= o[e0] * 0.95:
            add_i, add_px = j, o[e0] * 0.95
            break
    if add_i is not None:
        r1, s1, d1 = leg(e0, o[e0], 0.5); r2, s2, d2 = leg(add_i, add_px, 0.5)
        A = (0.5 * r1 + 0.5 * r2, s1 or s2, min(d1, d2))
    else:
        A = (0.5 * base_r, base_s, base_dd)   # 只有半仓

    # B 时间分散: 第5日开盘
    e5 = min(e0 + 5, n - 1)
    r1, s1, d1 = leg(e0, o[e0], 0.5); r2, s2, d2 = leg(e5, o[e5], 0.5)
    B = (0.5 * r1 + 0.5 * r2, s1 or s2, min(d1, d2))

    # C 确认补仓: ≤10日内收盘>信号日收盘
    ci, cp = None, None
    for j in range(e0, min(e0 + 10, end + 1)):
        if c[j] > c[i_sig]:
            ci, cp = j, c[j]
            break
    if ci is not None and ci <= end:
        r1, s1, d1 = leg(e0, o[e0], 0.5); r2, s2, d2 = leg(ci, cp, 0.5)
        C_ = (0.5 * r1 + 0.5 * r2, s1 or s2, min(d1, d2))
    else:
        C_ = (0.5 * base_r, base_s, base_dd)
    return base_r, base_s, base_dd, A, B, C_


def main():
    rows = []
    codes = [d for d in os.listdir(KDIR) if d.isdigit() and len(d) == 6
             and d[:2] not in ('51', '15', '56', '58')]
    for k, code in enumerate(codes):
        try:
            df = load(code)
            if len(df) < 200:
                continue
            df = df.tail(400).reset_index(drop=True)
            sig = signals(df).values
            last_i = -99
            for i in np.where(sig)[0]:
                if i - last_i < 10:      # 去重: 同股间隔≥10日
                    continue
                last_i = i
                r = sim(df, i)
                if r:
                    rows.append((code, i, r[0], r[1], r[2], *r[3], *r[4], *r[5]))
        except Exception:
            continue
        if k % 800 == 0:
            print(f'  {k}/{len(codes)}...', flush=True)
    d = pd.DataFrame(rows, columns=['code', 'i', 'base_r', 'base_s', 'base_dd',
                                    'A_r', 'A_s', 'A_dd', 'B_r', 'B_s', 'B_dd',
                                    'C_r', 'C_s', 'C_dd'])
    print(f'\n信号数: {len(d)} (全市场,2025-07→2026-08窗口,去重后)\n')
    print(f'{"策略":12s} {"平均收益":>8s} {"中位":>8s} {"盈利率":>7s} {"止损率":>7s} {"平均最大浮亏":>10s} {"P90":>8s} {"vs基线p":>8s}')
    for nm, rcol, scol, dcol in [('基线·满仓', 'base_r', 'base_s', 'base_dd'),
                                 ('A·越跌越买', 'A_r', 'A_s', 'A_dd'),
                                 ('B·时间分散', 'B_r', 'B_s', 'B_dd'),
                                 ('C·确认补仓', 'C_r', 'C_s', 'C_dd')]:
        r, s, dd = d[rcol], d[scol], d[dcol]
        t, p = stats.ttest_rel(r, d['base_r']) if nm != '基线·满仓' else (0, 1)
        print(f'{nm:12s} {r.mean():+8.2%} {r.median():+8.2%} {(r>0).mean()*100:6.0f}% '
              f'{s.mean()*100:6.0f}% {dd.mean()*100:9.1f}% {r.quantile(0.9):+8.1%} {p:8.3f}')
    d.to_csv('reports/oversold_stagger_entry_result.csv', index=False)
    print('\n明细已存 reports/oversold_stagger_entry_result.csv')

if __name__ == '__main__':
    main()
