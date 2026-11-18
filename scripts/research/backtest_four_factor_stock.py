#!/usr/bin/env python3
"""个股级四因子回测 — 成分股验证资金流前置(2026-06-17)。

ETF级样本太少(11), 改用板块成分股(41只, 样本多4倍)验证:
四因子(V3+回踩 ∩ 板块资金流流入) vs 三因子(V3+回踩)。
严格无未来: net_5d=rolling(5)无center; V3/回踩已shift防look-ahead; ret=shift(-10)仅衡量。

用法: PYTHONPATH=. python scripts/research/backtest_four_factor_stock.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask
from scripts.screener.sector_rotation.config import SECTOR_DEFS
from scripts.screener.sector_rotation.backtester import load_backtest_data

START, END, HOLD = '20250601', '20260531', 10


def metrics(s):
    s = s.dropna()
    if len(s) < 5:
        return None
    w, l = s[s > 0], s[s < 0]
    pf = w.mean() / abs(l.mean()) if len(l) and l.mean() != 0 else 0
    sh = s.mean() / s.std() * np.sqrt(252 / HOLD) if s.std() > 0 else 0
    from scipy import stats
    t, p = stats.ttest_1samp(s, 0) if len(s) > 5 else (0, 1)
    return {'n': len(s), 'mean': s.mean() * 100, 'win': (s > 0).mean() * 100,
            'PF': pf, 'sh': sh, 'p': p, 'med': s.median() * 100}


def main():
    print(f"个股级四因子回测 | 41只成分股 | 持有{HOLD}日 | {START}~{END} | 严格无未来")
    data = load_backtest_data(START, END)
    all_3f, all_4f = [], []
    for name in SECTOR_DEFS:
        d = data.get(name)
        if not d:
            continue
        sm = d['sector_moneyflow'].sort_values('trade_date').reset_index(drop=True)
        sm['net_5d'] = sm['net_mf_amount'].rolling(5).sum()   # 5日净流入(无未来)
        flow_by_date = dict(zip(sm['trade_date'], sm['net_5d'] > 0))
        sd = d.get('stocks_daily', {})
        for code, df in sd.items():
            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'}).copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if 'close' not in df.columns or len(df) < 130:
                continue
            df = compute_indicators(df)                    # V3/MA/DIF/dh60(无未来)
            df['tcl'] = tcl_mask(df)                       # 三因子: V3+回踩
            df['td'] = df['date'].dt.strftime('%Y%m%d')
            df['flow_in'] = df['td'].map(flow_by_date)     # 板块资金流(无未来)
            df['sig_3f'] = df['tcl']
            df['sig_4f'] = df['tcl'] & df['flow_in'].fillna(False)
            df['ret'] = df['close'].shift(-HOLD) / df['close'] - 1   # 未来收益仅衡量
            all_3f += df[df['sig_3f']]['ret'].dropna().tolist()
            all_4f += df[df['sig_4f']]['ret'].dropna().tolist()

    print(f"\n{'='*64}\n个股级汇总(41成分股)\n{'='*64}")
    for lbl, lst in [('三因子(V3+回踩)', all_3f), ('四因子(+板块资金流前置)', all_4f)]:
        m = metrics(pd.Series(lst))
        if m:
            print(f"  {lbl}: n={m['n']:4d} 均值{m['mean']:+6.2f}% 胜率{m['win']:5.1f}% "
                  f"PF{m['PF']:.2f} 夏普{m['sh']:.2f} 中位{m['med']:+.2f}% p={m['p']:.4f}")
    m3 = metrics(pd.Series(all_3f))
    m4 = metrics(pd.Series(all_4f))
    if m3 and m4:
        print(f"\n  → 资金流前置提升: 均值 {m3['mean']:+.2f}%→{m4['mean']:+.2f}% "
              f"({m4['mean']-m3['mean']:+.2f}%), 胜率 {m3['win']:.1f}%→{m4['win']:.1f}%, "
              f"夏普 {m3['sh']:.2f}→{m4['sh']:.2f}")
        if m4['mean'] > m3['mean'] and m4['n'] >= 20:
            print("  ✅ 资金流前置有效(个股级验证通过)")
        else:
            print("  ⚠️ 提升不显著或样本不足")


if __name__ == "__main__":
    main()
