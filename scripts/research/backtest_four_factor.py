#!/usr/bin/env python3
"""四因子回测 — 验证"板块资金流前置"是否提升三因子(2026-06-17)。

认知: 三因子精选Day1跑输, 因选了资金流出板块(AI/机器人)。资金流应前置。
四因子 = 板块资金流(流入) + V3趋势 + 回踩买点
对比: 三因子(V3+回踩, 无资金流) vs 四因子(+板块资金流前置)

数据: sector_rotation的板块缓存(etf_daily + sector_moneyflow, 1年)
严格无未来: net_5d=rolling(5)无center; f1/回踩=tcl_pattern已shift防look-ahead; ret=shift(-10)仅衡量。

用法: PYTHONPATH=. python scripts/research/backtest_four_factor.py
"""
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask
from scripts.screener.sector_rotation.config import SECTOR_DEFS
from scripts.screener.sector_rotation.backtester import load_backtest_data

START, END = '20250601', '20260531'
HOLD = 10


def metrics(s):
    s = s.dropna()
    if len(s) < 5:
        return None
    w, l = s[s > 0], s[s < 0]
    pf = w.mean() / abs(l.mean()) if len(l) and l.mean() != 0 else 0
    sh = s.mean() / s.std() * np.sqrt(252 / HOLD) if s.std() > 0 else 0
    return {'n': len(s), 'mean': s.mean() * 100, 'win': (s > 0).mean() * 100, 'PF': pf, 'sh': sh}


def main():
    print(f"四因子回测 | 板块ETF级 | 持有{HOLD}日 | {START}~{END} | 严格无未来")
    data = load_backtest_data(START, END)
    print(f"加载板块: {list(data.keys())}\n")

    all_3f, all_4f, all_flow = [], [], []
    print(f"{'板块':<8}{'三因子n/均值/胜率':<28}{'四因子n/均值/胜率':<28}{'资金流入n/均值'}")
    for name in SECTOR_DEFS:
        d = data.get(name)
        if not d or d['etf_daily'].empty:
            continue
        etf = d['etf_daily'].copy()
        sm = d['sector_moneyflow'].copy()
        # 标准化ETF列
        colmap = {'trade_date': 'date', 'vol': 'volume'}
        etf = etf.rename(columns={k: v for k, v in colmap.items() if k in etf.columns})
        etf['date'] = pd.to_datetime(etf['date'])
        etf = etf.sort_values('date').reset_index(drop=True)
        if 'close' not in etf.columns:
            continue
        etf = compute_indicators(etf)            # f1/MA/DIF/DEA/dh60 (无未来)
        etf['tcl'] = tcl_mask(etf)               # 三因子: V3+回踩
        # 板块资金流(无未来): 5日净流入 + 连入
        sm = sm.sort_values('trade_date').reset_index(drop=True)
        sm['net_5d'] = sm['net_mf_amount'].rolling(5).sum()    # 5日净流入(无center)
        sm['flow_in'] = sm['net_5d'] > 0                        # 资金流入
        etf['td'] = etf['date'].dt.strftime('%Y%m%d')
        m = etf.merge(sm[['trade_date', 'net_5d', 'flow_in']], left_on='td', right_on='trade_date', how='left')
        m['sig_3f'] = m['tcl']                                  # 三因子
        m['sig_4f'] = m['tcl'] & m['flow_in']                   # 四因子(+资金流前置)
        m['sig_flow'] = m['flow_in']                            # 纯资金流(对照)
        m['ret'] = m['close'].shift(-HOLD) / m['close'] - 1     # 未来收益仅衡量

        r3 = metrics(m[m['sig_3f']]['ret'])
        r4 = metrics(m[m['sig_4f']]['ret'])
        rf = metrics(m[m['sig_flow']]['ret'])
        s3 = f"n={r3['n']:3d} {r3['mean']:+5.2f}% 胜{r3['win']:.0f}%" if r3 else "n=0"
        s4 = f"n={r4['n']:3d} {r4['mean']:+5.2f}% 胜{r4['win']:.0f}%" if r4 else "n=0"
        sf = f"n={rf['n']:3d} {rf['mean']:+5.2f}%" if rf else "n=0"
        print(f"{name:<8}{s3:<28}{s4:<28}{sf}")
        all_3f += m[m['sig_3f']]['ret'].dropna().tolist()
        all_4f += m[m['sig_4f']]['ret'].dropna().tolist()
        all_flow += m[m['sig_flow']]['ret'].dropna().tolist()

    print(f"\n{'='*60}\n汇总(全部板块合并)\n{'='*60}")
    for lbl, lst in [('三因子(V3+回踩)', all_3f), ('四因子(+板块资金流前置)', all_4f), ('纯板块资金流(对照)', all_flow)]:
        m = metrics(pd.Series(lst))
        if m:
            print(f"  {lbl}: n={m['n']:4d} 均值{m['mean']:+6.2f}% 胜率{m['win']:5.1f}% PF{m['PF']:.2f} 夏普{m['sh']:.2f}")
    # 提升量
    m3 = metrics(pd.Series(all_3f))
    m4 = metrics(pd.Series(all_4f))
    if m3 and m4:
        print(f"\n  → 资金流前置提升: 均值 {m3['mean']:+.2f}% → {m4['mean']:+.2f}% ({m4['mean']-m3['mean']:+.2f}%), "
              f"胜率 {m3['win']:.1f}% → {m4['win']:.1f}%")


if __name__ == "__main__":
    main()
