#!/usr/bin/env python3
"""涨停回踩模式回测 — 验证"回踩MA20 + 前20日涨停 + 突破过MA20"(金财模式)是否有效。

金财(002657) 6/22 涨停前: 底部涨停(6/1)→突破MA20涨停(6/10)→回踩MA20(6/18)→涨停(6/22)。
本回测验证: 这个模式(加"前期涨停")是否显著优于纯回踩MA20(tcl_pattern弱+0.66%)。
无未来: 信号用T日及之前, 收益shift(-hold)。
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
from scipy import stats

STORE = Path('./data/store/kline')
HOLDS = [5, 10, 20]
codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
print(f"涨停回踩模式回测 | {len(codes)}只")
snaps = []
for i, code in enumerate(codes):
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet').sort_values('date').reset_index(drop=True)
        if len(df) < 80:
            continue
        c = df['close'].astype(float)
        df['chg'] = c.pct_change() * 100
        ma20 = c.rolling(20).mean()
        dist = (c / ma20 - 1) * 100
        thr = 19 if code.startswith(('30', '68')) else 9.5
        zt_count = (df['chg'] >= thr).astype(int).rolling(20).sum()
        broke = ((c / ma20).rolling(20).max().shift(1) > 1.03).fillna(False)
        sig_pb = ((dist >= -3) & (dist <= 3)).fillna(False)
        sig_ztpb = (sig_pb & (zt_count.shift(1) >= 1) & broke).fillna(False)
        out = pd.DataFrame({'sig_pb': sig_pb, 'sig_ztpb': sig_ztpb})
        for h in HOLDS:
            out[f'ret{h}'] = (c.shift(-h) / c - 1)
        out['code'] = code
        snaps.append(out)
    except Exception:
        continue
    if (i + 1) % 1000 == 0:
        print(f"  [{i+1}/{len(codes)}]")

snap = pd.concat(snaps, ignore_index=True).dropna(subset=['ret10'])
base = snap['ret10']
pb = snap.loc[snap.sig_pb, 'ret10']
ztpb = snap.loc[snap.sig_ztpb, 'ret10']
print(f"\n{'组':<28}{'N':>9}{'ret5':>9}{'ret10':>9}{'ret20':>9}{'胜率10':>7}")
for nm, mask in [('基准(全市场)', pd.Series(True, index=snap.index)),
                 ('单回踩MA20±3%', snap.sig_pb),
                 ('金财模式(回踩+前20日涨停)', snap.sig_ztpb)]:
    g = snap[mask]
    wr = (g['ret10'] > 0).mean() * 100
    print(f"{nm:<26}{len(g):>9,}{g['ret5'].mean()*100:>+8.2f}%{g['ret10'].mean()*100:>+8.2f}%"
          f"{g['ret20'].mean()*100:>+8.2f}%{wr:>6.0f}%")

print("\n[t检验 ret10] 金财模式 vs:")
for nm, x in [('基准', base), ('单回踩', pb)]:
    t, p = stats.ttest_ind(ztpb, x, equal_var=False)
    better = '✅显著更优' if (p < 0.05 and ztpb.mean() > x.mean()) else \
             ('⚠️显著更差' if (p < 0.05 and ztpb.mean() < x.mean()) else '—无显著差异')
    print(f"  {nm:<6}: t={t:+.2f} p={p:.4f}  {better}  (金财{ztpb.mean()*100:+.2f}% vs {x.mean()*100:+.2f}%)")
