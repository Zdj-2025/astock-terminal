#!/usr/bin/env python3
"""底背离修复回测: 对比"修复保留(新鲜+下跌趋势)" vs "修复剔除(过期/高位)"两组的前瞻收益。
验证修复剔除的确实是坏信号(高位追涨), 保留的仍有正期望。无未来: 信号用截至当日, fwd用t+H。"""
import os, glob, random
import pandas as pd, numpy as np
from scipy.signal import argrelextrema
from scipy import stats

KDIR = './data/store/kline'
random.seed(42)
codes = [os.path.basename(d) for d in glob.glob(KDIR + '/*') if os.path.isdir(d)]
sample = random.sample(codes, min(250, len(codes)))
H = 10  # 前瞻天数

kept, removed = [], []
for code in sample:
    p = f'{KDIR}/{code}/D.parquet'
    try:
        d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(d) < 80:
        continue
    c = d['close'].astype(float).values.copy()
    l = d['low'].astype(float).values.copy() if 'low' in d else c.copy()
    h = d['high'].astype(float).values.copy() if 'high' in d else c.copy()
    r = pd.Series(c).pct_change().values  # qfq
    for i in range(len(c) - 2, -1, -1):
        if not np.isnan(r[i + 1]) and r[i + 1] < -0.25:
            ratio = c[i + 1] / c[i]; c[:i + 1] *= ratio; l[:i + 1] *= ratio; h[:i + 1] *= ratio
    dif = pd.Series(c).ewm(12, adjust=False).mean().values - pd.Series(c).ewm(26, adjust=False).mean().values
    for t in range(60, len(c) - H):
        rc = c[t - 60:t]; rd = dif[t - 60:t]; rh = h[t - 60:t]
        tr = argrelextrema(rd, np.less, order=5)[0]
        if len(tr) < 2:
            continue
        t1, t2 = tr[-2], tr[-1]
        if rc[t2] < rc[t1] and rd[t2] > rd[t1]:  # 原始底背离条件
            bounce = c[t] / rc[t2] - 1            # 谷后反弹
            drawdown = c[t] / max(rh) - 1         # 距回溯高
            fwd = c[t + H] / c[t] - 1
            if bounce > 0.08 or drawdown > -0.05:
                removed.append(fwd)               # 修复会剔除
            else:
                kept.append(fwd)                  # 修复保留

kept = np.array(kept) * 100
removed = np.array(removed) * 100


def stat(a):
    if len(a) < 2:
        return a.mean() if len(a) else float('nan'), 1.0, 0
    t, p = stats.ttest_1samp(a, 0)
    return a.mean(), p, t


mk, pk, tk = stat(kept)
mr, pr, tr_ = stat(removed)
print(f"样本 {len(sample)} 股 | forward {H} 日 | 原始底背离信号(60日窗口DIF)")
print(f"\n[修复保留组] 新鲜+下跌趋势: {len(kept)} 只 → 均值 {mk:+.2f}%/{H}日  p={pk:.3e} t={tk:.2f}")
print(f"[修复剔除组] 过期/高位:     {len(removed)} 只 → 均值 {mr:+.2f}%/{H}日  p={pr:.3e} t={tr_:.2f}")
print(f"\n剔除组 vs 保留组差: {mr - mk:+.2f}% → {'剔除的更差(✅修复正确,移除了坏信号)' if mr < mk else '剔除的更好(⚠️修复可能误删好信号)'}")
if len(kept) > 5 and len(removed) > 5:
    tt, pt = stats.ttest_ind(kept, removed, equal_var=False)
    print(f"两组差异 t检验: p={pt:.3e} ({'显著' if pt < 0.05 else '不显著'})")
