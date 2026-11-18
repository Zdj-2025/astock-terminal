#!/usr/bin/env python3
# ⚠️ SUPERSEDED 2026-09-08: 原np.roll(,1)有savgol泄漏(9根未来数据),已修为shift(SG_W//2)
# 正确重测结论见 memory reverify_derivative_v2_conclusion: f2四象限全无效,defensive维度退役
"""二阶导回测v2: 分大盘涨/平/跌regime验证二阶导在不同环境下的有效性。
用上证指数20日涨跌幅分regime(<-2%跌 / -2~2%平 / >2%涨)。
每个regime内: 主升浪(f>0,f">0) vs 加速下跌(f<0,f"<0) + f"独立价值。
无未来: 导数shift防look-ahead, regime用截至当日指数, forward用t+H。"""
import os, glob, random
import pandas as pd, numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import tushare as ts

KDIR = './data/store/kline'
from scripts.core.tushare_client import get_token
ts.set_token(get_token())
pro = ts.pro_api()

# 1. 上证指数(分regime用)
idx = pro.index_daily(ts_code='000001.SH', start_date='20240101', end_date='20260730')
idx = idx.sort_values('trade_date').reset_index(drop=True)
idx['ret20'] = idx['close'].pct_change(20) * 100  # 20日涨跌幅
idx_dict = {str(r['trade_date']): r['ret20'] for _, r in idx.iterrows()}

# 2. 采样股票算f'/f''
random.seed(42)
codes = [os.path.basename(d) for d in glob.glob(KDIR + '/*') if os.path.isdir(d)]
sample = random.sample(codes, min(300, len(codes)))
H = 10
SG_W, SG_P = 21, 3

rows = []
for code in sample:
    p = f'{KDIR}/{code}/D.parquet'
    try:
        d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(d) < SG_W + H + 25:
        continue
    close = d['close'].astype(float).values
    ln_p = np.log(close)
    f1_raw = savgol_filter(ln_p, SG_W, SG_P, deriv=1) * 100; f1 = pd.Series(f1_raw).shift(SG_W // 2).values
    f2_raw = savgol_filter(ln_p, SG_W, SG_P, deriv=2) * 10000; f2 = pd.Series(f2_raw).shift(SG_W // 2).values
    dates = d['date'].astype(str).values
    for i in range(SG_W, len(close) - H):
        dt = dates[i][:10].replace('-', '')  # YYYYMMDD
        ret20 = idx_dict.get(dt, None)
        if ret20 is None or np.isnan(ret20):
            continue
        if np.isnan(f1[i]) or np.isnan(f2[i]):
            continue
        # regime: 跌<-2% / 平-2~2% / 涨>2%
        if ret20 < -2:
            regime = '跌'
        elif ret20 > 2:
            regime = '涨'
        else:
            regime = '平'
        fwd = close[i + H] / close[i] - 1
        rows.append(dict(f1=f1[i], f2=f2[i], fwd=fwd, regime=regime,
                         f1p=f1[i] > 0, f2p=f2[i] > 0))

D = pd.DataFrame(rows)
print(f"样本: {len(D)} 条 | 涨{len(D[D.regime=='涨'])} 平{len(D[D.regime=='平'])} 跌{len(D[D.regime=='跌'])}")
print(f"时间: 2024-01~2026-07 | forward{H}日 | S-G(w={SG_W},p={SG_P})\n")

# 每个regime内: 4组信号 + f"独立价值
for reg in ['涨', '平', '跌']:
    sub = D[D.regime == reg]
    print(f"{'='*72}")
    print(f"📊 大盘{reg}期（20日{'<-2%' if reg=='跌' else '>2%' if reg=='涨' else '-2~2%'}）| N={len(sub)}")
    print(f"{'='*72}")
    print(f"{'信号':<24}{'N':>7}{'均值':>8}{'胜率':>7}{'p值':>9}")
    print('-' * 58)

    groups = [
        ('🚀主升浪(f>0,f">0)', sub[sub.f1p & sub.f2p]),
        ('🔄底部拐点(f<0,f">0)', sub[~sub.f1p & sub.f2p]),
        ('⚠️减速见顶(f>0,f"<0)', sub[sub.f1p & ~sub.f2p]),
        ('🔴加速下跌(f<0,f"<0)', sub[~sub.f1p & ~sub.f2p]),
    ]
    for name, g in groups:
        if len(g) < 5:
            print(f"{name:<24}{len(g):>7}  (样本不足)"); continue
        r = g['fwd'] * 100
        _, p = stats.ttest_1samp(r, 0)
        print(f"{name:<24}{len(g):>7}{r.mean():>+6.2f}%{(r>0).mean()*100:>6.0f}%{p:>8.4f}")

    # 核心对比
    a = sub[sub.f1p & sub.f2p]['fwd'] * 100
    b = sub[~sub.f1p & ~sub.f2p]['fwd'] * 100
    if len(a) > 5 and len(b) > 5:
        _, pp = stats.ttest_ind(a, b, equal_var=False)
        print(f"  主升浪 vs 加速下跌: 差{a.mean()-b.mean():+.2f}%, p={pp:.4f} ({'✅' if pp<0.05 else '⚠️'})")

    # f"独立(控制f')
    for fs in [True, False]:
        s2 = sub[sub.f1p == fs]
        pos = s2[s2.f2p]['fwd'] * 100
        neg = s2[~s2.f2p]['fwd'] * 100
        if len(pos) > 5 and len(neg) > 5:
            _, pp = stats.ttest_ind(pos, neg, equal_var=False)
            lbl = 'f>0(涨势)' if fs else 'f<0(跌势)'
            print(f"  {lbl}: f\">0 {pos.mean():+.2f}% vs f\"<0 {neg.mean():+.2f}%, p={pp:.4f} ({'✅' if pp<0.05 else '⚠️'})")
    print()

# 汇总
print(f"{'='*72}")
print("📋 汇总: 主升浪(f>0,f\">0) 各regime均值")
for reg in ['涨', '平', '跌']:
    g = D[(D.regime==reg) & D.f1p & D.f2p]
    if len(g) > 5:
        r = g['fwd'] * 100
        print(f"  {reg}: {r.mean():+.2f}% (N={len(g)}, 胜率{(r>0).mean()*100:.0f}%)")
