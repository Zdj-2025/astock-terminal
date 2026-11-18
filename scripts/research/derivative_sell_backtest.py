#!/usr/bin/env python3
# ⚠️ SUPERSEDED 2026-09-08: 原np.roll(,1)有savgol泄漏(9根未来数据),已修为shift(SG_W//2)
# 正确重测结论见 memory reverify_derivative_v2_conclusion: f2四象限全无效,defensive维度退役
"""二阶导卖出策略回测: 验证'减速见顶'(f'>0,f''<0)作为减仓信号的有效性。
逻辑: 持仓中f''从正转负 = 动量在死 = 该不该减?
对比4组(分regime): 减速见顶减仓 vs 继续持有(主升浪) vs 全部持有(基准)
关键: 卖出信号的前瞻收益应为负(确认该卖) + 与主升浪持有对比(如果减速见顶收益远低于主升浪,说明该切换)。"""
import os, glob, random
import pandas as pd, numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import tushare as ts

KDIR = './data/store/kline'
from scripts.core.tushare_client import get_token
ts.set_token(get_token())
pro = ts.pro_api()

idx = pro.index_daily(ts_code='000001.SH', start_date='20240101', end_date='20260730').sort_values('trade_date').reset_index(drop=True)
idx['ret20'] = idx['close'].pct_change(20) * 100
idx_dict = {str(r['trade_date']): r['ret20'] for _, r in idx.iterrows()}

random.seed(42)
codes = [os.path.basename(d) for d in glob.glob(KDIR + '/*') if os.path.isdir(d)]
sample = random.sample(codes, min(300, len(codes)))
H = 10; SG_W, SG_P = 21, 3

rows = []
for code in sample:
    p = f'{KDIR}/{code}/D.parquet'
    try: d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except: continue
    if len(d) < SG_W + H + 25: continue
    close = d['close'].astype(float).values
    ln_p = np.log(close)
    f1_raw = savgol_filter(ln_p, SG_W, SG_P, deriv=1) * 100; f1 = pd.Series(f1_raw).shift(SG_W // 2).values
    f2_raw = savgol_filter(ln_p, SG_W, SG_P, deriv=2) * 10000; f2 = pd.Series(f2_raw).shift(SG_W // 2).values
    dates = d['date'].astype(str).values
    for i in range(SG_W, len(close) - H):
        dt = dates[i][:10].replace('-', '')
        ret20 = idx_dict.get(dt, None)
        if ret20 is None or np.isnan(ret20): continue
        if np.isnan(f1[i]) or np.isnan(f2[i]): continue
        regime = '跌' if ret20 < -2 else ('涨' if ret20 > 2 else '平')
        # 只看f'>0的(持仓中上涨的票, 该不该减)
        if f1[i] <= 0: continue  # 只看涨势中的卖出决策
        fwd = close[i + H] / close[i] - 1
        rows.append(dict(f1=f1[i], f2=f2[i], fwd=fwd, regime=regime, f2p=f2[i] > 0))

D = pd.DataFrame(rows)
print(f"样本(仅f'>0涨势中): {len(D)} 条 | 这是在'持仓上涨'时该不该减的决策")
print(f"涨{len(D[D.regime=='涨'])} 平{len(D[D.regime=='平'])} 跌{len(D[D.regime=='跌'])}\n")

# 核心对比: f'>0&f''>0(持有/主升浪) vs f'>0&f''<0(减仓/减速见顶)
print(f"{'='*72}")
print(f"📊 卖出策略: 涨势中(f'>0), f''<0(减速见顶)该不该减?")
print(f"{'='*72}")
print(f"{'regime':<8}{'信号':<20}{'N':>7}{'10日均':>8}{'胜率':>7}{'p值':>9}")
print('-' * 62)

for reg in ['涨', '平', '跌']:
    sub = D[D.regime == reg]
    hold = sub[sub.f2p]      # f''>0 继续持有(主升浪)
    sell = sub[~sub.f2p]    # f''<0 减仓信号(减速见顶)
    if len(hold) > 5:
        r = hold['fwd'] * 100; _, p = stats.ttest_1samp(r, 0)
        print(f"{reg:<8}{'持有(f\">0主升浪)':<20}{len(hold):>7}{r.mean():>+6.2f}%{(r>0).mean()*100:>6.0f}%{p:>8.4f}")
    if len(sell) > 5:
        r = sell['fwd'] * 100; _, p = stats.ttest_1samp(r, 0)
        print(f"{reg:<8}{'减仓(f\"<0减速见顶)':<20}{len(sell):>7}{r.mean():>+6.2f}%{(r>0).mean()*100:>6.0f}%{p:>8.4f}")
    if len(hold) > 5 and len(sell) > 5:
        _, pp = stats.ttest_ind(hold['fwd']*100, sell['fwd']*100, equal_var=False)
        print(f"{reg:<8}{'差(持有-减仓)':<20}{'':>7}{(hold['fwd']*100).mean()-(sell['fwd']*100).mean():>+6.2f}%{'':>6}  p={pp:.4f} {'✅该减' if pp<0.05 and sell['fwd'].mean()<hold['fwd'].mean() else ''}")
    print()

# 减速见顶 f''<0 的前瞻收益(应该为负才确认该卖)
print(f"{'='*72}")
print(f"📋 减速见顶(f'>0,f''<0) 前瞻收益(负=确认该减):")
for reg in ['涨', '平', '跌']:
    g = D[(D.regime == reg) & ~D.f2p]
    if len(g) > 5:
        r = g['fwd'] * 100
        print(f"  {reg}期: {r.mean():+.2f}% (N={len(g)}, 胜率{(r>0).mean()*100:.0f}%) {'✅负=该减' if r.mean() < 0 else '⚠️正=不该减'}")

# f''从正转负的时点(更精确的卖出信号)
print(f"\n{'='*72}")
print(f"📋 f''拐点(从正转负) vs 持续正: 卖出信号精确化")
rows2 = []
for code in sample:
    p = f'{KDIR}/{code}/D.parquet'
    try: d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except: continue
    if len(d) < SG_W + H + 25: continue
    close = d['close'].astype(float).values
    ln_p = np.log(close)
    f1_raw = savgol_filter(ln_p, SG_W, SG_P, deriv=1) * 100; f1 = pd.Series(f1_raw).shift(SG_W // 2).values
    f2_raw = savgol_filter(ln_p, SG_W, SG_P, deriv=2) * 10000; f2 = pd.Series(f2_raw).shift(SG_W // 2).values
    for i in range(SG_W + 1, len(close) - H):
        if np.isnan(f1[i]) or np.isnan(f2[i]) or np.isnan(f2[i-1]): continue
        if f1[i] <= 0: continue  # 只看涨势
        # f''拐点: 前一日正, 今日负 = 刚转负
        turn = (f2[i-1] > 0 and f2[i] < 0)
        fwd = close[i + H] / close[i] - 1
        rows2.append(dict(fwd=fwd, turn=turn, f2val=f2[i]))
D2 = pd.DataFrame(rows2)
turn_g = D2[D2.turn]; noturn_g = D2[~D2.turn & (D2.f2val > 0)]
if len(turn_g) > 10 and len(noturn_g) > 10:
    tt, pp = stats.ttest_ind(turn_g['fwd']*100, noturn_g['fwd']*100, equal_var=False)
    print(f"  f''刚转负(拐点): {turn_g['fwd'].mean()*100:+.2f}% (N={len(turn_g)})")
    print(f"  f''持续正(持有): {noturn_g['fwd'].mean()*100:+.2f}% (N={len(noturn_g)})")
    print(f"  差: {turn_g['fwd'].mean()*100 - noturn_g['fwd'].mean()*100:+.2f}%, p={pp:.4f} {'✅拐点该减' if pp<0.05 else '⚠️不显著'}")
