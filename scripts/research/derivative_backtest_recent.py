#!/usr/bin/env python3
# ⚠️ SUPERSEDED 2026-09-08: 原roll(1)有savgol泄漏,已修为shift(SG_WINDOW//2)
# 正确结论: f2全无效(reverify_derivative_v2_conclusion)
"""二阶导回测(近期验证): 用S-G滤波器算f'/f'', 验证在2026年5-7月数据上是否仍有效。
v3原始回测(IC=+0.459)是5月数据, 现在验证7月暴跌期间是否还work。
无未来: 导数用截至当日, forward用t+H。
关键发现回测: f''>0+f'>0(主升浪) 应显著跑赢 f''<0+f'<0(加速下跌)。"""
import os, glob, random
import pandas as pd, numpy as np
from scipy.signal import savgol_filter
from scipy import stats

KDIR = './data/store/kline'
random.seed(42)
codes = [os.path.basename(d) for d in glob.glob(KDIR + '/*') if os.path.isdir(d)]
sample = random.sample(codes, min(300, len(codes)))
H = 10  # forward天数
SG_WINDOW = 21  # S-G窗口(约4周)
SG_POLY = 3

rows = []
for code in sample:
    p = f'{KDIR}/{code}/D.parquet'
    try:
        d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(d) < SG_WINDOW + H + 5:
        continue
    close = d['close'].astype(float).values
    ln_p = np.log(close)
    # S-G求导(防look-ahead: shift 1)
    f1 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=1) * 100
    f2 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=2) * 10000
    f1 = pd.Series(f1).shift(SG_WINDOW // 2).values  # 正确因果shift(原roll(1)只消1根,漏9根)
    f2 = pd.Series(f2).shift(SG_WINDOW // 2).values
    # 只取2026-05以后(近期)
    dates = d['date'].astype(str).values
    for i in range(SG_WINDOW, len(close) - H):
        if str(dates[i]) < '2026-05-01':
            continue
        if np.isnan(f1[i]) or np.isnan(f2[i]):
            continue
        fwd = close[i + H] / close[i] - 1
        rows.append(dict(f1=f1[i], f2=f2[i], fwd=fwd,
                         f1_pos=f1[i] > 0, f2_pos=f2[i] > 0))

D = pd.DataFrame(rows)
print(f"样本: {len(D)} 条 (2026-05以后, {len(sample)}股, forward{H}日)")

# 分组对比
groups = {
    '🚀主升浪(f>0,f">0)': D[D.f1_pos & D.f2_pos],
    '🔄底部拐点(f<0,f">0)': D[~D.f1_pos & D.f2_pos],
    '⚠️减速见顶(f>0,f"<0)': D[D.f1_pos & ~D.f2_pos],
    '🔴加速下跌(f<0,f"<0)': D[~D.f1_pos & ~D.f2_pos],
    'f">0 全部': D[D.f2_pos],
    'f"<0 全部': D[~D.f2_pos],
}

print(f"\n{'组':<24}{'N':>7}{'均值':>8}{'胜率':>7}{'t':>6}{'p':>9}")
print('-' * 64)
for name, g in groups.items():
    if len(g) < 5:
        print(f"{name:<24}{len(g):>7}  (样本不足)")
        continue
    rets = g['fwd'] * 100
    t, p = stats.ttest_1samp(rets, 0)
    print(f"{name:<24}{len(g):>7}{rets.mean():>+6.2f}%{(rets > 0).mean() * 100:>6.0f}%{t:>6.2f}{p:>8.4f}")

# 核心对比: 主升浪 vs 加速下跌
a = groups['🚀主升浪(f>0,f">0)']['fwd'] * 100
b = groups['🔴加速下跌(f<0,f"<0)']['fwd'] * 100
if len(a) > 5 and len(b) > 5:
    tt, pp = stats.ttest_ind(a, b, equal_var=False)
    print(f"\n核心对比: 主升浪 vs 加速下跌 → 差 {a.mean() - b.mean():+.2f}%, p={pp:.4f} ({'✅显著' if pp < 0.05 else '⚠️不显著'})")

# f''独立价值: 控制f'后, f''>0 vs f''<0
for f1_sign in [True, False]:
    sub = D[D.f1_pos == f1_sign]
    pos = sub[sub.f2_pos]['fwd'] * 100
    neg = sub[~sub.f2_pos]['fwd'] * 100
    if len(pos) > 5 and len(neg) > 5:
        tt, pp = stats.ttest_ind(pos, neg, equal_var=False)
        label = 'f>0(涨)' if f1_sign else 'f<0(跌)'
        print(f"控制{label}后: f\">0 {pos.mean():+.2f}% vs f\"<0 {neg.mean():+.2f}%, p={pp:.4f}")
