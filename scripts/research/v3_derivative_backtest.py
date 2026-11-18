#!/usr/bin/env python3
"""V3二阶导信号扣成本回测 — 验证f''>0(加速度转正)在扣0.3%成本后是否还有alpha。

V3体系:
  f1 = 一阶导(趋势方向) — savgol对ln(price)求导
  f2 = 二阶导(加速度) — savgol二阶导
  信号: 🔄底部拐点 f1<0但f2>0(减速下跌,接近底) / 🚀主升 f1>0且f2>0

回测: f2>0 vs f2<0 / 底部拐点(f1<0+f2>0) / 主升(f1>0+f2>0) 分组,扣成本看收益。
无未来: savgol后shift(window//2)消除前瞻,收益shift(-hold)。
用法: cd stock && PYTHONPATH=. python scripts/research/v3_derivative_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

STORE = Path(__file__).resolve().parent.parent.parent / 'data/store/kline'
HOLDS = [5, 10, 20]
COST = 0.003
SG_WINDOW = 21
SG_POLY = 3


def compute_v3(df):
    """算V3二阶导信号 + 未来收益。无未来(savgol后shift消除前瞻)。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    ln = np.log(close)
    # savgol求导
    try:
        f1_raw = savgol_filter(ln.values, SG_WINDOW, SG_POLY, deriv=1)
        f2_raw = savgol_filter(ln.values, SG_WINDOW, SG_POLY, deriv=2)
    except Exception:
        return None
    # shift消除前瞻(savgol是centered,含未来window//2根)
    shift = SG_WINDOW // 2
    f1 = pd.Series(f1_raw).shift(shift)
    f2 = pd.Series(f2_raw).shift(shift)
    out = pd.DataFrame({'date': df['date'], 'f1': f1.values * 100, 'f2': f2.values * 10000})
    # 信号
    out['sig_f2_pos'] = (f2 > 0).fillna(False)           # 加速度>0(底部拐点或主升加速)
    out['sig_reversal'] = ((f1 < 0) & (f2 > 0)).fillna(False)  # 底部拐点: 跌但减速
    out['sig_accel'] = ((f1 > 0) & (f2 > 0)).fillna(False)     # 主升: 涨且加速
    out['sig_decel'] = ((f1 > 0) & (f2 < 0)).fillna(False)     # 减速见顶: 涨但减速
    # 低位过滤(距120日高<-15%)
    hi120 = close.rolling(120).max()
    out['low_pos'] = ((close / hi120 - 1) < -0.15).fillna(False)
    # 未来收益
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values
    return out


def main():
    print(f"V3二阶导回测 | savgol({SG_WINDOW},{SG_POLY}) 扣成本{COST*100:.1f}%")
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
    print(f"扫描 {len(codes)} 只...")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = compute_v3(pd.read_parquet(STORE / c / 'D.parquet'))
            if s is not None:
                s['code'] = c
                snaps.append(s)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True).dropna(subset=['ret10', 'f2'])
    print(f"总快照: {len(snap):,}\n")

    # 分组
    groups = [
        ('基准(全部)', snap),
        ('f2>0(加速度正)', snap[snap.sig_f2_pos]),
        ('└+低位(距120高<-15%)', snap[snap.sig_f2_pos & snap.low_pos]),
        ('底部拐点(f1<0+f2>0)', snap[snap.sig_reversal]),
        ('└+低位', snap[snap.sig_reversal & snap.low_pos]),
        ('主升(f1>0+f2>0)', snap[snap.sig_accel]),
        ('减速见顶(f1>0+f2<0)', snap[snap.sig_decel]),
    ]
    print(f"{'组':<28}{'N':>8}{'ret10毛':>9}{'ret10净':>9}{'胜率':>6}{'夏普':>7}")
    print('-' * 72)
    for nm, g in groups:
        n = len(g)
        if n < 100:
            print(f"{nm:<26}{n:>8,}  (样本不足)"); continue
        r = g['ret10']; rnet = r - COST
        wr = (r > COST).mean() * 100 if n else 0
        sh = rnet.mean() / rnet.std() * np.sqrt(252 / 10) if n and rnet.std() > 0 else 0
        print(f"{nm:<26}{n:>8,}{r.mean()*100:>+8.2f}%{rnet.mean()*100:>+8.2f}%{wr:>5.0f}%{sh:>7.2f}")

    # t检验
    base = snap['ret10'].dropna() - COST
    print(f"\n[t检验 ret10净]")
    for nm, mask in [('f2>0 vs 基准', snap.sig_f2_pos),
                     ('底部拐点+低位 vs 基准', snap.sig_reversal & snap.low_pos),
                     ('主升 vs 基准', snap.sig_accel),
                     ('减速见顶 vs 基准', snap.sig_decel)]:
        s = snap.loc[mask, 'ret10'].dropna() - COST
        if len(s) > 100:
            t, p = stats.ttest_ind(s, base, equal_var=False)
            verdict = '✅显著更优' if (p < 0.05 and s.mean() > base.mean()) else \
                      ('⚠️显著更差' if (p < 0.05 and s.mean() < base.mean()) else '—无显著差异')
            print(f"  {nm}: 净{s.mean()*100:+.2f}% vs {base.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {verdict} N={len(s)}")

    # f2强度分位(看f2越强收益是否越好)
    print(f"\n[f2强度分位]")

    snap_valid = snap.dropna(subset=['f2'])
    snap_valid['f2_pct'] = pd.qcut(snap_valid['f2'].rank(method='first'), 5, labels=['Q1最弱', 'Q2', 'Q3', 'Q4', 'Q5最强'])
    print(f"{'f2分位':<10}{'N':>8}{'ret10净':>9}{'夏普':>7}")
    for q in ['Q1最弱', 'Q2', 'Q3', 'Q4', 'Q5最强']:
        g = snap_valid[snap_valid.f2_pct == q]
        if len(g) < 100: continue
        rnet = g['ret10'] - COST
        sh = rnet.mean() / rnet.std() * np.sqrt(252 / 10) if rnet.std() > 0 else 0
        print(f"{q:<10}{len(g):>8,}{rnet.mean()*100:>+8.2f}%{sh:>7.2f}")

    # 不同持有期(底部拐点+低位)
    print(f"\n[底部拐点+低位 不同持有期]")
    g = snap[snap.sig_reversal & snap.low_pos]
    if len(g) >= 100:
        for h in HOLDS:
            r = g[f'ret{h}'].dropna() - COST
            wr = (g[f'ret{h}'] > COST).mean() * 100
            sh = r.mean() / r.std() * np.sqrt(252 / h) if r.std() > 0 else 0
            print(f"  持{h}日: 净{r.mean()*100:+.2f}% 胜率{wr:.0f}% 夏普{sh:.2f} N={len(r)}")


if __name__ == '__main__':
    main()
