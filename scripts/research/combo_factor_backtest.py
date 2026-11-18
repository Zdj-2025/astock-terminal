#!/usr/bin/env python3
"""组合因子回测 — 突破+非低位 × f''二阶导 × MTF简化, 验证扣成本后是否还赚。

基于 backtest_accumulation.py 向量回测模式, 全市场日线, 加交易成本。
回答核心问题: 有效因子组合后, 扣掉0.3%往返成本, 还能否显著跑赢基准?
无未来: 信号 rolling 不 center / shift(1); f'' savgol 后 shift(win//2) 消前瞻; 收益 shift(-hold)。

用法: cd stock && PYTHONPATH=. python scripts/research/combo_factor_backtest.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

STORE = ROOT / 'data/store/kline'
HOLDS = [5, 10, 20]
COST = 0.003  # 往返成本 0.3%(佣金0.025%×2+印花税0.1%+过户+滑点0.1%)


def signals(df):
    """单股: 突破+非低位 / f''二阶导 / MTF简化 + 未来收益。无未来。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    vol = df['volume'].astype(float).ffill()
    ln = np.log(close)

    # ── 因子1: 突破+非低位(平台+新高+放量+距120高<-10%) ──
    hi20 = close.rolling(20).max(); lo20 = close.rolling(20).min()
    platform = ((hi20 / lo20 - 1) < 0.15).shift(1)           # 前一日处平台
    prev_hi = close.shift(1).rolling(20).max()
    breakout = close > prev_hi
    vol_surge = vol > vol.rolling(20).mean() * 1.5
    hi120 = close.rolling(120).max(); dist120 = close / hi120 - 1
    sig_brk_low = (platform.fillna(False) & breakout.fillna(False)
                   & vol_surge.fillna(False) & (dist120 < -0.10).fillna(False))

    # ── 因子2: f'' 二阶导(加速度>0), savgol 无未来 ──
    win = 21
    try:
        f2_raw = savgol_filter(ln.values, win, 3, deriv=2) * 10000
        f2 = pd.Series(f2_raw, index=close.index).shift(win // 2)   # 消前瞻
    except Exception:
        f2 = pd.Series(np.nan, index=close.index)
    sig_acc = (f2 > 0).fillna(False)

    # ── 因子3: MTF简化(日线多周期趋势一致) 价格>MA20 & MA20>MA60 & MACD金叉 ──
    ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    dif = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    dea = dif.ewm(span=9).mean()
    sig_mtf = ((close > ma20) & (ma20 > ma60) & (dif > dea)).fillna(False)

    out = pd.DataFrame({'date': df['date'],
                        'sig_brk_low': sig_brk_low, 'sig_acc': sig_acc, 'sig_mtf': sig_mtf})
    out['sig_combo2'] = sig_brk_low & sig_acc          # 突破+非低 + f''
    out['sig_combo3'] = sig_brk_low & sig_acc & sig_mtf  # 三因子
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1)
    return out


def main():
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    print(f"组合因子回测 | {len(codes)}只 | 往返成本 {COST*100:.1f}% | 数据截至最新日线")
    snaps = []
    for i, c in enumerate(codes):
        try:
            df = pd.read_parquet(STORE / c / 'D.parquet')
            s = signals(df)
            if s is not None:
                s['code'] = c
                snaps.append(s)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True).dropna(subset=['ret10'])
    base = snap['ret10'].dropna()
    print(f"总快照: {len(snap):,}\n")

    groups = [
        ('基准(全市场)', snap),
        ('突破+非低位', snap[snap.sig_brk_low]),
        ('+f二阶导', snap[snap.sig_combo2]),
        ('+f+MTF(三因子)', snap[snap.sig_combo3]),
    ]
    print(f"{'组':<16}{'N':>9}{'ret10毛':>9}{'ret10净':>9}{'胜率净':>7}{'夏普净':>8}")
    print('-' * 62)
    for nm, g in groups:
        n = len(g); r = g['ret10']; rnet = r - COST
        wr = (r > COST).mean() * 100
        sharpe = rnet.mean() / rnet.std() * np.sqrt(252 / 10) if rnet.std() > 0 else 0
        print(f"{nm:<14}{n:>9,}{r.mean()*100:>+8.2f}%{rnet.mean()*100:>+8.2f}%{wr:>6.0f}%{sharpe:>8.2f}")

    print(f"\n[t检验 ret10 扣成本后 vs 基准]")
    base_net = base - COST
    for nm, mask in [('+f二阶导', snap.sig_combo2), ('+f+MTF', snap.sig_combo3), ('突破+非低', snap.sig_brk_low)]:
        s = snap.loc[mask, 'ret10'].dropna() - COST
        if len(s) < 100:
            print(f"  [{nm}] N={len(s)} 不足"); continue
        t, p = stats.ttest_ind(s, base_net, equal_var=False)
        verdict = '✅扣成本仍显著为正且跑赢' if (p < 0.05 and s.mean() > 0 and s.mean() > base_net.mean()) else \
                  ('⚠️扣成本后不赚' if s.mean() <= 0 else '—无显著优势')
        print(f"  {nm:<10}: t={t:+.2f} p={p:.4f} 净{s.mean()*100:+.2f}% vs 基准净{base_net.mean()*100:+.2f}%  {verdict}  N={len(s)}")

    # 多持有期对比(三因子)
    g3 = snap[snap.sig_combo3]
    if len(g3) >= 100:
        print(f"\n[三因子 不同持有期 净收益]")
        for h in HOLDS:
            r = g3[f'ret{h}'].dropna() - COST
            wr = (g3[f'ret{h}'] > COST).mean() * 100
            sh = r.mean() / r.std() * np.sqrt(252 / h) if r.std() > 0 else 0
            print(f"  持{h}日: 净{r.mean()*100:+.2f}% 胜率{wr:.0f}% 夏普{sh:.2f} N={len(r)}")


if __name__ == '__main__':
    main()
