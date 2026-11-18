#!/usr/bin/env python3
"""C:突破信号+右侧确认(次日不回落=真突破), 测能否从无效(p=0.58)变有效。
基础突破=平台+前20日高+放量(backtest_accumulation同款); 无未来用shift(1)。
右侧确认: 突破日t, 次日t+1 close>close[t](不回落=真突破), 信号在t+1, 收益从t+1算。
用法: cd stock && PYTHONPATH=. python scripts/research/breakout_confirm_backtest.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

STORE = ROOT / 'data/store/kline'
HOLDS = [5, 10, 20]
COST = 0.003


def signals(df):
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    hi20 = close.rolling(20).max(); lo20 = close.rolling(20).min()
    platform = ((hi20 / lo20 - 1) < 0.15).shift(1)            # 前一日处平台
    prev_hi = close.shift(1).rolling(20).max()                # 前20日高(不含当日)
    breakout = close > prev_hi
    vol_surge = vol > vol.rolling(20).mean() * 1.5
    hi120 = close.rolling(120).max(); dist120 = close / hi120 - 1

    sig_brk = (platform.fillna(False) & breakout.fillna(False) & vol_surge.fillna(False))   # 基础突破
    sig_brk_low = sig_brk & (dist120 < -0.10).fillna(False)                                 # +低位
    # 右侧确认: 突破日t的信号shift到t+1, 且t+1不回落(close[t+1]>close[t]) → 信号在确认日
    next_up = (close > close.shift(1)).fillna(False)
    sig_brk_conf = sig_brk.shift(1).fillna(False) & next_up                                 # 突破+次日确认
    sig_brk_low_conf = sig_brk_low.shift(1).fillna(False) & next_up                         # +低位+确认

    out = pd.DataFrame({'date': df['date'], 'sig_brk': sig_brk.values, 'sig_brk_low': sig_brk_low.values,
                        'sig_brk_conf': sig_brk_conf.values, 'sig_brk_low_conf': sig_brk_low_conf.values})
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values
    return out


def main():
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    print(f"C:突破+右侧确认 | {len(codes)}只 | 成本{COST*100:.1f}%")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = signals(pd.read_parquet(STORE / c / 'D.parquet'))
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

    groups = [('基准', snap), ('突破(基础)', snap[snap.sig_brk]), ('突破+低位', snap[snap.sig_brk_low]),
              ('突破+右侧确认', snap[snap.sig_brk_conf]), ('突破+低位+右侧确认', snap[snap.sig_brk_low_conf])]
    print(f"{'组':<22}{'N':>8}{'ret10净':>9}{'胜率':>6}{'夏普':>7}")
    print('-' * 55)
    for nm, g in groups:
        n = len(g); r = g['ret10']; rn = r - COST
        wr = (r > COST).mean() * 100 if n else 0
        sh = rn.mean() / rn.std() * np.sqrt(252 / 10) if n and rn.std() > 0 else 0
        print(f"{nm:<20}{n:>8,}{rn.mean()*100:>+8.2f}%{wr:>5.0f}%{sh:>7.2f}")

    print(f"\n[t检验 ret10扣成本 vs 基准]")
    for nm, mask in [('突破+低位+确认', snap.sig_brk_low_conf), ('突破+右侧确认', snap.sig_brk_conf),
                     ('突破+低位', snap.sig_brk_low)]:
        s = snap.loc[mask, 'ret10'].dropna() - COST
        if len(s) > 100:
            t, p = stats.ttest_ind(s, base - COST, equal_var=False)
            vd = '✅扣成本显著有效' if (p < 0.05 and s.mean() > 0 and s.mean() > (base - COST).mean()) else \
                 ('⚠️扣成本不赚' if s.mean() <= 0 else '—无显著优势')
            print(f"  {nm}: 净{s.mean()*100:+.2f}% vs 基准{(base-COST).mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {vd} N={len(s)}")


if __name__ == '__main__':
    main()
