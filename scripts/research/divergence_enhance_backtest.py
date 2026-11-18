#!/usr/bin/env python3
"""B:量能确认底背离叠加位置/强度/趋势过滤,看能否把夏普0.71继续推高。
基础=量能阳线确认底背离(右侧确认,夏普0.71,p=0.0000,已验证)。
无未来: argrelextrema order=5 + 谷后首根放量阳线确认 + 收益shift(-hold)。
用法: cd stock && PYTHONPATH=. python scripts/research/divergence_enhance_backtest.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

STORE = ROOT / 'data/store/kline'
HOLDS = [5, 10, 20]
COST = 0.003
ORDER = 5


def signals(df):
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    high = df['high'].astype(float).ffill() if 'high' in df.columns else close
    low = df['low'].astype(float).ffill() if 'low' in df.columns else close
    op = df['open'].astype(float).ffill() if 'open' in df.columns else close
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    vol_ma = vol.rolling(20).mean()
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    hi120 = close.rolling(120).max(); dist120 = close / hi120 - 1
    ma60 = close.rolling(60).mean()

    out = pd.DataFrame({'date': df['date'], 'close': close.values, 'dist120': dist120.values})
    out['sig_vc'] = False       # 量能阳线确认底背离(基础)
    out['hist_div'] = 0.0       # 背离强度(hist抬升比例)
    tr_idx = argrelextrema(low.values, np.less, order=ORDER)[0]
    h_arr = hist.values; lo_arr = low.values; cl_arr = close.values
    op_arr = op.values; vol_arr = vol.values; vm_arr = vol_ma.values; d120 = dist120.values
    c_vc = out.columns.get_loc('sig_vc'); c_hd = out.columns.get_loc('hist_div')
    for k in range(1, len(tr_idx)):
        t2 = tr_idx[k]; t1 = tr_idx[k - 1]
        if lo_arr[t2] < lo_arr[t1] and h_arr[t2] > h_arr[t1]:   # 底背离
            hd = (h_arr[t2] - h_arr[t1]) / (abs(h_arr[t1]) + 0.001)
            for j in range(t2 + 1, min(t2 + ORDER + 1, len(out))):
                if cl_arr[j] > op_arr[j] and vm_arr[j] > 0 and vol_arr[j] > vm_arr[j]:
                    out.iat[j, c_vc] = True
                    out.iat[j, c_hd] = float(hd)
                    break
    # 叠加过滤
    out['sig_vc_low'] = out['sig_vc'] & (out['dist120'] < -0.15)            # 低位
    out['sig_vc_strong'] = out['sig_vc'] & (out['hist_div'] > 0.30)         # 强背离
    out['sig_vc_down'] = out['sig_vc'] & (out['close'] < ma60.values)       # 下行中(真底部)
    for hld in HOLDS:
        out[f'ret{hld}'] = (close.shift(-hld) / close - 1).values
    return out


def main():
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    print(f"B:量能确认背离叠加过滤 | {len(codes)}只 | 成本{COST*100:.1f}%")
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
    print(f"总快照: {len(snap):,}\n")

    groups = [('基准(全市场)', snap), ('量能确认背离', snap[snap.sig_vc]),
              ('+低位(距120高<-15%)', snap[snap.sig_vc_low]),
              ('+强背离(hist抬>30%)', snap[snap.sig_vc_strong]),
              ('+下行中(close<MA60)', snap[snap.sig_vc_down])]
    print(f"{'组':<24}{'N':>7}{'ret10净':>9}{'胜率':>6}{'夏普':>7}")
    print('-' * 56)
    for nm, g in groups:
        n = len(g); r = g['ret10']; rn = r - COST
        wr = (r > COST).mean() * 100 if n else 0
        sh = rn.mean() / rn.std() * np.sqrt(252 / 10) if n and rn.std() > 0 else 0
        print(f"{nm:<22}{n:>7,}{rn.mean()*100:>+8.2f}%{wr:>5.0f}%{sh:>7.2f}")

    base_vc = snap.loc[snap.sig_vc, 'ret10'].dropna() - COST
    print(f"\n[t检验 叠加过滤 vs 纯量能确认]")
    for nm, mask in [('+低位', snap.sig_vc_low), ('+强背离', snap.sig_vc_strong), ('+下行', snap.sig_vc_down)]:
        s = snap.loc[mask, 'ret10'].dropna() - COST
        if len(s) > 100:
            t, p = stats.ttest_ind(s, base_vc, equal_var=False)
            vd = '✅显著提升' if (p < 0.05 and s.mean() > base_vc.mean()) else '—无显著'
            print(f"  {nm}: 净{s.mean()*100:+.2f}% vs 纯{base_vc.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {vd} N={len(s)}")


if __name__ == '__main__':
    main()
