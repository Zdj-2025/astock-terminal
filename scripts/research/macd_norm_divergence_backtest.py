#!/usr/bin/env python3
"""MACD柱ATR归一化改进回测 — 验证归一化是否让底背离信号从弱变强。

问题: 标准hist=2*(DIF-DEA)是绝对值, 10元股与100元股量级差10倍, 跨股票不可比 → 背离误判。
改进: hist_atr = hist / ATR14, 归一化后跨股票可比。
回测: 标准底背离 vs ATR归一化底背离, 扣0.3%成本, 看哪个alpha高。
无未来: argrelextrema用low(order=5局部极值), 收益shift(-hold)。

用法: cd stock && PYTHONPATH=. python scripts/research/macd_norm_divergence_backtest.py
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
ORDER = 5   # argrelextrema 局部极值阶数


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
    # MACD(标准)
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    # ATR14 归一化
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    hist_atr = hist / atr
    # 价格谷(argrelextrema on low)
    out = pd.DataFrame({'date': df['date'], 'hist': hist.values, 'hist_atr': hist_atr.values, 'low': low.values})
    out['sig_std'] = False       # centered: ⚠含未来函数(谷需未来ORDER根确认)
    out['sig_atr'] = False
    out['sig_std_rt'] = False    # 右侧确认: 无未来, 信号在 t2+ORDER(谷确认后)
    out['sig_atr_rt'] = False
    out['sig_std_vc'] = False    # 量能阳线确认: 谷后首根放量阳线买(减少右侧滞后)
    tr_idx = argrelextrema(low.values, np.less, order=ORDER)[0]
    h_arr = hist.values; ha_arr = hist_atr.values; lo_arr = low.values
    cl_arr = close.values; op_arr = op.values; vol_arr = vol.values; vm_arr = vol_ma.values
    c_std = out.columns.get_loc('sig_std'); c_atr = out.columns.get_loc('sig_atr')
    c_std_rt = out.columns.get_loc('sig_std_rt'); c_atr_rt = out.columns.get_loc('sig_atr_rt')
    c_std_vc = out.columns.get_loc('sig_std_vc')
    for k in range(1, len(tr_idx)):
        t2 = tr_idx[k]; t1 = tr_idx[k - 1]
        if lo_arr[t2] < lo_arr[t1]:   # 价格新低
            std_ok = h_arr[t2] > h_arr[t1]
            atr_ok = not (np.isnan(ha_arr[t2]) or np.isnan(ha_arr[t1])) and ha_arr[t2] > ha_arr[t1]
            out.iat[t2, c_std] = std_ok              # centered(未来函数)
            out.iat[t2, c_atr] = atr_ok
            rt = t2 + ORDER                          # 右侧确认: 谷需未来ORDER根形成
            if rt < len(out):
                out.iat[rt, c_std_rt] = std_ok       # 确认日才知t2是谷 → 无未来
                out.iat[rt, c_atr_rt] = atr_ok
            # 量能阳线确认: 谷后[1,ORDER]内首根放量阳线(启动确认,更早买入)
            if std_ok:
                for j in range(t2 + 1, min(t2 + ORDER + 1, len(out))):
                    if cl_arr[j] > op_arr[j] and vm_arr[j] > 0 and vol_arr[j] > vm_arr[j]:
                        out.iat[j, c_std_vc] = True
                        break
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values
    return out


def main():
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    print(f"MACD柱ATR归一化底背离回测 | {len(codes)}只 | 成本{COST*100:.1f}%")
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

    groups = [('基准(全市场)', snap),
              ('标准底背离[centered⚠未来函数]', snap[snap.sig_std]),
              ('标准底背离[右侧确认·可复现]', snap[snap.sig_std_rt]),
              ('标准底背离[量能阳线确认]', snap[snap.sig_std_vc]),
              ('ATR归一化[右侧确认]', snap[snap.sig_atr_rt])]
    print(f"{'组':<34}{'N':>8}{'ret10毛':>9}{'ret10净':>9}{'胜率净':>7}{'夏普净':>8}")
    print('-' * 78)
    for nm, g in groups:
        n = len(g); r = g['ret10']; rnet = r - COST
        wr = (r > COST).mean() * 100 if n else 0
        sh = rnet.mean() / rnet.std() * np.sqrt(252 / 10) if n and rnet.std() > 0 else 0
        print(f"{nm:<30}{n:>8,}{r.mean()*100:>+8.2f}%{rnet.mean()*100:>+8.2f}%{wr:>6.0f}%{sh:>8.2f}")

    # 关键: 计算方法改进(量能阳线确认)是否显著有效
    vc = snap.loc[snap.sig_std_vc, 'ret10'].dropna() - COST
    rt_s = snap.loc[snap.sig_std_rt, 'ret10'].dropna() - COST
    base_r = snap['ret10'].dropna() - COST
    print(f"\n[t检验 扣成本]")
    for nm, s, ref in [('量能阳线确认 vs 基准', vc, base_r), ('量能确认 vs 右侧死等', vc, rt_s)]:
        if len(s) > 100 and len(ref) > 100:
            t, p = stats.ttest_ind(s, ref, equal_var=False)
            vd = '✅显著更优' if (p < 0.05 and s.mean() > ref.mean()) else '—无显著差异'
            print(f"  {nm}: 净{s.mean()*100:+.2f}% vs {ref.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {vd} N={len(s)}")

    print(f"\n[量能阳线确认底背离 不同持有期]")
    g = snap[snap.sig_std_vc]
    for h in HOLDS:
        r = g[f'ret{h}'].dropna() - COST
        wr = (g[f'ret{h}'] > COST).mean() * 100
        sh = r.mean() / r.std() * np.sqrt(252 / h) if r.std() > 0 else 0
        print(f"  持{h}日: 净{r.mean()*100:+.2f}% 胜率{wr:.0f}% 夏普{sh:.2f} N={len(r)}")


if __name__ == '__main__':
    main()
