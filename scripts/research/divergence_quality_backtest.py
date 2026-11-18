#!/usr/bin/env python3
"""底背离股性回测 — 滚动历史胜率(无未来) × 分组收益比较。

股性 = 该股历史上底背离confirmed信号后的胜率(只用之前信号算,不含当前)。
验证: 高股性(历史底背离后大概率涨)的股票,当前底背离信号是否更可靠?

无未来: 对信号t_k,只用t_1...t_{k-1}的ret10算胜率(至少3个历史信号)。
用法: cd stock && PYTHONPATH=. python scripts/research/divergence_quality_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

STORE = Path(__file__).resolve().parent.parent.parent / 'data/store/kline'
COST = 0.003
ORDER = 5
MIN_HIST = 3  # 至少3个历史信号才算"有股性"


def compute_signals(df):
    """底背离confirmed(量能阳线确认)信号 + ret10。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    low = df['low'].astype(float).ffill() if 'low' in df.columns else close
    op = df['open'].astype(float).ffill() if 'open' in df.columns else close
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    vol_ma = vol.rolling(20).mean()
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    out = pd.DataFrame({'date': df['date'], 'sig': False})
    tr_idx = argrelextrema(low.values, np.less, order=ORDER)[0]
    h_arr = hist.values; lo_arr = low.values
    cl_arr = close.values; op_arr = op.values; vol_arr = vol.values; vm_arr = vol_ma.values
    c_sig = out.columns.get_loc('sig')
    for k in range(1, len(tr_idx)):
        t2 = tr_idx[k]; t1 = tr_idx[k - 1]
        if lo_arr[t2] < lo_arr[t1] and h_arr[t2] > h_arr[t1]:
            for j in range(t2 + 1, min(t2 + ORDER + 1, len(out))):
                if cl_arr[j] > op_arr[j] and vm_arr[j] > 0 and vol_arr[j] > vm_arr[j]:
                    out.iat[j, c_sig] = True
                    break
    out['ret10'] = (close.shift(-10) / close - 1).values
    return out[out['sig']].dropna(subset=['ret10']).copy()


def main():
    print("底背离股性回测 | 滚动历史胜率(无未来) × 分组收益")
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
    print(f"扫描 {len(codes)} 只...")
    records = []
    for i, c in enumerate(codes):
        try:
            sigs = compute_signals(pd.read_parquet(STORE / c / 'D.parquet'))
            if sigs is None or len(sigs) < (MIN_HIST + 1):
                continue
            sigs = sigs.sort_values('date').reset_index(drop=True)
            sigs['code'] = c
            # 滚动历史胜率(不含当前)
            hist_wr = []; hist_mean = []
            for k in range(len(sigs)):
                if k < MIN_HIST:
                    hist_wr.append(np.nan); hist_mean.append(np.nan)
                else:
                    prev = sigs['ret10'].iloc[:k]
                    hist_wr.append((prev > 0).mean())
                    hist_mean.append(prev.mean())
            sigs['hist_winrate'] = hist_wr
            sigs['hist_mean'] = hist_mean
            records.append(sigs)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")

    snap = pd.concat(records, ignore_index=True).dropna(subset=['hist_winrate'])
    print(f"有股性的信号(≥{MIN_HIST}个历史): {len(snap):,}\n")

    # 按历史胜率分组
    snap['quality_group'] = pd.cut(snap['hist_winrate'],
        bins=[-0.01, 0.3, 0.45, 0.55, 1.01],
        labels=['胜率<30%(差)', '30-45%', '45-55%', '≥55%(好)'])
    print(f"{'股性组':<14}{'N':>8}{'ret10净':>9}{'胜率':>6}{'夏普':>7}{'历史均值':>8}")
    print('-' * 56)
    for grp in ['胜率<30%(差)', '30-45%', '45-55%', '≥55%(好)']:
        g = snap[snap.quality_group == grp]
        n = len(g)
        if n < 30:
            print(f"{grp:<12}{n:>8,}  (样本不足)"); continue
        r = g['ret10'] - COST
        wr = (g['ret10'] > COST).mean() * 100
        sh = r.mean() / r.std() * np.sqrt(252 / 10) if r.std() > 0 else 0
        hm = g['hist_mean'].mean()
        print(f"{grp:<12}{n:>8,}{r.mean()*100:>+8.2f}%{wr:>5.0f}%{sh:>7.2f}{hm*100:>+7.1f}%")

    # t检验: 高股性vs低股性
    high = snap[snap.hist_winrate >= 0.55]['ret10'] - COST
    low = snap[snap.hist_winrate < 0.40]['ret10'] - COST
    if len(high) > 50 and len(low) > 50:
        t, p = stats.ttest_ind(high, low, equal_var=False)
        print(f"\n[t检验 股性好(≥55%) vs 差(<40%)]")
        print(f"  好净{high.mean()*100:+.2f}% vs 差净{low.mean()*100:+.2f}% t={t:+.2f} p={p:.4f}")
        verdict = '✅股性有效(高股性显著更优)' if (p < 0.05 and high.mean() > low.mean()) else \
                  ('⚠️股性反向(低股性反而好!)' if (p < 0.05 and low.mean() > high.mean()) else '—无显著差异(股性无效)')
        print(f"  {verdict}")

    # 按历史均值分组
    snap['mean_group'] = pd.cut(snap['hist_mean'],
        bins=[-9.99, 0, 0.01, 0.03, 9.99],
        labels=['历史均值<0(亏)', '0-1%', '1-3%', '>3%(好)'])
    print(f"\n[按历史均值分组]")
    print(f"{'组':<14}{'N':>8}{'ret10净':>9}{'夏普':>7}")
    for grp in ['历史均值<0(亏)', '0-1%', '1-3%', '>3%(好)']:
        g = snap[snap.mean_group == grp]
        if len(g) < 30: continue
        r = g['ret10'] - COST
        sh = r.mean() / r.std() * np.sqrt(252 / 10) if r.std() > 0 else 0
        print(f"{grp:<12}{len(g):>8,}{r.mean()*100:>+8.2f}%{sh:>7.2f}")


if __name__ == '__main__':
    main()
