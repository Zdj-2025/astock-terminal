#!/usr/bin/env python3
"""多层确认底背离回测 — 大盘好/坏环境 + 双MACD参数共振。

验证3个问题:
1. 量能阳线确认底背离(现有mainline用的) 在大盘好 vs 坏的表现差异?
2. 加快速MACD(5,34,5)柱双参数共振 是否提升信号?
3. 大盘坏时 底背离信号是否该降权/标注风险?

信号(全日线,无未来,argrelextrema右侧确认,收益shift):
- sig_base: 量能阳线确认底背离(现有mainline用的核心信号)
- sig_dual: sig_base + 快速MACD(5,34,5)柱也背离(双参数=多灵敏度,代理分钟共振)
大盘环境: 上证000001近20日涨跌 >0=好, <0=坏
用法: cd stock && PYTHONPATH=. python scripts/research/multi_confirm_regime_backtest.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter, argrelextrema

STORE = ROOT / 'data/store/kline'
COST = 0.003
ORDER = 5


def load_index_returns():
    """取上证000001的20日涨跌(大盘环境标签)。"""
    # 优先sh000001, 次sz399001
    for idx in ['000001SH', '000001']:
        p = STORE / idx / 'D.parquet'
        if p.exists():
            df = pd.read_parquet(p)
            for c in ['date', 'day', 'trade_date', 'datetime']:
                if c in df.columns:
                    df = df.rename(columns={c: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['idx_ret20'] = df['close'].pct_change(20)
            return df[['date', 'idx_ret20']].dropna()
    return None


def signals(df):
    """单股: 量能确认底背离(sig_base) + 双MACD共振(sig_dual) + 未来收益。无未来。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    high = df['high'].astype(float).ffill() if 'high' in df.columns else close
    low = df['low'].astype(float).ffill() if 'low' in df.columns else close
    op = df['open'].astype(float).ffill() if 'open' in df.columns else close
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    vol_ma = vol.rolling(20).mean()
    # 标准MACD
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    # 快速MACD(5,34,5) — 多灵敏度代理
    dif2 = close.ewm(span=5, adjust=False).mean() - close.ewm(span=34, adjust=False).mean()
    dea2 = dif2.ewm(span=5, adjust=False).mean()
    hist2 = 2 * (dif2 - dea2)

    out = pd.DataFrame({'date': df['date']})
    out['sig_base'] = False
    out['sig_dual'] = False
    tr_idx = argrelextrema(low.values, np.less, order=ORDER)[0]
    h_arr = hist.values; h2_arr = hist2.values; lo_arr = low.values
    cl_arr = close.values; op_arr = op.values; vol_arr = vol.values; vm_arr = vol_ma.values
    c_base = out.columns.get_loc('sig_base'); c_dual = out.columns.get_loc('sig_dual')
    for k in range(1, len(tr_idx)):
        t2 = tr_idx[k]; t1 = tr_idx[k - 1]
        if lo_arr[t2] < lo_arr[t1]:  # 价格新低
            std_ok = h_arr[t2] > h_arr[t1]  # 标准MACD柱背离
            dual_ok = not (np.isnan(h2_arr[t2]) or np.isnan(h2_arr[t1])) and h2_arr[t2] > h2_arr[t1]  # 快速MACD也背离
            if std_ok:
                for j in range(t2 + 1, min(t2 + ORDER + 1, len(out))):
                    if cl_arr[j] > op_arr[j] and vm_arr[j] > 0 and vol_arr[j] > vm_arr[j]:
                        out.iat[j, c_base] = True  # 量能阳线确认
                        if dual_ok:
                            out.iat[j, c_dual] = True  # 双参数共振
                        break
    for h in [5, 10, 20]:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values
    return out


def main():
    idx_ret = load_index_returns()
    if idx_ret is None:
        print("无法加载上证数据"); return
    print(f"多层确认底背离回测 | 大盘好/坏环境 + 双MACD共振 | 成本{COST*100:.1f}%")

    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
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
    # join大盘环境
    snap['date'] = pd.to_datetime(snap['date'])
    snap = snap.merge(idx_ret, on='date', how='left')
    snap['market_good'] = snap['idx_ret20'] > 0  # 大盘好(上证20日涨)
    snap['market_bad'] = snap['idx_ret20'] <= 0   # 大盘坏(上证20日跌)
    print(f"总快照: {len(snap):,} | 大盘好: {snap['market_good'].sum():,} | 大盘坏: {snap['market_bad'].sum():,}\n")

    # 分组
    groups = [
        ('基准', snap),
        ('量能确认底背离(全部)', snap[snap.sig_base]),
        ('└大盘好时', snap[snap.sig_base & snap.market_good]),
        ('└大盘坏时', snap[snap.sig_base & snap.market_bad]),
        ('双MACD共振(标准+快速)', snap[snap.sig_dual]),
        ('└大盘好时', snap[snap.sig_dual & snap.market_good]),
        ('└大盘坏时', snap[snap.sig_dual & snap.market_bad]),
    ]
    print(f"{'组':<28}{'N':>8}{'ret10净':>9}{'胜率':>6}{'夏普':>7}")
    print('-' * 62)
    for nm, g in groups:
        n = len(g); r = g['ret10']; rnet = r - COST
        wr = (r > COST).mean() * 100 if n else 0
        sh = rnet.mean() / rnet.std() * np.sqrt(252 / 10) if n and rnet.std() > 0 else 0
        print(f"{nm:<26}{n:>8,}{rnet.mean()*100:>+8.2f}%{wr:>5.0f}%{sh:>7.2f}")

    # t检验
    base = snap.loc[snap.sig_base, 'ret10'].dropna() - COST
    print(f"\n[t检验 ret10净]")
    for nm, mask in [('量能确认 vs 基准', snap.sig_base),
                     ('双MACD共振 vs 量能确认', snap.sig_dual),
                     ('大盘好 vs 大盘坏(量能确认)',
                      snap.sig_base & snap.market_good)]:
        s = snap.loc[mask, 'ret10'].dropna() - COST
        if len(s) > 100:
            if '大盘好 vs 大盘坏' in nm:
                s_bad = snap.loc[snap.sig_base & snap.market_bad, 'ret10'].dropna() - COST
                t, p = stats.ttest_ind(s, s_bad, equal_var=False)
                print(f"  {nm}: 好净{s.mean()*100:+.2f}% vs 坏净{s_bad.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} N_good={len(s)} N_bad={len(s_bad)}")
            else:
                ref = snap['ret10'].dropna() - COST if '基准' in nm else base
                t, p = stats.ttest_ind(s, ref, equal_var=False)
                print(f"  {nm}: 净{s.mean()*100:+.2f}% vs {ref.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} N={len(s)}")


if __name__ == '__main__':
    main()
