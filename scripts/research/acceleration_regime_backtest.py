#!/usr/bin/env python3
"""主升信号 × 大盘regime 回测 — 给 acceleration_scanner 的 regime gate 提供数据依据。

信号(对齐acceleration_scanner, 无未来): savgol(21,3)求ln(close)的f1/f2, shift(10)消前瞻
  主升 = f1>0(涨) + f2>0(加速) + 放量>1.2x + 不创新高(距120高<=0) + 价>2 + 不暴跌(>-75%)
regime: 上证20日涨跌 ±5% 三档
收益: ret5/10/20 扣0.3%

调研提示: market_regime_backtest 测过V3"趋势加速"回调期胜率29%。本脚本用统一20日涨跌法重测。

用法: cd stock && PYTHONPATH=. python scripts/research/acceleration_regime_backtest.py
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
SG_WINDOW, SG_POLY = 21, 3


def load_index_regime():
    p = STORE / '000001SH' / 'D.parquet'
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    for c in ['date', 'day', 'trade_date']:
        if c in df.columns:
            df = df.rename(columns={c: 'date'})
    df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['ret20'] = df['close'].pct_change(20)
    df['date_str'] = df['date'].dt.strftime('%Y%m%d')
    df['regime'] = df['ret20'].apply(lambda x: '上升' if x > 0.05 else ('下降' if x < -0.05 else '震荡'))
    return dict(zip(df['date_str'], df['regime']))


def compute_accel(df):
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    ln = np.log(close)
    try:
        f1 = pd.Series(savgol_filter(ln.values, SG_WINDOW, SG_POLY, deriv=1)).shift(SG_WINDOW // 2)
        f2 = pd.Series(savgol_filter(ln.values, SG_WINDOW, SG_POLY, deriv=2)).shift(SG_WINDOW // 2)
    except Exception:
        return None
    vol_ma = vol.rolling(20).mean()
    hi120 = close.rolling(120).max()
    dist120 = close / hi120 - 1
    sig = ((f1 > 0) & (f2 > 0) & (vol > vol_ma * 1.2) & (dist120 <= 0) & (close > 2) & (dist120 > -0.75)).fillna(False)
    out = pd.DataFrame({'date_str': pd.to_datetime(df['date']).dt.strftime('%Y%m%d'), 'sig_accel': sig})
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values
    return out


def main():
    print("主升信号 × 大盘regime 回测 | f1>0+f2>0+放量+不创新高 | 扣0.3%成本")
    regimes = load_index_regime()
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6')) and d.name != '000001SH'])
    print(f"扫描 {len(codes)} 只...")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = compute_accel(pd.read_parquet(STORE / c / 'D.parquet'))
            if s is not None:
                s['code'] = c; snaps.append(s)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True)
    snap['regime'] = snap['date_str'].map(regimes)
    snap = snap.dropna(subset=['ret10', 'regime'])
    sig = snap[snap.sig_accel].copy()
    print(f"主升信号总样本: {len(sig):,}\n")

    print(f"{'regime':<8}{'N':>8}{'ret5净':>9}{'ret10净':>9}{'ret20净':>9}{'胜率10':>7}{'夏普10':>8}")
    print('-' * 62)
    for reg in ['上升', '震荡', '下降']:
        g = sig[sig.regime == reg]
        if len(g) < 30:
            print(f"{reg:<8}{len(g):>8,}  样本不足"); continue
        r10 = g.ret10 - COST
        sh = r10.mean() / r10.std() * np.sqrt(252 / 10) if r10.std() > 0 else 0
        print(f"{reg:<8}{len(g):>8,}{(g.ret5-COST).mean()*100:>+8.2f}%{r10.mean()*100:>+8.2f}%{(g.ret20-COST).mean()*100:>+8.2f}%{(g.ret10>COST).mean()*100:>6.0f}%{sh:>8.2f}")
    r10 = sig.ret10 - COST
    sh = r10.mean() / r10.std() * np.sqrt(252 / 10) if r10.std() > 0 else 0
    print(f"{'全部':<8}{len(sig):>8,}{(sig.ret5-COST).mean()*100:>+8.2f}%{r10.mean()*100:>+8.2f}%{(sig.ret20-COST).mean()*100:>+8.2f}{(sig.ret10>COST).mean()*100:>6.0f}%{sh:>8.2f}")

    print(f"\n=== regime gate 效果(全部 vs 排除下降期) ===")
    no_down = sig[sig.regime != '下降']
    for h in HOLDS:
        full = sig[f'ret{h}'] - COST; gated = no_down[f'ret{h}'] - COST
        t, p = stats.ttest_ind(gated.dropna(), full.dropna(), equal_var=False)
        print(f"  ret{h}: 全部{full.mean()*100:+.2f}% vs gate后{gated.mean()*100:+.2f}% (排除{len(sig)-len(no_down)}样本) p={p:.4f} {'✅gate提升' if gated.mean()>full.mean() else '❌gate无效'}")
    sd = sig[sig.regime == '下降'].ret10.dropna() - COST
    su = sig[sig.regime == '上升'].ret10.dropna() - COST
    if len(sd) > 30 and len(su) > 30:
        t, p = stats.ttest_ind(sd, su, equal_var=False)
        print(f"\n=== t检验 下降{sd.mean()*100:+.2f}% vs 上升{su.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {'→下降期显著更差,gate成立' if sd.mean()<su.mean() and p<0.05 else '→无显著差异'}")


if __name__ == '__main__':
    main()
