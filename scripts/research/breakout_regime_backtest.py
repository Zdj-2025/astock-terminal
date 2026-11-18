#!/usr/bin/env python3
"""突破信号 × 大盘regime 回测 — 给 breakout_scanner 的 regime gate 提供数据依据。

缺口(2026-07-09调研): 突破信号×regime从未测过。本脚本补上。
信号(对齐breakout_scanner, 无未来): 平台(前20日振幅<15%) + 突破(当日收>前20日最高) + 放量(>20日均量×1.5) + 非高位(距120高<-5%)
regime(对齐sell_check被验证口径): 上证000001近20日涨跌, >5%=上升 / <-5%=下降 / 中间=震荡
收益: ret5/10/20, 扣0.3%成本。

预期假设: 突破在下降期失效(假突破多) → 若验证, breakout_scanner下降期gate成立。

用法: cd stock && PYTHONPATH=. python scripts/research/breakout_regime_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from scipy import stats

STORE = Path(__file__).resolve().parent.parent.parent / 'data/store/kline'
HOLDS = [5, 10, 20]
COST = 0.003


def load_index_regime():
    """上证20日涨跌三档regime。返回 {date_str: regime}。"""
    p = STORE / '000001SH' / 'D.parquet'
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    for c in ['date', 'day', 'trade_date']:
        if c in df.columns:
            df = df.rename(columns={c: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['ret20'] = df['close'].pct_change(20)
    df['date_str'] = df['date'].dt.strftime('%Y%m%d')
    df['regime'] = df['ret20'].apply(lambda x: '上升' if x > 0.05 else ('下降' if x < -0.05 else '震荡'))
    return dict(zip(df['date_str'], df['regime']))


def compute_breakout(df):
    """对齐breakout_scanner信号 + 未来收益。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 125:
        return None
    close = df['close'].astype(float).ffill()
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    date_str = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    out = pd.DataFrame({'date_str': date_str})
    # 信号(全部用T日及之前, 无未来)
    prev20_hi = close.rolling(20).apply(lambda x: x.iloc[:-1].max())  # 前20日最高(不含当日)
    prev20_lo = close.rolling(20).apply(lambda x: x.iloc[:-1].min())
    prev20_range = (prev20_hi / prev20_lo - 1)
    vol_ma20 = vol.rolling(20).mean().shift(1)
    hi120 = close.rolling(120).max().shift(1)
    platform = (prev20_range < 0.15).fillna(False)
    brk = (close > prev20_hi).fillna(False)
    surge = (vol > vol_ma20 * 1.5).fillna(False)
    not_high = ((close / hi120 - 1) < -0.05).fillna(False)
    out['sig_breakout'] = (platform & brk & surge & not_high)
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values
    return out


def main():
    print("突破信号 × 大盘regime 回测 | 平台+突破+放量+非高位 | 扣0.3%成本")
    regimes = load_index_regime()
    if not regimes:
        print("❌ 无上证数据"); return
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6')) and d.name != '000001SH'])
    print(f"扫描 {len(codes)} 只...")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = compute_breakout(pd.read_parquet(STORE / c / 'D.parquet'))
            if s is not None:
                s['code'] = c
                snaps.append(s)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True)
    snap['regime'] = snap['date_str'].map(regimes)
    snap = snap.dropna(subset=['ret10', 'regime'])
    sig = snap[snap.sig_breakout].copy()
    print(f"突破信号总样本: {len(sig):,}\n")

    # 分regime
    print(f"{'regime':<8}{'N':>8}{'ret5净':>9}{'ret10净':>9}{'ret20净':>9}{'胜率10':>7}{'夏普10':>8}")
    print('-' * 62)
    for reg in ['上升', '震荡', '下降']:
        g = sig[sig.regime == reg]
        if len(g) < 30:
            print(f"{reg:<8}{len(g):>8,}  样本不足"); continue
        r5 = g.ret5 - COST; r10 = g.ret10 - COST; r20 = g.ret20 - COST
        wr = (g.ret10 > COST).mean() * 100
        sh = r10.mean() / r10.std() * np.sqrt(252 / 10) if r10.std() > 0 else 0
        print(f"{reg:<8}{len(g):>8,}{r5.mean()*100:>+8.2f}%{r10.mean()*100:>+8.2f}%{r20.mean()*100:>+8.2f}%{wr:>6.0f}%{sh:>8.2f}")
    # 基准(全部信号)
    r10 = sig.ret10 - COST
    sh = r10.mean() / r10.std() * np.sqrt(252 / 10) if r10.std() > 0 else 0
    print(f"{'全部':<8}{len(sig):>8,}{(sig.ret5-COST).mean()*100:>+8.2f}%{r10.mean()*100:>+8.2f}%{(sig.ret20-COST).mean()*100:>+8.2f}{(sig.ret10>COST).mean()*100:>6.0f}%{sh:>8.2f}")

    # gate效果: 全部 vs 排除下降期
    print(f"\n=== regime gate 效果(全部 vs 排除下降期) ===")
    no_down = sig[sig.regime != '下降']
    for h in HOLDS:
        full = sig[f'ret{h}'] - COST
        gated = no_down[f'ret{h}'] - COST
        t, p = stats.ttest_ind(gated.dropna(), full.dropna(), equal_var=False)
        print(f"  ret{h}: 全部{full.mean()*100:+.2f}% vs gate后{gated.mean()*100:+.2f}% (排除下降期{len(sig)-len(no_down)}样本) t={t:+.2f} p={p:.4f} {'✅gate提升' if gated.mean()>full.mean() else '❌gate无效'}")

    # t检验: 下降期 vs 上升期
    print(f"\n=== t检验: 下降期 vs 上升期 ret10 ===")
    sd = sig[sig.regime == '下降'].ret10.dropna() - COST
    su = sig[sig.regime == '上升'].ret10.dropna() - COST
    if len(sd) > 30 and len(su) > 30:
        t, p = stats.ttest_ind(sd, su, equal_var=False)
        print(f"  下降{sd.mean()*100:+.2f}% vs 上升{su.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {'→下降期突破显著更差,gate成立' if sd.mean()<su.mean() and p<0.05 else '→无显著差异'}")


if __name__ == '__main__':
    main()
