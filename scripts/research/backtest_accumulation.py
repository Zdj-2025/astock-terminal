#!/usr/bin/env python3
"""量价吸筹共振回测 — 验证"波动收敛+缩量"共振是否预测上涨(机构吸筹蓄势)。

只用同步量价数据(价格/成交量), 不用滞后公告/股东数。
无未来: 信号用T日及之前rolling(不center), 收益用shift(-hold)。
对比: 波动收敛 / 缩量 / 共振 vs 基准, t检验p值。
预期: 若机构在低位吸筹, 会表现为"价格波动收敛+成交量萎缩"(低调收集),
      随后放量启动 → 共振状态未来收益应显著高于基准。

用法: cd stock && PYTHONPATH=. python scripts/research/backtest_accumulation.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

STORE = PROJECT_ROOT / 'data/store/kline'
HOLDS = [5, 10, 20]


def compute_signals(df):
    """单只股: 算收敛/缩量/共振信号 + 未来收益。无未来。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 80:
        return None
    close = df['close'].astype(float)
    vol = df['volume'].astype(float)
    # 布林带宽(归一化): 2*std20/ma20
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = 2 * std20 / ma20
    # 带宽在过去60日的分位(0=最窄=最收敛), rolling rank 无未来
    bb_rank = bb_width.rolling(60).rank(pct=True)
    # 量比
    vol_ratio = vol / vol.rolling(20).mean()
    # 信号
    out = pd.DataFrame({
        'date': df['date'],
        'sig_compress': (bb_rank < 0.2).fillna(False),   # 波动收敛: 带宽近60日最窄20%
        'sig_shrink': (vol_ratio < 0.8).fillna(False),    # 缩量: 量比<0.8
    })
    out['sig_resonance'] = out['sig_compress'] & out['sig_shrink']
    # 突破启动: 前期平台(20日振幅<15%) + 当日创20日新高 + 放量(>1.5倍均量)
    hi20 = close.rolling(20).max()
    lo20 = close.rolling(20).min()
    platform = ((hi20 / lo20 - 1) < 0.15).shift(1)   # 前一日处平台状态(无未来)
    prev_hi = close.shift(1).rolling(20).max()        # 前20日最高(不含当日)
    breakout = close > prev_hi                         # 突破前20日高点
    vol_surge = vol > vol.rolling(20).mean() * 1.5     # 放量
    out['sig_breakout'] = platform.fillna(False) & breakout.fillna(False) & vol_surge.fillna(False)
    # 突破位置: 距120日高点。低位突破=从深位启动(主力建仓后拉升), 高位突破=末升段?
    hi120 = close.rolling(120).max()
    dist120 = close / hi120 - 1
    out['sig_brk_low'] = out['sig_breakout'] & (dist120 < -0.10).fillna(False)   # 低位(距120高<-10%)
    out['sig_brk_high'] = out['sig_breakout'] & (dist120 > -0.05).fillna(False)  # 高位(距120高>-5%)
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1)
    return out


def main():
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    print(f"量价吸筹共振回测 | {len(codes)}只 | 收敛=带宽近60日最窄20%, 缩量=量比<0.8")
    snaps = []
    for i, code in enumerate(codes):
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            sub = compute_signals(df)
            if sub is not None:
                sub['code'] = code
                snaps.append(sub)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True)
    snap = snap.dropna(subset=['ret10'])
    print(f"总快照: {len(snap):,} (dropna ret10)\n")

    groups = {
        '基准(全市场)': snap,
        '波动收敛': snap[snap.sig_compress],
        '缩量': snap[snap.sig_shrink],
        '共振(收敛∩缩量)': snap[snap.sig_resonance],
        '突破启动(平台+新高+放量)': snap[snap.sig_breakout],
        '└低位突破(距120高<-10%)': snap[snap.sig_brk_low],
        '└高位突破(距120高>-5%)': snap[snap.sig_brk_high],
    }
    print(f"{'组':<16}{'N':>10}{'ret5':>9}{'ret10':>9}{'ret20':>9}{'胜率10':>8}")
    for name, g in groups.items():
        n = len(g)
        wr = (g['ret10'] > 0).mean() * 100
        print(f"{name:<14}{n:>10,}{g['ret5'].mean()*100:>+8.2f}%"
              f"{g['ret10'].mean()*100:>+8.2f}%{g['ret20'].mean()*100:>+8.2f}%{wr:>7.0f}%")

    base = snap['ret10'].dropna()
    print(f"\n[t检验 ret10]")
    for sig_name, sig_mask in [('共振', snap.sig_resonance), ('突破启动', snap.sig_breakout),
                                ('低位突破', snap.sig_brk_low), ('高位突破', snap.sig_brk_high)]:
        s = snap.loc[sig_mask, 'ret10'].dropna()
        if len(s) < 100:
            print(f"  [{sig_name}] 样本不足 N={len(s)}"); continue
        t, p = stats.ttest_ind(s, base, equal_var=False)
        better = '✅显著更优' if (p < 0.05 and s.mean() > base.mean()) else \
                 ('⚠️显著更差' if (p < 0.05 and s.mean() < base.mean()) else '—无显著差异')
        print(f"  {sig_name:<6} vs 基准: t={t:+.2f} p={p:.4f}  "
              f"{sig_name}{s.mean()*100:+.2f}% vs {base.mean()*100:+.2f}%  {better}  (N={len(s)})")


if __name__ == '__main__':
    main()
