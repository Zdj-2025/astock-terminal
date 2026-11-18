#!/usr/bin/env python3
"""TCL模式详细回测 — 多时间点+多维度稳健性验证(2026-06-17)。

之前整体回测只给平均(+0.66%), 需多维度验证稳健性:
1. 按年/分季(时段稳健性)
2. 按大盘regime(上证MA60多空, 看模式在什么环境有效)
3. 关键指标(胜率/盈亏比PF/夏普/最大单笔亏, 不只是均值)
4. 参数敏感性(dh60回踩深度/f1强度, 看条件是否过拟合)

严格无未来: 复用tcl_pattern的V3 savgol+shift; 信号指标用T及之前; 收益shift(-hold)仅衡量。

用法: PYTHONPATH=. python scripts/research/backtest_tcl_detailed.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask, load_codes, STORE

HOLD = 10


def metrics(s):
    """收益序列的关键指标。"""
    s = s.dropna()
    if len(s) < 5:
        return None
    win = s[s > 0]
    loss = s[s < 0]
    avg_win = win.mean() if len(win) else 0
    avg_loss = abs(loss.mean()) if len(loss) else 0
    pf = avg_win / avg_loss if avg_loss > 0 else float('inf')
    sharpe = s.mean() / s.std() * np.sqrt(252 / HOLD) if s.std() > 0 else 0
    return {'n': len(s), 'mean': s.mean() * 100, 'win%': (s > 0).mean() * 100,
            'PF': pf, 'median': s.median() * 100, 'max_loss': s.min() * 100, 'sharpe': sharpe}


def main():
    codes = load_codes()
    print(f"TCL模式详细回测 | {len(codes)}只 | 持有{HOLD}日 | 严格无未来")
    # 大盘MA60(上证综指, 判信号时点多空)
    bull = set()
    try:
        idx = pd.read_parquet(STORE / '000001SH' / 'D.parquet')
        idx['date'] = pd.to_datetime(idx['date'])
        idx['ma60'] = idx['close'].rolling(60).mean()
        bull = set(idx[idx['close'] > idx['ma60']]['date'])
        print(f"大盘regime基准: 上证综指MA60, 多头日{len(bull)}天")
    except Exception as e:
        print(f"(大盘数据不可用: {e})")

    rows = []
    for i, code in enumerate(codes):
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            df['date'] = pd.to_datetime(df['date'])
            df = compute_indicators(df)
            df['red'] = (df['DIF'] > df['DEA']).astype(int)
            df['red_ratio_20'] = df['red'].rolling(20).mean()   # 近20日MACD红柱占比(蓄能)
            df['ret20'] = df['close'].pct_change(20) * 100
            df['sig'] = tcl_mask(df)
            df['ret'] = df['close'].shift(-HOLD) / df['close'] - 1   # 未来收益仅衡量
            for _, r in df[df['sig'] & df['ret'].notna()].iterrows():
                rows.append({'date': r['date'], 'code': code, 'ret': r['ret'],
                             'f1': r['f1'], 'dh60': r['dh60'] * 100,
                             'red_ratio': r['red_ratio_20'], 'ret20': r['ret20'],
                             'vsMA20': (r['close'] / r['MA20'] - 1) * 100,
                             'bull': r['date'] in bull})
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}] 信号{len(rows)}", flush=True)

    rec = pd.DataFrame(rows)
    if rec.empty:
        print("无信号"); return

    print(f"\n{'='*70}\n信号总数 {len(rec)}\n{'='*70}")
    m = metrics(rec['ret'])
    print(f"\n【整体】 均值{m['mean']:+.2f}% 胜率{m['win%']:.1f}% 盈亏比{m['PF']:.2f} "
          f"夏普{m['sharpe']:.2f} 中位{m['median']:+.2f}% 最大单笔亏{m['max_loss']:+.1f}%")

    print("\n【按年】")
    rec['year'] = rec['date'].dt.year
    for y, g in rec.groupby('year'):
        m = metrics(g['ret'])
        if m:
            print(f"  {y}: n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win%']:5.1f}% PF{m['PF']:.2f} 夏普{m['sharpe']:.2f}")

    print("\n【按季度(近8季)】")
    rec['yq'] = rec['date'].dt.to_period('Q').astype(str)
    recent_qs = sorted(rec['yq'].unique())[-8:]
    for q in recent_qs:
        g = rec[rec['yq'] == q]
        m = metrics(g['ret'])
        if m:
            print(f"  {q}: n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win%']:5.1f}% PF{m['PF']:.2f}")

    if bull:
        print("\n【按大盘regime(信号时点上证MA60多空)】")
        for label, sub in [('多头(大盘>MA60)', rec[rec['bull']]), ('弱势(大盘<MA60)', rec[~rec['bull']])]:
            m = metrics(sub['ret'])
            if m:
                print(f"  {label}: n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win%']:5.1f}% PF{m['PF']:.2f} 夏普{m['sharpe']:.2f}")

    print("\n【参数敏感性: 回踩深度dh60】")
    for lo, hi, lbl in [(-5, -15, '浅回踩 dh60 -5~-15%'), (-15, -30, '深回踩 dh60 -15~-30%'),
                        (-5, -10, '极浅 dh60 -5~-10%'), (-10, -20, '中等 dh60 -10~-20%')]:
        sub = rec[(rec['dh60'] >= hi) & (rec['dh60'] <= lo)]
        m = metrics(sub['ret'])
        if m and m['n'] > 20:
            print(f"  {lbl}: n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win%']:5.1f}% PF{m['PF']:.2f}")

    print("\n【参数敏感性: 趋势强度f1】")
    for lo, hi, lbl in [(0.3, 0.6, '弱 f1 0.3-0.6'), (0.6, 1.2, '中 f1 0.6-1.2'), (1.2, 99, '强 f1>1.2')]:
        sub = rec[(rec['f1'] >= lo) & (rec['f1'] < hi)]
        m = metrics(sub['ret'])
        if m and m['n'] > 20:
            print(f"  {lbl}: n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win%']:5.1f}% PF{m['PF']:.2f}")

    print("\n【★★ 蓄能子集: TCL回踩 + MACD红柱持续 + 涨幅有限(TCL 6/1模式) ★★】")
    print("  (spring_scanner发现: 红柱100%是硬门槛, <100%胜率17% vs 100%胜率69%)")
    best_n, best_mean, best_lbl = 0, -99, ""
    for rr, lbl in [(1.0, '红柱100%'), (0.8, '红柱≥80%'), (0.6, '红柱≥60%')]:
        for r20, lbl2 in [(3, 'ret20<3%'), (5, 'ret20<5%'), (10, 'ret20<10%')]:
            sub = rec[(rec['red_ratio'] >= rr) & (rec['ret20'] < r20)]
            m = metrics(sub['ret'])
            if m and m['n'] > 20:
                print(f"  {lbl}+{lbl2}: n={m['n']:5d} 均值{m['mean']:+6.2f}% 胜率{m['win%']:5.1f}% "
                      f"PF{m['PF']:.2f} 夏普{m['sharpe']:.2f} 中位{m['median']:+.2f}%")
                if m['mean'] > best_mean and m['n'] > 30:
                    best_mean, best_lbl, best_n = m['mean'], f"{lbl}+{lbl2}", m['n']
    print(f"\n  → 最佳蓄能组合: {best_lbl} (n={best_n}, 均值{best_mean:+.2f}%)")
    print(f"  → 对比整体: 均值+0.66% 胜率47.9% 夏普0.32")

    rec.to_csv(PROJECT_ROOT / 'reports' / 'tcl_detailed_backtest.csv', index=False)
    print(f"\n明细 → reports/tcl_detailed_backtest.csv")


if __name__ == "__main__":
    main()
