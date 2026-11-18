#!/usr/bin/env python3
"""高开低走重心上移扫描器 — 数据港形态(2026-08-14涨停前8/11-8/13形态提炼)。

形态定义(每天):
  高开: open > prev_close × (1+gap_min)     隔夜买盘强
  低走: open > close                          盘中获利回吐/洗盘
  重心上移: close > prev_close                收盘承接力强于抛压
连续N天(默认3,允许破1天) → "洗盘上移", 常在主升前(数据港案例: 3日后平开涨停)。

CLI:
  python scripts/screener/gap_up_drift_scanner.py            # 默认扫最近3交易日
  python scripts/screener/gap_up_drift_scanner.py --days 3 --strict
  python scripts/screener/gap_up_drift_scanner.py --gap 1.0  # 高开下限%
"""
import argparse
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')


def scan(days: int = 3, gap: float = 1.0, strict: bool = False, min_hits: int = None):
    """返回符合形态的股票列表"""
    if min_hits is None:
        min_hits = days if strict else max(2, days - 1)
    codes_dir = 'data/store/kline'
    results = []
    for code in os.listdir(codes_dir):
        p = os.path.join(codes_dir, code, 'D.parquet')
        if not os.path.isfile(p) or not code.isdigit() or len(code) != 6:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        dcol = [c for c in df.columns if 'date' in c.lower()]
        if not dcol:
            continue
        df = df.sort_values(dcol[0])
        if len(df) < days + 1:
            continue
        tail = df.tail(days + 1).reset_index(drop=True)
        hits, opens, closes, gaps = 0, [], [], []
        for i in range(1, len(tail)):
            o, c = float(tail.loc[i, 'open']), float(tail.loc[i, 'close'])
            pc = float(tail.loc[i - 1, 'close'])
            po = float(tail.loc[i - 1, 'open'])
            gp = (o / pc - 1) * 100
            cond = (gp >= gap) and (o > c) and (c > pc)
            if cond:
                hits += 1
            opens.append(o); closes.append(c); gaps.append(gp)
        if hits >= min_hits:
            last = tail.iloc[-1]
            results.append({
                'code': code,
                'close': float(last['close']),
                'hits': f'{hits}/{days}',
                'gaps': '/'.join(f'{g:+.1f}%' for g in gaps),
                'ret3': (float(last['close']) / float(tail.loc[len(tail) - days - 1, 'close']) - 1) * 100,
            })
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description='高开低走重心上移(数据港形态)扫描')
    ap.add_argument('--days', type=int, default=3, help='回看天数(默认3)')
    ap.add_argument('--gap', type=float, default=1.0, help='高开下限%(默认1.0)')
    ap.add_argument('--strict', action='store_true', help='严格模式(N天全满足)')
    a = ap.parse_args()
    res = scan(a.days, a.gap, a.strict)
    if len(res) == 0:
        print(f'无符合形态标的(days={a.days} gap>={a.gap}% strict={a.strict})')
        return
    res = res.sort_values('hits', ascending=False)
    print(f'=== 高开低走·重心上移 扫描 | 近{a.days}日 高开>={a.gap}% | 命中{len(res)}只 ===')
    print(f"{'代码':<8}{'现价':>8}{'命中':>6}{'高开序列':>26}{'3日涨幅':>8}")
    for _, r in res.head(30).iterrows():
        print(f"{r['code']:<8}{r['close']:>8.2f}{r['hits']:>6}{r['gaps']:>26}{r['ret3']:>+7.1f}%")
    print('\n💡 数据港案例: 8/11-8/13此形态(高开5-6%低走但收盘逐日抬升) → 8/14平开涨停')


if __name__ == '__main__':
    main()
