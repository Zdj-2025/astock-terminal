#!/usr/bin/env python3
"""R0-记账闭环 ③: 推荐 outcome 自动回填 — 让"推荐到底赚不赚钱"第一次可统计。

口径(诚实, 防未来数据 + 不臆造):
- 基准价 = rec_date 当日 K 线收盘(rec_date 非交易日则取之后第一根); 前向窗口 = 其后的第
  +5/+10/+20 根【交易日】收盘。pnlN = close[+N]/基准−1。纯粹衡量"推完之后市场是否认可"。
  注意: 多数推荐给的是 buy_point(回踩低吸点), 真实入场通常优于基准价; 故 pnlN 是
  【保守下限】口径, 不是实际成交盈亏。实际盈亏取决于是否执行, 属另一本账。
- buy_hit: 基准日后 N 个交易日内, 日线 low 曾 <= buy_point → 该买点在窗口内"可达"。
  用日线 low 作代理(无分钟数据时最接近可执行判据), 不保证真实成交。
- 未到期(pnl20 尚不可算)的行留空, 不填/不臆造; 下次运行自动补。
- +N 需 N 根完整收盘bar; 若样本尾部不足 → NaN。
输出: 直接改写 reports/recommendations.csv, 追加列 rec_close,pnl5,pnl10,pnl20,buy_hit。
用法: PYTHONPATH=. python scripts/accounting/rec_outcome.py [--quiet]
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
STORE = PROJECT / 'data' / 'store' / 'kline'
CSV = PROJECT / 'reports' / 'recommendations.csv'
HORIZONS = ((5, 'pnl5'), (10, 'pnl10'), (20, 'pnl20'))


def load(code):
    p = STORE / f'{code}' / 'D.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    if 'date' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for c in ('close', 'low'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def compute_outcomes(recs: pd.DataFrame) -> pd.DataFrame:
    """给 recs 逐行算 outcome 列并写回 DataFrame(不动原列). 无数据的行原样保留."""
    res = []
    for _, r in recs.iterrows():
        code = str(r['code']).zfill(6)
        rec_date = pd.to_datetime(r['rec_date'])
        out = {'code': r['code']}
        df = load(code)
        if df is None or len(df) < 2:
            # 无K线(如转债/停牌/代码映射差异) → 保持空, 不臆造
            for n, c in HORIZONS:
                out[c] = None
            out['rec_close'] = None
            out['buy_hit'] = ''
            res.append(out)
            continue
        i0 = df.index[df['date'] >= rec_date]
        if i0.empty:
            i0 = df.index[-1]
        else:
            i0 = i0[0]
        base = float(df['close'].iloc[i0])
        out['rec_close'] = round(base, 3)
        # buy_point 可达性
        bp = None
        try:
            bpv = float(r['buy_point'])
            if bpv > 0:
                bp = bpv
        except (TypeError, ValueError):
            bp = None
        if bp is not None:
            wins_low = df['low'].iloc[i0:min(i0 + 20, len(df))]
            out['buy_hit'] = bool(wins_low.min() <= bp) if len(wins_low) else ''
        else:
            out['buy_hit'] = ''
        for n, c in HORIZONS:
            j = i0 + n
            if j < len(df):
                out[c] = round((float(df['close'].iloc[j]) / base - 1) * 100, 2)
            else:
                out[c] = None   # 未到期
        res.append(out)
    outdf = pd.DataFrame(res)
    if 'buy_hit' in outdf.columns:
        outdf['buy_hit'] = outdf['buy_hit'].astype(object)
        outdf.loc[outdf['buy_hit'].isna(), 'buy_hit'] = ''   # 无buy_point=空, 而非 nan
    # 按行序与 recs 对齐, 加列
    for col in outdf.columns:
        recs[col] = outdf[col].values
    return recs


def main(quiet=False):
    if not CSV.exists():
        print(f'❌ {CSV} 不存在')
        sys.exit(1)
    recs = pd.read_csv(CSV)
    for col in ['rec_close', 'pnl5', 'pnl10', 'pnl20', 'buy_hit']:
        if col not in recs.columns:
            recs[col] = ''
    orig = recs.copy()
    recs = compute_outcomes(recs)
    # 仅在确有变化时写回
    if not recs.equals(orig):
        recs.to_csv(CSV, index=False)
        if not quiet:
            print('✅ recommendations.csv outcome 列已更新')

    if not quiet:
        matured = recs[recs['pnl20'].notna()]
        print(f'总推荐 {len(recs)} 条, 已满20交易日可评 {len(matured)} 条, 未到期 {len(recs)-len(matured)} 条\n')
        for hor, col in HORIZONS:
            sub = recs[recs[col].notna()]
            if len(sub) == 0:
                continue
            wr = (sub[col] > 0).mean() * 100
            avg = sub[col].mean()
            med = sub[col].median()
            print(f'  +{hor}日: n={len(sub):>3}  胜率 {wr:.0f}%  平均 {avg:+.2f}%  中位 {med:+.2f}%')
        print('\n  按 regime 分组(20日):')
        for regime, g in matured.groupby('regime'):
            if len(g):
                wr = (g['pnl20'] > 0).mean() * 100
                print(f'    {regime:<4} 期推荐 {len(g):>2} 条: 平均 {g["pnl20"].mean():+.2f}%  胜率 {wr:.0f}%')
        print('\n  当前浮在水上(现价>基准)推荐:')
        # 现价需当日K线, 这里用 pnl 列最后可得 → 输出 matured 里 pnl20>0 的
        tops = matured[matured['pnl20'] > 0].sort_values('pnl20', ascending=False).head(8)
        for _, r in tops.iterrows():
            print(f"    {str(r['code']).zfill(6)} {r['name'][:8]:<9} {str(r['rec_date'])[:10]} "
                  f"pnl5={r['pnl5']:+.1f} pnl10={r['pnl10']:+.1f} pnl20={r['pnl20']:+.1f}%")
        print('\n⚠️ 口径: 基准=推荐日收盘, pnl 是"推后市场认可度"保守下限; 真实盈亏看是否执行(buy_hit 仅低吸可达代理)。')


if __name__ == '__main__':
    main(quiet='--quiet' in sys.argv)
