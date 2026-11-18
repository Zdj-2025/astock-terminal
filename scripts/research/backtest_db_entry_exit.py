#!/usr/bin/env python3
"""背离+双底: 买点×卖点矩阵回测 + 分时段/分regime稳健性(2026-09-02)。

数据: store混合深度(2020起老股子集~2500只 + 2025-05起全市场)。
买点: A1反弹带[1.04,1.12](现行) A2贴底带[1.02,1.06] A3确认即买(Lc+4开) A4回踩带[0.99,1.03]
卖点: E1纯结构止损 E2棘轮(破N/破N+(N-Lb)移止损) E3量度目标T=N+(N-Lb) E4 MA10跟踪 E5固定40日
时段: 分年2020-2026 + regime(上证vs MA60上行/下行/其他=牛/熊/震荡)
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')

COST = 0.0013
RALLY = 0.20
MAX_HOLD = 60
sys.path.insert(0, 'scripts/research')
from backtest_div_double_bottom import detect_patterns, load  # noqa: E402


def get_index():
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    if idx['d'].iloc[0] > '20200110' or idx['d'].iloc[-1] < '20260820':
        from scripts.core.tushare_client import TushareClient
        pro = TushareClient().pro
        old = pro.index_daily(ts_code='000001.SH', start_date='20191201', end_date='20260901')
        idx = pd.concat([old.drop_duplicates('trade_date'), idx], ignore_index=True).drop_duplicates('d').sort_values('d')
    ic = idx.set_index('d')['close'].astype(float)
    ma60 = ic.rolling(60).mean()
    regime = {}  # date -> '牛'/'熊'/'震荡'
    prev_slope = None
    for d in ic.index:
        m = ma60.get(d, np.nan)
        if np.isnan(m):
            regime[d] = '未知'
            continue
        slope = m > ma60.shift(5).get(d, np.nan) if d in ma60.index else None
        if ic[d] > m and slope:
            regime[d] = '牛'
        elif ic[d] < m and not slope:
            regime[d] = '熊'
        else:
            regime[d] = '震荡'
    return ic, regime


def entries(df, p):
    """返回{变体: 买入日索引}"""
    l = df['low'].astype(float).values
    c = df['close'].astype(float).values
    Lc = p['Lc']
    out = {}
    if Lc + 4 >= len(c) or not all(l[Lc + 1:Lc + 4] > l[Lc]):
        return out
    for name, lo, hi, wait in [('A1', 1.04, 1.12, 15), ('A2', 1.02, 1.06, 15), ('A4', 0.99, 1.03, 15)]:
        for t in range(Lc + 3, min(Lc + 3 + wait, len(c))):
            if lo * l[Lc] <= c[t] <= hi * l[Lc]:
                out[name] = t
                break
    out['A3'] = Lc + 3   # 确认即买(第4根)
    return {k: v for k, v in out.items() if v + 1 < len(c)}


def exits(df, p, ei):
    """ei=买入日(买入在ei+1开盘). 返回{变体: (卖出价, 持有天数)}"""
    l = df['low'].astype(float).values
    h = df['high'].astype(float).values
    c = df['close'].astype(float).values
    o = df['open'].astype(float).values
    Lb, Lc, N = p['Lb'], p['Lc'], p['N']
    stop0 = min(l[Lb], l[Lc])
    entry = o[ei + 1]
    end = min(ei + 1 + MAX_HOLD, len(c) - 1)
    res = {}

    def walk(stop_fn, target=None):
        stop = stop0
        days = None
        for t in range(ei + 1, end + 1):
            if target is not None and h[t] >= target:
                return target, t - ei
            if c[t] < stop:
                return o[min(t + 1, len(c) - 1)], t - ei
            stop = max(stop, stop_fn(t, stop))
        return c[end], end - ei

    res['E1纯止损'] = walk(lambda t, s: s)
    T2 = N + (N - l[Lb])
    res['E2棘轮'] = walk(lambda t, s: N if c[t] > N and s < N else (T2 if c[t] > T2 else s))
    res['E3量度目标'] = walk(lambda t, s: s, target=T2)
    ma10 = pd.Series(c).rolling(10).mean().values
    res['E4MA10跟踪'] = walk(lambda t, s: max(s, ma10[t] * 0.995) if not np.isnan(ma10[t]) else s)
    d40 = min(ei + 1 + 40, len(c) - 1)
    res['E5固定40日'] = (c[d40], d40 - ei)
    return entry, res


def main():
    ic, regime = get_index()
    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))
    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])
    n_old = 0
    trades = []  # dict(code, t_date, entry_variant, exit_variant, ret, days, year, regime)
    for code in codes:
        try:
            df = load(code)
        except Exception:
            continue
        if len(df) < 320:
            continue
        if 'ST' in nm.get(code, '') or float(df['close'].iloc[-1]) < 2:
            continue
        if df.d.iloc[0] <= '20200115':
            n_old += 1
        c = df['close'].astype(float)
        if c.iloc[-1] / c.rolling(120).max().iloc[-1] - 1 < -0.75:
            continue
        for p in detect_patterns(df, rally=RALLY):
            for av, ei in entries(df, p).items():
                entry, ex = exits(df, p, ei)
                t_date = df.d.iloc[ei]
                if not ('20200101' <= t_date <= '20260901'):
                    continue
                for xv, (px, days) in ex.items():
                    trades.append(dict(code=code, name=nm.get(code, ''), t=t_date,
                                       av=av, xv=xv, ret=(px / entry - 1 - COST) * 100,
                                       days=days, year=t_date[:4], rg=regime.get(df.d.iloc[ei + 1], '未知')))
    t = pd.DataFrame(trades)
    print(f'老股票(2020起)子集: {n_old}只 | 总信号交易: {len(t)}笔')
    t.to_parquet('data/store/db_entry_exit_bt.parquet')

    print('\n═══ 买点×卖点矩阵 (净收益%: 中位/均值/胜率/均持有日) ═══')
    for av in ['A1', 'A2', 'A3', 'A4']:
        sub = t[t.av == av]
        if sub.empty:
            continue
        n1 = sub[sub.xv == 'E1纯止损'].shape[0]
        print(f'\n买点{av} (N信号{n1}):')
        for xv in ['E1纯止损', 'E2棘轮', 'E3量度目标', 'E4MA10跟踪', 'E5固定40日']:
            s = sub[sub.xv == xv]
            if s.empty:
                continue
            print(f'  {xv:<10} N={len(s):>4} 中位{s.ret.median():+6.2f}% 均{s.ret.mean():+6.2f}% '
                  f'胜{(s.ret>0).mean()*100:3.0f}% 持有{s.days.mean():5.1f}d 最差{s.ret.min():+6.1f}%')

    print('\n═══ 现行组合(A1×E2棘轮) 分年 ═══')
    s = t[(t.av == 'A1') & (t.xv == 'E2棘轮')]
    for y, g in s.groupby('year'):
        print(f'  {y}: N={len(g):>3} 中位{g.ret.median():+6.2f}% 均{g.ret.mean():+6.2f}% 胜{(g.ret>0).mean()*100:3.0f}%')
    print('\n═══ A1×E2 分regime ═══')
    for r, g in s.groupby('rg'):
        print(f'  {r}: N={len(g):>3} 中位{g.ret.median():+6.2f}% 均{g.ret.mean():+6.2f}% 胜{(g.ret>0).mean()*100:3.0f}%')


if __name__ == '__main__':
    main()
