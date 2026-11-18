#!/usr/bin/env python3
"""双底 vs 深回调二买 × 背离口径 2×2回测(2026-09-03,用户提议).

四组(互斥):
  G1 双抬背离×浅带(|Lc/Lb-1|<=7%)   = 现行双底
  G2 双抬背离×深带(+7%~+30%)        = 深回调二买(缠论二买)
  G3 DIF-only背离×浅带(hist新低但DIF抬)
  G4 DIF-only背离×深带              = 欢瑞世纪所属组
其余条件同双底: 间距/B过滤/反弹N>=Lb*1.20/Lc<N; A3确认即买; E2棘轮+E5固定40日出场.
验证: --verify 欢瑞000892 as-of 8/28应触发G4, 8/28开盘买入.
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
SW = 5


def load(code):
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def macd(c):
    ef = c.ewm(span=12, adjust=False).mean(); es = c.ewm(span=26, adjust=False).mean()
    dif = ef - es
    return dif.values, (dif - dif.ewm(span=9, adjust=False).mean()).values


def swings(l, n=SW):
    # 左严格/右允许相等: 双底的本义=同一水平被测两次,精确触碰相等不算破(欢瑞8/24=8/25低3.60案例)
    return [i for i in range(n, len(l) - n)
            if all(l[i] < l[i - j] for j in range(1, n + 1)) and all(l[i] <= l[i + j] for j in range(1, n + 1))]


def patterns(df):
    """返回四组触发列表: [(group, ei, Lb, Lc, N)]"""
    l = df['low'].astype(float).values
    h = df['high'].astype(float).values
    c = df['close'].astype(float).values
    dif, hist = macd(pd.Series(c))
    out = []
    for La, Lb, Lc in zip(swings(l), swings(l)[1:], swings(l)[2:]):
        if not (15 <= Lb - La <= 60 and Lc - Lb <= 40):
            continue
        if not l[Lb] < l[La]:
            continue
        if Lb + 1 < Lc and np.min(l[Lb + 1:Lc + 1]) < l[Lb] * 0.90:
            continue
        band = l[Lc] / l[Lb] - 1
        if not (0 <= band <= 0.30):
            continue
        N = float(np.max(h[Lb:Lc + 1]))
        if N < l[Lb] * (1 + RALLY) or l[Lc] >= N:
            continue
        dif_lift = dif[Lb] > dif[La]
        hist_lift = hist[Lb] > hist[La]
        if not dif_lift:
            continue
        deep = band > 0.07
        g = ('G2深带·双抬' if deep else 'G1浅带·双抬') if hist_lift else ('G4深带·DIFonly' if deep else 'G3浅带·DIFonly')
        if Lc + 4 >= len(c) or not all(l[Lc + 1:Lc + 4] >= l[Lc]):  # 相等回踩=不破(欢瑞8/25=3.60精确复测)
            continue
        ei = Lc + 3   # A3确认即买
        out.append((g, ei, l[Lb], l[Lc], N))
    return out


def exits(df, ei, Lb, Lc, N):
    c = df['close'].astype(float).values
    o = df['open'].astype(float).values
    h = df['high'].astype(float).values
    stop0 = min(Lb, Lc)
    entry = o[ei + 1]
    end = min(ei + 1 + MAX_HOLD, len(c) - 1)
    T2 = N + (N - Lb)

    def ratchet():
        stop = stop0
        for t in range(ei + 1, end + 1):
            if c[t] < stop:
                return o[min(t + 1, len(c) - 1)], t - ei
            if c[t] > T2:
                stop = max(stop, T2)
            elif c[t] > N:
                stop = max(stop, N)
        return c[end], end - ei
    px2, d2 = ratchet()
    d40 = min(ei + 1 + 40, len(c) - 1)
    return entry, {'E2': (px2, d2), 'E5': (c[d40], d40 - ei)}


def verify():
    df = load('000892')
    for g, ei, Lb, Lc, N in patterns(df):
        t = df.d.iloc[ei]
        if t >= '20260820':
            print(f'✓ {g} 触发{t}(买次日{df.d.iloc[ei+1]}开{df.open.iloc[ei+1]:.2f}) Lb={Lb:.2f} Lc={Lc:.2f} N={N:.2f} 止损{min(Lb,Lc):.2f}')
    print('(预期: G4深带·DIFonly, 触发~20260827, 买8/28开盘~3.83)')


def main():
    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))
    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists() and d.name.isdigit()
                    and len(d.name) == 6 and d.name[:2] not in ('51', '15', '56', '58')])
    rows = []
    for code in codes:
        n = nm.get(code, '')
        if 'ST' in n or '退' in n:
            continue
        try:
            df = load(code)
        except Exception:
            continue
        if len(df) < 320 or float(df['close'].iloc[-1]) < 2:
            continue
        c = df['close'].astype(float)
        if c.iloc[-1] / c.rolling(120).max().iloc[-1] - 1 < -0.75:
            continue
        for g, ei, Lb, Lc, N in patterns(df):
            if not ('20200101' <= df.d.iloc[ei] <= '20260901') or ei + 42 >= len(df):
                continue
            entry, ex = exits(df, ei, Lb, Lc, N)
            for xv, (px, days) in ex.items():
                rows.append(dict(g=g, xv=xv, t=df.d.iloc[ei], year=df.d.iloc[ei][:4],
                                 ret=(px / entry - 1 - COST) * 100, days=days))
    t = pd.DataFrame(rows)
    print(f'总信号×出场: {len(t)} | 2020-2026.9')
    print('\n═══ 四组 × 出场 (中位/均值/胜率/均持有) ═══')
    for g in ['G1浅带·双抬', 'G2深带·双抬', 'G3浅带·DIFonly', 'G4深带·DIFonly']:
        for xv in ['E2', 'E5']:
            s = t[(t.g == g) & (t.xv == xv)]
            if s.empty:
                print(f'{g:<14} {xv} N=0')
                continue
            print(f'{g:<14} {xv} N={len(s):>4} 中位{s.ret.median():+6.2f}% 均{s.ret.mean():+6.2f}% '
                  f'胜{(s.ret>0).mean()*100:3.0f}% 持{s.days.mean():4.1f}d 最差{s.ret.min():+6.1f}%')
    print('\n═══ E2棘轮 分年(各组) ═══')
    s = t[t.xv == 'E2']
    for g in ['G1浅带·双抬', 'G2深带·双抬', 'G4深带·DIFonly']:
        sg = s[s.g == g]
        if sg.empty:
            continue
        line = f'{g:<14}'
        for y, gy in sg.groupby('year'):
            line += f' | {y}:{gy.ret.median():+.1f}%/{(gy.ret>0).mean()*100:.0f}%N{len(gy)}'
        print(line)
    t.to_parquet('data/store/second_buy_bt.parquet')


if __name__ == '__main__':
    verify() if '--verify' in sys.argv else main()
