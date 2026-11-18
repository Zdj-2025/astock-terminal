#!/usr/bin/env python3
"""筹码阻力地图 — 利弗莫尔"最小阻力线"的量化落地(2026-09-10).

原理: 阻力的本体是密集成交区的套牢盘(memory: 最小阻力线深研).
用近N日成交额按价格分桶=筹码分布近似; 输出最大筹码峰(主分水岭)/上方阻力层/
下方支撑层/获利套牢比例 → 每日盘前回答"方向在不对称的哪一边".

用法: PYTHONPATH=. python scripts/analyzers/volume_profile.py
      from scripts.analyzers.volume_profile import resistance_map
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import pandas as pd

INDEX_PARQUET = PROJECT_ROOT / 'data/store/index/sh000001_long.parquet'
N_DAYS = 90      # 筹码统计窗口(交易日)
BIN = 20         # 价格桶宽(指数点)


def _load():
    df = pd.read_parquet(INDEX_PARQUET)
    df['d'] = pd.to_datetime(df['d'])
    return df.sort_values('d')


def volume_profile(df=None, n_days=N_DAYS, bin_size=BIN):
    """返回DataFrame[bin_left, bin_right, amt, pct]."""
    if df is None:
        df = _load()
    w = df.tail(n_days).copy()
    lo = int(np.floor(w['c'].min() / bin_size) * bin_size)
    hi = int(np.ceil(w['c'].max() / bin_size) * bin_size)
    w['bin'] = pd.cut(w['c'], np.arange(lo, hi + bin_size, bin_size))
    prof = w.groupby('bin', observed=True)['amt'].sum().rename('amt').reset_index()
    prof['pct'] = (prof['amt'] / prof['amt'].sum() * 100).round(1)
    prof['bin_left'] = prof['bin'].astype(str).str.extract(r'([\d.]+),')[0].astype(float).astype(int)
    prof['bin_right'] = prof['bin_left'] + bin_size
    return prof[['bin_left', 'bin_right', 'amt', 'pct']]


def resistance_map(n_days=N_DAYS, bin_size=BIN, verbose=True):
    """计算并(可选)打印阻力地图. 返回dict供其他模块调用."""
    df = _load()
    px = float(df['c'].iloc[-1])
    date = df['d'].iloc[-1].strftime('%Y-%m-%d')
    prof = volume_profile(df, n_days, bin_size)
    prof = prof[prof['pct'] >= 3].reset_index(drop=True)  # 剔除噪声桶

    peak = prof.loc[prof['pct'].idxmax()]
    above = prof[prof['bin_left'] > px].head(3)
    below = prof[prof['bin_right'] <= px].tail(3)
    win_pct = round(float(prof.loc[prof['bin_right'] <= px, 'pct'].sum()), 1)
    lose_pct = round(float(prof.loc[prof['bin_left'] > px, 'pct'].sum()), 1)

    if verbose:
        print(f"\n--- 🗺️ 上证筹码阻力地图({date}收盘{px:.0f}, 近{n_days}日成交额分布) ---")
        print(f"  最大筹码峰: {peak['bin_left']:.0f}-{peak['bin_right']:.0f} ({peak['pct']}%) ← 主分水岭")
        if len(above):
            ups = ' → '.join(f"{r['bin_left']:.0f}-{r['bin_right']:.0f}({r['pct']}%)" for _, r in above.iterrows())
            print(f"  上方阻力: {ups}")
        if len(below):
            dns = ' ← '.join(f"{r['bin_left']:.0f}-{r['bin_right']:.0f}({r['pct']}%)" for _, r in below.iloc[::-1].iterrows())
            print(f"  下方支撑: {dns}")
        print(f"  获利盘{win_pct}% / 套牢盘{lose_pct}%")
        up_w = above['pct'].sum(); dn_w = below['pct'].sum()
        bias = '偏下(上压厚)' if up_w > dn_w * 1.3 else ('偏上(下垫厚)' if dn_w > up_w * 1.3 else '均衡')
        print(f"  最小阻力速读: 上方近3层{up_w:.0f}% vs 下方近3层{dn_w:.0f}% → {bias}")
        print(f"  💡 突破最大峰需放量; 无量突破=供给未消化=假关键点(利弗莫尔)")
    return {'date': date, 'price': px, 'peak': (int(peak['bin_left']), int(peak['bin_right'])),
            'win_pct': win_pct, 'lose_pct': lose_pct,
            'above': above[['bin_left', 'bin_right', 'pct']].to_dict('records'),
            'below': below[['bin_left', 'bin_right', 'pct']].to_dict('records')}


if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)
    resistance_map()


# ============ 升级(2026-09-10晚): 两套成熟算法 ============
# ① Volume Profile标准算法(Steidlmayer/CQG): POC + Value Area双向扩张70%
# ② CYQ筹码分布(通达信口径): 换手率衰减 + 三角形日内分布 + WINNER获利盘

def value_area(prof, target=0.70):
    """CQG官方算法: 从POC起逐档双向比较扩张, 直到累积量>=target.
    返回dict(poc_bin, val, vah, coverage)."""
    p = prof.sort_values('amt', ascending=False).reset_index(drop=True)
    if p.empty:
        return None
    lo = hi = 0
    cum = p.at[0, 'amt']
    total = p['amt'].sum()
    while cum < total * target and (lo > 0 or hi < len(p) - 1):
        down = p.at[lo - 1, 'amt'] if lo > 0 else -1
        up = p.at[hi + 1, 'amt'] if hi < len(p) - 1 else -1
        if up >= down:
            hi += 1; cum += up
        else:
            lo -= 1; cum += down
    sub = p.iloc[lo:hi + 1]
    return {'poc_bin': (int(p.at[0, 'bin_left']), int(p.at[0, 'bin_right'])),
            'val': float(sub['bin_left'].min()), 'vah': float(sub['bin_right'].max()),
            'coverage': round(float(cum / total * 100), 1)}


def cyq_distribution(code, n_days=250, decay=1.0):
    """CYQ筹码分布(通达信近似): 历史筹码x(1-换手率xdecay), 当日量按[low,high]三角形分布.
    返回(分布DataFrame, 获利盘%, 现价)."""
    df = pd.read_parquet(PROJECT_ROOT / f'data/store/kline/{code}/D.parquet')
    for col in ('datetime', 'day', 'trade_date'):
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['date'] = df['date'].astype(str).str.replace('-', '').str[:8]
    df = df.sort_values('date').tail(n_days).reset_index(drop=True)
    from scripts.core.tushare_client import TushareClient
    c = TushareClient()
    suffix = '.SH' if code[0] in '689' or code[:2] in ('51', '56', '58') else '.SZ'
    db = c.pro.daily_basic(ts_code=code + suffix, start_date=df['date'].iloc[0],
                           end_date=df['date'].iloc[-1], fields='trade_date,turnover_rate')
    tmap = dict(zip(db['trade_date'].astype(str), db['turnover_rate'].fillna(2.0)))

    price_range = float(df['high'].max() - df['low'].min())
    bin_w = max(0.5, round(price_range / 100 * 2) / 2)  # 自适应: 约100桶,千元股不炸桶
    bins = np.arange(np.floor(df['low'].min()), np.ceil(df['high'].max()) + bin_w, bin_w)
    dist = np.zeros(len(bins) - 1)
    centers = (bins[:-1] + bins[1:]) / 2
    for _, r in df.iterrows():
        tr = min(tmap.get(r['date'], 2.0) / 100, 1.0)
        dist *= (1 - tr * decay)
        lo, hi = float(r['low']), float(r['high'])
        avg = float(r['close']) if hi <= lo else min(max(float(r['amount']) / float(r['volume']) / 100 if float(r.get('amount', 0) or 0) > 0 and float(r['volume']) > 0 else float(r['close']), lo), hi)
        w = np.clip(1 - np.abs(centers - avg) / max(hi - lo, 1e-9), 0.05, None)
        w[(centers < lo - 0.25) | (centers > hi + 0.25)] = 0
        if w.sum() > 0:
            dist += w / w.sum() * tr
    total = dist.sum()
    px = float(df['close'].iloc[-1])
    win = float(dist[centers <= px].sum() / total * 100) if total > 0 else 50.0
    out = pd.DataFrame({'bin_left': bins[:-1], 'bin_right': bins[1:], 'amt': dist})
    out['pct'] = (out['amt'] / total * 100).round(1) if total > 0 else 0
    out = out[out['pct'] >= 0.5].reset_index(drop=True)
    return out, round(win, 1), px
