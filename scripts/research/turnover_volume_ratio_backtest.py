#!/usr/bin/env python3
"""换手率×量比 信号回测 — 验证抖音周言周语的4信号(启动/拉升/分歧/诱多)。
数据: tushare daily_basic(换手率turnover_rate+量比volume_ratio, 官方) join parquet(价格)。
重点验证"分歧(换手>10%+量比>5)": 高位应跌(见顶)/低位应涨(吸筹)。
无未来: 信号用当日换手率/量比, 收益shift(-hold)。
用法: cd stock && PYTHONPATH=. python scripts/research/turnover_volume_ratio_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import time
import tushare as ts
import numpy as np
import pandas as pd
from scipy import stats

pro = ts.pro_api()
STORE = Path(__file__).resolve().parent.parent.parent / 'data/store/kline'
NDAYS = 150   # 最近150个交易日(扩样本,排除80天下跌市偏差)

# 1. 交易日列表(从长期票取)
ref = pd.read_parquet(STORE / '000001' / 'D.parquet')
for c in ['datetime', 'day', 'trade_date']:
    if c in ref.columns:
        ref = ref.rename(columns={c: 'date'})
ref['date'] = pd.to_datetime(ref['date'])
all_dates = sorted(ref['date'].dt.strftime('%Y%m%d').unique())
recent = all_dates[-NDAYS:]
print(f"取最近 {NDAYS} 个交易日 daily_basic(换手率+量比)...")

# 2. 全市场换手率+量比
parts = []
for i, d in enumerate(recent):
    try:
        db = pro.daily_basic(trade_date=d, fields='ts_code,trade_date,turnover_rate,volume_ratio')
        parts.append(db)
    except Exception:
        pass
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{NDAYS}]")
    time.sleep(0.13)
panel = pd.concat(parts, ignore_index=True)
panel['code'] = panel['ts_code'].str[:6]
panel['date'] = pd.to_datetime(panel['trade_date'])
print(f"  daily_basic: {len(panel):,} 条")

# 3. join parquet 价格(close/dist120/ret)
print("join 价格数据...")
recs = []
codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
for code in codes:
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet')
        for c in ['datetime', 'day', 'trade_date']:
            if c in df.columns:
                df = df.rename(columns={c: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 130:
            continue
        close = df['close'].astype(float)
        hi120 = close.rolling(120).max()
        dist120 = (close / hi120 - 1) * 100
        sub = pd.DataFrame({'date': df['date'], 'code': code, 'dist120': dist120.values,
                            'ret5': (close.shift(-5) / close - 1).values,
                            'ret10': (close.shift(-10) / close - 1).values,
                            'ret20': (close.shift(-20) / close - 1).values})
        recs.append(sub)
    except Exception:
        pass
prices = pd.concat(recs, ignore_index=True)
m = panel.merge(prices, on=['code', 'date'], how='inner').dropna(subset=['ret10'])
print(f"  merge: {len(m):,} 条\n")

# 4. 4信号(按视频)
m['sig_start'] = (m['turnover_rate'] > 5) & m['volume_ratio'].between(2, 3)          # 启动
m['sig_pull']  = m['turnover_rate'].between(5, 10) & m['volume_ratio'].between(3, 5) # 拉升
m['sig_div']   = (m['turnover_rate'] > 10) & (m['volume_ratio'] > 5)                 # 分歧
m['sig_div_high'] = m['sig_div'] & (m['dist120'] > -5)                               # 高位分歧
m['sig_div_low']  = m['sig_div'] & (m['dist120'] < -15)                              # 低位分歧
m['sig_trap']  = (m['turnover_rate'] > 5) & (m['volume_ratio'] < 2)                  # 诱多

base = m['ret10'].dropna()
print(f"{'信号':<20}{'N':>8}{'ret10':>9}{'ret20':>9}{'胜率10':>7}")
print('-' * 55)
for nm, mask in [('基准(全市场)', None), ('启动(>5%,量比2-3)', m.sig_start),
                 ('拉升(5-10%,3-5)', m.sig_pull), ('分歧(>10%,>5)', m.sig_div),
                 ('└高位分歧(距120高>-5%)', m.sig_div_high), ('└低位分歧(<-15%)', m.sig_div_low),
                 ('诱多(>5%,量比<2)', m.sig_trap)]:
    g = m if mask is None else m[mask]
    n = len(g); r10 = g['ret10']; r20 = g['ret20']
    wr = (r10 > 0).mean() * 100 if n else 0
    print(f"{nm:<18}{n:>8,}{r10.mean()*100:>+8.2f}%{r20.mean()*100:>+8.2f}%{wr:>6.0f}%")

print(f"\n[t检验 ret10 vs 基准]")
for nm, mask, expect in [('高位分歧(视频:应跌/见顶)', m.sig_div_high, '跌'),
                         ('低位分歧(视频:吸筹应涨)', m.sig_div_low, '涨'),
                         ('启动(视频:应涨)', m.sig_start, '涨'),
                         ('诱多(视频:应跌)', m.sig_trap, '跌')]:
    s = m.loc[mask, 'ret10'].dropna()
    if len(s) > 50:
        t, p = stats.ttest_ind(s, base, equal_var=False)
        ok = '✅验证' if (p < 0.05) else '—不显著'
        print(f"  {nm}: {s.mean()*100:+.2f}% vs 基准{base.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {ok} N={len(s)}")
