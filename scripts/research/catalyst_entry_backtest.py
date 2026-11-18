#!/usr/bin/env python3
"""催化驱动入场回测(E)：业绩预增(>30%)公告后, '次日直接买(shallow)' vs '等回踩MA5/10/20'。
核心看: 多少催化股根本不回踩(踏空率) + 不回踩的那些涨多少。
entry: shallow=次日收盘买; MAx=10日内首次触及MAx当日收盘买; 都持H日。
无未来: MA用截至当日, entry在touch日, forward从entry。"""
import pandas as pd, numpy as np, os
from scipy import stats

FCACHE = './data/store/forecast/forecast_all.parquet'
KDIR = './data/store/kline'
H = 10

fc = pd.read_parquet(FCACHE)
fc = fc[(fc['type'] == '预增') & (fc['p_change_min'] > 30)].copy()
fc['code'] = fc['ts_code'].str.split('.').str[0]
fc['ann'] = fc['ann_date'].astype(str)
print(f"预增>30% 事件: {len(fc)}")

KCACHE = {}

def get_k(code):
    if code in KCACHE:
        return KCACHE[code]
    p = f'{KDIR}/{code}/D.parquet'
    df = None
    if os.path.exists(p):
        try:
            d = pd.read_parquet(p)
            dc = 'trade_date' if 'trade_date' in d.columns else ('date' if 'date' in d.columns else d.columns[0])
            d['td'] = pd.to_datetime(d[dc]).dt.strftime('%Y%m%d')
            d = d.sort_values('td').reset_index(drop=True)
            df = d[['td', 'close', 'low']]
        except Exception:
            df = None
    KCACHE[code] = df
    return df

rows = []
for k, (_, r) in enumerate(fc.iterrows()):
    df = get_k(r['code'])
    if df is None:
        continue
    if r['ann'] not in set(df['td'].values):
        continue
    close = df['close'].astype(float).values
    low = df['low'].astype(float).values
    i0 = df.index[df['td'] == r['ann']][0]
    n = len(close)
    if i0 + 12 >= n or i0 < 20:
        continue
    ma5 = pd.Series(close).rolling(5).mean().values
    ma10 = pd.Series(close).rolling(10).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values

    def touch(ma):
        for j in range(i0 + 1, min(i0 + 11, n)):
            if not np.isnan(ma[j]) and low[j] <= ma[j]:
                return j
        return None

    t5, t10, t20 = touch(ma5), touch(ma10), touch(ma20)

    def ret(e):
        return (close[e + H] / close[e] - 1) if (e is not None and e + H < n) else np.nan

    rows.append(dict(shallow=ret(i0 + 1), ma5=ret(t5) if t5 else np.nan,
                     ma10=ret(t10) if t10 else np.nan, ma20=ret(t20) if t20 else np.nan,
                     hit5=t5 is not None, hit10=t10 is not None, hit20=t20 is not None))
    if (k + 1) % 1000 == 0:
        print(f"  {k+1}...")

R = pd.DataFrame(rows)
print(f"\n样本: {len(R)} 事件(有足够K线) | 持有{H}日")

# 回踩率
print(f"\n10日内回踩率: MA5 {R['hit5'].mean()*100:.1f}% | MA10 {R['hit10'].mean()*100:.1f}% | MA20 {R['hit20'].mean()*100:.1f}%")
print(f"MA20不触及(等MA20会踏空): {(~R['hit20']).mean()*100:.1f}%")

# 各策略收益
def summ(col, sub):
    s = sub[col].dropna()
    t, p = stats.ttest_1samp(s, 0)
    return s.mean() * 100, p, len(s)

sh, psh, nsh = summ('shallow', R)
print(f"\n[1] shallow 次日买(全样本, 始终入场): 均值 {sh:+.2f}%/{H}日, p={psh:.2e}, n={nsh}")

m5h, _, _ = summ('ma5', R[R['hit5']])
m10h, _, _ = summ('ma10', R[R['hit10']])
m20h, p20, n20 = summ('ma20', R[R['hit20']])
print(f"[2] MA5触及后买: {m5h:+.2f}% (触及率{R['hit5'].mean()*100:.0f}%)")
print(f"[3] MA10触及后买: {m10h:+.2f}% (触及率{R['hit10'].mean()*100:.0f}%)")
print(f"[4] MA20触及后买: {m20h:+.2f}% (触及率{R['hit20'].mean()*100:.0f}%, p={p20:.2e})")

# 踏空子集(不触及MA20)的shallow收益 = 等待错过的
miss = R[~R['hit20']]
mh, _, _ = summ('shallow', miss)
print(f"\n[5] MA20踏空子例({len(miss)}只, 占{len(miss)/len(R)*100:.0f}%): shallow买仍 {mh:+.2f}%/{H}日 ← 这就是'等回踩'错过的")

# 期望收益比较(等MA20: 触及才买,踏空=持币0)
exp_shallow = R['shallow'].mean() * 100
exp_ma20wait = R['hit20'].mean() * (R.dropna(subset=['ma20'])['ma20'].mean() * 100)
print(f"\n期望收益: shallow(总入场) {exp_shallow:+.2f}%  vs  等MA20(踏空=0) {exp_ma20wait:+.2f}%")
print(f"→ {'shallow胜(等回踩亏)' if exp_shallow > exp_ma20wait else '等MA20胜'}")
