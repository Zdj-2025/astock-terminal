#!/usr/bin/env python3
"""板块强度信号回测：Top3强行业 是否显著跑赢全行业基准。
无未来：score用截至当日(截面z-score),收益用forward。
全向量化(transform/cumsum),避免groupby.apply丢列。"""
import sys, os
sys.path.insert(0, '.')
import pandas as pd, numpy as np
import tushare as ts
from scipy import stats
from scripts.core.tushare_client import TushareClient

CACHE = './data/store/sector/panel.parquet'
START, END = '20240801', '20260722'

TushareClient()
pro = ts.pro_api()

if os.path.exists(CACHE):
    P = pd.read_parquet(CACHE)
    print(f"用缓存面板 {len(P)} 行 ({P['trade_date'].min()}~{P['trade_date'].max()})")
else:
    cls = pro.index_classify(level='L1', src='SW2021')
    panels = []
    import time
    for code, name in zip(cls['index_code'], cls['industry_name']):
        try:
            df = pro.index_daily(ts_code=code, start_date=START, end_date=END)
            time.sleep(0.12)
            if df is None or len(df) == 0:
                continue
            df = df[['trade_date', 'close', 'amount']].copy()
            df['sector'] = name
            panels.append(df)
        except Exception as e:
            print('skip', code, str(e)[:40])
    P = pd.concat(panels, ignore_index=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    P.to_parquet(CACHE)
    print(f"已拉取缓存 {len(P)} 行")

P['close'] = pd.to_numeric(P['close'], errors='coerce')
P['amount'] = pd.to_numeric(P['amount'], errors='coerce')
P = P.dropna(subset=['close', 'amount']).sort_values(['sector', 'trade_date']).reset_index(drop=True)

# 2. 每行业指标(向量化, 截至当日无未来)
gs = P.groupby('sector')
P['ret1'] = gs['close'].pct_change() * 100
P['ret5'] = gs['close'].pct_change(5) * 100
amt_prev5 = P['amount'].groupby(P['sector']).transform(lambda s: s.shift(1).rolling(5).mean())
P['amt_ratio'] = P['amount'] / amt_prev5
# 连续上涨天数(cumsum run-id trick)
P['_up'] = (gs['close'].diff() > 0).astype(int)
P['_nup'] = (P['_up'] == 0).astype(int)
P['run_id'] = P.groupby('sector')['_nup'].cumsum()
P['streak'] = P.groupby(['sector', 'run_id']).cumcount() + P['_up']

# 3. 每交易日截面z-score(供各变体复用)
gt = P.groupby('trade_date')
for col in ['ret5', 'amt_ratio', 'ret1']:
    m, sd = gt[col].transform('mean'), gt[col].transform('std')
    P['z_' + col] = (P[col] - m) / (sd + 1e-9)

# 4. forward收益(每行业)
for h in [1, 3, 5]:
    P[f'fwd{h}'] = P.groupby('sector')['close'].shift(-h) / P['close'] - 1
P = P.dropna(subset=['fwd1', 'fwd3', 'fwd5'])

# 5. 多变体测试
VARIANTS = {
    '综合(涨+量+连续)': lambda d: 0.35 * d['z_ret5'] + 0.30 * d['z_amt_ratio'] + 0.20 * d['z_ret1'] + 0.15 * (d['streak'] / 5),
    '纯动量5日': lambda d: d['z_ret5'],
    '纯放量(额环比)': lambda d: d['z_amt_ratio'],
    '连续天数': lambda d: d['streak'] / 5,
    '当日动量': lambda d: d['z_ret1'],
    '当日反转(买弱)': lambda d: -d['z_ret1'],
    '5日反转(买弱)': lambda d: -d['z_ret5'],
}

def eval_variant(P, score_col):
    Q = P.copy()
    Q['rank'] = Q.groupby('trade_date')[score_col].rank(ascending=False, method='first')
    rows = []
    for td, g in Q.groupby('trade_date'):
        top3 = g.nsmallest(3, 'rank')
        bot3 = g.nlargest(3, 'rank')
        rows.append({'top3': top3['fwd1'].mean(), 'all': g['fwd1'].mean(), 'bot3': bot3['fwd1'].mean()})
    D = pd.DataFrame(rows)
    exc = (D['top3'] - D['all']) * 100
    spread = (D['top3'] - D['bot3']) * 100
    t, p = stats.ttest_1samp(exc, 0)
    return exc.mean(), p, (exc > 0).mean() * 100, spread.mean()

print(f"\n样本: {P['trade_date'].nunique()} 交易日 | 测次日(1日)forward, 7个变体")
print(f"\n{'变体':<20}{'Top3超额':>9}{'p值':>9}{'胜率':>7}{'Top3-Bot3':>10}")
print('-' * 58)
for name, fn in VARIANTS.items():
    col = '_score'
    P[col] = fn(P)
    exc, p, hit, spread = eval_variant(P, col)
    P.drop(columns=[col], inplace=True)
    sig = '✅' if p < 0.05 else ('·' if p < 0.15 else '  ')
    print(f"{sig}{name:<18}{exc:>+7.3f}%{p:>9.4f}{hit:>6.1f}%{spread:>+9.3f}%")
print("\n✅=p<0.05显著 ·=p<0.15弱信号 (Top3超额=vs全行业, Top3-Bot3=多空 spread)")

