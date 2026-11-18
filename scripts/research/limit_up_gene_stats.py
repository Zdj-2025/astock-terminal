#!/usr/bin/env python3
"""连板基因回测 阶段2: 4基因显著性统计(预增/主线/超跌/小盘)。
基因(全部无未来,首板日及之前):
  g_yjyg  业绩预增>50%(首板前30日内公告)
  g_mainline 主线TOP10(首板日行业涨幅排名)
  g_oversold 超跌(距120日高<-30%)
  g_small  小盘(流通市值<80亿)
目标: 各基因单独+组合的2板/3板率,卡方p值。"""
import pandas as pd, numpy as np, glob, os
from scipy.stats import chi2_contingency
os.chdir('.')

fb = pd.read_parquet('data/store/limit_up_first_boards.parquet').reset_index(names='_i')
fb['date'] = pd.to_datetime(fb['date'])
print(f"首板样本 {len(fb)}")

# === 加载全市场kline + industry ===
sl = pd.read_parquet('data/store/stock_list.parquet')
ind_map = dict(zip(sl['symbol'], sl['industry']))
st_codes = set(sl.loc[sl['name'].astype(str).str.contains('ST'), 'symbol'])

print("加载全市场kline(算超跌+主线)...")
chunks = []
for code_dir in glob.glob('data/store/kline/*'):
    code = os.path.basename(code_dir)
    if code in st_codes or code[0] not in '036':
        continue
    try:
        d = pd.read_parquet(f'{code_dir}/D.parquet')[['date', 'close', 'change_pct']]
    except Exception:
        continue
    if len(d) < 120:
        continue
    d['code'] = code; d['industry'] = ind_map.get(code, '')
    chunks.append(d)
ALL = pd.concat(chunks, ignore_index=True)
ALL['date'] = pd.to_datetime(ALL['date'])
print(f"  ALL {len(ALL)}行 {ALL['code'].nunique()}股")

# 基因3 超跌: 距120日高
ALL['high120'] = ALL.groupby('code')['close'].transform(lambda x: x.rolling(120, min_periods=90).max())
ALL['dd120'] = ALL['close'] / ALL['high120'] - 1
# 基因2 主线: 每日行业涨幅排名
dir_ret = ALL.groupby(['date', 'industry'])['change_pct'].mean().reset_index()
dir_ret['ind_rank'] = dir_ret.groupby('date')['change_pct'].rank(ascending=False, method='min')

fb = fb.merge(ALL[['code', 'date', 'dd120']], on=['code', 'date'], how='left')
fb = fb.merge(dir_ret[['date', 'industry', 'ind_rank']], on=['date', 'industry'], how='left')

# 基因1 业绩预增
fc = pd.read_parquet('data/store/forecast/forecast_all.parquet')
fc_zz = fc[(fc['type'] == '预增') & (fc['p_change_min'] > 50)].copy()
fc_zz['code'] = fc_zz['ts_code'].str[:6]
fc_zz['ann_date'] = pd.to_datetime(fc_zz['ann_date'], format='%Y%m%d', errors='coerce')
fc_zz = fc_zz[['code', 'ann_date']].dropna().drop_duplicates()
m = fb[['_i', 'code', 'date']].merge(fc_zz, on='code', how='left')
m['gap'] = (m['date'] - m['ann_date']).dt.days
hit = set(m[(m['gap'] >= 0) & (m['gap'] <= 30)]['_i'])
fb['g_yjyg'] = fb['_i'].isin(hit)

# 基因4 小盘
print("加载daily_basic(流通市值)...")
db = pd.read_parquet('data/store/daily_basic')
print(f"  cols: {db.columns.tolist()[:8]}")
code_col = 'ts_code' if 'ts_code' in db.columns else 'symbol'
date_col = 'trade_date' if 'trade_date' in db.columns else 'date'
mv_col = 'circ_mv' if 'circ_mv' in db.columns else ('total_mv' if 'total_mv' in db.columns else None)
if mv_col:
    db2 = db[[code_col, date_col, mv_col]].copy()
    db2['code'] = db2[code_col].astype(str).str[:6]
    db2['date'] = pd.to_datetime(db2[date_col].astype(str), format='%Y%m%d', errors='coerce')
    db2 = db2[['code', 'date', mv_col]].dropna().drop_duplicates(['code', 'date'], keep='last')
    fb = fb.merge(db2, on=['code', 'date'], how='left')
    fb['g_small'] = fb[mv_col] < 800000  # 80亿(万元)
else:
    fb['g_small'] = False

fb['g_mainline'] = fb['ind_rank'] <= 10
fb['g_oversold'] = fb['dd120'] < -0.30

# === 统计 ===
n = len(fb)
print(f"\n=== 基准(全{n}首板) ===")
print(f"  2板率 {(fb['streak']>=2).mean()*100:.1f}% | 3板率 {(fb['streak']>=3).mean()*100:.1f}%")

genes = {'预增>50%': 'g_yjyg', '主线TOP10': 'g_mainline', '超跌<-30%': 'g_oversold', '小盘<80亿': 'g_small'}
print("\n=== 单基因: 2板率(有 vs 无) + 卡方p ===")
for name, col in genes.items():
    yes = fb[fb[col]]; no = fb[~fb[col]]
    if len(yes) < 30:
        print(f"  {name}: 样本不足({len(yes)})"); continue
    ry = (yes['streak'] >= 2).mean() * 100; rn = (no['streak'] >= 2).mean() * 100
    table = [[(yes['streak'] >= 2).sum(), (yes['streak'] < 2).sum()],
             [(no['streak'] >= 2).sum(), (no['streak'] < 2).sum()]]
    p = chi2_contingency(table)[1]
    sig = '✅✅' if p < 0.001 else ('✅' if p < 0.01 else ('?' if p < 0.05 else ''))
    print(f"  {name}: 有{ry:5.1f}%(n={len(yes):>5}) vs 无{rn:5.1f}% | p={p:.4f} {sig}")

print("\n=== 基因数累积: 2板率/3板率 ===")
fb['gene_cnt'] = fb[['g_yjyg', 'g_mainline', 'g_oversold', 'g_small']].sum(axis=1)
for cnt in range(5):
    sub = fb[fb['gene_cnt'] == cnt]
    if len(sub) < 20:
        continue
    print(f"  ≥{cnt}基因... " if cnt == 4 else f"  {cnt}基因: 2板{(sub['streak']>=2).mean()*100:5.1f}% 3板{(sub['streak']>=3).mean()*100:5.1f}% (n={len(sub)})")
# 累积≥3
for thr in [3, 4]:
    sub = fb[fb['gene_cnt'] >= thr]
    if len(sub) >= 10:
        print(f"  ≥{thr}基因: 2板{(sub['streak']>=2).mean()*100:5.1f}% 3板{(sub['streak']>=3).mean()*100:5.1f}% (n={len(sub)})")

# 宝鼎模式
bm = fb[fb['g_yjyg'] & fb['g_mainline'] & fb['g_oversold']]
if len(bm) >= 5:
    print(f"\n=== 宝鼎模式(预增+主线+超跌) ===")
    print(f"  2板率{(bm['streak']>=2).mean()*100:.1f}% 3板率{(bm['streak']>=3).mean()*100:.1f}% (n={len(bm)})")
fb.to_parquet('data/store/limit_up_with_genes.parquet', index=False)
print("\n存档: data/store/limit_up_with_genes.parquet")
