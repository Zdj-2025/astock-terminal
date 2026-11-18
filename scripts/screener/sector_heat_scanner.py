#!/usr/bin/env python3
"""行业热度scanner(业界标准): 全市场个股按申万行业分组,等权平均涨跌=行业指数近似.
替代scanner计数法(有筛选偏差),用市场真实涨跌判断热点(资金用脚投票).
输出: 行业涨跌排序(本周/上周/本月/变化), TOP热点+降温+逆转.
每日盘后跑,判断真实主线轮动. 全绝对路径,无cwd依赖."""
import os, glob, pandas as pd, numpy as np, tushare as ts
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
KDIR = './data/store/kline'
from scripts.core.tushare_client import get_token
ts.set_token(get_token())
pro = ts.pro_api()

# 行业映射(申万)
bb = pro.stock_basic(fields='ts_code,name,industry')
bb['code'] = bb['ts_code'].str.split('.').str[0]
ind_map = dict(zip(bb['code'], bb['industry']))

# 全市场个股算 近5日(本周)/前5日(上周)/近20日(本月) 涨跌
rows = []
for p in glob.glob(KDIR + '/*/D.parquet'):
    code = os.path.basename(os.path.dirname(p))
    industry = ind_map.get(code)
    if not industry:
        continue
    try:
        d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(d) < 25:
        continue
    c = d['close'].astype(float).values
    w = c[-1] / c[-6] - 1      # 本周(近5交易日)
    l = c[-6] / c[-11] - 1      # 上周(前5交易日)
    m = c[-1] / c[-21] - 1      # 本月(近20交易日)
    rows.append((industry, w, l, m))

D = pd.DataFrame(rows, columns=['ind', 'w', 'l', 'm'])
# 行业分组等权平均(>=10只才有代表性)
g = D.groupby('ind').agg(本周=('w', 'mean'), 上周=('l', 'mean'), 本月=('m', 'mean'), N=('w', 'count'))
g = g[g.N >= 10].copy()
g['变化'] = g['本周'] - g['上周']
g = g.sort_values('本周', ascending=False)

today = datetime.now().strftime('%Y-%m-%d')
print(f"🔥 行业热度scanner(业界标准·个股等权聚合) | {today}")
print(f"   全市场{len(D)}只个股 → {len(g)}个行业(>=10只) | 本周=近5日 上周=前5日 本月=近20日\n")

print(f"{'行业':<10}{'本周':>8}{'上周':>8}{'变化':>8}{'本月':>8}{'N':>5}")
print('-' * 50)
for ind, r in g.head(15).iterrows():
    print(f"{ind:<10}{r['本周']*100:>+7.2f}%{r['上周']*100:>+7.2f}%{r['变化']*100:>+7.2f}%{r['本月']*100:>+7.2f}%{int(r['N']):>5}")
print(f"  ... (TOP15热点)\n")

print(f"=== 📈逆转最强(上周跌→本周涨, B反主线) ===")
g2 = g.sort_values('变化', ascending=False)
for ind, r in g2.head(5).iterrows():
    print(f"  {ind:<10} 上周{r['上周']*100:+.1f}%→本周{r['本周']*100:+.1f}% (变化{r['变化']*100:+.1f}%)")

print(f"\n=== 📉降温最强(上周涨多→本周弱, 获利了结) ===")
g3 = g.sort_values('变化')
for ind, r in g3.head(5).iterrows():
    print(f"  {ind:<10} 上周{r['上周']*100:+.1f}%→本周{r['本周']*100:+.1f}% (变化{r['变化']*100:+.1f}%)")

print(f"\n=== 💀本周最弱(垫底5) ===")
for ind, r in g.tail(5).iterrows():
    print(f"  {ind:<10}{r['本周']*100:>+7.2f}%")

print(f"\n💡 业界标准(个股聚合等权), 替代scanner计数(筛选偏差). 热点=资金用脚投票的涨幅,非选股偏好.")
print(f"💡 轮动看'变化'(逆转=B反新主线, 降温=获利了结). 持仓契合度看持仓行业在TOP还是垫底.")
