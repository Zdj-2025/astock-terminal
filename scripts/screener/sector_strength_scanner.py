#!/usr/bin/env python3
"""行业强度扫描器 v1：用申万一级指数的 涨跌幅+成交额环比+连续性 合成"资金流入强度分"。
用法: python sector_strength_scanner.py [end_date YYYYMMDD]   # 不传=最新
代理指标(资金流入难直接拿):
  - 5日涨跌幅(持续力) + 当日涨跌幅(爆发力)
  - 成交额环比 = 今日/前5日均 (放量=钱在进)
  - 连续上涨天数 (真资金流入非一日游)
回放: end_date=20260722 验证能否提前抓到资源/有色资金流入。
强行业(强度分高+连续≥2日): 选龙头浅回踩/突破买,别等MA20(能源踏空教训)。"""
import sys, time
sys.path.insert(0, '.')
import pandas as pd, numpy as np
import tushare as ts
from scripts.core.tushare_client import TushareClient

END = sys.argv[1] if len(sys.argv) > 1 else '20260723'
cli = TushareClient()          # 设token
pro = ts.pro_api()

cls = pro.index_classify(level='L1', src='SW2021')
sectors = cls[['index_code', 'industry_name']]
print(f"申万一级 {len(sectors)} 行业, end_date={END}, 拉取日线...")

rows = []
for _, r in sectors.iterrows():
    code, name = r['index_code'], r['industry_name']
    try:
        df = pro.index_daily(ts_code=code, start_date='20260520', end_date=END)
        time.sleep(0.12)
        if df is None or len(df) < 7:
            continue
        df = df.sort_values('trade_date').reset_index(drop=True)
        c = df['close'].astype(float).values
        amt = df['amount'].astype(float).values
        ret1 = (c[-1] / c[-2] - 1) * 100
        ret5 = (c[-1] / c[-6] - 1) * 100 if len(c) >= 6 else np.nan
        amt_ratio = amt[-1] / np.mean(amt[-6:-1]) if len(amt) >= 7 else np.nan
        streak = 0
        for i in range(len(c) - 1, 0, -1):
            if c[i] > c[i - 1]:
                streak += 1
            else:
                break
        rows.append(dict(code=code, name=name, close=c[-1], ret1=ret1,
                         ret5=ret5, amt_ratio=amt_ratio, streak=streak,
                         date=df['trade_date'].iloc[-1]))
    except Exception as e:
        print('  skip', code, name, str(e)[:50])

D = pd.DataFrame(rows)

def z(s):
    s = pd.to_numeric(s, errors='coerce').astype(float)
    return (s - s.mean()) / (s.std() + 1e-9)

D['zr5'] = z(D['ret5']); D['za'] = z(D['amt_ratio']); D['z1'] = z(D['ret1'])
D['score'] = 0.35 * D['zr5'] + 0.30 * D['za'] + 0.20 * D['z1'] + 0.15 * (D['streak'] / 5.0)
D = D.sort_values('score', ascending=False).reset_index(drop=True)

latest = D['date'].iloc[0] if len(D) else '?'
print(f"\n📊 行业强度榜(资金流入代理) | 截至 {latest} | 分=0.35×5日涨+0.30×额环比+0.20×当日+0.15×连续")
print(f"{'#':<3}{'行业':<9}{'当日':>7}{'5日':>8}{'额环比':>7}{'连续':>5}  强度分")
print('-' * 58)
for i, r in D.head(12).iterrows():
    mark = '🥇' if i < 3 else ('  ' if r['score'] < 0.5 else '✅')
    print(f"{mark}{i+1:<3}{r['name']:<9}{r['ret1']:>+6.2f}%{r['ret5']:>+7.2f}%"
          f"{r['amt_ratio']:>6.2f}x{r['streak']:>4}日  {r['score']:>+5.2f}")

# 龙虎榜行业聚集(机构活跃方向): top_list个股 → stock_basic行业 → 分组计数
try:
    tl = pro.top_list(trade_date=str(latest))
    if tl is not None and len(tl):
        bb = pro.stock_basic(fields='ts_code,industry')
        ind_map = dict(zip(bb['ts_code'], bb['industry']))
        tl = tl.assign(industry=tl['ts_code'].map(ind_map))
        agg = tl.dropna(subset=['industry']).groupby('industry').size().sort_values(ascending=False)
        print(f"\n🐉 龙虎榜行业聚集 @ {latest} (共{len(tl)}只, 机构活跃方向):")
        for ind, n in agg.head(8).items():
            if n >= 2:
                print(f"   {ind}: {n}只")
except Exception as e:
    print('龙虎榜skip:', str(e)[:60])

print(f"\n💡 强度分>0.5 且连续≥2日 = 资金持续流入,选龙头浅回踩/突破买(别等MA20)")
print(f"   弱区(负分)不选股,或只深回踩")
