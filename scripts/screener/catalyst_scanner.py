#!/usr/bin/env python3
"""催化剂scanner: 业绩预增>30%公告 ∩ 聪明钱 ∩ 主线 → 仅信息名单(追买edge已死,禁止作为入场指令)。
回测验证: 预增>30%次日买+2.32%/10日(p=1e-15), 等MA20踏空36%(backtest_catalyst_entry_shallow)。
规则: 预增>30% → 次日/MA5浅入场, 别等MA20。每日盘后跑, 次日开盘执行。"""
import sys, os, time
from datetime import date, timedelta
sys.path.insert(0, '.')
import pandas as pd, numpy as np
import tushare as ts
from scipy.signal import argrelextrema
from scripts.core.tushare_client import TushareClient

KDIR = './data/store/kline'
MAINLINE = ['有色', '铝', '铜', '煤', '电', '石油', '钢', '金属', '军工', '通信', '算', '半导体',
            '电池', '新能源', '机械', '化工', '银行', '食', '医', '汽']

cli = TushareClient()
pro = ts.pro_api()

# 1. 最新预增>30% (循环ann_date近16天)
today = date(2026, 7, 23)
chunks = []
for back in range(0, 17):
    ad = (today - timedelta(days=back)).strftime('%Y%m%d')
    try:
        f = pro.forecast(ann_date=ad, fields='ts_code,ann_date,type,p_change_min,summary')
        if f is not None and len(f):
            chunks.append(f)
    except Exception:
        pass
    time.sleep(0.1)
fc = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
fc = fc[(fc['type'] == '预增') & (fc['p_change_min'] > 30)].copy()
fc['code'] = fc['ts_code'].str.split('.').str[0]
fc['ann'] = fc['ann_date'].astype(str)
print(f"近16天 预增>30%: {len(fc)} 只")

# 2. name + industry
bb = pro.stock_basic(fields='ts_code,name,industry')
bb['code'] = bb['ts_code'].str.split('.').str[0]
nmap = dict(zip(bb['code'], bb['name']))
imap = dict(zip(bb['code'], bb['industry']))


def smart_score(close, vol, dif):
    """4特征聪明钱(简化): 底部/放量异动/底背离/金叉。"""
    s = 0
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    vma = pd.Series(vol).rolling(20).mean().values
    if not np.isnan(ma20[-1]) and close[-1] / ma20[-1] < 0.92:
        s += 1
    for j in range(-5, 0):
        if not np.isnan(vma[j]) and vma[j] > 0 and vol[j] > vma[j] * 1.3 and (close[j] / close[j - 1] - 1) > 0.04:
            s += 1
            break
    rd, rc = dif[-60:], close[-60:]
    tr = argrelextrema(rd, np.less, order=5)[0]
    if len(tr) >= 2 and rc[tr[-1]] < rc[tr[-2]] and rd[tr[-1]] > rd[tr[-2]]:
        s += 1
    if not np.isnan(ma5[-1]) and not np.isnan(ma20[-1]) and ma5[-1] > ma20[-1]:
        s += 1
    return s


rows = []
for _, r in fc.iterrows():
    code = r['code']
    p = f'{KDIR}/{code}/D.parquet'
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    if len(d) < 30:
        continue
    close = d['close'].astype(float).values
    vol = (d['volume'].astype(float).values if 'volume' in d else d['vol'].astype(float).values)
    price = float(close[-1])
    ma5 = float(pd.Series(close).rolling(5).mean().values[-1])
    ma20 = float(pd.Series(close).rolling(20).mean().values[-1])
    dif = pd.Series(close).ewm(12, adjust=False).mean().values - pd.Series(close).ewm(26, adjust=False).mean().values
    sc = smart_score(close, vol, dif)
    ind = imap.get(code, '?')
    main = any(k in str(ind) for k in MAINLINE)
    pct = (price / close[-2] - 1) * 100 if len(close) > 1 else 0
    limup = pct >= 9.5  # 涨停(主板10/创业20, 粗略)
    # 仅状态标注(2026-11-14审计:追买建议文案已删,与总账'催化预增追买edge已死'对齐)
    if limup:
        advice = '今日涨停'
    elif price > ma5 * 1.03:
        advice = '高于MA5超3%'
    else:
        advice = 'MA5附近'
    rows.append(dict(code=code, name=nmap.get(code, '?'), ann=r['ann'], pcg=float(r['p_change_min']),
                     ind=ind, main=main, price=price, vsma5=(price / ma5 - 1) * 100,
                     vsma20=(price / ma20 - 1) * 100, sc=sc, pct=pct, advice=advice))

D = pd.DataFrame(rows).sort_values(['main', 'sc', 'pcg'], ascending=[False, False, False]).reset_index(drop=True)
main_n = D['main'].sum()
print(f"有K线: {len(D)} | 其中主线相关 {main_n} 只\n")

print(f"{'代码':<7}{'名称':<9}{'公告':<9}{'预增%':>6}{'行业':<9}{'现价':>7}{'vsMA5':>7}{'vsMA20':>7}{'聪明':>4}{'今日':>6}  建议")
print('-' * 100)
for _, r in D.head(20).iterrows():
    star = '⭐' if (r['main'] and r['sc'] >= 2) else ('✅' if r['main'] else '  ')
    print(f"{star}{r['code']:<7}{r['name']:<9}{r['ann']:<9}{r['pcg']:>5.0f}%{str(r['ind'])[:7]:<9}"
          f"{r['price']:>7.2f}{r['vsma5']:>+6.1f}%{r['vsma20']:>+6.1f}%{r['sc']:>4}{r['pct']:>+5.1f}%  {r['advice']}")

print(f"\n⭐=主线+聪明钱≥2(优先) ✅=主线  | 入场: 次日/MA5浅买, 别等MA20(36%踏空)")
print(f"   预增>30%是已验证edge(+2.32%/10日p=1e-15), 聪明钱≥2加分确认")
