#!/usr/bin/env python3
"""明天关注标的扫描：合并 7/24 scan 的全部候选(主线🥇+催化⭐)，
逐只过 7/24 技术位置 + 基本面，输出"可买位置+盈利+无高质押"的清单。
过滤: vsMA20∈[-3,+5](回踩到位,非追高非破位) + PE>0(盈利) + 质押<50%。
底背离用修后逻辑(未兑现+下跌趋势)。排序: 催化>底背离>贴近MA20。"""
import re, os, time
import pandas as pd, numpy as np
import tushare as ts
from scipy.signal import argrelextrema

KDIR = './data/store/kline'
from scripts.core.tushare_client import get_token
ts.set_token(get_token())
pro = ts.pro_api()

log = open('/tmp/scan_0724.log').read()
def sec(a, b):
    i, j = log.find(a), log.find(b)
    return log[i:j] if i >= 0 and j >= 0 else ''
cat_codes = set(re.findall(r'\b(\d{6})\b', sec('[5/6]', '[6/6]')))
main_codes = re.findall(r'\b(\d{6})\b', sec('[2/6]', '[3/6]'))
all_codes = list(dict.fromkeys(list(cat_codes) + main_codes))
print(f"候选 {len(all_codes)} 只（催化{len(cat_codes)} + 主线{len(main_codes)}）")


def qfq(s):
    c = s.astype(float).values.copy()
    r = pd.Series(c).pct_change().values
    for i in range(len(c) - 2, -1, -1):
        if not np.isnan(r[i + 1]) and r[i + 1] < -0.25:
            c[:i + 1] = c[:i + 1] * (c[i + 1] / c[i])
    return c


rows = []
for code in all_codes:
    p = f'{KDIR}/{code}/D.parquet'
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    if len(d) < 30:
        continue
    c = qfq(d['close']); l = qfq(d['low']) if 'low' in d else c.copy(); h = qfq(d['high']) if 'high' in d else c.copy()
    v = (d['volume'].astype(float) if 'volume' in d else d['vol'].astype(float)).values
    price = float(c[-1]); ma20 = float(pd.Series(c).tail(20).mean()); vs20 = (price / ma20 - 1) * 100
    tr = argrelextrema(l[-90:], np.less, order=5)[0]
    pl = float(l[-90:][tr[-1]]) if len(tr) else float(np.min(l[-30:]))
    dl = (price / pl - 1) * 100
    vr = float(v[-1] / pd.Series(v).tail(20).mean())
    ef = pd.Series(c).ewm(12, adjust=False).mean().values; es = pd.Series(c).ewm(26, adjust=False).mean().values
    dif = ef - es; rd = dif[-60:]; rc = c[-60:]; rh = h[-60:]; t = argrelextrema(rd, np.less, order=5)[0]
    div = False
    if len(t) >= 2 and rc[t[-1]] < rc[t[-2]] and rd[t[-1]] > rd[t[-2]]:  # 底背离条件
        bounce = c[-1] / rc[t[-1]] - 1; drawdown = c[-1] / max(rh) - 1
        if bounce < 0.08 and drawdown < -0.05:  # 修后: 未兑现+下跌趋势
            div = True
    rows.append(dict(code=code, price=price, vs20=vs20, dl=dl, vr=vr, div=div, cat=(code in cat_codes)))

buy = [r for r in rows if -3 <= r['vs20'] <= 5]  # 可买位置
print(f"可买位置(vsMA20∈[-3,+5]): {len(buy)} 只，查基本面(剔除亏损/高质押)...")

bb = pro.stock_basic(fields='ts_code,name,industry')
bb['code'] = bb['ts_code'].str.split('.').str[0]
nm = dict(zip(bb['code'], bb['name'])); ind = dict(zip(bb['code'], bb['industry']))
final = []
for r in sorted(buy, key=lambda x: abs(x['vs20'])):
    code = r['code']
    tc = [x for x in bb['ts_code'] if x.split('.')[0] == code]
    if not tc:
        continue
    try:
        db = pro.daily_basic(ts_code=tc[0], fields='pe_ttm')  # 默认DESC,iloc[0]=最新
        pe = float(db.iloc[0]['pe_ttm']) if db is not None and len(db) else float('nan')
    except Exception:
        pe = float('nan')
    try:
        pg = pro.pledge_stat(ts_code=tc[0])
        pledge = float(pg.iloc[0]['pledge_ratio']) if len(pg) and pg.iloc[0]['pledge_ratio'] is not None else 0.0
    except Exception:
        pledge = 0.0
    time.sleep(0.08)
    if np.isnan(pe) or pe <= 0 or pledge > 50:  # 亏损/高质押剔除
        continue
    r.update(name=nm.get(code, '?'), ind=ind.get(code, '?'), pe=pe, pledge=pledge)
    r['score'] = (3 if r['cat'] else 0) + (2 if r['div'] else 0) - abs(r['vs20']) * 0.1
    final.append(r)

final.sort(key=lambda x: -x['score'])
print(f"\n✅ 明天关注清单（可买位置 + 盈利 + 质押<50%，按催化/底背离/位置排序）")
print(f"{'代码':<7}{'名称':<9}{'行业':<8}{'现价':>7}{'vsMA20':>8}{'前低距':>7}{'底背离':>6}{'PE':>6}{'质押':>5}{'催化':>5}")
print('-' * 78)
for r in final[:14]:
    print(f"{r['code']:<7}{r['name']:<9}{str(r['ind'])[:7]:<8}{r['price']:>7.2f}{r['vs20']:>+7.1f}%{r['dl']:>+6.1f}%"
          f"{'✅' if r['div'] else '-':>6}{r['pe']:>6.0f}{r['pledge']:>4.0f}%{'预增' if r['cat'] else '-':>5}")
