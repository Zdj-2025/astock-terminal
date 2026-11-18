#!/usr/bin/env python3
"""方法论验证(无作弊): 用7/27收盘数据筛"回踩到位+缩量+盈利+主线"模板,
然后查7/28股价验证。screen只用7/27数据(parquet+tushare到7/27), 7/28价格单独curl。
模板=回踩(vsMA20∈[-3,+5])+缩量(量比<0.9)+盈利(PE>0)+无高质押(<50%)+主线。"""
import os, glob, pandas as pd, numpy as np, tushare as ts, time
KDIR = './data/store/kline'
from scripts.core.tushare_client import get_token
ts.set_token(get_token())
pro = ts.pro_api()
MAIN = ['食', '消费', '饮', '医', '药', '军', '航空', '航天', '船舶', '电', '有色', '煤', '钢',
        '石油', '石', '化工', '化纤', '机械', '汽车', '通信', '算', '半导体', '元件', '电气',
        '仪器', '塑料', '橡胶', '纺织', '家电', '轻工', '玻璃', '建材', '运输', '物流']

def qfq(s):
    c = s.astype(float).values.copy()
    r = pd.Series(c).pct_change().values
    for i in range(len(c) - 2, -1, -1):
        if not np.isnan(r[i + 1]) and r[i + 1] < -0.25:
            c[:i + 1] = c[:i + 1] * (c[i + 1] / c[i])
    return c

# stage1: 全市场扫 回踩+缩量 (7/27收盘)
cands = []
files = glob.glob(KDIR + '/*/D.parquet')
for p in files:
    code = os.path.basename(os.path.dirname(p))
    try:
        df = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(df) < 25:
        continue
    c = qfq(df['close'])
    v = (df['volume'].astype(float) if 'volume' in df else df['vol'].astype(float)).values
    price = float(c[-1]); ma20 = float(pd.Series(c).tail(20).mean())
    vs20 = (price / ma20 - 1) * 100
    vr = float(v[-1] / pd.Series(v).tail(20).mean())
    if -3 <= vs20 <= 5 and vr < 0.9:  # 回踩到位 + 缩量
        cands.append((code, price, vs20, vr))
print(f"7/27收盘 全市场扫: 回踩+缩量 {len(cands)} 只 / 总 {len(files)}")

# 按贴近MA20+缩量程度排序, 取前40查基本面(控制API调用)
cands.sort(key=lambda x: (abs(x[2]), -x[3]))
bb = pro.stock_basic(fields='ts_code,name,industry')
bb['code'] = bb['ts_code'].str.split('.').str[0]
nm = dict(zip(bb['code'], bb['name'])); ind = dict(zip(bb['code'], bb['industry']))
final = []
for code, price, vs20, vr in cands[:45]:
    name = nm.get(code); industry = ind.get(code, '')
    if not name or not any(k in str(industry) for k in MAIN):
        continue
    tc = [x for x in bb['ts_code'] if x.split('.')[0] == code]
    if not tc:
        continue
    try:
        db = pro.daily_basic(ts_code=tc[0], fields='pe_ttm')
        pe = float(db.iloc[0]['pe_ttm']) if db is not None and len(db) else float('nan')
    except Exception:
        pe = float('nan')
    try:
        pg = pro.pledge_stat(ts_code=tc[0])
        pledge = float(pg.iloc[0]['pledge_ratio']) if len(pg) and pg.iloc[0]['pledge_ratio'] is not None else 0
    except Exception:
        pledge = 0
    time.sleep(0.06)
    if np.isnan(pe) or pe <= 0 or pledge > 50:
        continue
    final.append((code, name, str(industry)[:7], price, vs20, vr, round(pe), pledge))

print(f"\n✅ 模板命中(回踩+缩量+盈利+主线, 7/27收盘) {len(final)} 只:")
print(f"{'代码':<7}{'名称':<10}{'行业':<8}{'7/27价':>8}{'vsMA20':>8}{'量比':>6}{'PE':>5}{'质押':>5}")
for c in final:
    print(f"{c[0]:<7}{c[1]:<10}{c[2]:<8}{c[3]:>8.2f}{c[4]:>+7.1f}%{c[5]:>6.2f}{c[6]:>5}{c[7]:>4.0f}%")
# 输出代码列表供7/28验证
print("\n验证代码:", ",".join(c[0] for c in final))
