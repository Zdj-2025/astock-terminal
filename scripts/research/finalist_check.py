#!/usr/bin/env python3
"""聪明钱score5候选技术位置检查：判断回踩(可买) vs 急涨(追高)。
算 MA5/MA20/vsMA20/近5日涨幅/量比/前低。"""
import pandas as pd, numpy as np
from scipy.signal import argrelextrema

FINAL = [
    ('603993', '洛阳钼业', '小金属'),
    ('600312', '平高电气', '电网'),
    ('600219', '南山铝业', '铝'),
    ('002128', '电投能源', '煤'),
    ('000899', '赣能股份', '火电'),
    ('002847', '盐津铺子', '食品'),
]

def qfq(s):
    c = s.astype(float).copy()
    rets = pd.Series(c).pct_change().values
    for i in range(len(c) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            c[:i + 1] *= c[i + 1] / c[i]
    return c

print(f"{'代码':<7}{'名称':<9}{'行业':<6}{'现价':>7}{'MA20':>7}{'vsMA20':>8}{'5日涨':>7}{'量比':>6}{'前低':>7}{'距前低':>8}  判定")
print('-' * 95)
for code, name, ind in FINAL:
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet').sort_values('date').reset_index(drop=True)
    c = qfq(df['close'].values)
    low = qfq(df['low'].values) if 'low' in df.columns else c.copy()
    price = float(c[-1])
    ma5, ma20 = float(pd.Series(c).tail(5).mean()), float(pd.Series(c).tail(20).mean())
    ret5 = (price / c[-6] - 1) * 100 if len(c) > 6 else 0
    vol_col = 'vol' if 'vol' in df.columns else 'volume'
    vol = df[vol_col].values.astype(float)
    vr = float(vol[-1] / pd.Series(vol).tail(20).mean())
    tr = argrelextrema(low[-90:], np.less, order=5)[0]
    prev_low = float(low[-90:][tr[-1]]) if len(tr) else float(low[-30:].min())
    dist_low = (price / prev_low - 1) * 100
    vs20 = (price / ma20 - 1) * 100
    # 判定: vsMA20<3% 且 5日涨幅<8% = 回踩可买; 否则急涨追高
    if vs20 < 3 and ret5 < 8:
        tag = '✅回踩区(可买)'
    elif vs20 > 8 or ret5 > 10:
        tag = '⚠️急涨(别追)'
    else:
        tag = '中性(小仓)'
    print(f"{code:<7}{name:<9}{ind:<6}{price:>7.2f}{ma20:>7.2f}{vs20:>+7.1f}%{ret5:>+6.1f}%{vr:>6.2f}{prev_low:>7.2f}{dist_low:>+7.1f}%  {tag}")
