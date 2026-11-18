#!/usr/bin/env python3
"""八只手排雷三查 — 2026-10-18(陈杰森视频落地)
①商誉占净资产>30%(第2手·洗澡) ②质押率>50%(第4手·回购对照) ③未来90日解禁(第3手·利好对照)
用法: PYTHONPATH=. python scripts/screener/invisible_hands_check.py [codes...]
默认跑持仓+超跌池头部; 推荐前必跑
"""
import sys, time; sys.path.insert(0, '.')

DEFAULT = ['600522.SH','002250.SZ','688122.SH','601888.SH','000100.SZ','601006.SH','688615.SH',
           '601138.SH','603986.SH','300433.SZ']

def check(ts_code, tc):
    gw_ratio, pledge = -1.0, -1.0
    try:
        bs = tc.pro.balancesheet(ts_code=ts_code, fields='end_date,goodwill,total_hldr_eqy_exc_min_int')
        bs = bs.dropna(subset=['end_date']).sort_values('end_date')
        if len(bs):
            r = bs.iloc[-1]
            gw_ratio = float(r['goodwill'] or 0)/max(float(r['total_hldr_eqy_exc_min_int'] or 1),1)*100
    except Exception: pass
    try:
        pl = tc.pro.pledge_stat(ts_code=ts_code)
        if len(pl): pledge = float(pl.iloc[0]['pledge_ratio'] or 0)
    except Exception: pass
    unlocks = []
    try:
        f = tc.pro.share_float(ts_code=ts_code, start_date='20261018', end_date='20270118')
        for _, r in (f.iterrows() if len(f) else []):
            unlocks.append(f'{r["float_date"]}解禁{float(r["float_ratio"] or 0):.1f}%')
    except Exception: pass
    flags = ([f'🚩商誉{gw_ratio:.0f}%'] if gw_ratio > 30 else []) + ([f'🚩质押{pledge:.0f}%'] if pledge > 50 else []) + ([f'⚠️{u}' for u in unlocks])
    return gw_ratio, pledge, unlocks, flags

if __name__ == '__main__':
    from scripts.core.tushare_client import TushareClient
    tc = TushareClient()
    codes = sys.argv[1:] or DEFAULT
    print(f'{"代码":<11}{"商誉/净资产":>9}{"质押率":>7}  三查')
    for c in codes:
        if '.' not in c: c += '.SH' if c.startswith('6') else '.SZ'
        gw, pl, un, fl = check(c, tc)
        print(f'{c:<11}{gw:>8.1f}%{pl:>6.1f}%  {" ".join(fl) if fl else "✅"}')
        time.sleep(0.3)
