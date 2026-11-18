#!/usr/bin/env python3
"""利润增速二阶导告警 — P1-3(2026-10-18, 张湧财报四要点增量·茅台2018案例)
检测: 本期净利增速 vs 上期骤降>30pct 且上期>20% = ⚠️增速失速(抱团股杀手)
用途: 三季报披露时跑, 持仓+watchlist优先
"""
import sys; sys.path.insert(0, '.')

def growth_decel(ts_code, tc=None):
    """返回 (本期增速%, 上期增速%, 降速pct, 告警bool)"""
    if tc is None:
        from scripts.core.tushare_client import TushareClient
        tc = TushareClient()
    import pandas as pd
    # 近5期累计净利同比(npy)
    d = tc.pro.fina_indicator(ts_code=ts_code, fields='end_date,npy')
    d = d.dropna().sort_values('end_date')
    # 取年报口径(1231)序列算YoY最稳, 简化: 用最近两期npy的增速对比
    if len(d) < 2: return None
    cur, prev = d.iloc[-1], d.iloc[-2]
    # 累计净利同比直接有: dbrg? 用 netprofit_yoy 更准, 尝试
    try:
        d2 = tc.pro.fina_indicator(ts_code=ts_code, fields='end_date,netprofit_yoy')
        d2 = d2.dropna().sort_values('end_date').drop_duplicates('end_date', keep='last')
        if len(d2) >= 2:
            g_now, g_prev = float(d2.iloc[-1]['netprofit_yoy']), float(d2.iloc[-2]['netprofit_yoy'])
            drop = g_prev - g_now
            warn = g_prev > 20 and drop > 30
            return g_now, g_prev, drop, warn
    except Exception:
        pass
    return None

if __name__ == '__main__':
    from scripts.core.tushare_client import TushareClient
    tc = TushareClient()
    import time
    POS = [('600522.SH','中天科技'),('002250.SZ','联化科技'),('688122.SH','西部超导'),
           ('601888.SH','中国中免'),('000100.SZ','TCL科技')]
    for code, name in POS:
        r = growth_decel(code, tc)
        if r:
            g_now, g_prev, drop, warn = r
            tag = '🚨茅台式失速!' if warn else ('⚠️注意' if drop>20 else '✅')
            print(f'{name}: 最新增速{g_now:+.0f}% vs 上期{g_prev:+.0f}% (降{drop:.0f}pct) {tag}')
        else:
            print(f'{name}: 数据不足')
        time.sleep(0.3)
