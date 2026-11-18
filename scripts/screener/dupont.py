#!/usr/bin/env python3
"""杜邦三因子分析 — 2026-10-07 上线(P系列盲区审计后新增)
ROE = 净利率(NPM) × 总资产周转率(AT) × 权益乘数(EM)
用途: 识别"借出来的ROE"—— 杠杆型ROE在质量卡降档(与排雷哲学同源)。
数据: tushare fina_indicator (roe / netprofit_margin / debt_to_assets)
分类: 金融股(银行/证券/保险)天然高杠杆, 豁免扣分只标注。
"""
import sys; sys.path.insert(0, '.')

def dupont_factors(ts_code, tc=None, periods_back=0):
    """返回 dict(roe, npm, debt_ratio, em, at, dtype) 或 None。
    dtype: 质量驱动(净利率) / 周转驱动 / 杠杆驱动 / 杠杆警示 / 金融豁免"""
    if tc is None:
        from scripts.core.tushare_client import TushareClient
        tc = TushareClient()
    import pandas as pd
    # 季末序列(9/30 6/30 3/31 12/31), 从最近季末往前试, 取最新有值的
    q = pd.Period(pd.Timestamp('today'), freq='Q')
    for pb in range(periods_back, periods_back + 5):
        per = (q - pb).end_time.strftime('%Y%m%d')
        try:
            d = tc.pro.fina_indicator(ts_code=ts_code, period=per,
                fields='roe,netprofit_margin,debt_to_assets').dropna()
            if len(d) and d.iloc[0].get('roe') is not None and d.iloc[0]['roe'] == d.iloc[0]['roe']:
                r = d.iloc[0]
                roe, npm, debt = float(r['roe']), float(r['netprofit_margin'] or 0), float(r['debt_to_assets'] or 0)
                break
        except Exception:
            continue
    else:
        return None
    if npm <= 0 or debt >= 99:
        return dict(roe=roe, npm=npm, debt_ratio=debt, em=None, at=None, dtype='数据异常')
    em = 1 / (1 - debt/100)                    # 权益乘数
    at = roe / (npm * em) if em else None      # 倒推周转率(%/%=倍)
    # 驱动分类: 三因子各自相对典型水平的倍数
    n_npm, n_at, n_em = npm/8, (at or 0)/0.6, em/1.8
    if em >= 3.5:
        dtype = '杠杆警示'
    elif em >= 3.0:
        dtype = '偏杠杆'
    else:
        m = max(n_npm, n_at, n_em)
        dtype = '净利率驱动' if m == n_npm else ('周转驱动' if m == n_at else '均衡/偏杠杆')
    return dict(roe=roe, npm=npm, debt_ratio=debt, em=round(em,2), at=round(at,3) if at else None, dtype=dtype)

FIN_KEYWORDS = ('银行', '证券', '保险', '金融', '投行')

def dupont_penalty(name, dup):
    """质量卡扣分规则。返回 (扣分, 说明)。金融豁免。"""
    if not dup or dup.get('em') is None:
        return 0, ''
    if any(k in (name or '') for k in FIN_KEYWORDS):
        return 0, f"金融豁免(EM{dup['em']})"
    em = dup['em']
    if em >= 3.5:
        return -15, f"杠杆型ROE⚠️ EM{em}(负债率{dup['debt_ratio']:.0f}%)"
    if em >= 3.0:
        return -5, f"偏杠杆 EM{em}"
    return 0, f"{dup['dtype']} EM{em}"

def dupont_line(name, dup):
    """一行展示: 杜邦: ROE6.3% = 净利率7.9% × 周转0.47 × 杠杆1.69 (净利率驱动)"""
    if not dup:
        return '杜邦: 无数据'
    return (f"杜邦: ROE{dup['roe']:.1f}% = 净利率{dup['npm']:.1f}% × 周转{dup.get('at') or '?'} × 杠杆{dup.get('em') or '?'} ({dup['dtype']})")

if __name__ == '__main__':
    # 自测: 持仓+金融对照
    for code, name in [('600522.SH','中天科技'), ('603881.SH','数据港'), ('600030.SH','中信证券'), ('000100.SZ','TCL科技')]:
        d = dupont_factors(code)
        p = dupont_penalty(name, d)
        print(f"{name}: {dupont_line(name, d)} | 质量卡{p[0]}分 {p[1]}")
