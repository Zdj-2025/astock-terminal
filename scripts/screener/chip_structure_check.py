#!/usr/bin/env python3
"""筹码三维检查 — 2026-10-12 上线(大波哥框架落地)
维度: ①股东户数变化率(筹码集中/分散) ②人均持股 ③前十大机构持仓变化(聪明钱方向)
数据: tushare stk_holdernumber + top10_holders + daily_basic
分级: 🔴人山人海换手 / ⚠️分散中 / ✅集中或健康
用法:
    PYTHONPATH=. python scripts/screener/chip_structure_check.py 600522.SH   # 单只
    PYTHONPATH=. python scripts/screener/chip_structure_check.py             # 持仓批量
灵感: 紫金矿业案例(9个月户数3.7倍+北向/社保减持=聪明钱把筹码导给散户, 上方全是套牢盘)
"""
import sys, time
sys.path.insert(0, '.')

INSTITUTION_KW = ['基金', '资管', '社保', '保险', 'QFII', '信托', '资产管理', '香港中央结算',
                  '中央汇金', '中国证券金融', '国新', '养老', '计划', '银行']

PORTFOLIO_STOCKS = [
    ('000100.SZ', 'TCL科技'), ('002250.SZ', '联化科技'), ('603881.SH', '数据港'),
    ('600522.SH', '中天科技'), ('688122.SH', '西部超导'), ('601888.SH', '中国中免'),
    ('601006.SH', '大秦铁路'), ('688615.SH', '合合信息'),
]


def chip_check(ts_code, tc, name=''):
    """返回 (level, 一行摘要)。level: ok/warn/danger/na"""
    # ① 户数序列
    h = tc.pro.stk_holdernumber(ts_code=ts_code, start_date='20240101', end_date='20261012')
    if h is None or not len(h):
        return 'na', f'{name or ts_code} 无户数数据'
    h = h.drop_duplicates('end_date').sort_values('end_date')
    # 只取季末口径(剔除临时披露的月末快照), end_date 形如 20260630
    h = h[h['end_date'].str.match(r'^\d{4}(0331|0630|0930|1231)$')]
    latest, latest_v = h['end_date'].iloc[-1], int(h['holder_num'].iloc[-1])
    year_ago = h[h['end_date'] <= f'{int(latest[:4])-1}{latest[4:]}']
    yoy = (latest_v / int(year_ago['holder_num'].iloc[-1]) - 1) * 100 if len(year_ago) else None
    q2 = (latest_v / int(h['holder_num'].iloc[-3]) - 1) * 100 if len(h) >= 3 else None

    # ② 人均持股(万股/户)
    db = None
    for td in ['20261009', '20261008', '20260930']:
        try:
            db = tc.pro.daily_basic(ts_code=ts_code, trade_date=td, fields='total_share')
            if len(db) and db.iloc[0]['total_share']:
                break
        except Exception:
            pass
    per_holder = float(db.iloc[0]['total_share']) * 1e4 / latest_v if db is not None and len(db) else None

    # ③ 前十大机构(最近两期)
    inst_chg, inst_detail = None, ''
    try:
        periods = ['20260630', '20260331', '20251231']
        ratios = []
        for p in periods:
            t = tc.pro.top10_holders(ts_code=ts_code, period=p)
            if t is None or not len(t):
                continue
            m = t['holder_name'].fillna('').apply(lambda s: any(k in s for k in INSTITUTION_KW))
            ratios.append((p, t.loc[m, 'hold_ratio'].sum(), m.sum()))
            if len(ratios) == 2:
                break
        if len(ratios) == 2:
            inst_chg = ratios[0][1] - ratios[1][1]
            inst_detail = f'机构top10持股{ratios[0][1]:.1f}%({ratios[0][2]}家) vs 上期{ratios[1][1]:.1f}%({ratios[1][2]}家)'
    except Exception as e:
        inst_detail = f'top10拉取失败:{e}'

    # 分级(阈值经紫金案例校准: 9个月+270%=极端)
    level, tag = 'ok', '✅'
    if yoy is not None and yoy > 80:
        level, tag = 'danger', '🔴'
    elif yoy is not None and yoy > 30:
        level, tag = '⚠️'
        if inst_chg is not None and inst_chg < -0.5:
            level = 'danger'
    elif yoy is not None and yoy < -10 and (inst_chg is None or inst_chg >= -0.5):
        tag = '✅集中'
    if q2 is not None and q2 > 150:
        tag += '(单季暴涨数据待核)'
    yoy_s = f'{yoy:+.0f}%' if yoy is not None else 'n/a'
    q2_s = f'{q2:+.0f}%' if q2 is not None else 'n/a'
    ph_s = f'{per_holder/1e4:.1f}万股/户' if per_holder else 'n/a'
    ic_s = f'机构{"增持" if inst_chg and inst_chg>0 else "减持" if inst_chg and inst_chg<0 else "持平"}{abs(inst_chg):.1f}pct' if inst_chg is not None else '机构n/a'
    line = (f'{tag} {name or ts_code} 户数{latest_v/1e4:.1f}万({latest}) 1年{yoy_s} 近2季{q2_s} '
            f'人均{ph_s} {ic_s}')
    if inst_detail:
        line += f' | {inst_detail}'
    return level, line


def main():
    from scripts.core.tushare_client import TushareClient
    tc = TushareClient()
    codes = [(sys.argv[1], '')] if len(sys.argv) > 1 else PORTFOLIO_STOCKS
    results = {'ok': [], 'warn': [], 'danger': [], 'na': []}
    for code, nm in codes:
        try:
            lvl, line = chip_check(code if '.' in code else code + ('.SH' if code.startswith('6') else '.SZ'), tc, nm)
        except Exception as e:
            lvl, line = 'na', f'{nm or code} 异常: {e}'
        results[lvl if lvl in results else 'na'].append(line)
        print(line)
        time.sleep(0.4)
    print(f'\n汇总: 🔴{len(results["danger"])} ⚠️{len(results["warn"])} ✅{len(results["ok"])} na{len(results["na"])}')
    print('规则: 1年户数+80%以上=🔴人山人海(紫金式换手); +30~80%或伴机构减=⚠️分散; 户数下降=✅集中')


if __name__ == '__main__':
    main()
