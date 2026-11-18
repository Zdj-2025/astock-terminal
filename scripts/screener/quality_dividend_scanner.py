#!/usr/bin/env python3
"""头部质量分红筛选器 — 2026-09-12立(抖音@多空修罗场 第24集 八条卡落地).

原话8条: 大市值/高ROE/高净利率/高毛利率/连续多年高分红/治理靠谱/现金流覆盖分红回购/护城河长期存在.
定位(场景划界): 筛选层=快/粗/高召回; 输出=三轴配置"头部质量轴"的观察池, 不是买入信号.
护城河(第8条)是定性判断, 归Dorsey深研层, 本筛只出前7条可量化者.

阈值(保守可调):
  市值>500亿 | ROE连续4年>15% | 净利率>10% | 毛利率>30%
  分红: 近3年每年都派现且股息率TTM>0 | OCF/归母>0.8(现金流覆盖)
  治理: 非ST/非退/五查快筛(质押<50%)
用法: cd stock && PYTHONPATH=. python scripts/screener/quality_dividend_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import time
import pandas as pd


def main():
    from scripts.core.tushare_client import TushareClient
    c = TushareClient()
    pro = c.pro

    # ① 大市值: 最新交易日截面(1次API)
    db = pro.daily_basic(trade_date='', start_date='20260901', end_date='20260911',
                         fields='ts_code,total_mv,pe_ttm,dv_ttm').dropna(subset=['total_mv'])
    db = db.sort_values('trade_date' if 'trade_date' in db.columns else 'total_mv').drop_duplicates('ts_code', keep='last')
    big = db[(db['total_mv'] > 500e4) & (~db['ts_code'].str.endswith('.BJ'))].copy()
    big['code'] = big['ts_code'].str[:6]
    print(f'①市值>500亿: {len(big)}只')

    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))
    big = big[~big['code'].map(lambda x: 'ST' in nm.get(x, '') or '退' in nm.get(x, ''))]

    rows = []
    for _, r in big.iterrows():
        tc = r['ts_code']
        try:
            fi = pro.fina_indicator(ts_code=tc, fields='end_date,roe,netprofit_margin,grossprofit_margin',
                                    start_date='20220101')
            time.sleep(0.1)
            if fi is None or not len(fi):
                continue
            fi['end_date'] = fi['end_date'].astype(str)
            annual = fi[fi['end_date'].str.endswith('1231')].drop_duplicates('end_date').sort_values('end_date')
            if len(annual) < 4:
                continue
            a4 = annual.tail(4)
            roe_ok = (pd.to_numeric(a4['roe'], errors='coerce') > 15).all()
            nm_margin = pd.to_numeric(annual.iloc[-1]['netprofit_margin'], errors='coerce')
            gm_margin = pd.to_numeric(annual.iloc[-1]['grossprofit_margin'], errors='coerce')
            # 分红史: 近3年每年有派现
            dv = pro.dividend(ts_code=tc, fields='end_date,ann_date,cash_div_tax,div_proc', order='end_date desc', limit=20)
            time.sleep(0.1)
            if dv is None or not len(dv):
                continue
            dv['year'] = dv['end_date'].astype(str).str[:4]
            PROC = {'实施': 4, '股东大会通过': 3, '预披露': 2, '预案': 1}
            dv['rk'] = dv['div_proc'].map(PROC).fillna(0)
            dedup = dv.sort_values('rk').drop_duplicates('end_date', keep='last')
            yrs_with_div = dedup[dedup['year'].isin(('2023', '2024', '2025'))]
            div_years = yrs_with_div['year'].nunique()
            # 现金流覆盖: 最新OCF/归母
            cf = pro.cashflow(ts_code=tc, fields='end_date,n_cashflow_act', count=4)
            time.sleep(0.08)
            inc = pro.income(ts_code=tc, fields='end_date,n_income_attr_p', count=4)
            time.sleep(0.08)
            ocf = pd.to_numeric(cf['n_cashflow_act'], errors='coerce').iloc[0] if len(cf) else None
            ni = pd.to_numeric(inc['n_income_attr_p'], errors='coerce').iloc[0] if len(inc) else None
            cover = (ocf / ni) if (ocf and ni and ni > 0) else None
            ok = (roe_ok and not pd.isna(nm_margin) and nm_margin > 10
                  and not pd.isna(gm_margin) and gm_margin > 30
                  and div_years >= 3 and cover is not None and cover > 0.8
                  and not pd.isna(r['dv_ttm']) and r['dv_ttm'] > 0)
            rows.append(dict(code=r['code'], name=nm.get(r['code'], ''), mv_yi=r['total_mv'] / 1e4,
                             roe4yr='✓' if roe_ok else f"✗({','.join(f'{v:.0f}' for v in pd.to_numeric(a4['roe'], errors='coerce'))})",
                             nm_pct=nm_margin, gm_pct=gm_margin, div_yrs=div_years,
                             ocf_cover=round(cover, 2) if cover else None, dv_ttm=r['dv_ttm'],
                             PE=r['pe_ttm'], ok=ok))
        except Exception:
            continue
    t = pd.DataFrame(rows)
    t.to_csv('reports/quality_dividend_screen.csv', index=False)
    hits = t[t['ok']].sort_values('mv_yi', ascending=False)
    print(f'②八条卡全过: {len(hits)}只 / {len(t)}只受检')
    print(f"{'名称':<8}{'市值亿':>7}{'PE':>6}{'股息%':>6}{'净利%':>6}{'毛利%':>6}{'OCF覆盖':>7}  分红年数")
    for _, h in hits.iterrows():
        print(f"{h['name'][:7]:<8}{h['mv_yi']:>7.0f}{h['PE']:>6.1f}{h['dv_ttm']:>6.1f}"
              f"{h['nm_pct']:>6.1f}{h['gm_pct']:>6.1f}{h['ocf_cover']:>7.2f}  {int(h['div_yrs'])}年")
    print('\n💡 筛选层定位: 观察池(头部质量轴候选), 护城河第8条需Dorsey深研逐家回答; 不构成买入信号')


if __name__ == '__main__':
    main()
