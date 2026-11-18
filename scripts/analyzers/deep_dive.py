#!/usr/bin/env python3
"""深度分析数据包 — 深研场景专用(2026-09-11立,用户划界:筛选≠深研).

场景边界(用户2026-09-11定):
  筛选层(mainline_leader) = 快/粗/高召回,只认硬雷(亏损/高质押/存贷双高类)
  深研层(本工具) = 慢/细/个案裁断,六查全套+估值+筹码+预期+定性问题清单
  需要人工"辨"的检查(如存货-预付-减值三角)只属于这里.

一条命令输出单票深研数据包,判断归深研报告(工具供数,不替人下结论).
用法: cd stock && PYTHONPATH=. python scripts/analyzers/deep_dive.py 688256
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd


def deep_dive(code, verbose=True):
    code = str(code).zfill(6)
    from scripts.core.tushare_client import TushareClient
    c = TushareClient()
    suffix = '.SH' if code[0] in '689' or code[:2] in ('51', '56', '58') else ('.BJ' if code[0] in '48' else '.SZ')
    tc = code + suffix

    out = {'code': code}

    # ① 六查(全套含软旗,深研场景专用)
    from scripts.analyzers.financial_forensic import FinancialForensic
    ff = FinancialForensic(use_cache=False)
    out['forensic'] = ff.check_code(code)

    # ② 归母口径利润史(finance_utils,防总净利口径事故)
    from scripts.core.finance_utils import attr_net_profit
    out['profit'] = attr_net_profit(c.pro, tc, start='20240630')

    # ③ 估值带
    db = c.pro.daily_basic(ts_code=tc, fields='trade_date,close,pe_ttm,pb,total_mv,turnover_rate',
                           start_date='20250101', end_date='20260911').dropna().sort_values('trade_date')
    r = db.iloc[-1]
    y26 = db[db['trade_date'] >= '20260101']
    out['valuation'] = {
        'date': r['trade_date'], 'close': r['close'], 'pe_ttm': round(float(r['pe_ttm']), 1),
        'pb': round(float(r['pb']), 2), 'mv_yi': round(float(r['total_mv']) / 1e4, 0),
        'pb_pct_2026': round(float(y26['pb'].rank(pct=True).iloc[-1]) * 100) if len(y26) else None,
    }

    # ④ CYQ筹码分布
    try:
        from scripts.analyzers.volume_profile import cyq_distribution, value_area
        dist, win, px = cyq_distribution(code)
        out['chips'] = {'winner_pct': win, 'price': px,
                        'va': value_area(dist),
                        'top_bins': dist.sort_values('pct', ascending=False).head(3)[
                            ['bin_left', 'bin_right', 'pct']].to_dict('records')}
    except Exception as e:
        out['chips'] = {'error': str(e)}

    # ⑤ ROE成色: 杜邦分解+5年稳定性(2026-09-11用户ROE教学落地)
    # ROE = ROA × 权益乘数(杠杆); ROA ≈ 净利率 × 周转 — 高ROE是赚的还是借的要拆开看
    try:
        fi = c.pro.fina_indicator(ts_code=tc, fields='end_date,roe,roa,debt_to_assets',
                                  start_date='20210101').drop_duplicates('end_date').sort_values('end_date')
        annual = fi[fi['end_date'].astype(str).str.endswith('1231')]
        cur = fi.iloc[-1]
        em = float(cur['roe']) / float(cur['roa']) if float(cur['roa'] or 0) != 0 else None  # 权益乘数
        out['roe'] = {
            'latest': {'end_date': cur['end_date'], 'roe': float(cur['roe']),
                       'roa': float(cur['roa']), 'equity_mult': round(em, 2) if em else None,
                       'debt_ratio': float(cur['debt_to_assets'])},
            'annual_roe_5y': {str(r['end_date'])[:4]: round(float(r['roe']), 1)
                              for _, r in annual.tail(5).iterrows()},
            'stable_above15': bool(len(annual) >= 4 and (annual.tail(4)['roe'].astype(float) > 15).all()),
        }
    except Exception as e:
        out['roe'] = {'error': str(e)}

    # ⑦ 技术位置
    try:
        k = pd.read_parquet(PROJECT_ROOT / f'data/store/kline/{code}/D.parquet')
        for col in ('datetime', 'day', 'trade_date'):
            if col in k.columns and col != 'date':
                k = k.rename(columns={col: 'date'})
        cc = k['close'].astype(float)
        px = float(cc.iloc[-1])
        w250 = cc.tail(250)
        out['tech'] = {
            'price': px, 'pos250_pct': round((px - w250.min()) / (w250.max() - w250.min()) * 100),
            'ret20': round((px / float(cc.iloc[-21]) - 1) * 100, 1),
            'ret60': round((px / float(cc.iloc[-61]) - 1) * 100, 1),
        }
    except Exception as e:
        out['tech'] = {'error': str(e)}

    if verbose:
        _print(code, out)
    return out


def _print(code, out):
    v = out['valuation']
    print(f"━━━ 深研数据包 {code} | {v['date']} ━━━")
    print(f"估值: 现价{v['close']} PE_ttm {v['pe_ttm']} PB {v['pb']}(2026年{v['pb_pct_2026']}%分位) 市值{v['mv_yi']:.0f}亿")
    t = out.get('tech', {})
    if 'price' in t:
        print(f"技术: 250日分位{t['pos250_pct']}% ret20 {t['ret20']:+.1f}% ret60 {t['ret60']:+.1f}%")
    ch = out.get('chips', {})
    if 'winner_pct' in ch:
        va = ch.get('va') or {}
        print(f"筹码: 获利盘{ch['winner_pct']}% POC={va.get('poc_bin')} 价值区{va.get('val')}-{va.get('vah')}")
    print(f"\n归母净利史(亿):")
    print(out['profit'].tail(5).to_string(index=False))
    roe = out.get('roe', {})
    if 'latest' in roe:
        L = roe['latest']
        lev_tag = '⚠️杠杆驱动' if (L.get('equity_mult') or 0) > 3 else ('杠杆温和' if (L.get('equity_mult') or 0) > 1.8 else '低杠杆')
        print(f"\nROE成色(杜邦): 最新ROE {L['roe']:.1f}% = ROA {L['roa']:.1f}% × 权益乘数 {L['equity_mult'] or '?'}({lev_tag}, 负债率{L['debt_ratio']:.0f}%)")
        print(f"  年度ROE序列: {roe.get('annual_roe_5y')} → {'✅连续4年>15%(护城河候选)' if roe.get('stable_above15') else '未达连续>15%'}")
    f = out['forensic']
    print(f"\n六查(报告期{f.get('end_date')}):")
    print(f"  {'🚩命中' if f.get('hard_hit') else '✅无硬雷'} 旗: {f.get('flags') or '无'}")
    if f.get('reason'):
        print(f"  明细: {f['reason']}")
    print(f"\n定性清单(深研报告必答,工具不替人回答):")
    print("  1.这门生意靠什么赚钱? 2.护城河哪一类,能否量化?")
    print("  3.存货/预付异动(若有⑥旗):订单备货还是计提不足?行业惯例如何?")
    print("  4.预期差在哪,市场在定价什么? 5.证伪条件与跟踪点(预注册)")


if __name__ == '__main__':
    deep_dive(sys.argv[1] if len(sys.argv) > 1 else '688256')
