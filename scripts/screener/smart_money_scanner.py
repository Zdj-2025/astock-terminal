#!/usr/bin/env python3
"""利好预知scanner — 扫全市场聪明钱痕迹信号(可能有利好未公告)。

回测验证(smart_money_backtest.py, 1744预增公告, p=2.7e-91, lift3.36x):
  业绩预增公告前 score≥3 触发率 71.5% vs 随机 21.2%。
核心特征(实验组触发率): 金叉89.6% + 稳步抬升81.4% + 量价配合57% + 放量异动56%。
(底背离仅15.9%是加分项; 底部区域3.1%无用已忽略)

定位: 信息不对称探测器——在利好公告前13-20天发现候选(浪潮case公告前13天score=5)。
叠加 mine_filter 避雷(亏损/ST/低价)。质押/公告跟进时人工查(scanner初筛)。

用法: cd stock && PYTHONPATH=. python scripts/screener/smart_money_scanner.py
"""
import sys
import math
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import tushare as ts
from scripts.core.tushare_client import TushareClient
from scripts.screener.mainline_leader_scanner import mine_filter
from scripts.research.smart_money_backtest import smart_money_score, load

STORE = PROJECT_ROOT / 'data/store/kline'
pro = ts.pro_api()


def scan_smart_money(threshold=4):
    """扫全市场, 返回 score>=threshold 且过mine_filter的票。"""
    tc = TushareClient()
    sl = tc.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))
    ind_map = dict(zip(sym, sl['industry']))
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])

    # 最新PE/市值(全市场单日daily_basic)
    latest = None
    for code in codes[:3]:
        try:
            d = pro.daily_basic(ts_code=f"{code}.{'SH' if code[:1] in '6' else 'SZ'}", fields='trade_date').head(1)
            if not d.empty and (latest is None or d.iloc[0]['trade_date'] > latest):
                latest = d.iloc[0]['trade_date']
        except Exception:
            pass
    db = {}
    if latest:
        try:
            dd = pro.daily_basic(trade_date=latest, fields='ts_code,pe_ttm,circ_mv')
            db = {r['ts_code'][:6]: (r['pe_ttm'], r['circ_mv'] / 1e4 if pd.notna(r['circ_mv']) else None) for _, r in dd.iterrows()}
        except Exception:
            pass

    hits = []
    ge3 = 0
    for i, code in enumerate(codes):
        try:
            df = load(code)
            if df is None or len(df) < 60:
                continue
            t = len(df) - 1
            s, f = smart_money_score(df, t)
            if s >= 3:
                ge3 += 1
            if s < threshold:
                continue
            pe, mv = db.get(code, (None, None))
            price = float(df['close'].iloc[-1])
            nm = name_map.get(code, '')
            # mine_filter避雷(亏损/ST/低价; 质押跟进时查, 此处=0)
            passed, reason = mine_filter(pe, 0, price, nm, None)
            if not passed:
                continue
            hits.append({'code': code, 'name': nm, 'industry': ind_map.get(code, ''),
                         'price': price, 'score': s, 'features': f, 'pe': pe, 'mv': mv})
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}] 命中{len(hits)}", flush=True)
    return hits, latest, ge3


def main():
    hits, latest, ge3 = scan_smart_money(threshold=4)
    hits.sort(key=lambda h: (-h['score'], -(h['mv'] or 0)))
    print(f"\n📊 利好预知scanner | 聪明钱痕迹 score≥4 | 数据{latest}")
    print(f"   回测依据: 预增公告前 score≥4 实验34.3% vs 随机7.5%(lift高, p=2.7e-91)")
    print(f"   全市场 score≥3: {ge3}只 | score≥4过避雷: {len(hits)}只\n")
    print(f"{'代码':<7}{'名称':<10}{'行业':<10}{'价':>8}{'score':>6}{'PE':>6}{'市值亿':>7}  触发特征")
    print('-' * 86)
    for h in hits[:30]:
        trig = '/'.join([k for k, v in h['features'].items() if v])
        pe = h['pe']
        if pe is None or (isinstance(pe, float) and math.isnan(pe)):
            pe_s = 'NaN'
        elif pe > 0:
            pe_s = f"{pe:.0f}"
        else:
            pe_s = '亏'
        mv_s = f"{h['mv']:.0f}" if h['mv'] else '-'
        star = '⭐' if h['score'] >= 5 else ('◆' if h['score'] == 4 else ' ')
        print(f"{star}{h['code']:<6}{h['name'][:9]:<10}{h['industry'][:8]:<10}{h['price']:>8.2f}{h['score']:>5}{pe_s:>6}{mv_s:>7}  {trig}")
    print(f"\n⭐score=5(五信号共振,最强) | ◆score=4(强信号)")
    print(f"💡 这些票有聪明钱埋伏痕迹, 可能有未公告利好。跟进: 查近期公告+质押+60min背离, 回踩买入。")


if __name__ == '__main__':
    main()
