#!/usr/bin/env python3
"""突破启动scanner — 找当前满足"有效突破"信号的股。

回测验证(backtest_accumulation.py, 127万样本):
- 突破启动(平台+新高+放量): +1.29% vs基准+1.09%, p=0.0001 ✅有效
- 高位突破(距120高>-5%): +0.71% vs+1.09%, p=0.013 ⚠️陷阱(末升段)
所以: 取突破信号 + 排除高位。

信号(全部同步量价, 无未来, 与回测一致):
- 平台: 前20日振幅(close.max/min-1)<15%
- 突破: 当日close > 前20日最高close
- 放量: 当日量 > 20日均量×1.5
- 非高位: 距120日高点<-5%(避开高位陷阱)

定位: 借机构拉升力(突破=机构启动, 同步留痕)。触发后需AI深度验证(板块/资金/背离/避雷), 不全信。
用法: cd stock && PYTHONPATH=. python scripts/screener/breakout_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.core.tushare_client import TushareClient
from scripts.screener.mainline_leader_scanner import mine_filter

STORE = PROJECT_ROOT / 'data/store/kline'


def scan_breakout():
    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))
    ind_map = dict(zip(sym, sl['industry']))

    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    hits = []
    for code in codes:
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet', columns=['date', 'close', 'volume'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) < 125:
                continue
            close = df['close'].astype(float)
            vol = df['volume'].astype(float)
            i = len(df) - 1
            prev20 = close.iloc[i - 20:i]          # 前20日(不含当日)
            platform = (prev20.max() / prev20.min() - 1) < 0.15
            brk = close.iloc[i] > prev20.max()      # 突破前20日最高
            vol_ma = vol.iloc[i - 20:i].mean()
            # 2026-08-28回测: 放量过滤让突破更差(+0.54%→+0.30%),停用
            # surge = vol.iloc[i] > vol_ma * 1.5
            surge = True  # 不再作为入选条件,仅显示
            hi120 = close.iloc[i - 120:i].max()
            not_high = (close.iloc[i] / hi120 - 1) < -0.05   # 距120高<-5%
            if platform and brk and surge and not_high:
                hits.append({
                    'code': code, 'name': name_map.get(code, ''), 'industry': ind_map.get(code, ''),
                    'price': float(close.iloc[i]),
                    'vol_surge': float(vol.iloc[i] / vol_ma),
                    'dist_hi120': (close.iloc[i] / hi120 - 1) * 100,
                    'ret20': (close.iloc[i] / close.iloc[i - 21] - 1) * 100 if i >= 21 else 0,
                    'date': str(df['date'].iloc[i])[:10],
                })
        except Exception:
            continue
    return hits


def _fund_for_mine(hits):
    """取PE/质押(仅给mine_filter硬排除用, 不打分)。"""
    if not hits:
        return
    import tushare as ts
    from scripts.analyzers.news_analyzer import NewsAnalyzer
    pro = ts.pro_api(); na = NewsAnalyzer()
    codes = [h['code'] for h in hits]
    db = {}
    try:
        latest = pro.daily_basic(ts_code=f"{codes[0]}.{'SH' if codes[0][:1] in '6' else 'SZ'}", fields='trade_date').head(1).iloc[0]['trade_date']
        dd = pro.daily_basic(trade_date=latest, fields='ts_code,pe_ttm')
        db = {r['ts_code'][:6]: r['pe_ttm'] for _, r in dd.iterrows()}
    except Exception:
        pass
    pz = {}
    for code in codes:
        try:
            p = na.get_pledge_info(code)
            if p is not None and not p.empty and '状态' in p.columns:
                wk = p[p['状态'].astype(str).str.contains('未解押')]
                pz[code] = pd.to_numeric(wk['占所持股份比例'], errors='coerce').max() if not wk.empty else 0
            else:
                pz[code] = 0
        except Exception:
            pz[code] = 0
    for h in hits:
        h['pe'] = db.get(h['code']); h['pledge'] = pz.get(h['code'], 0)


def main():
    from scripts.research.breakout_quality import breakout_quality
    hits = scan_breakout()
    if not hits:
        print("无突破信号"); return
    # 硬排除(与mainline/acceleration一致: 亏损/高质押/面值/暴跌雷, 2026-07-10补漏)
    _fund_for_mine(hits)
    _mine = [(h, mine_filter(h.get('pe'), h.get('pledge'), h.get('price'), h.get('name'), h.get('dist_hi120'))) for h in hits]
    filtered = [h for h, (p, r) in _mine if not p]
    for h, (p, r) in _mine:
        if not p:
            h['mine_reason'] = r
    hits = [h for h, (p, r) in _mine if p]
    if filtered:
        from collections import Counter
        cnt = Counter(h['mine_reason'] for h in filtered)
        print(f"   ⚠️硬排除{len(filtered)}只雷(突破也不碰): {dict(cnt)} | 剩{len(hits)}只打股性")
    # 股性验证(历史突破胜率, AI维度: 散户只看当下K线看不出股性)
    for h in hits:
        q = breakout_quality(h['code'])
        h['win'] = q['win'] if q else None
        h['hmean'] = q['mean'] if q else None

    def qscore(h):
        if h['win'] is not None and h['win'] >= 55 and h['hmean'] > 1:
            return 0  # 真突破(股性好)
        if h['win'] is not None and h['win'] < 40:
            return 2  # 假突破(股性差)
        return 1  # 中性

    hits.sort(key=lambda h: (qscore(h), -h['vol_surge']))
    print(f"📊 突破启动scanner | 平台+突破+放量+非高位+股性验证")
    print(f"   有效突破: {len(hits)}只(按股性排序: ★真/中性/⚠️假)\n")
    print(f"{'代码':<7}{'名称':<10}{'行业':<10}{'价':>8}{'放量':>6}{'距120高':>9}{'历史胜率':>8}{'历史均值':>9}")
    for h in hits:
        qm = f"{h['win']:.0f}%" if h['win'] is not None else '   -'
        qmm = f"{h['hmean']:+.1f}%" if h['hmean'] is not None else '    -'
        star = '★真' if qscore(h) == 0 else ('⚠️假' if qscore(h) == 2 else '  ?')
        print(f"{h['code']:<7}{h['name'][:9]:<10}{h['industry'][:8]:<10}{h['price']:>8.2f}"
              f"{h['vol_surge']:>5.1f}x{h['dist_hi120']:>+8.1f}%{qm:>8}{qmm:>9} {star}")
    good = [h for h in hits if qscore(h) == 0]
    print(f"\n💡 股性验证后: {len(good)}只真突破(历史胜率≥55%且均值>+1%) — 只跟这些, 避开⚠️假突破")


if __name__ == '__main__':
    main()
