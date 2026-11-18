#!/usr/bin/env python3
"""主升信号扫描器 — V3二阶导f1>0+f2>0(主升) + 放量 + 基本面质量 + 双信号共振标注。

回测验证(v3_derivative_backtest.py, 127万样本, 扣0.3%成本):
- 主升(f1>0+f2>0): 净+1.00% 夏普0.45 p=0.0000 ✅有效(追涨策略)
- 与底背离(divergence_scanner)互补: 底背离=抄底, 主升=追涨/做T

双信号共振: 同时出现在divergence_scanner底背离🥇 + 主升信号 = 最强组合(从底部进入主升)。
标记为⭐双共振重点推荐。

定位: 追涨/加仓/做T策略。仓位减半, 3S止盈(涨减速即卖)。
用法: cd stock && PYTHONPATH=. python scripts/screener/acceleration_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tushare as ts
from scipy.signal import savgol_filter
from scripts.core.tushare_client import TushareClient
from scripts.analyzers.news_analyzer import NewsAnalyzer
from scripts.screener.mainline_leader_scanner import quality_scorecard, mine_filter

STORE = PROJECT_ROOT / 'data/store/kline'
pro = ts.pro_api()
SG_WINDOW = 21
SG_POLY = 3


def scan_acceleration():
    """扫描主升信号(f1>0+f2>0+放量+非极端高位)。返回 list[dict]。"""
    _tc = TushareClient()
    sl = _tc.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))
    ind_map = dict(zip(sym, sl['industry']))

    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
    hits = []
    for i, c in enumerate(codes):
        try:
            df = pd.read_parquet(STORE / c / 'D.parquet')
            df = df.sort_values('date').reset_index(drop=True) if 'date' in df.columns else df
            for col in ['datetime', 'day', 'trade_date']:
                if col in df.columns and col != 'date':
                    df = df.rename(columns={col: 'date'})
            if len(df) < 150:
                continue
            close = df['close'].astype(float).ffill()
            ln = np.log(close)
            from scripts.core.derivatives import causal_savgol  # 2026-09-10: 权威因果, 弃中心化+shift滞后
            f1_val = float(causal_savgol(ln.values, SG_WINDOW, SG_POLY, deriv=1)[-1] * 100)
            f2_val = float(causal_savgol(ln.values, SG_WINDOW, SG_POLY, deriv=2)[-1] * 10000)
            if np.isnan(f1_val) or np.isnan(f2_val):
                continue
            # 主升: f1>0 + f2>0
            if f1_val <= 0 or f2_val <= 0:
                continue
            # 放量
            vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
            vol_ma = vol.rolling(20).mean().iloc[-1]
            if not (vol.iloc[-1] > vol_ma * 1.2):
                continue
            # 非极端高位
            hi120 = close.rolling(120).max().iloc[-1]
            dist120 = (close.iloc[-1] / hi120 - 1) * 100
            if dist120 > 0:  # 创新高=末升段,排除
                continue
            # 避雷
            nm = name_map.get(c, '')
            if 'ST' in nm or '退' in nm:
                continue
            if close.iloc[-1] < 2 or dist120 < -75:
                continue
            hits.append({'code': c, 'name': nm, 'industry': ind_map.get(c, ''),
                         'price': float(close.iloc[-1]), 'f1': float(f1_val), 'f2': float(f2_val),
                         'dist120': float(dist120), 'vol_ratio': float(vol.iloc[-1] / vol_ma),
                         'date': str(df['date'].iloc[-1])[:10]})
        except Exception:
            continue
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    return hits


def add_fundamentals(hits):
    """对主升信号叠加基本面质量打分 + 质押。"""
    na = NewsAnalyzer()
    codes = [h['code'] for h in hits]
    # PE/PB/市值
    latest = None
    for code in codes[:5]:
        try:
            d = pro.daily_basic(ts_code=f"{code}.{'SH' if code[:1] in '6' else 'SZ'}", fields='trade_date').head(1)
            if not d.empty and (latest is None or d.iloc[0]['trade_date'] > latest):
                latest = d.iloc[0]['trade_date']
        except Exception:
            pass
    db = {}
    if latest:
        try:
            dd = pro.daily_basic(trade_date=latest, fields='ts_code,pe_ttm,pb,circ_mv')
            db = {r['ts_code'][:6]: (r['pe_ttm'], r['pb'], r['circ_mv'] / 1e4 if r['circ_mv'] else None)
                  for _, r in dd.iterrows()}
        except Exception:
            pass
    # 质押
    pz = {}
    for i, code in enumerate(codes):
        try:
            p = na.get_pledge_info(code)
            if p is not None and not p.empty and '状态' in p.columns:
                wk = p[p['状态'].astype(str).str.contains('未解押')]
                pz[code] = pd.to_numeric(wk['占所持股份比例'], errors='coerce').max() if not wk.empty else 0
            else:
                pz[code] = 0
        except Exception:
            pz[code] = 0
        if (i + 1) % 50 == 0:
            print(f"  [质押 {i+1}/{len(codes)}]")
    # 打分
    for h in hits:
        pe, pb, mv = db.get(h['code'], (None, None, None))
        pledge = pz.get(h['code'], 0)
        h['pe'] = pe; h['pb'] = pb; h['mv'] = mv; h['pledge'] = pledge
        h['quality'] = quality_scorecard(pe, pb, mv, None, None, pledge, False)
    return latest


def check_divergence_overlap(hits):
    """检查哪些主升票同时出现在底背离🥇(双信号共振)。返回 set of codes。"""
    try:
        from scripts.screener.divergence_scanner import scan_divergence
        div_hits = scan_divergence()
        div_confirmed = {h['code'] for h in div_hits if h.get('confirmed') and h.get('low_pos')}
        return div_confirmed
    except Exception:
        return set()


def main():
    print("📊 主升信号扫描器 | V3 f1>0+f2>0(加速度) + 放量 + 基本面质量 + 双信号共振")
    print("   回测: 主升净+1.00% 夏普0.45(追涨策略, 仓位减半+3S止盈)\n")
    hits = scan_acceleration()
    if not hits:
        print("无主升信号"); return
    latest = add_fundamentals(hits)
    # 硬排除(与mainline一致: 亏损/高质押/面值/暴跌雷, 追涨也不碰雷)
    _mine = [(h, mine_filter(h.get('pe'), h.get('pledge'), h.get('price'), h.get('name'), h.get('dist120'))) for h in hits]
    filtered = [h for h, (p, r) in _mine if not p]
    for h, (p, r) in _mine:
        if not p:
            h['mine_reason'] = r
    hits = [h for h, (p, r) in _mine if p]
    if filtered:
        from collections import Counter
        cnt = Counter(h['mine_reason'] for h in filtered)
        print(f"   ⚠️硬排除{len(filtered)}只雷(追涨也不碰): {dict(cnt)} | 剩{len(hits)}只")
    div_codes = check_divergence_overlap(hits)
    # 标注双共振
    for h in hits:
        h['dual'] = h['code'] in div_codes
    # 排序: 双共振优先 > 质量高 > f1强
    hits.sort(key=lambda h: (not h['dual'], -h['quality'], -h['f1']))

    dual = [h for h in hits if h['dual']]
    print(f"   主升信号: {len(hits)}只 | ⭐双共振(底背离+主升): {len(dual)}只 | 数据{latest}")
    print(f"   双共振=从底部进入主升的最强组合\n")
    print(f"{'代码':<7}{'名称':<10}{'行业':<10}{'价':>7}{'f1':>5}{'f2':>5}{'距120高':>8}{'质量':>5}{'PE':>5}{'质押':>5}  标记")
    print('-' * 80)
    for h in hits[:30]:
        pe_s = f"{h['pe']:.0f}" if h['pe'] and h['pe'] > 0 else '亏'
        pz_s = f"{h['pledge']:.0f}%" if h['pledge'] is not None else '-'
        tag = '⭐双共振重点!' if h['dual'] else ''
        print(f"{h['code']:<7}{h['name'][:9]:<10}{h['industry'][:8]:<10}{h['price']:>7.2f}"
              f"{h['f1']:>+4.1f}{h['f2']:>+4.0f}{h['dist120']:>+7.0f}%{h['quality']:>5}{pe_s:>5}{pz_s:>5}  {tag}")
    print(f"\n💡 主升=追涨策略(夏普0.45), 仓位减半+3S止盈。⭐双共振=底背离→主升的最强组合。")


if __name__ == '__main__':
    main()
