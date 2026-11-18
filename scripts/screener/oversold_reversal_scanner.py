#!/usr/bin/env python3
"""超跌反转scanner — 20日跌幅>15% + 阳线确认(最简版)

逻辑: A股月频反转因子IC=0.046(全市场最强), 20日跌>15%后前瞻10日均值正.
信号审计t=6.2. 族归属: bottom(与divergence/double_bottom同族).
2026-09-10并行化: 逐股循环走scripts/core/parallel.process_map(8进程).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.core.tushare_client import TushareClient
from scripts.screener.scan_history import record_batch

STORE = PROJECT_ROOT / 'data/store/kline'
THRESHOLD = 0.15  # 20日跌幅>15%
LOW_POS = 0.30    # 距250日高分位<30%(低位超跌才有弹性)


def _scan_one(code):
    """单只计算(worker,可pickle). 返回dict或None. 名称/ST过滤在主进程."""
    try:
        df = pd.read_parquet(STORE / code / 'D.parquet')
        for col in ['datetime', 'day', 'trade_date']:
            if col in df.columns and col != 'date':
                df = df.rename(columns={col: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 60:
            return None
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        close = float(c.iloc[-1])
        if close < 2:
            return None
        px20 = float(c.iloc[-21]) if len(c) > 21 else float(c.iloc[0])
        ret20 = close / px20 - 1
        w = c.tail(250)
        pos = (close - w.min()) / (w.max() - w.min()) if w.max() > w.min() else 0.5
        yang = close > float(o.iloc[-1])
        if ret20 < -THRESHOLD and pos < LOW_POS and yang:
            upside = float(w.max()) / close - 1
            return {
                'code': code,
                'price': close, 'ret20': round(ret20 * 100, 1),
                'pos250': round(pos * 100, 0), 'upside': round(upside * 100, 0),
            }
        return None
    except Exception:
        return None


def scan_oversold(codes=None, record=True):
    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name'].astype(str)))
    ind_map = dict(zip(sym, sl['industry'].astype(str)))

    if codes is None:
        codes = sorted([d.name for d in STORE.iterdir()
                        if (d / 'D.parquet').exists() and d.name.isdigit()
                        and len(d.name) == 6 and d.name[:2] not in ('51', '15', '56', '58')])

    from scripts.core.parallel import process_map
    results = process_map(_scan_one, codes)
    hits = []
    for r in results:
        if r is None:
            continue
        nm = name_map.get(r['code'], '')
        if 'ST' in nm or '退' in nm:
            continue
        r['name'] = nm
        r['industry'] = ind_map.get(r['code'], '')
        hits.append(r)
    hits.sort(key=lambda x: x['ret20'])  # 跌最多的排前面
    return hits


def main():
    hits = scan_oversold()
    print(f"📉 超跌反转scanner | 20日跌>{THRESHOLD * 100:.0f}% + 低位 + 阳线确认")
    print(f"   回测: t=6.2, 全信号审计验证")
    print(f"   命中: {len(hits)}只\n")
    if hits:
        print(f"{'代码':<8}{'名称':<10}{'价':>7}{'20日跌':>8}{'分位':>5}{'反弹空间':>8}")
        print('-' * 55)
        for h in hits[:20]:
            print(f"  {h['code']:<8}{h['name'][:9]:<10}{h['price']:>7.2f}"
                  f"{h['ret20']:>+7.1f}%{h['pos250']:>4.0f}%{h['upside']:>+7.0f}%")
        record_batch('oversold_reversal', hits)
        print(f"\n  已写入信号累积器(scan_history)")


if __name__ == '__main__':
    main()
