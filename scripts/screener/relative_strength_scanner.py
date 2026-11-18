#!/usr/bin/env python3
"""相对强度scanner — 逆势抗跌: 市场跌时个股不跌(有资金护盘)

逻辑: 市场下跌期间个股跑赢指数=相对强度高→市场企稳后这些票领涨.
与现有7个策略全部负相关(底部策略买弱,这个买强) — 唯一的趋势信号.
族归属: trend(与golden_cross/smart_money同族).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.core.tushare_client import TushareClient
from scripts.screener.scan_history import record_batch

STORE = PROJECT_ROOT / 'data/store/kline'
INDEX = 'data/store/index/sh000001_long.parquet'
RS_WINDOW = 20  # 相对强度计算窗口
RS_THRESHOLD = 10.0  # 跑赢指数至少5pct
MIN_ABS_RET = -15.0  # 自身也不能跌太深(排除补跌票)


def load_index():
    idx = pd.read_parquet(INDEX)
    ic = idx.set_index('d')['c'].astype(float)
    return ic


def scan_relative_strength(codes=None, record=True):
    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name'].astype(str)))
    ind_map = dict(zip(sym, sl['industry'].astype(str)))

    ic = load_index()
    idx_ret20 = float(ic.iloc[-1] / ic.iloc[-RS_WINDOW-1] - 1) * 100

    if codes is None:
        codes = sorted([d.name for d in STORE.iterdir()
                        if (d / 'D.parquet').exists() and d.name.isdigit()
                        and len(d.name) == 6 and d.name[:2] not in ('51','15','56','58')])

    hits = []
    for code in codes:
        nm = name_map.get(code, '')
        if 'ST' in nm or '退' in nm:
            continue
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            for col in ['datetime','day','trade_date']:
                if col in df.columns and col != 'date':
                    df = df.rename(columns={col:'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) < 60:
                continue
            cl = df['close'].astype(float)
            close = float(cl.iloc[-1])
            if close < 2:
                continue

            # 个股20日收益
            px20 = float(cl.iloc[-RS_WINDOW-1]) if len(cl) > RS_WINDOW else float(cl.iloc[0])
            ret20 = (close / px20 - 1) * 100

            # 相对强度 = 个股收益 - 指数收益
            rs = ret20 - idx_ret20

            # 站在MA20上方(趋势完好)
            ma20 = float(cl.tail(20).mean())
            above_ma20 = close > ma20

            if rs > RS_THRESHOLD and above_ma20 and ret20 > MIN_ABS_RET:
                hits.append({
                    'code': code, 'name': nm, 'industry': ind_map.get(code,''),
                    'price': close, 'ret20': round(ret20, 1),
                    'rs': round(rs, 1), 'idx_ret20': round(idx_ret20, 1),
                })
        except Exception:
            continue

    hits.sort(key=lambda x: x['rs'], reverse=True)  # RS最高的排前面
    return hits


def main():
    hits = scan_relative_strength()
    print(f"💪 相对强度scanner | 逆势跑赢指数>{RS_THRESHOLD}pct + 站上MA20")
    print(f"   指数20日: {hits[0]['idx_ret20'] if hits else 'N/A'}%")
    print(f"   命中: {len(hits)}只\n")
    if hits:
        print(f"{'代码':<8}{'名称':<10}{'价':>7}{'20日收益':>8}{'相对强度':>8}")
        print('-' * 50)
        for h in hits[:20]:
            print(f"  {h['code']:<8}{h['name'][:9]:<10}{h['price']:>7.2f}"
                  f"{h['ret20']:>+7.1f}%{h['rs']:>+7.1f}%")
        record_batch('relative_strength', hits)
        print(f"\n  已写入信号累积器")


if __name__ == '__main__':
    main()
