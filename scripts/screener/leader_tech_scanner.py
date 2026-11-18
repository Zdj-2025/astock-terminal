#!/usr/bin/env python3
"""龙头+技术选股器 — alpha = 主线龙头 × 技术择时(2026-06-17 认知落地)。

认知(memory pure_technical_not_alpha, 经蓄能/波动/三因子/四因子系列回测收敛):
  成分股(龙头)三因子 +3.09% 夏普1.43 p<0.0001, 远强于全市场 +0.66%。
  alpha 在选股质量(龙头), 不在资金流前置(边际不大, 龙头已含资金信息)。

本扫描器聚焦 41 只龙头成分股(非全市场1940只科技):
- 选股: sector_rotation 10板块的龙头成分股(权重股)
- 择时: V3趋势加速(f1>0.3) + 回踩买点(tcl_mask)

用法: PYTHONPATH=. python scripts/screener/leader_tech_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask, STORE
from scripts.screener.sector_rotation.config import SECTOR_DEFS
from scripts.screener.sector_rotation.backtester import load_backtest_data
from scripts.core.tushare_client import TushareClient

START, END = '20250601', '20260531'


def get_leaders():
    """从sector_rotation缓存拿龙头成分股(code, 板块), 去重。"""
    data = load_backtest_data(START, END)
    leaders, seen = [], set()
    for name in SECTOR_DEFS:
        sd = data.get(name, {}).get('stocks_daily', {})
        for ts_code in sd.keys():
            code = ts_code.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
            if code not in seen:
                seen.add(code)
                leaders.append((code, name))
    return leaders


def main():
    leaders = get_leaders()
    print(f"龙头+技术选股 | {len(leaders)}只龙头成分股 | V3趋势+回踩买点")
    print("认知: alpha=主线龙头×技术(成分股三因子+3.09%夏普1.43, 回测验证)\n")
    name_map = {}
    try:
        sl = TushareClient().get_stock_list()
        name_map = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name']))
    except Exception:
        pass

    buy, watch, missing = [], [], 0
    for code, sector in leaders:
        p = STORE / code / 'D.parquet'
        if not p.exists():
            missing += 1
            continue
        try:
            df = pd.read_parquet(p)
            df['date'] = pd.to_datetime(df['date'])
            df = compute_indicators(df)
            last = df.iloc[-1]
            f1 = last['f1']
            tcl = bool(tcl_mask(df).iloc[-1])
            above_ma20 = last['close'] > last['MA20']
            info = {'code': code, 'name': name_map.get(code, ''), 'sector': sector,
                    'price': last['close'], 'f1': round(f1, 2),
                    'date': last['date'].date(),
                    'dh60': f"{last['dh60']*100:+.1f}%",
                    'vsMA20': f"{last['close']/last['MA20']*100-100:+.1f}%"}
            if f1 > 0.3 and above_ma20 and tcl:
                buy.append(info)         # 🎯龙头+回踩买点(主力信号)
            elif f1 > 0.3 and above_ma20:
                watch.append(info)       # 趋势在, 待回踩
        except Exception:
            pass

    print(f"=== 🎯 龙头+V3趋势+回踩买点(主力): {len(buy)}只 ===")
    for h in sorted(buy, key=lambda x: -x['f1']):
        print(f"  {h['code']} {h['name']:<8}[{h['sector']}] 价{h['price']:.2f} "
              f"f1={h['f1']:+.2f} dh60={h['dh60']} vsMA20={h['vsMA20']} ({h['date']})")
    print(f"\n=== 趋势在(待回踩): {len(watch)}只 ===")
    for h in sorted(watch, key=lambda x: -x['f1'])[:15]:
        print(f"  {h['code']} {h['name']:<8}[{h['sector']}] 价{h['price']:.2f} "
              f"f1={h['f1']:+.2f} dh60={h['dh60']} ({h['date']})")
    if missing:
        print(f"\n(成分股{missing}只无parquet, 已跳过)")


if __name__ == "__main__":
    main()
