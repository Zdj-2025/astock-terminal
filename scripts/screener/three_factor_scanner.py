#!/usr/bin/env python3
"""三因子精选扫描器 — 真 alpha = 主线 × 选股 × 择时(2026-06-17)。

认知(memory feedback_pure_technical_not_alpha): 纯技术形态非alpha, 三因子叠加才是 TCL 6/1 路径。
1. 选股质量: V3趋势加速(f1>0.3 + above_ma20) — 回测+6.94%(系统核心)
2. 主线催化: 科技赛道(AI/半导体/通信/电气设备/机器人/航天)
3. 技术择时: TCL回踩买点(回踩MA20 + MACD金叉 + dh60中等, 不追高)

三者交集 = "现在的TCL"。任一因子单独不够(已回测证伪), 乘积才是alpha。

用法: PYTHONPATH=. python scripts/screener/three_factor_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask, load_codes, STORE
from scripts.core.tushare_client import TushareClient

# 科技赛道行业(tushare industry字段)
TECH_INDUSTRIES = {
    '半导体', '元器件', '软件服务', 'IT设备', '互联网', '通信设备',  # AI/算力/通信
    '电气设备',           # 储能/新能源
    '专用机械', '机械基件', '电器仪表',  # 机器人/航天(含部分非科技, 名称关键词再筛)
    '光学光电子', '电子制造', '通信服务', '航天军工',
}
# 科技名称关键词(补行业分类不细的: 航天/机器人/卫星/储能等)
TECH_NAME_KEYS = ['航天', '航空', '卫星', '机器人', '储能', '锂电', '光伏',
                  '芯片', '半导', '算力', '智能', '数据', '云端', '光电',
                  '科技', '信息', '电子', '通讯', '导航', '军工']


def is_tech(name, industry):
    if industry in TECH_INDUSTRIES:
        return True
    if any(k in str(name) for k in TECH_NAME_KEYS):
        return True
    return False


def main():
    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    ind_map = dict(zip(sym, sl['industry']))
    name_map = dict(zip(sym, sl['name']))
    codes = load_codes()
    tech_count = sum(1 for code in codes if is_tech(name_map.get(code, ''), ind_map.get(code, '')))
    print(f"三因子精选 | {len(codes)}只(科技赛道{tech_count}只) | V3趋势加速 ∩ 科技赛道 ∩ TCL回踩买点")
    print("认知: 真 alpha = 主线 × 选股 × 择时 (纯技术已证伪)\n")

    hits = []
    for code in codes:
        try:
            ind = ind_map.get(code, '')
            nm = name_map.get(code, '')
            if not is_tech(nm, ind):
                continue  # 因子1: 主线过滤
            df = pd.read_parquet(STORE / code / 'D.parquet')
            df['date'] = pd.to_datetime(df['date'])
            df = compute_indicators(df)
            last = df.iloc[-1]
            # 因子2: V3选股质量 + 因子3: TCL回踩择时
            if (last['f1'] > 0.3 and last['close'] > last['MA20']
                    and bool(tcl_mask(df).iloc[-1])):
                hits.append({'code': code, 'name': nm, 'industry': ind,
                             'price': last['close'], 'f1': round(last['f1'], 2),
                             'dh60': f"{last['dh60']*100:+.1f}%",
                             'vsMA20': f"{last['close']/last['MA20']*100-100:+.1f}%",
                             'date': last['date'].date()})
        except Exception:
            pass

    hits = sorted(hits, key=lambda x: x['f1'], reverse=True)
    print(f"=== 三因子精选(V3趋势 + 科技赛道 + TCL回踩): {len(hits)}只 ===")
    for h in hits[:30]:
        print(f"  {h['code']} {h['name']:<8}[{h['industry']}] 价{h['price']:.2f} "
              f"f1={h['f1']:+.2f} dh60={h['dh60']} vsMA20={h['vsMA20']} ({h['date']})")
    if not hits:
        print("  (无三因子叠加股 — 当前科技赛道无'V3趋势+回踩买点'同时满足的)")
    return hits


if __name__ == "__main__":
    main()
