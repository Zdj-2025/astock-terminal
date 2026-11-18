#!/usr/bin/env python3
"""板块内所有V3选股(2026-06-17, 扩板块22+不限龙头避免抱团)。

调整: ① 板块10→22(加商业航天/白酒等); ② 选股从只龙头→板块内所有V3股。
避免龙头抱团风险(资金过度集中→反转踩踏)。

用法: PYTHONPATH=. python scripts/screener/sector_v3_scanner.py [--top-sectors N]
"""
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask, STORE
from scripts.core.tushare_client import TushareClient

# 22板块映射: industry精确匹配 + 名称关键词兜底
SECTOR_MAP = {
    '半导体': {'ind': ['半导体', '元器件'], 'names': []},
    'AI算力': {'ind': ['IT设备', '互联网'], 'names': ['算力', '数据', '智能', '云端']},
    '消费电子': {'ind': ['电器仪表'], 'names': ['电子', '光电']},
    '新能源车': {'ind': ['汽车配件'], 'names': ['汽车', '电车', '宁德']},
    '光伏': {'ind': ['电气设备'], 'names': ['光伏', '太阳']},
    '通信': {'ind': ['通信设备'], 'names': ['通信', '通讯']},
    '软件信创': {'ind': ['软件服务'], 'names': ['信创', '软件', '信息']},
    '机器人': {'ind': ['专用机械', '机械基件'], 'names': ['机器人', '自动化']},
    '军工': {'ind': [], 'names': ['军工', '兵器', '航空', '防务']},
    '生物医药': {'ind': ['生物制药', '化学制药', '医疗保健', '中成药'], 'names': []},
    '商业航天': {'ind': [], 'names': ['卫星', '北斗', '导航', '航天']},
    '低空经济': {'ind': [], 'names': ['低空', '通航', '无人机']},
    '储能': {'ind': [], 'names': ['储能']},
    '固态电池': {'ind': [], 'names': ['固态', '锂电', '电池']},
    '创新药': {'ind': [], 'names': ['创新药', '生物', '基因']},
    '医疗器械': {'ind': [], 'names': ['医疗', '器械', '迈瑞']},
    '金融科技': {'ind': [], 'names': ['金融科技', '恒生', '同花顺']},
    '传媒游戏': {'ind': [], 'names': ['传媒', '游戏', '影视']},
    '有色金属': {'ind': ['小金属'], 'names': ['有色', '稀土', '铜', '铝', '黄金', '紫金']},
    '化工新材料': {'ind': ['化工原料', '塑料'], 'names': ['化工', '化学', '新材料', '万华']},
    '煤炭': {'ind': [], 'names': ['煤']},
    '白酒': {'ind': ['食品'], 'names': ['白酒', '茅台', '五粮液', '酒']},
    '家电': {'ind': ['家用电器'], 'names': ['家电', '美的', '格力']},
}


def classify_sector(name, industry):
    """股票→板块(首个匹配)。"""
    for sector, rule in SECTOR_MAP.items():
        if industry in rule['ind']:
            return sector
    for sector, rule in SECTOR_MAP.items():
        if any(k in str(name) for k in rule['names']):
            return sector
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-sectors', type=int, default=0, help='只看前N板块(0=全部22)')
    args = ap.parse_args()

    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))
    ind_map = dict(zip(sym, sl['industry']))

    # 全市场→22板块
    all_codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    sector_codes = {s: [] for s in SECTOR_MAP}
    for code in all_codes:
        sec = classify_sector(name_map.get(code, ''), ind_map.get(code, ''))
        if sec:
            sector_codes[sec].append(code)

    print(f"板块内所有V3选股 | 22板块 | 不限龙头(避免抱团)\n")
    print("【板块股池分布】")
    for s, codes in sorted(sector_codes.items(), key=lambda x: -len(x[1])):
        print(f"  {s:<8}{len(codes):>5}只")

    # 板块内所有股 → V3趋势
    print(f"\n【板块内V3趋势加速股(f1>0.3 + above_ma20)】")
    hits = []
    sectors_to_scan = list(SECTOR_MAP.keys()) if args.top_sectors == 0 else \
        [s for s, _ in sorted(sector_codes.items(), key=lambda x: -len(x[1]))[:args.top_sectors]]
    for sector in sectors_to_scan:
        for code in sector_codes[sector]:
            try:
                df = pd.read_parquet(STORE / code / 'D.parquet')
                df['date'] = pd.to_datetime(df['date'])
                df = compute_indicators(df)
                last = df.iloc[-1]
                if last['f1'] > 0.3 and last['close'] > last['MA20']:
                    hits.append({'code': code, 'name': name_map.get(code, ''), 'sector': sector,
                                 'price': last['close'], 'f1': round(last['f1'], 2),
                                 'pullback': bool(tcl_mask(df).iloc[-1]),
                                 'dh60': f"{last['dh60']*100:+.1f}%"})
            except Exception:
                pass

    hits = sorted(hits, key=lambda x: (not x['pullback'], -x['f1']))
    buy = [h for h in hits if h['pullback']]
    print(f"  V3趋势股: {len(hits)}只(回踩买点{len(buy)}只)\n")
    # 按板块分组
    by_sector = {}
    for h in hits:
        by_sector.setdefault(h['sector'], []).append(h)
    for sector in sorted(by_sector, key=lambda s: -len(by_sector[s])):
        hs = by_sector[sector]
        buy_s = [h for h in hs if h['pullback']]
        print(f"  [{sector}] {len(hs)}只(回踩{len(buy_s)})" + (f" 🎯{','.join(h['code'] for h in buy_s[:3])}" if buy_s else ""))
    print(f"\n💡 板块内所有V3(不限龙头): 趋势{len(hits)}只, 回踩买点{len(buy)}只")


if __name__ == '__main__':
    main()
