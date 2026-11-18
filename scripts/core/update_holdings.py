#!/usr/bin/env python3
"""从券商APP持仓截图更新 portfolio.xlsx + capital.xlsx(自动备份)。"""
import pandas as pd
import shutil
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent.parent / 'config'

ACCOUNT = {
    'total': 435300.82, 'avail': 99335.28, 'mv': 335965.04,
    'pnl': 43818.99, 'position': 0.772, 'date': '2026-07-03',
}
HOLDINGS = [
    ('000100', 'TCL科技',   3.924, 10000, 5.720, 57200, '核心', '7/3收盘|+45.8%|3S持有|移动止损5.02'),
    ('603881', '数据港',    19.772, 2072, 25.320, 52463, '核心', '7/3收盘|+28.1%|3S观望'),
    ('002250', '联化科技',  15.952, 3000, 15.510, 46530, '卫星', '7/3收盘|-2.8%|底背离confirmed持有'),
    ('603617', '君禾股份',  7.503, 5000, 7.020, 35100, '卫星', '7/3收盘|-6.4%'),
    ('301518', '长华化学',  27.238, 1000, 27.500, 27500, '卫星', '7/3新买|质量90最高|底背离买点'),
    ('512480', '半导体ETF', 0.509, 20000, 1.361, 27220, '核心', '7/3收盘|加仓10000→20000|成本1.018→0.509'),
    ('510500', '500ETF',    8.862, 3000, 8.971, 26913, '卫星', '7/3收盘|+1.2%'),
    ('562500', '机器人ETF', 1.188, 20000, 1.207, 24140, '卫星', '7/3收盘|+1.6%'),
    ('301110', '青木科技',  39.381, 500, 39.510, 19755, '卫星', '7/3新买|质量87|互联网PE33'),
    ('300455', '航天智装',  22.651, 800, 22.680, 18144, '卫星', '7/3新买|旧推荐|亏损股5d外'),
]

def main():
    T = ACCOUNT['total']
    for f in ['portfolio.xlsx', 'capital.xlsx']:
        src = CONFIG / f
        if src.exists():
            shutil.copy(src, CONFIG / f"{f}.bak")
    rows = []
    for code, name, cost, shares, price, mv, pt, notes in HOLDINGS:
        rows.append({
            'code': code, 'name': name, 'buy_date': ACCOUNT['date'], 'avg_cost': cost,
            'shares': shares, 'stop_loss': 0, 'take_profit': 0,
            'position_pct': round(mv / T * 100, 1), 'position_type': pt, 'notes': notes,
        })
    df = pd.DataFrame(rows)[['code', 'name', 'buy_date', 'avg_cost', 'shares', 'stop_loss',
                             'take_profit', 'position_pct', 'position_type', 'notes']]
    df.to_excel(CONFIG / 'portfolio.xlsx', sheet_name='持仓', index=False)
    cap = pd.DataFrame({
        '项目': ['总资金', '可用资金', '当前持仓总市值', '单只股票最大仓位限制', '总持仓占总资金比例上限',
                '风险偏好', '投资策略', '当前仓位', '浮动盈亏', '更新时间'],
        '金额/数值': [ACCOUNT['total'], ACCOUNT['avail'], ACCOUNT['mv'], 0.2, 0.8,
                    '中等偏激进', '混合策略', ACCOUNT['position'], ACCOUNT['pnl'], ACCOUNT['date']],
        '说明': ['总投资资金(元)', '未投入资金', '当前持仓市值', '单股最大15%', '总仓位最大80%',
                '', '', '当前仓位比例', '当前浮动盈亏(元)', '券商截图校准']})
    cap.to_excel(CONFIG / 'capital.xlsx', sheet_name='Sheet1', index=False)
    print(f"✓ 已更新 portfolio.xlsx({len(df)}只) + capital.xlsx(仓位{ACCOUNT['position']*100:.1f}% 浮盈{ACCOUNT['pnl']:+,.0f})")

if __name__ == '__main__':
    main()
