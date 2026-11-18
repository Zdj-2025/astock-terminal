#!/usr/bin/env python3
"""用 7/22 券商截图更新 capital.xlsx（停在 7/9，全旧）。
总资产/现金/仓位取券商权威；持仓市值+浮盈从 portfolio 成本×现价 code-compute 并与券商核对。
openpyxl 保留格式，先备份。"""
import openpyxl, shutil, pandas as pd

PATH = 'config/capital.xlsx'
shutil.copy(PATH, PATH + '.bak.0723')

# 持仓（与 premarket 一致；500ETF 成本用券商 8.668；半导体用 portfolio 0.509）
HOLDINGS = [
    ('000100', 4.152, 12000), ('002250', 15.475, 2400), ('510500', 8.668, 4000),
    ('603881', 16.443, 1072), ('512480', 0.509, 10000), ('512100', 3.012, 20000),
    ('159841', 1.012, 30000), ('300003', 12.844, 2000), ('300171', 11.842, 2000),
    ('515120', 0.642, 40000),
]
mkt = cost_tot = pnl = 0.0
for code, cost, sh in HOLDINGS:
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet').sort_values('date')
    price = float(df['close'].iloc[-1])
    mkt += price * sh
    cost_tot += cost * sh
    pnl += (price - cost) * sh
    print(f"  {code}: {sh}@{cost} 现{price:.3f} 市值{price*sh:>8.0f} 盈亏{(price-cost)*sh:>+8.0f}")

TOTAL = 403962.13   # 券商 7/22 总资产（权威）
CASH = 67812        # 券商可用现金
pos_pct = (TOTAL - CASH) / TOTAL
print(f"\n持仓市值(算) {mkt:.0f} | 成本 {cost_tot:.0f} | 浮盈(算) {pnl:+.0f}")
print(f"券商总资产 {TOTAL:.2f} | 现金 {CASH} | 仓位 {pos_pct*100:.1f}%")
print(f"核对: 算市值 {mkt:.0f} vs 券商推算 {TOTAL-CASH:.0f} (差 {mkt-(TOTAL-CASH):+.0f})")

wb = openpyxl.load_workbook(PATH)
ws = wb['Sheet1']
ws['B2'] = TOTAL
ws['B3'] = CASH
ws['B4'] = round(mkt, 2)
ws['B9'] = round(pos_pct, 4)
ws['B10'] = round(pnl, 2)
ws['B11'] = '2026-07-22'
wb.save(PATH)
print(f"\n✅ capital.xlsx 已更新（备份 {PATH}.bak.0723）")
