#!/usr/bin/env python3
"""重建trade_log.xlsx — 从之前读取的数据恢复 + 补录期初持仓。"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

# 之前读到的原始数据(42行) + 期初补录
# 格式: [date, code, name, direction, price, shares, amount, trade_type, commission, notes, source]

HEADER = ['trade_date', 'stock_code', 'stock_name', 'direction', 'price', 'shares', 'amount', 'trade_type', 'commission', 'notes', 'source']

# 期初持仓（4/20之前就持有的）
OPENING = [
    ['2026-04-20', '603881', '数据港', 'buy', 26.172, 2160, 56531.5, 'opening', 0, '期初持仓（2024年建仓）', 'opening_balance'],
    ['2026-04-20', '603617', '君禾股份', 'buy', 7.485, 5000, 37425, 'opening', 0, '期初持仓（2024年建仓）', 'opening_balance'],
    ['2026-04-20', '512480', '半导体ETF', 'buy', 1.018, 20000, 20360, 'opening', 0, '期初持仓（2024年建仓）', 'opening_balance'],
    ['2026-04-20', '300291', '百纳千成', 'buy', 6.277, 4000, 25108, 'opening', 0, '期初持仓', 'opening_balance'],
]

# 原始trade_log数据（4/21起，从之前读取恢复）
ORIGINAL = [
    ['2026-04-21', '603881', '数据港', 'buy', 38.1, 1760, 64584, 'normal', 16.15, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '515070', 'AI智能', 'buy', 1.223, 30000, 43380, 'normal', 10.85, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '300059', '东方财富', 'buy', 26.181, 2000, 45200, 'normal', 11.3, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '603617', '君禾股份', 'buy', 7.695, 5000, 36400, 'normal', 9.1, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '512760', '芯片ETF', 'buy', 0.549, 40000, 30680, 'normal', 7.67, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '512480', '半导体', 'buy', 1.596, 26000, 42380, 'normal', 10.6, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '515700', '大数据', 'buy', 0.971, 40000, 38400, 'normal', 9.6, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '510050', '50ETF', 'buy', 2.906, 12000, 32880, 'normal', 8.22, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '000159', '国际实业', 'buy', 5.498, 5000, 29150, 'normal', 7.29, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '300459', '汤姆猫', 'buy', 3.93, 5800, 23432, 'normal', 5.86, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-21', '515120', '创新药', 'buy', 0.623, 30000, 18750, 'normal', 5, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-22', '159995', '芯片ETF', 'buy', 0.549, 60000, 54480, 'normal', 13.62, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-22', '159819', 'AI智能', 'buy', 0.984, 20000, 35540, 'normal', 8.89, '快照自动检测：新建仓', 'snapshot_diff'],
    ['2026-04-22', '000159', '国际实业', 'sell', 5.83, 5000, 29150, 'normal', 7.29, '快照自动检测：清仓', 'snapshot_diff'],
    ['2026-04-22', '512760', '芯片ETF', 'sell', 0.767, 40000, 30680, 'normal', 7.67, '快照自动检测：清仓', 'snapshot_diff'],
    ['2026-04-22', '515070', 'AI智能', 'sell', 1.446, 30000, 43380, 'normal', 10.85, '快照自动检测：清仓', 'snapshot_diff'],
    ['2026-04-22', '510050', '50ETF', 'sell', 2.74, 12000, 32880, 'normal', 8.22, '快照自动检测：清仓', 'snapshot_diff'],
    ['2026-04-22', '515700', '大数据', 'sell', 0.96, 40000, 38400, 'normal', 9.6, '快照自动检测：清仓', 'snapshot_diff'],
    ['2026-04-22', '603881', '数据港', 'buy', 39.57, 500, 19785, 'normal', 5, '快照自动检测：加仓500股', 'snapshot_diff'],
    ['2026-04-22', '512480', '半导体', 'sell', 1.677, 6000, 10062, 'normal', 5, '快照自动检测：减仓6000股', 'snapshot_diff'],
    ['2026-04-22', '515120', '创新药', 'sell', 0.629, 10000, 6290, 'normal', 5, '快照自动检测：减仓10000股', 'snapshot_diff'],
    ['2026-04-23', '515120', '创新药', 'buy', 0.621, 16000, 9936, 'normal', 5, '快照自动检测：加仓16000股', 'snapshot_diff'],
    ['2026-04-23', '159995', '芯片ETF', 'sell', 0.897, 10000, 8970, 'normal', 5, '快照自动检测：减仓10000股', 'snapshot_diff'],
    ['2026-04-23', '515120', '创新药', 'buy', 0.622, 16000, 9952, 'normal', 5, '加仓创新药，从20000→36000股', 'manual'],
    ['2026-04-23', '159995', '芯片ETF', 'sell', 0.9, 10000, 9000, 'normal', 5, '减仓芯片ETF做T，从60000→50000股，成本从0.549降至0.479', 'manual'],
    ['2026-04-24', '603881', '数据港', 'buy', 37.26, 500, 18630, 't_buy', 5, '做T买入，低吸', 'manual'],
    ['2026-04-24', '603881', '数据港', 'buy', 37.26, 500, 18630, 't_buy', 5, '做T买入，低吸', 'manual'],
    ['2026-04-24', '603881', '数据港', 'sell', 38.18, 500, 19090, 't_sell', 5, '做T卖出，高抛', 'manual'],
    ['2026-05-19', '603881', '数据港', 'buy', 41.18, 500, 20590, 't_buy', 5, '做T买入，分两笔400+100', 'manual'],
    ['2026-05-19', '603881', '数据港', 'sell', 42.71, 500, 21355, 't_sell', 5, '做T卖出，+765元(+3.72%)', 'manual'],
    ['2026-05-19', '002250', '联化科技', 'buy', 15.18, 1000, 15180, 'normal', 5, '建仓，分两笔500+500', 'manual'],
    ['2026-05-19', '002250', '联化科技', 'buy', 15.18, 1000, 15180, 'normal', 5, '加仓', 'manual'],
    ['2026-05-19', '002250', '联化科技', 'buy', 15.18, 2000, 30360, 'normal', 5, '加仓至4000股', 'manual'],
    ['2026-06-02', '000100', 'TCL科技', 'buy', 4.505, 12000, 54060, 'normal', 5, '建仓12000股', 'manual'],
    ['2026-06-08', '603617', '君禾股份', 'buy', 7.11, 2000, 14220, 't_buy', 5, '做T买入2000股', 'manual'],
    ['2026-06-09', '603617', '君禾股份', 'sell', 7.23, 600, 4338, 't_sell', 5, '做T卖出600股', 'manual'],
    ['2026-06-09', '603617', '君禾股份', 'sell', 7.23, 1400, 10122, 't_sell', 5, '做T卖出1400股', 'manual'],
    ['2026-06-09', '000100', 'TCL科技', 'sell', 4.84, 2000, 9680, 'normal', 5, '减仓2000股，+6.1%', 'manual'],
    ['2026-06-10', '000100', 'TCL科技', 'buy', 4.62, 5000, 23100, 'normal', 5, '加仓5000股(1700+3300)', 'manual'],
    ['2026-06-10', '159995', '芯片ETF', 'sell', 1.126, 12000, 13512, 'normal', 5, '清仓12000份', 'manual'],
    ['2026-06-10', '562500', '机器人ETF', 'buy', 1.142, 10000, 11420, 'normal', 5, '加仓10000份(10000→20000)', 'manual'],
]

ALL_DATA = OPENING + ORIGINAL

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "trade_log"

# 表头
header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
for col, title in enumerate(HEADER, 1):
    cell = ws.cell(row=1, column=col)
    cell.value = title
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = Alignment(horizontal='center')

# 数据行
for row_idx, row_data in enumerate(ALL_DATA, 2):
    for col_idx, val in enumerate(row_data, 1):
        ws.cell(row=row_idx, column=col_idx).value = val

# 列宽
for col in range(1, 12):
    ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 14

wb.save('./config/trade_log.xlsx')
print(f"✅ trade_log重建完成: {len(ALL_DATA)}条记录")
print(f"   期初: {len(OPENING)}条")
print(f"   交易: {len(ORIGINAL)}条")
