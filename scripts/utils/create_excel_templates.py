#!/usr/bin/env python3
"""
创建Excel配置模板文件
"""
import pandas as pd
from pathlib import Path

def create_portfolio_template():
    """创建持仓配置模板"""
    data = {
        '股票代码': ['600519', '000858', '002304'],
        '股票名称': ['贵州茅台', '五粮液', '洋河股份'],
        '买入日期': ['2024-01-15', '2024-02-20', '2024-03-10'],
        '买入价格': [1680.00, 120.50, 85.00],
        '持仓数量': [100, 200, 300],
        '止损价': [1500.00, 108.00, 76.50],
        '止盈价': [1900.00, 135.00, 97.75],
        '仓位%': [0.10, 0.08, 0.06],
        '持仓类型': ['核心持仓', '核心持仓', '卫星持仓'],
        '备注': ['白酒龙头', '白酒龙头', '白酒二线']
    }

    df = pd.DataFrame(data)

    # 添加说明sheet
    instructions = pd.DataFrame({
        '字段说明': [
            '股票代码: 6位数字代码',
            '股票名称: 股票中文名称',
            '买入日期: YYYY-MM-DD格式',
            '买入价格: 成本价',
            '持仓数量: 股数',
            '止损价: 止损触发价格',
            '止盈价: 止盈触发价格',
            '仓位%: 该股票占总资金的比例（小数）',
            '持仓类型: 核心持仓/卫星持仓/短线T',
            '备注: 自定义备注信息'
        ]
    })

    output_file = Path(__file__).parent.parent.parent / 'config' / 'portfolio.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='持仓信息', index=False)
        instructions.to_excel(writer, sheet_name='字段说明', index=False)

    print(f"✅ 已创建: {output_file}")

def create_capital_template():
    """创建资金池配置模板"""
    data = {
        '项目': [
            '总资金',
            '可用资金',
            '持仓市值',
            '最大单股仓位',
            '总仓位上限',
            '风险偏好',
            '投资风格'
        ],
        '数值': [
            500000,
            200000,
            300000,
            0.15,
            0.80,
            '中等',
            '混合策略'
        ],
        '说明': [
            '总投资资金（元）',
            '未投入资金（元）',
            '当前持仓总市值（元）',
            '单只股票最大仓位限制（小数）',
            '总持仓占总资金比例上限（小数）',
            '保守/中等/激进',
            '价值投资/趋势交易/混合策略'
        ]
    }

    df = pd.DataFrame(data)

    output_file = Path(__file__).parent.parent.parent / 'config' / 'capital.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='资金池', index=False)

    print(f"✅ 已创建: {output_file}")

if __name__ == '__main__':
    print("创建Excel配置模板...")

    try:
        create_portfolio_template()
        create_capital_template()
        print("\n✅ 所有Excel模板创建完成！")
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        print("\n请确保已安装依赖: pip install pandas openpyxl")
