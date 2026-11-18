#!/usr/bin/env python3
"""
完整分析演示 - 展示整个系统的工作流程
包括：多时间尺度MACD分析 + 技术指标分析 + 仓位建议
"""
import sys
sys.path.insert(0, '/mnt/d/claude-code/stock')

import pandas as pd
import numpy as np
from datetime import datetime
from scripts.core import DataFetcher, PortfolioManager, PositionCalculator
from scripts.analyzers import TechnicalAnalyzer


def create_mock_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """创建模拟K线数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # 根据不同股票设置不同基础价格
    base_prices = {
        '600519': 1500,  # 贵州茅台
        '603881': 30,    # 数据港
        '512760': 0.7,   # 芯片ETF
    }

    base_price = base_prices.get(symbol, 50)
    trend = np.linspace(0, np.random.uniform(-20, 50), days)
    noise = np.random.randn(days) * base_price * 0.03
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(days) * 0.02),
        'high': close_prices * (1 + np.abs(np.random.randn(days) * 0.03)),
        'low': close_prices * (1 - np.abs(np.random.randn(days) * 0.03)),
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, days)
    })

    return df


def analyze_single_stock(symbol: str, name: str):
    """分析单只股票"""
    print(f"\n{'=' * 80}")
    print(f"  {name} ({symbol}) - 完整分析报告")
    print(f"{'=' * 80}")

    # 创建模拟数据
    df = create_mock_data(symbol, 200)
    current_price = df['close'].iloc[-1]

    # 1. 技术指标分析
    print(f"\n【技术指标分析】")
    print("-" * 80)

    tech_analyzer = TechnicalAnalyzer(symbol)
    tech_result = tech_analyzer.analyze_indicators(df)

    if tech_result:
        # 显示各指标分析
        indicators = ['MACD', 'RSI', 'KDJ', '布林带', '均线']
        signals = [
            tech_result['macd'],
            tech_result['rsi'],
            tech_result['kdj'],
            tech_result['bollinger'],
            tech_result['moving_averages']
        ]

        for ind, sig in zip(indicators, signals):
            signal_emoji = {
                'buy': '📈', 'strong_buy': '📈', 'golden_cross': '📈',
                'sell': '📉', 'strong_sell': '📉', 'death_cross': '📉',
                'hold': '➡️'
            }.get(sig['signal'], '➡️')

            print(f"{signal_emoji} {ind:6s}: {sig['description']}")

        # 综合信号
        overall = tech_result['overall']
        overall_emoji = {
            'buy': '✅', 'sell': '❌', 'hold': '⏸️'
        }.get(overall['signal'], '➡️')

        print(f"\n{overall_emoji} 综合信号: {overall['signal'].upper()} (强度: {overall['strength']:.2f})")

    # 2. 仓位建议
    print(f"\n【仓位建议】")
    print("-" * 80)

    calc = PositionCalculator()

    # 基于技术信号强度计算仓位
    signal_strength = tech_result['overall']['strength'] if tech_result else 0.5

    position_suggestion = calc.suggest_position(
        signal_strength=signal_strength,
        total_capital=484271.94,  # 从资金池配置获取
        win_rate=0.55,
        avg_win=0.12,
        avg_loss=0.08
    )

    print(f"信号强度: {signal_strength:.2f}")
    print(f"建议仓位: {position_suggestion['suggested_fraction']:.1%}")
    print(f"建议金额: ¥{position_suggestion['suggested_amount']:,.2f}")
    print(f"计算方法: {position_suggestion['method']}")

    # 3. 风险提示
    print(f"\n【风险提示】")
    print("-" * 80)

    if tech_result and tech_result['macd']['signal'] == 'death_cross':
        print("⚠️  MACD死叉，短期可能下跌")
    elif tech_result and tech_result['macd']['signal'] == 'golden_cross':
        print("✅ MACD金叉，短期可能上涨")

    if tech_result and tech_result['rsi']['value'] > 70:
        print("⚠️  RSI超买，注意回调风险")
    elif tech_result and tech_result['rsi']['value'] < 30:
        print("✅ RSI超卖，可能出现反弹")

    # 止损止盈建议
    stop_loss = current_price * 0.92
    take_profit = current_price * 1.15

    print(f"\n止损价: ¥{stop_loss:.2f} (-8%)")
    print(f"止盈价: ¥{take_profit:.2f} (+15%)")

    return tech_result


def main():
    """主函数"""
    print("=" * 80)
    print(" " * 20 + "A股混合策略分析系统 - 完整演示")
    print("=" * 80)

    print("\n⚠️  注意：由于网络连接问题，本演示使用模拟数据")
    print("实际使用时会从AKShare获取真实数据\n")

    # 加载用户持仓
    pm = PortfolioManager()
    portfolio = pm.load_portfolio()

    if portfolio.empty:
        print("❌ 无法加载持仓配置")
        return

    print(f"✅ 已加载持仓: {len(portfolio)}只股票\n")

    # 分析前3只股票
    print("\n" + "=" * 80)
    print(" " * 25 + "持仓股票分析")
    print("=" * 80)

    for idx, row in portfolio.head(3).iterrows():
        symbol = row['股票代码']
        name = row['股票名称']

        analyze_single_stock(symbol, name)

        print("\n" + "─" * 80)

    # 总结
    print("\n" + "=" * 80)
    print(" " * 30 + "系统总结")
    print("=" * 80)

    print("""
✅ Phase 1: 项目基础设施 - 完成
   • 项目结构搭建
   • 配置文件系统
   • 完整文档

✅ Phase 2: 核心数据模块 - 完成
   • DataFetcher - 数据获取
   • PortfolioManager - 组合管理
   • PositionCalculator - 仓位计算

✅ Phase 3: 技术分析模块 - 完成
   • MultiScaleMACDAnalyzer - 多时间尺度MACD分析⭐
   • TechnicalAnalyzer - 综合技术指标分析
   • RSI, KDJ, 布林带, 均线等指标

🎯 下一步:
   Phase 4: 基本面分析整合
   • 整合china-stock-analysis skill
   • 财务健康度分析
   • 估值计算

   Phase 5: 回测与报告
   • 策略回测引擎
   • 自动报告生成

   Phase 6: 自动化功能
   • 每日自动分析
   • 价格预警
   • 定时任务

💡 系统已可用于:
   ✓ 技术指标分析
   ✓ 仓位建议
   ✓ 风险预警
   ✓ 持仓管理

⚠️  注意事项:
   • 当前使用模拟数据，实际使用需配置网络
   • 所有分析仅供参考，不构成投资建议
   • 投资有风险，决策需谨慎
    """)

    print("=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
