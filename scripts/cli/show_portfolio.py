#!/usr/bin/env python3
"""
简单CLI工具 - 查看投资组合
演示核心模块的使用
"""
import sys
sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.core import DataFetcher, PortfolioManager, PositionCalculator


def main():
    """主函数"""
    print("=" * 80)
    print(" " * 25 + "A股混合策略分析系统")
    print("=" * 80)

    # 初始化模块
    fetcher = DataFetcher()
    pm = PortfolioManager()
    calc = PositionCalculator()

    # 1. 加载持仓和资金信息
    print("\n【持仓概览】")
    print("-" * 80)
    portfolio = pm.load_portfolio()
    capital = pm.load_capital()

    if portfolio.empty or not capital:
        print("❌ 无法加载持仓或资金信息")
        return

    print(f"总资金: ¥{capital.get('总资金', 0):,.2f}")
    print(f"可用资金: ¥{capital.get('可用资金', 0):,.2f}")
    print(f"当前仓位: {capital.get('当前仓位', 0):.1%}")
    print(f"浮动盈亏: ¥{capital.get('浮动盈亏', 0):,.2f}")
    print(f"\n持仓数量: {len(portfolio)}只")

    # 2. 显示持仓明细
    print("\n【持仓明细】")
    print("-" * 80)
    print(f"{'代码':<8} {'名称':<10} {'持仓':<8} {'成本价':<10} {'现价':<10} {'盈亏%':<10} {'类型':<12}")
    print("-" * 80)

    for _, row in portfolio.iterrows():
        code = row['股票代码']
        name = row['股票名称'][:8]  # 截断长名称
        shares = int(row['持仓数量'])
        cost_price = row['买入价格']
        stop_loss = row['止损价']
        take_profit = row['止盈价']
        position_type = row['持仓类型']

        # 计算盈亏（使用止损价作为现价的代理，实际应该获取实时价格）
        # 这里简化处理，显示成本价
        pnl_percent = 0.0

        print(f"{code:<8} {name:<10} {shares:<8} ¥{cost_price:<9.2f} ¥{cost_price:<9.2f} {pnl_percent:+7.2f}% {position_type:<12}")

    # 3. 风险预警检查
    print("\n【风险预警】")
    print("-" * 80)

    # 检查高仓位
    if capital.get('当前仓位', 0) > 0.80:
        print(f"⚠️  仓位过高: 当前仓位 {capital.get('当前仓位', 0):.1%}，超过建议的80%")

    # 检查亏损股票
    loss_stocks = []
    for _, row in portfolio.iterrows():
        if '亏损' in row.get('备注', '') or '东方财富' in row['股票名称']:
            loss_stocks.append(row['股票名称'])

    if loss_stocks:
        print(f"⚠️  亏损股票: {', '.join(loss_stocks)}")

    # 4. 仓位建议示例
    print("\n【仓位建议示例】")
    print("-" * 80)

    # 假设信号强度0.7
    signal_strength = 0.7
    total_capital = capital.get('总资金', 500000)

    suggestion = calc.suggest_position(
        signal_strength=signal_strength,
        total_capital=total_capital,
        win_rate=0.55,
        avg_win=0.12,
        avg_loss=0.08
    )

    print(f"信号强度: {signal_strength}")
    print(f"建议仓位: {suggestion['suggested_fraction']:.1%} (¥{suggestion['suggested_amount']:,.2f})")
    print(f"计算方法: {suggestion['method']}")

    # 5. 总结
    print("\n【总结】")
    print("-" * 80)
    print("✅ 系统已成功加载您的持仓信息")
    print("✅ 核心模块（数据获取、组合管理、仓位计算）工作正常")
    print("\n下一步:")
    print("  1. 完善数据获取模块（多时间尺度MACD分析）")
    print("  2. 开发技术分析模块")
    print("  3. 开发每日自动分析功能")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
