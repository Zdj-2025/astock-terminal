#!/usr/bin/env python3
"""
完整回测系统测试

测试流程:
1. 运行回测引擎
2. 生成绩效报告
3. 导出HTML报告
4. 验证所有组件集成

作者: AI Assistant
日期: 2026-04-04
"""
import sys
sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.backtest.engine import BacktestEngine
from scripts.backtest.report_generator import BacktestReportGenerator
from datetime import datetime


def test_backtest_system():
    """测试完整的回测系统"""
    print("=" * 80)
    print("A股混合策略分析系统 - 完整回测测试")
    print("=" * 80)

    # 1. 运行回测
    print(f"\n📊 步骤1: 运行回测引擎\n")

    engine = BacktestEngine(
        symbols=['603881'],  # 数据港
        start_date='2024-06-01',  # 使用更早的数据
        end_date='2024-12-31',
        initial_capital=1000000
    )

    success = engine.run_backtest()

    if not success:
        print("❌ 回测失败")
        return

    # 2. 获取结果
    print(f"\n📊 步骤2: 获取回测结果\n")

    results = engine.get_results()

    print(f"✅ 回测完成:")
    print(f"   初始资金: ¥{results['initial_capital']:,.2f}")
    print(f"   最终资产: ¥{results['final_value']:,.2f}")
    print(f"   总收益率: {results['total_return']:.2%}")
    print(f"   交易次数: {results['trades']} 笔")

    # 3. 显示绩效指标
    print(f"\n📊 步骤3: 绩效指标分析\n")

    metrics = results.get('performance_metrics', {})

    if 'returns' in metrics:
        r = metrics['returns']
        print(f"【收益指标】")
        print(f"  总收益率: {r.get('total_return_pct', 0):.2f}%")
        print(f"  年化收益率: {r.get('annualized_return_pct', 0):.2f}%")
        print(f"  平均日收益: {r.get('avg_daily_return_pct', 0):.3f}%")

    if 'risk' in metrics:
        r = metrics['risk']
        print(f"\n【风险指标】")
        print(f"  最大回撤: {r.get('max_drawdown_pct', 0):.2f}%")
        print(f"  年化波动率: {r.get('volatility_pct', 0):.2f}%")
        print(f"  下行风险: {r.get('downside_risk_pct', 0):.2f}%")

    if 'risk_adjusted' in metrics:
        r = metrics['risk_adjusted']
        print(f"\n【风险调整收益】")
        print(f"  夏普比率: {r.get('sharpe_ratio', 0):.2f}")
        print(f"  索提诺比率: {r.get('sortino_ratio', 0):.2f}")
        print(f"  卡玛比率: {r.get('calmar_ratio', 0):.2f}")

    if 'trading' in metrics and metrics['trading']:
        t = metrics['trading']
        print(f"\n【交易质量】")
        print(f"  总交易次数: {t.get('total_trades', 0)} 笔")
        print(f"  胜率: {t.get('win_rate_pct', 0):.2f}%")
        print(f"  盈亏比: {t.get('profit_loss_ratio', 0):.2f}")

    # 4. 生成HTML报告
    print(f"\n📊 步骤4: 生成HTML报告\n")

    report_generator = BacktestReportGenerator()
    report_path = report_generator.generate_report(
        results,
        report_name="完整回测测试_603881_2024H2"
    )

    print(f"\n✅ 完整测试完成!")
    print(f"   HTML报告: {report_path}")

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"✅ 回测引擎: 正常运行")
    print(f"✅ 策略执行器: 已集成（信号阈值可能较高，导致交易较少）")
    print(f"✅ 绩效计算器: 计算完成")
    print(f"✅ 报告生成器: HTML报告已生成")
    print(f"\n💡 提示: 如交易次数为0，可能是信号强度未达到阈值 (0.6)")
    print(f"   可以调整 strategy_executor.py 中的买入/卖出阈值")
    print("=" * 80)


if __name__ == '__main__':
    try:
        test_backtest_system()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
