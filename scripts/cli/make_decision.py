#!/usr/bin/env python3
"""
决策子系统CLI - 命令行入口

用法:
  python scripts/cli/make_decision.py 600519              # 分析单只股票
  python scripts/cli/make_decision.py 600519 贵州茅台      # 带名称
  python scripts/cli/make_decision.py --portfolio          # 分析整个组合
  python scripts/cli/make_decision.py 510300 --format md   # Markdown输出

作者: AI Assistant
日期: 2026-04-07
"""
import sys
import argparse

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.decision.decision_executor import DecisionExecutor


def main():
    parser = argparse.ArgumentParser(
        description='A股混合策略决策系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/cli/make_decision.py 600519              分析单只股票
  python scripts/cli/make_decision.py --portfolio          分析整个组合
  python scripts/cli/make_decision.py 510300 --format md   Markdown格式输出
        """
    )

    parser.add_argument('symbol', nargs='?', help='股票/ETF代码 (6位)')
    parser.add_argument('name', nargs='?', default='', help='股票名称 (可选)')
    parser.add_argument('--portfolio', '-p', action='store_true',
                        help='分析整个投资组合')
    parser.add_argument('--format', '-f', choices=['text', 'md'], default='text',
                        dest='output_format', help='输出格式 (默认text)')
    parser.add_argument('--no-macd', action='store_true',
                        help='跳过多尺度MACD共振分析')
    parser.add_argument('--config-dir', default='config',
                        help='配置文件目录 (默认config)')

    args = parser.parse_args()

    if not args.symbol and not args.portfolio:
        parser.print_help()
        return

    executor = DecisionExecutor(args.config_dir)

    if args.portfolio:
        result = executor.analyze_portfolio(
            skip_macd_resonance=args.no_macd
        )
        print(result)
    else:
        output_format = 'markdown' if args.output_format == 'md' else 'text'
        result = executor.analyze_and_decide(
            symbol=args.symbol,
            name=args.name,
            skip_macd_resonance=args.no_macd,
            output_format=output_format
        )
        print(result)


if __name__ == '__main__':
    main()
