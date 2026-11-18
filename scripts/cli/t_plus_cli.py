#!/usr/bin/env python3
"""做T分析CLI — 日内/短线波段交易信号。

用法:
    # 分析单只股票做T机会
    PYTHONPATH=. python scripts/cli/t_plus_cli.py 603881

    # 分析所有持仓的做T机会
    PYTHONPATH=. python scripts/cli/t_plus_cli.py --holdings
"""
import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.trading.t_plus_system import (
    TPlusAnalyzer,
    format_report,
    format_holdings_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="做T（日内/短线波段）分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/cli/t_plus_cli.py 603881       # 分析数据港做T机会
  python scripts/cli/t_plus_cli.py --holdings    # 分析所有持仓
        """,
    )
    parser.add_argument("symbol", nargs="?", help="股票代码，如 603881")
    parser.add_argument("--holdings", action="store_true", help="分析所有持仓的做T机会")

    args = parser.parse_args()

    if not args.symbol and not args.holdings:
        parser.print_help()
        sys.exit(1)

    analyzer = TPlusAnalyzer()

    if args.holdings:
        print("正在分析所有持仓...\n")
        results = analyzer.analyze_holdings()
        print(format_holdings_report(results))
        # 对有信号的股票输出详细报告
        for r in results:
            if r.signal in ("买入", "卖出") and r.signal_strength >= 3:
                print()
                print(format_report(r))
    else:
        symbol = args.symbol.strip()
        print(f"正在分析 {symbol} ...\n")
        result = analyzer.analyze(symbol)
        print(format_report(result))


if __name__ == "__main__":
    main()
