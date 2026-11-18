#!/usr/bin/env python3
"""
4D信号回测系统 CLI

用法:
  # 单只股票回测
  python scripts/backtest/cli.py 603881 --start 20230101

  # 指定结束日期
  python scripts/backtest/cli.py 603881 --start 20230101 --end 20260430

  # 批量回测
  python scripts/backtest/cli.py --batch --symbols 603881,512480,512760 --start 20230101

  # 使用股票池
  python scripts/backtest/cli.py --batch --pool index300 --start 20240101

  # 单维度分析
  python scripts/backtest/cli.py 603881 --dimension macd --start 20230101
"""
import argparse
import sys
import os
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.backtest.signal_backtest import SignalBacktester, run_batch, format_batch_report
from scripts.backtest.metrics import format_backtest_report

warnings.filterwarnings('ignore')

# 股票池预设
POOLS = {
    'hold': ['603881', '603617', '512480', '512760'],
    'ai_chain': ['603881', '002230', '300496', '688256', '300474'],
    'index300_sample': [
        '600519', '000858', '601318', '600036', '000651',
        '603259', '601012', '600276', '000333', '002475',
    ],
}


def run_single(symbol: str, start: str, end: str = None,
               dimension: str = None) -> None:
    """单只股票回测"""
    print(f"\n{'='*70}")
    print(f"  4D信号回测: {symbol}")
    print(f"  期间: {start} ~ {end or '最新'}")
    print(f"{'='*70}\n")

    t0 = time.time()
    bt = SignalBacktester(symbol, start, end)
    results = bt.run()
    elapsed = time.time() - t0

    if 'error' in results:
        print(f"  错误: {results['error']}")
        return

    # 报告
    report = format_backtest_report(results)
    print(report)

    # IC衰减
    if results.get('ic_decay'):
        print(f"\n{'─'*70}")
        print(f"  IC衰减 (信号对不同期限收益的预测力)")
        print(f"{'─'*70}")
        for period, ic_val in results['ic_decay'].items():
            bar = '+' * int(abs(ic_val) * 100) if ic_val > 0 else '-' * int(abs(ic_val) * 100)
            print(f"  {period:>4}: {ic_val:+.4f} {'|' + bar}")

    # 维度对比
    if results.get('dimension_analysis'):
        print(f"\n{'─'*70}")
        print(f"  各维度独立预测力")
        print(f"{'─'*70}")
        print(f"  {'维度':<10} {'IC':>8} {'胜率':>8} {'看多次数':>8} {'看空次数':>8} {'看多均收':>10} {'看空均收':>10}")
        print(f"  {'─'*62}")
        for dim, data in results['dimension_analysis'].items():
            print(f"  {dim:<10} {data['ic']:>+8.4f} {data['win_rate']:>7.1%} "
                  f"{data['bull_count']:>8} {data['bear_count']:>8} "
                  f"{data['bull_fwd_avg']:>+9.2%} {data['bear_fwd_avg']:>+9.2%}")

    # 冲突统计
    signals = results.get('signals')
    if signals is not None:
        conflict_rate = signals['conflict'].mean()
        print(f"\n{'─'*70}")
        print(f"  冲突降级统计")
        print(f"{'─'*70}")
        print(f"  冲突率: {conflict_rate:.1%}")
        verdict_counts = signals['verdict'].value_counts()
        for v in ['做多', '偏多', '中性', '偏空', '做空']:
            cnt = verdict_counts.get(v, 0)
            pct = cnt / len(signals) if len(signals) > 0 else 0
            bar = '#' * int(pct * 50)
            print(f"  {v:<6}: {cnt:>5} ({pct:>5.1%}) {'|' + bar}")

    print(f"\n  耗时: {elapsed:.1f}s | 数据点: {results.get('trading_days', 0)}")
    print()


def run_batch_cli(symbols: list, start: str, end: str = None) -> None:
    """批量回测"""
    print(f"\n{'='*70}")
    print(f"  批量4D信号回测: {len(symbols)}只")
    print(f"  期间: {start} ~ {end or '最新'}")
    print(f"{'='*70}\n")

    t0 = time.time()
    all_results = run_batch(symbols, start, end)
    elapsed = time.time() - t0

    report = format_batch_report(all_results)
    print(report)

    # 每只单独的维度分析
    print(f"\n{'─'*90}")
    print(f"  各股详细维度IC")
    print(f"{'─'*90}")
    print(f"  {'代码':<10} {'MA':>8} {'结构':>8} {'MACD':>8} {'量价':>8} {'冲突率':>8}")
    print(f"  {'─'*50}")
    for sym, r in all_results.items():
        if 'error' in r or 'dimension_analysis' not in r:
            continue
        da = r['dimension_analysis']
        conflict = r.get('signals', None)
        cr = conflict['conflict'].mean() if conflict is not None else 0
        print(f"  {sym:<10} "
              f"{da.get('ma', {}).get('ic', 0):>+8.4f} "
              f"{da.get('struct', {}).get('ic', 0):>+8.4f} "
              f"{da.get('macd', {}).get('ic', 0):>+8.4f} "
              f"{da.get('vol', {}).get('ic', 0):>+8.4f} "
              f"{cr:>7.1%}")

    print(f"\n  总耗时: {elapsed:.1f}s")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='4D信号回测系统 — 验证指标体系是否有效',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/backtest/cli.py 603881 --start 20230101
  python scripts/backtest/cli.py --batch --symbols 603881,512480 --start 20240101
  python scripts/backtest/cli.py --batch --pool hold --start 20230101
        """,
    )
    parser.add_argument('symbol', nargs='?', help='股票代码 (如 603881)')
    parser.add_argument('--start', required=True, help='回测起始日期 (如 20230101)')
    parser.add_argument('--end', default=None, help='回测结束日期 (默认最新)')
    parser.add_argument('--batch', action='store_true', help='批量模式')
    parser.add_argument('--symbols', default=None, help='批量股票，逗号分隔 (如 603881,512480)')
    parser.add_argument('--pool', default=None,
                        choices=list(POOLS.keys()),
                        help=f'预设股票池 ({", ".join(POOLS.keys())})')
    parser.add_argument('--dimension', default=None,
                        choices=['ma', 'struct', 'macd', 'vol'],
                        help='单维度分析 (仅显示该维度详情)')

    args = parser.parse_args()

    if args.batch:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        elif args.pool:
            symbols = POOLS[args.pool]
        else:
            parser.error('批量模式需要 --symbols 或 --pool')
        run_batch_cli(symbols, args.start, args.end)
    else:
        if not args.symbol:
            parser.error('单只回测需要提供股票代码')
        run_single(args.symbol, args.start, args.end, args.dimension)


if __name__ == '__main__':
    main()
