#!/usr/bin/env python3
"""增幅/动量多维评分 CLI

用法:
    PYTHONPATH=/path/to/stock python scripts/cli/gain_momentum_cli.py 603881
    PYTHONPATH=/path/to/stock python scripts/cli/gain_momentum_cli.py 603881 002993 300323
    PYTHONPATH=/path/to/stock python scripts/cli/gain_momentum_cli.py --date 20260519 603881
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="增幅/动量多维评分")
    parser.add_argument("symbols", nargs="+", help="股票代码（6位）")
    parser.add_argument("--date", default=None, help="截止日期 YYYYMMDD")
    args = parser.parse_args()

    from scripts.data.tushare_client import TushareClient
    from scripts.screener.gain_momentum_scorer import GainMomentumScorer

    client = TushareClient()
    scorer = GainMomentumScorer(client)

    results = []
    for symbol in args.symbols:
        symbol = symbol.strip()[:6]
        r = scorer.score(symbol, trade_date=args.date)
        results.append(r)

    for r in results:
        print(f"\n{'='*55}")
        print(f"  📈 增幅/动量评分: {r.symbol} {r.name}")
        print(f"{'='*55}")
        print(f"  📊 总分: {r.total_score:.1f}/100 ({r.grade})")
        print(f"     弹性: {r.elasticity_score:.1f} | "
              f"动量: {r.momentum_score:.1f} | "
              f"回撤: {r.drawdown_score:.1f}")
        print(f"     交叉信号: {r.cross_signal}")
        print()
        print(f"  📐 弹性详情: {r.elasticity_detail}")
        print(f"  📐 动量详情: {r.momentum_detail}")
        print(f"  📐 回撤详情: {r.drawdown_detail}")
        print(f"  📐 原始收益: {r.raw_returns}")
        print(f"  📐 原始回撤: {r.raw_drawdowns}")

    if len(results) > 1:
        print(f"\n{'='*55}")
        print("  📊 排名:")
        print(f"{'='*55}")
        ranked = sorted(results, key=lambda x: x.total_score, reverse=True)
        for i, r in enumerate(ranked, 1):
            print(f"  {i}. {r.symbol} — {r.total_score:.1f} ({r.grade}) "
                  f"弹性:{r.elasticity_score:.0f} "
                  f"动量:{r.momentum_score:.0f} "
                  f"回撤:{r.drawdown_score:.0f}")


if __name__ == "__main__":
    main()
