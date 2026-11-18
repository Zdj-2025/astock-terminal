#!/usr/bin/env python3
"""多周期K线数据下载 CLI。

用法:
  # 下载所有关注标的的日线+周线
  PYTHONPATH=. python scripts/cli/download_data_cli.py --periods W,D

  # 下载单只股票所有周期
  PYTHONPATH=. python scripts/cli/download_data_cli.py --symbol 603881 --all

  # 下载所有关注标的所有周期（注意分钟线限额）
  PYTHONPATH=. python scripts/cli/download_data_cli.py --all

  # 强制全量重下
  PYTHONPATH=. python scripts/cli/download_data_cli.py --symbol 603881 --periods D --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保 project root 在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.multi_period_downloader import (
    PERIOD_CONFIG,
    MultiPeriodDownloader,
    load_watchlist,
    print_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="多周期K线数据下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="单只标的代码（如 603881），不指定则下载 watchlist 全部",
    )
    parser.add_argument(
        "--periods", type=str, default=None,
        help="指定周期，逗号分隔（如 W,D,60min）。可选: W,D,60min,30min,15min,5min",
    )
    parser.add_argument(
        "--all", action="store_true", dest="download_all",
        help="下载所有周期（W,D,60min,30min,15min,5min）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制全量重下，忽略增量",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_symbols",
        help="列出 watchlist 中的所有标的",
    )

    args = parser.parse_args()

    # 列出标的
    if args.list_symbols:
        items = load_watchlist()
        print(f"📋 关注列表 ({len(items)} 只):")
        for it in items:
            print(f"  {it['code']:>6s}  {it['name']:<10s}  {it['type']}")
        return

    # 确定周期
    if args.download_all:
        periods = list(PERIOD_CONFIG.keys())
    elif args.periods:
        periods = [p.strip() for p in args.periods.split(",")]
    else:
        # 默认日线+周线
        periods = ["W", "D"]

    print(f"🔄 周期: {', '.join(periods)}")

    # 确定标的
    if args.symbol:
        symbols = [args.symbol.strip()]
    else:
        items = load_watchlist()
        symbols = [it["code"] for it in items]
        if not symbols:
            print("❌ watchlist 中无有效标的")
            sys.exit(1)
        print(f"📋 共 {len(symbols)} 只标的")

    # 下载
    downloader = MultiPeriodDownloader()
    results = downloader.download_all(
        periods=periods,
        symbols=symbols,
        force=args.force,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
