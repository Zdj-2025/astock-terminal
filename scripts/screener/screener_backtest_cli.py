#!/usr/bin/env python3
"""选股器回测CLI。

用法:
    # 6个月月度回测
    python scripts/screener/screener_backtest_cli.py \
        --template wave3 --from 20240601 --windows 6

    # 指定日期范围+自定义前瞻天数
    python scripts/screener/screener_backtest_cli.py \
        --template wave3_startup --from 20240101 --to 20241231 \
        --step monthly --forward 5,10,20,40

    # 快速测试（1个窗口，少量股票）
    python scripts/screener/screener_backtest_cli.py \
        --template wave3 --from 20241201 --windows 1 --quick --top 5

    # 清除缓存
    python scripts/screener/screener_backtest_cli.py --clear-cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="选股器回测")
    parser.add_argument("--template", "-t",
                        choices=["wave3", "wave3_startup", "bottom_reversal",
                                 "trend_scan", "chanlun_t3",
                                 "growth", "value", "momentum", "ai_focus",
                                 "wave", "davis_double", "turnaround"],
                        default="wave3",
                        help="选股模板")
    parser.add_argument("--from", dest="from_date", required=False,
                        help="起始日期YYYYMMDD")
    parser.add_argument("--to", dest="to_date",
                        help="结束日期YYYYMMDD（默认今天）")
    parser.add_argument("--windows", type=int,
                        help="回测窗口数（优先于--to）")
    parser.add_argument("--step",
                        choices=["monthly", "biweekly", "weekly"],
                        default="monthly",
                        help="窗口间隔")
    parser.add_argument("--forward", default="5,10,20",
                        help="前瞻天数，逗号分隔")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（仅行情数据评分）")
    parser.add_argument("--top", type=int, default=20,
                        help="每个窗口最多选N只")
    parser.add_argument("--benchmark", default="000001.SH",
                        help="基准指数")
    parser.add_argument("--clear-cache", action="store_true",
                        help="清除回测缓存")
    parser.add_argument("--min-confidence", type=float, default=50.0,
                        help="W3最低置信度")
    parser.add_argument("--min-w3-gain", type=float, default=10.0,
                        help="W3最低涨幅%%")
    parser.add_argument("--min-score", type=float, default=70.0,
                        help="最低综合分(默认70)")

    args = parser.parse_args()

    from scripts.screener.screener_backtester import ScreenerBacktester

    bt = ScreenerBacktester()

    if args.clear_cache:
        count = bt.clear_cache(args.template)
        print(f"清除 {count} 个缓存文件")
        return

    if not args.from_date:
        parser.error("需要 --from 日期 或 --clear-cache")

    forward_days = [int(x) for x in args.forward.split(",")]

    dates = ScreenerBacktester.generate_snapshot_dates(
        from_date=args.from_date,
        to_date=args.to_date,
        step=args.step,
        count=args.windows,
    )

    print(f"回测计划: {len(dates)}个窗口")
    print(f"  模板: {args.template}")
    print(f"  日期: {dates[0]} → {dates[-1]}")
    print(f"  前瞻: {forward_days}")

    results = bt.run(
        template=args.template,
        snapshot_dates=dates,
        forward_days=forward_days,
        benchmark=args.benchmark,
        quick=args.quick,
        top=args.top,
        min_w3_confidence=args.min_confidence,
        min_w3_gain=args.min_w3_gain,
        min_score=args.min_score,
    )

    report = ScreenerBacktester.format_report(results, forward_days)
    print(report)

    # 保存报告
    report_dir = Path("reports") / "backtest"
    report_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = report_dir / f"{args.template}_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  报告已保存: {report_path}")


if __name__ == "__main__":
    main()
