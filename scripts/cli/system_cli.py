#!/usr/bin/env python3
"""系统管理CLI — 运行完整流水线、查看状态、情景分析。

用法:
    python scripts/cli/system_cli.py status             # 系统状态
    python scripts/cli/system_cli.py run-daily          # 运行每日流水线
    python scripts/cli/system_cli.py run-weekly         # 运行每周流水线
    python scripts/cli/system_cli.py scenario           # 493K→100万情景分析
"""
import argparse
import json
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.orchestrator.data_backbone import DataBackbone


def cmd_status(args):
    bb = DataBackbone.get()
    health = bb.health_check()
    trade_date = bb.get_latest_trade_date()

    print("📊 系统状态")
    print("=" * 40)
    print(f"最近交易日: {trade_date}")
    for source, status in health.items():
        icon = "✓" if status == "ok" else "✗"
        print(f"  {icon} {source}: {status}")

    # 检查配置
    portfolio = bb.load_portfolio()
    if portfolio is not None:
        print(f"  ✓ 持仓: {len(portfolio)} 只")
    else:
        print("  ✗ 持仓: 未加载")

    # 检查关注列表
    wl_path = f"{project_root}/config/watchlist.json"
    if Path(wl_path).exists():
        with open(wl_path) as f:
            wl = json.load(f)
        print(f"  ✓ 关注列表: {len(wl)} 只")
    else:
        print("  - 关注列表: 未创建")


def cmd_daily(args):
    from scripts.orchestrator.scheduler import TradingScheduler
    sched = TradingScheduler()
    result = sched.run_daily_pipeline()
    if result:
        print(f"\n✓ 每日流水线完成")
        print(f"  仪表盘: {result['dashboard_path']}")


def cmd_weekly(args):
    from scripts.orchestrator.scheduler import TradingScheduler
    sched = TradingScheduler()
    result = sched.run_weekly_pipeline()
    if result:
        radar = result.get("radar")
        lab = result.get("lab")
        print(f"\n✓ 每周流水线完成")
        if radar:
            print(f"  新候选: {len(radar.new_candidates)} 只")
        if lab:
            print(f"  情景: {len(lab.scenarios)} 个")


def cmd_scenario(args):
    from scripts.orchestrator.scheduler import TradingScheduler
    sched = TradingScheduler()
    report = sched.run_scenario_analysis()
    print(report)


def main():
    parser = argparse.ArgumentParser(description="投资系统管理")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="系统状态")
    sub.add_parser("run-daily", help="运行每日流水线")
    sub.add_parser("run-weekly", help="运行每周流水线")
    sub.add_parser("scenario", help="493K→100万情景分析")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "run-daily": cmd_daily,
        "run-weekly": cmd_weekly,
        "scenario": cmd_scenario,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
