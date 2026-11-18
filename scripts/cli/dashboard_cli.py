#!/usr/bin/env python3
"""每日仪表盘CLI — 生成并展示/推送每日交易仪表盘。

用法:
    python scripts/cli/dashboard_cli.py                    # 生成仪表盘
    python scripts/cli/dashboard_cli.py --push             # 生成并推送到飞书
    python scripts/cli/dashboard_cli.py --date 20260512    # 指定日期
"""
import argparse
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.orchestrator.data_backbone import DataBackbone
from scripts.orchestrator.market_pulse import MarketPulse
from scripts.orchestrator.portfolio_guardian import PortfolioGuardian
from scripts.orchestrator.decision_console import DecisionConsole


def main():
    parser = argparse.ArgumentParser(description="每日交易仪表盘")
    parser.add_argument("--date", help="交易日期 YYYYMMDD")
    parser.add_argument("--push", action="store_true", help="推送到飞书")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    args = parser.parse_args()

    bb = DataBackbone.get()
    trade_date = args.date or bb.get_latest_trade_date()

    print(f"📊 生成每日仪表盘 | {trade_date}")

    # 1. 市场脉搏
    print("  ⏳ 市场脉搏...")
    pulse = MarketPulse(bb)
    pulse_report = pulse.run(trade_date)

    # 2. 持仓守护
    print("  ⏳ 持仓守护...")
    guardian = PortfolioGuardian(bb)
    guardian_report = guardian.run(trade_date)

    # 3. 生成仪表盘
    print("  ⏳ 生成仪表盘...")
    console = DecisionConsole()
    market_data = {
        "regime": pulse_report.regime,
        "score": pulse_report.regime_score,
        "hot_sectors": [{"name": s["name"], "pct": s["pct"]}
                       for s in pulse_report.hot_sectors],
    }
    dashboard = console.generate_dashboard(guardian_report, market_data)
    path = console.save_report(dashboard, trade_date)

    # 输出
    print("\n" + "=" * 60)
    print(dashboard)
    print("=" * 60)
    print(f"\n📁 已保存: {path}")

    # 推送到飞书
    if args.push:
        print("  ⏳ 推送到飞书...")
        card_json = console.generate_feishu_card(guardian_report, market_data)
        try:
            # 使用MCP工具推送
            print("  ⚠️ 飞书推送需要通过Claude Code的MCP工具执行")
            print("  卡片内容已生成，请通过Claude手动推送或集成到调度器")
        except Exception as e:
            print(f"  ✗ 推送失败: {e}")


if __name__ == "__main__":
    main()
