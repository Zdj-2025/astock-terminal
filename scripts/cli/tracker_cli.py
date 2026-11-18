#!/usr/bin/env python3
"""Portfolio tracker CLI: snapshot, trades, compare, summary.

Usage:
    python scripts/cli/tracker_cli.py snapshot --date 2026-04-22 --source screenshot
    python scripts/cli/tracker_cli.py trades --last 7
    python scripts/cli/tracker_cli.py compare --from 2026-04-21 --to 2026-04-22
    python scripts/cli/tracker_cli.py summary --last 30
    python scripts/cli/tracker_cli.py holding --code 603881
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.tracker.portfolio_tracker import PortfolioTracker
from scripts.tracker.snapshot_diff import format_diff_report
from scripts.tracker.trade_recorder import TradeRecorder


def load_config() -> dict:
    config_path = Path(project_root) / "config" / "tracker_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def cmd_snapshot(args):
    """Save a snapshot from a JSON data file or stdin."""
    config = load_config()
    db_path = str(Path(project_root) / config.get("db_path", "data/tracker.db"))
    tracker = PortfolioTracker(db_path)

    # Read data from file or stdin
    if args.data_file:
        with open(args.data_file) as f:
            data = json.load(f)
    else:
        print("请通过 --data-file 提供JSON数据文件，或使用管道输入")
        print("格式: {\"holdings\": [...], \"capital\": {...}}")
        return

    holdings = data.get("holdings", [])
    capital = data.get("capital")
    date = args.date
    source = args.source

    result = tracker.save_snapshot(date, source, holdings, capital)

    print(f"✅ 快照已保存: {date} ({source}), {result['saved_count']}只持仓")

    if result["changes"]:
        print()
        print(format_diff_report(result["changes"]))

    if result["trades_detected"]:
        print(f"\n📝 自动检测到 {len(result['trades_detected'])} 笔交易:")
        for t in result["trades_detected"]:
            print(f"  - {t['action']}: {t['stock']}")


def cmd_trades(args):
    """Show trade history."""
    config = load_config()
    db_path = str(Path(project_root) / config.get("db_path", "data/tracker.db"))
    recorder = TradeRecorder(db_path)

    kwargs = {"days": args.last}
    if args.code:
        kwargs["stock_code"] = args.code
    if args.type:
        kwargs["trade_type"] = args.type

    trades = recorder.get_trades(**kwargs)

    if not trades:
        print(f"最近{args.last}天无交易记录。")
        return

    print(f"📋 最近{args.last}天交易记录（共{len(trades)}笔）\n")
    print(recorder.format_trades_table(trades))

    summary = recorder.get_trade_summary(args.last)
    print(f"\n汇总: 买入{summary['buy_count']}笔(¥{summary['buy_amount']:,.2f}), "
          f"卖出{summary['sell_count']}笔(¥{summary['sell_amount']:,.2f})")


def cmd_compare(args):
    """Compare two snapshot dates."""
    config = load_config()
    db_path = str(Path(project_root) / config.get("db_path", "data/tracker.db"))
    tracker = PortfolioTracker(db_path)

    prev = tracker.get_snapshot_at(args.from_date)
    curr = tracker.get_snapshot_at(args.to_date)

    if not prev:
        print(f"⚠️ 未找到 {args.from_date} 的快照")
        return
    if not curr:
        print(f"⚠️ 未找到 {args.to_date} 的快照")
        return

    from scripts.tracker.snapshot_diff import diff_snapshots
    changes = diff_snapshots(prev, curr)

    print(f"📊 持仓对比: {args.from_date} → {args.to_date}\n")
    print(format_diff_report(changes))

    # Also show capital changes
    prev_cap = tracker.get_capital_at(args.from_date)
    curr_cap = tracker.get_capital_at(args.to_date)
    if prev_cap and curr_cap:
        print(f"\n💰 资金变化:")
        print(f"  总资产: ¥{prev_cap.get('total_assets', 0):,.2f} → ¥{curr_cap.get('total_assets', 0):,.2f}")
        print(f"  可用资金: ¥{prev_cap.get('available_cash', 0):,.2f} → ¥{curr_cap.get('available_cash', 0):,.2f}")


def cmd_summary(args):
    """Show portfolio summary over a period."""
    config = load_config()
    db_path = str(Path(project_root) / config.get("db_path", "data/tracker.db"))
    tracker = PortfolioTracker(db_path)

    # Latest holdings
    latest = tracker.get_latest_snapshot()
    if not latest:
        print("暂无持仓快照数据。")
        return

    latest_cap = tracker.get_latest_capital()

    print("📈 当前持仓概况\n")
    print(f"{'代码':<8} {'名称':<10} {'数量':>8} {'成本':>10} {'现价':>10} {'盈亏%':>8}")
    print("-" * 60)
    for h in sorted(latest, key=lambda x: x.get("market_value", 0), reverse=True):
        pnl = h.get("pnl_pct", 0) or 0
        pnl_str = f"{pnl:+.2f}%" if pnl else "N/A"
        print(f"{h['stock_code']:<8} {h['stock_name']:<10} {h['shares']:>8} "
              f"¥{h['cost_price']:>9.3f} ¥{h.get('current_price', 0):>9.3f} {pnl_str:>8}")

    if latest_cap:
        print(f"\n💰 资金状况:")
        print(f"  总资产: ¥{latest_cap.get('total_assets', 0):,.2f}")
        print(f"  可用资金: ¥{latest_cap.get('available_cash', 0):,.2f}")
        print(f"  持仓市值: ¥{latest_cap.get('market_value', 0):,.2f}")
        print(f"  总盈亏: ¥{latest_cap.get('total_pnl', 0):,.2f}")

    # Trade summary
    recorder = TradeRecorder(db_path)
    summary = recorder.get_trade_summary(args.last)
    print(f"\n📋 最近{args.last}天交易: "
          f"买入{summary['buy_count']}笔, 卖出{summary['sell_count']}笔")

    # Snapshot dates
    dates = tracker.get_snapshot_dates(10)
    if dates:
        print(f"\n📅 快照记录: {', '.join(dates[:5])}")


def cmd_holding(args):
    """Show current holding for a specific stock."""
    config = load_config()
    db_path = str(Path(project_root) / config.get("db_path", "data/tracker.db"))
    tracker = PortfolioTracker(db_path)

    code = args.code
    if not code.isdigit() or len(code) != 6:
        print(f"⚠️ 无效股票代码: {code}（需要6位数字）")
        return

    holding = tracker.get_cost_basis(code)
    if not holding:
        print(f"未找到 {code} 的持仓记录。")
        return

    print(f"📊 {holding['stock_name']}({holding['stock_code']})")
    print(f"  持仓: {holding['shares']}股")
    print(f"  成本: ¥{holding['cost_price']:.3f}")
    print(f"  现价: ¥{holding.get('current_price', 'N/A')}")
    print(f"  市值: ¥{holding['shares'] * holding['cost_price']:,.2f}")

    # Show recent trades for this stock
    recorder = TradeRecorder(db_path)
    trades = recorder.get_trades(days=30, stock_code=code)
    if trades:
        print(f"\n  最近30天交易:")
        for t in trades:
            arrow = "买" if t["direction"] == "buy" else "卖"
            print(f"    {t['trade_date']} {arrow} {t['shares']}股 × ¥{t['price']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="持仓跟踪与交易记录")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # snapshot
    p_snap = subparsers.add_parser("snapshot", help="保存持仓快照")
    p_snap.add_argument("--date", required=True, help="快照日期 YYYY-MM-DD")
    p_snap.add_argument("--source", default="screenshot",
                        choices=["screenshot", "jqktrader", "manual"])
    p_snap.add_argument("--data-file", help="JSON数据文件路径")

    # trades
    p_trades = subparsers.add_parser("trades", help="查看交易记录")
    p_trades.add_argument("--last", type=int, default=7, help="查看最近N天")
    p_trades.add_argument("--code", help="按股票代码筛选")
    p_trades.add_argument("--type", help="按交易类型筛选 (normal/t_buy/t_sell/t_buyback)")

    # compare
    p_comp = subparsers.add_parser("compare", help="对比两个日期的持仓")
    p_comp.add_argument("--from", dest="from_date", required=True, help="起始日期")
    p_comp.add_argument("--to", dest="to_date", required=True, help="结束日期")

    # summary
    p_sum = subparsers.add_parser("summary", help="持仓汇总")
    p_sum.add_argument("--last", type=int, default=30, help="统计最近N天")

    # holding
    p_hold = subparsers.add_parser("holding", help="查看单只股票持仓")
    p_hold.add_argument("--code", required=True, help="6位股票代码")

    args = parser.parse_args()

    if args.command == "snapshot":
        cmd_snapshot(args)
    elif args.command == "trades":
        cmd_trades(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "holding":
        cmd_holding(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
