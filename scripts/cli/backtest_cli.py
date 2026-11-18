"""
选股回测命令行工具

用法:
    # 查看今日推荐
    PYTHONPATH=... python scripts/cli/backtest_cli.py list --today

    # 查看某日推荐
    PYTHONPATH=... python scripts/cli/backtest_cli.py list --date 2026-05-20

    # 查看所有推荐
    PYTHONPATH=... python scripts/cli/backtest_cli.py list --all

    # 回测摘要
    PYTHONPATH=... python scripts/cli/backtest_cli.py summary

    # 手动记录一条推荐
    PYTHONPATH=... python scripts/cli/backtest_cli.py record --symbol 601678 --name 滨化股份 ...

    # 手动更新状态
    PYTHONPATH=... python scripts/cli/backtest_cli.py update --symbol 601678 --date 2026-05-20 --status hit_target --result 5.2

    # 收盘后跟踪
    PYTHONPATH=... python scripts/cli/backtest_cli.py track --date 2026-05-20
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest.pick_recorder import (
    record_pick, update_status, load_picks, load_all_picks, 
    backtest_summary, print_summary
)
from scripts.backtest.pick_tracker import track_daily, track_all
from datetime import datetime


def cmd_list(args):
    if args.all:
        df = load_all_picks()
    elif args.date:
        df = load_picks(args.date)
    else:
        df = load_picks()  # 今天
    
    if len(df) == 0:
        print("无推荐记录")
        return
    
    # 选择显示列
    display_cols = ["date", "symbol", "name", "source", "strategy", "grade", 
                    "recommend_price", "buy_zone", "target_price", "stop_loss",
                    "action", "status", "result_pct", "notes"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(df[display_cols].to_string(index=False))


def cmd_summary(args):
    df = load_all_picks()
    print_summary(picks_df=df)
    
    if args.detail and len(df) > 0:
        print("\n📋 详细记录:")
        for _, row in df.iterrows():
            status_icon = {"pending": "⏳", "triggered": "🟡", "hit_target": "🟢", 
                          "stopped_out": "🔴", "expired": "⚪"}.get(row["status"], "?")
            result = f"{row['result_pct']:+.1f}%" if pd.notna(row.get("result_pct")) else ""
            print(f"  {status_icon} {row['date']} {row['name']}({row['symbol']}) "
                  f"{row['grade']} → {row['status']} {result} | {row.get('notes','')}")


def cmd_record(args):
    df = record_pick(
        symbol=args.symbol,
        name=args.name,
        source=args.source or "manual_pick",
        strategy=args.strategy or "trend_v2",
        score=args.score or 0,
        grade=args.grade or "A",
        recommend_price=args.price,
        buy_zone=args.buy_zone or "",
        target_price=args.target or 0,
        stop_loss=args.stop or 0,
        reason=args.reason or "",
        algorithm_tags=args.tags or "",
        risk_tags=args.risk_tags or "",
        position_score=args.position_score,
        rsi=args.rsi,
        action=args.action or "pending",
        industry=args.industry or "",
        notes=args.notes or "",
    )
    print(f"✅ 已记录 {args.name}({args.symbol})")
    print(df.tail(1).to_string(index=False))


def cmd_update(args):
    update_status(
        symbol=args.symbol,
        date=args.date,
        status=args.status,
        result_pct=args.result,
        notes=args.notes or "",
    )
    print(f"✅ 已更新 {args.symbol} ({args.date}) → {args.status}")


def cmd_track(args):
    if args.all:
        track_all()
    else:
        track_daily(args.date)


def main():
    parser = argparse.ArgumentParser(description="选股回测工具")
    subparsers = parser.add_subparsers(dest="command")
    
    # list
    p_list = subparsers.add_parser("list", help="查看推荐记录")
    p_list.add_argument("--date", help="日期 YYYY-MM-DD")
    p_list.add_argument("--all", action="store_true", help="所有记录")
    p_list.set_defaults(func=cmd_list)
    
    # summary
    p_summary = subparsers.add_parser("summary", help="回测摘要")
    p_summary.add_argument("--detail", action="store_true", help="显示详细记录")
    p_summary.set_defaults(func=cmd_summary)
    
    # record
    p_record = subparsers.add_parser("record", help="记录推荐")
    p_record.add_argument("--symbol", required=True)
    p_record.add_argument("--name", required=True)
    p_record.add_argument("--price", type=float, required=True, help="推荐价格")
    p_record.add_argument("--target", type=float, required=True, help="目标价")
    p_record.add_argument("--stop", type=float, required=True, help="止损价")
    p_record.add_argument("--buy-zone", help="买入区间 如 11.5-11.7")
    p_record.add_argument("--source", default="manual_pick")
    p_record.add_argument("--strategy", default="trend_v2")
    p_record.add_argument("--score", type=float)
    p_record.add_argument("--grade")
    p_record.add_argument("--reason")
    p_record.add_argument("--tags", help="算法标签 逗号分隔")
    p_record.add_argument("--risk-tags", help="风险标签 逗号分隔")
    p_record.add_argument("--position-score", type=float)
    p_record.add_argument("--rsi", type=float)
    p_record.add_argument("--action", default="pending", choices=["buy_now", "wait_pullback", "watch"])
    p_record.add_argument("--industry")
    p_record.add_argument("--notes")
    p_record.set_defaults(func=cmd_record)
    
    # update
    p_update = subparsers.add_parser("update", help="更新状态")
    p_update.add_argument("--symbol", required=True)
    p_update.add_argument("--date", required=True, help="推荐日期")
    p_update.add_argument("--status", required=True, 
                          choices=["pending", "triggered", "hit_target", "stopped_out", "expired"])
    p_update.add_argument("--result", type=float, help="收益率%")
    p_update.add_argument("--notes")
    p_update.set_defaults(func=cmd_update)
    
    # track
    p_track = subparsers.add_parser("track", help="收盘后跟踪")
    p_track.add_argument("--date", help="指定日期")
    p_track.add_argument("--all", action="store_true", help="跟踪所有")
    p_track.set_defaults(func=cmd_track)
    
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
