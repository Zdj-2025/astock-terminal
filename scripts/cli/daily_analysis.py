#!/usr/bin/env python3
"""
每日自动分析 + 微信推送

功能:
- 分析整个投资组合
- 检查止损/止盈触发
- 通过Server酱推送分析结果到微信
- 可设定时运行（cron或schedule）

用法:
  # 立即运行一次
  python scripts/cli/daily_analysis.py

  # 定时运行（每天9:45）
  python scripts/cli/daily_analysis.py --schedule

  # 设置cron（WSL/Linux）
  # 45 9 * * 1-5 cd /mnt/d/claude-code/stock && python scripts/cli/daily_analysis.py

作者: AI Assistant
日期: 2026-04-08
"""
import sys
import argparse
from datetime import datetime

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.decision.decision_executor import DecisionExecutor
from scripts.utils.notifier import Notifier


def run_daily_analysis(push: bool = True) -> list:
    """
    执行每日分析

    参数:
        push: 是否推送通知

    返回:
        决策结果列表
    """
    print("=" * 60)
    print(f"📊 每日持仓分析")
    print(f"   日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 初始化
    executor = DecisionExecutor()
    notifier = Notifier()

    # 分析组合
    try:
        portfolio = executor.engine.portfolio_mgr.load_portfolio()
        if portfolio.empty:
            print("   ⚠️  无持仓")
            return []
    except Exception as e:
        print(f"   ❌ 加载组合失败: {e}")
        return []

    decisions = []
    alerts = []

    for _, row in portfolio.iterrows():
        symbol = str(row.get('股票代码', '')).zfill(6)
        name = str(row.get('股票名称', ''))

        print(f"\n{'─'*50}")
        try:
            decision = executor.engine.decide(
                symbol, name, skip_macd_resonance=True
            )
            decisions.append({
                'symbol': decision.symbol,
                'name': decision.name,
                'action': decision.action,
                'action_text': decision.action_text,
                'overall_score': decision.overall_score,
                'is_etf': decision.is_etf,
                'trading_advice': decision.trading_advice,
                'current_price': decision.current_price,
            })

            # 检查预警
            if decision.action == 'sell' and decision.holding_info:
                alerts.append({
                    'type': 'stop_loss' if '止损' in decision.action_text else 'take_profit',
                    'symbol': decision.symbol,
                    'name': decision.name,
                    'message': decision.action_text
                })

        except Exception as e:
            print(f"   ❌ {symbol} 分析失败: {str(e)[:50]}")

    # 推送通知
    if push and notifier.enabled and decisions:
        # 1. 推送每日摘要
        notifier.send_daily_summary(decisions)

        # 2. 推送预警（如果有）
        for alert in alerts:
            notifier.send_alert(
                alert['type'],
                alert['symbol'],
                alert['name'],
                alert['message']
            )

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"📊 分析完成: {len(decisions)}只")
    for d in decisions:
        action_icons = {'buy': '📈', 'sell': '📉', 'hold': '⏸️'}
        icon = action_icons.get(d['action'], '⏸️')
        print(f"  {icon} {d['symbol']} {d['name']} | "
              f"{d['overall_score']:.0f}分 | {d['action_text']}")
    print("=" * 60)

    return decisions


def main():
    parser = argparse.ArgumentParser(description='每日持仓分析')
    parser.add_argument('--schedule', action='store_true',
                        help='定时运行模式（每天9:45）')
    parser.add_argument('--no-push', action='store_true',
                        help='不推送通知（仅终端输出）')
    args = parser.parse_args()

    if args.schedule:
        # 定时模式
        try:
            import schedule
        except ImportError:
            print("❌ 请安装 schedule: pip install schedule")
            return

        print("⏰ 定时模式启动，每天 09:45 执行分析")
        schedule.every().monday.at("09:45").do(run_daily_analysis)
        schedule.every().tuesday.at("09:45").do(run_daily_analysis)
        schedule.every().wednesday.at("09:45").do(run_daily_analysis)
        schedule.every().thursday.at("09:45").do(run_daily_analysis)
        schedule.every().friday.at("09:45").do(run_daily_analysis)

        print("   等待中... (Ctrl+C 退出)")
        import time
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        # 立即运行一次
        run_daily_analysis(push=not args.no_push)


if __name__ == '__main__':
    main()
