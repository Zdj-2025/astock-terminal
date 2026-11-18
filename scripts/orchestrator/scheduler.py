#!/usr/bin/env python3
"""调度器 — 每日/每周/每月自动运行模块。

使用APScheduler + A股交易日历。支持前台运行和后台daemon模式。
"""
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.orchestrator.data_backbone import DataBackbone


class TradingScheduler:
    """A股交易日调度器"""

    def __init__(self):
        self.bb = DataBackbone.get()

    def is_trading_day(self) -> bool:
        return self.bb.is_trade_day()

    def _resolve_trade_date(self) -> str:
        """获取有数据的最近交易日。当天数据未更新时回退到前一天。"""
        trade_date = self.bb.get_latest_trade_date()
        # 验证当天是否有日线数据
        df = self.bb.get_market_data(trade_date)
        if df is not None and not df.empty:
            return trade_date
        # 回退：用前10天的日历找最近有数据的交易日
        try:
            from datetime import timedelta
            end = trade_date
            start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
            cal = self.bb.ts.get_trade_cal(start=start, end=end)
            if cal is not None and not cal.empty:
                open_days = cal[cal["is_open"] == 1]["cal_date"].tolist()
                for d in open_days:
                    df2 = self.bb.get_market_data(d)
                    if df2 is not None and not df2.empty:
                        return d
        except Exception:
            pass
        return trade_date

    def run_daily_pipeline(self, trade_date: str = None):
        """运行每日流水线"""
        if not trade_date:
            trade_date = self._resolve_trade_date()
            if not trade_date:
                print("非交易日且无历史数据可用，跳过")
                return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动每日流水线 | {trade_date}")

        # 1. 市场脉搏
        from scripts.orchestrator.market_pulse import MarketPulse
        pulse = MarketPulse(self.bb)
        pulse_report = pulse.run(trade_date)

        # 2. 持仓守护
        from scripts.orchestrator.portfolio_guardian import PortfolioGuardian
        guardian = PortfolioGuardian(self.bb)
        guardian_report = guardian.run(trade_date)

        # 3. 决策控制台
        from scripts.orchestrator.decision_console import DecisionConsole
        console = DecisionConsole()
        market_data = {
            "regime": pulse_report.regime,
            "score": pulse_report.regime_score,
            "hot_sectors": [{"name": s["name"], "pct": s["pct"]}
                           for s in pulse_report.hot_sectors],
        }
        dashboard = console.generate_dashboard(guardian_report, market_data)
        path = console.save_report(dashboard, trade_date)
        print(f"仪表盘已保存: {path}")

        return {
            "pulse": pulse_report,
            "guardian": guardian_report,
            "dashboard_path": path,
        }

    def run_weekly_pipeline(self):
        """运行每周流水线"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动每周流水线")

        # 1. 股票雷达
        from scripts.orchestrator.stock_radar import StockRadar
        radar = StockRadar(self.bb)
        radar_report = radar.run_weekly()

        # 2. 策略实验室
        from scripts.orchestrator.strategy_lab import StrategyLab
        lab = StrategyLab(self.bb)
        lab_report = lab.run_full_report()

        # 3. 表现追踪
        from scripts.orchestrator.performance_tracker import PerformanceTracker
        tracker = PerformanceTracker()
        perf = tracker.calculate()

        print(f"雷达发现: {len(radar_report.new_candidates)} 个候选股")
        print(f"情景分析: {len(lab_report.scenarios)} 个场景")

        return {
            "radar": radar_report,
            "lab": lab_report,
            "performance": perf,
        }

    def run_scenario_analysis(self):
        """运行493K→100万情景分析"""
        from scripts.orchestrator.strategy_lab import StrategyLab
        lab = StrategyLab(self.bb)
        scenarios = lab.run_scenarios()

        lines = ["# 493K→100万 情景分析", ""]
        lines.append(f"当前资金: ¥493,138.70 | 目标: ¥1,000,000 | 剩余: 11个月")
        lines.append("")

        for s in scenarios:
            lines.append(f"## {s.name}")
            lines.append(f"- 年化收益: {s.annual_return}% | 月均: {s.monthly_return}%")
            lines.append(f"- 达标需: {s.months_to_target}个月 | 概率: {s.probability*100:.0f}%")
            lines.append(f"- 策略: {s.strategy}")
            lines.append(f"- 仓位: {s.position_style}")
            lines.append("")

        return "\n".join(lines)

    def start_daemon(self):
        """启动daemon模式（使用APScheduler）"""
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
        except ImportError:
            print("需要安装: pip install apscheduler")
            return

        sched = BlockingScheduler()

        # 每日流水线
        sched.add_job(self.run_daily_pipeline, "cron",
                      hour=15, minute=15, day_of_week="mon-fri",
                      id="daily_pipeline")

        # 每周流水线
        sched.add_job(self.run_weekly_pipeline, "cron",
                      hour=10, minute=0, day_of_week="sat",
                      id="weekly_pipeline")

        print("调度器已启动:")
        print("  - 每日 15:15 (周一至周五): 每日流水线")
        print("  - 每周 10:00 (周六): 每周流水线")
        print("按 Ctrl+C 停止")
        sched.start()
