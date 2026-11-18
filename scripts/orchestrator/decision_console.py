#!/usr/bin/env python3
"""决策控制台 — 生成每日仪表盘，聚合所有模块输出。

输入: MarketPulseReport + GuardianReport + 雷达/实验室数据
输出: Markdown仪表盘 + HTML仪表盘 + Feishu卡片
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.orchestrator.portfolio_guardian import GuardianReport, PositionStatus, PositionAlert


class DecisionConsole:
    """决策控制台 — 生成每日仪表盘"""

    def __init__(self):
        pass

    def generate_dashboard(self, guardian: GuardianReport,
                           market_data: Dict = None) -> str:
        """生成Markdown格式的每日仪表盘"""
        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines.append(f"# 每日交易仪表盘 | {guardian.trade_date}")
        lines.append(f"> 生成时间: {now} | 数据源: Tushare")
        lines.append("")

        # === 市场状态 ===
        lines.append("## 市场状态")
        if market_data:
            regime = market_data.get("regime", "unknown")
            score = market_data.get("score", 0)
            hot = market_data.get("hot_sectors", [])
            emoji = {"risk_on": "🟢", "neutral": "🟡", "risk_off": "🔴"}.get(regime, "⚪")
            lines.append(f"{emoji} {regime} | Score: {score:+.2f}")
            if hot:
                names = [h.get("name", str(h)) if isinstance(h, dict) else str(h) for h in hot[:5]]
                lines.append(f"热点板块: {', '.join(names)}")
        else:
            lines.append("*市场数据未加载*")
        lines.append("")

        # === 组合健康 ===
        health = guardian.portfolio_health_score
        health_emoji = "🟢" if health >= 80 else "🟡" if health >= 60 else "🟠" if health >= 40 else "🔴"
        lines.append(f"## 组合健康: {health_emoji} {health}/100")
        lines.append(f"总资产: ¥{guardian.total_capital:,.0f} | "
                     f"市值: ¥{guardian.total_market_value:,.0f} | "
                     f"现金: ¥{guardian.cash:,.0f} | "
                     f"仓位: {guardian.position_ratio:.1f}%")
        lines.append("")

        # === 目标进度 ===
        tp = guardian.target_progress
        if tp:
            lines.append("## 百万目标进度")
            lines.append(f"当前: ¥{tp['current']:,.0f} → 目标: ¥{tp['target']:,.0f} | "
                        f"进度: {tp['progress_pct']:.1f}% | "
                        f"差距: ¥{tp['gap']:,.0f}")
            lines.append(f"月均需: +{tp['monthly_target_pct']:.2f}% | 剩余: 11个月")
            lines.append("")

        # === 警报 ===
        if guardian.alerts:
            lines.append("## 警报")
            red = [a for a in guardian.alerts if a.severity == "red"]
            yellow = [a for a in guardian.alerts if a.severity == "yellow"]
            blue = [a for a in guardian.alerts if a.severity == "blue"]

            for a in red:
                lines.append(f"🔴 **{a.name}({a.code})**: {a.message} → {a.action}")
            for a in yellow[:5]:
                lines.append(f"🟡 {a.name}({a.code}): {a.message}")
            for a in blue[:3]:
                lines.append(f"🔵 {a.name}({a.code}): {a.message}")
            lines.append("")

        # === 持仓详情 ===
        lines.append("## 持仓状态")
        lines.append("")
        lines.append("| 持仓 | 现价 | 盈亏% | 市值 | 仓位% | 效率 | 操作 |")
        lines.append("|------|------|-------|------|-------|------|------|")
        for pos in guardian.positions:
            pnl_str = f"{pos.pnl_pct:+.1f}%"
            eff_str = f"{pos.efficiency_grade}({pos.efficiency_score:.0f})" if pos.efficiency_grade else "-"
            lines.append(f"| {pos.name} | {pos.current_price:.2f} | {pnl_str} | "
                        f"¥{pos.market_value:,.0f} | {pos.position_pct:.1f}% | "
                        f"{eff_str} | {pos.action_text} |")
        lines.append("")

        # === T+0机会 ===
        if guardian.t_opportunities:
            lines.append("## T+0机会")
            for opp in guardian.t_opportunities:
                lines.append(f"- **{opp['name']}({opp['code']})**: {opp['reason']}")
            lines.append("")

        # === 效率分布 ===
        edist = guardian.efficiency_distribution
        if edist:
            lines.append("## 资金效率分布")
            total = sum(edist.values())
            for grade in ["A", "B", "C", "D"]:
                cnt = edist.get(grade, 0)
                if total > 0:
                    pct = cnt / total * 100
                    lines.append(f"- {grade}级: {cnt}只 ({pct:.0f}%)")
            lines.append("")

        return "\n".join(lines)

    def generate_feishu_card(self, guardian: GuardianReport,
                             market_data: Dict = None) -> str:
        """生成飞书交互卡片的JSON内容"""
        now = datetime.now().strftime("%H:%M")
        health = guardian.portfolio_health_score
        health_text = "健康" if health >= 70 else "注意" if health >= 50 else "警告"

        elements = []

        # 头部后的市场状态
        if market_data:
            regime = market_data.get("regime", "unknown")
            score = market_data.get("score", 0)
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**市场状态**: {regime} ({score:+.2f})"}
            })

        # 组合概要
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md",
                     "content": f"**组合健康**: {health}/100 ({health_text})\n"
                                f"总资产 ¥{guardian.total_capital:,.0f} | "
                                f"仓位 {guardian.position_ratio:.1f}%"}
        })

        # 警报（最多5条）
        alert_lines = []
        for a in guardian.alerts[:5]:
            icon = {"red": "🔴", "yellow": "🟡", "blue": "🔵"}.get(a.severity, "⚪")
            alert_lines.append(f"{icon} {a.name}: {a.message[:30]}")
        if alert_lines:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md",
                         "content": "**警报**\n" + "\n".join(alert_lines)}
            })

        # 目标进度
        tp = guardian.target_progress
        if tp:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md",
                         "content": f"**百万目标**: {tp['progress_pct']:.1f}% | "
                                    f"月均需+{tp['monthly_target_pct']:.2f}%"}
            })

        # 持仓摘要表
        rows = []
        for pos in guardian.positions[:8]:
            pnl = f"{pos.pnl_pct:+.1f}%"
            eff = pos.efficiency_grade or "-"
            rows.append(f"{pos.name} {pnl} ({eff})")
        if rows:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md",
                         "content": "**持仓**\n" + " | ".join(rows)}
            })

        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text",
                          "content": f"每日仪表盘 {guardian.trade_date} {now}"},
                "template": "blue"
            },
            "elements": elements
        }

        import json
        return json.dumps(card, ensure_ascii=False)

    def save_report(self, content: str, trade_date: str, prefix: str = "每日仪表盘"):
        """保存报告到文件"""
        date_dir = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        out_dir = os.path.join(PROJECT_ROOT, "reports", date_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}_{trade_date}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        return out_path
