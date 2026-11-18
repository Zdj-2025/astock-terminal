#!/usr/bin/env python3
"""
决策执行器 - 输出标准化交易建议

职责:
1. 将TradeDecision格式化为标准交易建议（按CLAUDE.md模板）
2. 提供CLI友好输出
3. 支持批量决策（投资组合分析）

作者: AI Assistant
日期: 2026-04-07
"""
import sys
from typing import Dict, Optional, List

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.decision.decision_engine import DecisionEngine, TradeDecision


class DecisionExecutor:
    """
    决策执行器

    将决策引擎的输出格式化为用户可读的交易建议。
    """

    def __init__(self, config_dir: str = 'config'):
        self.engine = DecisionEngine(config_dir)

    def analyze_and_decide(
        self,
        symbol: str,
        name: str = '',
        kline_data=None,
        financial_data=None,
        skip_macd_resonance: bool = False,
        output_format: str = 'text'
    ) -> str:
        """
        分析并决策，返回格式化的交易建议

        参数:
            symbol: 股票/ETF代码
            name: 名称
            kline_data: 预加载K线数据
            financial_data: 预加载财务数据
            skip_macd_resonance: 跳过多尺度MACD
            output_format: 输出格式 text/markdown

        返回:
            格式化的交易建议文本
        """
        decision = self.engine.decide(
            symbol, name, kline_data, financial_data,
            skip_macd_resonance=skip_macd_resonance
        )

        if output_format == 'markdown':
            return self._format_markdown(decision)
        return self._format_text(decision)

    def analyze_portfolio(self, skip_macd_resonance: bool = True) -> str:
        """
        分析整个投资组合，生成综合决策

        参数:
            skip_macd_resonance: 跳过多尺度MACD（组合分析通常不需要）

        返回:
            组合分析报告
        """
        print("\n" + "=" * 80)
        print("📊 投资组合综合决策分析")
        print("=" * 80)

        decisions = []

        try:
            portfolio = self.engine.portfolio_mgr.load_portfolio()
            if portfolio.empty:
                print("   ⚠️  持仓为空")
                return "当前无持仓"

            for _, row in portfolio.iterrows():
                symbol = str(row.get('股票代码', '')).zfill(6)
                name = str(row.get('股票名称', ''))

                print(f"\n{'─'*60}")
                try:
                    decision = self.engine.decide(
                        symbol, name,
                        skip_macd_resonance=skip_macd_resonance
                    )
                    decisions.append(decision)
                except Exception as e:
                    print(f"   ❌ {symbol} {name} 决策失败: {str(e)[:50]}")

        except Exception as e:
            print(f"   ❌ 加载组合失败: {e}")
            return f"组合分析失败: {e}"

        return self._format_portfolio_report(decisions)

    # ── 格式化方法 ──

    def _format_text(self, d: TradeDecision) -> str:
        """格式化为纯文本建议（按CLAUDE.md模板）"""
        lines = []

        # 标题
        icon = '📈' if d.action == 'buy' else ('📉' if d.action == 'sell' else '💼')
        action_label = {'buy': '买入建议', 'sell': '卖出建议', 'hold': '持有/观望建议'}
        lines.append(f"{icon} {action_label.get(d.action, '建议')}: {d.symbol} {d.name}")
        lines.append("━" * 50)

        # 基本信息
        lines.append(f"\n**基本信息**")
        lines.append(f"当前价格: ¥{d.current_price:.2f}")
        lines.append(f"综合评分: {d.overall_score:.1f}/100")
        lines.append(f"信号置信度: {d.confidence}")

        # 操作建议
        action_icon = {'buy': '✅', 'sell': '❌', 'hold': '⏸️'}
        lines.append(f"\n**建议操作**")
        lines.append(f"{action_icon.get(d.action, '⏸️')} {d.action_text}")
        if d.action == 'buy':
            lines.append(f"建议仓位: {d.position_fraction*100:.1f}% (约¥{d.position_amount:,.0f})")
        lines.append(f"评级: {d.rating}")

        # 分析依据
        lines.append(f"\n**分析依据**")
        lines.append(f"🔹 技术面: {d.tech_score:.1f}分")
        lines.append(f"🔹 基本面: {d.fund_score:.1f}分")
        lines.append(f"🔹 MACD共振: {d.macd_score:.1f}分")
        if d.is_etf:
            lines.append(f"🔹 ETF指标: {d.etf_score:.1f}分")

        # ETF专用信息
        if d.is_etf:
            if d.cost_basis > 0:
                pnl_pct = d.pnl_vs_cost * 100
                lines.append(f"\n**ETF持仓分析**")
                lines.append(f"  成本价: ¥{d.cost_basis:.3f}")
                lines.append(f"  当前价: ¥{d.current_price:.3f}")
                lines.append(f"  浮动盈亏: {pnl_pct:+.2f}%")

            if d.core_advice:
                lines.append(f"\n**核心仓位建议**")
                lines.append(f"  {d.core_advice.get('action_text', '')}")
                for r in d.core_advice.get('reasons', [])[:3]:
                    lines.append(f"  • {r}")

            if d.trading_advice:
                lines.append(f"\n**交易仓位建议**")
                lines.append(f"  {d.trading_advice.get('action_text', '')}")
                if d.trading_advice.get('trade_size_pct', 0) > 0:
                    lines.append(f"  建议幅度: {d.trading_advice['trade_size_pct']:.1f}%")
                if d.trading_advice.get('trigger_price', 0) > 0:
                    lines.append(f"  触发价: ¥{d.trading_advice['trigger_price']:.3f}")
                for r in d.trading_advice.get('reasons', [])[:3]:
                    lines.append(f"  • {r}")

        # 买入/卖出理由
        reasons = d.buy_reasons if d.action == 'buy' else d.sell_reasons
        if reasons:
            label = '买入理由' if d.action == 'buy' else '卖出理由'
            lines.append(f"\n**{label}**")
            for r in reasons:
                lines.append(f"  • {r}")

        # 风控
        lines.append(f"\n**风险控制**")
        lines.append(f"⚠️ 止损价位: ¥{d.stop_loss_price:.2f} ({d.stop_loss_pct*100:.0f}%)")
        lines.append(f"🎯 止盈价位: ¥{d.take_profit_price:.2f} (+{d.take_profit_pct*100:.0f}%)")

        # 持仓信息
        if d.holding_info:
            lines.append(f"\n**持仓信息**")
            buy_price = d.holding_info.get('buy_price', 0)
            if buy_price > 0 and d.current_price > 0:
                pnl_pct = (d.current_price / buy_price - 1) * 100
                pnl_icon = '📈' if pnl_pct > 0 else '📉'
                lines.append(f"{pnl_icon} 盈亏: {pnl_pct:+.2f}% (成本¥{buy_price:.2f})")

        # 风险提示
        if d.risk_warnings:
            lines.append(f"\n**风险提示**")
            for w in d.risk_warnings:
                lines.append(f"  • {w}")
        lines.append("  • 市场有风险，投资需谨慎")

        return '\n'.join(lines)

    def _format_markdown(self, d: TradeDecision) -> str:
        """格式化为Markdown"""
        lines = []
        action_map = {'buy': '📈 买入建议', 'sell': '📉 卖出建议', 'hold': '💼 持有/观望建议'}
        lines.append(f"## {action_map.get(d.action, '建议')}: {d.symbol} {d.name}")
        lines.append("")
        lines.append(f"---")
        lines.append("")

        lines.append("### 基本信息")
        lines.append(f"- 当前价格: ¥{d.current_price:.2f}")
        lines.append(f"- 综合评分: **{d.overall_score:.1f}/100** ({d.rating})")
        lines.append(f"- 置信度: {d.confidence}")
        lines.append("")

        action_icon = {'buy': '✅ 买入', 'sell': '❌ 卖出', 'hold': '⏸️ 观望'}
        lines.append("### 建议操作")
        lines.append(f"**{action_icon.get(d.action, '观望')}**")
        if d.action == 'buy':
            lines.append(f"- 建议仓位: {d.position_fraction*100:.1f}% (约¥{d.position_amount:,.0f})")
        lines.append("")

        lines.append("### 分析依据")
        lines.append(f"| 维度 | 评分 |")
        lines.append(f"|------|------|")
        lines.append(f"| 技术面 | {d.tech_score:.1f} |")
        lines.append(f"| 基本面 | {d.fund_score:.1f} |")
        lines.append(f"| MACD共振 | {d.macd_score:.1f} |")
        if d.is_etf:
            lines.append(f"| ETF指标 | {d.etf_score:.1f} |")
        lines.append("")

        lines.append("### 风险控制")
        lines.append(f"- 止损: ¥{d.stop_loss_price:.2f} ({d.stop_loss_pct*100:.0f}%)")
        lines.append(f"- 止盈: ¥{d.take_profit_price:.2f} (+{d.take_profit_pct*100:.0f}%)")
        lines.append("")
        lines.append("> ⚠️ 市场有风险，投资需谨慎。本分析仅供参考，不构成投资建议。")

        return '\n'.join(lines)

    def _format_portfolio_report(self, decisions: List[TradeDecision]) -> str:
        """格式化组合分析报告"""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("📊 投资组合决策汇总")
        lines.append("=" * 80)

        if not decisions:
            lines.append("  无决策结果")
            return '\n'.join(lines)

        # 汇总表格
        buy_count = sum(1 for d in decisions if d.action == 'buy')
        sell_count = sum(1 for d in decisions if d.action == 'sell')
        hold_count = sum(1 for d in decisions if d.action == 'hold')

        lines.append(f"\n  总计: {len(decisions)}只 | 买入{buy_count} | 卖出{sell_count} | 持有/观望{hold_count}")
        lines.append("")

        # 每只股票的简要建议
        for d in decisions:
            action_icons = {'buy': '📈', 'sell': '📉', 'hold': '⏸️'}
            icon = action_icons.get(d.action, '⏸️')

            lines.append(f"  {icon} {d.symbol} {d.name}")
            lines.append(f"     评分: {d.overall_score:.1f} | {d.action_text} | {d.rating}")

            if d.holding_info and d.current_price > 0:
                buy_price = d.holding_info.get('buy_price', 0)
                if buy_price > 0:
                    pnl_pct = (d.current_price / buy_price - 1) * 100
                    lines.append(f"     盈亏: {pnl_pct:+.2f}% | 止损¥{d.stop_loss_price:.2f} | 止盈¥{d.take_profit_price:.2f}")

            if d.risk_warnings:
                for w in d.risk_warnings[:2]:
                    lines.append(f"     {w}")

            lines.append("")

        # 风险提醒
        lines.append("━" * 50)
        lines.append("⚠️ 以上分析仅供参考，不构成投资建议")
        lines.append("⚠️ 请结合市场环境和个人判断做出最终决策")

        return '\n'.join(lines)


# ── 测试 ──

if __name__ == '__main__':
    print("=" * 80)
    print("测试决策执行器")
    print("=" * 80)

    executor = DecisionExecutor()

    # 测试1: 单只股票决策
    print("\n--- 测试1: 单只股票决策 ---")
    result = executor.analyze_and_decide('603881', '数据港', skip_macd_resonance=True)
    print(result)

    print("\n\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
