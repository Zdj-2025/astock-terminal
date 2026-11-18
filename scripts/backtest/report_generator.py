#!/usr/bin/env python3
"""
报告生成器 - 生成交互式HTML回测报告

特性:
- 使用Plotly生成交互式图表
- 净值曲线、回撤分析、月度收益、持仓分布
- 完整的绩效指标展示
- 专业的HTML排版

作者: AI Assistant
日期: 2026-04-04
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import sys

sys.path.insert(0, '/mnt/d/claude-code/stock')

# 尝试导入plotly，如果没有则使用基础图表
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly未安装，将使用简化版报告")


class BacktestReportGenerator:
    """
    回测报告生成器

    生成交互式HTML报告，包含:
    1. 执行摘要
    2. 绩效分析（可视化图表）
    3. 交易记录
    4. 风险分析
    """

    def __init__(self, output_dir: str = "data/backtest_results"):
        """
        初始化报告生成器

        参数:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"📝 初始化报告生成器")
        print(f"   输出目录: {output_dir}")
        print(f"   Plotly可用: {'✅' if PLOTLY_AVAILABLE else '❌'}")

    def generate_report(
        self,
        backtest_results: Dict,
        report_name: Optional[str] = None
    ) -> str:
        """
        生成完整的回测报告

        参数:
            backtest_results: 回测结果字典
            report_name: 报告名称（可选）

        返回:
            报告文件路径
        """
        print(f"\n📝 生成回测报告...")

        # 创建报告目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(report_dir, exist_ok=True)

        # 生成报告
        if PLOTLY_AVAILABLE:
            report_path = self._generate_html_report(
                backtest_results,
                report_dir,
                report_name or f"backtest_report_{timestamp}"
            )
        else:
            report_path = self._generate_text_report(
                backtest_results,
                report_dir,
                report_name or f"backtest_report_{timestamp}"
            )

        # 导出数据
        self._export_data(backtest_results, report_dir)

        print(f"✅ 报告生成完成")
        print(f"   报告路径: {report_path}")

        return report_path

    def _generate_html_report(
        self,
        results: Dict,
        report_dir: str,
        report_name: str
    ) -> str:
        """
        生成HTML格式报告（使用Plotly）

        参数:
            results: 回测结果
            report_dir: 报告目录
            report_name: 报告名称

        返回:
            HTML文件路径
        """
        # 提取数据
        metrics = results.get('performance_metrics', {})
        portfolio_values = results.get('portfolio_values', [])
        trades = results.get('trade_history', [])

        # 创建图表
        charts = {}

        # 1. 净值曲线
        if portfolio_values:
            charts['equity_curve'] = self._create_equity_curve(portfolio_values)

        # 2. 回撤分析
        if portfolio_values:
            charts['drawdown'] = self._create_drawdown_chart(portfolio_values)

        # 3. 月度收益热力图
        if portfolio_values and len(portfolio_values) > 20:
            charts['monthly_returns'] = self._create_monthly_heatmap(portfolio_values)

        # 4. 持仓分布（如果有交易）
        if trades:
            charts['position_dist'] = self._create_position_distribution(trades)

        # 生成HTML
        html_content = self._build_html(
            results,
            metrics,
            charts,
            report_name
        )

        # 保存报告
        report_path = os.path.join(report_dir, f"{report_name}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_path

    def _create_equity_curve(self, portfolio_values: List[float]) -> str:
        """
        创建净值曲线图

        参数:
            portfolio_values: 组合净值列表

        返回:
            Plotly图表HTML
        """
        fig = go.Figure()

        # 净值曲线
        fig.add_trace(go.Scatter(
            y=portfolio_values,
            mode='lines',
            name='组合净值',
            line=dict(color='#2E86DE', width=2)
        ))

        # 初始资金参考线
        initial_capital = portfolio_values[0]
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="初始资金",
            annotation_position="right"
        )

        fig.update_layout(
            title='净值曲线',
            xaxis_title='交易日',
            yaxis_title='资产价值 (¥)',
            template='plotly_white',
            hovermode='x unified',
            height=400
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _create_drawdown_chart(self, portfolio_values: List[float]) -> str:
        """
        创建回撤分析图

        参数:
            portfolio_values: 组合净值列表

        返回:
            Plotly图表HTML
        """
        # 计算回撤
        peak = portfolio_values[0]
        drawdowns = []

        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            drawdowns.append(dd * 100)  # 转换为百分比

        fig = go.Figure()

        # 回撤区域图
        fig.add_trace(go.Scatter(
            y=drawdowns,
            mode='lines',
            name='回撤',
            fill='tozeroy',
            line=dict(color='#FF6B6B', width=2),
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))

        # 最大回撤线
        max_dd = max(drawdowns)
        fig.add_hline(
            y=max_dd,
            line_dash="dash",
            line_color="red",
            annotation_text=f"最大回撤: {max_dd:.2f}%",
            annotation_position="right"
        )

        fig.update_layout(
            title='回撤分析',
            xaxis_title='交易日',
            yaxis_title='回撤 (%)',
            template='plotly_white',
            hovermode='x unified',
            height=300
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _create_monthly_heatmap(self, portfolio_values: List[float]) -> str:
        """
        创建月度收益热力图

        参数:
            portfolio_values: 组合净值列表

        返回:
            Plotly图表HTML
        """
        # 计算月度收益率（简化：按20个交易日为一个月）
        monthly_returns = []
        for i in range(20, len(portfolio_values), 20):
            ret = (portfolio_values[i] - portfolio_values[i-20]) / portfolio_values[i-20] * 100
            monthly_returns.append(ret)

        if not monthly_returns:
            return ""

        # 创建热力图数据
        # 简化：使用单行显示
        z = [monthly_returns]
        x = [f"Month {i+1}" for i in range(len(monthly_returns))]
        y = ["月度收益"]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{v:.2f}%" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="收益率 (%)")
        ))

        fig.update_layout(
            title='月度收益热力图',
            xaxis_title='月份',
            template='plotly_white',
            height=250
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _create_position_distribution(self, trades: List[Dict]) -> str:
        """
        创建持仓分布饼图

        参数:
            trades: 交易记录

        返回:
            Plotly图表HTML
        """
        # 统计每个股票的交易次数
        symbol_counts = {}
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        if not symbol_counts:
            return ""

        fig = go.Figure(data=[go.Pie(
            labels=list(symbol_counts.keys()),
            values=list(symbol_counts.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])

        fig.update_layout(
            title='交易分布',
            template='plotly_white',
            height=300
        )

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _build_html(
        self,
        results: Dict,
        metrics: Dict,
        charts: Dict,
        report_name: str
    ) -> str:
        """
        构建完整HTML报告

        参数:
            results: 回测结果
            metrics: 绩效指标
            charts: 图表字典
            report_name: 报告名称

        返回:
            HTML内容
        """
        # 提取指标
        returns = metrics.get('returns', {})
        risk = metrics.get('risk', {})
        risk_adjusted = metrics.get('risk_adjusted', {})
        trading = metrics.get('trading', {})

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_name} - 回测报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .metric-card .value.positive {{ color: #27ae60; }}
        .metric-card .value.negative {{ color: #e74c3c; }}
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 回测绩效报告</h1>
            <p>{report_name} | 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="section">
            <h2>📈 执行摘要</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>总收益率</h3>
                    <div class="value {'positive' if returns.get('total_return', 0) >= 0 else 'negative'}">
                        {returns.get('total_return_pct', 0):.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <h3>年化收益率</h3>
                    <div class="value {'positive' if returns.get('annualized_return', 0) >= 0 else 'negative'}">
                        {returns.get('annualized_return_pct', 0):.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <h3>最大回撤</h3>
                    <div class="value negative">
                        {risk.get('max_drawdown_pct', 0):.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <h3>夏普比率</h3>
                    <div class="value {'positive' if risk_adjusted.get('sharpe_ratio', 0) >= 1 else ''}">
                        {risk_adjusted.get('sharpe_ratio', 0):.2f}
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📊 收益分析</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>平均日收益</h3>
                    <div class="value">{returns.get('avg_daily_return_pct', 0):.3f}%</div>
                </div>
                <div class="metric-card">
                    <h3>年化波动率</h3>
                    <div class="value">{risk.get('volatility_pct', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>下行风险</h3>
                    <div class="value">{risk.get('downside_risk_pct', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>胜率</h3>
                    <div class="value">{trading.get('win_rate_pct', 0):.1f}%</div>
                </div>
            </div>
        </div>
"""

        # 添加图表
        if 'equity_curve' in charts:
            html += f"""
        <div class="section">
            <h2>📉 净值曲线</h2>
            <div class="chart-container">
                {charts['equity_curve']}
            </div>
        </div>
"""

        if 'drawdown' in charts:
            html += f"""
        <div class="section">
            <h2>⬇️ 回撤分析</h2>
            <div class="chart-container">
                {charts['drawdown']}
            </div>
        </div>
"""

        if 'monthly_returns' in charts:
            html += f"""
        <div class="section">
            <h2>📅 月度收益</h2>
            <div class="chart-container">
                {charts['monthly_returns']}
            </div>
        </div>
"""

        if 'position_dist' in charts:
            html += f"""
        <div class="section">
            <h2>🥧 交易分布</h2>
            <div class="chart-container">
                {charts['position_dist']}
            </div>
        </div>
"""

        # 添加详细指标
        html += f"""
        <div class="section">
            <h2>📋 详细指标</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>索提诺比率</h3>
                    <div class="value">{risk_adjusted.get('sortino_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>卡玛比率</h3>
                    <div class="value">{risk_adjusted.get('calmar_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>盈亏比</h3>
                    <div class="value">{trading.get('profit_loss_ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <h3>交易次数</h3>
                    <div class="value">{trading.get('total_trades', 0)}笔</div>
                </div>
                <div class="metric-card">
                    <h3>VaR (95%)</h3>
                    <div class="value">{risk.get('var_95_pct', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <h3>CVaR</h3>
                    <div class="value">{risk.get('cvar_pct', 0):.2f}%</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>A股混合策略分析系统 | 回测报告 | 生成于 {datetime.now().strftime("%Y年%m月%d日 %H:%M")}</p>
            <p style="margin-top: 10px; font-size: 0.8em;">
                ⚠️ 免责声明：本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。
            </p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def _generate_text_report(
        self,
        results: Dict,
        report_dir: str,
        report_name: str
    ) -> str:
        """
        生成文本格式报告（Plotly不可用时的降级方案）

        参数:
            results: 回测结果
            report_dir: 报告目录
            report_name: 报告名称

        返回:
            文本文件路径
        """
        metrics = results.get('performance_metrics', {})

        # 使用绩效计算器的摘要报告
        if 'performance_metrics' in results:
            from scripts.backtest.performance import PerformanceCalculator
            calculator = PerformanceCalculator()
            summary = calculator.generate_summary_report(metrics)
        else:
            summary = "绩效指标未计算"

        # 保存报告
        report_path = os.path.join(report_dir, f"{report_name}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{report_name} - 回测报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(summary)

        return report_path

    def _export_data(self, results: Dict, report_dir: str):
        """
        导出详细数据

        参数:
            results: 回测结果
            report_dir: 报告目录
        """
        # 导出绩效指标
        metrics_path = os.path.join(report_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results.get('performance_metrics', {}), f, indent=2, ensure_ascii=False)

        # 导出交易记录
        trades = results.get('trade_history', [])
        if trades:
            trades_path = os.path.join(report_dir, "trades.csv")
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')

        # 导出净值数据
        portfolio_values = results.get('portfolio_values', [])
        if portfolio_values:
            values_path = os.path.join(report_dir, "portfolio_values.csv")
            values_df = pd.DataFrame({
                'day': range(len(portfolio_values)),
                'value': portfolio_values
            })
            values_df.to_csv(values_path, index=False)


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试报告生成器")
    print("=" * 80)

    # 创建模拟回测结果
    np.random.seed(42)
    days = 252
    initial_capital = 1000000

    portfolio_values = [initial_capital]
    for i in range(days):
        daily_return = np.random.normal(0.0005, 0.02)
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)

    # 模拟交易记录
    trades = []
    for i in range(0, days, 60):
        trades.append({
            'date': datetime.now(),
            'symbol': '603881',
            'quantity': 100,
            'price': portfolio_values[i] / 10000,
            'commission': 5
        })

    # 计算绩效指标
    from scripts.backtest.performance import PerformanceCalculator
    calculator = PerformanceCalculator()
    metrics = calculator.calculate_all_metrics(portfolio_values, trades)

    # 构建回测结果
    backtest_results = {
        'initial_capital': initial_capital,
        'final_value': portfolio_values[-1],
        'portfolio_values': portfolio_values,
        'trade_history': trades,
        'performance_metrics': metrics
    }

    # 生成报告
    generator = BacktestReportGenerator()
    report_path = generator.generate_report(backtest_results, "测试报告")

    print(f"\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
