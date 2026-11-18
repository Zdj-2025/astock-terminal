#!/usr/bin/env python3
"""
报告组装器 — 将分析数据和图表组装为完整HTML报告

使用Jinja2模板 + ChartGenerator图表
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from jinja2 import Environment, FileSystemLoader

from scripts.analyzers.chart_generator import ChartGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, 'templates')
# PROJECT_ROOT/scripts/analyzers/ -> PROJECT_ROOT
_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'reports', 'html')


class ReportBuilder:
    """HTML可视化报告组装器"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.chart_gen = ChartGenerator()
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=False,
        )
        self.env.filters['sign'] = lambda v: f"{v:+.1f}" if isinstance(v, (int, float)) else v

    def build_stock_report(self, data: Dict[str, Any]) -> str:
        """
        构建个股分析HTML报告

        Parameters:
            data: {
                'stock_code': '603881',
                'stock_name': '数据港',
                'industry': '数据中心',
                'current_price': 38.81,
                'daily_change': -1.2,
                'report_date': '2026-04-26',
                'position': {'shares': 2000, 'cost_price': 26.08, 'pnl_pct': -24.5, 'pnl_amount': -12760},
                'metrics': {
                    'revenue_growth': 44.3, 'profit_growth': 37.7,
                    'gross_margin': 74.0, 'roe': 14.03, 'pe': 20.8, 'debt_ratio': 78.3,
                },
                'quarterly': {
                    'quarters': ['24Q1', '24Q2', ...],
                    'revenues': [24.56, 24.89, ...],
                    'profits': [19.54, 21.02, ...],
                    'revenue_growth': [-12.6, -16.5, ...],
                    'profit_growth': [-3.7, -4.3, ...],
                    'rows': [
                        {'label': 'EPS', 'values': ['0.124', '0.133', ...]},
                        {'label': '毛利率', 'values': ['57.6%', '58.2%', ...]},
                        ...
                    ],
                },
                'kline': {
                    'df': DataFrame (OHLCV),
                    'ma': {5: Series, 10: Series, 20: Series, 60: Series},
                    'macd': {'dif': Series, 'dea': Series, 'hist': Series},
                    'rsi': Series,
                },
                'key_levels': [
                    {'role': '支撑', 'value': 19.43, 'source': 'MA20', 'strength': '强'},
                    ...
                ],
                'scores': {'成长性': 9, '盈利能力': 8, ...},
                'total_score': 7.5,
                'advice': {
                    'title': '保留持仓，设好止损',
                    'content': '基本面强劲...',
                    'stop_loss': '18.0',
                    'take_profit': '22.0',
                },
                'portfolio': [
                    {'name': '东方财富', 'market_value': 39400, 'pnl_pct': -24.5},
                    ...
                ],
            }

        Returns:
            HTML文件路径
        """
        charts = {}

        # 1. 财务趋势柱状图
        q = data.get('quarterly')
        if q and q.get('quarters'):
            charts['financial'] = self.chart_gen.create_financial_bars(
                quarters=q['quarters'],
                revenues=q.get('revenues', []),
                profits=q.get('profits', []),
                revenue_growth=q.get('revenue_growth'),
                profit_growth=q.get('profit_growth'),
                title="季度营收与净利润趋势",
            )

        # 2. K线四面板
        kl = data.get('kline')
        if kl and kl.get('df') is not None:
            charts['kline'] = self.chart_gen.create_kline_panel(
                df=kl['df'],
                ma_data=kl.get('ma'),
                macd_data=kl.get('macd'),
                macd_fast=kl.get('macd_fast'),
                rsi_data=kl.get('rsi'),
                title=f"{data['stock_name']} 技术分析",
            )

        # 3. 关键价位图
        levels = data.get('key_levels')
        if levels and kl and kl.get('df') is not None:
            charts['levels'] = self.chart_gen.create_key_levels_chart(
                df=kl['df'],
                levels=levels,
                title=f"关键支撑/压力位 (现价 {data['current_price']})",
            )

        # 4. 评分雷达图
        scores = data.get('scores')
        if scores:
            charts['radar'] = self.chart_gen.create_score_radar(
                scores=scores,
                title="六维评分",
            )

        # 5. 持仓饼图
        portfolio = data.get('portfolio')
        if portfolio:
            charts['portfolio'] = self.chart_gen.create_portfolio_pie(
                holdings=portfolio,
            )

        # 渲染模板
        template = self.env.get_template('stock_report.html')
        html = template.render(
            stock_code=data.get('stock_code', ''),
            stock_name=data.get('stock_name', ''),
            industry=data.get('industry', ''),
            current_price=data.get('current_price', ''),
            daily_change=data.get('daily_change'),
            report_date=data.get('report_date', datetime.now().strftime('%Y-%m-%d')),
            position=data.get('position'),
            metrics=data.get('metrics', {}),
            quarterly_data=data.get('quarterly'),
            charts=charts,
            scores=scores,
            total_score=data.get('total_score'),
            advice=data.get('advice'),
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        )

        # 写文件
        code = data.get('stock_code', 'unknown')
        name = data.get('stock_name', '')
        filename = f"{name}{code}_可视化报告.html"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return filepath
