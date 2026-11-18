#!/usr/bin/env python3
"""
图表生成器 - 生成财报分析可视化图表

支持图表:
- K线+MACD+RSI+成交量 四面板图 (Plotly)
- 财务指标季度趋势柱状图 (Plotly)
- 六维评分雷达图 (PyEcharts)
- 关键支撑压力位图 (Plotly)
- 持仓分布饼图 (Plotly)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from pyecharts import options as opts
    from pyecharts.charts import Radar
    PYECHARTS_AVAILABLE = True
except ImportError:
    PYECHARTS_AVAILABLE = False


# A股配色：红涨绿跌
COLOR_UP = '#e74c3c'
COLOR_DOWN = '#2ecc71'
COLOR_MA5 = '#f39c12'
COLOR_MA10 = '#3498db'
COLOR_MA20 = '#9b59b6'
COLOR_MA60 = '#1abc9c'


class ChartGenerator:
    """图表生成器 — 产出HTML片段供ReportBuilder组装"""

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise RuntimeError("Plotly未安装，无法生成图表")

    # ── K线四面板图 ──────────────────────────────────────────

    def create_kline_panel(
        self,
        df: pd.DataFrame,
        ma_data: Optional[Dict[int, pd.Series]] = None,
        macd_data: Optional[Dict[str, pd.Series]] = None,
        macd_fast: Optional[Dict[str, pd.Series]] = None,
        rsi_data: Optional[pd.Series] = None,
        title: str = "技术分析",
        show_days: int = 120,
    ) -> str:
        """
        K线+成交量+MACD+RSI 四合一面板

        Parameters:
            df: OHLCV DataFrame (columns: open, high, low, close, volume, date)
            ma_data: {5: Series, 10: Series, 20: Series, 60: Series}
            macd_data: {'dif': Series, 'dea': Series, 'hist': Series} — 标准12,26,9
            macd_fast: {'dif': Series, 'dea': Series, 'hist': Series} — 快速5,34,5（背离检测）
            rsi_data: RSI值Series
            title: 图表标题
            show_days: 显示最近N天
        """
        df = df.tail(show_days).copy()
        if ma_data:
            ma_data = {k: v.tail(show_days) for k, v in ma_data.items()}
        if macd_data:
            macd_data = {k: v.tail(show_days) for k, v in macd_data.items()}
        if rsi_data is not None:
            rsi_data = rsi_data.tail(show_days)

        rows = 4
        row_heights = [0.45, 0.15, 0.22, 0.18]
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03,
            subplot_titles=('', '成交量', 'MACD', 'RSI'),
        )

        # Row 1: K线
        colors = [COLOR_UP if c >= o else COLOR_DOWN for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
            increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
            name='K线',
            showlegend=False,
        ), row=1, col=1)

        # MA叠加
        ma_colors = {5: COLOR_MA5, 10: COLOR_MA10, 20: COLOR_MA20, 60: COLOR_MA60}
        if ma_data:
            for period, series in ma_data.items():
                if period in ma_colors and len(series) > 0:
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=series.values,
                        mode='lines', line=dict(width=1, color=ma_colors[period]),
                        name=f'MA{period}', opacity=0.8,
                    ), row=1, col=1)

        # Row 2: 成交量
        vol_colors = [COLOR_UP if df['close'].iloc[i] >= df['open'].iloc[i]
                      else COLOR_DOWN for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df['date'], y=df['volume'],
            marker_color=vol_colors, name='成交量',
            showlegend=False,
        ), row=2, col=1)

        # Row 3: MACD (标准 12,26,9 — 主趋势)
        if macd_data:
            fig.add_trace(go.Scatter(
                x=df['date'], y=macd_data['dif'].values,
                mode='lines', line=dict(width=1.5, color='#e74c3c'),
                name='DIF(12,26)',
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df['date'], y=macd_data['dea'].values,
                mode='lines', line=dict(width=1.5, color='#3498db'),
                name='DEA(12,26)',
            ), row=3, col=1)
            hist = macd_data['hist'].values
            hist_colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in hist]
            fig.add_trace(go.Bar(
                x=df['date'], y=hist,
                marker_color=hist_colors, name='MACD柱',
                showlegend=False,
            ), row=3, col=1)
            fig.add_hline(y=0, row=3, col=1, line=dict(width=0.5, color='gray'))

        # Row 3 overlay: MACD快速 5,34,5 — 背离检测用，虚线叠加
        if macd_fast:
            fig.add_trace(go.Scatter(
                x=df['date'], y=macd_fast['dif'].values,
                mode='lines', line=dict(width=1, color='#f39c12', dash='dot'),
                name='DIF(5,34)', opacity=0.7,
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df['date'], y=macd_fast['dea'].values,
                mode='lines', line=dict(width=1, color='#1abc9c', dash='dot'),
                name='DEA(5,34)', opacity=0.7,
            ), row=3, col=1)

        # Row 4: RSI
        if rsi_data is not None:
            fig.add_trace(go.Scatter(
                x=df['date'], y=rsi_data.values,
                mode='lines', line=dict(width=1.5, color='#9b59b6'),
                name='RSI',
            ), row=4, col=1)
            fig.add_hline(y=70, row=4, col=1, line=dict(width=0.8, color='red', dash='dash'))
            fig.add_hline(y=30, row=4, col=1, line=dict(width=0.8, color='green', dash='dash'))
            fig.add_hrect(y0=70, y1=100, row=4, col=1, fillcolor='red', opacity=0.05)
            fig.add_hrect(y0=0, y1=30, row=4, col=1, fillcolor='green', opacity=0.05)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=700, template='plotly_white',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=50, r=20, t=60, b=30),
        )
        fig.update_yaxes(title_text='', row=1, col=1)
        fig.update_yaxes(title_text='', row=2, col=1)
        fig.update_yaxes(title_text='', row=3, col=1)
        fig.update_yaxes(title_text='', row=4, col=1)

        return fig.to_html(include_plotlyjs=False, full_html=False, div_id='kline-chart')

    # ── 财务指标季度趋势柱状图 ─────────────────────────────────

    def create_financial_bars(
        self,
        quarters: List[str],
        revenues: List[float],
        profits: List[float],
        revenue_growth: Optional[List[float]] = None,
        profit_growth: Optional[List[float]] = None,
        title: str = "季度财务趋势",
    ) -> str:
        """
        营收+利润季度对比柱状图，可选叠加增长率折线

        Parameters:
            quarters: ['24Q1', '24Q2', ...]
            revenues: 营收（亿元）
            profits: 净利润（亿元）
            revenue_growth: 营收同比增长率(%)
            profit_growth: 净利润同比增长率(%)
        """
        has_growth = revenue_growth is not None or profit_growth is not None

        if has_growth:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
            )
        else:
            fig = go.Figure()

        # 营收柱
        fig.add_trace(go.Bar(
            x=quarters, y=revenues, name='营收(亿)',
            marker_color='#3498db', marker_line_width=0,
            offsetgroup='revenue',
        ), secondary_y=False)

        # 净利润柱
        fig.add_trace(go.Bar(
            x=quarters, y=profits, name='净利润(亿)',
            marker_color='#2ecc71', marker_line_width=0,
            offsetgroup='profit',
        ), secondary_y=False)

        # 增长率折线
        if revenue_growth:
            fig.add_trace(go.Scatter(
                x=quarters, y=revenue_growth, name='营收增长(%)',
                mode='lines+markers', line=dict(width=2, color='#e74c3c'),
                marker=dict(size=5),
            ), secondary_y=True)

        if profit_growth:
            fig.add_trace(go.Scatter(
                x=quarters, y=profit_growth, name='利润增长(%)',
                mode='lines+markers', line=dict(width=2, color='#f39c12', dash='dash'),
                marker=dict(size=5),
            ), secondary_y=True)

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=400, template='plotly_white',
            barmode='group',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=50, r=50, t=60, b=30),
        )
        fig.update_yaxes(title_text='金额(亿元)', secondary_y=False)
        if has_growth:
            fig.update_yaxes(title_text='增长率(%)', secondary_y=True)

        if has_growth:
            return fig.to_html(include_plotlyjs=False, full_html=False, div_id='financial-chart')
        return fig.to_html(include_plotlyjs=False, full_html=False, div_id='financial-chart')

    # ── 评分雷达图 ────────────────────────────────────────────

    def create_score_radar(
        self,
        scores: Dict[str, float],
        title: str = "综合评分",
    ) -> str:
        """
        六维评分雷达图 (PyEcharts)

        Parameters:
            scores: {'成长性': 8, '盈利能力': 7, ...} 各项1-10分
        """
        if not PYECHARTS_AVAILABLE:
            return self._fallback_radar(scores, title)

        schema = [opts.RadarIndicatorItem(name=k, max_=10) for k in scores.keys()]
        values = [list(scores.values())]

        radar = Radar(init_opts=opts.InitOpts(
            width="600px", height="400px",
            theme="white",
        ))
        radar.add_schema(
            schema=schema,
            shape="circle",
            splitarea_opt=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0.3)),
            splitline_opt=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#ddd")),
            axisline_opt=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="#999")),
            textstyle_opts=opts.TextStyleOpts(font_size=13, color="#333"),
        )
        radar.add(
            series_name="评分",
            data=values,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color="#3498db"),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#2980b9"),
            color="#2980b9",
        )
        radar.set_global_opts(
            title_opts=opts.TitleOpts(title=title, pos_left="center", pos_top="5px",
                                       title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
            legend_opts=opts.LegendOpts(is_show=False),
        )

        return radar.render_embed()

    def _fallback_radar(self, scores: Dict[str, float], title: str) -> str:
        """Plotly备用雷达图"""
        categories = list(scores.keys())
        values = list(scores.values())
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=categories_closed,
            fill='toself', fillcolor='rgba(52,152,219,0.3)',
            line=dict(color='#2980b9', width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title=dict(text=title, font=dict(size=16)),
            height=400, showlegend=False,
            margin=dict(l=40, r=40, t=60, b=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False, div_id='radar-chart')

    # ── 关键支撑压力位图 ──────────────────────────────────────

    def create_key_levels_chart(
        self,
        df: pd.DataFrame,
        levels: List[Dict[str, Any]],
        title: str = "关键支撑/压力位",
        show_days: int = 60,
    ) -> str:
        """
        价格走势 + 支撑/压力位标注

        Parameters:
            df: OHLCV with 'date', 'close'
            levels: [{'role': '支撑'|'压力', 'value': float, 'source': str, 'strength': str}]
        """
        df = df.tail(show_days).copy()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'],
            mode='lines', line=dict(width=2, color='#2c3e50'),
            name='收盘价',
        ))

        for lv in levels[:8]:  # 最多显示8条线
            color = '#27ae60' if lv['role'] == '支撑' else '#e74c3c'
            dash = 'solid' if lv['strength'] == '强' else 'dash'
            width = 2 if lv['strength'] == '强' else 1
            label = f"{lv['role']} {lv['value']:.2f} ({lv.get('source', '')})"

            fig.add_hline(
                y=lv['value'], line=dict(width=width, color=color, dash=dash),
                annotation_text=label,
                annotation=dict(
                    font=dict(size=10, color=color),
                    bgcolor='rgba(255,255,255,0.8)',
                ),
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=350, template='plotly_white',
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=50, r=20, t=60, b=30),
        )

        return fig.to_html(include_plotlyjs=False, full_html=False, div_id='levels-chart')

    # ── 持仓分布饼图 ──────────────────────────────────────────

    def create_portfolio_pie(
        self,
        holdings: List[Dict[str, Any]],
        title: str = "持仓分布",
    ) -> str:
        """
        持仓分布饼图

        Parameters:
            holdings: [{'name': '东方财富', 'market_value': 39400, 'pnl_pct': -24.5}, ...]
        """
        names = [h['name'] for h in holdings]
        values = [h['market_value'] for h in holdings]
        colors = []
        for h in holdings:
            pnl = h.get('pnl_pct', 0)
            if pnl > 10:
                colors.append('#27ae60')
            elif pnl > 0:
                colors.append('#2ecc71')
            elif pnl > -10:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        fig = go.Figure(data=[go.Pie(
            labels=names, values=values,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textfont=dict(size=12),
            hole=0.4,
        )])

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=350, template='plotly_white',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
            margin=dict(l=20, r=20, t=60, b=50),
        )

        return fig.to_html(include_plotlyjs=False, full_html=False, div_id='portfolio-chart')

    # ── 四维评分卡片数据 ──────────────────────────────────────

    def create_score_summary(self, scores: Dict[str, float], total: float) -> str:
        """生成评分摘要HTML片段"""
        grade = '优秀' if total >= 7.5 else '良好' if total >= 6 else '一般' if total >= 4.5 else '较弱'
        grade_color = '#27ae60' if total >= 7.5 else '#3498db' if total >= 6 else '#f39c12' if total >= 4.5 else '#e74c3c'

        items = []
        for dim, score in scores.items():
            bar_pct = score * 10
            if score >= 8:
                bar_color = '#27ae60'
            elif score >= 6:
                bar_color = '#3498db'
            elif score >= 4:
                bar_color = '#f39c12'
            else:
                bar_color = '#e74c3c'
            items.append(f'''
            <div class="score-item">
                <span class="score-label">{dim}</span>
                <div class="score-bar-bg">
                    <div class="score-bar" style="width:{bar_pct}%; background:{bar_color}"></div>
                </div>
                <span class="score-value">{score}</span>
            </div>''')

        return f'''
        <div class="score-summary">
            <div class="total-score" style="color:{grade_color}">
                <span class="score-number">{total}</span>
                <span class="score-grade">{grade}</span>
            </div>
            <div class="score-details">{"".join(items)}</div>
        </div>'''
