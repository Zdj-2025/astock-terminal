#!/usr/bin/env python3
"""缠论专用图表生成器 — Plotly K线+笔+线段+中枢+买卖点"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .engine import LevelResult, Stroke, Segment, Pivot, BSPoint

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None

COLOR_UP = '#e74c3c'
COLOR_DOWN = '#2ecc71'
COLOR_BI = '#f39c12'
COLOR_SEG = '#3498db'
COLOR_ZS = 'rgba(52,152,219,0.15)'
COLOR_ZS_BORDER = 'rgba(52,152,219,0.5)'
COLOR_BUY = '#27ae60'
COLOR_SELL = '#e74c3c'


def create_chanlun_chart(
    df: pd.DataFrame,
    level_result: LevelResult,
    title: str = "缠论分析",
    show_days: int = 120,
) -> str:
    """
    生成缠论分析图：K线+笔+线段+中枢+买卖点

    Parameters:
        df: OHLCV DataFrame (columns: open, high, low, close, volume, date)
        level_result: 缠论分析结果
        title: 图表标题
        show_days: 显示最近N天
    """
    if go is None:
        return "<p>Plotly未安装，无法生成图表</p>"

    df = df.tail(show_days).copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    rows = 2
    row_heights = [0.75, 0.25]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.03,
        subplot_titles=('', '成交量'),
    )

    # Row 1: K线
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN,
        increasing_fillcolor=COLOR_UP, decreasing_fillcolor=COLOR_DOWN,
        name='K线', showlegend=False,
    ), row=1, col=1)

    # 笔标注（连线）
    _add_strokes(fig, level_result.strokes, df, row=1)

    # 线段标注（粗连线）
    _add_segments(fig, level_result.segments, df, row=1)

    # 中枢区间（矩形）
    _add_pivots(fig, level_result.pivots, df, row=1)

    # 买卖点标注
    _add_bs_points(fig, level_result.bs_points, df, row=1)

    # Row 2: 成交量
    vol_colors = [COLOR_UP if df['close'].iloc[i] >= df['open'].iloc[i]
                  else COLOR_DOWN for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'],
        marker_color=vol_colors, name='成交量', showlegend=False,
    ), row=2, col=1)

    level_cn = {"daily": "日线", "weekly": "周线", "60min": "60分钟", "30min": "30分钟", "15min": "15分钟", "5min": "5分钟", "1min": "1分钟"}
    fig.update_layout(
        title=dict(text=f"{title} ({level_cn.get(level_result.level, '')})", font=dict(size=16)),
        height=600, template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=30),
    )

    return fig.to_html(include_plotlyjs=False, full_html=False, div_id='chanlun-chart')


def _find_date_index(df: pd.DataFrame, time_str: str) -> Optional[int]:
    """通过时间字符串找到DataFrame中对应的行索引"""
    if not time_str:
        return None
    try:
        target = pd.to_datetime(time_str)
        dates = pd.to_datetime(df['date'])
        mask = abs((dates - target).dt.total_seconds()) < 86400
        indices = df.index[mask]
        return indices[0] if len(indices) > 0 else None
    except Exception:
        return None


def _add_strokes(fig, strokes: List[Stroke], df: pd.DataFrame, row: int):
    """添加笔连线"""
    x_pts, y_pts = [], []
    for bi in strokes:
        begin_idx = _find_date_index(df, bi.begin_time)
        end_idx = _find_date_index(df, bi.end_time)
        if begin_idx is not None and end_idx is not None:
            x_pts.extend([df.loc[begin_idx, 'date'], df.loc[end_idx, 'date'], None])
            y_pts.extend([bi.begin_price, bi.end_price, None])

    if x_pts:
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts,
            mode='lines', line=dict(width=1.5, color=COLOR_BI),
            name='笔', opacity=0.9,
        ), row=row, col=1)


def _add_segments(fig, segments: List[Segment], df: pd.DataFrame, row: int):
    """添加线段连线"""
    x_pts, y_pts = [], []
    for seg in segments:
        begin_idx = _find_date_index(df, seg.begin_time)
        end_idx = _find_date_index(df, seg.end_time)
        if begin_idx is not None and end_idx is not None:
            x_pts.extend([df.loc[begin_idx, 'date'], df.loc[end_idx, 'date'], None])
            y_pts.extend([seg.begin_price, seg.end_price, None])

    if x_pts:
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts,
            mode='lines', line=dict(width=2.5, color=COLOR_SEG, dash='dot'),
            name='线段', opacity=0.8,
        ), row=row, col=1)


def _add_pivots(fig, pivots: List[Pivot], df: pd.DataFrame, row: int):
    """添加中枢矩形区间"""
    for i, zs in enumerate(pivots[-5:]):  # 最多显示5个中枢
        begin_idx = _find_date_index(df, zs.begin_time)
        end_idx = _find_date_index(df, zs.end_time)
        if begin_idx is not None and end_idx is not None:
            fig.add_shape(
                type="rect",
                x0=df.loc[begin_idx, 'date'], x1=df.loc[end_idx, 'date'],
                y0=zs.low, y1=zs.high,
                line=dict(color=COLOR_ZS_BORDER, width=1),
                fillcolor=COLOR_ZS,
                row=row, col=1,
            )
            fig.add_annotation(
                x=df.loc[end_idx, 'date'], y=zs.high,
                text=f"ZS[{zs.low:.1f},{zs.high:.1f}]",
                font=dict(size=9, color='#3498db'),
                showarrow=False, yshift=10,
                row=row, col=1,
            )


def _add_bs_points(fig, bs_points: List[BSPoint], df: pd.DataFrame, row: int):
    """添加买卖点标注"""
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    buy_text, sell_text = [], []

    for bsp in bs_points:
        idx = _find_date_index(df, bsp.time)
        if idx is None:
            continue
        date = df.loc[idx, 'date']
        types_str = ",".join(bsp.bsp_types)
        if bsp.is_buy:
            buy_x.append(date)
            buy_y.append(bsp.price)
            buy_text.append(f"买{types_str}")
        else:
            sell_x.append(date)
            sell_y.append(bsp.price)
            sell_text.append(f"卖{types_str}")

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y, mode='markers+text',
            marker=dict(symbol='triangle-up', size=12, color=COLOR_BUY),
            text=buy_text, textposition='bottom center',
            textfont=dict(size=9, color=COLOR_BUY),
            name='买点',
        ), row=row, col=1)

    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y, mode='markers+text',
            marker=dict(symbol='triangle-down', size=12, color=COLOR_SELL),
            text=sell_text, textposition='top center',
            textfont=dict(size=9, color=COLOR_SELL),
            name='卖点',
        ), row=row, col=1)
