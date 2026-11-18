"""
大A投研PDF报告生成器

调用 free_data_client 获取数据 → matplotlib画图 → reportlab合成PDF
支持飞书直接发送

Usage:
    python scripts/generate_report.py 603881
    python scripts/generate_report.py 603881 --output /tmp/report.pdf

Author: Hermes Agent
Created: 2026-06-07
"""

import argparse
import os
import sys
import io
import math
from datetime import datetime, timedelta
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from core.free_data_client import (
    get_realtime_quote, get_dragon_tiger, get_lockup_expiry,
    get_fund_flow, get_industry_ranking, get_northbound_flow,
    get_hot_stocks, get_stock_info, get_concept_blocks,
    get_holder_change, get_stock_info as get_em_info,
    normalize_code,
)

# ============================================================================
# 中文字体 & 颜色配置
# ============================================================================

# matplotlib 中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 暗黑主题色
BG_COLOR = '#0f0f1a'
CARD_BG = '#1a1a2e'
TEXT_COLOR = '#e0e0e0'
ACCENT = '#ff5a1f'
RED = '#ff4444'
GREEN = '#00cc66'
BLUE = '#4488ff'
ORANGE = '#ff8800'
GRAY = '#666666'

# reportlab 颜色
RL_BG = colors.HexColor('#0f0f1a')
RL_CARD = colors.HexColor('#1a1a2e')
RL_TEXT = colors.HexColor('#cccccc')
RL_ACCENT = colors.HexColor('#ff5a1f')
RL_RED = colors.HexColor('#ff4444')
RL_GREEN = colors.HexColor('#00cc66')

# 注册中文字体（reportlab）
def _register_chinese_font():
    """尝试注册系统中文字体"""
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Songti.ttc',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont('Chinese', fp))
                return 'Chinese'
            except Exception:
                continue
    # fallback
    return 'Helvetica'

FONT_NAME = _register_chinese_font()

# ============================================================================
# matplotlib 图表生成
# ============================================================================

def _style_ax(ax, title=""):
    """统一暗黑风格"""
    ax.set_facecolor(CARD_BG)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.grid(axis='y', alpha=0.15, color='#444444')


def chart_fund_flow_daily(day_flow_df, buf):
    """日级资金流柱状图"""
    fig, ax = plt.subplots(figsize=(7.5, 3), facecolor=BG_COLOR)
    _style_ax(ax, "主力资金流向（近20日）")

    if day_flow_df.empty:
        ax.text(0.5, 0.5, "暂无数据", ha='center', va='center',
                color=GRAY, fontsize=14, transform=ax.transAxes)
    else:
        dates = day_flow_df['日期'].tolist()
        vals = day_flow_df['主力净流入万'].tolist()
        bar_colors = [RED if v > 0 else GREEN for v in vals]
        ax.bar(range(len(dates)), vals, color=bar_colors, width=0.7, alpha=0.85)
        ax.set_xticks(range(0, len(dates), max(1, len(dates) // 6)))
        ax.set_xticklabels([dates[i] for i in range(0, len(dates), max(1, len(dates) // 6))],
                           rotation=30, fontsize=8)
        ax.axhline(y=0, color='#444444', linewidth=0.8, linestyle='--')
        ax.set_ylabel("净流入(万)", color=GRAY, fontsize=9)

    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)


def chart_northbound(nb_data, buf):
    """北向资金分钟走势图"""
    fig, ax = plt.subplots(figsize=(7.5, 2.8), facecolor=BG_COLOR)
    _style_ax(ax, "北向资金分钟级累计净买入（亿）")

    minutes_df = nb_data.get('minutes', pd.DataFrame())
    if minutes_df.empty:
        ax.text(0.5, 0.5, "暂无数据（非交易时间）", ha='center', va='center',
                color=GRAY, fontsize=14, transform=ax.transAxes)
    else:
        ax.plot(minutes_df['时间'], minutes_df['沪股通亿'], color=ORANGE, linewidth=1.5, label='沪股通')
        ax.plot(minutes_df['时间'], minutes_df['深股通亿'], color=BLUE, linewidth=1.5, label='深股通')
        ax.plot(minutes_df['时间'], minutes_df['合计亿'], color='white', linewidth=1.5,
                linestyle='--', label='合计', alpha=0.7)
        ax.axhline(y=0, color='#444444', linewidth=0.8, linestyle='--')
        ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor='#333', labelcolor=TEXT_COLOR)
        # 只显示部分 x 标签
        n = len(minutes_df)
        step = max(1, n // 8)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([minutes_df['时间'].iloc[i] for i in range(0, n, step)],
                           rotation=30, fontsize=8)

    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)


def chart_hot_tags(hot_df, buf):
    """热点题材频次图"""
    fig, ax = plt.subplots(figsize=(7.5, 3), facecolor=BG_COLOR)
    _style_ax(ax, "今日热门题材 TOP15")

    if hot_df.empty:
        ax.text(0.5, 0.5, "暂无数据（非交易日）", ha='center', va='center',
                color=GRAY, fontsize=14, transform=ax.transAxes)
    else:
        all_tags = []
        for reason in hot_df['题材标签'].dropna():
            tags = [t.strip() for t in str(reason).split("+") if t.strip()]
            all_tags.extend(tags)
        if all_tags:
            cnt = Counter(all_tags)
            top15 = cnt.most_common(15)
            names = [t[0] for t in top15][::-1]
            freqs = [t[1] for t in top15][::-1]
            ax.barh(range(len(names)), freqs, color=ACCENT, alpha=0.85, height=0.6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("频次", color=GRAY, fontsize=9)

    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)


def chart_industry(industry_df, buf):
    """行业涨跌幅图"""
    fig, ax = plt.subplots(figsize=(7.5, 3.5), facecolor=BG_COLOR)
    _style_ax(ax, "行业板块涨跌幅 TOP20")

    if industry_df.empty:
        ax.text(0.5, 0.5, "暂无数据", ha='center', va='center',
                color=GRAY, fontsize=14, transform=ax.transAxes)
    else:
        # 前10涨 + 后10跌
        top10 = industry_df.head(10)
        bot10 = industry_df.tail(10)
        show = pd.concat([top10, bot10.iloc[::-1]])
        names = show['行业'].tolist()[::-1]
        vals = [float(v) for v in show['涨跌幅%'].tolist()][::-1]
        bar_colors = [RED if v > 0 else GREEN for v in vals]
        ax.barh(range(len(names)), vals, color=bar_colors, alpha=0.85, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(x=0, color='#444444', linewidth=0.8)
        ax.set_xlabel("涨跌幅%", color=GRAY, fontsize=9)

    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)


def chart_concept_radar(concept_df, buf):
    """概念板块词云式水平图"""
    fig, ax = plt.subplots(figsize=(7.5, 2.5), facecolor=BG_COLOR)
    _style_ax(ax)

    if concept_df.empty:
        ax.text(0.5, 0.5, "概念板块暂无", ha='center', va='center',
                color=GRAY, fontsize=14, transform=ax.transAxes)
    else:
        # 只画概念类别
        concepts = concept_df[concept_df['类别'] == '概念'].head(12)
        if not concepts.empty:
            names = concepts['板块'].tolist()[::-1]
            # 尝试解析涨跌幅
            vals = []
            for v in concepts['涨跌幅%'].tolist()[::-1]:
                try:
                    vals.append(float(str(v).replace('%', '')))
                except:
                    vals.append(0)
            bar_colors = [RED if v > 0 else GREEN for v in vals]
            ax.barh(range(len(names)), vals, color=bar_colors, alpha=0.8, height=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.axvline(x=0, color='#444444', linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "暂无概念标签", ha='center', va='center',
                    color=GRAY, fontsize=12, transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# PDF 报告合成
# ============================================================================

def build_pdf(code, output_path=None):
    """生成完整PDF报告

    Args:
        code: 6位股票代码
        output_path: PDF输出路径（默认 /tmp/stock_report_{code}.pdf）

    Returns:
        str: PDF文件路径
    """
    code = normalize_code(code)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    if output_path is None:
        output_path = f"/tmp/stock_report_{code}_{date_str}.pdf"

    # ---- 获取所有数据 ----
    print(f"[1/6] 获取 {code} 数据...")
    quote_df = get_realtime_quote([code])
    info = get_stock_info(code)
    concept_df = get_concept_blocks(code)
    flow = get_fund_flow(code, days=20)
    lockup_df = get_lockup_expiry(code=code)
    holder_df = get_holder_change(code)
    dt_data = get_dragon_tiger(code=code)
    nb_data = get_northbound_flow()
    industry_df = get_industry_ranking()
    hot_df = get_hot_stocks()

    # ---- 生成图表 ----
    print("[2/6] 生成图表...")
    buf_flow = io.BytesIO(); chart_fund_flow_daily(flow['日流'], buf_flow); buf_flow.seek(0)
    buf_nb = io.BytesIO(); chart_northbound(nb_data, buf_nb); buf_nb.seek(0)
    buf_hot = io.BytesIO(); chart_hot_tags(hot_df, buf_hot); buf_hot.seek(0)
    buf_ind = io.BytesIO(); chart_industry(industry_df, buf_ind); buf_ind.seek(0)
    buf_concept = io.BytesIO(); chart_concept_radar(concept_df, buf_concept); buf_concept.seek(0)

    # ---- 构建 PDF ----
    print("[3/6] 合成PDF...")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=12*mm,
    )

    styles = getSampleStyleSheet()
    # 自定义样式
    s_title = ParagraphStyle('Title2', parent=styles['Title'],
                             fontName=FONT_NAME, fontSize=20, textColor=RL_ACCENT,
                             spaceAfter=4, spaceBefore=0, alignment=TA_CENTER)
    s_subtitle = ParagraphStyle('Sub', parent=styles['Normal'],
                                fontName=FONT_NAME, fontSize=9, textColor=RL_TEXT,
                                alignment=TA_CENTER, spaceAfter=8)
    s_section = ParagraphStyle('Section', parent=styles['Heading2'],
                               fontName=FONT_NAME, fontSize=13, textColor=RL_ACCENT,
                               spaceBefore=12, spaceAfter=4)
    s_body = ParagraphStyle('Body2', parent=styles['Normal'],
                            fontName=FONT_NAME, fontSize=9, textColor=RL_TEXT,
                            leading=14, spaceAfter=3)
    s_warn = ParagraphStyle('Warn', parent=styles['Normal'],
                            fontName=FONT_NAME, fontSize=7, textColor=colors.HexColor('#555555'),
                            alignment=TA_CENTER, spaceBefore=6)

    story = []

    # ---- 封面/标题 ----
    stock_name = info.get('名称', '') if info else ''
    q = quote_df.iloc[0] if not quote_df.empty else {}
    price = q.get('现价', 0)
    change = q.get('涨跌幅%', 0)
    mcap = q.get('总市值亿', 0)
    pe = q.get('PE_TTM', 0)
    pb = q.get('PB', 0)

    title_text = f"{code} {stock_name}" if stock_name else code
    story.append(Paragraph(f"📊 {title_text} 投研日报", s_title))
    story.append(Paragraph(f"{date_str} {time_str} | 数据源: 腾讯/东财/同花顺/百度", s_subtitle))
    story.append(HRFlowable(width="100%", thickness=1, color=RL_ACCENT, spaceAfter=8))

    # ---- 核心估值卡片 ----
    print("[4/6] 写入估值数据...")
    card_data = [
        ['现价', f'{price:.2f}', '涨跌幅', f'{change:+.2f}%',
         'PE(TTM)', f'{pe:.1f}', 'PB', f'{pb:.2f}'],
    ]
    if mcap > 10000:
        card_data.append(['总市值', f'{mcap/10000:.1f}万亿', '换手率',
                          f'{q.get("换手率%", 0):.2f}%',
                          '行业', info.get('行业', 'N/A'),
                          '上市', str(info.get('上市日期', 'N/A'))[:4]])
    else:
        card_data.append(['总市值', f'{mcap:.0f}亿', '换手率',
                          f'{q.get("换手率%", 0):.2f}%',
                          '行业', info.get('行业', 'N/A'),
                          '上市', str(info.get('上市日期', 'N/A'))[:4]])

    card_table = Table(card_data, colWidths=[35*mm, 30*mm, 25*mm, 25*mm, 25*mm, 30*mm, 15*mm, 20*mm])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), RL_CARD),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#888888')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (3, 0), (3, -1), RL_RED if change > 0 else RL_GREEN),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ('ALIGN', (5, 0), (5, -1), 'RIGHT'),
        ('ALIGN', (7, 0), (7, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#333333')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 6))

    # ---- 概念板块 ----
    print("[5/6] 写入板块/资金...")
    story.append(Paragraph("🏷️ 概念板块", s_section))
    story.append(Image(buf_concept, width=175*mm, height=55*mm))

    # ---- 资金流向 ----
    story.append(Paragraph("💰 个股资金流向（近20日）", s_section))
    story.append(Image(buf_flow, width=175*mm, height=65*mm))

    # ---- 北向资金 ----
    story.append(Paragraph("🧭 北向资金", s_section))
    nb_total = nb_data.get('total', 0)
    nb_text = (f"沪股通: {nb_data.get('hgt', 0):.2f}亿  "
               f"深股通: {nb_data.get('sgt', 0):.2f}亿  "
               f"合计: {nb_total:.2f}亿")
    story.append(Paragraph(nb_text, s_body))
    story.append(Image(buf_nb, width=175*mm, height=60*mm))

    # ---- 龙虎榜 ----
    story.append(PageBreak())
    story.append(Paragraph("🐉 龙虎榜", s_section))
    dt_records = dt_data.get('上榜记录', pd.DataFrame())
    if not dt_records.empty:
        # 简表
        rows = [['日期', '名称', '上榜原因', '净买入万', '换手率%']]
        for _, r in dt_records.head(10).iterrows():
            rows.append([
                str(r.get('日期', '')),
                str(r.get('名称', '')),
                str(r.get('上榜原因', ''))[:20],
                f"{r.get('净买入万', 0):.0f}",
                f"{r.get('换手率%', 0):.2f}",
            ])
        dt_table = Table(rows, colWidths=[25*mm, 20*mm, 55*mm, 25*mm, 20*mm])
        dt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), RL_ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), RL_CARD),
            ('TEXTCOLOR', (0, 1), (-1, -1), RL_TEXT),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#333333')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(dt_table)

        # 买卖席位
        buy_df = dt_data.get('买入席位', pd.DataFrame())
        sell_df = dt_data.get('卖出席位', pd.DataFrame())
        if not buy_df.empty or not sell_df.empty:
            story.append(Spacer(1, 6))
            seat_rows = [['营业部', '买入万', '卖出万', '净额万', '方向']]
            for _, r in buy_df.head(5).iterrows():
                seat_rows.append([str(r.get('营业部', ''))[:25],
                                  f"{r.get('买入万', 0):.0f}",
                                  f"{r.get('卖出万', 0):.0f}",
                                  f"{r.get('净额万', 0):.0f}", '买'])
            for _, r in sell_df.head(5).iterrows():
                seat_rows.append([str(r.get('营业部', ''))[:25],
                                  f"{r.get('买入万', 0):.0f}",
                                  f"{r.get('卖出万', 0):.0f}",
                                  f"{r.get('净额万', 0):.0f}", '卖'])
            seat_table = Table(seat_rows, colWidths=[65*mm, 22*mm, 22*mm, 22*mm, 15*mm])
            seat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), RL_CARD),
                ('TEXTCOLOR', (0, 1), (-1, -1), RL_TEXT),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#333333')),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(seat_table)
    else:
        story.append(Paragraph(f"{code} 近30日未上龙虎榜", s_body))

    # ---- 解禁日历 ----
    story.append(Spacer(1, 8))
    story.append(Paragraph("🔓 限售解禁", s_section))
    if not lockup_df.empty:
        lock_rows = [['解禁日期', '类型', '解禁数量', '占比%']]
        for _, r in lockup_df.head(8).iterrows():
            lock_rows.append([
                str(r.get('解禁日期', ''))[:10],
                str(r.get('解禁类型', ''))[:15],
                str(r.get('解禁数量', '')),
                str(r.get('解禁占比%', '')),
            ])
        lock_table = Table(lock_rows, colWidths=[30*mm, 40*mm, 40*mm, 25*mm])
        lock_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#663300')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), RL_CARD),
            ('TEXTCOLOR', (0, 1), (-1, -1), RL_TEXT),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#333333')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(lock_table)
    else:
        story.append(Paragraph("暂无解禁记录", s_body))

    # ---- 行业排名 ----
    story.append(Spacer(1, 8))
    story.append(Paragraph("📊 行业板块排名", s_section))
    story.append(Image(buf_ind, width=175*mm, height=72*mm))

    # ---- 热点题材 ----
    story.append(PageBreak())
    story.append(Paragraph("🔥 今日热点题材", s_section))
    story.append(Image(buf_hot, width=175*mm, height=65*mm))

    if not hot_df.empty:
        hot_rows = [['代码', '名称', '涨跌幅%', '题材标签']]
        for _, r in hot_df.head(30).iterrows():
            hot_rows.append([
                str(r.get('代码', '')),
                str(r.get('名称', '')),
                str(r.get('涨跌幅%', '')),
                str(r.get('题材标签', ''))[:35],
            ])
        hot_table = Table(hot_rows, colWidths=[20*mm, 22*mm, 18*mm, 100*mm])
        hot_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), RL_ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), RL_CARD),
            ('TEXTCOLOR', (0, 1), (-1, -1), RL_TEXT),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#333333')),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(hot_table)

    # ---- 页脚 ----
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#333333')))
    story.append(Paragraph(
        f"📊 大A投研日报 v1.0 | 生成时间: {date_str} {time_str} | "
        "数据: 腾讯/东财/同花顺/百度 | ⚠️ 仅供参考，不构成投资建议",
        s_warn
    ))

    # ---- 生成 ----
    print("[6/6] 输出PDF...")
    doc.build(story)
    print(f"✅ PDF报告已生成: {output_path}")
    return output_path


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大A投研PDF报告生成器")
    parser.add_argument("code", help="股票代码，如 603881")
    parser.add_argument("--output", "-o", help="输出PDF路径", default=None)
    args = parser.parse_args()

    pdf_path = build_pdf(args.code, args.output)
    print(f"\n发送到飞书: 在飞书聊天中发送 MEDIA:{pdf_path}")
