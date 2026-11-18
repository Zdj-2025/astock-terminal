#!/usr/bin/env python3
"""选股报告PDF生成器。

使用fpdf2生成专业PDF报告，包含：
- 封面、结论速览、全景表格
- 因子评分详解（原始指标+分段逻辑）
- 企业深度分析（收入/利润/现金流/风险）
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

from fpdf import FPDF

from scripts.core.tushare_client import from_ts_code

# ── 颜色体系 ──────────────────────────────────────────────────────
C_PRIMARY    = (26, 82, 118)
C_BLUE       = (41, 128, 185)
C_LIGHT_BLUE = (214, 234, 248)
C_ORANGE     = (230, 126, 34)
C_LIGHT_ORANGE = (253, 235, 220)
C_WHITE      = (255, 255, 255)
C_BLACK      = (44, 62, 80)
C_GRAY       = (149, 165, 166)
C_LIGHT_GRAY = (245, 245, 245)
C_DARK_GRAY  = (100, 100, 100)
C_GREEN      = (39, 174, 96)
C_LIGHT_GREEN = (234, 250, 241)
C_RED        = (192, 57, 43)
C_LIGHT_RED  = (250, 230, 230)
C_YELLOW     = (241, 196, 15)
C_LIGHT_YELLOW = (255, 248, 220)

# ── 字体 ──────────────────────────────────────────────────────────
FONT_DIR = "/mnt/c/Windows/Fonts"
FONT_REGULAR = os.path.join(FONT_DIR, "msyh.ttc")
FONT_BOLD = os.path.join(FONT_DIR, "msyhbd.ttc")


def _fonts_available() -> bool:
    return os.path.exists(FONT_REGULAR)


class ScreenerPDF(FPDF):
    """选股报告PDF生成器"""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        if _fonts_available():
            self.add_font("msyh", "", FONT_REGULAR)
            self.add_font("msyh", "B", FONT_BOLD)
        self._font_name = "msyh" if _fonts_available() else "Helvetica"

    # ── UI组件 ──────────────────────────────────────────────────────

    def draw_top_bar(self, height=6):
        self.set_fill_color(*C_PRIMARY)
        self.rect(0, 0, 210, height, "F")
        self.set_fill_color(*C_ORANGE)
        self.rect(0, height, 210, 1.5, "F")

    def draw_footer(self, text=""):
        self.set_font(self._font_name, "", 7)
        self.set_text_color(*C_GRAY)
        self.set_xy(15, 285)
        self.cell(180, 5, text or "报告由选股子系统自动生成。不构成投资建议。",
                  align="C")

    def section_title(self, title, y=None):
        if y is not None:
            self.set_y(y)
        y = self.get_y()
        self.set_fill_color(*C_BLUE)
        self.rect(15, y, 3, 8, "F")
        self.set_xy(21, y)
        self.set_font(self._font_name, "B", 13)
        self.set_text_color(*C_PRIMARY)
        self.cell(0, 8, title)
        self.ln(12)

    def draw_table(self, x, y, headers, rows, col_widths,
                   header_color=C_PRIMARY):
        """画表格，返回表格底部y坐标"""
        # 检查是否需要换页
        table_h = 7 + len(rows) * 6.5
        if y + table_h > 275:
            self.add_page()
            self.draw_top_bar()
            y = 15

        # 表头
        self.set_fill_color(*header_color)
        self.set_text_color(*C_WHITE)
        self.set_font(self._font_name, "B", 8)
        cx = x
        for i, h in enumerate(headers):
            self.set_xy(cx, y)
            self.cell(col_widths[i], 7, str(h), border=0, align="C",
                      fill=True)
            cx += col_widths[i]

        # 数据行
        for ri, row in enumerate(rows):
            ry = y + 7 + ri * 6.5
            bg = C_LIGHT_GRAY if ri % 2 == 0 else C_WHITE
            self.set_fill_color(*bg)
            self.set_text_color(*C_BLACK)
            self.set_font(self._font_name, "", 7.5)
            cx = x
            for ci, cell in enumerate(row):
                self.set_xy(cx, ry)
                align = "C" if ci > 0 else "L"
                self.cell(col_widths[ci], 6.5, str(cell), border=0,
                          align=align, fill=True)
                cx += col_widths[ci]

        return y + 7 + len(rows) * 6.5

    def draw_big_number(self, x, y, number, label, color=C_BLUE):
        self.set_xy(x, y)
        self.set_font(self._font_name, "B", 22)
        self.set_text_color(*color)
        self.cell(60, 12, str(number), align="C")
        self.set_xy(x, y + 12)
        self.set_font(self._font_name, "", 8)
        self.set_text_color(*C_GRAY)
        self.cell(60, 5, label, align="C")

    def draw_card(self, x, y, w, h, title, items,
                  bg_color=C_LIGHT_BLUE, tag=""):
        self.set_fill_color(*bg_color)
        self.rect(x, y, w, h, "F")
        self.set_fill_color(*C_PRIMARY)
        self.rect(x, y, w, 1.2, "F")
        if tag:
            colors = {
                "推荐": C_GREEN, "关注": C_BLUE,
                "high": C_RED, "medium": C_ORANGE, "low": C_YELLOW,
            }
            tc = colors.get(tag, C_BLUE)
            self.set_fill_color(*tc)
            self.set_xy(x + w - 24, y + 3)
            self.set_font(self._font_name, "B", 7)
            self.set_text_color(*C_WHITE)
            self.cell(20, 5, tag, fill=True, align="C")
        self.set_xy(x + 4, y + 4)
        self.set_font(self._font_name, "B", 10)
        self.set_text_color(*C_PRIMARY)
        self.cell(w - 8, 6, title)
        cy = y + 13
        self.set_font(self._font_name, "", 8)
        self.set_text_color(*C_BLACK)
        for item in items:
            if cy > y + h - 5:
                break
            self.set_xy(x + 5, cy)
            self.cell(w - 10, 5, str(item))
            cy += 6

    def draw_risk_tag(self, x, y, level, text):
        colors = {
            "high": (C_RED, C_LIGHT_RED),
            "medium": (C_ORANGE, C_LIGHT_ORANGE),
            "low": (C_YELLOW, C_LIGHT_YELLOW),
        }
        label_c, bg_c = colors.get(level, (C_GRAY, C_LIGHT_GRAY))
        level_text = {"high": "高风险", "medium": "中风险",
                      "low": "低风险"}.get(level, level)
        self.set_fill_color(*bg_c)
        self.rect(x, y, 180, 7, "F")
        self.set_fill_color(*label_c)
        self.set_xy(x, y)
        self.set_font(self._font_name, "B", 7)
        self.set_text_color(*C_WHITE)
        self.cell(18, 7, level_text, fill=True, align="C")
        self.set_xy(x + 20, y)
        self.set_font(self._font_name, "", 8)
        self.set_text_color(*C_BLACK)
        self.cell(158, 7, text)

    def draw_score_bar(self, x, y, label, value, w=60):
        """画一个分数条"""
        self.set_xy(x, y)
        self.set_font(self._font_name, "", 8)
        self.set_text_color(*C_BLACK)
        self.cell(18, 5, label)
        # 背景条
        self.set_fill_color(220, 220, 220)
        self.rect(x + 18, y + 1, w, 4, "F")
        # 分数条
        bar_w = value / 100 * w
        bar_color = C_GREEN if value >= 70 else C_BLUE if value >= 50 else C_ORANGE
        self.set_fill_color(*bar_color)
        self.rect(x + 18, y + 1, bar_w, 4, "F")
        # 分数
        self.set_xy(x + 18 + w + 2, y)
        self.set_font(self._font_name, "", 8)
        self.cell(15, 5, f"{value:.0f}")


# ── 主生成函数 ─────────────────────────────────────────────────────

def generate_pdf_report(
    results,
    meta: dict,
    factor_details: dict,
    company_analyses: list,
    output_path: str,
) -> str:
    """生成完整PDF选股报告。"""
    pdf = ScreenerPDF()
    keyword = meta.get("concept") or meta.get("industry") or \
              meta.get("template_name", "选股")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # ═══════════════════════════════════════════════════════════════
    # 第1页 — 封面
    # ═══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.draw_top_bar(height=50)
    pdf.set_fill_color(*C_PRIMARY)
    pdf.rect(0, 0, 210, 50, "F")
    pdf.set_fill_color(*C_ORANGE)
    pdf.rect(0, 50, 210, 2, "F")

    pdf.set_xy(20, 14)
    pdf.set_font(pdf._font_name, "B", 24)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(170, 12, f"{keyword}选股报告", align="C")

    pdf.set_xy(20, 28)
    pdf.set_font(pdf._font_name, "", 12)
    pdf.set_text_color(200, 220, 240)
    scope = meta.get("concept") or meta.get("industry") or "全市场"
    pdf.cell(170, 8,
             f"{date_str} | {meta.get('template_name', '自定义')} | {scope}",
             align="C")

    # 大数字
    pdf.draw_big_number(20, 70, str(meta.get("candidates_count", 0)),
                        "候选池", C_BLUE)
    pdf.draw_big_number(80, 70, str(meta.get("results_count", 0)),
                        "入选", C_GREEN)
    pdf.draw_big_number(140, 70,
                        str(meta.get("min_score", 70)),
                        "最低分", C_ORANGE)

    # 结论速览
    pdf.section_title("结论速览", y=100)

    if not results.empty:
        top3 = results.head(3)
        for i, row in top3.iterrows():
            code = from_ts_code(row["ts_code"])
            name = str(row.get("name", ""))
            score = row.get("composite", 0)
            tag = "推荐" if i == 0 else "关注"
            pdf.draw_card(15, pdf.get_y(), 180, 20,
                          f"{i+1}. {name}({code})",
                          [f"综合分 {score:.1f} | "
                           f"成长{row.get('growth_score', 0):.0f} "
                           f"质量{row.get('quality_score', 0):.0f} "
                           f"价值{row.get('value_score', 0):.0f} "
                           f"动量{row.get('momentum_score', 0):.0f} "
                           f"主线{row.get('theme_score', 0):.0f}"],
                          tag=tag)
            pdf.set_y(pdf.get_y() + 22)

    pdf.draw_footer()

    # ═══════════════════════════════════════════════════════════════
    # 第2页 — 入选股全景表 + 分数条
    # ═══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.draw_top_bar()
    pdf.section_title("入选股全景")

    if not results.empty:
        # 精简表格：排名、代码、名称、综合、成长、质量、价值、动量、PE、PB、市值
        headers = ["#", "代码", "名称", "综合", "成长", "质量",
                    "价值", "动量", "PE", "PB", "市值(亿)"]
        # 总宽度 = 180 (15mm左边距 → 195mm右边界)
        col_w = [8, 14, 18, 13, 13, 13, 13, 13, 14, 10, 16]
        # 总和 = 8+14+18+13+13+13+13+13+14+10+16 = 145，居中偏左
        tbl_x = 12
        rows = []
        for i, row in results.iterrows():
            code = from_ts_code(row["ts_code"])
            name = str(row.get("name", ""))[:5]
            pe = row.get("pe_ttm")
            pb = row.get("pb")
            mv = row.get("total_mv", 0) / 1e4 if row.get("total_mv") else 0
            pe_str = f"{pe:.0f}" if pe and not np.isnan(pe) else "-"
            pb_str = f"{pb:.1f}" if pb and not np.isnan(pb) else "-"
            rows.append([
                str(i + 1), code, name,
                f"{row.get('composite', 0):.1f}",
                f"{row.get('growth_score', 0):.0f}",
                f"{row.get('quality_score', 0):.0f}",
                f"{row.get('value_score', 0):.0f}",
                f"{row.get('momentum_score', 0):.0f}",
                pe_str, pb_str, f"{mv:.0f}",
            ])
        bottom_y = pdf.draw_table(tbl_x, pdf.get_y(), headers, rows, col_w)
        pdf.set_y(bottom_y + 5)

    # 分数条展示（每只股票一行5个因子条）
    pdf.ln(3)
    if not results.empty:
        for i, row in results.head(5).iterrows():
            name = str(row.get("name", ""))[:5]
            code = from_ts_code(row["ts_code"])
            pdf.set_font(pdf._font_name, "B", 9)
            pdf.set_text_color(*C_PRIMARY)
            pdf.cell(0, 6, f"{i+1}. {name}({code})")
            pdf.ln(7)

            labels = ["成长", "质量", "价值", "动量", "主线"]
            vals = [
                row.get("growth_score", 0),
                row.get("quality_score", 0),
                row.get("value_score", 0),
                row.get("momentum_score", 0),
                row.get("theme_score", 0),
            ]
            # 5个分数条横排
            bar_x = 20
            for lbl, val in zip(labels, vals):
                pdf.draw_score_bar(bar_x, pdf.get_y(), lbl, val, w=24)
                bar_x += 35
            pdf.ln(9)

    pdf.draw_footer()

    # ═══════════════════════════════════════════════════════════════
    # 第3页 — 因子评分详解
    # ═══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.draw_top_bar()
    pdf.section_title("因子评分详解")

    # 因子说明表
    factor_desc = [
        ["成长性", "营收增速+利润增速", "各0-50分，分段累加"],
        ["质量", "ROE+毛利率+负债率", "ROE(0-40)+毛利(0-30)+负债(0-30)"],
        ["价值", "PE(TTM)+PB", "PE分段(0-55)+PB分段(0-45)"],
        ["动量", "20日涨幅+量比", "涨幅(0-60)+量能(0-40)"],
        ["主线", "概念板块匹配", "基础30+命中加分x10"],
    ]
    pdf.draw_table(15, pdf.get_y(),
                   ["因子", "计算方法", "分段规则"],
                   factor_desc, [15, 70, 100])

    # 每只股票的因子拆解
    pdf.ln(5)
    pdf.set_font(pdf._font_name, "B", 11)
    pdf.set_text_color(*C_PRIMARY)
    pdf.cell(0, 8, "个股因子拆解")
    pdf.ln(10)

    for i, row in results.head(5).iterrows():
        ts_code = row["ts_code"]
        name = str(row.get("name", ""))
        code = from_ts_code(ts_code)
        details = factor_details.get(ts_code, {})

        # 检查剩余空间，不够就换页
        if pdf.get_y() > 240:
            pdf.add_page()
            pdf.draw_top_bar()
            pdf.ln(5)

        pdf.set_font(pdf._font_name, "B", 9)
        pdf.set_text_color(*C_BLACK)
        pdf.cell(0, 6, f"{i+1}. {name}({code})")
        pdf.ln(7)

        detail_rows = []
        for dim in ["growth", "quality", "value"]:
            d = details.get(dim, {})
            raw_parts = " | ".join(
                f"{k}={v}" for k, v in d.get("raw", {}).items())
            score_parts = " + ".join(
                f"{k.split('(')[0]}{v:.0f}"
                for k, v in d.get("breakdown", {}).items())
            total = d.get("total", 0)
            detail_rows.append([
                {"growth": "成长", "quality": "质量",
                 "value": "价值"}.get(dim, dim),
                raw_parts[:50],
                f"{score_parts} = {total:.0f}",
            ])

        pdf.draw_table(15, pdf.get_y(),
                       ["维度", "原始指标", "评分"],
                       detail_rows, [15, 95, 70])
        pdf.ln(8)

    pdf.draw_footer()

    # ═══════════════════════════════════════════════════════════════
    # 第4-6页 — 企业深度分析（每只1页）
    # ═══════════════════════════════════════════════════════════════
    for ca in company_analyses:
        pdf.add_page()
        pdf.draw_top_bar()

        name = ca.get("name", "")
        industry = ca.get("industry", "")
        ts_code = ca.get("ts_code", "")
        code = from_ts_code(ts_code)

        # 标题
        pdf.set_xy(15, 10)
        pdf.set_font(pdf._font_name, "B", 16)
        pdf.set_text_color(*C_PRIMARY)
        pdf.cell(0, 10, f"{name}({code})深度分析")
        pdf.set_xy(15, 20)
        pdf.set_font(pdf._font_name, "", 9)
        pdf.set_text_color(*C_GRAY)
        pdf.cell(0, 5, f"{industry}行业")
        pdf.ln(8)

        # 综合评价
        summary = ca.get("summary", "")
        if summary:
            pdf.set_font(pdf._font_name, "B", 10)
            pdf.set_text_color(*C_BLACK)
            pdf.cell(0, 7, f"综合评价: {summary}")
            pdf.ln(10)

        # 收入利润趋势
        pdf.section_title("收入与利润趋势")
        rev = ca.get("revenue", {})
        prof = ca.get("profit", {})

        rev_trend = rev.get("trend", [])
        prof_trend = prof.get("trend", [])

        if rev_trend or prof_trend:
            n_periods = len(rev_trend[:4]) if rev_trend else 1
            trend_headers = ["指标"] + [
                t["period"][:10] for t in rev_trend[:4]]
            trend_rows = []
            if rev_trend:
                trend_rows.append(
                    ["营收(亿)"] +
                    [f"{t['revenue']:.1f}" for t in rev_trend[:4]])
            if prof_trend:
                trend_rows.append(
                    ["净利润(亿)"] +
                    [f"{t['net_profit']:.1f}" for t in prof_trend[:4]])
            trend_rows.append([
                "增速判断", rev.get("stability", "-")])
            col_w = [30] + [36] * n_periods
            pdf.draw_table(15, pdf.get_y(),
                           trend_headers, trend_rows, col_w)
            pdf.set_y(pdf.get_y() + 5)

        # 现金流
        pdf.ln(4)
        pdf.section_title("现金流分析")
        cf = ca.get("cashflow", {})
        cf_items = [
            f"经营现金流: {cf.get('operating_cf', 'N/A')}亿",
            f"投资现金流: {cf.get('investing_cf', 'N/A')}亿",
            f"自由现金流: {cf.get('fcf', 'N/A')}亿",
            f"状态: {cf.get('status', 'N/A')}",
        ]
        cf_color = C_LIGHT_GREEN if "健康" in cf.get("status", "") else \
            C_LIGHT_YELLOW if "一般" in cf.get("status", "") else C_LIGHT_RED
        pdf.draw_card(15, pdf.get_y(), 180, 32, "现金流", cf_items,
                      bg_color=cf_color)
        pdf.set_y(pdf.get_y() + 35)

        # 资产负债关键指标
        bal = ca.get("balance", {})
        if bal:
            pdf.section_title("资产负债关键指标")
            bal_rows = []
            if "goodwill_ratio" in bal:
                bal_rows.append(["商誉/净资产", f"{bal['goodwill_ratio']:.1f}%"])
            if "accounts_rec_ratio" in bal:
                bal_rows.append(["应收/总资产", f"{bal['accounts_rec_ratio']:.1f}%"])
            if "cash_ratio" in bal:
                bal_rows.append(["现金/总资产", f"{bal['cash_ratio']:.1f}%"])
            if "inventory_ratio" in bal:
                bal_rows.append(["存货/总资产", f"{bal['inventory_ratio']:.1f}%"])
            if bal_rows:
                pdf.draw_table(15, pdf.get_y(),
                               ["指标", "数值"],
                               bal_rows, [60, 60])
                pdf.set_y(pdf.get_y() + 5)

        # 风险清单
        risks = ca.get("risks", [])
        if risks:
            pdf.section_title("风险提示")
            for risk in risks:
                pdf.draw_risk_tag(15, pdf.get_y(),
                                  risk["level"], risk["desc"])
                pdf.set_y(pdf.get_y() + 9)

        # 亮点
        highlights = ca.get("highlights", [])
        if highlights:
            pdf.ln(3)
            pdf.section_title("亮点")
            hl_items = [f"* {h}" for h in highlights]
            card_h = 8 + len(hl_items) * 6
            pdf.draw_card(15, pdf.get_y(), 180, card_h,
                          "亮点", hl_items, bg_color=C_LIGHT_GREEN)
            pdf.set_y(pdf.get_y() + card_h + 3)

        # 业绩预告
        forecast = ca.get("forecast", "")
        if forecast:
            pdf.ln(5)
            pdf.set_font(pdf._font_name, "", 8)
            pdf.set_text_color(*C_BLUE)
            pdf.cell(0, 5, forecast)
            pdf.ln(3)

        pdf.draw_footer()

    # ═══════════════════════════════════════════════════════════════
    # 末页 — 漏斗概览 + 免责声明
    # ═══════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.draw_top_bar()
    pdf.section_title("筛选漏斗概览")

    funnel_rows = [
        ["全A股", "-", "~5000"],
        ["Layer 1 基础过滤",
         "排除ST/新股/停牌/北交所/市值<" + str(meta.get("min_mcap", 20)) + "亿",
         str(meta.get("candidates_count", 0))],
        ["Layer 2 因子评分",
         f"{meta.get('template_name', '自定义')}策略，综合分>={meta.get('min_score', 70)}",
         str(meta.get("results_count", 0))],
        ["最终输出", f"TOP {len(results)}", str(len(results))],
    ]
    pdf.draw_table(15, pdf.get_y(),
                   ["阶段", "条件", "数量"],
                   funnel_rows, [45, 90, 30])

    pdf.ln(8)
    pdf.set_font(pdf._font_name, "", 8)
    pdf.set_text_color(*C_DARK_GRAY)
    pdf.set_xy(15, pdf.get_y())
    pdf.multi_cell(180, 5,
                   "免责声明: 本报告由选股子系统基于公开财务数据和量化模型"
                   "自动生成，仅供参考学习使用，不构成任何投资建议。"
                   "投资有风险，入市需谨慎。数据来源: Tushare Pro。")

    pdf.output(output_path)
    return output_path
