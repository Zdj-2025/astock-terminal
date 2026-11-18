#!/usr/bin/env python3
"""公告与新闻分析器 — 从AKShare获取股票公告、新闻、研报等信息"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import akshare as ak
import pandas as pd


# 公告重要性分级关键词
_A_KEYWORDS = ['业绩预告', '业绩快报', '退市', 'ST', '重组', '收购', '停牌', '复牌',
               '风险提示', '质押', '减持', '增持', '重大合同', '重大事项']
_B_KEYWORDS = ['年报', '一季报', '半年报', '三季报', '分红', '送转', '除权',
               '合同', '调研', '股东大会', '激励', '解禁']
# 不在A/B中的为C级


def _classify_announcement(title: str) -> str:
    """根据标题关键词分级"""
    for kw in _A_KEYWORDS:
        if kw in title:
            return 'A'
    for kw in _B_KEYWORDS:
        if kw in title:
            return 'B'
    return 'C'


class NewsAnalyzer:
    """股票公告与新闻查询"""

    def get_announcements(
        self, symbol: str, days: int = 30, category: str = ""
    ) -> pd.DataFrame:
        """
        获取巨潮资讯公告

        Parameters:
            symbol: 股票代码如 "603881"
            days: 往前查多少天
            category: 公告类别筛选（如 "业绩预告", "年报"），空字符串=全部
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=symbol, market="沪深京",
            category=category,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if df.empty:
            return df
        df['级别'] = df['公告标题'].apply(_classify_announcement)
        df = df.sort_values('级别').reset_index(drop=True)
        return df

    def get_news(self, symbol: str) -> pd.DataFrame:
        """获取东方财富个股新闻（约10条最新）"""
        try:
            return ak.stock_news_em(symbol=symbol)
        except Exception:
            return pd.DataFrame()

    def get_research_reports(self, symbol: str) -> pd.DataFrame:
        """获取个股研报+评级"""
        try:
            df = ak.stock_research_report_em(symbol=symbol)
            if df.empty:
                return df
            cols = ['股票简称', '报告名称', '东财评级', '机构', '日期',
                    '近一月个股研报数']
            profit_cols = [c for c in df.columns if '盈利预测' in c]
            cols += profit_cols
            cols = [c for c in cols if c in df.columns]
            return df[cols].head(20)
        except Exception:
            return pd.DataFrame()

    def get_dividend_history(self, symbol: str) -> pd.DataFrame:
        """获取分红送配历史"""
        try:
            return ak.stock_fhps_detail_em(symbol=symbol)
        except Exception:
            return pd.DataFrame()

    def get_pledge_info(self, symbol: str) -> pd.DataFrame:
        """获取个股股权质押详情"""
        try:
            return ak.stock_gpzy_individual_pledge_ratio_detail_em(symbol=symbol)
        except Exception:
            return pd.DataFrame()

    def get_full_briefing(self, symbol: str, days: int = 30) -> dict:
        """
        综合信息简报

        Returns:
            {
                'symbol': '603881',
                'announcements': DataFrame,
                'a_level_count': 3,
                'news': DataFrame,
                'research': DataFrame,
                'research_rating': '增持',
                'dividend': DataFrame,
                'pledge': DataFrame or None,
            }
        """
        result = {'symbol': symbol}

        # 公告
        ann = self.get_announcements(symbol, days=days)
        result['announcements'] = ann
        result['a_level_count'] = len(ann[ann['级别'] == 'A']) if not ann.empty else 0

        # 新闻
        result['news'] = self.get_news(symbol)

        # 研报
        research = self.get_research_reports(symbol)
        result['research'] = research
        if not research.empty and '东财评级' in research.columns:
            ratings = research['东财评级'].value_counts()
            result['research_rating'] = ratings.index[0] if len(ratings) > 0 else None
            result['research_count'] = len(research)
        else:
            result['research_rating'] = None
            result['research_count'] = 0

        # 分红
        result['dividend'] = self.get_dividend_history(symbol)

        # 质押
        pledge = self.get_pledge_info(symbol)
        result['pledge'] = pledge if not pledge.empty else None

        return result


def format_briefing(briefing: dict, name: str = "") -> str:
    """将综合简报格式化为可读文本"""
    lines = []
    sym = briefing['symbol']
    n = name or sym
    lines.append(f"# {n}({sym}) 信息简报")
    lines.append("")

    # A级公告
    ann = briefing.get('announcements')
    if ann is not None and not ann.empty:
        a_level = ann[ann['级别'] == 'A']
        lines.append(f"## 重要公告 (A级: {len(a_level)}条)")
        if a_level.empty:
            lines.append("  近期无A级重要公告")
        else:
            for _, row in a_level.head(10).iterrows():
                lines.append(f"  [{row['公告时间']}] {row['公告标题']}")
        lines.append("")

        # B级摘要
        b_level = ann[ann['级别'] == 'B']
        if not b_level.empty:
            lines.append(f"## 一般公告 (B级: {len(b_level)}条)")
            for _, row in b_level.head(5).iterrows():
                lines.append(f"  [{row['公告时间']}] {row['公告标题']}")
            lines.append("")

    # 新闻
    news = briefing.get('news')
    if news is not None and not news.empty:
        lines.append(f"## 最新新闻 ({len(news)}条)")
        for _, row in news.head(5).iterrows():
            lines.append(f"  [{row['发布时间']}] {row['新闻标题'][:60]}")
            lines.append(f"    来源: {row['文章来源']}")
        lines.append("")

    # 研报
    research = briefing.get('research')
    if research is not None and not research.empty:
        rating = briefing.get('research_rating', '')
        count = briefing.get('research_count', 0)
        lines.append(f"## 研报 ({count}份, 主流评级: {rating or '无'})")
        for _, row in research.head(5).iterrows():
            line = f"  [{row['日期']}] {row['机构']} - {row['报告名称']}"
            if '东财评级' in row:
                line += f" [{row['东财评级']}]"
            lines.append(line)
        lines.append("")

    # 分红
    dividend = briefing.get('dividend')
    if dividend is not None and not dividend.empty:
        lines.append("## 分红送配 (最近3次)")
        for _, row in dividend.head(3).iterrows():
            desc = []
            for col in dividend.columns:
                if '现金' in col and '比例' in col and pd.notna(row[col]) and row[col] != 0:
                    try:
                        desc.append(f"每10股派{float(row[col]):.2f}元")
                    except (ValueError, TypeError):
                        desc.append(f"每10股派{row[col]}元")
                if '送转' in col and '总比例' in col and pd.notna(row[col]) and row[col] != 0:
                    try:
                        desc.append(f"每10股送转{float(row[col]):.1f}股")
                    except (ValueError, TypeError):
                        desc.append(f"每10股送转{row[col]}股")
            if desc:
                lines.append(f"  {row.get('股权登记日', row.get('除权除息日', ''))} {' '.join(desc)}")
        lines.append("")

    # 质押
    pledge = briefing.get('pledge')
    if pledge is not None and not pledge.empty:
        lines.append("## 股权质押")
        for _, row in pledge.head(3).iterrows():
            lines.append(f"  {row.to_dict()}")
        lines.append("")
    else:
        lines.append("## 股权质押: 无数据或无质押")
        lines.append("")

    return "\n".join(lines)
