#!/usr/bin/env python3
"""公告新闻查询CLI — 查看股票公告、新闻、研报等信息。

用法:
    python scripts/cli/news_cli.py 603881                       # 综合简报
    python scripts/cli/news_cli.py 603881 --name 数据港
    python scripts/cli/news_cli.py 603881 --announcements        # 仅公告
    python scripts/cli/news_cli.py 603881 --news                 # 仅新闻
    python scripts/cli/news_cli.py 603881 --research             # 仅研报
    python scripts/cli/news_cli.py 603881 --dividend             # 分红送配
    python scripts/cli/news_cli.py 603881 --pledge               # 股权质押
    python scripts/cli/news_cli.py 603881 --days 60              # 最近60天
    python scripts/cli/news_cli.py 603881 --announcements --category 业绩预告
"""
import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.analyzers.news_analyzer import NewsAnalyzer, format_briefing


def main():
    parser = argparse.ArgumentParser(description='股票公告新闻查询')
    parser.add_argument('symbol', help='股票代码 (如 603881)')
    parser.add_argument('--name', help='股票名称')
    parser.add_argument('--days', type=int, default=30, help='查询天数 (默认30)')
    parser.add_argument('--announcements', action='store_true', help='仅查看公告')
    parser.add_argument('--category', default='', help='公告类别筛选 (如 业绩预告, 年报)')
    parser.add_argument('--news', action='store_true', help='仅查看新闻')
    parser.add_argument('--research', action='store_true', help='仅查看研报')
    parser.add_argument('--dividend', action='store_true', help='仅查看分红送配')
    parser.add_argument('--pledge', action='store_true', help='仅查看股权质押')
    args = parser.parse_args()

    symbol = args.symbol.strip()
    name = args.name or symbol
    analyzer = NewsAnalyzer()

    # 单项查询模式
    if args.announcements:
        print(f"=== {name}({symbol}) 公告 (最近{args.days}天) ===\n")
        df = analyzer.get_announcements(symbol, days=args.days, category=args.category)
        if df.empty:
            print("  无数据")
        else:
            for _, row in df.iterrows():
                level_tag = {'A': '🔴', 'B': '🟡', 'C': '⚪'}.get(row['级别'], '  ')
                print(f"  {level_tag}[{row['级别']}][{row['公告时间']}] {row['公告标题']}")
        return

    if args.news:
        print(f"=== {name}({symbol}) 最新新闻 ===\n")
        df = analyzer.get_news(symbol)
        if df.empty:
            print("  无数据")
        else:
            for _, row in df.iterrows():
                print(f"  [{row['发布时间']}] {row['新闻标题']}")
                print(f"    来源: {row['文章来源']}")
        return

    if args.research:
        print(f"=== {name}({symbol}) 研报 ===\n")
        df = analyzer.get_research_reports(symbol)
        if df.empty:
            print("  无数据")
        else:
            for _, row in df.iterrows():
                line = f"  [{row['日期']}] {row['机构']} - {row['报告名称']}"
                if '东财评级' in row and pd.notna(row['东财评级']):
                    line += f" [{row['东财评级']}]"
                print(line)
        return

    if args.dividend:
        print(f"=== {name}({symbol}) 分红送配 ===\n")
        df = analyzer.get_dividend_history(symbol)
        if df.empty:
            print("  无数据")
        else:
            print(df.to_string(index=False))
        return

    if args.pledge:
        print(f"=== {name}({symbol}) 股权质押 ===\n")
        df = analyzer.get_pledge_info(symbol)
        if df.empty:
            print("  无质押或无数据")
        else:
            print(df.to_string(index=False))
        return

    # 综合简报模式（默认）
    print(f"=== {name}({symbol}) 信息简报 ===\n")
    briefing = analyzer.get_full_briefing(symbol, days=args.days)
    print(format_briefing(briefing, name=name))


if __name__ == '__main__':
    import pandas as pd
    main()
