#!/usr/bin/env python3
"""缠论分析CLI — 从AKShare获取数据执行缠论分析。

用法:
    python scripts/cli/chanlun_cli.py 603881
    python scripts/cli/chanlun_cli.py 603881 --levels daily,60min
    python scripts/cli/chanlun_cli.py 603881 --name 数据港 --begin 2025-06-01
"""
import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts', 'analyzers', 'chanlun', 'vendor'))

import akshare as ak
import pandas as pd

from scripts.analyzers.chanlun.engine import ChanLunEngine
from scripts.analyzers.chanlun.analyzer import generate_chanlun_report


_MIN_PERIODS = {"1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60"}


def fetch_kline(symbol: str, period: str = "daily", begin: str = "20240101") -> pd.DataFrame:
    """获取K线数据"""
    adjust = "qfq"
    if period in ("daily", "weekly", "monthly"):
        df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=begin.replace("-", ""), end_date="20991231", adjust=adjust)
    elif period in _MIN_PERIODS:
        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=_MIN_PERIODS[period], start_date=f"{begin} 09:30:00", end_date="2099-12-31 15:00:00", adjust=adjust)
    else:
        raise ValueError(f"不支持的周期: {period}")

    # 统一列名
    col_map = {'日期': 'date', '时间': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df['date'] = pd.to_datetime(df['date'])
    return df


def main():
    parser = argparse.ArgumentParser(description='缠论分析')
    parser.add_argument('symbol', help='股票代码 (如 603881)')
    parser.add_argument('--name', help='股票名称')
    parser.add_argument('--levels', default='daily', help='分析级别，逗号分隔 (daily,60min,weekly)')
    parser.add_argument('--begin', default='2025-01-01', help='起始日期')
    parser.add_argument('--report', action='store_true', help='生成HTML可视化报告')
    args = parser.parse_args()

    symbol = args.symbol.strip()
    name = args.name or symbol
    levels = [l.strip() for l in args.levels.split(',')]

    print(f"=== {name}({symbol}) 缠论分析 ===")
    print(f"级别: {levels}")

    engine = ChanLunEngine()
    result = engine.analyze(symbol, levels=levels, begin_date=args.begin)

    # 生成文字报告
    report = generate_chanlun_report(result, stock_name=name)
    print(report)

    if args.report:
        # 生成HTML报告（带图表）
        from scripts.analyzers.chart_chanlun import create_chanlun_chart

        html_parts = []
        for lr in result.levels:
            period = lr.level  # "daily", "5min", "15min", "30min", "60min", "weekly"
            try:
                df = fetch_kline(symbol, period=period, begin=args.begin)
                chart_html = create_chanlun_chart(df, lr, title=f"{name} 缠论分析")
                html_parts.append(chart_html)
                print(f"  {lr.level} 图表生成完成")
            except Exception as e:
                html_parts.append(f"<p>{lr.level} 图表生成失败: {e}</p>")
                print(f"  {lr.level} 图表生成失败: {e}")

        if html_parts:
            report_dir = os.path.join(project_root, 'reports', 'html')
            os.makedirs(report_dir, exist_ok=True)
            filepath = os.path.join(report_dir, f"{name}{symbol}_缠论分析.html")
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{name} 缠论分析</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#f5f5f5}}</style>
</head><body>
<h1>{name}({symbol}) 缠论分析</h1>
<pre style="background:white;padding:20px;border-radius:8px">{report}</pre>
{''.join(html_parts)}
</body></html>"""
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"\nHTML报告已生成: {filepath}")


if __name__ == '__main__':
    main()
