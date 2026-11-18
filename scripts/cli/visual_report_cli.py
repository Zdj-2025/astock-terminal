#!/usr/bin/env python3
"""可视化报告生成CLI — 从AKShare获取数据生成带图表的HTML报告。

用法:
    python scripts/cli/visual_report_cli.py 603881
    python scripts/cli/visual_report_cli.py 300059 --name 东方财富 --industry 证券
"""
import argparse
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import akshare as ak
import pandas as pd
import numpy as np

from scripts.analyzers.report_builder import ReportBuilder


def fetch_kline(symbol: str) -> pd.DataFrame:
    """获取日K线数据"""
    prefix = 'sh' if symbol.startswith('6') else 'sz'
    df = ak.stock_zh_a_daily(
        symbol=f"{prefix}{symbol}",
        start_date="20250101", adjust="qfq",
    )
    df.columns = [c.lower() for c in df.columns]
    if 'date' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_indicators(df: pd.DataFrame) -> dict:
    """计算MA/MACD(双参数)/RSI"""
    ma = {p: df['close'].rolling(p).mean() for p in [5, 10, 20, 60]}

    # MACD 标准 12,26,9
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd = {'dif': dif, 'dea': dea, 'hist': 2 * (dif - dea)}

    # MACD 快速 5,34,5 (背离检测+短线做T)
    ema5 = df['close'].ewm(span=5).mean()
    ema34 = df['close'].ewm(span=34).mean()
    dif_f = ema5 - ema34
    dea_f = dif_f.ewm(span=5).mean()
    macd_fast = {'dif': dif_f, 'dea': dea_f, 'hist': 2 * (dif_f - dea_f)}

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    return {'ma': ma, 'macd': macd, 'macd_fast': macd_fast, 'rsi': rsi}


def main():
    parser = argparse.ArgumentParser(description='生成可视化HTML分析报告')
    parser.add_argument('symbol', help='股票代码 (如 603881, 300059)')
    parser.add_argument('--name', help='股票名称 (如 数据港)')
    parser.add_argument('--industry', help='行业 (如 数据中心)')
    args = parser.parse_args()

    symbol = args.symbol.strip()
    name = args.name or symbol

    print(f"获取 {name}({symbol}) 数据...")
    df = fetch_kline(symbol)
    if df.empty:
        print(f"错误：未获取到 {symbol} 的K线数据")
        sys.exit(1)

    indicators = compute_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    print(f"  最新收盘: {latest['close']:.2f}  数据量: {len(df)}行")

    data = {
        'stock_code': symbol,
        'stock_name': name,
        'industry': args.industry or '',
        'current_price': round(latest['close'], 2),
        'daily_change': round((latest['close'] / prev['close'] - 1) * 100, 2),
        'report_date': str(latest['date'].date()),
        'kline': {
            'df': df,
            'ma': indicators['ma'],
            'macd': indicators['macd'],
            'macd_fast': indicators['macd_fast'],
            'rsi': indicators['rsi'],
        },
        'key_levels': _extract_levels(indicators, latest['close']),
    }

    builder = ReportBuilder()
    filepath = builder.build_stock_report(data)
    print(f"\n报告已生成: {filepath}")


def _extract_levels(indicators: dict, price: float) -> list:
    """从MA和布林带提取关键价位"""
    levels = []
    ma = indicators['ma']
    for p, series in ma.items():
        if len(series) > 0 and not np.isnan(series.iloc[-1]):
            val = round(series.iloc[-1], 2)
            dist = (val / price - 1) * 100
            role = '支撑' if val < price else '压力'
            strength = '强' if p in [60, 120] else '中'
            levels.append({'role': role, 'value': val, 'source': f'MA{p}', 'strength': strength})

    levels.sort(key=lambda x: abs(x['value'] - price))
    return levels[:6]


if __name__ == '__main__':
    main()
