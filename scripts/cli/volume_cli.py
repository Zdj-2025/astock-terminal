#!/usr/bin/env python3
"""成交量分析CLI"""
import argparse
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.analyzers.volume_analyzer import VolumeAnalyzer


def main():
    parser = argparse.ArgumentParser(description='成交量分析')
    parser.add_argument('symbol', help='股票代码（如 603881）')
    parser.add_argument('--timeframes', '-t', default='daily,60min',
                        help='时间级别 (daily,60min,15min,weekly)，逗号分隔')
    parser.add_argument('--json', action='store_true', help='JSON输出')
    args = parser.parse_args()

    tfs = [t.strip() for t in args.timeframes.split(',')]
    analyzer = VolumeAnalyzer(symbol=args.symbol)
    result = analyzer.analyze(timeframes=tfs)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(VolumeAnalyzer.format_report(result))


if __name__ == '__main__':
    main()
