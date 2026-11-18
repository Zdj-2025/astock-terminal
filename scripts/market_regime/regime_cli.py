#!/usr/bin/env python3
"""大盘环境检测 CLI。

用法:
    python scripts/market_regime/regime_cli.py
    python scripts/market_regime/regime_cli.py --json
    python scripts/market_regime/regime_cli.py --as-of 20250101
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from scripts.market_regime.regime_detector import MarketRegimeDetector


def main():
    parser = argparse.ArgumentParser(description="大盘环境检测")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")
    parser.add_argument("--as-of", dest="as_of", help="参考日期YYYYMMDD")
    args = parser.parse_args()

    detector = MarketRegimeDetector()
    result = detector.detect(as_of_date=args.as_of)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(MarketRegimeDetector.format_report(result))


if __name__ == "__main__":
    main()
