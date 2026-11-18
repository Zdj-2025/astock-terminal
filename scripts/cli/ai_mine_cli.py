#!/usr/bin/env python3
"""AI概念股挖掘CLI — 从AKShare概念板块发现、筛选、评分AI供应链优质标的。

用法:
    python scripts/cli/ai_mine_cli.py                    # 完整挖掘
    python scripts/cli/ai_mine_cli.py --quick             # 快速模式（仅行情数据）
    python scripts/cli/ai_mine_cli.py --sector optical    # 仅光模块板块
    python scripts/cli/ai_mine_cli.py --top 30            # 前30名
    python scripts/cli/ai_mine_cli.py --output report.json
"""
import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.analyzers.ai_stock_miner import AIStockMiner


def main():
    parser = argparse.ArgumentParser(description='AI概念股挖掘')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式（仅行情数据，不获取财务指标）')
    parser.add_argument('--sector', type=str, default=None,
                        help='指定板块: ai_chips/optical/servers/cloud/ai_app/data')
    parser.add_argument('--top', type=int, default=20,
                        help='每个类别展示前N只 (默认20)')
    parser.add_argument('--pe-max', type=float, default=None)
    parser.add_argument('--roe-min', type=float, default=None)
    parser.add_argument('--revenue-min', type=float, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件路径')
    args = parser.parse_args()

    overrides = {}
    if args.pe_max is not None:
        overrides["value_pe_max"] = args.pe_max
    if args.roe_min is not None:
        overrides["value_roe_min"] = args.roe_min
    if args.revenue_min is not None:
        overrides["value_revenue_growth_min"] = args.revenue_min

    miner = AIStockMiner(config=overrides if overrides else None)
    results = miner.run(quick_mode=args.quick, sector_filter=args.sector)

    print(miner.format_results(results, top_n=args.top))

    if args.output:
        filepath = args.output
        if not os.path.isabs(filepath):
            filepath = os.path.join(project_root, filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已保存: {filepath}")


if __name__ == '__main__':
    main()
