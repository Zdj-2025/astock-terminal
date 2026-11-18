#!/usr/bin/env python3
"""选股子系统 CLI 入口。

用法:
  # 使用预设模板
  python scripts/screener/screener_cli.py --template growth
  python scripts/screener/screener_cli.py --template davis_double

  # 指定概念板块
  python scripts/screener/screener_cli.py --concept AI芯片
  python scripts/screener/screener_cli.py --concept 光通信 --template value

  # 列出可用概念板块/模板
  python scripts/screener/screener_cli.py --list-concepts
  python scripts/screener/screener_cli.py --list-concepts AI
  python scripts/screener/screener_cli.py --list-templates

  # 自定义参数（覆盖模板默认值）
  python scripts/screener/screener_cli.py --concept 锂电池 --rev-growth-min 30 --top 15

  # 快速模式（跳过财务数据，只用daily_basic）
  python scripts/screener/screener_cli.py --template momentum --quick
"""
import argparse
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.screener.screener import StockScreener, SCREENER_TEMPLATES

warnings.filterwarnings('ignore')


def list_templates():
    """列出所有预设模板"""
    print(f"\n{'='*50}")
    print("  可用筛选模板")
    print(f"{'='*50}\n")
    for key, tpl in SCREENER_TEMPLATES.items():
        weights = tpl["factor_weights"]
        print(f"  {key:<16} {tpl['name']:<10} {tpl['desc']}")
        print(f"  {'':>16} 权重: "
              f"成长{weights['growth']:.0%} 质量{weights['quality']:.0%} "
              f"价值{weights['value']:.0%} 动量{weights['momentum']:.0%} "
              f"主线{weights['theme']:.0%}")
        filters = tpl.get("filters", {})
        if filters:
            filter_str = " | ".join(f"{k}={v}" for k, v in filters.items())
            print(f"  {'':>16} 过滤: {filter_str}")
        print()


def list_concepts(screener: StockScreener, keyword: str = None):
    """列出概念板块"""
    concepts = screener.list_concepts(keyword)
    if concepts.empty:
        print("  未找到匹配的概念板块")
        return
    print(f"\n  共 {len(concepts)} 个概念板块")
    print(f"{'─'*50}")
    for _, row in concepts.head(50).iterrows():
        print(f"  {row['code']:<12} {row['name']}")
    if len(concepts) > 50:
        print(f"  ... 还有 {len(concepts) - 50} 个，请用关键词缩小范围")


def main():
    parser = argparse.ArgumentParser(
        description='全市场选股子系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/screener/screener_cli.py --template growth
  python scripts/screener/screener_cli.py --concept AI芯片 --top 10
  python scripts/screener/screener_cli.py --template davis_double --quick
  python scripts/screener/screener_cli.py --concept 锂电池 --rev-growth-min 30
        """,
    )

    # 查询类
    parser.add_argument('--list-templates', action='store_true',
                        help='列出可用筛选模板')
    parser.add_argument('--list-concepts', nargs='?', const='', default=None,
                        help='列出概念板块（可选关键词过滤）')

    # 筛选参数
    parser.add_argument('--template', '-t', default=None,
                        choices=list(SCREENER_TEMPLATES.keys()),
                        help='预设模板')
    parser.add_argument('--concept', '-c', default=None,
                        help='概念板块关键词（如 AI芯片、光通信）')
    parser.add_argument('--industry', '-i', default=None,
                        help='行业名称')

    # 筛选条件覆盖
    parser.add_argument('--min-score', type=float, default=70,
                        help='最低综合分 (默认70)')
    parser.add_argument('--top', type=int, default=20,
                        help='返回前N只 (默认20)')
    parser.add_argument('--min-mcap', type=float, default=20,
                        help='最小市值/亿元 (默认20)')
    parser.add_argument('--max-mcap', type=float, default=None,
                        help='最大市值/亿元')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='快速模式（跳过逐只财务查询）')
    parser.add_argument('--feishu', '-f', action='store_true',
                        help='发送报告摘要到飞书群')
    parser.add_argument('--pdf', '-p', action='store_true',
                        help='生成PDF报告（含企业深度分析）')

    # 模板参数覆盖
    parser.add_argument('--rev-growth-min', type=float, default=None,
                        help='营收增速最低%% (覆盖模板)')
    parser.add_argument('--profit-growth-min', type=float, default=None,
                        help='利润增速最低%%')
    parser.add_argument('--profit-growth-max', type=float, default=None,
                        help='利润增速最高%%')
    parser.add_argument('--roe-min', type=float, default=None,
                        help='ROE最低%%')
    parser.add_argument('--pe-max', type=float, default=None,
                        help='PE最高')
    parser.add_argument('--pb-max', type=float, default=None,
                        help='PB最高')
    parser.add_argument('--debt-max', type=float, default=None,
                        help='负债率最高%%')

    # W3浪专用参数
    parser.add_argument('--min-w3-gain', type=float, default=10,
                        help='W3最低涨幅%% (wave3模板专用，默认10)')
    parser.add_argument('--min-confidence', type=float, default=50,
                        help='W3最低置信度 (wave3/wave3_startup模板专用，默认50)')

    # W3刚启动专用参数
    parser.add_argument('--verify-chanlun', action='store_true',
                        help='对W3刚启动候选做缠论深度验证 (wave3_startup专用)')

    args = parser.parse_args()

    # 列表类命令
    if args.list_templates:
        list_templates()
        return

    screener = StockScreener()

    if args.list_concepts is not None:
        list_concepts(screener, args.list_concepts or None)
        return

    # 构建覆盖过滤条件
    override_filters = {}
    if args.rev_growth_min is not None:
        override_filters["rev_growth_min"] = args.rev_growth_min
    if args.profit_growth_min is not None:
        override_filters["profit_growth_min"] = args.profit_growth_min
    if args.profit_growth_max is not None:
        override_filters["profit_growth_max"] = args.profit_growth_max
    if args.roe_min is not None:
        override_filters["roe_min"] = args.roe_min
    if args.pe_max is not None:
        override_filters["pe_max"] = args.pe_max
    if args.pb_max is not None:
        override_filters["pb_max"] = args.pb_max
    if args.debt_max is not None:
        override_filters["debt_max"] = args.debt_max

    # 执行筛选
    result = screener.run(
        template=args.template,
        concept=args.concept,
        industry=args.industry,
        min_mcap=args.min_mcap,
        max_mcap=args.max_mcap,
        min_score=args.min_score,
        top=args.top,
        quick=args.quick,
        override_filters=override_filters or None,
        min_w3_gain=args.min_w3_gain,
        min_w3_confidence=args.min_confidence,
    )

    meta = result["meta"]

    if "error" in meta:
        print(f"\n  错误: {meta['error']}\n")
        return

    # 控制台报告
    if args.template == "wave_position" and not result["results"].empty:
        # wave_position用专用格式化，通用format_report缺太多列
        from scripts.screener.wave_position_screener import WavePositionScreener
        wp_fmt = WavePositionScreener()
        print(wp_fmt.format_report(result["results"]))
    else:
        report = screener.format_report(result["results"], meta)
        print(report)

    # Phase 2: 缠论深度验证（wave3_startup专用）
    if args.verify_chanlun and args.template == "wave3_startup" and not result["results"].empty:
        from scripts.screener.wave3_startup_screener import Wave3StartupScreener
        startup_scanner = Wave3StartupScreener(screener.client)
        result["results"] = startup_scanner.verify_with_chanlun(result["results"])
        confirmed = result["results"]['w3s_confirmed_ermai'].sum()
        print(f"\n  缠论验证: {confirmed}/{len(result['results'])}只确认二买\n")

    # 保存文件
    if not result["results"].empty and args.template != "wave_position":
        path = screener.save_report(result["results"], meta)
        print(f"  报告已保存: {path}\n")

        # PDF报告
        if args.pdf:
            deep_analyses = screener.run_deep_analysis(result["results"], top_n=3)
            pdf_path = screener.save_pdf_report(result["results"], meta, deep_analyses)
            print(f"  PDF报告已保存: {pdf_path}\n")

        # 飞书推送
        if args.feishu:
            screener.send_to_feishu(result["results"], meta, path)
            msg = screener._feishu_msg
            print(f"  飞书消息已准备: 群 {msg['chat_id']}")
            # 输出JSON供MCP发送
            import json
            feishu_file = "/tmp/screener_feishu_msg.json"
            with open(feishu_file, "w") as f:
                json.dump(msg, f, ensure_ascii=False, indent=2)

            # 导入MD全文为飞书文档
            if hasattr(screener, '_feishu_md_content') and screener._feishu_md_content:
                doc_file = "/tmp/screener_feishu_doc.json"
                with open(doc_file, "w") as f:
                    json.dump({
                        "title": screener._feishu_doc_title,
                        "content": screener._feishu_md_content,
                    }, f, ensure_ascii=False)
                print(f"  飞书文档内容已准备: {doc_file}")
            print()


if __name__ == '__main__':
    main()
