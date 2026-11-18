#!/usr/bin/env python3
"""综合技术分析CLI — 两步六维度框架。

用法:
    python scripts/cli/tech_analysis_cli.py 603881
    python scripts/cli/tech_analysis_cli.py 300059 --with-15min
    python scripts/cli/tech_analysis_cli.py 512480 --json
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.analyzers.comprehensive_analyzer import ComprehensiveAnalyzer
from scripts.analyzers.discipline_checker import DisciplineChecker


def save_html_report(symbol: str, report_text: str, result: dict):
    """保存分析报告为HTML文件到 reports/html/"""
    try:
        import markdown as md_lib
    except ImportError:
        return
    from scripts.core.tushare_client import TushareClient

    ts = TushareClient()
    # 优先从实时行情获取名称，fallback到stock_info
    rt = result.get('realtime', {})
    name = rt.get('name', '')
    if not name:
        info = ts.get_stock_info(symbol) or {}
        name = info.get('name', symbol)
    today = datetime.now().strftime('%Y%m%d')

    out_dir = os.path.join(project_root, 'reports', 'html')
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{name}{symbol}_分析报告.html"
    filepath = os.path.join(out_dir, filename)

    body = md_lib.markdown(report_text, extensions=['tables', 'fenced_code'])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name}({symbol}) 综合分析报告 {today}</title>
<style>
body {{ font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
       max-width: 900px; margin: 0 auto; padding: 20px;
       background: #1a1a2e; color: #e0e0e0; line-height: 1.7; }}
h1 {{ color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 8px; }}
h2 {{ color: #4ecdc4; margin-top: 1.5em; }}
h3 {{ color: #43S7d1; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
th {{ background: #2a2a4a; color: #00d4aa; }}
td {{ background: #1e1e38; }}
code {{ background: #2a2a4a; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
pre {{ background: #12122a; padding: 12px; border-radius: 6px; overflow-x: auto; }}
hr {{ border: none; border-top: 1px solid #333; margin: 1.5em 0; }}
blockquote {{ border-left: 3px solid #00d4aa; margin: 1em 0; padding: 0.5em 1em;
             background: #1e1e38; }}
strong {{ color: #fff; }}
.meta {{ color: #888; font-size: 0.85em; margin-bottom: 1em; }}
</style>
</head>
<body>
<h1>{name}({symbol}) 综合分析报告</h1>
<p class="meta">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
{body}
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n📄 报告已保存: {filepath}")


def load_holding(symbol):
    """从portfolio.xlsx加载指定股票的持仓数据"""
    import openpyxl
    portfolio_path = os.path.join(os.path.dirname(__file__), '../../config/portfolio.xlsx')
    if not os.path.exists(portfolio_path):
        return None
    try:
        wb = openpyxl.load_workbook(portfolio_path)
        ws = wb.active
        headers = [cell.value for cell in ws[1]]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] is None:
                continue
            h = dict(zip(headers, row))
            code = str(h.get('股票代码', '')).strip()
            if code == symbol:
                wb.close()
                return {
                    'name': str(h.get('股票名称', '')).strip(),
                    'cost': float(h.get('买入价格', 0) or 0),
                    'shares': int(h.get('持仓数量', 0) or 0),
                    'buy_date': h.get('买入日期', ''),
                    'stop_loss': float(h.get('止损价', 0) or 0),
                    'take_profit': float(h.get('止盈价', 0) or 0),
                }
        wb.close()
    except Exception:
        pass
    return None


def _load_all_holdings():
    """加载全量持仓和资金数据（用于纪律检查的总仓位计算）"""
    import openpyxl
    holdings = []
    capital = {'total': 0, 'cash': 0}

    portfolio_path = os.path.join(os.path.dirname(__file__), '../../config/portfolio.xlsx')
    if os.path.exists(portfolio_path):
        try:
            wb = openpyxl.load_workbook(portfolio_path)
            ws = wb.active
            headers = [cell.value for cell in ws[1]]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] is None:
                    continue
                h = dict(zip(headers, row))
                code = str(h.get('股票代码', '')).strip()
                if code:
                    holdings.append({
                        'code': code,
                        'name': str(h.get('股票名称', '')).strip(),
                        'cost': float(h.get('买入价格', 0) or 0),
                        'shares': int(h.get('持仓数量', 0) or 0),
                    })
            wb.close()
        except Exception:
            pass

    capital_path = os.path.join(os.path.dirname(__file__), '../../config/capital.xlsx')
    if os.path.exists(capital_path):
        try:
            wb = openpyxl.load_workbook(capital_path)
            ws = wb.active
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] is None:
                    continue
                key = str(row[0]).strip()
                val = row[1]
                if key == '总资金':
                    capital['total'] = float(val)
                elif key == '可用资金':
                    capital['cash'] = float(val)
            wb.close()
        except Exception:
            pass

    return holdings, capital


def main():
    parser = argparse.ArgumentParser(description='综合技术分析 — 两步六维度框架')
    parser.add_argument('symbol', help='股票代码 (如 603881, 300059)')
    parser.add_argument('--with-15min', action='store_true', help='包含15分钟级别分析')
    parser.add_argument('--json', action='store_true', dest='json_output', help='输出JSON格式')
    args = parser.parse_args()

    symbol = args.symbol.strip()
    analyzer = ComprehensiveAnalyzer(symbol)
    result = analyzer.analyze(include_15min=args.with_15min)

    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        holding = load_holding(symbol)
        report = ComprehensiveAnalyzer.format_report(result, holding=holding)
        print(report)

        # 纪律检查（仅对持仓股票运行）
        if holding:
            rt = result.get('realtime', {})
            price = rt.get('price', 0) or result.get('price', 0)
            checker = DisciplineChecker()

            # 加载全量持仓和资金用于总仓位检查
            all_holdings_raw, capital_data = _load_all_holdings()
            all_h = []
            if all_holdings_raw and price > 0:
                for h in all_holdings_raw:
                    h_price = price if h['code'] == symbol else h.get('cost', 0)
                    all_h.append({'code': h['code'], 'shares': h['shares'], 'price': h_price})

            check = checker.check(
                holding=holding,
                price=price,
                analysis_result=result,
                all_holdings=all_h,
                total_capital=capital_data.get('total', 0),
            )
            print(DisciplineChecker.format_report(check))

        save_html_report(symbol, report, result)


if __name__ == '__main__':
    main()
