"""
科技板块轮动监控系统 - CLI入口 v2
"""
import argparse
import json
import logging
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from scripts.screener.sector_rotation.config import SECTORS, SECTOR_DEFS, init_sectors
from scripts.screener.sector_rotation.data_fetcher import fetch_all_sectors_data
from scripts.screener.sector_rotation.signal_detector import scan_all_sectors
from scripts.screener.sector_rotation.reporter import generate_daily_report, generate_sector_detail

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_sectors():
    """获取板块配置，如果未初始化则初始化"""
    if not SECTORS:
        logger.info("初始化板块配置（获取ETF成分股）...")
        init_sectors()
    return SECTORS if SECTORS else SECTOR_DEFS


def cmd_scan(args):
    end_date = args.end_date or datetime.now().strftime('%Y%m%d')
    start_date = args.start_date or (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

    sectors = get_sectors()
    logger.info(f"开始扫描 {len(sectors)} 个板块: {start_date} ~ {end_date}")

    all_data = fetch_all_sectors_data(start_date, end_date)

    # 加载历史信号
    prev_signals = {}
    history_file = os.path.join(os.path.dirname(__file__), '.sector_signals_history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                dates = sorted(history.keys(), reverse=True)
                if dates:
                    prev_signals = history[dates[0]]
        except:
            pass

    results = scan_all_sectors(all_data, prev_signals)

    # 个股推荐（仅对有信号的板块）
    from scripts.screener.sector_rotation.stock_picker import pick_stocks
    for r in results:
        if r['signal'] in ('关注', '潜伏', '启动', '强势'):
            try:
                picks = pick_stocks(r['sector'], top_n=3)
                r['stock_picks'] = picks
            except Exception as e:
                logger.warning(f"个股推荐失败 {r['sector']}: {e}")
                r['stock_picks'] = []
        else:
            r['stock_picks'] = []
    report = generate_daily_report(results, datetime.now().strftime('%Y-%m-%d'))
    print(report)

    # 保存信号历史
    today_signals = {r['sector']: r['signal'] for r in results}
    try:
        history = {}
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        today_key = datetime.now().strftime('%Y-%m-%d')
        history[today_key] = today_signals
        cutoff = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        history = {k: v for k, v in history.items() if k >= cutoff}
        with open(history_file, 'w') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存信号历史失败: {e}")

    # 保存报告
    report_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reports', 'sector_rotation')
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"rotation_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"报告已保存: {report_file}")


def cmd_detail(args):
    sectors = get_sectors()
    sector_name = args.sector
    if sector_name not in sectors and sector_name not in SECTOR_DEFS:
        print(f"未知板块: {sector_name}")
        print(f"可选: {', '.join(list(sectors.keys()) or list(SECTOR_DEFS.keys()))}")
        return

    end_date = args.end_date or datetime.now().strftime('%Y%m%d')
    start_date = args.start_date or (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')

    from scripts.screener.sector_rotation.data_fetcher import (
        fetch_etf_daily, fetch_sector_moneyflow, fetch_sector_stocks_daily
    )
    config = sectors.get(sector_name) or SECTOR_DEFS[sector_name]

    etf_daily = fetch_etf_daily(config['etf_primary'], start_date, end_date)
    sector_flow = fetch_sector_moneyflow(sector_name, start_date, end_date)
    stocks_daily = fetch_sector_stocks_daily(sector_name, start_date, end_date)

    from scripts.screener.sector_rotation.scorer import score_sector
    from scripts.screener.sector_rotation.signal_detector import detect_signal

    score_result = score_sector(etf_daily, sector_flow, stocks_daily)
    signal_result = detect_signal(score_result)

    detail = {
        'sector': sector_name, 'total_score': score_result['total_score'],
        'signal': signal_result['signal'], 'emoji': signal_result['emoji'],
        'action': signal_result['action'], 'change': signal_result['change'],
        'price_volume': score_result['price_volume'],
        'fund_flow': score_result['fund_flow'],
        'technicals': score_result['technicals'],
        'breadth': score_result['breadth'],
    }
    print(generate_sector_detail(detail))


def cmd_backtest(args):
    print("回测模块开发中，敬请期待")


def main():
    parser = argparse.ArgumentParser(description='科技板块轮动监控系统 v2')
    subparsers = parser.add_subparsers(dest='command')

    scan_p = subparsers.add_parser('scan')
    scan_p.add_argument('--start-date')
    scan_p.add_argument('--end-date')

    detail_p = subparsers.add_parser('detail')
    detail_p.add_argument('sector')
    detail_p.add_argument('--start-date')
    detail_p.add_argument('--end-date')

    bt_p = subparsers.add_parser('backtest')
    bt_p.add_argument('--start', default='20250601')
    bt_p.add_argument('--end', default='20260531')

    args = parser.parse_args()
    if args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'detail':
        cmd_detail(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
