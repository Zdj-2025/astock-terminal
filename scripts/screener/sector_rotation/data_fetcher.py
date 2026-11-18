"""
科技板块轮动监控系统 - 数据获取层
统一封装Tushare数据获取，处理频率限制
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import tushare as ts

from .config import SECTORS, SECTOR_DEFS, TUSHARE_API_LIMIT, init_sectors

logger = logging.getLogger(__name__)

# 请求间隔（毫秒），避免触发频率限制
_REQUEST_INTERVAL = 60.0 / TUSHARE_API_LIMIT * 1000  # ms per request
_last_request_time = 0.0


def _rate_limit():
    """简单的频率限制"""
    global _last_request_time
    now = time.time() * 1000
    elapsed = now - _last_request_time
    if elapsed < _REQUEST_INTERVAL:
        time.sleep((_REQUEST_INTERVAL - elapsed) / 1000.0)
    _last_request_time = time.time() * 1000


def _get_pro():
    """获取tushare pro接口"""
    return ts.pro_api()


def fetch_etf_daily(etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取ETF日线数据
    Args:
        etf_code: 如 '512480.SH'
        start_date: '20250601'
        end_date: '20260607'
    Returns:
        DataFrame with columns: trade_date, open, high, low, close, vol, amount, pct_chg
    """
    _rate_limit()
    pro = _get_pro()
    df = pro.fund_daily(ts_code=etf_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        logger.warning(f"ETF日线数据为空: {etf_code}")
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def fetch_stock_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取个股日线数据"""
    _rate_limit()
    pro = _get_pro()
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def fetch_moneyflow(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取个股资金流向
    关键列: buy_elg_amount(超大单买), sell_elg_amount, buy_lg_amount(大单买), sell_lg_amount,
            net_mf_amount(主力净流入), buy_sm_amount(小单买), sell_sm_amount
    """
    _rate_limit()
    pro = _get_pro()
    df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def fetch_daily_basic(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取个股每日指标（换手率等）"""
    _rate_limit()
    pro = _get_pro()
    df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date,
                         fields='ts_code,trade_date,turnover_rate,volume,amount,pe,pb')
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def fetch_sector_moneyflow(sector_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取某板块所有个股的资金流数据并汇总为板块级
    Returns:
        DataFrame with: trade_date, net_mf_amount(主力净流入), buy_elg_amount, sell_elg_amount,
                        buy_lg_amount, sell_lg_amount, total_amount
    """
    sector = SECTORS.get(sector_name) or SECTOR_DEFS.get(sector_name, {})
    stock_codes = list(sector.get("stocks", {}).keys())
    if not stock_codes:
        stock_codes = sector.get("seed_stocks", [])

    all_flows = []
    for code in stock_codes:
        df = fetch_moneyflow(code, start_date, end_date)
        if df.empty:
            logger.warning(f"资金流数据为空: {code}")
            continue
        all_flows.append(df)

    if not all_flows:
        return pd.DataFrame()

    # 合并所有个股资金流
    combined = pd.concat(all_flows, ignore_index=True)

    # 按日期汇总
    sector_flow = combined.groupby('trade_date').agg({
        'net_mf_amount': 'sum',
        'buy_elg_amount': 'sum',
        'sell_elg_amount': 'sum',
        'buy_lg_amount': 'sum',
        'sell_lg_amount': 'sum',
        'buy_md_amount': 'sum',
        'sell_md_amount': 'sum',
        'buy_sm_amount': 'sum',
        'sell_sm_amount': 'sum',
    }).reset_index()

    sector_flow = sector_flow.sort_values('trade_date').reset_index(drop=True)

    # 计算超大单净流入
    sector_flow['elg_net_amount'] = sector_flow['buy_elg_amount'] - sector_flow['sell_elg_amount']
    # 计算板块总成交额
    sector_flow['total_amount'] = (sector_flow['buy_elg_amount'] + sector_flow['sell_elg_amount'] +
                                    sector_flow['buy_lg_amount'] + sector_flow['sell_lg_amount'] +
                                    sector_flow['buy_md_amount'] + sector_flow['sell_md_amount'] +
                                    sector_flow['buy_sm_amount'] + sector_flow['sell_sm_amount']) / 2

    return sector_flow


def fetch_sector_stocks_daily(sector_name: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """获取板块内所有个股的日线数据"""
    sector = SECTORS[sector_name]
    result = {}
    for code in sector["stocks"]:
        df = fetch_stock_daily(code, start_date, end_date)
        if not df.empty:
            result[code] = df
    return result


def fetch_all_sectors_data(start_date: str, end_date: str) -> Dict[str, Dict]:
    """
    一次性获取所有板块数据
    Returns:
        {sector_name: {
            'etf_daily': DataFrame,
            'sector_moneyflow': DataFrame,
            'stocks_daily': {code: DataFrame, ...},
            'stocks_basic': {code: DataFrame, ...},
        }}
    """
    results = {}
    total_sectors = len(SECTORS)
    
    for i, (name, config) in enumerate(SECTORS.items() if SECTORS else SECTOR_DEFS.items()):
        logger.info(f"获取数据 [{i+1}/{total_sectors}]: {name}")
        
        sector_data = {}
        
        # ETF日线
        etf_code = config.get('etf_primary', '')
        if etf_code:
            sector_data['etf_daily'] = fetch_etf_daily(etf_code, start_date, end_date)
        else:
            sector_data['etf_daily'] = pd.DataFrame()
        
        # 板块资金流
        sector_data['sector_moneyflow'] = fetch_sector_moneyflow(name, start_date, end_date)
        
        # 个股日线
        stocks_cfg = config.get('stocks', {})
        if stocks_cfg:
            sector_data['stocks_daily'] = fetch_sector_stocks_daily(name, start_date, end_date)
        else:
            sector_data['stocks_daily'] = {}
        
        results[name] = sector_data
    
    return results
