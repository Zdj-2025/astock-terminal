#!/usr/bin/env python3
"""证券代码工具 — ETF/股票判定 + ts_code后缀(唯一权威实现)。

2026-08-17 P0-1A建。背景: update_all_kline.py曾用 `code.startswith('5')` 判ETF,
漏掉深市159xxx(走了个股pro.daily分支拉不到数),且沪ETF走了fund_daily未复权。
本模块统一判定, update_all_kline/minute_data/signal_backtest三方共用。

用法:
    from scripts.core.instrument import is_etf, to_ts_code
    is_etf('512480')  # True (沪ETF)
    is_etf('159841')  # True (深ETF)
    is_etf('000100')  # False
    to_ts_code('512480')  # '512480.SH'
    to_ts_code('159841')  # '159841.SZ'
"""


def is_etf(code) -> bool:
    """是否场内基金(ETF)。沪: 50/51/56/58开头; 深: 15开头(159xxx)。
    注: 深市股票只以00/30开头,15开头必为基金,无歧义。"""
    c = str(code).split('.')[0]
    return c[:2] in ('50', '51', '56', '58', '15')


def to_ts_code(code) -> str:
    """项目内纯数字code → tushare ts_code(带交易所后缀)。"""
    c = str(code).split('.')[0]
    if c[0] == '6' or c[:2] in ('50', '51', '56', '58'):
        return c + '.SH'
    return c + '.SZ'   # 00/30个股 + 159xxx ETF → 深


def to_bs_code(code) -> str:
    """项目内纯数字code → baostock code。指数请用sh.000001格式自行传参。"""
    c = str(code).split('.')[0]
    return ('sh.' if c[0] == '6' or c[:2] in ('50', '51', '56', '58') else 'sz.') + c
