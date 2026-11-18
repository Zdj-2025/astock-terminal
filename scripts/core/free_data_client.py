"""
免费A股数据源客户端 — 直连HTTP/TCP，零第三方数据依赖

数据源优先级：mootdx(TCP) > 腾讯(HTTP) > 同花顺/新浪/百度 > 东财(限流)
参考: simonlin1212/a-stock-data + TradingAgents-astock

Author: Hermes Agent (借鉴 Simon林 架构)
Created: 2026-06-07
"""

import json
import logging
import math
import os
import random
import re
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# 基础设施
# ============================================================================

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _get_prefix(code: str) -> str:
    """6位A股代码 -> 市场前缀 (sh/sz/bj)"""
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith("8"):
        return "bj"
    return "sz"


def normalize_code(symbol: str) -> str:
    """统一为纯6位代码，兼容 600379/SH600379/600379.SH 等格式"""
    s = symbol.strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    for prefix in ("SH", "SZ", "BJ"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip()


# ============================================================================
# 东财统一限流（防封三件套：串行+抖动+会话复用）
# ============================================================================

_EM_SESSION = requests.Session()
_EM_SESSION.headers.update({"User-Agent": _UA})
_EM_MIN_INTERVAL = float(os.environ.get("EM_MIN_INTERVAL", "1.0"))
_em_last_call = [0.0]


def em_get(url, params=None, headers=None, timeout=15, **kwargs):
    """东财统一请求入口：自动节流 + 复用 session + 默认 UA"""
    wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return _EM_SESSION.get(url, params=params, headers=headers,
                               timeout=timeout, **kwargs)
    finally:
        _em_last_call[0] = time.time()


_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _eastmoney_datacenter(report_name, filter_str="", page_size=50,
                           sort_columns="", sort_types="-1"):
    """东财数据中心统一查询 — 龙虎榜/解禁/融资融券等共用"""
    params = {
        "reportName": report_name,
        "columns": "ALL",
        "filter": filter_str,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortColumns": sort_columns,
        "sortTypes": sort_types,
        "source": "WEB",
        "client": "WEB",
    }
    try:
        r = em_get(_DATACENTER_URL, params=params, timeout=15)
        d = r.json()
        if d.get("result") and d["result"].get("data"):
            return d["result"]["data"]
    except Exception as e:
        logger.warning("东财datacenter查询失败: %s", e)
    return []


# ============================================================================
# 1. 腾讯财经 — 实时报价（PE/PB/市值/换手率）
# ============================================================================

def get_realtime_quote(codes):
    """腾讯财经批量实时报价
    
    Args:
        codes: 股票代码列表，如 ['603881', '688222'] 或单个 '603881'
    
    Returns:
        DataFrame: columns=[代码, 名称, 现价, 涨跌幅%, 换手率%, 
                          PE_TTM, PB, 总市值亿, 流通市值亿, 涨停价, 跌停价]
    """
    if isinstance(codes, str):
        codes = [codes]
    
    prefixed = [f"{_get_prefix(c)}{c}" for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        raw = resp.read().decode("gbk")
    except Exception as e:
        logger.error("腾讯报价获取失败: %s", e)
        return pd.DataFrame()
    
    rows = []
    for line in raw.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue
        code = key[2:]  # 去掉 sh/sz/bj 前缀
        
        def safe_float(idx):
            try:
                return float(vals[idx]) if vals[idx] else 0.0
            except (ValueError, IndexError):
                return 0.0
        
        rows.append({
            "代码": code,
            "名称": vals[1],
            "现价": safe_float(3),
            "昨收": safe_float(4),
            "今开": safe_float(5),
            "涨跌幅%": safe_float(32),
            "最高": safe_float(33),
            "最低": safe_float(34),
            "换手率%": safe_float(38),
            "PE_TTM": safe_float(39),
            "总市值亿": safe_float(44),
            "流通市值亿": safe_float(45),
            "PB": safe_float(46),
            "涨停价": safe_float(47),
            "跌停价": safe_float(48),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# 2. 龙虎榜
# ============================================================================

def get_dragon_tiger(code=None, trade_date=None, look_back_days=30):
    """龙虎榜数据（东财 datacenter）
    
    Args:
        code: 股票代码（可选，为空则返回全市场）
        trade_date: 交易日期 YYYY-MM-DD（默认今天）
        look_back_days: 回溯天数
    
    Returns:
        dict: {上榜记录: DataFrame, 买入席位: DataFrame, 卖出席位: DataFrame}
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)
    start_str = start_dt.strftime("%Y-%m-%d")
    
    result = {"上榜记录": pd.DataFrame(), "买入席位": pd.DataFrame(), "卖出席位": pd.DataFrame()}
    
    # 上榜记录
    filter_str = (f"(TRADE_DATE>='{start_str}')"
                  f"(TRADE_DATE<='{trade_date}')")
    if code:
        code = normalize_code(code)
        filter_str += f'(SECURITY_CODE="{code}")'
    
    data = _eastmoney_datacenter(
        "RPT_DAILYBILLBOARD_DETAILSNEW",
        filter_str=filter_str,
        page_size=100,
        sort_columns="TRADE_DATE",
        sort_types="-1",
    )
    
    if data:
        rows = []
        for row in data:
            rows.append({
                "日期": str(row.get("TRADE_DATE", ""))[:10],
                "代码": row.get("SECURITY_CODE", ""),
                "名称": row.get("SECURITY_NAME_ABBR", ""),
                "收盘价": row.get("CLOSE_PRICE", 0),
                "涨跌幅%": row.get("CHANGE_RATE", 0),
                "上榜原因": row.get("EXPLANATION", ""),
                "净买入万": round((row.get("BILLBOARD_NET_AMT") or 0) / 10000, 1),
                "买入额万": round((row.get("BILLBOARD_BUY_AMT") or 0) / 10000, 1),
                "卖出额万": round((row.get("BILLBOARD_SELL_AMT") or 0) / 10000, 1),
                "换手率%": round(float(row.get("TURNOVERRATE") or 0), 2),
            })
        result["上榜记录"] = pd.DataFrame(rows)
        
        # 最近上榜日期的席位明细
        if result["上榜记录"].shape[0] > 0:
            latest_date = result["上榜记录"].iloc[0]["日期"]
            target_code = code or result["上榜记录"].iloc[0]["代码"]
            
            # 买入席位
            buy_data = _eastmoney_datacenter(
                "RPT_BILLBOARD_DAILYDETAILSBUY",
                filter_str=f"(TRADE_DATE='{latest_date}')(SECURITY_CODE=\"{target_code}\")",
                page_size=10, sort_columns="BUY", sort_types="-1",
            )
            if buy_data:
                result["买入席位"] = pd.DataFrame([{
                    "营业部": r.get("OPERATEDEPT_NAME", ""),
                    "买入万": round((r.get("BUY") or 0) / 10000, 1),
                    "卖出万": round((r.get("SELL") or 0) / 10000, 1),
                    "净额万": round((r.get("NET") or 0) / 10000, 1),
                } for r in buy_data])
            
            # 卖出席位
            sell_data = _eastmoney_datacenter(
                "RPT_BILLBOARD_DAILYDETAILSSELL",
                filter_str=f"(TRADE_DATE='{latest_date}')(SECURITY_CODE=\"{target_code}\")",
                page_size=10, sort_columns="SELL", sort_types="-1",
            )
            if sell_data:
                result["卖出席位"] = pd.DataFrame([{
                    "营业部": r.get("OPERATEDEPT_NAME", ""),
                    "买入万": round((r.get("BUY") or 0) / 10000, 1),
                    "卖出万": round((r.get("SELL") or 0) / 10000, 1),
                    "净额万": round((r.get("NET") or 0) / 10000, 1),
                } for r in sell_data])
    
    return result


# ============================================================================
# 3. 限售解禁日历
# ============================================================================

def get_lockup_expiry(code=None, forward_days=90, trade_date=None):
    """限售解禁日历（东财 datacenter）
    
    Args:
        code: 股票代码（可选）
        forward_days: 向前看天数
        trade_date: 基准日期
    
    Returns:
        DataFrame: 解禁记录
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    filter_str = ""
    if code:
        code = normalize_code(code)
        filter_str = f'(SECURITY_CODE="{code}")'
    
    data = _eastmoney_datacenter(
        "RPT_LIFT_STAGE",
        filter_str=filter_str,
        page_size=50,
        sort_columns="FREE_DATE",
        sort_types="-1",
    )
    
    if not data:
        return pd.DataFrame()
    
    rows = []
    for row in data:
        free_date = str(row.get("FREE_DATE", ""))[:10]
        rows.append({
            "解禁日期": free_date,
            "代码": row.get("SECURITY_CODE", ""),
            "名称": row.get("SECURITY_NAME_ABBR", ""),
            "解禁类型": row.get("LIMITED_STOCK_TYPE", ""),
            "解禁数量": row.get("FREE_SHARES_NUM", 0),
            "解禁占比%": row.get("FREE_RATIO", 0),
            "解禁市值万": row.get("FREE_MARKET_CAP", 0),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# 4. 个股资金流向
# ============================================================================

def get_fund_flow(code, days=20):
    """个股资金流向（东财 push2/push2his）
    
    Args:
        code: 股票代码
        days: 历史天数
    
    Returns:
        dict: {分钟流: DataFrame, 日流: DataFrame}
    """
    code = normalize_code(code)
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    
    result = {"分钟流": pd.DataFrame(), "日流": pd.DataFrame()}
    
    # 分钟级资金流
    try:
        url_rt = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
        params_rt = {
            "secid": secid, "klt": 1,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
        }
        r = em_get(url_rt, params=params_rt, timeout=10)
        d = r.json()
        klines = d.get("data", {}).get("klines", [])
        
        if klines:
            rows = []
            for line in klines:
                parts = line.split(",")
                if len(parts) >= 6:
                    rows.append({
                        "时间": parts[0],
                        "主力净流入万": round(float(parts[1]) / 1e4, 1),
                        "小单净流入万": round(float(parts[2]) / 1e4, 1),
                        "中单净流入万": round(float(parts[3]) / 1e4, 1),
                        "大单净流入万": round(float(parts[4]) / 1e4, 1),
                        "超大单净流入万": round(float(parts[5]) / 1e4, 1),
                    })
            result["分钟流"] = pd.DataFrame(rows)
    except Exception as e:
        logger.warning("分钟资金流获取失败 %s: %s", code, e)
    
    # 日级资金流
    try:
        url_hist = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params_hist = {
            "secid": secid, "lmt": days, "klt": 101,
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
        }
        rh = em_get(url_hist, params=params_hist, timeout=10)
        dh = rh.json()
        hist_klines = dh.get("data", {}).get("klines", [])
        
        if hist_klines:
            rows = []
            for line in hist_klines:
                parts = line.split(",")
                if len(parts) >= 6:
                    rows.append({
                        "日期": parts[0],
                        "主力净流入万": round(float(parts[1]) / 1e4, 1),
                        "小单万": round(float(parts[2]) / 1e4, 1),
                        "中单万": round(float(parts[3]) / 1e4, 1),
                        "大单万": round(float(parts[4]) / 1e4, 1),
                        "超大单万": round(float(parts[5]) / 1e4, 1),
                    })
            result["日流"] = pd.DataFrame(rows)
    except Exception as e:
        logger.warning("日级资金流获取失败 %s: %s", code, e)
    
    return result


# ============================================================================
# 5. 融资融券
# ============================================================================

def get_margin_trading(code=None, trade_date=None):
    """融资融券数据（东财 datacenter）
    
    Returns:
        DataFrame
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    filter_str = ""
    if code:
        code = normalize_code(code)
        filter_str = f'(SECURITY_CODE="{code}")'
    
    data = _eastmoney_datacenter(
        "RPT_RZRQ_LSHJ",
        filter_str=filter_str,
        page_size=30,
        sort_columns="TRADE_DATE",
        sort_types="-1",
    )
    
    if not data:
        return pd.DataFrame()
    
    rows = []
    for row in data:
        rows.append({
            "日期": str(row.get("TRADE_DATE", ""))[:10],
            "代码": row.get("SECURITY_CODE", ""),
            "名称": row.get("SECURITY_NAME_ABBR", ""),
            "融资买入万": round((row.get("RZYE") or 0) / 10000, 1),
            "融资余额万": round((row.get("RZMJE") or 0) / 10000, 1),
            "融券余量": row.get("RQYE", 0),
            "融券余额万": round((row.get("RQMJE") or 0) / 10000, 1),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# 6. 大宗交易
# ============================================================================

def get_block_trade(code=None, trade_date=None):
    """大宗交易数据（东财 datacenter）
    
    Returns:
        DataFrame
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    start_dt = datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)
    start_str = start_dt.strftime("%Y-%m-%d")
    
    filter_str = (f"(TRADE_DATE>='{start_str}')"
                  f"(TRADE_DATE<='{trade_date}')")
    if code:
        code = normalize_code(code)
        filter_str += f'(SECURITY_CODE="{code}")'
    
    data = _eastmoney_datacenter(
        "RPT_DATATRANSACTION",
        filter_str=filter_str,
        page_size=50,
        sort_columns="TRADE_DATE",
        sort_types="-1",
    )
    
    if not data:
        return pd.DataFrame()
    
    rows = []
    for row in data:
        rows.append({
            "日期": str(row.get("TRADE_DATE", ""))[:10],
            "代码": row.get("SECURITY_CODE", ""),
            "名称": row.get("SECURITY_NAME_ABBR", ""),
            "成交价": row.get("DEAL_PRICE", 0),
            "成交量": row.get("DEAL_VOL", 0),
            "成交额万": round((row.get("DEAL_AMT") or 0) / 10000, 1),
            "折溢率%": row.get("PREMIUM_RATIO", 0),
            "买方营业部": row.get("BUYER_DEPT_NAME", ""),
            "卖方营业部": row.get("SELLER_DEPT_NAME", ""),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# 7. 股东户数变化
# ============================================================================

def get_holder_change(code):
    """股东户数变化（东财 datacenter）
    
    Returns:
        DataFrame
    """
    code = normalize_code(code)
    data = _eastmoney_datacenter(
        "RPT_F10_EH_HOLDERSNUM",
        filter_str=f'(SECURITY_CODE="{code}")',
        page_size=10,
        sort_columns="END_DATE",
        sort_types="-1",
    )
    
    if not data:
        return pd.DataFrame()
    
    rows = []
    for row in data:
        rows.append({
            "截止日期": str(row.get("END_DATE", ""))[:10],
            "股东户数": row.get("HOLDER_NUM", 0),
            "户数变化": row.get("HOLDER_NUM_CHANGE", 0),
            "变化比例%": row.get("HOLDER_NUM_RATIO", 0),
            "户均持股数量": row.get("AVG_HOLD_NUM", 0),
            "户均持股金额万": round((row.get("AVG_HOLD_AMT") or 0) / 10000, 1),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# 8. 行业板块排名
# ============================================================================

def get_industry_ranking():
    """行业板块排名（东财 push2）
    
    Returns:
        DataFrame: 全行业涨跌幅排名
    """
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1", "pz": "100", "po": "1", "np": "1",
            "fltt": "2", "invt": "2", "fs": "m:90+t:2",
            "fields": "f2,f3,f4,f12,f13,f14,f104,f105,f128,f136,f140,f141",
        }
        r = em_get(url, params=params, timeout=15)
        d = r.json()
        items = d.get("data", {}).get("diff", [])
        
        if not items:
            return pd.DataFrame()
        
        rows = []
        for i, item in enumerate(items):
            rows.append({
                "排名": i + 1,
                "行业": item.get("f14", ""),
                "涨跌幅%": item.get("f3", 0),
                "上涨数": item.get("f104", 0),
                "下跌数": item.get("f105", 0),
                "领涨股": item.get("f140", ""),
                "领涨股涨幅%": item.get("f141", 0),
            })
        
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error("行业排名获取失败: %s", e)
        return pd.DataFrame()


# ============================================================================
# 9. 北向资金（同花顺）
# ============================================================================

def get_northbound_flow():
    """北向资金流向（同花顺 hsgtApi）
    
    Returns:
        dict: {hgt: 沪股通净买入亿, sgt: 深股通净买入亿, total: 合计, 
               minutes: DataFrame分钟级数据}
    """
    headers = {
        "User-Agent": _UA,
        "Host": "data.hexin.cn",
        "Referer": "https://data.hexin.cn/",
    }
    
    result = {"hgt": 0.0, "sgt": 0.0, "total": 0.0, "minutes": pd.DataFrame()}
    
    try:
        url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
        r = requests.get(url, headers=headers, timeout=10)
        d = r.json()
        
        times = d.get("time", [])
        hgt = d.get("hgt", [])
        sgt = d.get("sgt", [])
        
        if times:
            result["hgt"] = float(hgt[-1]) if hgt else 0
            result["sgt"] = float(sgt[-1]) if sgt else 0
            result["total"] = result["hgt"] + result["sgt"]
            
            # 分钟级数据
            rows = []
            for i, t in enumerate(times):
                rows.append({
                    "时间": t,
                    "沪股通亿": float(hgt[i]) if i < len(hgt) else 0,
                    "深股通亿": float(sgt[i]) if i < len(sgt) else 0,
                    "合计亿": (float(hgt[i]) if i < len(hgt) else 0) + 
                              (float(sgt[i]) if i < len(sgt) else 0),
                })
            result["minutes"] = pd.DataFrame(rows)
    except Exception as e:
        logger.warning("北向资金获取失败: %s", e)
    
    return result


# ============================================================================
# 10. 同花顺热点强势股
# ============================================================================

def get_hot_stocks(trade_date=None):
    """同花顺热点强势股（含题材归因标签）
    
    Args:
        trade_date: 日期 YYYY-MM-DD，默认今天
    
    Returns:
        DataFrame: 当日强势股 + 题材标签
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        url = (f"http://zx.10jqka.com.cn/event/api/getharden/"
               f"date/{trade_date}/orderby/date/orderway/desc/charset/GBK/")
        headers = {"User-Agent": _UA}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        
        if data.get("errocode", 0) != 0:
            return pd.DataFrame()
        
        rows_data = data.get("data") or []
        if not rows_data:
            return pd.DataFrame()
        
        records = []
        for row in rows_data:
            records.append({
                "代码": row.get("code", ""),
                "名称": row.get("name", ""),
                "涨跌幅%": row.get("zhangfu", ""),
                "换手率%": row.get("huanshou", ""),
                "成交额": row.get("chengjiaoe", ""),
                "大单净量": row.get("ddejingliang", ""),
                "题材标签": row.get("reason", ""),
            })
        
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning("热点强势股获取失败: %s", e)
        return pd.DataFrame()


# ============================================================================
# 11. 东财个股信息（补充估值/行业等）
# ============================================================================

def get_stock_info(code):
    """东财 push2 个股基本信息
    
    Returns:
        dict: 行业/总股本/流通股本/总市值/流通市值/上市日期
    """
    code = normalize_code(code)
    market = 1 if code.startswith("6") else 0
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "fltt": "2", "invt": "2",
        "fields": "f57,f58,f84,f85,f127,f116,f117,f189,f43",
        "secid": f"{market}.{code}",
    }
    try:
        r = em_get(url, params=params, timeout=10)
        d = r.json().get("data", {})
        if d:
            return {
                "代码": d.get("f57", ""),
                "名称": d.get("f58", ""),
                "总股本": d.get("f84", 0),
                "流通股本": d.get("f85", 0),
                "行业": d.get("f127", ""),
                "总市值": d.get("f116", 0),
                "流通市值": d.get("f117", 0),
                "上市日期": d.get("f189", ""),
            }
    except Exception as e:
        logger.warning("东财个股信息获取失败 %s: %s", code, e)
    return {}


# ============================================================================
# 12. 概念板块（百度股市通）
# ============================================================================

def get_concept_blocks(code):
    """概念板块归属（百度股市通）
    
    Returns:
        DataFrame
    """
    code = normalize_code(code)
    headers = {
        "Host": "finance.pae.baidu.com",
        "User-Agent": _UA,
        "Accept": "application/vnd.finance-web.v1+json",
        "Origin": "https://gushitong.baidu.com",
        "Referer": "https://gushitong.baidu.com/",
    }
    
    try:
        url = (f"https://finance.pae.baidu.com/api/getrelatedblock"
               f'?stock=[{{"code":"{code}","market":"ab","type":"stock"}}]'
               "&finClientType=pc")
        r = requests.get(url, headers=headers, timeout=10)
        d = r.json()
        
        if str(d.get("ResultCode", -1)) != "0":
            return pd.DataFrame()
        
        categories = d.get("Result", {}).get(code, [])
        if not categories:
            return pd.DataFrame()
        
        rows = []
        for cat in categories:
            cat_name = cat.get("name", "")
            for item in cat.get("list", []):
                rows.append({
                    "类别": cat_name,
                    "板块": item.get("name", ""),
                    "涨跌幅%": item.get("ratio", ""),
                    "描述": item.get("describe", ""),
                })
        
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("概念板块获取失败 %s: %s", code, e)
        return pd.DataFrame()


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("免费数据源客户端 — 快速测试")
    print("=" * 60)
    
    # 1. 实时报价
    print("\n[1] 腾讯实时报价 603881,688222")
    df = get_realtime_quote(["603881", "688222"])
    if not df.empty:
        print(df[["代码", "名称", "现价", "涨跌幅%", "PE_TTM", "PB", "总市值亿"]].to_string(index=False))
    
    # 2. 行业排名
    print("\n[2] 行业排名 TOP5")
    df = get_industry_ranking()
    if not df.empty:
        print(df.head(5).to_string(index=False))
    
    # 3. 北向资金
    print("\n[3] 北向资金")
    nb = get_northbound_flow()
    print(f"沪股通: {nb['hgt']:.2f}亿  深股通: {nb['sgt']:.2f}亿  合计: {nb['total']:.2f}亿")
    
    print("\n测试完成!")
