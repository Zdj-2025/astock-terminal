#!/usr/bin/env python3
"""Layer 1: 全A股基础过滤。

从~5000只过滤到~3000只候选池。
排除: ST/新股/停牌/涨跌停/北交所/微盘股。
支持概念板块和行业过滤。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from scripts.core.tushare_client import TushareClient, from_ts_code


class UniverseFilter:
    """全A股基础过滤器"""

    def __init__(self, client: TushareClient):
        self.client = client

    def filter(
        self,
        concept: Optional[str] = None,
        concept_id: Optional[str] = None,
        industry: Optional[str] = None,
        min_mcap: float = 20,
        max_mcap: Optional[float] = None,
        exclude_st: bool = True,
        min_list_days: int = 60,
        exclude_bj: bool = True,
        exclude_limit: bool = True,
        trade_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """执行基础过滤。

        Args:
            concept: 概念板块名称关键词（如"AI芯片"），模糊匹配
            concept_id: 精确概念板块ID（优先于concept）
            industry: Tushare行业名称
            min_mcap: 最小总市值（亿元），默认20亿
            max_mcap: 最大总市值（亿元），可选
            exclude_st: 排除ST/*ST
            min_list_days: 最小上市天数，默认60天
            exclude_bj: 排除北交所
            exclude_limit: 排除涨跌停
            trade_date: 交易日期YYYYMMDD，默认最近交易日

        Returns:
            DataFrame: ts_code, symbol, name, industry, market, list_date, total_mv, ...
        """
        # 1. 获取交易日
        if not trade_date:
            trade_date = self.client.get_recent_trade_day()

        # 2. 获取全A股列表
        stocks = self.client.get_stock_list()
        if stocks.empty:
            return pd.DataFrame()
        stocks["symbol"] = stocks["symbol"].astype(str)

        # 3. 概念板块过滤（如指定）
        if concept_id or concept:
            stocks = self._filter_by_concept(stocks, concept, concept_id)
            if stocks.empty:
                return stocks

        # 4. 行业过滤（如指定）
        if industry:
            stocks = stocks[stocks["industry"] == industry]
            if stocks.empty:
                return stocks

        # 5. 排除ST
        if exclude_st:
            stocks = stocks[~stocks["name"].str.contains("ST", na=False)]

        # 6. 排除北交所（代码以4/8开头但排除沪深）
        if exclude_bj:
            stocks = stocks[~stocks["ts_code"].str.endswith(".BJ")]

        # 7. 排除新股
        if min_list_days > 0:
            anchor = datetime.strptime(trade_date, "%Y%m%d") if trade_date else datetime.now()
            cutoff = (anchor - timedelta(days=min_list_days)).strftime("%Y%m%d")
            stocks = stocks[stocks["list_date"] <= cutoff]

        if stocks.empty:
            return stocks

        # 8. 获取当日行情数据（市值+涨跌停过滤）
        daily = self.client.get_daily_basic(trade_date=trade_date)
        if daily.empty:
            return stocks

        # 合并市值数据
        merged = stocks.merge(daily[["ts_code", "total_mv", "pe_ttm", "pb",
                                      "turnover_rate", "close", "volume_ratio"]],
                              on="ts_code", how="inner")
        # inner join 排除停牌（停牌股无daily_basic数据）

        # 9. 市值过滤
        if min_mcap > 0:
            merged = merged[merged["total_mv"] >= min_mcap * 1e4]
        if max_mcap is not None:
            merged = merged[merged["total_mv"] <= max_mcap * 1e4]

        # 10. 排除涨跌停
        if exclude_limit:
            kline_data = self._get_today_kline(merged["ts_code"].tolist(), trade_date)
            if not kline_data.empty:
                limit_mask = kline_data["change_pct"].abs() >= 9.9
                # 创业板/科创板20%涨跌停
                cyb_mask = kline_data["ts_code"].str.startswith(("3", "68"))
                limit_mask = limit_mask & ~cyb_mask
                cyb_limit = kline_data["change_pct"].abs() >= 19.9
                limit_mask = limit_mask | (cyb_limit & cyb_mask)
                limit_codes = set(kline_data.loc[limit_mask, "ts_code"])
                merged = merged[~merged["ts_code"].isin(limit_codes)]

        return merged.reset_index(drop=True)

    def _filter_by_concept(self, stocks: pd.DataFrame,
                           keyword: Optional[str],
                           concept_id: Optional[str]) -> pd.DataFrame:
        """按概念板块过滤股票列表"""
        if concept_id:
            concept_stocks = self.client.get_concept_stocks(concept_id)
        elif keyword:
            concepts = self.client.search_concepts(keyword)
            if concepts.empty:
                return pd.DataFrame()
            # 取匹配的第一个概念
            concept_code = concepts.iloc[0]["code"]
            concept_stocks = self.client.get_concept_stocks(concept_code)
        else:
            return stocks

        if concept_stocks.empty:
            return pd.DataFrame()

        concept_codes = set(concept_stocks["ts_code"].tolist())
        return stocks[stocks["ts_code"].isin(concept_codes)]

    def _get_today_kline(self, ts_codes: list[str],
                         trade_date: str) -> pd.DataFrame:
        """批量获取当日K线（用于涨跌停判断）"""
        from scripts.core.tushare_client import to_ts_code
        rows = []
        for code in ts_codes[:500]:  # 限制数量避免限流
            try:
                df = self.client.get_kline(from_ts_code(code), freq="D",
                                            start=trade_date, end=trade_date)
                if not df.empty:
                    rows.append({
                        "ts_code": code,
                        "change_pct": df.iloc[-1].get("change_pct", 0),
                    })
            except Exception:
                continue
        return pd.DataFrame(rows)
