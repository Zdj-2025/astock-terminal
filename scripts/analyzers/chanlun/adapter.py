#!/usr/bin/env python3
"""数据适配器 — Tushare日线 + AKShare分钟线 → chan.py CKLine_Unit格式"""
import re

import akshare as ak
import pandas as pd

from scripts.core.tushare_client import TushareClient

from .vendor.chanpy.Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from .vendor.chanpy.Common.CTime import CTime
from .vendor.chanpy.KLine.KLine_Unit import CKLine_Unit
from .vendor.chanpy.DataAPI.CommonStockAPI import CCommonStockApi

_MIN_PERIOD_MAP = {
    KL_TYPE.K_1M: "1",
    KL_TYPE.K_5M: "5",
    KL_TYPE.K_15M: "15",
    KL_TYPE.K_30M: "30",
    KL_TYPE.K_60M: "60",
}


def _parse_date(val) -> tuple:
    """从各种日期格式中提取 (year, month, day, hour, minute)"""
    if isinstance(val, pd.Timestamp):
        return val.year, val.month, val.day, val.hour, val.minute
    s = str(val).strip()
    # "2026-04-01 10:30:00"
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})', s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    # "20260401"
    m = re.match(r'(\d{4})(\d{2})(\d{2})', s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0
    # "2026-04-01"
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0
    raise ValueError(f"无法解析日期: {s}")


def _to_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


class CAkshareAPI(CCommonStockApi):
    """混合数据适配器: Tushare日线/周线 + AKShare分钟线"""

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        self._code = code
        super().__init__(code, k_type, begin_date, end_date, autype)

    def SetBasciInfo(self):
        self.name = self._code
        self.is_stock = True

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

    def get_kl_data(self):
        adjust = {AUTYPE.QFQ: "qfq", AUTYPE.HFQ: "hfq", AUTYPE.NONE: ""}.get(self.autype, "qfq")
        start = self.begin_date.replace("-", "") if self.begin_date else "20240101"
        end = self.end_date.replace("-", "") if self.end_date else "20991231"

        if self.k_type in (KL_TYPE.K_DAY, KL_TYPE.K_WEEK, KL_TYPE.K_MON):
            # 日线/周线/月线 — Tushare
            freq_map = {KL_TYPE.K_DAY: "D", KL_TYPE.K_WEEK: "W", KL_TYPE.K_MON: "M"}
            df = TushareClient().get_kline(
                self._code, freq=freq_map[self.k_type],
                start=start, end=end, adj=adjust,
            )
            for _, row in df.iterrows():
                y, mo, d, h, mi = _parse_date(row['date'])
                yield CKLine_Unit({
                    DATA_FIELD.FIELD_TIME: CTime(y, mo, d, h, mi),
                    DATA_FIELD.FIELD_OPEN: _to_float(row['open']),
                    DATA_FIELD.FIELD_HIGH: _to_float(row['high']),
                    DATA_FIELD.FIELD_LOW: _to_float(row['low']),
                    DATA_FIELD.FIELD_CLOSE: _to_float(row['close']),
                    DATA_FIELD.FIELD_VOLUME: _to_float(row['volume']),
                    DATA_FIELD.FIELD_TURNOVER: _to_float(row.get('amount', 0)),
                })

        elif self.k_type in _MIN_PERIOD_MAP:
            # 分钟线 — AKShare（Tushare分钟线限2次/天）
            period = _MIN_PERIOD_MAP[self.k_type]
            start_dt = f"{start[:4]}-{start[4:6]}-{start[6:8]} 09:30:00"
            end_dt = f"{end[:4]}-{end[4:6]}-{end[6:8]} 15:00:00"
            df = ak.stock_zh_a_hist_min_em(
                symbol=self._code, period=period,
                start_date=start_dt, end_date=end_dt, adjust=adjust,
            )
            col_time = '时间' if '时间' in df.columns else '日期'
            for _, row in df.iterrows():
                y, mo, d, h, mi = _parse_date(row[col_time])
                yield CKLine_Unit({
                    DATA_FIELD.FIELD_TIME: CTime(y, mo, d, h, mi),
                    DATA_FIELD.FIELD_OPEN: _to_float(row['开盘']),
                    DATA_FIELD.FIELD_HIGH: _to_float(row['最高']),
                    DATA_FIELD.FIELD_LOW: _to_float(row['最低']),
                    DATA_FIELD.FIELD_CLOSE: _to_float(row['收盘']),
                    DATA_FIELD.FIELD_VOLUME: _to_float(row['成交量']),
                    DATA_FIELD.FIELD_TURNOVER: _to_float(row.get('成交额', 0)),
                })

        else:
            raise ValueError(f"不支持的时间级别: {self.k_type}")
