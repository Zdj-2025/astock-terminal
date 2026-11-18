#!/usr/bin/env python3
"""缠论分析引擎 — 封装chan.py，提供简洁的分析接口"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .vendor.chanpy.Chan import CChan
from .vendor.chanpy.ChanConfig import CChanConfig
from .vendor.chanpy.Common.CEnum import AUTYPE, KL_TYPE
from .vendor.chanpy.DataAPI.CommonStockAPI import CCommonStockApi
from .adapter import CAkshareAPI


@dataclass
class Stroke:
    """笔"""
    index: int
    direction: str          # "up" / "down"
    begin_price: float
    end_price: float
    begin_time: str
    end_time: str
    is_sure: bool
    high: float
    low: float


@dataclass
class Segment:
    """线段"""
    index: int
    direction: str
    begin_price: float
    end_price: float
    begin_time: str
    end_time: str
    is_sure: bool
    stroke_count: int


@dataclass
class Pivot:
    """中枢"""
    index: int
    low: float              # ZD 中枢下沿
    high: float             # ZG 中枢上沿
    peak_low: float         # 最低点
    peak_high: float        # 最高点
    begin_time: str
    end_time: str
    is_sure: bool
    stroke_count: int       # 构成中枢的笔数


@dataclass
class BSPoint:
    """买卖点"""
    is_buy: bool
    bsp_types: List[str]    # ["1"], ["2", "3a"], etc.
    price: float
    time: str
    stroke_index: int
    is_sure: bool


@dataclass
class LevelResult:
    """单个级别的缠论分析结果"""
    level: str              # "daily", "60min", "weekly"
    strokes: List[Stroke]
    segments: List[Segment]
    pivots: List[Pivot]
    bs_points: List[BSPoint]
    latest_price: float
    latest_time: str


@dataclass
class ChanLunResult:
    """缠论多级别联立分析结果"""
    stock_code: str
    stock_name: str
    levels: List[LevelResult] = field(default_factory=list)

    @property
    def daily(self) -> Optional[LevelResult]:
        return next((r for r in self.levels if r.level == "daily"), None)

    @property
    def hourly(self) -> Optional[LevelResult]:
        return next((r for r in self.levels if r.level == "60min"), None) if self.levels else None


def _fmt_time(kl) -> str:
    try:
        # CKLine has time_begin/time_end, CKLine_Unit has time
        t = getattr(kl, 'time', None) or getattr(kl, 'time_begin', None)
        if t is None and hasattr(kl, 'lst') and kl.lst:
            t = kl.lst[-1].time
        if t is None:
            return ""
        if hasattr(t, 'hour') and t.hour > 0:
            return f"{t.year}-{t.month:02d}-{t.day:02d} {t.hour:02d}:{t.minute:02d}"
        return f"{t.year}-{t.month:02d}-{t.day:02d}"
    except Exception:
        return ""


def _dir_name(bi_dir) -> str:
    return "up" if "UP" in str(bi_dir) else "down"


def _bsp_type_names(bsp_types) -> List[str]:
    return [str(t).split(".")[-1] for t in bsp_types]


def _extract_level(chan: CChan, kl_type: KL_TYPE, level_name: str) -> LevelResult:
    kl_list = chan[kl_type]

    strokes = []
    for bi in kl_list.bi_list:
        strokes.append(Stroke(
            index=bi.idx,
            direction=_dir_name(bi.dir),
            begin_price=bi.get_begin_val(),
            end_price=bi.get_end_val(),
            begin_time=_fmt_time(bi.get_begin_klu()),
            end_time=_fmt_time(bi.get_end_klu()),
            is_sure=bi.is_sure,
            high=bi._high(),
            low=bi._low(),
        ))

    segments = []
    for seg in kl_list.seg_list:
        segments.append(Segment(
            index=seg.idx,
            direction=_dir_name(seg.dir),
            begin_price=seg.start_bi.get_begin_val() if seg.start_bi else 0,
            end_price=seg.end_bi.get_end_val() if seg.end_bi else 0,
            begin_time=_fmt_time(seg.get_begin_klu()),
            end_time=_fmt_time(seg.get_end_klu()),
            is_sure=seg.is_sure,
            stroke_count=len(seg.bi_list) if hasattr(seg, 'bi_list') else 0,
        ))

    pivots = []
    for zs in kl_list.zs_list:
        pivots.append(Pivot(
            index=zs.begin_bi.idx if zs.begin_bi else 0,
            low=zs.low,
            high=zs.high,
            peak_low=zs.peak_low,
            peak_high=zs.peak_high,
            begin_time=_fmt_time(zs.begin_bi.get_begin_klu()) if zs.begin_bi else "",
            end_time=_fmt_time(zs.end_bi.get_end_klu()) if zs.end_bi else "",
            is_sure=zs.is_sure,
            stroke_count=getattr(zs, '_stroke_count', 3),
        ))

    bs_points = []
    for bsp in kl_list.bs_point_lst.bsp_iter():
        bs_points.append(BSPoint(
            is_buy=bsp.is_buy,
            bsp_types=_bsp_type_names(bsp.type),
            price=bsp.klu.close if bsp.klu else 0,
            time=_fmt_time(bsp.klu) if bsp.klu else "",
            stroke_index=bsp.bi.idx if bsp.bi else 0,
            is_sure=bsp.bi.is_sure if bsp.bi else False,
        ))

    latest_kl = kl_list.lst[-1] if kl_list.lst else None
    return LevelResult(
        level=level_name,
        strokes=strokes,
        segments=segments,
        pivots=pivots,
        bs_points=bs_points,
        latest_price=latest_kl.lst[-1].close if latest_kl and latest_kl.lst else 0,
        latest_time=_fmt_time(latest_kl) if latest_kl else "",
    )


def _run_chan_single(code: str, kl_type: KL_TYPE, begin_date: str, end_date: str, config: CChanConfig) -> CChan:
    """运行单级别CChan分析，直接注入数据适配器"""
    adapter = CAkshareAPI(code=code, k_type=kl_type, begin_date=begin_date, end_date=end_date, autype=AUTYPE.QFQ)
    klu_iter = CChan.load_stock_data(None, adapter, kl_type)

    # 手动构建CChan，不走其data_src解析
    chan = object.__new__(CChan)
    chan.code = code
    chan.begin_time = begin_date
    chan.end_time = end_date
    chan.autype = AUTYPE.QFQ
    chan.data_src = None
    chan.lv_list = [kl_type]
    chan.conf = config
    chan.kl_misalign_cnt = 0
    chan.kl_inconsistent_detail = {}
    chan.g_kl_iter = {}
    chan.do_init()

    # 直接喂K线数据
    for klu in klu_iter:
        chan[kl_type].add_single_klu(klu)

    # 计算线段、中枢、买卖点
    for lv in chan.lv_list:
        chan.kl_datas[lv].cal_seg_and_zs()

    return chan


class ChanLunEngine:
    """缠论分析引擎"""

    def __init__(self, config: Optional[dict] = None):
        self._config = CChanConfig(config or {})

    def analyze(
        self,
        stock_code: str,
        levels: Optional[List[str]] = None,
        begin_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ChanLunResult:
        """
        执行缠论分析

        Parameters:
            stock_code: 6位股票代码，如 "603881"
            levels: 分析级别列表，如 ["daily", "60min"]
            begin_date: 起始日期 "2026-01-01"
            end_date: 结束日期 "2026-04-26"
        """
        levels = levels or ["daily"]
        level_map = {
            "1min": KL_TYPE.K_1M,
            "5min": KL_TYPE.K_5M,
            "15min": KL_TYPE.K_15M,
            "30min": KL_TYPE.K_30M,
            "60min": KL_TYPE.K_60M,
            "daily": KL_TYPE.K_DAY,
            "weekly": KL_TYPE.K_WEEK,
        }

        result = ChanLunResult(stock_code=stock_code, stock_name=stock_code)

        for level_name in levels:
            kl_type = level_map.get(level_name)
            if kl_type is None:
                continue
            chan = _run_chan_single(stock_code, kl_type, begin_date, end_date, self._config)
            result.levels.append(_extract_level(chan, kl_type, level_name))

        return result
