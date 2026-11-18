#!/usr/bin/env python3
"""缠论分析报告生成器 — 从ChanLunResult生成文字分析报告"""
from __future__ import annotations

from typing import Optional

from .engine import ChanLunResult, LevelResult, Stroke, Segment, Pivot, BSPoint


def _dir_cn(d: str) -> str:
    return "向上" if d == "up" else "向下"


def _bsp_cn(types: list) -> str:
    name_map = {"T1": "一买", "T1P": "一买'", "T2": "二买", "T2S": "二买'",
                "T3A": "三买a", "T3B": "三买b"}
    return "/".join(name_map.get(t, t) for t in types)


def generate_level_report(lr: LevelResult) -> str:
    """生成单级别分析报告"""
    lines = []
    level_cn = {"daily": "日线", "weekly": "周线", "60min": "60分钟", "30min": "30分钟", "15min": "15分钟", "5min": "5分钟", "1min": "1分钟"}
    lv_name = level_cn.get(lr.level, lr.level)

    lines.append(f"### {lv_name}缠论分析")
    lines.append(f"- 最新价: {lr.latest_price:.2f} ({lr.latest_time})")
    lines.append(f"- 笔: {len(lr.strokes)}根 | 线段: {len(lr.segments)}段 | 中枢: {len(lr.pivots)}个 | 买卖点: {len(lr.bs_points)}个")

    # 当前笔状态
    if lr.strokes:
        last = lr.strokes[-1]
        sure_tag = "已确认" if last.is_sure else "未确认(形成中)"
        lines.append(f"- 当前笔: {_dir_cn(last.direction)} {last.begin_price:.2f}→{last.end_price:.2f} ({sure_tag})")
        if not last.is_sure:
            lines.append(f"  ⚠ 笔未确认，可能随时反转")

    # 最近中枢
    if lr.pivots:
        zs = lr.pivots[-1]
        pos = "上方" if lr.latest_price > zs.high else "下方" if lr.latest_price < zs.low else "内部"
        lines.append(f"- 最近中枢: [{zs.low:.2f}, {zs.high:.2f}] 当前价格在{pos}")

        dist_up = (zs.high / lr.latest_price - 1) * 100
        dist_down = (lr.latest_price / zs.low - 1) * 100
        if pos == "下方":
            lines.append(f"  距中枢上沿 +{dist_up:.1f}%")
        elif pos == "上方":
            lines.append(f"  距中枢下沿 -{dist_down:.1f}%")

    # 最近买卖点
    if lr.bs_points:
        recent = lr.bs_points[-5:]
        lines.append(f"- 最近买卖点:")
        for bsp in reversed(recent):
            tag = "🟢买" if bsp.is_buy else "🔴卖"
            sure_tag = "" if bsp.is_sure else "(未确认)"
            lines.append(f"  {tag} {_bsp_cn(bsp.bsp_types)} @ {bsp.price:.2f} ({bsp.time}){sure_tag}")

    return "\n".join(lines)


def generate_chanlun_report(result: ChanLunResult, stock_name: str = "") -> str:
    """生成完整的缠论多级别分析报告"""
    lines = []
    name = stock_name or result.stock_code
    lines.append(f"# {name}({result.stock_code}) 缠论分析")
    lines.append("")

    for lr in result.levels:
        lines.append(generate_level_report(lr))
        lines.append("")

    # 多级别综合判断
    lines.append("### 综合判断")
    daily = result.daily
    hourly = result.hourly

    if daily and daily.pivots and daily.strokes:
        zs = daily.pivots[-1]
        bi = daily.strokes[-1]
        if bi.direction == "up" and bi.is_sure and lr.latest_price > zs.high:
            lines.append("- 日线笔向上突破中枢，偏多")
        elif bi.direction == "down" and bi.is_sure and lr.latest_price < zs.low:
            lines.append("- 日线笔向下突破中枢，偏空")
        elif zs.low <= lr.latest_price <= zs.high:
            lines.append("- 日线价格在中枢内震荡，观望")

    if hourly and hourly.bs_points:
        last_bsp = hourly.bs_points[-1]
        tag = "买入" if last_bsp.is_buy else "卖出"
        lines.append(f"- 60分钟最新信号: {tag}({_bsp_cn(last_bsp.bsp_types)})")

    return "\n".join(lines)
