#!/usr/bin/env python3
"""事件驱动选股器。

从产业新闻/公告中识别重大事件，自动匹配受益链条，
结合技术面筛选出"有催化但还没被充分定价"的标的。

用法：
  # 手动输入事件分析
  python event_driven_screener.py --event "长鑫科技IPO重启，募资295亿扩产DRAM"

  # 分析指定事件+受益股
  python event_driven_screener.py --event "长鑫科技IPO" --stocks 603986,000753,000672,603626,002049

  # 扫描近期重大事件（从新闻源抓取）
  python event_driven_screener.py --scan
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

STOCK_ROOT = str(Path("."))
sys.path.insert(0, STOCK_ROOT)

from scripts.core.tushare_client import TushareClient


# ── 事件类型定义 ──────────────────────────────────────────

EVENT_TYPES = {
    "ipo": {
        "name": "IPO/上市",
        "benefit_chain": ["直接持股", "供应链(设备/材料)", "同赛道竞品", "下游客户"],
        "time_horizon": "中期(1-3月)",
        "typical_strength": 8,
    },
    "merger": {
        "name": "并购重组",
        "benefit_chain": ["被并购方", "同行业整合", "供应链", "竞争对手估值提升"],
        "time_horizon": "中期(1-6月)",
        "typical_strength": 7,
    },
    "policy": {
        "name": "政策利好",
        "benefit_chain": ["直接受益", "行业龙头", "产业链"],
        "time_horizon": "中短期(1-4周)",
        "typical_strength": 6,
    },
    "product": {
        "name": "产品/技术突破",
        "benefit_chain": ["自身", "供应链", "竞争对手跟风"],
        "time_horizon": "短期(1-2周)",
        "typical_strength": 5,
    },
    "earnings": {
        "name": "业绩爆发",
        "benefit_chain": ["自身", "同行业估值提升", "供应链"],
        "time_horizon": "短期(1-2周)",
        "typical_strength": 6,
    },
    "capacity": {
        "name": "扩产/投资",
        "benefit_chain": ["自身", "设备供应商", "材料供应商", "同赛道"],
        "time_horizon": "中期(3-6月)",
        "typical_strength": 7,
    },
    "supply_shortage": {
        "name": "供需失衡/涨价",
        "benefit_chain": ["产能最大者", "替代品", "上游原材料"],
        "time_horizon": "中短期(2-8周)",
        "typical_strength": 7,
    },
}

# ── 受益链知识库 ──────────────────────────────────────────

# 长鑫科技IPO受益链（示例，后续可扩展）
BENEFIT_CHAINS = {
    "长鑫科技": {
        "event_type": "ipo",
        "chains": {
            "直接持股": [
                {"code": "603986", "name": "兆易创新", "relation": "持股1.8%+董事长兼任", "benefit_score": 10},
                {"code": "000753", "name": "合肥城建", "relation": "间接持股", "benefit_score": 8},
                {"code": "000672", "name": "上峰水泥", "relation": "投资4亿(兰璞创投)", "benefit_score": 6},
                {"code": "603626", "name": "朗迪集团", "relation": "通过基金间接持股", "benefit_score": 5},
            ],
            "供应链(设备/材料)": [
                {"code": "002049", "name": "紫光国微", "relation": "存储芯片设计", "benefit_score": 7},
                {"code": "688396", "name": "华润微", "relation": "晶圆代工", "benefit_score": 6},
                {"code": "688012", "name": "中微公司", "relation": "刻蚀设备", "benefit_score": 8},
                {"code": "688200", "name": "华峰测控", "relation": "测试设备", "benefit_score": 6},
                {"code": "300346", "name": "南大光电", "relation": "光刻胶", "benefit_score": 6},
                {"code": "603698", "name": "航天工程", "relation": "气体供应", "benefit_score": 4},
            ],
            "同赛道竞品": [
                {"code": "002049", "name": "紫光国微", "relation": "国产存储芯片", "benefit_score": 5},
                {"code": "688981", "name": "中芯国际", "relation": "晶圆代工龙头", "benefit_score": 5},
            ],
            "下游客户": [
                {"code": "000977", "name": "浪潮信息", "relation": "服务器用DRAM", "benefit_score": 4},
                {"code": "000063", "name": "中兴通讯", "relation": "通信设备用DRAM", "benefit_score": 4},
            ],
        },
    },
}


class EventDrivenScreener:
    """事件驱动选股器"""

    def __init__(self):
        self.client = TushareClient()

    def analyze_event(self, event_text: str, stock_codes: list[str] | None = None) -> dict:
        """分析事件并匹配受益链。

        Args:
            event_text: 事件描述
            stock_codes: 可选，指定分析的股票代码列表
        """
        # 1. 识别事件类型
        event_type = self._classify_event(event_text)
        event_info = EVENT_TYPES.get(event_type, EVENT_TYPES["ipo"])

        # 2. 匹配受益链
        chain = self._match_benefit_chain(event_text)

        # 3. 如果有指定股票，分析它们在受益链中的位置
        if stock_codes:
            chain = self._analyze_stocks_in_chain(chain, stock_codes)

        # 4. 对受益股做技术面快速检查
        for chain_name, stocks in chain.get("chains", {}).items():
            for stock in stocks:
                tech = self._quick_tech_check(stock["code"])
                stock.update(tech)

        # 5. 综合评分
        results = self._score_and_rank(chain)

        return {
            "event": event_text,
            "event_type": event_type,
            "event_info": event_info,
            "benefit_chain": chain,
            "ranked_stocks": results,
        }

    def _classify_event(self, text: str) -> str:
        """简单的事件分类（关键词匹配）"""
        text = text.lower()
        if any(k in text for k in ["ipo", "上市", "招股", "审核"]):
            return "ipo"
        elif any(k in text for k in ["并购", "重组", "收购"]):
            return "merger"
        elif any(k in text for k in ["政策", "补贴", "扶持", "国策"]):
            return "policy"
        elif any(k in text for k in ["突破", "发布", "新品", "量产"]):
            return "product"
        elif any(k in text for k in ["业绩", "净利", "营收", "增长"]):
            return "earnings"
        elif any(k in text for k in ["扩产", "投资", "建厂", "募资"]):
            return "capacity"
        elif any(k in text for k in ["涨价", "短缺", "供需", "缺货"]):
            return "supply_shortage"
        return "ipo"  # default

    def _match_benefit_chain(self, event_text: str) -> dict:
        """从知识库匹配受益链"""
        for key, chain_data in BENEFIT_CHAINS.items():
            if key in event_text:
                return chain_data

        # 未匹配到，返回空结构
        return {
            "event_type": "unknown",
            "chains": {},
            "note": "未在知识库中找到匹配，需要手动指定受益链",
        }

    def _analyze_stocks_in_chain(self, chain: dict, codes: list[str]) -> dict:
        """分析指定股票在受益链中的位置"""
        # 如果chain已经有数据，保留
        if chain.get("chains"):
            return chain
        # 否则创建一个待分析列表
        chain["chains"] = {"待分析": [
            {"code": c, "name": "", "relation": "需确认", "benefit_score": 5}
            for c in codes
        ]}
        return chain

    def _quick_tech_check(self, code: str) -> dict:
        """快速技术面检查"""
        result = {
            "current_price": None,
            "rsi": None,
            "ma_trend": "未知",
            "near_high_pct": None,
            "tech_score": 50,
            "tech_note": "数据获取失败",
        }

        try:
            df = self.client.get_kline(code, freq="D", start="20260301", end="20260518")
            if df is None or len(df) < 30:
                return result

            df = df.sort_values("date")
            close = df["close"].astype(float)

            current = float(close.iloc[-1])
            period_high = float(close.max())
            period_low = float(close.min())

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = float(100 - 100 / (1 + gain.iloc[-1] / max(loss.iloc[-1], 0.01)))

            # MA趋势
            ma5 = float(close.rolling(5).mean().iloc[-1])
            ma10 = float(close.rolling(10).mean().iloc[-1])
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma_trend = "多头" if ma5 > ma10 > ma20 else ("空头" if ma5 < ma10 < ma20 else "震荡")

            # 距离阶段高点的百分比
            near_high_pct = round((current - period_high) / period_high * 100, 1)

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            macd_golden = bool(dif.iloc[-1] > dea.iloc[-1])

            # 技术面评分
            tech_score = 50
            if ma_trend == "多头":
                tech_score += 15
            elif ma_trend == "空头":
                tech_score -= 15
            if macd_golden:
                tech_score += 10
            else:
                tech_score -= 10
            if 40 <= rsi <= 60:
                tech_score += 15  # 最佳区间
            elif rsi > 70:
                tech_score -= 10  # 超买
            elif rsi < 30:
                tech_score += 5  # 超卖可能是机会

            # 离高点远加分（说明还没被炒过）
            if near_high_pct < -15:
                tech_score += 10  # 离高点很远=还没被炒
            elif near_high_pct > -3:
                tech_score -= 5  # 已经接近高点了

            tech_score = max(0, min(100, tech_score))

            result.update({
                "current_price": round(current, 2),
                "rsi": round(rsi, 1),
                "ma_trend": ma_trend,
                "near_high_pct": near_high_pct,
                "macd_golden": macd_golden,
                "tech_score": tech_score,
                "tech_note": f"{'金叉' if macd_golden else '死叉'} | RSI {rsi:.0f} | {ma_trend} | 距高点{near_high_pct}%",
            })

        except Exception as e:
            result["tech_note"] = f"检查失败: {e}"

        return result

    def _score_and_rank(self, chain: dict) -> list[dict]:
        """综合评分并排名"""
        all_stocks = []
        for chain_name, stocks in chain.get("chains", {}).items():
            for stock in stocks:
                benefit = stock.get("benefit_score", 5)
                tech = stock.get("tech_score", 50)

                # 综合分 = 事件受益分 × 0.5 + 技术面分 × 0.3 + 位置分 × 0.2
                position_score = 50
                near_high = stock.get("near_high_pct")
                if near_high is not None:
                    if near_high < -15:
                        position_score = 90  # 离高点很远=空间大
                    elif near_high < -5:
                        position_score = 70
                    elif near_high < 0:
                        position_score = 50
                    else:
                        position_score = 30  # 已在高点

                composite = benefit * 0.5 + tech * 0.3 + position_score * 0.2

                all_stocks.append({
                    **stock,
                    "chain_position": chain_name,
                    "position_score": position_score,
                    "composite_score": round(composite, 1),
                    "recommendation": self._make_recommendation(benefit, tech, position_score),
                })

        all_stocks.sort(key=lambda x: x["composite_score"], reverse=True)
        return all_stocks

    def _make_recommendation(self, benefit: int, tech: int, position: int) -> str:
        """生成推荐"""
        if benefit >= 8 and tech >= 60 and position >= 60:
            return "⭐⭐⭐ 强烈关注"
        elif benefit >= 6 and tech >= 50:
            return "⭐⭐ 值得关注"
        elif benefit >= 5 and tech >= 40:
            return "⭐ 可以观察"
        elif tech < 30:
            return "⚠️ 技术面偏弱，等企稳"
        elif position < 40:
            return "⚠️ 已在高点，追高风险"
        else:
            return "观望"


def format_event_report(results: dict) -> str:
    """格式化事件驱动分析报告"""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  📰 事件驱动选股分析")
    lines.append(f"  事件: {results['event']}")
    lines.append(f"  类型: {results['event_info']['name']} | {results['event_info']['time_horizon']}")
    lines.append("=" * 65)
    lines.append("")

    # 受益链
    chain = results.get("benefit_chain", {})
    lines.append("  📋 受益链条:")
    for chain_name, stocks in chain.get("chains", {}).items():
        lines.append(f"    【{chain_name}】")
        for s in stocks:
            lines.append(f"      - {s.get('name','')}({s['code']}) | {s.get('relation','')} | 受益分{s.get('benefit_score','?')}")
    lines.append("")

    # 排名结果
    ranked = results.get("ranked_stocks", [])
    if ranked:
        lines.append("  🏆 综合排名（受益分×0.5 + 技术面×0.3 + 位置×0.2）:")
        lines.append(f"  {'排名':4} {'代码':8} {'名称':8} {'受益链':12} {'综合分':6} {'现价':8} {'距高点':8} {'技术面':30} {'推荐'}")
        lines.append("  " + "-" * 100)

        for i, s in enumerate(ranked, 1):
            price = f"{s.get('current_price', '?')}" if s.get('current_price') else "?"
            near = f"{s.get('near_high_pct', '?')}%" if s.get('near_high_pct') is not None else "?"
            lines.append(
                f"  {i:4} {s['code']:8} {str(s.get('name',''))[:8]:8} "
                f"{str(s.get('chain_position',''))[:12]:12} "
                f"{s.get('composite_score',0):6.1f} "
                f"{price:>8} {near:>8} "
                f"{str(s.get('tech_note',''))[:30]:30} "
                f"{s.get('recommendation','')}"
            )

    lines.append("")
    lines.append("  💡 说明:")
    lines.append("    - 距高点越远=还没被充分定价=空间越大")
    lines.append("    - 技术面金叉+多头=时机好")
    lines.append("    - 受益分高+技术面好+位置低=最佳标的")
    lines.append("=" * 65)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="事件驱动选股器")
    parser.add_argument("--event", type=str, help="事件描述")
    parser.add_argument("--stocks", type=str, help="股票代码，逗号分隔")
    parser.add_argument("--scan", action="store_true", help="扫描近期重大事件")
    parser.add_argument("--json", action="store_true", help="输出JSON")
    args = parser.parse_args()

    screener = EventDrivenScreener()

    if args.event:
        codes = [c.strip() for c in args.stocks.split(",")] if args.stocks else None
        results = screener.analyze_event(args.event, stock_codes=codes)

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        else:
            print(format_event_report(results))
    elif args.scan:
        print("⚠️ 新闻扫描功能待实现（需要接入新闻API）")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
