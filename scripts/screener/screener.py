#!/usr/bin/env python3
"""选股子系统主编排器。

四层漏斗: universe_filter → factor_scorer → (tech_scanner) → 排序输出
内置预设模板，支持概念板块筛选和参数覆盖。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from scripts.core.tushare_client import TushareClient, from_ts_code
from scripts.screener.universe_filter import UniverseFilter
from scripts.screener.factor_scorer import FactorScorer
from scripts.screener.wave_screener import WaveScreener


# ── 预设模板 ──────────────────────────────────────────────────────────

SCREENER_TEMPLATES = {
    "growth": {
        "name": "成长股",
        "desc": "高营收增速、合理估值",
        "factor_weights": {
            "growth": 0.35, "quality": 0.25, "value": 0.15,
            "momentum": 0.15, "theme": 0.10,
        },
        "filters": {
            "rev_growth_min": 20,
            "roe_min": 8,
            "pe_max": 80,
            "debt_max": 60,
        },
    },
    "value": {
        "name": "价值股",
        "desc": "低估值+稳定盈利",
        "factor_weights": {
            "value": 0.35, "quality": 0.30, "growth": 0.15,
            "momentum": 0.10, "theme": 0.10,
        },
        "filters": {
            "pe_max": 25,
            "pb_max": 3,
            "roe_min": 10,
            "debt_max": 50,
        },
    },
    "momentum": {
        "name": "强势股",
        "desc": "近期涨幅大+资金关注",
        "factor_weights": {
            "momentum": 0.35, "growth": 0.20, "quality": 0.20,
            "value": 0.15, "theme": 0.10,
        },
        "filters": {
            "mcap_min": 50,
            "turnover_min": 3,
        },
    },
    "davis_double": {
        "name": "戴维斯双击",
        "desc": "高增长+低估值，业绩和估值双升",
        "factor_weights": {
            "growth": 0.30, "value": 0.30, "quality": 0.20,
            "momentum": 0.10, "theme": 0.10,
        },
        "filters": {
            "rev_growth_min": 25,
            "pe_max": 40,
        },
    },
    "turnaround": {
        "name": "困境反转",
        "desc": "业绩触底回升+技术面企稳",
        "factor_weights": {
            "growth": 0.25, "momentum": 0.25, "quality": 0.20,
            "value": 0.20, "theme": 0.10,
        },
        "filters": {
            "pe_max": 60,
        },
    },
    "ai_focus": {
        "name": "AI产业链",
        "desc": "AI主线+基本面确认",
        "factor_weights": {
            "theme": 0.30, "growth": 0.30, "quality": 0.20,
            "value": 0.10, "momentum": 0.10,
        },
        "filters": {
            "rev_growth_min": 15,
            "mcap_min": 30,
        },
    },
    "wave3": {
        "name": "W3浪主升",
        "desc": "Elliott Wave第三浪主升段，动量+基本面确认",
        "factor_weights": {
            "growth": 0.25, "quality": 0.20, "value": 0.15,
            "momentum": 0.25, "theme": 0.15,
        },
        "filters": {
            "mcap_min": 30,
            "pe_max": 100,
            "turnover_min": 2,
        },
    },
    "wave3_startup": {
        "name": "W3刚启动",
        "desc": "三系统共振：波浪W3刚启动+缠论二买+MACD看涨确认",
        "factor_weights": {
            "growth": 0.20, "quality": 0.15, "value": 0.15,
            "momentum": 0.30, "theme": 0.20,
        },
        "filters": {
            "mcap_min": 30,
            "pe_max": 100,
            "turnover_min": 2,
        },
    },
    "trend_scan": {
        "name": "趋势扫描",
        "desc": "全市场趋势扫描：多头排列+MACD金叉+量价配合+周线确认",
        "factor_weights": {
            "momentum": 0.40, "growth": 0.20, "quality": 0.15,
            "value": 0.10, "theme": 0.15,
        },
        "filters": {
            "mcap_min": 20,
            "turnover_min": 1,
        },
    },
    "chanlun_t3": {
        "name": "缠论三买点",
        "desc": "三买点扫描：离开中枢+回调不破中枢+MACD确认+量价配合",
        "factor_weights": {
            "momentum": 0.35, "growth": 0.20, "quality": 0.15,
            "value": 0.15, "theme": 0.15,
        },
        "filters": {
            "mcap_min": 20,
            "turnover_min": 1,
        },
    },
    "fusion": {
        "name": "信号融合",
        "desc": "多策略交叉验证：W3启动+缠论T3+趋势扫描+底部反转融合评分",
        "factor_weights": {
            "momentum": 0.30, "growth": 0.20, "quality": 0.15,
            "value": 0.15, "theme": 0.20,
        },
        "filters": {
            "mcap_min": 20,
            "turnover_min": 1,
        },
    },
    "bottom_reversal": {
        "name": "底部反转",
        "desc": "牛股启动前特征研究：极度超卖+缩量+近低点+基本面不差+反转信号",
        "factor_weights": {
            "growth": 0.10, "quality": 0.15, "value": 0.25,
            "momentum": 0.20, "theme": 0.30,
        },
        "filters": {
            "mcap_min": 20,
            "pe_max": 100,
        },
    },
    "wave_position": {
        "name": "波浪位置",
        "desc": "下跌5浪结束后W2回调底部/W3刚启动买点（子模型）",
        "factor_weights": {
            "momentum": 0.30, "quality": 0.20, "growth": 0.15,
            "value": 0.15, "theme": 0.20,
        },
        "filters": {
            "mcap_min": 20,
        },
    },
    "event_driven": {
        "name": "事件驱动",
        "desc": "产业事件催化选股：IPO/并购/政策/扩产等重大事件受益链条分析",
        "factor_weights": {
            "theme": 0.40, "growth": 0.20, "quality": 0.15,
            "value": 0.10, "momentum": 0.15,
        },
        "filters": {
            "mcap_min": 20,
        },
    },
}


class StockScreener:
    """选股子系统"""

    def __init__(self, token: Optional[str] = None):
        self.client = TushareClient(token)
        self.universe = UniverseFilter(self.client)
        self.scorer = FactorScorer(self.client)

    def run(
        self,
        template: Optional[str] = None,
        concept: Optional[str] = None,
        concept_id: Optional[str] = None,
        industry: Optional[str] = None,
        min_mcap: float = 20,
        max_mcap: Optional[float] = None,
        min_score: float = 70,
        top: int = 20,
        quick: bool = False,
        override_filters: Optional[dict] = None,
        min_w3_gain: float = 10.0,
        min_w3_confidence: float = 50.0,
        as_of_date: Optional[str] = None,
    ) -> dict:
        """执行选股流程。

        Args:
            template: 模板名称
            concept: 概念板块关键词
            concept_id: 精确概念板块ID
            industry: 行业名称
            min_mcap: 最小市值（亿元）
            max_mcap: 最大市值（亿元）
            min_score: 最低综合分
            top: 返回前N只
            quick: 快速模式
            override_filters: 覆盖模板过滤条件
            min_w3_gain: W3最低涨幅%(wave3模板)
            min_w3_confidence: W3最低置信度(wave3模板)
            as_of_date: 历史截断日期YYYYMMDD（回测用，None=当天）

        Returns:
            dict: {candidates, results, meta}
        """
        # 加载模板
        tpl = SCREENER_TEMPLATES.get(template, {}) if template else {}
        factor_weights = tpl.get("factor_weights")
        filters = {**tpl.get("filters", {}), **(override_filters or {})}

        # AI产业链模板自动设概念
        if template == "ai_focus" and not concept:
            concept = "AI"

        print(f"\n{'='*60}")
        print(f"  选股子系统")
        print(f"  模板: {tpl.get('name', '自定义')} | "
              f"概念: {concept or '全市场'} | "
              f"行业: {industry or '不限'}")
        print(f"{'='*60}\n")

        # 大盘环境检测（软参考）
        regime_info = None
        if not as_of_date:
            try:
                from scripts.market_regime.regime_detector import MarketRegimeDetector
                regime_detector = MarketRegimeDetector()
                regime_info = regime_detector.detect()
                regime_label = {
                    "risk_on": "多头🟢", "neutral": "震荡🟡", "risk_off": "空头🔴",
                }.get(regime_info["regime"], regime_info["regime"])
                print(f"  大盘环境: {regime_label} "
                      f"(环境分{regime_info['score']:+.2f}, "
                      f"置信度{regime_info['confidence']:.0%})")
                print()
            except Exception:
                pass

        # Layer 1: 基础过滤
        step_label = "[1/3]" if template in ("wave3", "wave3_startup", "bottom_reversal", "trend_scan", "chanlun_t3", "wave_position") else "[1/2]"
        print(f"{step_label} 基础过滤...")
        candidates = self.universe.filter(
            concept=concept,
            concept_id=concept_id,
            industry=industry,
            min_mcap=min_mcap,
            max_mcap=max_mcap,
            trade_date=as_of_date,
        )
        print(f"  候选池: {len(candidates)} 只")

        if candidates.empty:
            return {"candidates": candidates, "results": pd.DataFrame(),
                    "meta": {"error": "Layer 1过滤后无候选股"}}

        # Layer 1.5: 技术形态检测（wave3/wave3_startup/bottom_reversal模板）
        wave_results = None
        if template == "wave3":
            print("[1.5/3] W3浪检测...")
            wave_scanner = WaveScreener(self.client)
            wave_results = wave_scanner.scan(
                candidates,
                lookback_days=500,
                min_confidence=min_w3_confidence,
                min_w3_gain=min_w3_gain,
                top=top * 3,
                as_of_date=as_of_date,
            )
            if wave_results.empty:
                print("  未找到W3候选股")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            candidates = wave_results
            print(f"  W3候选: {len(candidates)} 只")

        elif template == "wave3_startup":
            print("[1.5/3] W3刚启动检测（三系统共振）...")
            from scripts.screener.wave3_startup_screener import Wave3StartupScreener
            startup_scanner = Wave3StartupScreener(self.client)
            wave_results = startup_scanner.scan(
                candidates,
                lookback_days=500,
                min_confidence=min_w3_confidence,
                top=top * 3,
                as_of_date=as_of_date,
            )
            if wave_results.empty:
                print("  未找到W3刚启动候选股")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            candidates = wave_results
            print(f"  W3刚启动候选: {len(candidates)} 只")

        elif template == "bottom_reversal":
            print("[1.5/3] 底部反转检测（牛股启动前特征）...")
            from scripts.screener.bottom_reversal_screener import BottomReversalScreener
            br_scanner = BottomReversalScreener(self.client)
            wave_results = br_scanner.scan(
                candidates,
                lookback_days=200,
                min_confidence=40.0,
                top=top * 3,
                as_of_date=as_of_date,
            )
            if wave_results.empty:
                print("  未找到底部反转候选股")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            candidates = wave_results
            print(f"  底部反转候选: {len(candidates)} 只")

        elif template == "trend_scan":
            print("[1.5/3] 全市场趋势扫描（多头排列+MACD+量价）...")
            from scripts.screener.trend_scanner import TrendScanner
            trend_scanner = TrendScanner(self.client)
            wave_results = trend_scanner.scan(
                candidates,
                lookback_days=120,
                min_grade="A",
                top=top * 3,
                as_of_date=as_of_date,
                concept=concept,
                industry=industry,
            )
            if wave_results.empty:
                print("  未找到上升趋势股票")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            candidates = wave_results
            print(f"  趋势候选: {len(candidates)} 只")

        elif template == "chanlun_t3":
            print("[1.5/3] 缠论三买点扫描（离开中枢+回调不破+MACD确认）...")
            from scripts.screener.chanlun_t3_screener import ChanLunT3Screener
            t3_scanner = ChanLunT3Screener(self.client)
            wave_results = t3_scanner.scan(
                candidates,
                lookback_days=200,
                min_confidence=50.0,
                top=top * 3,
                as_of_date=as_of_date,
            )
            if wave_results.empty:
                print("  未找到缠论三买点股票")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            candidates = wave_results
            print(f"  三买点候选: {len(candidates)} 只")

        elif template == "fusion":
            print("[1.5/3] 多策略信号融合扫描...")
            from scripts.screener.signal_fusion import SignalFusion
            fusion = SignalFusion(self.client)
            wave_results = fusion.scan(
                candidates,
                as_of_date=as_of_date,
                concept=concept,
                industry=industry,
                min_fusion_score=min_score,
                top=top,
                lookback_days=200,
            )
            if wave_results.empty:
                print("  融合无结果")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            candidates = wave_results
            print(f"  融合候选: {len(candidates)} 只")

        elif template == "wave_position":
            print("[1.5/3] 波浪位置子模型扫描（5浪下跌→W2/W3买点）...")
            from pathlib import Path as _Path
            from scripts.screener.wave_position_screener import WavePositionScreener
            data_root = _Path("data/store")
            wp_scanner = WavePositionScreener(data_root=data_root)
            codes = candidates["ts_code"].apply(lambda x: x[:6] if "." in str(x) else str(x)).tolist()
            wave_results = wp_scanner.scan(codes, min_score=30, top=top, as_of_date=as_of_date)
            if wave_results.empty:
                print("  未找到波浪位置候选股")
                return {"candidates": candidates, "results": pd.DataFrame(),
                        "meta": {"template": template, "candidates_count": len(candidates),
                                 "results_count": 0, "trade_date": self.client.get_recent_trade_day()}}
            # wave_position返回code列(纯数字)，补ts_code列供format_report使用
            wave_results["ts_code"] = wave_results["code"].apply(
                lambda c: f"{c}.SH" if c.startswith("6") else f"{c}.SZ"
            )
            # 补齐format_report/save_report需要的通用列
            wave_results["name"] = ""
            wave_results["composite"] = wave_results["composite_score"]
            wave_results["growth_score"] = 0
            wave_results["quality_score"] = 0
            wave_results["value_score"] = 0
            wave_results["momentum_score"] = 0
            wave_results["theme_score"] = 0
            candidates = wave_results
            print(f"  波浪位置候选: {len(candidates)} 只")

        elif template == "event_driven":
            # 事件驱动模板：使用独立的事件驱动选股器
            print("[1/2] 事件驱动选股...")
            print("  ⚠️ event_driven 模板请使用 scripts/screener/event_driven_screener.py 单独运行")
            print("  用法: python scripts/screener/event_driven_screener.py --event \"事件描述\"")
            return {"candidates": pd.DataFrame(), "results": pd.DataFrame(),
                    "meta": {"template": template, "note": "请使用 event_driven_screener.py"}}

        # Layer 2: 多因子评分
        # 有Layer 1.5技术形态筛选的模板，直接用技术面得分排序，不走多因子
        has_layer15 = template in ("wave3", "wave3_startup", "bottom_reversal", "trend_scan", "chanlun_t3", "wave_position", "fusion", "event_driven")
        step2_label = "[2/3]" if has_layer15 else "[2/2]"

        if has_layer15:
            # 技术形态模板：Layer 1.5已有专项评分，直接排序输出
            print(f"{step2_label} 技术形态自带评分，跳过多因子评分...")
            results = candidates
            if template == "bottom_reversal" and "br_score" in results.columns:
                results = results.sort_values("br_score", ascending=False)
            elif template == "wave3_startup" and "w3s_composite" in results.columns:
                results = results.sort_values("w3s_composite", ascending=False)
            elif template == "wave3" and "w3_gain_pct" in results.columns:
                results = results.sort_values("w3_gain_pct", ascending=False)
            elif template == "trend_scan" and "trend_strength" in results.columns:
                results = results.sort_values("trend_strength", ascending=False)
            elif template == "chanlun_t3" and "t3_composite" in results.columns:
                results = results.sort_values("t3_composite", ascending=False)
            elif template == "fusion" and "fusion_score" in results.columns:
                results = results.sort_values("fusion_score", ascending=False)
            elif template == "wave_position" and "composite_score" in results.columns:
                results = results.sort_values("composite_score", ascending=False)
        elif quick:
            print(f"{step2_label} 快速评分（仅行情数据）...")
            results = self._quick_score(candidates, factor_weights, min_score)
        else:
            print(f"{step2_label} 多因子评分...")
            results = self.scorer.score(
                candidates,
                weights=factor_weights,
                min_score=min_score,
                hard_filters=filters,
            )

        print(f"  入选: {len(results)} 只")

        # 取TOP N
        results = results.head(top)

        # 输出
        meta = {
            "template": template,
            "template_name": tpl.get("name", "自定义"),
            "concept": concept,
            "industry": industry,
            "min_mcap": min_mcap,
            "min_score": min_score,
            "candidates_count": len(candidates),
            "results_count": len(results),
            "quick": quick,
            "trade_date": as_of_date or self.client.get_recent_trade_day(),
            "wave_filter_count": len(wave_results) if wave_results is not None else None,
            "wave_avg_gain": float(wave_results['w3_gain_pct'].mean()) if wave_results is not None and 'w3_gain_pct' in wave_results.columns and len(wave_results) > 0 else None,
            "startup_avg_composite": float(wave_results['w3s_composite'].mean()) if wave_results is not None and 'w3s_composite' in wave_results.columns and len(wave_results) > 0 else None,
            "br_avg_score": float(wave_results['br_score'].mean()) if wave_results is not None and 'br_score' in wave_results.columns and len(wave_results) > 0 else None,
            "market_regime": regime_info,
        }

        # 环境自适应建议（在meta定义之后）
        if regime_info is not None:
            try:
                from scripts.market_regime.regime_advisor import RegimeAdvisor
                advisor = RegimeAdvisor()
                regime = regime_info.get("regime", "neutral")
                regime_config = advisor.get_config(regime)
                meta["regime_advice"] = regime_config
                print(f"  💡 环境建议: {regime_config['label']} → "
                      f"推荐 {regime_config['recommended_templates'][:2]}, "
                      f"单仓≤{regime_config['max_position_per_stock']*100:.0f}%, "
                      f"持有{regime_config['preferred_hold_days']}天")
            except Exception:
                pass

        return {"candidates": candidates, "results": results, "meta": meta}

    def _quick_score(self, candidates: pd.DataFrame,
                     weights: Optional[dict],
                     min_score: float) -> pd.DataFrame:
        """快速模式：仅用daily_basic数据评分（无逐只财务查询）"""
        df = candidates.copy()
        w = weights or {}

        # 补充 daily_basic 数据（如果 candidates 中没有 pe_ttm/total_mv 等）
        if "pe_ttm" not in df.columns or "total_mv" not in df.columns:
            trade_date = self.client.get_recent_trade_day()
            try:
                basic = self.client.get_daily_basic(trade_date=trade_date)
                if basic is not None and not basic.empty:
                    merge_cols = ["ts_code"]
                    for col in ["pe_ttm", "pb", "total_mv", "turnover_rate", "circ_mv"]:
                        if col in basic.columns:
                            merge_cols.append(col)
                    if len(merge_cols) > 1:
                        df = df.merge(basic[merge_cols], on="ts_code", how="left")
            except Exception:
                pass

        # 价值评分（PE/PB从daily_basic已有）
        df["value_score"] = df.apply(self._quick_value_score, axis=1)

        # 市值评分作为质量代理
        if "total_mv" in df.columns:
            df["quality_score"] = df["total_mv"].apply(self._quick_quality_score)
        else:
            df["quality_score"] = 50.0

        # 成长/动量/主线给默认中间分
        # NOTE: 快速模式使用固定中位分数，仅用于初步粗筛，不可用于最终选股决策
        df["growth_score"] = 50.0
        df["momentum_score"] = 50.0
        df["theme_score"] = 30.0

        gw = w.get("growth", 0.30)
        qw = w.get("quality", 0.25)
        vw = w.get("value", 0.20)
        mw = w.get("momentum", 0.15)
        tw = w.get("theme", 0.10)

        df["composite"] = (
            df["growth_score"] * gw
            + df["quality_score"] * qw
            + df["value_score"] * vw
            + df["momentum_score"] * mw
            + df["theme_score"] * tw
        )

        return df[df["composite"] >= min_score].sort_values(
            "composite", ascending=False).reset_index(drop=True)

    @staticmethod
    def _quick_value_score(row) -> float:
        pe = FactorScorer._safe_float(row.get("pe_ttm"))
        pb = FactorScorer._safe_float(row.get("pb"))
        if pe <= 0:
            pe_s = 10
        elif pe <= 15:
            pe_s = 50
        elif pe <= 25:
            pe_s = 55
        elif pe <= 40:
            pe_s = 40
        elif pe <= 60:
            pe_s = 25
        else:
            pe_s = 5
        if pb <= 0:
            pb_s = 10
        elif pb <= 1:
            pb_s = 40
        elif pb <= 2:
            pb_s = 45
        elif pb <= 3:
            pb_s = 35
        elif pb <= 5:
            pb_s = 20
        else:
            pb_s = 10
        return min(100, pe_s + pb_s)

    @staticmethod
    def _quick_quality_score(total_mv: float) -> float:
        """市值代理质量评分"""
        if total_mv <= 0:
            return 20.0
        mv_yi = total_mv / 1e4  # 转为亿元
        if mv_yi < 30:
            return 25.0
        elif mv_yi < 50:
            return 35.0
        elif mv_yi < 100:
            return 45.0
        elif mv_yi < 300:
            return 55.0
        elif mv_yi < 1000:
            return 65.0
        else:
            return 75.0

    @staticmethod
    def list_templates() -> dict:
        """列出所有预设模板"""
        return {
            key: {"name": val["name"], "desc": val["desc"]}
            for key, val in SCREENER_TEMPLATES.items()
        }

    def list_concepts(self, keyword: Optional[str] = None) -> pd.DataFrame:
        """列出概念板块"""
        concepts = self.client.get_concepts()
        if concepts.empty:
            return concepts
        if keyword:
            return concepts[concepts["name"].str.contains(
                keyword, case=False, na=False)]
        return concepts

    def format_report(self, results: pd.DataFrame, meta: dict) -> str:
        """格式化控制台报告"""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"  选股报告  {meta.get('trade_date', '')}")
        lines.append(f"  策略: {meta.get('template_name', '自定义')} | "
                      f"候选池: {meta.get('candidates_count', 0)}只 → "
                      f"入选: {meta.get('results_count', 0)}只")
        lines.append(f"{'='*70}")

        if results.empty:
            lines.append("\n  无符合条件的股票")
            return "\n".join(lines)

        # 表头
        lines.append(f"\n{'排名':>4}  {'代码':<8} {'名称':<10} "
                      f"{'综合分':>6} {'成长':>5} {'质量':>5} "
                      f"{'价值':>5} {'动量':>5} {'主线':>5}")

        lines.append(f"{'─'*65}")

        for i, row in results.iterrows():
            name = str(row.get("name", ""))[:8]
            lines.append(
                f"{i+1:>4}  {from_ts_code(row['ts_code']):<8} {name:<10} "
                f"{row.get('composite', 0):>6.1f} "
                f"{row.get('growth_score', 0):>5.0f} "
                f"{row.get('quality_score', 0):>5.0f} "
                f"{row.get('value_score', 0):>5.0f} "
                f"{row.get('momentum_score', 0):>5.0f} "
                f"{row.get('theme_score', 0):>5.0f}"
            )

        lines.append(f"\n  注意: 以上为量化筛选结果，不构成投资建议。")
        lines.append(f"  买入前请用技术分析工具进一步确认。\n")

        return "\n".join(lines)

    def save_report(self, results: pd.DataFrame, meta: dict) -> str:
        """保存专业选股报告到 reports/choice/{关键词}_{date}.md"""
        from pathlib import Path

        today = datetime.now().strftime("%Y%m%d")
        date_str = f"{today[:4]}-{today[4:6]}-{today[6:8]}"

        # 文件名关键词：concept > industry > template_name
        keyword = meta.get("concept") or meta.get("industry") or meta.get("template_name", "选股")

        report_dir = Path("reports") / "choice"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{keyword}_{today}.md"

        content = self._generate_detailed_report(results, meta, date_str, keyword)
        report_path.write_text(content, encoding="utf-8")
        return str(report_path)

    def _generate_detailed_report(self, results: pd.DataFrame,
                                   meta: dict, date_str: str,
                                   keyword: str) -> str:
        """生成详细报告内容（结论先行）"""
        import numpy as np

        lines = []

        # ── 标题 ──
        lines.append(f"# {keyword}选股报告")
        lines.append("")
        scope = meta.get("concept") or meta.get("industry") or "全市场"
        lines.append(f"**日期**: {date_str} | **策略**: {meta.get('template_name', '自定义')} | **范围**: {scope}")
        lines.append(f"**筛选结果**: 候选池{meta.get('candidates_count', 0)}只 "
                      f"→ 入选{meta.get('results_count', 0)}只（综合分≥{meta.get('min_score', 70)}）")
        lines.append("")
        lines.append("---")
        lines.append("")

        if results.empty:
            lines.append("## 结论速览")
            lines.append("")
            lines.append("无符合条件的股票。")
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("*报告由选股子系统自动生成。不构成投资建议。*")
            return "\n".join(lines)

        # ── 结论速览 ──
        lines.append("## 结论速览")
        lines.append("")

        # TOP 3
        top3 = results.head(3)
        top3_str = " | ".join(
            f"**{i+1}** {row['name']}({from_ts_code(row['ts_code'])}) "
            f"{row.get('composite', 0):.1f}分"
            for i, row in top3.iterrows()
        )
        lines.append(f"**TOP 3**: {top3_str}")
        lines.append("")

        # 共性特征
        avg_growth = results["growth_score"].mean()
        avg_value = results["value_score"].mean()
        avg_momentum = results["momentum_score"].mean()
        avg_quality = results["quality_score"].mean()
        traits = []
        if avg_growth >= 65:
            traits.append("高成长")
        elif avg_growth >= 45:
            traits.append("中等成长")
        else:
            traits.append("成长性偏弱")

        if avg_value >= 65:
            traits.append("估值较低")
        elif avg_value >= 40:
            traits.append("估值适中")
        else:
            traits.append("估值偏高")

        if avg_momentum >= 65:
            traits.append("动量向上")
        elif avg_momentum >= 45:
            traits.append("动量平稳")
        else:
            traits.append("动量不足")

        if avg_quality >= 65:
            traits.append("基本面优良")

        lines.append(f"**共性特征**: {'+'.join(traits)}")
        lines.append("")

        # 风险提示
        pe_vals = results["pe_ttm"].dropna()
        if not pe_vals.empty and pe_vals.mean() > 60:
            lines.append("**风险提示**: 多数标的PE偏高，注意估值回调风险")
        elif not pe_vals.empty and pe_vals.mean() < 0:
            lines.append("**风险提示**: 部分标的处于亏损状态，关注盈利持续性")
        else:
            lines.append("**风险提示**: 筛选为量化结果，买入前务必用技术分析确认时机")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── 一、入选股全景 ──
        lines.append("## 一、入选股全景")
        lines.append("")
        lines.append("| 排名 | 代码 | 名称 | 综合分 | 成长 | 质量 | 价值 | 动量 | 主线 | PE(TTM) | PB | 市值(亿) |")
        lines.append("|------|------|------|--------|------|------|------|------|------|---------|-----|---------|")

        for i, row in results.iterrows():
            code = from_ts_code(row["ts_code"])
            name = str(row.get("name", ""))
            pe = row.get("pe_ttm")
            pb = row.get("pb")
            mv = row.get("total_mv", 0) / 1e4 if row.get("total_mv") else 0
            pe_str = f"{pe:.1f}" if pe and not np.isnan(pe) else "N/A"
            pb_str = f"{pb:.1f}" if pb and not np.isnan(pb) else "N/A"
            mv_str = f"{mv:.0f}" if mv else "N/A"
            lines.append(
                f"| {i+1} | {code} | {name} | "
                f"{row.get('composite', 0):.1f} | "
                f"{row.get('growth_score', 0):.0f} | "
                f"{row.get('quality_score', 0):.0f} | "
                f"{row.get('value_score', 0):.0f} | "
                f"{row.get('momentum_score', 0):.0f} | "
                f"{row.get('theme_score', 0):.0f} | "
                f"{pe_str} | {pb_str} | {mv_str} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── 二、因子雷达分析 ──
        lines.append("## 二、因子雷达分析")
        lines.append("")

        # 2.1 成长性TOP
        by_growth = results.sort_values("growth_score", ascending=False).head(5)
        lines.append("### 2.1 成长性TOP5")
        lines.append("")
        for i, row in by_growth.iterrows():
            code = from_ts_code(row["ts_code"])
            lines.append(f"- **{row['name']}**({code}) 成长分{row['growth_score']:.0f}")
        lines.append("")

        # 2.2 价值洼地TOP
        by_value = results.sort_values("value_score", ascending=False).head(5)
        lines.append("### 2.2 价值洼地TOP5")
        lines.append("")
        for i, row in by_value.iterrows():
            code = from_ts_code(row["ts_code"])
            pe = row.get("pe_ttm")
            pe_str = f"PE={pe:.1f}" if pe and not np.isnan(pe) else ""
            lines.append(f"- **{row['name']}**({code}) 价值分{row['value_score']:.0f} {pe_str}")
        lines.append("")

        # 2.3 动量强势TOP
        by_momentum = results.sort_values("momentum_score", ascending=False).head(5)
        lines.append("### 2.3 动量强势TOP5")
        lines.append("")
        for i, row in by_momentum.iterrows():
            code = from_ts_code(row["ts_code"])
            lines.append(f"- **{row['name']}**({code}) 动量分{row['momentum_score']:.0f}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── 三、重点个股解读 ──
        lines.append("## 三、重点个股解读")
        lines.append("")

        detail_count = min(3, len(results))
        for idx in range(detail_count):
            row = results.iloc[idx]
            code = from_ts_code(row["ts_code"])
            name = str(row["name"])
            composite = row.get("composite", 0)
            lines.append(f"### {idx+1}. {name}（{code}）— 综合分 {composite:.1f}")
            lines.append("")

            # 指标表
            pe = row.get("pe_ttm")
            pb = row.get("pb")
            mv = row.get("total_mv", 0) / 1e4 if row.get("total_mv") else 0

            lines.append("| 维度 | 分数 | 判定 | 估值指标 | 数值 |")
            lines.append("|------|------|------|---------|------|")

            for dim, label in [("growth_score", "成长"), ("quality_score", "质量"),
                               ("value_score", "价值"), ("momentum_score", "动量"),
                               ("theme_score", "主线")]:
                score = row.get(dim, 0)
                if score >= 70:
                    verdict = "优良"
                elif score >= 50:
                    verdict = "中等"
                else:
                    verdict = "偏弱"
                lines.append(f"| {label} | {score:.0f} | {verdict} | | |")

            pe_str = f"{pe:.1f}" if pe and not np.isnan(pe) else "N/A"
            pb_str = f"{pb:.1f}" if pb and not np.isnan(pb) else "N/A"
            lines.append(f"| | | | PE(TTM) | {pe_str} |")
            lines.append(f"| | | | PB | {pb_str} |")
            lines.append(f"| | | | 市值(亿) | {mv:.0f} |")
            lines.append("")

            # 入选理由
            reasons = []
            if row.get("growth_score", 0) >= 70:
                reasons.append("成长性突出")
            if row.get("quality_score", 0) >= 70:
                reasons.append("基本面优良")
            if row.get("value_score", 0) >= 60:
                reasons.append("估值有吸引力")
            if row.get("momentum_score", 0) >= 70:
                reasons.append("短期动量强劲")
            if row.get("theme_score", 0) >= 50:
                reasons.append("契合主线方向")

            if reasons:
                lines.append(f"**入选理由**: {'、'.join(reasons)}")
            else:
                lines.append("**入选理由**: 综合评分靠前，各维度无明显短板")
            lines.append("")

        lines.append("---")
        lines.append("")

        # ── 四、筛选漏斗概览 ──
        lines.append("## 四、筛选漏斗概览")
        lines.append("")
        lines.append("| 阶段 | 条件 | 数量 |")
        lines.append("|------|------|------|")
        lines.append("| 全A股 | — | ~5000 |")
        lines.append(f"| Layer 1 基础过滤 | 排除ST/新股/停牌/北交所/市值<{meta.get('min_mcap', 20)}亿 | {meta.get('candidates_count', 0)} |")

        tpl_name = meta.get("template_name", "自定义")
        min_sc = meta.get("min_score", 70)
        lines.append(f"| Layer 2 因子评分 | {tpl_name}策略，综合分≥{min_sc} | {meta.get('results_count', 0)} |")
        lines.append(f"| 最终输出 | TOP {len(results)} | {len(results)} |")
        lines.append("")

        # 策略权重
        tpl = SCREENER_TEMPLATES.get(meta.get("template", ""), {})
        if tpl.get("factor_weights"):
            w = tpl["factor_weights"]
            lines.append(f"**策略权重**: 成长{w['growth']:.0%} | 质量{w['quality']:.0%} | "
                          f"价值{w['value']:.0%} | 动量{w['momentum']:.0%} | 主线{w['theme']:.0%}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*报告由选股子系统自动生成。不构成投资建议，买入前请用技术分析工具进一步确认。*")

        return "\n".join(lines)

    def run_deep_analysis(self, results: pd.DataFrame, top_n: int = 3) -> list:
        """对TOP N候选做企业深度分析"""
        from scripts.screener.company_analyzer import CompanyAnalyzer

        analyzer = CompanyAnalyzer(self.client)
        analyses = []
        count = min(top_n, len(results))

        print(f"\n  企业深度分析 TOP {count}...")
        for idx in range(count):
            ts_code = results.iloc[idx]["ts_code"]
            name = results.iloc[idx].get("name", "")
            print(f"    分析 {idx+1}/{count}: {name}({ts_code})")
            try:
                analysis = analyzer.analyze(ts_code)
                analyses.append(analysis)
            except Exception as e:
                print(f"    ⚠ 分析失败: {e}")

        return analyses

    def save_pdf_report(self, results: pd.DataFrame, meta: dict,
                        deep_analyses: Optional[list] = None) -> str:
        """生成PDF报告"""
        from pathlib import Path
        from scripts.screener.report_pdf import generate_pdf_report

        today = datetime.now().strftime("%Y%m%d")
        keyword = meta.get("concept") or meta.get("industry") or meta.get("template_name", "选股")

        report_dir = Path("reports") / "choice"
        report_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = report_dir / f"{keyword}_{today}.pdf"

        # 收集因子详情
        factor_details = {}
        for _, row in results.iterrows():
            ts_code = row["ts_code"]
            factor_details[ts_code] = self.scorer.get_factor_details(ts_code, row)

        generate_pdf_report(
            results=results,
            meta=meta,
            factor_details=factor_details,
            company_analyses=deep_analyses or [],
            output_path=str(pdf_path),
        )
        return str(pdf_path)

    def send_to_feishu(self, results: pd.DataFrame, meta: dict,
                       report_path: str,
                       chat_id: str = "oc_351f0783247122c4712afd7631d6bb50") -> bool:
        """准备飞书消息：摘要发群里，MD全文导入飞书文档"""
        if results.empty:
            return False

        keyword = meta.get("concept") or meta.get("industry") or meta.get("template_name", "选股")
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 群消息摘要
        top3 = results.head(3)
        top3_parts = []
        for i, row in top3.iterrows():
            code = from_ts_code(row["ts_code"])
            top3_parts.append(f"{i+1}.{row['name']}({code}) {row.get('composite', 0):.1f}分")

        table_lines = []
        for i, row in results.iterrows():
            code = from_ts_code(row["ts_code"])
            table_lines.append(
                f"{i+1}. {row['name']}({code}) "
                f"综合{row.get('composite', 0):.1f} "
                f"成长{row.get('growth_score', 0):.0f} "
                f"动量{row.get('momentum_score', 0):.0f}"
            )

        msg_text = (
            f"【{keyword}选股报告】{date_str}\n\n"
            f"候选池{meta.get('candidates_count', 0)}只 → 入选{meta.get('results_count', 0)}只\n\n"
            f"TOP3推荐:\n" + "\n".join(top3_parts) + "\n\n"
            + "全部入选:\n" + "\n".join(table_lines)
        )

        # 群消息JSON
        self._feishu_msg = {
            "chat_id": chat_id,
            "msg_type": "text",
            "content": msg_text,
        }

        # 读取MD全文用于导入飞书文档
        from pathlib import Path
        md_file = Path(report_path)
        if md_file.exists():
            self._feishu_md_content = md_file.read_text(encoding="utf-8")
            self._feishu_doc_title = f"{keyword}选股报告_{date_str}"
        return True
