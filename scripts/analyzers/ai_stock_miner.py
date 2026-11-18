#!/usr/bin/env python3
"""AI产业链股票挖掘器 — 从概念板块发现、基本面筛选、评分排名AI供应链优质标的。

数据源: Tushare Pro（主力）, AKShare（备用降级）
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.core.tushare_client import TushareClient, from_ts_code, to_ts_code

# ── AI板块定义 ──────────────────────────────────────────────────────────
# 每个板块含概念关键词（用于软匹配AKShare概念名）和阵营潜力权重(1-5)
AI_SECTORS: Dict[str, Dict[str, Any]] = {
    "ai_chips": {
        "name": "AI芯片",
        "concepts": ["AI芯片", "GPU", "芯片概念"],
        "camp_weight": 4,
    },
    "optical": {
        "name": "光模块",
        "concepts": ["光纤", "光通信", "激光"],
        "camp_weight": 5,
    },
    "servers": {
        "name": "算力基础设施",
        "concepts": ["数据中心", "大数据"],
        "camp_weight": 4,
    },
    "cloud": {
        "name": "算力租赁/云",
        "concepts": ["云计算"],
        "camp_weight": 3,
    },
    "ai_app": {
        "name": "AI应用",
        "concepts": ["智能金融", "智能物流", "智能医疗"],
        "camp_weight": 3,
    },
    "data": {
        "name": "数据要素",
        "concepts": ["数据存储", "数据安全"],
        "camp_weight": 3,
    },
}

# ── 默认筛选阈值 ────────────────────────────────────────────────────────
_DEFAULT_CONFIG = {
    "value_pe_max": 40,
    "value_roe_min": 8,
    "value_revenue_growth_min": 10,
    "value_debt_max": 60,
    "momentum_revenue_growth_min": 20,
    "momentum_market_cap_min": 50,  # 亿元
    "momentum_turnover_min": 2,
    "cache_dir": os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "ai_mine"),
    "rate_limit": 0.5,  # 秒
}


def _safe_float(v, default=0.0) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _retry(fn=None, retries=3, delay=2):
    """重试装饰器"""
    if fn is None:
        return lambda f: _retry(f, retries=retries, delay=delay)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(delay * (i + 1))
    return wrapper


class AIStockMiner:
    """AI供应链股票挖掘器 — 三阶段: 发现 → 获取 → 筛选评分"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self._cache_dir = self.config["cache_dir"]
        os.makedirs(self._cache_dir, exist_ok=True)
        self._ts = TushareClient()

    # ── Phase 1: 发现 ───────────────────────────────────────────────

    def discover_stocks(self) -> Dict[str, List[Dict]]:
        """通过概念关键词匹配发现AI板块股票（Tushare数据源）"""
        print("Phase 1: 发现AI板块股票...")

        all_concepts = self._ts.get_concepts()
        if all_concepts.empty:
            print("  ⚠ 未获取到概念板块数据")
            return {}

        result: Dict[str, List[Dict]] = {}
        for sector_key, sector_info in AI_SECTORS.items():
            matched = []  # (concept_id, concept_name, matched_kw)
            for kw in sector_info["concepts"]:
                mask = all_concepts["name"].str.contains(kw, case=False, na=False)
                for _, row in all_concepts[mask].iterrows():
                    cid, cname = row["code"], row["name"]
                    if cid not in [m[0] for m in matched]:
                        matched.append((cid, cname, kw))

            stocks = []
            seen_codes = set()
            for concept_id, concept_name, matched_kw in matched:
                try:
                    df = self._ts.get_concept_stocks(concept_id)
                    if df.empty:
                        continue
                    for _, row in df.iterrows():
                        code = from_ts_code(row["ts_code"]) if "ts_code" in row else str(row.get("code", ""))
                        if code and code not in seen_codes:
                            seen_codes.add(code)
                            stocks.append({
                                "code": code,
                                "name": str(row.get("name", "")).strip(),
                                "concept": concept_name,
                                "concept_keyword": matched_kw,
                            })
                except Exception as e:
                    print(f"  ⚠ 获取概念'{concept_name}'失败: {e}")

            result[sector_key] = stocks
            print(f"  {sector_info['name']}: {len(stocks)}只 (匹配概念: {', '.join(m[1] for m in matched)})")

        return result

    # ── Phase 2: 获取基本面 ─────────────────────────────────────────

    def _get_cache_path(self, code: str) -> str:
        today = datetime.now().strftime("%Y%m%d")
        return os.path.join(self._cache_dir, f"{code}_{today}.json")

    def _load_cache(self, code: str) -> Optional[Dict]:
        path = self._get_cache_path(code)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_cache(self, code: str, data: Dict):
        path = self._get_cache_path(code)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def fetch_fundamentals(self, sector_stocks: Dict[str, List[Dict]],
                           quick_mode: bool = False) -> pd.DataFrame:
        """批量获取行情+财务数据（Tushare数据源）"""
        print(f"\nPhase 2: 获取基本面数据{'(快速模式)' if quick_mode else ''}...")

        # 去重合并所有股票，记录板块和概念标签
        stock_map: Dict[str, Dict] = {}
        for sector_key, stocks in sector_stocks.items():
            for s in stocks:
                code = s["code"]
                if code not in stock_map:
                    stock_map[code] = {
                        "code": code, "name": s["name"],
                        "sectors": [], "concepts": [],
                    }
                sector_name = AI_SECTORS[sector_key]["name"]
                if sector_name not in stock_map[code]["sectors"]:
                    stock_map[code]["sectors"].append(sector_name)
                if s["concept"] not in stock_map[code]["concepts"]:
                    stock_map[code]["concepts"].append(s["concept"])

        codes = list(stock_map.keys())
        ts_codes = [to_ts_code(c) for c in codes]
        print(f"  去重后: {len(codes)}只")

        # 用daily_basic一次获取全市场行情
        print("  获取全A行情指标...")
        trade_date = self._ts.get_recent_trade_day()
        basic_df = self._ts.get_daily_basic(trade_date=trade_date)
        basic_df["pure_code"] = basic_df["ts_code"].apply(from_ts_code)
        basic_df = basic_df[basic_df["pure_code"].isin(codes)]

        rows = []
        for _, row in basic_df.iterrows():
            code = row["pure_code"]
            info = stock_map.get(code, {})
            r = {
                "code": code,
                "name": info.get("name", ""),
                "sectors": "|".join(info.get("sectors", [])),
                "concepts": "|".join(info.get("concepts", [])),
                "n_concepts": len(info.get("concepts", [])),
                "price": _safe_float(row.get("close")),
                "change_pct": 0,  # daily_basic不含涨跌幅，用前日计算
                "pe": _safe_float(row.get("pe_ttm")),
                "pb": _safe_float(row.get("pb")),
                "market_cap": _safe_float(row.get("total_mv")) / 1e4,  # 万元→亿元
                "turnover": _safe_float(row.get("turnover_rate")),
                "volume_ratio": _safe_float(row.get("volume_ratio")),
            }

            # 财务指标（Tushare fina_indicator，带缓存）
            if not quick_mode:
                cached = self._load_cache(code)
                if cached:
                    r.update(cached)
                else:
                    try:
                        fi_df = self._ts.get_financial(code)
                        fi = {}
                        if not fi_df.empty:
                            latest = fi_df.iloc[0]
                            fi["roe"] = _safe_float(latest.get("roe"))
                            fi["revenue_growth"] = _safe_float(latest.get("or_yoy"))
                            fi["profit_growth"] = _safe_float(latest.get("q_gr_yoy"))
                            fi["debt_ratio"] = _safe_float(latest.get("debt_to_assets"))
                        else:
                            fi = {"roe": 0, "revenue_growth": 0,
                                  "profit_growth": 0, "debt_ratio": 0}
                        self._save_cache(code, fi)
                        r.update(fi)
                    except Exception as e:
                        print(f"  ⚠ {code} 财务指标获取失败: {e}")
                        r.update({"roe": 0, "revenue_growth": 0,
                                  "profit_growth": 0, "debt_ratio": 0})
            else:
                r.update({"roe": 0, "revenue_growth": 0,
                          "profit_growth": 0, "debt_ratio": 0})

            rows.append(r)

        print(f"  获取到 {len(rows)} 只股票数据")
        return pd.DataFrame(rows)

    # ── Phase 3: 筛选评分 ────────────────────────────────────────────

    @staticmethod
    def _score_valuation(pe: float, pb: float) -> float:
        if pe <= 0:
            return 10
        if pe < 15:
            score = 100
        elif pe < 25:
            score = 80
        elif pe < 40:
            score = 60
        elif pe < 60:
            score = 40
        elif pe < 100:
            score = 20
        else:
            score = 10
        # PB修正
        if pb < 3:
            mod = 1.0
        elif pb < 5:
            mod = 0.8
        elif pb < 10:
            mod = 0.6
        else:
            mod = 0.4
        return score * mod

    @staticmethod
    def _score_growth(rev_growth: float, profit_growth: float) -> float:
        if rev_growth > 50:
            score = 100
        elif rev_growth > 30:
            score = 80
        elif rev_growth > 20:
            score = 60
        elif rev_growth > 10:
            score = 40
        elif rev_growth > 0:
            score = 20
        else:
            score = 0
        # 净利润增速 > 营收增速 → 经营杠杆加分
        if profit_growth > rev_growth and rev_growth > 0:
            score = min(100, score + 15)
        return score

    @staticmethod
    def _score_camp(sector_key: str, n_concepts: int) -> float:
        weight = AI_SECTORS.get(sector_key, {}).get("camp_weight", 3)
        base = weight * 3  # max 15
        bonus = min(n_concepts - 1, 5) * 3  # 每多一个概念+3
        return min(100, (base + bonus) / 15 * 100)

    @staticmethod
    def _score_health(roe: float, debt: float) -> float:
        # ROE
        if roe > 20:
            roe_score = 40
        elif roe > 15:
            roe_score = 30
        elif roe > 10:
            roe_score = 20
        elif roe > 8:
            roe_score = 10
        else:
            roe_score = 0
        # 负债率
        if debt < 30:
            debt_score = 30
        elif debt < 50:
            debt_score = 20
        elif debt < 60:
            debt_score = 10
        else:
            debt_score = 0
        return min(100, (roe_score + debt_score) / 70 * 100)

    def _compute_total_score(self, row: pd.Series, sector_key: str) -> Dict:
        pe = row.get("pe", 0)
        pb = row.get("pb", 0)
        rev_g = row.get("revenue_growth", 0)
        prof_g = row.get("profit_growth", 0)
        n_con = row.get("n_concepts", 1)
        roe = row.get("roe", 0)
        debt = row.get("debt_ratio", 100)

        v = self._score_valuation(pe, pb)
        g = self._score_growth(rev_g, prof_g)
        c = self._score_camp(sector_key, n_con)
        h = self._score_health(roe, debt)

        total = v * 0.35 + g * 0.35 + c * 0.15 + h * 0.15
        return {"valuation": round(v, 1), "growth": round(g, 1),
                "camp": round(c, 1), "health": round(h, 1),
                "total": round(total, 1)}

    def screen_and_score(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """筛选 + 评分 + 分类"""
        print("\nPhase 3: 筛选评分...")

        value_picks = []
        momentum_picks = []
        all_scored = []

        for _, row in df.iterrows():
            # 找到该股票的主板块（取camp_weight最高的）
            primary_sector = "ai_app"
            sectors_str = str(row.get("sectors", ""))
            for sk, si in AI_SECTORS.items():
                if si["name"] in sectors_str:
                    if AI_SECTORS[sk]["camp_weight"] >= AI_SECTORS[primary_sector]["camp_weight"]:
                        primary_sector = sk

            scores = self._compute_total_score(row, primary_sector)

            item = {
                "code": row["code"],
                "name": row["name"],
                "sectors": row.get("sectors", ""),
                "concepts": row.get("concepts", ""),
                "price": round(_safe_float(row.get("price")), 2),
                "change_pct": round(_safe_float(row.get("change_pct")), 2),
                "pe": round(_safe_float(row.get("pe")), 1),
                "pb": round(_safe_float(row.get("pb")), 2),
                "market_cap_yi": round(_safe_float(row.get("market_cap")), 1),
                "turnover": round(_safe_float(row.get("turnover")), 2),
                "roe": round(_safe_float(row.get("roe")), 2),
                "revenue_growth": round(_safe_float(row.get("revenue_growth")), 2),
                "profit_growth": round(_safe_float(row.get("profit_growth")), 2),
                "debt_ratio": round(_safe_float(row.get("debt_ratio")), 2),
                "scores": scores,
            }
            all_scored.append(item)

            # 价值精选筛选
            pe = _safe_float(row.get("pe"))
            roe = _safe_float(row.get("roe"))
            rev_g = _safe_float(row.get("revenue_growth"))
            debt = _safe_float(row.get("debt_ratio"))
            mcap = _safe_float(row.get("market_cap"))

            if (0 < pe <= self.config["value_pe_max"]
                    and roe >= self.config["value_roe_min"]
                    and rev_g >= self.config["value_revenue_growth_min"]
                    and debt <= self.config["value_debt_max"]):
                item_v = {**item, "tag": "VALUE"}
                value_picks.append(item_v)

            # 动量精选筛选
            if (rev_g >= self.config["momentum_revenue_growth_min"]
                    and mcap >= self.config["momentum_market_cap_min"]
                    and _safe_float(row.get("turnover")) >= self.config["momentum_turnover_min"]):
                item_m = {**item, "tag": "MOMENTUM"}
                momentum_picks.append(item_m)

        value_picks.sort(key=lambda x: x["scores"]["total"], reverse=True)
        momentum_picks.sort(key=lambda x: x["scores"]["total"], reverse=True)
        all_scored.sort(key=lambda x: x["scores"]["total"], reverse=True)

        print(f"  价值精选: {len(value_picks)}只")
        print(f"  动量精选: {len(momentum_picks)}只")
        print(f"  全部评分: {len(all_scored)}只")

        return {
            "value_picks": value_picks,
            "momentum_picks": momentum_picks,
            "all_scored": all_scored,
        }

    # ── 编排器 ──────────────────────────────────────────────────────

    def run(self, quick_mode: bool = False,
            sector_filter: Optional[str] = None) -> Dict:
        """执行完整挖掘流程"""
        sector_stocks = self.discover_stocks()

        if sector_filter:
            sector_stocks = {k: v for k, v in sector_stocks.items()
                             if k == sector_filter}
            print(f"\n  仅分析: {AI_SECTORS.get(sector_filter, {}).get('name', sector_filter)}")

        df = self.fetch_fundamentals(sector_stocks, quick_mode=quick_mode)
        results = self.screen_and_score(df)

        sector_counts = {AI_SECTORS[k]["name"]: len(v) for k, v in sector_stocks.items()}

        return {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quick_mode": quick_mode,
            "total_discovered": sum(len(v) for v in sector_stocks.values()),
            "total_with_data": len(df),
            "sectors": sector_counts,
            **results,
        }

    # ── 格式化输出 ──────────────────────────────────────────────────

    @staticmethod
    def format_results(results: Dict, top_n: int = 20) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(f"🤖 AI产业链股票挖掘报告 | {results['run_time']}")
        if results.get("quick_mode"):
            lines.append("  (快速模式 — 无财务指标)")
        lines.append("=" * 80)

        # 板块概览
        lines.append(f"\n📊 板块发现: {results['total_discovered']}只 → 有效数据: {results['total_with_data']}只")
        for name, count in results.get("sectors", {}).items():
            lines.append(f"  {name}: {count}只")

        # 价值精选
        vp = results.get("value_picks", [])
        lines.append(f"\n{'─' * 80}")
        lines.append(f"💎 价值精选 ({len(vp)}只) — PE<{_DEFAULT_CONFIG['value_pe_max']}, ROE>{_DEFAULT_CONFIG['value_roe_min']}%, 营收增长>{_DEFAULT_CONFIG['value_revenue_growth_min']}%")
        lines.append(f"{'─' * 80}")
        if vp:
            lines.append(f"{'排名':>4} {'代码':>8} {'名称':<10} {'板块':<12} {'PE':>7} {'ROE':>6} {'营收增':>8} {'市值亿':>8} {'评分':>6}")
            lines.append("-" * 80)
            for i, s in enumerate(vp[:top_n]):
                lines.append(
                    f"{i+1:>4} {s['code']:>8} {s['name']:<10} "
                    f"{s['sectors'].split('|')[0]:<10} "
                    f"{s['pe']:>7.1f} {s['roe']:>6.1f}% "
                    f"{s['revenue_growth']:>+7.1f}% "
                    f"{s['market_cap_yi']:>8.1f} "
                    f"{s['scores']['total']:>6.1f}"
                )
        else:
            lines.append("  无符合条件标的")

        # 动量精选
        mp = results.get("momentum_picks", [])
        lines.append(f"\n{'─' * 80}")
        lines.append(f"🚀 动量精选 ({len(mp)}只) — 营收增长>{_DEFAULT_CONFIG['momentum_revenue_growth_min']}%, 市值>{_DEFAULT_CONFIG['momentum_market_cap_min']}亿")
        lines.append(f"{'─' * 80}")
        if mp:
            lines.append(f"{'排名':>4} {'代码':>8} {'名称':<10} {'板块':<12} {'PE':>7} {'营收增':>8} {'净利增':>8} {'市值亿':>8} {'评分':>6}")
            lines.append("-" * 80)
            for i, s in enumerate(mp[:top_n]):
                lines.append(
                    f"{i+1:>4} {s['code']:>8} {s['name']:<10} "
                    f"{s['sectors'].split('|')[0]:<10} "
                    f"{s['pe']:>7.1f} "
                    f"{s['revenue_growth']:>+7.1f}% "
                    f"{s['profit_growth']:>+7.1f}% "
                    f"{s['market_cap_yi']:>8.1f} "
                    f"{s['scores']['total']:>6.1f}"
                )
        else:
            lines.append("  无符合条件标的")

        return "\n".join(lines)
