#!/usr/bin/env python3
"""信号融合器 — 多选股策略交叉验证。

五套选股器独立评分后加权融合：
  1. W3启动（wave3_startup）   权重0.25
  2. 缠论三买点（chanlun_t3）   权重0.20
  3. 趋势扫描（trend_scan）     权重0.20
  4. 底部反转（bottom_reversal）权重0.20
  5. 增幅动量（gain_momentum）  权重0.15

融合规则：
  - 同向增强：≥3个系统看多 → 信号强度×1.5
  - 反向取消：≤1个系统看多 → 信号作废
  - 输出：融合分 + 建议仓位 + 目标价 + 止损价

用法：
  python scripts/screener/screener_cli.py --template fusion
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient, from_ts_code


# 各策略的评分字段名 → 标准化键名
SCORE_FIELDS = {
    "wave3_startup": "w3s_composite",
    "chanlun_t3": "t3_composite",
    "trend_scan": "trend_strength",
    "bottom_reversal": "br_score",
    "gain_momentum": "gm_total_score",
}

# 各策略权重
STRATEGY_WEIGHTS = {
    "wave3_startup": 0.25,
    "chanlun_t3": 0.20,
    "trend_scan": 0.20,
    "bottom_reversal": 0.20,
    "gain_momentum": 0.15,
}

# 仓位建议（基于融合分）
POSITION_TIERS = [
    (80, 0.10),   # ≥80 → 10%
    (65, 0.07),   # ≥65 → 7%
    (50, 0.05),   # ≥50 → 5%
    (40, 0.03),   # ≥40 → 3%
    (0, 0.0),     # <40 → 不建仓
]

# 多系统共振加成
MIN_SYSTEMS_FOR_BOOST = 3  # ≥3个系统看多才加成
BOOST_FACTOR = 1.5
MIN_SYSTEMS_TO_PASS = 2    # <2个系统看多则作废


class SignalFusion:
    """多策略信号融合器"""

    def __init__(self, client: Optional[TushareClient] = None):
        self.client = client or TushareClient()

    def scan(
        self,
        candidates: pd.DataFrame,
        as_of_date: Optional[str] = None,
        concept: Optional[str] = None,
        industry: Optional[str] = None,
        min_fusion_score: float = 40.0,
        top: int = 20,
        quick: bool = False,
        lookback_days: int = 200,
    ) -> pd.DataFrame:
        """执行多策略融合扫描。

        Args:
            candidates: 候选股列表（来自 universe_filter）
            as_of_date: 截断日期
            concept: 概念板块
            industry: 行业
            min_fusion_score: 最低融合分
            top: 返回前N只
            quick: 快速模式（只用趋势扫描）
            lookback_days: K线回溯天数

        Returns:
            DataFrame: 融合评分结果
        """
        from scripts.screener.wave3_startup_screener import Wave3StartupScreener
        from scripts.screener.chanlun_t3_screener import ChanLunT3Screener
        from scripts.screener.trend_scanner import TrendScanner

        # 底部反转可能不存在，可选
        try:
            from scripts.screener.bottom_reversal_screener import BottomReversalScreener
            has_br = True
        except ImportError:
            has_br = False

        strategies_results = {}

        # ── 策略1: W3启动 ──
        print("  [1/5] W3启动扫描...")
        try:
            w3s = Wave3StartupScreener(self.client)
            w3s_result = w3s.scan(
                candidates, lookback_days=lookback_days,
                min_confidence=30.0, top=top * 5,
                as_of_date=as_of_date)
            if not w3s_result.empty:
                strategies_results["wave3_startup"] = w3s_result
                print(f"    W3启动: {len(w3s_result)} 只")
            else:
                print("    W3启动: 0 只")
        except Exception as e:
            print(f"    W3启动失败: {e}")

        # ── 策略2: 缠论三买点 ──
        print("  [2/5] 缠论三买点扫描...")
        try:
            t3s = ChanLunT3Screener(self.client)
            t3s_result = t3s.scan(
                candidates, lookback_days=lookback_days,
                min_confidence=30.0, top=top * 5,
                as_of_date=as_of_date)
            if not t3s_result.empty:
                strategies_results["chanlun_t3"] = t3s_result
                print(f"    缠论T3: {len(t3s_result)} 只")
            else:
                print("    缠论T3: 0 只")
        except Exception as e:
            print(f"    缠论T3失败: {e}")

        # ── 策略3: 趋势扫描 ──
        print("  [3/5] 趋势扫描...")
        try:
            ts = TrendScanner(self.client)
            ts_result = ts.scan(
                candidates, lookback_days=120,
                min_grade="A", top=top * 5,
                as_of_date=as_of_date,
                concept=concept, industry=industry)
            if not ts_result.empty:
                strategies_results["trend_scan"] = ts_result
                print(f"    趋势扫描: {len(ts_result)} 只")
            else:
                print("    趋势扫描: 0 只")
        except Exception as e:
            print(f"    趋势扫描失败: {e}")

        # ── 策略4: 底部反转 ──
        if has_br:
            print("  [4/5] 底部反转扫描...")
            try:
                brs = BottomReversalScreener(self.client)
                br_result = brs.scan(
                    candidates, lookback_days=200,
                    min_confidence=30.0, top=top * 5,
                    as_of_date=as_of_date)
                if not br_result.empty:
                    strategies_results["bottom_reversal"] = br_result
                    print(f"    底部反转: {len(br_result)} 只")
                else:
                    print("    底部反转: 0 只")
            except Exception as e:
                print(f"    底部反转失败: {e}")
        else:
            print("  [4/5] 底部反转: 模块未安装，跳过")

        # ── 策略5: 增幅动量 ──
        print("  [5/5] 增幅动量扫描...")
        try:
            from scripts.screener.gain_momentum_scorer import GainMomentumScorer
            gm_scorer = GainMomentumScorer(self.client)
            gm_rows = []
            code_list = candidates["ts_code"].tolist() if "ts_code" in candidates.columns else []
            for code in code_list:
                try:
                    gm_result = gm_scorer.score(code)
                    if gm_result and gm_result.total_score >= 30:
                        gm_rows.append({
                            "ts_code": code,
                            "name": gm_result.name or "",
                            "gm_total_score": gm_result.total_score,
                            "gm_grade": gm_result.grade,
                            "current_price": 0,
                        })
                except Exception:
                    continue
            if gm_rows:
                gm_df = pd.DataFrame(gm_rows)
                strategies_results["gain_momentum"] = gm_df
                print(f"    增幅动量: {len(gm_df)} 只")
            else:
                print("    增幅动量: 0 只")
        except Exception as e:
            print(f"    增幅动量失败: {e}")

        if not strategies_results:
            print("  ⚠️ 所有策略均无结果")
            return pd.DataFrame()

        # ── 融合 ──
        return self._fuse(strategies_results, min_fusion_score, top)

    def _fuse(
        self,
        strategies_results: dict[str, pd.DataFrame],
        min_fusion_score: float,
        top: int,
    ) -> pd.DataFrame:
        """多策略融合核心逻辑。"""
        # 收集所有出现过的股票
        all_codes = {}
        for strategy_name, df in strategies_results.items():
            for _, row in df.iterrows():
                ts_code = row.get("ts_code", "")
                if not ts_code:
                    continue
                if ts_code not in all_codes:
                    all_codes[ts_code] = {
                        "ts_code": ts_code,
                        "name": row.get("name", ""),
                        "industry": row.get("industry", ""),
                        "_scores": {},
                        "_details": {},
                    }
                # 记录该策略的评分
                score_field = SCORE_FIELDS.get(strategy_name)
                if score_field and score_field in row.index:
                    score = float(row.get(score_field, 0))
                    if pd.notna(score):
                        all_codes[ts_code]["_scores"][strategy_name] = score
                        # 保留一些策略特有信息
                        all_codes[ts_code]["_details"][strategy_name] = {
                            "score": score,
                            "current_price": row.get("w3s_current_price",
                                             row.get("t3_current_price",
                                             row.get("current_price", 0))),
                        }

        if not all_codes:
            return pd.DataFrame()

        # 计算融合分
        fusion_rows = []
        for ts_code, info in all_codes.items():
            scores = info["_scores"]
            n_systems = len(scores)

            if n_systems < MIN_SYSTEMS_TO_PASS:
                continue  # 太少系统认可，作废

            # 加权平均分
            raw_fusion = 0.0
            total_weight = 0.0
            for strategy_name, score in scores.items():
                w = STRATEGY_WEIGHTS.get(strategy_name, 0.1)
                raw_fusion += score * w
                total_weight += w

            if total_weight == 0:
                continue

            fusion_score = raw_fusion / total_weight

            # 多系统共振加成
            boost = 1.0
            if n_systems >= MIN_SYSTEMS_FOR_BOOST:
                boost = BOOST_FACTOR

            final_score = fusion_score * boost

            # 建议仓位
            suggested_position = 0.0
            for threshold, pct in POSITION_TIERS:
                if final_score >= threshold:
                    suggested_position = pct
                    break

            # 目标价/止损价（从各策略的估值中取平均）
            prices = []
            for details in info["_details"].values():
                p = details.get("current_price", 0)
                if p and p > 0:
                    prices.append(float(p))

            avg_price = np.mean(prices) if prices else 0

            fusion_rows.append({
                "ts_code": ts_code,
                "name": info["name"],
                "industry": info["industry"],
                "fusion_score": round(final_score, 1),
                "raw_fusion": round(fusion_score, 1),
                "n_systems": n_systems,
                "boost": boost,
                "suggested_position_pct": round(suggested_position, 3),
                "current_price": round(avg_price, 2),
                "target_price": round(avg_price * 1.15, 2),  # 目标+15%
                "stop_loss_price": round(avg_price * 0.92, 2),  # 止损-8%
                "systems": "+".join(sorted(scores.keys())),
                "w3s_score": scores.get("wave3_startup"),
                "t3_score": scores.get("chanlun_t3"),
                "trend_score": scores.get("trend_scan"),
                "br_score": scores.get("bottom_reversal"),
                "gm_score": scores.get("gain_momentum"),
            })

        if not fusion_rows:
            return pd.DataFrame()

        result = pd.DataFrame(fusion_rows)
        result = result[result["fusion_score"] >= min_fusion_score]
        result = result.sort_values("fusion_score", ascending=False)
        result = result.head(top).reset_index(drop=True)

        print(f"\n  融合结果: {len(result)} 只（{len(all_codes)} 只候选）")
        return result

    @staticmethod
    def format_report(results: pd.DataFrame) -> str:
        """格式化融合结果报告"""
        if results.empty:
            return "无融合结果"

        lines = []
        lines.append("=" * 72)
        lines.append("  🔀 信号融合报告")
        lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                      f"候选{len(results)}只")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"  {'#':>3}  {'代码':8} {'名称':8} {'融合分':>6} "
                      f"{'系统':>4} {'加成':>4} {'仓位':>5} "
                      f"{'现价':>7} {'目标':>7} {'止损':>7}")
        lines.append("  " + "-" * 65)

        for i, (_, row) in enumerate(results.iterrows(), 1):
            name = str(row.get("name", ""))[:6]
            fusion = row.get("fusion_score", 0)
            n_sys = row.get("n_systems", 0)
            boost = row.get("boost", 1.0)
            pos = row.get("suggested_position_pct", 0) * 100
            price = row.get("current_price", 0)
            target = row.get("target_price", 0)
            sl = row.get("stop_loss_price", 0)
            boost_str = f"×{boost:.1f}" if boost > 1 else ""
            systems = row.get("systems", "")

            lines.append(
                f"  {i:3}  {row.get('ts_code',''):8} {name:8} "
                f"{fusion:6.1f} {n_sys:4} {boost_str:>4} "
                f"{pos:4.0f}% {price:7.2f} {target:7.2f} {sl:7.2f}"
            )
            # 各系统分数
            parts = []
            for key, label in [("w3s_score", "W3"), ("t3_score", "T3"),
                               ("trend_score", "趋势"), ("br_score", "反转"),
                               ("gm_score", "动量")]:
                v = row.get(key)
                if pd.notna(v):
                    parts.append(f"{label}:{v:.0f}")
            if parts:
                lines.append(f"       {'  '.join(parts)}  [{systems}]")

        lines.append("")
        lines.append("  加成规则: ≥3系统共振 ×1.5 | <2系统作废")
        lines.append("  注意: 多系统交叉验证结果，不构成投资建议。")
        return "\n".join(lines)
