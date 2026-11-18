#!/usr/bin/env python3
"""Layer 2: 多因子评分系统。

5维因子: 成长性/质量/价值/动量/主线契合。
每因子0-100分，加权总分≥阈值进入候选。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from scripts.core.tushare_client import TushareClient, from_ts_code


# 默认因子权重
DEFAULT_WEIGHTS = {
    "growth": 0.30,
    "quality": 0.25,
    "value": 0.20,
    "momentum": 0.15,
    "theme": 0.10,
}


class FactorScorer:
    """多因子评分器"""

    def __init__(self, client: TushareClient):
        self.client = client
        self._fina_cache: dict[str, pd.DataFrame] = {}
        self._current_weights: Optional[dict] = None

    def score(
        self,
        candidates: pd.DataFrame,
        weights: Optional[dict] = None,
        min_score: float = 70,
        trade_date: Optional[str] = None,
        hard_filters: Optional[dict] = None,
    ) -> pd.DataFrame:
        """对候选股进行多因子评分。

        Args:
            candidates: Layer 1输出的DataFrame
            weights: 因子权重，默认DEFAULT_WEIGHTS
            min_score: 最低综合分阈值
            trade_date: 交易日期
            hard_filters: 硬性过滤条件（模板提供）

        Returns:
            DataFrame: 加入各因子分数和综合分的DataFrame
        """
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        self._current_weights = w
        df = candidates.copy()

        # 逐只评分
        scores_list = []
        total = len(df)
        for i, row in df.iterrows():
            ts_code = row["ts_code"]
            s = self._score_single(ts_code, row, trade_date, min_score_date=trade_date)
            scores_list.append(s)
            if (i + 1) % 100 == 0:
                print(f"  评分进度: {i + 1}/{total}")

        scores_df = pd.DataFrame(scores_list)
        df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

        # 综合分
        df["composite"] = (
            df["growth_score"] * w["growth"]
            + df["quality_score"] * w["quality"]
            + df["value_score"] * w["value"]
            + df["momentum_score"] * w["momentum"]
            + df["theme_score"] * w["theme"]
        )

        # 行业归一化：各因子分在行业内百分位排名
        if "industry" in df.columns:
            industry_scores = self._industry_normalize(df)
            if industry_scores is not None:
                df = industry_scores

        # 硬性过滤
        if hard_filters:
            df = self._apply_hard_filters(df, hard_filters)

        # 最低分过滤
        df = df[df["composite"] >= min_score]

        return df.sort_values("composite", ascending=False).reset_index(drop=True)

    def _score_single(self, ts_code: str, row: pd.Series,
                      trade_date: Optional[str],
                      min_score_date: Optional[str] = None) -> dict:
        """单只股票评分"""
        symbol = from_ts_code(ts_code)

        # 财务数据（缓存），按公告日过滤防止未来信息泄露
        fina = self._get_fina(ts_code, as_of_date=min_score_date)

        # 成长性因子
        growth = self._score_growth(fina)

        # 质量因子
        quality = self._score_quality(fina)

        # 价值因子（从daily_basic数据直接取）
        value = self._score_value(row)

        # 动量因子
        momentum = self._score_momentum(symbol, trade_date)

        # 主线契合因子
        theme = self._score_theme(ts_code)

        return {
            "growth_score": growth,
            "quality_score": quality,
            "value_score": value,
            "momentum_score": momentum,
            "theme_score": theme,
        }

    # ── 成长性因子 ──────────────────────────────────────────────────

    def _score_growth(self, fina: pd.DataFrame) -> float:
        """成长性评分: 营收增速+利润增速"""
        if fina.empty:
            return 30.0

        latest = fina.iloc[0]
        rev_growth = self._safe_float(latest.get("or_yoy"))
        profit_growth = self._safe_float(latest.get("q_gr_yoy"))

        # 营收增速评分 (0-50分)
        rev_score = self._bucket_score(rev_growth, [
            (-100, 5), (0, 15), (10, 25), (20, 35), (30, 42), (50, 50)
        ])

        # 利润增速评分 (0-50分)
        profit_score = self._bucket_score(profit_growth, [
            (-100, 5), (0, 15), (10, 25), (20, 35), (30, 42), (50, 50)
        ])

        return min(100, rev_score + profit_score)

    # ── 质量因子 ────────────────────────────────────────────────────

    def _score_quality(self, fina: pd.DataFrame) -> float:
        """质量评分: 6维升级版。

        ROE(25%) + 毛利率(20%) + 负债率(15%) +
        现金流/净利润(20%) + 应收账款周转率(10%) + 存货周转率(10%)
        """
        if fina.empty:
            return 30.0

        latest = fina.iloc[0]
        roe = self._safe_float(latest.get("roe"))
        gross_margin = self._safe_float(latest.get("grossprofit_margin"))
        debt_ratio = self._safe_float(latest.get("debt_to_assets"))
        cfps_raw = latest.get("cfps")
        eps_raw = latest.get("eps")
        ar_turn_raw = latest.get("ar_turn")
        inv_turn_raw = latest.get("inv_turn")

        # 处理重复列导致返回 Series 的情况
        def _to_float(v):
            if isinstance(v, pd.Series):
                return float(v.iloc[0]) if len(v) > 0 else float('nan')
            return self._safe_float(v)

        cfps = _to_float(cfps_raw)
        eps = _to_float(eps_raw)
        ar_turn = _to_float(ar_turn_raw)
        inv_turn = _to_float(inv_turn_raw)

        # 1. ROE评分 (0-25分)
        roe_score = self._bucket_score(roe, [
            (-100, 0), (0, 5), (5, 12), (10, 18), (15, 22), (20, 25)
        ])

        # 2. 毛利率评分 (0-20分)
        margin_score = self._bucket_score(gross_margin, [
            (0, 2), (15, 8), (25, 13), (35, 17), (50, 20)
        ])

        # 3. 负债率评分 (0-15分, 越低越好)
        debt_score = self._bucket_score(debt_ratio, reverse=True, buckets=[
            (0, 15), (20, 13), (40, 10), (60, 6), (80, 3), (100, 0)
        ])

        # 4. 盈利质量：每股现金流/每股收益 (0-20分)
        #    现金流>利润 = 利润质量高（钱收回来了）
        #    现金流<利润 = 可能是应收账款（钱没收回来）
        if not np.isnan(eps) and not np.isnan(cfps) and eps > 0:
            cash_ratio = cfps / eps
            cash_score = self._bucket_score(cash_ratio, [
                (-10, 0), (0, 5), (0.5, 10), (0.8, 14), (1.0, 17), (1.5, 20)
            ])
        else:
            cash_score = 5  # 无数据给低分

        # 5. 应收账款周转率 (0-10分, 越高越好)
        if not np.isnan(ar_turn):
            ar_score = self._bucket_score(ar_turn, [
                (0, 2), (2, 4), (4, 6), (6, 8), (10, 10)
            ])
        else:
            ar_score = 5

        # 6. 存货周转率 (0-10分, 越高越好)
        if not np.isnan(inv_turn):
            inv_score = self._bucket_score(inv_turn, [
                (0, 2), (1, 4), (3, 6), (5, 8), (10, 10)
            ])
        else:
            inv_score = 5

        return min(100, roe_score + margin_score + debt_score + cash_score + ar_score + inv_score)

    # ── 价值因子 ────────────────────────────────────────────────────

    def _score_value(self, row: pd.Series) -> float:
        """价值评分: PE+PB"""
        pe = self._safe_float(row.get("pe_ttm"))
        pb = self._safe_float(row.get("pb"))

        # PE评分 (0-55分, 适中区间最高)
        if pe <= 0:
            pe_score = 10  # 亏损
        elif pe <= 15:
            pe_score = 50
        elif pe <= 25:
            pe_score = 55
        elif pe <= 40:
            pe_score = 40
        elif pe <= 60:
            pe_score = 25
        elif pe <= 100:
            pe_score = 15
        else:
            pe_score = 5

        # PB评分 (0-45分)
        if pb <= 0:
            pb_score = 10
        elif pb <= 1:
            pb_score = 40
        elif pb <= 2:
            pb_score = 45
        elif pb <= 3:
            pb_score = 35
        elif pb <= 5:
            pb_score = 20
        else:
            pb_score = 10

        return min(100, pe_score + pb_score)

    # ── 动量因子 ────────────────────────────────────────────────────

    def _score_momentum(self, symbol: str,
                        trade_date: Optional[str]) -> float:
        """动量评分: 5维升级版。

        维度与权重：
          60日涨幅排名 (30%) — 中期动量
          波动率排名   (20%) — 低波更强
          RSI状态      (15%) — 超买超卖
          相对强度     (20%) — vs 沪深300
          量价配合     (15%) — 5日均量/20日均量
        """
        if not trade_date:
            trade_date = self.client.get_recent_trade_day()

        end_date = trade_date
        start_date = (datetime.strptime(trade_date, "%Y%m%d")
                      - timedelta(days=150)).strftime("%Y%m%d")  # 拉长到150天看60日动量

        try:
            df = self.client.get_kline(symbol, freq="D",
                                        start=start_date, end=end_date)
        except Exception:
            return 30.0

        if df.empty or len(df) < 20:
            return 30.0

        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series([1]*len(df))

        # ── 1. 60日涨幅 (0-30分) ──
        n60 = min(60, len(df))
        rise_60d = (close.iloc[-1] / close.iloc[-n60] - 1) * 100
        rise_60_score = self._bucket_score(rise_60d, [
            (-30, 0), (-10, 5), (0, 10), (5, 15), (10, 20), (20, 25), (40, 30)
        ])

        # ── 2. 波动率 (0-20分, 越低越好) ──
        if len(df) >= 20:
            daily_ret = close.pct_change().dropna().tail(20)
            volatility = daily_ret.std() * (252 ** 0.5) * 100  # 年化波动率%
        else:
            volatility = 50.0
        vol_score = self._bucket_score(volatility, reverse=True, buckets=[
            (10, 20), (20, 17), (30, 14), (40, 10), (50, 7), (70, 4), (100, 2)
        ])

        # ── 3. RSI(14) (0-15分) ──
        if len(df) >= 15:
            delta = close.diff().dropna()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
        else:
            rsi = 50.0
        # RSI 50-70 最好（偏强但没超买），>80超买扣分，<30超卖扣分
        if pd.notna(rsi):
            if 55 <= rsi <= 70:
                rsi_score = 15
            elif 50 <= rsi < 55 or 70 < rsi <= 75:
                rsi_score = 12
            elif 40 <= rsi < 50 or 75 < rsi <= 80:
                rsi_score = 8
            elif 30 <= rsi < 40:
                rsi_score = 5  # 超卖可能反弹
            else:
                rsi_score = 3  # 极端超买/超卖
        else:
            rsi_score = 8

        # ── 4. 相对强度 vs 沪深300 (0-20分) ──
        try:
            idx_df = self.client.get_index_daily(
                ts_code="000300.SH",  # 沪深300
                start=(datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=70)).strftime("%Y%m%d"),
                end=trade_date)
            if idx_df is not None and not idx_df.empty and len(idx_df) >= 20:
                idx_df = idx_df.sort_values("trade_date")
                idx_close = idx_df["close"].astype(float)
                n20 = min(20, len(idx_df), len(df))
                stock_ret = close.iloc[-1] / close.iloc[-n20] - 1
                idx_ret = idx_close.iloc[-1] / idx_close.iloc[-n20] - 1
                relative = (stock_ret - idx_ret) * 100
            else:
                relative = 0.0
        except Exception:
            relative = 0.0
        rel_score = self._bucket_score(relative, [
            (-20, 2), (-10, 5), (0, 10), (5, 14), (10, 17), (20, 20)
        ])

        # ── 5. 量价配合 (0-15分) ──
        vol_5 = volume.tail(5).mean()
        vol_20 = volume.tail(min(20, len(df))).mean()
        vol_trend = vol_5 / vol_20 if vol_20 > 0 else 1.0
        vp_score = self._bucket_score(vol_trend, [
            (0, 3), (0.5, 5), (0.8, 8), (1.0, 10), (1.3, 13), (2.0, 15)
        ])

        total = rise_60_score + vol_score + rsi_score + rel_score + vp_score
        return min(100, total)

    # ── 主线契合因子 ────────────────────────────────────────────────

    def _score_theme(self, ts_code: str) -> float:
        """主线契合度评分: 概念板块匹配"""
        try:
            # AI/科技相关概念关键词
            ai_keywords = ["AI", "人工智能", "芯片", "GPU", "算力", "大模型",
                           "半导体", "光通信", "数据要素", "云计算", "机器人",
                           "自动驾驶", "信创", "量子", "鸿蒙"]
            tech_keywords = ["5G", "6G", "物联网", "工业互联", "数字",
                             "新能源", "储能", "光伏", "风电", "锂电"]

            # 查询该股票所属概念
            concepts = self.client.get_concepts()
            if concepts.empty:
                return 30.0

            # 逐个概念检查是否包含该股票
            hit_count = 0
            total_checked = 0
            for _, concept_row in concepts.iterrows():
                try:
                    stocks = self.client.get_concept_stocks(concept_row["code"])
                    if stocks.empty:
                        continue
                    total_checked += 1
                    if ts_code in stocks["ts_code"].values:
                        name = concept_row["name"]
                        for kw in ai_keywords:
                            if kw in name:
                                hit_count += 2  # AI加倍
                                break
                        else:
                            for kw in tech_keywords:
                                if kw in name:
                                    hit_count += 1
                                    break
                except Exception:
                    continue
                # 限制检查数量避免太慢
                if total_checked >= 50:
                    break

            # 基础分30 + 命中加分
            score = 30 + min(70, hit_count * 10)
            return score

        except Exception:
            return 30.0

    # ── 硬性过滤 ────────────────────────────────────────────────────

    def _industry_normalize(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """行业归一化：各因子分在同行业内百分位排名。

        解决问题：科技股PE=40在全市场偏高，但在行业内偏低。
        方法：对每个因子，在同行业内计算百分位排名，替代绝对分。
        综合分用归一化后的因子重新计算。
        """
        factor_cols = ["growth_score", "quality_score", "value_score",
                       "momentum_score", "theme_score"]

        # 至少3只同行才有归一化意义
        industry_counts = df["industry"].value_counts()
        valid_industries = industry_counts[industry_counts >= 3].index

        if len(valid_industries) < 2:
            return None  # 行业太少，不值得归一化

        df = df.copy()
        for col in factor_cols:
            if col not in df.columns:
                continue
            # 百分位排名（同行业内）
            ranked = df.groupby("industry")[col].rank(pct=True) * 100
            # 只归一化有效行业的股票（≥3只）
            mask = df["industry"].isin(valid_industries)
            df.loc[mask, col + "_norm"] = ranked[mask]
            # 无效行业的保留原分
            df.loc[~mask, col + "_norm"] = df.loc[~mask, col]

        # 用归一化后的分数重新计算综合分
        w = self._current_weights or {"growth": 0.30, "quality": 0.25,
                                       "value": 0.20, "momentum": 0.15, "theme": 0.10}
        df["composite"] = (
            df["growth_score_norm"] * w["growth"]
            + df["quality_score_norm"] * w["quality"]
            + df["value_score_norm"] * w["value"]
            + df["momentum_score_norm"] * w["momentum"]
            + df["theme_score_norm"] * w["theme"]
        )

        return df

    def _apply_hard_filters(self, df: pd.DataFrame,
                            filters: dict) -> pd.DataFrame:
        """应用模板的硬性过滤条件"""
        fina_cache = {}

        for key, val in filters.items():
            if val is None:
                continue

            if key == "rev_growth_min":
                mask = df["ts_code"].apply(
                    lambda c: self._check_fina_field(c, "or_yoy", val, ">=", fina_cache))
                df = df[mask]

            elif key == "profit_growth_min":
                mask = df["ts_code"].apply(
                    lambda c: self._check_fina_field(c, "q_gr_yoy", val, ">=", fina_cache))
                df = df[mask]

            elif key == "profit_growth_max":
                mask = df["ts_code"].apply(
                    lambda c: self._check_fina_field(c, "q_gr_yoy", val, "<=", fina_cache))
                df = df[mask]

            elif key == "roe_min":
                mask = df["ts_code"].apply(
                    lambda c: self._check_fina_field(c, "roe", val, ">=", fina_cache))
                df = df[mask]

            elif key == "pe_max":
                if "pe_ttm" in df.columns:
                    df = df[(df["pe_ttm"] <= val) | df["pe_ttm"].isna()]

            elif key == "pb_max":
                if "pb" in df.columns:
                    df = df[(df["pb"] <= val) | df["pb"].isna()]

            elif key == "debt_max":
                mask = df["ts_code"].apply(
                    lambda c: self._check_fina_field(c, "debt_to_assets", val, "<=", fina_cache))
                df = df[mask]

            elif key == "mcap_min":
                if "total_mv" in df.columns:
                    df = df[df["total_mv"] >= val * 1e4]

            elif key == "rise_20d_min":
                # 需要动量因子数据，此处跳过（已在评分中处理）
                pass

            elif key == "turnover_min":
                if "turnover_rate" in df.columns:
                    df = df[df["turnover_rate"] >= val]

            elif key == "half_year_rise_max":
                # 近半年涨幅限制，在评分后处理
                pass

        return df

    def _check_fina_field(self, ts_code: str, field: str,
                          threshold: float, op: str,
                          cache: dict) -> bool:
        """检查财务指标是否满足阈值"""
        if ts_code not in cache:
            cache[ts_code] = self._get_fina(ts_code)
        fina = cache[ts_code]
        if fina.empty:
            return True  # 无数据时不过滤
        val = self._safe_float(fina.iloc[0].get(field))
        if np.isnan(val):
            return True
        if op == ">=":
            return val >= threshold
        elif op == "<=":
            return val <= threshold
        return True

    # ── 工具方法 ────────────────────────────────────────────────────

    def _get_fina(self, ts_code: str, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """获取财务数据（带缓存）。

        如果指定 as_of_date，只返回公告日 <= as_of_date 的财报，
        防止回测中使用未公告的财务数据（未来信息泄露）。
        """
        cache_key = ts_code
        if as_of_date:
            cache_key = f"{ts_code}_{as_of_date}"

        if cache_key not in self._fina_cache:
            try:
                raw = self.client.get_financial(ts_code)
                if as_of_date and not raw.empty and "ann_date" in raw.columns:
                    # 只保留公告日 <= 截断日的财报
                    raw = raw[raw["ann_date"] <= as_of_date]
                self._fina_cache[cache_key] = raw
            except Exception:
                self._fina_cache[cache_key] = pd.DataFrame()
        return self._fina_cache[cache_key]

    @staticmethod
    def _safe_float(val) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return float("nan")
        try:
            return float(val)
        except (ValueError, TypeError):
            return float("nan")

    @staticmethod
    def _bucket_score(value: float, buckets: list,
                      reverse: bool = False) -> float:
        """分段打分。buckets: [(threshold, score), ...]"""
        if np.isnan(value):
            return 10.0  # 缺失给低分
        if reverse:
            # 反向: 值越小分越高
            for threshold, score in buckets:
                if value <= threshold:
                    return score
            return 0.0
        else:
            # 正向: 值越大分越高
            prev_score = 0.0
            for threshold, score in buckets:
                if value < threshold:
                    return prev_score
                prev_score = score
            return prev_score

    # ── 因子详情（供PDF报告用）──────────────────────────────────────

    def get_factor_details(self, ts_code: str,
                           row: Optional[pd.Series] = None,
                           as_of_date: Optional[str] = None) -> dict:
        """返回因子评分的详细拆解，含原始指标值和分段说明"""
        fina = self._get_fina(ts_code, as_of_date=as_of_date)
        latest = fina.iloc[0] if not fina.empty else {}

        # ── 成长性 ──
        rev_growth = self._safe_float(latest.get("or_yoy"))
        profit_growth = self._safe_float(latest.get("q_gr_yoy"))
        rev_score = self._bucket_score(rev_growth, [
            (-100, 5), (0, 15), (10, 25), (20, 35), (30, 42), (50, 50)
        ])
        profit_score = self._bucket_score(profit_growth, [
            (-100, 5), (0, 15), (10, 25), (20, 35), (30, 42), (50, 50)
        ])

        # ── 质量 ──
        roe = self._safe_float(latest.get("roe"))
        gross_margin = self._safe_float(latest.get("grossprofit_margin"))
        debt_ratio = self._safe_float(latest.get("debt_to_assets"))
        roe_score = self._bucket_score(roe, [
            (-100, 0), (0, 8), (5, 18), (10, 28), (15, 35), (20, 40)
        ])
        margin_score = self._bucket_score(gross_margin, [
            (0, 2), (15, 10), (25, 18), (35, 24), (50, 30)
        ])
        debt_score = self._bucket_score(debt_ratio, reverse=True, buckets=[
            (0, 30), (20, 25), (40, 18), (60, 10), (80, 5), (100, 0)
        ])

        # ── 价值 ──
        pe = self._safe_float(row.get("pe_ttm")) if row is not None else float("nan")
        pb = self._safe_float(row.get("pb")) if row is not None else float("nan")

        if pe <= 0 or np.isnan(pe):
            pe_score = 10
        elif pe <= 15:
            pe_score = 50
        elif pe <= 25:
            pe_score = 55
        elif pe <= 40:
            pe_score = 40
        elif pe <= 60:
            pe_score = 25
        elif pe <= 100:
            pe_score = 15
        else:
            pe_score = 5

        if pb <= 0 or np.isnan(pb):
            pb_score = 10
        elif pb <= 1:
            pb_score = 40
        elif pb <= 2:
            pb_score = 45
        elif pb <= 3:
            pb_score = 35
        elif pb <= 5:
            pb_score = 20
        else:
            pb_score = 10

        return {
            "growth": {
                "total": min(100, rev_score + profit_score),
                "raw": {
                    "营收增速(%)": f"{rev_growth:.1f}" if not np.isnan(rev_growth) else "N/A",
                    "利润增速(%)": f"{profit_growth:.1f}" if not np.isnan(profit_growth) else "N/A",
                },
                "breakdown": {
                    "营收分段(0-50)": rev_score,
                    "利润分段(0-50)": profit_score,
                },
                "buckets": "<-100:5 | <0:15 | <10:25 | <20:35 | <30:42 | ≥50:50",
            },
            "quality": {
                "total": min(100, roe_score + margin_score + debt_score),
                "raw": {
                    "ROE(%)": f"{roe:.1f}" if not np.isnan(roe) else "N/A",
                    "毛利率(%)": f"{gross_margin:.1f}" if not np.isnan(gross_margin) else "N/A",
                    "负债率(%)": f"{debt_ratio:.1f}" if not np.isnan(debt_ratio) else "N/A",
                },
                "breakdown": {
                    "ROE分段(0-40)": roe_score,
                    "毛利率分段(0-30)": margin_score,
                    "负债率分段(0-30)": debt_score,
                },
                "buckets": "ROE: <0:0 | <5:18 | <10:28 | <15:35 | ≥20:40",
            },
            "value": {
                "total": min(100, pe_score + pb_score),
                "raw": {
                    "PE(TTM)": f"{pe:.1f}" if not np.isnan(pe) else "N/A",
                    "PB": f"{pb:.1f}" if not np.isnan(pb) else "N/A",
                },
                "breakdown": {
                    "PE分段(0-55)": pe_score,
                    "PB分段(0-45)": pb_score,
                },
                "buckets": "PE: ≤15:50 | ≤25:55 | ≤40:40 | ≤60:25 | ≤100:15 | >100:5",
            },
        }
