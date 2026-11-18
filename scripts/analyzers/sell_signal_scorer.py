#!/usr/bin/env python3
"""五维卖出信号评分器 — 区分洗盘 vs 真跌（v3 含MACD多周期共振）。

核心思想：卖出信号不能只看 f'<0 + f''<0（减速见顶），需要多维度确认。
通过 5 个维度综合评分，区分"主力洗盘"和"趋势反转真跌"。

评分维度（满分100，v3 含MACD共振）：
  ┌──────────────┬──────────────────────────┬──────┬──────────────────────────┐
  │ 维度         │ 指标                     │ 权重 │ 含义                     │
  ├──────────────┼──────────────────────────┼──────┼──────────────────────────┤
  │ 趋势衰减     │ f''+3日前向差分          │ 15分 │ 含前向差分弥补f''滞后    │
  │ MACD共振     │ 日线双参数+60min共振     │ 25分 │ 大跌时MACD最有用         │
  │ 量能异动     │ 放量下跌 vs 缩量下跌     │ 20分 │ 放量跌=真跑，缩量=洗盘   │
  │ 大盘环境     │ 大盘周线+板块联动        │ 20分 │ 大盘弱+板块弱=真跌       │
  │ 关键位破位   │ 是否破MA20/MA60          │ 20分 │ MA破位是最直接的信号     │
  └──────────────┴──────────────────────────┴──────┴──────────────────────────┘

铁律3S加成：MA5死叉MA10 + MACD柱转负/DIF死叉DEA + 收盘破MA20 → 额外+15分

洗盘 vs 真跌判别：
  洗盘: 缩量下跌(vol_ratio<0.8) + 大盘稳 + 板块不弱 + 不破MA20 + f''即将转正
  真跌: 放量下跌(vol_ratio>1.5) + 大盘弱 + 板块联动跌 + 破MA20 + f''持续负

评分阈值（v3 版）：
  ≥50: 减仓（真跌概率高）
  35-49: 观望
  <35: 可能是洗盘（不要卖）

用法:
    from scripts.analyzers.sell_signal_scorer import SellSignalScorer
    scorer = SellSignalScorer()
    result = scorer.score(code='603881', daily_df=df, minute_df=minute_df)
    # result.score, result.action, result.confidence, result.details
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from typing import Optional


# ── 默认参数 ──────────────────────────────────────────────────────────

DEFAULT_SG_WINDOW = 21   # S-G 窗口（约4周日线）
DEFAULT_SG_POLY = 3      # 三次多项式

# ── 评分阈值（v2 调优版）──────────────────────────────────────────────

THRESHOLD_REDUCE = 50        # ≥50: 减仓
THRESHOLD_WATCH = 35         # 35-49: 观望
# <35: 可能是洗盘

# ── 各维度满分（v3: 5维度含MACD共振）──────────────────────────────────

MAX_TREND_DECAY = 15     # 趋势衰减维度满分（原20，f''滞后严重降权）
MAX_MACD = 25            # MACD多周期共振维度满分（新维度）
MAX_VOLUME = 20          # 量能异动维度满分（原25，让出给MACD）
MAX_MARKET = 20          # 大盘环境维度满分（原25，让出给MACD）
MAX_KEY_LEVEL = 20       # 关键位破位维度满分（原30，让出给MACD）
# 铁律3S加成
IRON_5B_BONUS = 10       # 铁律3S三条件同时满足时的额外加分

# 数据缺失时给中间分
NEUTRAL_TREND_DECAY = 7   # 趋势衰减中间分
NEUTRAL_MACD = 12         # MACD共振中间分（无数据时）
NEUTRAL_VOLUME = 10       # 量能中间分
NEUTRAL_MARKET = 10       # 大盘环境中间分
NEUTRAL_KEY_LEVEL = 10    # 关键位破位中间分


@dataclass
class SellScoreDetail:
    """单个维度的评分详情。"""
    dimension: str = ""         # 维度名称
    score: float = 0.0          # 该维度得分
    max_score: float = 0.0      # 该维度满分
    description: str = ""       # 描述
    is_wash: bool = False       # 是否指向洗盘（True=偏洗盘，False=偏真跌）


@dataclass
class SellSignalResult:
    """卖出信号评分结果。"""
    code: str = ""
    score: float = 0.0          # 总分 0-100（越高越可能是真跌）
    action: str = "观望"        # 操作建议
    confidence: float = 0.0     # 置信度 0-1
    details: list[SellScoreDetail] = field(default_factory=list)
    # 快速判断字段
    is_wash_probability: float = 0.0  # 洗盘概率 0-1
    signal_type: str = ""              # 信号类型描述
    # 原始指标
    indicators: dict = field(default_factory=dict)


class SellSignalScorer:
    """三维卖出信号评分器 — 区分洗盘 vs 真跌。

    使用方法：
        scorer = SellSignalScorer()
        result = scorer.score(
            code='603881',
            daily_df=df,              # DataFrame, 至少包含 close, volume
            index_daily=index_df,     # 可选: 大盘日线
            sector_daily=sector_df,   # 可选: 板块日线
        )
    """

    def __init__(
        self,
        sg_window: int = DEFAULT_SG_WINDOW,
        sg_poly: int = DEFAULT_SG_POLY,
    ):
        self.sg_window = sg_window
        self.sg_poly = sg_poly

    # ── 公开接口 ──────────────────────────────────────────────────────

    def score(
        self,
        code: str,
        daily_df: pd.DataFrame,
        minute_df: Optional[pd.DataFrame] = None,
        index_daily: Optional[pd.DataFrame] = None,
        sector_daily: Optional[pd.DataFrame] = None,
    ) -> SellSignalResult:
        """对个股进行卖出信号评分。

        Args:
            code: 股票代码（纯数字，如 '603881'）
            daily_df: 日线数据 DataFrame，需包含 close, volume 列，
                      至少120天数据。建议包含 high, low, open 列。
            minute_df: 60分钟线数据（可选），需包含 close 列。
                      截断到评估日当天，用于60min MACD共振计算。
            index_daily: 大盘日线数据（可选，如上证指数），
                         需包含 close 列
            sector_daily: 板块日线数据（可选），需包含 close 列

        Returns:
            SellSignalResult: 评分结果
        """
        # ── 数据准备 ──
        df = self._prepare_dataframe(daily_df)
        if df is None or len(df) < 60:
            return SellSignalResult(
                code=code, score=0.0, action="数据不足",
                confidence=0.0,
                details=[SellScoreDetail(
                    dimension="数据", score=0, max_score=100,
                    description="日线数据不足60天，无法评分"
                )],
                indicators={},
            )

        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        high = df["high"].values.astype(float) if "high" in df.columns else None
        low = df["low"].values.astype(float) if "low" in df.columns else None

        # ── 计算技术指标 ──
        f1, f2 = self._compute_derivatives(close)
        f2_neg_days = self._count_consecutive_negative(f2)
        f1_val = float(f1[-1])
        f2_val = float(f2[-1])

        # 量能比
        vol_ratio = self._compute_volume_ratio(volume)
        is_price_down = close[-1] < close[-2] if len(close) >= 2 else False

        # MA20 / MA60
        ma20 = self._compute_ma(close, 20)
        ma60 = self._compute_ma(close, 60)
        cur_price = float(close[-1])
        below_ma20 = cur_price < ma20 if ma20 is not None else False
        below_ma60 = cur_price < ma60 if ma60 is not None else False

        # ── 前向差分信号（3日斜率，弥补f''滞后）──
        close_slope_3d = None  # 近3日close斜率
        vol_increasing = None  # 近3日量能是否递增（放量）
        if len(close) >= 4:
            # 3日线性回归斜率（比简单差分更稳健）
            x = np.arange(3, dtype=float)
            y_close = close[-3:]
            close_slope_3d = float(np.polyfit(x, y_close, 1)[0])
        if volume is not None and len(volume) >= 4:
            # 近3日量能是否递增
            vol_increasing = float(volume[-1]) > float(volume[-2]) > float(volume[-3])

        # ── 大盘环境 ──
        index_f1 = None
        index_f2 = None
        index_neg_days = None
        if index_daily is not None:
            index_f1, index_f2, index_neg_days = self._analyze_index(index_daily)

        # ── 板块环境 ──
        sector_down = None
        if sector_daily is not None:
            sector_down = self._analyze_sector(sector_daily)

        # ── 汇总指标 ──
        indicators = {
            "f1": round(f1_val, 5),
            "f2": round(f2_val, 5),
            "f2_neg_days": f2_neg_days,
            "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
            "is_price_down": is_price_down,
            "below_ma20": below_ma20,
            "below_ma60": below_ma60,
            "ma20": round(ma20, 3) if ma20 is not None else None,
            "ma60": round(ma60, 3) if ma60 is not None else None,
            "cur_price": round(cur_price, 3),
            "index_f2_neg_days": index_neg_days,
            "sector_down": sector_down,
            "close_slope_3d": round(close_slope_3d, 4) if close_slope_3d is not None else None,
            "vol_increasing": vol_increasing,
        }

        # ── 五维度评分 ──
        details: list[SellScoreDetail] = []
        total_score = 0.0

        # 1. 趋势衰减 (15分，含前向差分)
        d1 = self._score_trend_decay(f2_val, f2_neg_days, f1_val, f2,
                                      close_slope_3d, vol_increasing, is_price_down)
        details.append(d1)
        total_score += d1.score

        # 2. MACD多周期共振 (25分)
        d2 = self._score_macd_resonance(close, minute_df)
        details.append(d2)
        total_score += d2.score

        # 3. 量能异动 (20分)
        d3 = self._score_volume_anomaly(vol_ratio, is_price_down)
        details.append(d3)
        total_score += d3.score

        # 4. 大盘环境 (20分)
        d4 = self._score_market_environment(
            index_f1, index_f2, index_neg_days, sector_down
        )
        details.append(d4)
        total_score += d4.score

        # 5. 关键位破位 (20分)
        d5 = self._score_key_level_break(below_ma20, below_ma60, cur_price, ma20, ma60)
        details.append(d5)
        total_score += d5.score

        # ── 铁律3S硬过滤加成 ──
        # MA5死叉MA10 + MACD柱转负/DIF死叉DEA + 收盘破MA20 → 三条件同时满足+15分
        iron3S_bonus = 0.0
        iron3S_triggered = self._check_iron_3S(close, d2.description)
        if iron3S_triggered:
            iron3S_bonus = IRON_5B_BONUS
            total_score += iron3S_bonus
            indicators["iron_3S_triggered"] = True
        else:
            indicators["iron_3S_triggered"] = False

        # MACD指标放入indicators供报告使用
        indicators["macd_detail"] = d2.description

        # ── 洗盘概率评估 ──
        wash_prob = self._estimate_wash_probability(
            vol_ratio, is_price_down,
            index_f2, index_neg_days,
            sector_down,
            below_ma20, f2_val, f2_neg_days
        )

        # ── 综合判断 ──
        action = self._determine_action(total_score, wash_prob)
        confidence = self._compute_confidence(
            index_daily is not None, sector_daily is not None,
            volume is not None, len(df)
        )
        signal_type = self._describe_signal_type(f1_val, f2_val, vol_ratio, wash_prob)

        return SellSignalResult(
            code=code,
            score=round(total_score, 1),
            action=action,
            confidence=round(confidence, 3),
            details=details,
            is_wash_probability=round(wash_prob, 3),
            signal_type=signal_type,
            indicators=indicators,
        )

    # ── 维度1: 趋势衰减 (满分15，含前向差分信号) ────────────────────────

    def _score_trend_decay(
        self,
        f2_val: float,
        f2_neg_days: int,
        f1_val: float,
        f2_arr: np.ndarray,
        close_slope_3d: Optional[float] = None,
        vol_increasing: Optional[bool] = None,
        is_price_down: bool = False,
    ) -> SellScoreDetail:
        """趋势衰减评分（v3 版）。

        评分规则（满分15）：
        - f''连续负天数：0-6分
        - f'/f''组合形态：0-4分
        - 前向差分（3日斜率）：0-5分（弥补f''滞后）

        当f''滞后时，前向差分能更早捕捉趋势转向。
        """
        score = 0.0
        reasons = []
        is_wash = False

        # ── A. f''连续负天数评分（0-6分）──
        if f2_neg_days >= 10:
            score += 6
            reasons.append(f"f''连续{f2_neg_days}天负（深度见顶）")
        elif f2_neg_days >= 5:
            score += 5
            reasons.append(f"f''连续{f2_neg_days}天负（明显减速）")
        elif f2_neg_days >= 3:
            score += 3
            reasons.append(f"f''连续{f2_neg_days}天负（减速中）")
        elif f2_neg_days >= 1:
            score += 1.5
            reasons.append(f"f''刚转负{f2_neg_days}天")
        else:
            score += 0
            reasons.append("f''为正（加速中）")

        # ── B. f'/f''组合形态（0-4分）──
        if f1_val > 0 and f2_val < 0:
            score += 2
            reasons.append("f'>0+f''<0 减速见顶形态")
        elif f1_val < 0 and f2_val < 0:
            score += 4  # 加速下跌最危险
            reasons.append("f'<0+f''<0 加速下跌")
        elif f1_val < 0 and f2_val > 0:
            score += 1   # 可能是底部减速
            reasons.append("下跌中f''转正（可能企稳）")
            is_wash = True

        # ── C. 前向差分信号（0-5分，弥补f''滞后）──
        if close_slope_3d is not None:
            if close_slope_3d < 0:
                # 3日斜率向下 → 趋势走坏
                if is_price_down:
                    score += 3.5
                    reasons.append(f"3日斜率向下(slope={close_slope_3d:.3f})")
                else:
                    score += 1.5
                    reasons.append(f"3日斜率微降(slope={close_slope_3d:.3f})")

                # 放量下跌额外加分
                if vol_increasing:
                    score += 1.5
                    reasons.append("放量下跌（量能递增确认）")
            else:
                # 3日斜率向上 → 趋势尚好
                reasons.append(f"3日斜率向上(slope={close_slope_3d:.3f})")
                is_wash = True
        else:
            # 无前向差分数据，用f''回升检测
            if len(f2_arr) >= 5 and f2_val < 0:
                f2_recent = f2_arr[-5:]
                f2_trend = f2_recent[-1] - f2_recent[0]
                if f2_trend > 0:
                    score -= 1.5
                    reasons.append("f''正在回升（可能洗盘）")
                    is_wash = True

        score = max(0.0, min(score, MAX_TREND_DECAY))
        return SellScoreDetail(
            dimension="趋势衰减",
            score=round(score, 1),
            max_score=MAX_TREND_DECAY,
            description="; ".join(reasons),
            is_wash=is_wash,
        )

    # ── 维度3: 量能异动 (满分20) ─────────────────────────────────────

    def _score_volume_anomaly(
        self,
        vol_ratio: Optional[float],
        is_price_down: bool,
    ) -> SellScoreDetail:
        """量能异动评分。

        评分规则：
        - 放量下跌(vol_ratio > 1.5): 真跑，高分
        - 缩量下跌(vol_ratio < 0.8): 洗盘概率，低分
        - 温和放量下跌: 中间分
        """
        if vol_ratio is None:
            return SellScoreDetail(
                dimension="量能异动",
                score=NEUTRAL_VOLUME,
                max_score=MAX_VOLUME,
                description="无成交量数据，给中间分",
                is_wash=False,
            )

        score = 0.0
        reasons = []
        is_wash = False

        if is_price_down:
            # ── 下跌时的量能分析 ──
            if vol_ratio > 2.0:
                score += 20
                reasons.append(f"暴跌放量(vol_ratio={vol_ratio:.1f}，主力出逃)")
            elif vol_ratio > 1.5:
                score += 16
                reasons.append(f"放量下跌(vol_ratio={vol_ratio:.1f}，真跌信号)")
            elif vol_ratio > 1.2:
                score += 11
                reasons.append(f"温和放量下跌(vol_ratio={vol_ratio:.1f})")
            elif vol_ratio > 0.8:
                score += 8
                reasons.append(f"正常量下跌(vol_ratio={vol_ratio:.1f})")
            elif vol_ratio > 0.5:
                score += 3
                reasons.append(f"缩量下跌(vol_ratio={vol_ratio:.1f}，洗盘特征)")
                is_wash = True
            else:
                score += 1.5
                reasons.append(f"极度缩量(vol_ratio={vol_ratio:.1f}，强洗盘特征)")
                is_wash = True
        else:
            # ── 上涨/横盘时不太需要卖出 ──
            if vol_ratio > 2.0:
                score += 6  # 涨但巨量，可能见顶
                reasons.append(f"上涨巨量(vol_ratio={vol_ratio:.1f}，警惕天量见顶)")
            else:
                score += 1.5
                reasons.append(f"非下跌日(vol_ratio={vol_ratio:.1f})")
                is_wash = True

        score = max(0.0, min(score, MAX_VOLUME))
        return SellScoreDetail(
            dimension="量能异动",
            score=round(score, 1),
            max_score=MAX_VOLUME,
            description="; ".join(reasons),
            is_wash=is_wash,
        )

    # ── 维度4: 大盘环境 (满分20) ─────────────────────────────────────

    def _score_market_environment(
        self,
        index_f1: Optional[float],
        index_f2: Optional[float],
        index_neg_days: Optional[int],
        sector_down: Optional[bool],
    ) -> SellScoreDetail:
        """大盘环境评分。

        评分规则：
        - 大盘 f''持续负 + 板块联动跌 → 系统性风险，高分
        - 大盘稳定 + 板块不弱 → 个股下跌可能是独立洗盘
        - 数据缺失时给中间分
        """
        has_index = index_f1 is not None and index_f2 is not None
        has_sector = sector_down is not None

        # 两个都没有：中间分
        if not has_index and not has_sector:
            return SellScoreDetail(
                dimension="大盘环境",
                score=NEUTRAL_MARKET,
                max_score=MAX_MARKET,
                description="无大盘/板块数据，给中间分",
                is_wash=False,
            )

        score = 0.0
        reasons = []
        is_wash = False
        max_possible = 0.0

        # ── 大盘分析 (满分12) ──
        if has_index:
            max_possible += 12
            if index_neg_days is not None and index_neg_days >= 5:
                score += 10
                reasons.append(f"大盘f''连续{index_neg_days}天负（系统性风险）")
            elif index_neg_days is not None and index_neg_days >= 3:
                score += 6
                reasons.append(f"大盘f''连续{index_neg_days}天负（大盘偏弱）")
            elif index_neg_days is not None and index_neg_days >= 1:
                score += 3
                reasons.append("大盘f''刚转负")
            else:
                score += 1
                reasons.append("大盘f''为正（大盘稳定）")
                is_wash = True

            # 大盘一阶导也负，更危险
            if index_f1 is not None and index_f1 < 0:
                score += 2
                reasons.append("大盘f'<0（大盘下跌趋势）")
                is_wash = False
        else:
            # 无大盘数据，给一半分
            max_possible += 12
            score += 6
            reasons.append("无大盘数据（给半分）")

        # ── 板块分析 (满分8) ──
        if has_sector:
            max_possible += 8
            if sector_down:
                score += 7
                reasons.append("板块联动下跌（行业性风险）")
            else:
                score += 1
                reasons.append("板块不弱")
                is_wash = True
        else:
            max_possible += 8
            score += 4
            reasons.append("无板块数据（给半分）")

        # 按比例缩放到满分20
        if max_possible > 0:
            score = score / max_possible * MAX_MARKET

        score = max(0.0, min(score, MAX_MARKET))
        return SellScoreDetail(
            dimension="大盘环境",
            score=round(score, 1),
            max_score=MAX_MARKET,
            description="; ".join(reasons),
            is_wash=is_wash,
        )

    # ── 维度5: 关键位破位 (满分20) ───────────────────────────────────

    def _score_key_level_break(
        self,
        below_ma20: bool,
        below_ma60: bool,
        cur_price: float,
        ma20: Optional[float],
        ma60: Optional[float],
    ) -> SellScoreDetail:
        """关键位破位评分（v3 版，满分20）。

        评分规则：
        - 破MA60: 趋势反转信号，高分（18分）
        - 破MA20: 短期趋势走坏，中等分（13分）
        - 逼近MA20（距离<3%）: 风险升高（5分）
        - 均未破: 安全，低分（1.5分）
        """
        if ma20 is None and ma60 is None:
            return SellScoreDetail(
                dimension="关键位破位",
                score=NEUTRAL_KEY_LEVEL,
                max_score=MAX_KEY_LEVEL,
                description="数据不足，无法计算均线，给中间分",
                is_wash=False,
            )

        score = 0.0
        reasons = []
        is_wash = False

        if below_ma60:
            score += 18
            pct_below = (cur_price / ma60 - 1) * 100 if ma60 else 0
            reasons.append(f"已破MA60({pct_below:+.1f}%)，趋势反转信号")
        elif below_ma20:
            score += 13
            pct_below = (cur_price / ma20 - 1) * 100 if ma20 else 0
            reasons.append(f"已破MA20({pct_below:+.1f}%)，短期走坏")
        else:
            # 在MA20之上，检查距离
            if ma20:
                pct_above = (cur_price / ma20 - 1) * 100
                if pct_above < 3.0:
                    score += 5
                    reasons.append(f"逼近MA20(仅+{pct_above:.1f}%)，风险升高")
                else:
                    score += 1.5
                    reasons.append(f"MA20之上(+{pct_above:.1f}%)，趋势完好")
                    is_wash = True
            else:
                score += 1.5
                reasons.append("MA20之上")
                is_wash = True

        score = max(0.0, min(score, MAX_KEY_LEVEL))
        return SellScoreDetail(
            dimension="关键位破位",
            score=round(score, 1),
            max_score=MAX_KEY_LEVEL,
            description="; ".join(reasons),
            is_wash=is_wash,
        )

    # ── 维度2: MACD多周期共振 (满分25) ───────────────────────────────

    @staticmethod
    def _compute_macd(
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算MACD指标（EMA递推，无前瞻偏差）。

        Args:
            close: 收盘价数组
            fast: 快线EMA周期（默认12）
            slow: 慢线EMA周期（默认26）
            signal: 信号线EMA周期（默认9）

        Returns:
            (macd_line/DIF, signal_line/DEA, histogram/MACD柱)
            histogram = 2 * (DIF - DEA)，负值表示DIF < DEA
        """
        if len(close) < slow + signal:
            n = len(close)
            return np.zeros(n), np.zeros(n), np.zeros(n)

        s = pd.Series(close.astype(float))
        ema_fast = s.ewm(span=fast, adjust=False).mean().values
        ema_slow = s.ewm(span=slow, adjust=False).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
        hist = (dif - dea) * 2

        return dif, dea, hist

    @staticmethod
    def _detect_bottom_divergence(
        close: np.ndarray, hist: np.ndarray, lookback: int = 40
    ) -> bool:
        """检测MACD底背离：价格创新低但MACD柱没创新低。

        在最近lookback根K线内找两个低点：
        - 价格：最近低点 < 前一个低点（价格创新低）
        - MACD柱：最近低点对应的hist > 前一个低点对应的hist（MACD没创新低）
        - 两个低点之间至少间隔10根K线
        """
        n = min(len(close), len(hist), lookback)
        if n < 15:
            return False

        close_recent = close[-n:]
        hist_recent = hist[-n:]

        # 用局部最小值窗口（5根K线）找低点
        window = 5
        lows = []  # (index_in_recent, close_val, hist_val)
        for i in range(window, n - window):
            local_min = True
            for j in range(i - window, i + window + 1):
                if j != i and close_recent[j] < close_recent[i]:
                    local_min = False
                    break
            if local_min:
                lows.append((i, close_recent[i], hist_recent[i]))

        # 需要至少2个低点
        if len(lows) < 2:
            return False

        # 取最近的两个低点
        recent_low = lows[-1]
        prev_low = lows[-2]

        # 两个低点之间至少间隔10根K线
        if recent_low[0] - prev_low[0] < 10:
            return False

        # 价格创新低 但 MACD柱没创新低 → 底背离
        if recent_low[1] < prev_low[1] and recent_low[2] > prev_low[2]:
            return True

        return False

    def _score_macd_resonance(
        self,
        daily_close: np.ndarray,
        minute_df: Optional[pd.DataFrame] = None,
    ) -> SellScoreDetail:
        """MACD多周期共振评分（满分25）。

        日线MACD信号：
        - MACD柱转负（hist < 0）: +3分（收紧，减少假阳性）
        - DIF死叉DEA（刚发生，从前一天上方穿越到下方）: +5分
        - 已经在死叉状态（非刚发生）: 不触发，不给分
        - MACD柱连续5天以上为负: +5分（持续确认，条件从3天收紧到5天）
        - 标准(12,26,9)和快速(5,34,5)都触发: +2分（双参数共振加成）

        60分钟MACD信号：
        - 60min MACD柱转负: 0分（太容易触发，已删除）
        - 60min DIF死叉DEA（刚发生）: +3分
        - 60min+日线MACD共振（都死叉）: +3分（核心信号）
        - 60min MACD底背离: -5分（洗盘对冲）
        """
        score = 0.0
        reasons = []
        is_wash = False

        # ═══ 日线MACD ═══
        # 标准参数 (12, 26, 9)
        std_triggered = False
        std_dead_cross = False
        std_hist_neg_5d = False

        if len(daily_close) >= 35:
            dif_std, dea_std, hist_std = self._compute_macd(daily_close, 12, 26, 9)

            # MACD柱转负 (hist < 0): 3分（从5分收紧）
            if hist_std[-1] < 0:
                score += 3
                reasons.append("日线MACD柱转负")
                std_triggered = True

            # DIF死叉DEA（DIF从上方穿越到下方，刚发生）
            if len(dif_std) >= 2:
                if dif_std[-1] < dea_std[-1] and dif_std[-2] >= dea_std[-2]:
                    score += 5
                    reasons.append("日线DIF死叉DEA")
                    std_dead_cross = True
                    std_triggered = True
                elif dif_std[-1] < dea_std[-1]:
                    # 已经在死叉状态（非刚发生）
                    # 只标记死叉状态用于共振判定，不触发得分
                    std_dead_cross = True

            # MACD柱连续5天以上为负（条件从3天收紧到5天）
            if len(hist_std) >= 5 and all(h < 0 for h in hist_std[-5:]):
                score += 5
                reasons.append("日线MACD柱连续≥5天负")
                std_hist_neg_5d = True
        else:
            reasons.append("日线数据不足（<35天），跳过标准MACD")

        # 快速参数 (5, 34, 5)
        fast_triggered = False
        if len(daily_close) >= 39:
            dif_fast, dea_fast, hist_fast = self._compute_macd(daily_close, 5, 34, 5)

            if hist_fast[-1] < 0:
                fast_triggered = True

            if len(dif_fast) >= 2:
                if dif_fast[-1] < dea_fast[-1] and dif_fast[-2] >= dea_fast[-2]:
                    fast_triggered = True
                elif dif_fast[-1] < dea_fast[-1]:
                    fast_triggered = True

        # 双参数共振加成（从3分降到2分）
        if std_triggered and fast_triggered:
            score += 2
            reasons.append("标准+快速双参数MACD共振")

        # 日线MACD无信号 → 偏洗盘
        if not std_triggered and not fast_triggered:
            reasons.append("日线MACD未触发卖出信号")
            is_wash = True

        # ═══ 60分钟MACD ═══
        min60_dead_cross = False
        min60_hist_neg = False

        if minute_df is not None and len(minute_df) >= 35:
            # 准备分钟数据
            min_df = self._prepare_dataframe(minute_df)
            if min_df is not None and "close" in min_df.columns:
                min_close = min_df["close"].values.astype(float)

                if len(min_close) >= 35:
                    dif_m, dea_m, hist_m = self._compute_macd(min_close, 12, 26, 9)

                    # 60min MACD柱转负: 已删除（0分，太容易触发）
                    if hist_m[-1] < 0:
                        min60_hist_neg = True

                    # 60min DIF死叉DEA（刚发生，从4分降到3分）
                    if len(dif_m) >= 2:
                        if dif_m[-1] < dea_m[-1] and dif_m[-2] >= dea_m[-2]:
                            score += 3
                            reasons.append("60min DIF死叉DEA")
                            min60_dead_cross = True
                        elif dif_m[-1] < dea_m[-1]:
                            min60_dead_cross = True  # 已经在死叉状态

                    # 60min+日线MACD共振加成（从2分升到3分，核心信号）
                    if min60_dead_cross and std_dead_cross:
                        score += 3
                        reasons.append("60min+日线MACD多周期共振")

                    # 60min MACD底背离检测：价格创新低但MACD柱没创新低 → 洗盘
                    if self._detect_bottom_divergence(min_close, hist_m, lookback=40):
                        score -= 5
                        reasons.append("60min MACD底背离（洗盘信号）")

                    if not min60_dead_cross and not min60_hist_neg:
                        reasons.append("60min MACD未触发")
                else:
                    # 分钟数据不足，从5分降到3分
                    score += 3
                    reasons.append("60min数据不足35根K线，给中间分")
            else:
                score += 3
                reasons.append("60min数据格式异常，给中间分")
        else:
            # 无60min数据 → 从5分降到3分
            score += 3
            reasons.append("无60min数据，MACD部分给中间分")

        score = max(0.0, min(score, MAX_MACD))
        return SellScoreDetail(
            dimension="MACD共振",
            score=round(score, 1),
            max_score=MAX_MACD,
            description="; ".join(reasons) if reasons else "无MACD信号",
            is_wash=is_wash,
        )

    @staticmethod
    def _check_iron_3S(close: np.ndarray, macd_description: str) -> bool:
        """检查铁律3S三条件是否同时满足。

        铁律3S：建议减仓必须同时满足三个条件——
        ① MA5死叉MA10且维持≥3天
        ② MACD柱转负或DIF死叉DEA
        ③ 收盘破MA20

        Returns:
            True: 三个条件同时满足（真跌概率极高）
        """
        if len(close) < 20:
            return False

        close_f = close.astype(float)

        # 条件①: MA5死叉MA10且维持≥3天
        ma5_cross_ma10 = False
        if len(close_f) >= 10:
            ma5 = np.convolve(close_f, np.ones(5) / 5, mode='valid')
            ma10 = np.convolve(close_f, np.ones(10) / 10, mode='valid')
            # ma5和ma10对齐后，ma5的末尾对应close[-5:]
            # ma10长度 = len(close)-9, ma5长度 = len(close)-4
            # 对齐：取ma5的后(len(ma10))个值
            offset = len(ma5) - len(ma10)
            ma5_aligned = ma5[offset:]
            # 检查MA5 < MA10连续≥3天
            below_count = 0
            for i in range(len(ma5_aligned) - 1, -1, -1):
                if ma5_aligned[i] < ma10[i]:
                    below_count += 1
                else:
                    break
            ma5_cross_ma10 = below_count >= 3

        # 条件②: MACD柱转负或DIF死叉DEA（从macd_description解析）
        macd_sell = False
        if "MACD柱转负" in macd_description or "DIF死叉DEA" in macd_description:
            macd_sell = True

        # 条件③: 收盘破MA20
        cur_price = float(close_f[-1])
        ma20 = float(np.mean(close_f[-20:]))
        below_ma20 = cur_price < ma20

        # 三个条件同时满足
        return ma5_cross_ma10 and macd_sell and below_ma20

    # ── 洗盘概率估计 ─────────────────────────────────────────────────

    @staticmethod
    def _estimate_wash_probability(
        vol_ratio: Optional[float],
        is_price_down: bool,
        index_f2: Optional[float],
        index_neg_days: Optional[int],
        sector_down: Optional[bool],
        below_ma20: bool,
        f2_val: float,
        f2_neg_days: int,
    ) -> float:
        """估计洗盘概率 (0-1) — v2 收紧版。

        洗盘判别必须同时满足四项硬条件：
        1. 缩量下跌 (vol_ratio < 0.8)
        2. 不破MA20 (below_ma20 == False)
        3. 大盘强 (index_neg_days <= 1)
        4. f''明显转正 (f2_val > 0)

        四项全满足 → 洗盘概率 0.8-0.9
        满足3项 → 洗盘概率 0.5-0.6
        满足≤2项 → 洗盘概率 ≤0.3
        """
        # 计算各条件是否满足
        shrink_vol = (vol_ratio is not None and is_price_down and vol_ratio < 0.8)
        above_ma20 = not below_ma20
        strong_market = (index_neg_days is not None and index_neg_days <= 1)
        f2_positive = (f2_val > 0)

        conditions_met = sum([shrink_vol, above_ma20, strong_market, f2_positive])

        # 全满足 → 高概率洗盘
        if conditions_met >= 4:
            return 0.85
        elif conditions_met == 3:
            return 0.55
        elif conditions_met == 2:
            return 0.30
        else:
            return 0.10

    # ── 动作判断 ──────────────────────────────────────────────────────

    @staticmethod
    def _determine_action(total_score: float, wash_prob: float) -> str:
        """根据总分和洗盘概率确定操作建议。"""
        if total_score >= THRESHOLD_REDUCE:
            if wash_prob < 0.3:
                return "减仓（真跌信号强）"
            return "减仓"
        elif total_score >= THRESHOLD_WATCH:
            if wash_prob > 0.5:
                return "可能是洗盘"  # 分数不高且洗盘概率大
            return "观望"
        else:
            return "可能是洗盘"

    # ── 置信度计算 ────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        has_index: bool,
        has_sector: bool,
        has_volume: bool,
        data_length: int,
    ) -> float:
        """计算置信度 (0-1)。

        数据越完整，置信度越高。
        """
        confidence = 0.5  # 基础置信度

        # 数据完整性加分
        if has_volume:
            confidence += 0.15
        if has_index:
            confidence += 0.15
        if has_sector:
            confidence += 0.10

        # 数据量加分
        if data_length >= 120:
            confidence += 0.10
        elif data_length >= 90:
            confidence += 0.05

        return min(confidence, 1.0)

    # ── 信号类型描述 ──────────────────────────────────────────────────

    @staticmethod
    def _describe_signal_type(
        f1_val: float,
        f2_val: float,
        vol_ratio: Optional[float],
        wash_prob: float,
    ) -> str:
        """生成人类可读的信号类型描述。"""
        if f1_val > 0 and f2_val < 0:
            base = "⚠️ 减速见顶"
        elif f1_val < 0 and f2_val < 0:
            base = "🔴 加速下跌"
        elif f1_val < 0 and f2_val > 0:
            base = "🔄 底部减速"
        elif f1_val > 0 and f2_val > 0:
            base = "🚀 主升浪"
        else:
            base = "➡️ 横盘"

        if wash_prob > 0.6:
            suffix = "（偏洗盘）"
        elif wash_prob < 0.3:
            suffix = "（偏真跌）"
        else:
            suffix = ""

        vol_desc = ""
        if vol_ratio is not None:
            if vol_ratio > 1.5:
                vol_desc = " [放量]"
            elif vol_ratio < 0.8:
                vol_desc = " [缩量]"

        return f"{base}{vol_desc}{suffix}"

    # ── 技术计算辅助 ──────────────────────────────────────────────────

    def _compute_derivatives(
        self,
        close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对 ln(price) 求一阶和二阶导（S-G滤波器）。

        ⚠️ shift(sg_window//2) 防止未来数据泄漏。

        Returns:
            (f1, f2): 一阶导(×100) 和 二阶导(×10000)
        """
        ln_p = np.log(close.astype(float))
        window = min(self.sg_window, len(close) // 2 * 2 + 1)
        if window < 5:
            return np.zeros(len(close)), np.zeros(len(close))

        f1 = savgol_filter(ln_p, window, self.sg_poly, deriv=1) * 100
        f2 = savgol_filter(ln_p, window, self.sg_poly, deriv=2) * 10000

        # shift 防止未来数据泄漏
        shift = window // 2
        f1_safe = np.empty_like(f1)
        f2_safe = np.empty_like(f2)
        f1_safe[:shift] = f1[:shift]
        f2_safe[:shift] = f2[:shift]
        f1_safe[shift:] = f1[:-shift]
        f2_safe[shift:] = f2[:-shift]

        return f1_safe, f2_safe

    @staticmethod
    def _count_consecutive_negative(arr: np.ndarray) -> int:
        """从数组末尾向前数连续负值的个数。"""
        count = 0
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] < 0:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _compute_volume_ratio(
        volume: Optional[np.ndarray],
    ) -> Optional[float]:
        """计算量能比: 当日成交量 / MA(volume, 20)。"""
        if volume is None or len(volume) < 21:
            return None
        ma20_vol = np.mean(volume[-21:-1])  # 前20天的均量
        if ma20_vol <= 0:
            return None
        return float(volume[-1] / ma20_vol)

    @staticmethod
    def _compute_ma(close: np.ndarray, period: int) -> Optional[float]:
        """计算简单移动平均线（最新值）。"""
        if len(close) < period:
            return None
        return float(np.mean(close[-period:]))

    def _analyze_index(
        self,
        index_daily: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float], Optional[int]]:
        """分析大盘日线数据，返回 (f1, f2, f2连续负天数)。"""
        try:
            df = self._prepare_dataframe(index_daily)
            if df is None or len(df) < 30:
                return None, None, None
            close = df["close"].values.astype(float)
            f1, f2 = self._compute_derivatives(close)
            neg_days = self._count_consecutive_negative(f2)
            return float(f1[-1]), float(f2[-1]), neg_days
        except Exception:
            return None, None, None

    def _analyze_sector(
        self,
        sector_daily: pd.DataFrame,
    ) -> Optional[bool]:
        """分析板块日线数据，返回板块是否在下跌。"""
        try:
            df = self._prepare_dataframe(sector_daily)
            if df is None or len(df) < 5:
                return None
            close = df["close"].values.astype(float)
            # 板块近期是否下跌：5日收益率 < -2%
            if len(close) >= 5:
                ret5 = (close[-1] / close[-5] - 1)
                return ret5 < -0.02
            return None
        except Exception:
            return None

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """准备DataFrame：确保升序排列、包含必要列。"""
        if df is None or df.empty:
            return None

        # 复制避免修改原数据
        df = df.copy()

        # 确定日期列
        date_col = None
        for col in ["trade_date", "date", "datetime"]:
            if col in df.columns:
                date_col = col
                break

        # 按日期升序排列
        if date_col is not None:
            df = df.sort_values(date_col).reset_index(drop=True)

        # 标准化列名（vol → volume）
        if "vol" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"vol": "volume"})

        # 检查必要列
        if "close" not in df.columns:
            return None

        return df

    # ── 格式化报告 ────────────────────────────────────────────────────

    @staticmethod
    def format_report(result: SellSignalResult) -> str:
        """将评分结果格式化为可读报告。"""
        lines = [
            f"🔴 卖出信号评分: {result.code}",
            f"{'─' * 50}",
            f"总分: {result.score:.1f}/100 | 操作: {result.action} | "
            f"置信度: {result.confidence:.0%}",
            f"信号类型: {result.signal_type}",
            f"洗盘概率: {result.is_wash_probability:.0%}",
            f"",
            f"── 各维度评分 ──",
        ]

        for d in result.details:
            wash_tag = "🟢洗盘" if d.is_wash else "🔴真跌"
            pct = d.score / d.max_score * 100 if d.max_score > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(
                f"  {d.dimension}({d.max_score:.0f}分): "
                f"{d.score:.1f} [{bar}] {wash_tag}"
            )
            lines.append(f"    → {d.description}")

        # 铁律3S加成
        ind = result.indicators
        if ind and ind.get("iron_3S_triggered"):
            lines.append(f"  🔥 铁律3S加成: +{IRON_5B_BONUS}分（三条件同时满足）")

        # 关键指标
        if ind:
            lines.append("")
            lines.append("── 关键指标 ──")
            lines.append(f"  f' = {ind.get('f1', 'N/A')}")
            lines.append(f"  f'' = {ind.get('f2', 'N/A')}")
            lines.append(f"  f''连续负天数 = {ind.get('f2_neg_days', 'N/A')}")
            lines.append(f"  量能比 = {ind.get('vol_ratio', 'N/A')}")
            lines.append(
                f"  破MA20 = {'是' if ind.get('below_ma20') else '否'} | "
                f"破MA60 = {'是' if ind.get('below_ma60') else '否'}"
            )
            # MACD详情
            macd_detail = ind.get('macd_detail')
            if macd_detail:
                lines.append(f"  MACD详情: {macd_detail}")
            idx_neg = ind.get('index_f2_neg_days')
            if idx_neg is not None:
                lines.append(f"  大盘f''连续负天数 = {idx_neg}")
            sec_down = ind.get('sector_down')
            if sec_down is not None:
                lines.append(f"  板块联动下跌 = {'是' if sec_down else '否'}")

        return "\n".join(lines)
