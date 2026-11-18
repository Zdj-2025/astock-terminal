#!/usr/bin/env python3
"""多周期导数确认系统 (Multi-Timeframe Derivative Confirmation System).

核心思想：真正的趋势转折从短周期开始，逐级传导到长周期。
在日线/60分钟/15分钟三个周期上分别计算 f' 和 f''，
通过周期一致性和统计置信度来区分真正的拐点和噪声。

两大改进方向：
  1. 多周期一致性：D/60min/15min 三级导数同向确认
  2. 统计置信度：Z-score、信噪比、信号持续性检验

与 V3DerivativeScorer 的关系：
  - 本模块不替代 V3，而是在 V3 基础上增加一个增强层
  - V3 的日线 f'/f'' 保持不变，本模块追加多周期维度
  - 最终评分 = V3原始分 × MTF置信系数

用法：
  from scripts.screener.multi_timeframe_scorer import MultiTimeframeScorer
  mtf = MultiTimeframeScorer(data_root=Path("data/store"))
  result = mtf.score("603881")
  print(result)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from scipy.signal import savgol_filter

SG_WINDOW = 21
SG_POLY = 3


@dataclass
class TFResult:
    """单周期导数结果。"""
    timeframe: str = ""
    f1: float = 0.0
    f2: float = 0.0
    f1_zscore: float = 0.0
    f2_zscore: float = 0.0
    f1_snr: float = 0.0
    f2_snr: float = 0.0
    f2_persistence: float = 0.0
    n_bars: int = 0


@dataclass
class MTFResult:
    """多周期导数确认结果。"""
    daily: TFResult = field(default_factory=TFResult)
    hourly: TFResult = field(default_factory=TFResult)
    min15: TFResult = field(default_factory=TFResult)
    # 综合指标
    consensus_score: float = 0.0
    consensus_level: str = "无确认"
    confidence: float = 0.0
    signal_quality: str = "低"
    # 详情
    details: dict = field(default_factory=dict)


class MultiTimeframeScorer:
    """多周期导数确认评分器。"""

    def __init__(self, data_root: str | Path | None = None):
        self.data_root = Path(data_root) if data_root else None

    @staticmethod
    def _compute_tf_derivatives(
        close: np.ndarray,
        sg_window: int = SG_WINDOW,
        sg_poly: int = SG_POLY,
        lookback: int = 60,
    ) -> TFResult:
        """对单个时间周期的收盘价计算导数 + 统计置信度。

        Args:
            close: 收盘价序列
            sg_window: Savgol 窗口
            sg_poly: Savgol 多项式阶数
            lookback: 统计置信度的回看窗口长度
        """
        n = len(close)
        result = TFResult(n_bars=n)

        if n < sg_window + 3:
            return result

        ln_p = np.log(close.astype(float))
        from scripts.core.derivatives import causal_savgol  # 2026-09-10: 权威因果(pos=0右端), 弃中心化+shift滞后
        f1 = causal_savgol(ln_p, sg_window, sg_poly, deriv=1) * 100
        f2 = causal_savgol(ln_p, sg_window, sg_poly, deriv=2) * 10000

        result.f1 = float(f1[-1])
        result.f2 = float(f2[-1])

        # 统计置信度(因果导数前 sg_window-1 位为NaN, 起点从首个有效位取)
        hist_start = max(sg_window - 1, n - lookback)
        if n - hist_start >= 10:
            f1_hist = f1[hist_start:]
            f2_hist = f2[hist_start:]

            f1_mean = np.mean(f1_hist)
            f1_std = np.std(f1_hist)
            f2_mean = np.mean(f2_hist)
            f2_std = np.std(f2_hist)

            result.f1_zscore = float((result.f1 - f1_mean) / f1_std) if f1_std > 1e-8 else 0.0
            result.f2_zscore = float((result.f2 - f2_mean) / f2_std) if f2_std > 1e-8 else 0.0
            result.f1_snr = float(abs(result.f1) / f1_std) if f1_std > 1e-8 else 0.0
            result.f2_snr = float(abs(result.f2) / f2_std) if f2_std > 1e-8 else 0.0

            # f'' 持续性：过去 N 根中 f''>0 的比例
            recent = min(10, len(f2_hist))
            result.f2_persistence = float(np.sum(f2_hist[-recent:] > 0) / recent)

        return result

    def _load_kline(self, code: str, freq: str, cutoff_date: str | None = None) -> pd.DataFrame | None:
        """加载 K 线数据。"""
        if self.data_root is None:
            return None
        path = self.data_root / "kline" / code / f"{freq}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if "datetime" in df.columns:
                dt_col = "datetime"
            elif "trade_date" in df.columns:
                dt_col = "trade_date"
            else:
                dt_col = "date"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)
            if cutoff_date is not None:
                df = df[df[dt_col] <= pd.to_datetime(cutoff_date)].reset_index(drop=True)
            return df
        except Exception:
            return None

    def score(self, code: str, df_daily: pd.DataFrame | None = None,
              cutoff_date: str | None = None) -> MTFResult:
        """对单只股票做多周期导数评估。

        Args:
            code: 纯数字代码
            df_daily: 可选日线 DataFrame（不提供则从 data_root 加载）
            cutoff_date: 截断日期（YYYY-MM-DD），分钟数据只用到此日期（含）
        """
        result = MTFResult()

        # ── 日线 ──
        if df_daily is not None:
            close_d = df_daily["close"].values.astype(float)
            daily_last_dt = None
            if "trade_date" in df_daily.columns:
                daily_last_dt = pd.to_datetime(df_daily.iloc[-1]["trade_date"])
            elif "date" in df_daily.columns:
                daily_last_dt = pd.to_datetime(df_daily.iloc[-1]["date"])
        else:
            df = self._load_kline(code, "D")
            if df is None:
                return result
            close_d = df["close"].values.astype(float)
            dt_col = "trade_date" if "trade_date" in df.columns else "date"
            daily_last_dt = pd.to_datetime(df.iloc[-1][dt_col])

        result.daily = self._compute_tf_derivatives(close_d)
        result.daily.timeframe = "D"

        # ── 60分钟 ──
        df60 = self._load_kline(code, "60min", cutoff_date=cutoff_date)
        if df60 is not None and len(df60) >= SG_WINDOW + 3:
            dt60 = pd.to_datetime(df60.iloc[-1]["datetime"])
            result.details["min_data_stale"] = False
            if daily_last_dt is not None:
                gap_days = (daily_last_dt - dt60).days
                if gap_days > 2:
                    result.details["min_data_stale"] = True
                    result.details["min_data_gap_days"] = gap_days
            close_h = df60["close"].values.astype(float)
            result.hourly = self._compute_tf_derivatives(close_h)
            result.hourly.timeframe = "60min"

        # ── 15分钟 ──
        df15 = self._load_kline(code, "15min", cutoff_date=cutoff_date)
        if df15 is not None and len(df15) >= SG_WINDOW + 3:
            close_m = df15["close"].values.astype(float)
            result.min15 = self._compute_tf_derivatives(close_m)
            result.min15.timeframe = "15min"

        # ── 综合评估 ──
        self._compute_consensus(result)

        # 数据陈旧时降级
        if result.details.get("min_data_stale"):
            result.details["stale_note"] = "分钟数据滞后，60min/15min信号权重降低"
            result.confidence *= 0.7
            result.consensus_score *= 0.8

        return result

    def _compute_consensus(self, result: MTFResult) -> None:
        """计算多周期共识分和统计置信度。"""
        # 收集有效周期
        tf_results = []
        if result.daily.n_bars > 0:
            tf_results.append(result.daily)
        if result.hourly.n_bars > 0:
            tf_results.append(result.hourly)
        if result.min15.n_bars > 0:
            tf_results.append(result.min15)

        n_valid = len(tf_results)
        if n_valid < 2:
            # 只有日线数据，无法做多周期确认
            result.consensus_score = 0.3
            result.consensus_level = "数据不足"
            result.confidence = 0.3
            result.signal_quality = "低(无分钟数据)"
            result.details = {"n_timeframes": n_valid}
            return

        # ── 1. 方向一致性 (Direction Consensus) ──
        # f''>0 的周期数
        f2_positive_count = sum(1 for t in tf_results if t.f2 > 0)
        f1_positive_count = sum(1 for t in tf_results if t.f1 > 0)

        # f''>0 的比例（0-1）
        f2_consensus = f2_positive_count / n_valid
        f1_consensus = f1_positive_count / n_valid

        # ── 2. 统计置信度加权 ──
        # 每个周期的 f'' z-score 平均值
        f2_zscores = [t.f2_zscore for t in tf_results]
        avg_f2_zscore = np.mean(f2_zscores)

        # f'' 持续性加权平均
        f2_persistences = [t.f2_persistence for t in tf_results]
        avg_f2_persistence = np.mean(f2_persistences)

        # ── 3. 周期传导检测 (Cascade Detection) ──
        # 真正的趋势变化：短周期先转，长周期后转
        cascade_score = 0.0
        if n_valid >= 2:
            # 短周期(15min/60min)的 f'' 应该更早转正
            # 检测：短周期 f2 > 长周期 f2（短周期更敏感）
            short_tfs = [t for t in tf_results if t.timeframe in ("15min", "60min")]
            if short_tfs:
                short_f2_avg = np.mean([t.f2 for t in short_tfs])
                long_f2 = result.daily.f2
                # 短周期加速度更强 = 传导正在进行
                if short_f2_avg > 0 and long_f2 > 0:
                    cascade_score = min(short_f2_avg / max(abs(long_f2), 0.001), 2.0) / 2.0
                elif short_f2_avg > 0 and long_f2 <= 0:
                    # 短周期已转正但日线还没跟 = 早期信号
                    cascade_score = 0.3

        # ── 4. 综合评分 ──
        # consensus_score = f''方向一致性 × 统计置信 × 传导分数
        # 范围 0-1
        stat_confidence = self._sigmoid(avg_f2_zscore, k=0.8)  # z-score → 0-1

        raw_consensus = f2_consensus * 0.5 + stat_confidence * 0.3 + cascade_score * 0.2
        raw_consensus = min(max(raw_consensus, 0.0), 1.0)

        result.consensus_score = round(raw_consensus, 3)

        # ── 5. 分级标签 ──
        if f2_positive_count == n_valid and avg_f2_zscore > 0.5:
            result.consensus_level = "强确认"
        elif f2_positive_count >= n_valid - 1 and avg_f2_zscore > 0:
            result.consensus_level = "部分确认"
        elif f2_positive_count >= n_valid - 1:
            result.consensus_level = "弱确认"
        elif f2_positive_count == 1 and n_valid >= 3:
            result.consensus_level = "单周期信号"
        else:
            result.consensus_level = "无共识"

        # ── 6. 综合置信度 ──
        # 结合 f''方向性 + z-score + 持续性 + 传导
        confidence = (
            f2_consensus * 0.30
            + self._sigmoid(avg_f2_zscore, k=0.6) * 0.25
            + avg_f2_persistence * 0.25
            + cascade_score * 0.20
        )
        result.confidence = round(min(max(confidence, 0.0), 1.0), 3)

        # ── 7. 信号质量标签 ──
        if result.confidence >= 0.7 and f2_positive_count >= n_valid - 1:
            result.signal_quality = "高"
        elif result.confidence >= 0.5 and f2_positive_count >= n_valid - 1:
            result.signal_quality = "中"
        elif result.confidence >= 0.4:
            result.signal_quality = "中低"
        else:
            result.signal_quality = "低"

        result.details = {
            "n_timeframes": n_valid,
            "f2_positive_count": f2_positive_count,
            "f1_positive_count": f1_positive_count,
            "f2_consensus": round(f2_consensus, 2),
            "avg_f2_zscore": round(float(avg_f2_zscore), 3),
            "avg_f2_persistence": round(float(avg_f2_persistence), 3),
            "cascade_score": round(float(cascade_score), 3),
        }

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0) -> float:
        """Sigmoid 映射，将 z-score 类指标归一化到 0-1。"""
        return 1.0 / (1.0 + np.exp(-k * x))

    def compute_mtf_penalty(self, result: MTFResult) -> float:
        """基于多周期置信度计算惩罚系数（用于调整 V3 原始分）。

        Returns:
            penalty: 0.3 ~ 1.0
            - 1.0: 多周期强确认，不降分
            - 0.7: 部分确认，轻微降分
            - 0.5: 弱确认或无分钟数据，显著降分
            - 0.3: 无共识，大幅降分
        """
        if result.consensus_level == "数据不足":
            return 0.7  # 没有分钟数据，不能确认但也不能否定

        if result.consensus_level == "强确认":
            return 1.0
        elif result.consensus_level == "部分确认":
            return 0.9
        elif result.consensus_level == "弱确认":
            return 0.7
        elif result.consensus_level == "单周期信号":
            return 0.5
        else:  # 无共识
            return 0.3

    def score_batch(
        self,
        codes: list[str],
        df_dict: dict[str, pd.DataFrame] | None = None,
        cutoff_date: str | None = None,
    ) -> pd.DataFrame:
        """批量多周期评分。

        Args:
            codes: 股票代码列表
            df_dict: 可选预加载数据
            cutoff_date: 截断日期，分钟数据只用到此日期（含）
        """
        rows = []
        for code in codes:
            try:
                df_daily = None
                if df_dict and code in df_dict:
                    df_daily = df_dict[code]
                elif self.data_root:
                    path = self.data_root / "kline" / code / "D.parquet"
                    if not path.exists():
                        continue
                    df_daily = pd.read_parquet(path)
                    dt_col = "trade_date" if "trade_date" in df_daily.columns else "date"
                    df_daily[dt_col] = pd.to_datetime(df_daily[dt_col])
                    df_daily = df_daily.sort_values(dt_col).reset_index(drop=True)

                r = self.score(code, df_daily, cutoff_date=cutoff_date)
                rows.append({
                    "code": code,
                    "mtf_consensus": r.consensus_score,
                    "mtf_confidence": r.confidence,
                    "mtf_level": r.consensus_level,
                    "mtf_quality": r.signal_quality,
                    "mtf_penalty": self.compute_mtf_penalty(r),
                    "d_f1": r.daily.f1,
                    "d_f2": r.daily.f2,
                    "d_f2_zscore": r.daily.f2_zscore,
                    "d_f2_persistence": r.daily.f2_persistence,
                    "h_f1": r.hourly.f1,
                    "h_f2": r.hourly.f2,
                    "h_f2_zscore": r.hourly.f2_zscore,
                    "h_f2_persistence": r.hourly.f2_persistence,
                    "m_f1": r.min15.f1,
                    "m_f2": r.min15.f2,
                    "m_f2_zscore": r.min15.f2_zscore,
                    "m_f2_persistence": r.min15.f2_persistence,
                    "n_tf": r.details.get("n_timeframes", 0),
                })
            except Exception:
                continue

        return pd.DataFrame(rows)

    def augment_v3_batch(
        self,
        v3_batch: pd.DataFrame,
        cutoff_date: str | None = None,
    ) -> pd.DataFrame:
        """用多周期确认增强 V3 评分结果。

        Args:
            v3_batch: score_dual_branch() 输出的 DataFrame
            cutoff_date: 截断日期，分钟数据只用到此日期（含）
        """
        if len(v3_batch) == 0 or "code" not in v3_batch.columns:
            return v3_batch

        codes = v3_batch["code"].tolist()
        mtf_df = self.score_batch(codes, cutoff_date=cutoff_date)

        if len(mtf_df) == 0:
            return v3_batch

        # 合并 MTF 数据
        merge_cols = [
            "code", "mtf_consensus", "mtf_confidence", "mtf_level",
            "mtf_quality", "mtf_penalty",
            "d_f2_zscore", "h_f2", "h_f2_zscore", "m_f2", "m_f2_zscore",
            "n_tf",
        ]
        available_cols = [c for c in merge_cols if c in mtf_df.columns]
        result = v3_batch.merge(
            mtf_df[available_cols], on="code", how="left"
        )

        # 填充没有分钟数据的股票
        result["mtf_level"] = result["mtf_level"].fillna("数据不足")
        result["mtf_quality"] = result["mtf_quality"].fillna("低(无分钟数据)")
        result["mtf_penalty"] = result["mtf_penalty"].fillna(0.7)
        result["mtf_confidence"] = result["mtf_confidence"].fillna(0.3)

        # 用 MTF penalty 调整 v3_total
        if "v3_total" in result.columns:
            result["v3_total_original"] = result["v3_total"]
            result["v3_total"] = (result["v3_total"] * result["mtf_penalty"]).round(2)

        return result
