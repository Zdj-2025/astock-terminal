#!/usr/bin/env python3
"""V3 Derivative Scorer — 二阶导数核心评分系统 + 惩罚函数。

用 Savitzky-Golay 滤波器对 ln(price) 求导：
  f'(t)  = 一阶导（趋势方向/涨速）
  f''(t) = 二阶导（加速度/曲率/拐点）

全市场回测 IC=+0.341, v3>=6 胜率 80.3%, v3>=7 胜率 85.6%。

评分公式（满分10）：
  (位置距离(0-2) + 一阶导趋势(0-3) + 二阶导曲率(0-3) + 分钟信号(0-2)) × penalty

惩罚函数（方案一，2026-05-24上线）：
  penalty = 1 / (1 + exp(5 * (ret20/30 - CI)))
  CI = 涨势延续性指数 = 0.40×f''趋势 + 0.20×量能配合 + 0.40×位置空间
  暴涨股(20日涨幅>30%)被智能降分，CI高的温和惩罚，CI低的强惩罚。
  (v3.5: 量能权重从35%降到20%，位置从30%升到40%)

信号分类：
  🚀 主升浪: f'>0 且 f''>0 → 持有/加仓
  🔄 底部拐点: f'<0 但 f''>0 → 关注等f'转正
  ⚠️ 减速见顶: f'>0 但 f''<0 → 警惕
  🔴 加速下跌: f'<0 且 f''<0 → 回避

v3.5 更新（2026-05-24）：
  - 底部反转三级确认（L1预警/L2关注/L3确认买入）
  - 顶部卖出信号（减速+MACD绿柱+高位）
  - CI权重调整（量能35%→20%，位置30%→40%）
  - PB加分移除（因果验证确认是伪相关，动量效应代理）
  - 顶背离减分加大（-0.5→-1.5）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from scipy.signal import argrelextrema
from scripts.core.derivatives import causal_savgol   # 2026-08-19 P0-1C: 唯一权威因果实现


# ── 默认参数 ──────────────────────────────────────────────────────────

DEFAULT_SG_WINDOW = 21   # S-G 窗口（约4周日线）
DEFAULT_SG_POLY = 3      # 三次多项式


@dataclass
class V3Score:
    """V3 评分结果。"""
    total: float = 0.0
    raw_total: float = 0.0  # 惩罚前原始分
    penalty: float = 1.0    # 惩罚系数 (0~1)
    ci: float = 0.0         # 涨势延续性指数
    position_score: float = 0.0
    trend_score: float = 0.0
    curvature_score: float = 0.0
    minute_score: float = 0.0
    f1: float = 0.0       # 一阶导值
    f2: float = 0.0       # 二阶导值
    signal: str = "观望"
    details: dict = field(default_factory=dict)


class V3DerivativeScorer:
    """V3 二阶导数评分器。"""

    # HMM 市场状态权重配置
    HMM_WEIGHTS = {
        'trend': {  # 🟢趋势态: 趋势加速全力，底部正常
            'accel_weight': 1.0,
            'reversal_weight': 1.0,
            'accel_min_score': 4.0,
            'reversal_min_score_ratio': 0.7,
        },
        'range': {  # 🟡震荡态: 趋势降权，底部略升
            'accel_weight': 0.85,
            'reversal_weight': 1.1,
            'accel_min_score': 5.0,
            'reversal_min_score_ratio': 0.6,
        },
        'panic': {  # 🔴恐慌态: 趋势大幅降权，底部重点关注
            'accel_weight': 0.6,
            'reversal_weight': 1.3,
            'accel_min_score': 6.0,
            'reversal_min_score_ratio': 0.5,
        },
    }

    def __init__(
        self,
        sg_window: int = DEFAULT_SG_WINDOW,
        sg_poly: int = DEFAULT_SG_POLY,
        data_root: str | Path | None = None,
        use_kalman: bool = False,
    ):
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.data_root = Path(data_root) if data_root else None
        self.use_kalman = use_kalman
        self._hmm_model = None
        self._hmm_state_names = None

    # ── 核心：从日线数据计算导数 ─────────────────────────────────────

    @staticmethod
    def compute_derivatives(
        close: np.ndarray,
        sg_window: int = DEFAULT_SG_WINDOW,
        sg_poly: int = DEFAULT_SG_POLY,
        use_kalman: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对 ln(price) 求一阶和二阶导。

        默认用 S-G 滤波器（中心对称，需 shift 消除 look-ahead）。
        use_kalman=True 时用 Kalman 2-state 滤波器（天然因果，无滞后）。

        Returns:
            (f1, f2): 一阶导(×100, ≈日收益率%), 二阶导(×10000, 无量纲)
        """
        if use_kalman:
            from scripts.analyzers.kalman_derivative_filter import kalman_derivatives
            return kalman_derivatives(close)

        ln_p = np.log(close.astype(float))
        # 2026-08-19 P0-1C: 中心化savgol_filter+shift补丁(双重错误)→core.derivatives.causal_savgol
        # 因果左导数(pos=0右端无未来);原"假IC0.459/真实0.029"结论均作废(见memory scipy_savgol_pos_semantics)
        f1 = causal_savgol(ln_p, sg_window, sg_poly, deriv=1) * 100
        f2 = causal_savgol(ln_p, sg_window, sg_poly, deriv=2) * 10000
        # 前window-1位NaN→用首有效值回填(下游rolling/索引安全)
        _first = sg_window - 1
        if len(f1) > _first:
            f1[:_first] = f1[_first]
            f2[:_first] = f2[_first]

        return f1, f2

    @staticmethod
    def compute_derivatives_dual(
        close: np.ndarray,
        slow_window: int = 21,
        fast_window: int = 11,
        sg_poly: int = DEFAULT_SG_POLY,
        use_kalman: bool = False,
    ) -> dict:
        """双窗口导数计算 — 慢通道(稳定) + 快通道(响应)。

        回测: 快慢一致确认趋势加速胜率80.8%(vs慢通道单独76.2%)。
        快通道IC低但作为确认信号非常有效——过滤弱信号。

        use_kalman=True: 用不同的 q_scale 模拟快慢通道。
          - slow: q_scale=1e-5 (平滑，类似 window=21)
          - fast: q_scale=5e-5 (响应快，类似 window=11)

        Returns:
            dict with f1_slow, f2_slow, f1_fast, f2_fast
        """
        if use_kalman:
            from scripts.analyzers.kalman_derivative_filter import kalman_derivatives
            f1_slow, f2_slow = kalman_derivatives(close, q_scale=1e-5)
            f1_fast, f2_fast = kalman_derivatives(close, q_scale=5e-5)
            return {
                'f1_slow': f1_slow, 'f2_slow': f2_slow,
                'f1_fast': f1_fast, 'f2_fast': f2_fast,
            }

        ln_p = np.log(close.astype(float))
        # 2026-08-19 P0-1C: causal_savgol替换(原中心化+shift双重错误,见memory scipy_savgol_pos_semantics)
        f1_slow = causal_savgol(ln_p, slow_window, sg_poly, deriv=1) * 100
        f2_slow = causal_savgol(ln_p, slow_window, sg_poly, deriv=2) * 10000
        f1_fast = causal_savgol(ln_p, fast_window, sg_poly, deriv=1) * 100
        f2_fast = causal_savgol(ln_p, fast_window, sg_poly, deriv=2) * 10000
        # NaN回填(前window-1位)
        for _arr, _win in [(f1_slow, slow_window), (f2_slow, slow_window),
                           (f1_fast, fast_window), (f2_fast, fast_window)]:
            _f = _win - 1
            if len(_arr) > _f:
                _arr[:_f] = _arr[_f]

        return {
            'f1_slow': f1_slow, 'f2_slow': f2_slow,
            'f1_fast': f1_fast, 'f2_fast': f2_fast,
        }

    # ── 位置距离分 (0-2) ─────────────────────────────────────────────

    @staticmethod
    def score_position(close: np.ndarray, idx: int = -1) -> tuple[float, dict]:
        """距多周期高点距离评分。

        IC 加权: dh180(0.7) + dh360(0.6) + dh1000(0.5) + dh60(0.2) = max 2.0
        """
        n = len(close)
        current = close[idx]
        score = 0.0
        details = {}

        configs = [
            (180, 0.7, [(10, 0), (3, 0.1), (0, 0.3), (-10, 0.5), (-20, 0.6), (-999, 0.7)]),
            (360, 0.6, [(7, 0), (0, 0.2), (-20, 0.3), (-30, 0.5), (-999, 0.6)]),
            (1000, 0.5, [(5, 0), (0, 0.1), (-30, 0.2), (-999, 0.5)]),
            (60, 0.2, [(5, 0), (0, 0.1), (-999, 0.2)]),
        ]

        for period, max_s, thresholds in configs:
            i = n + idx if idx < 0 else idx
            if i >= period:
                high = np.max(close[i - period: i])
            else:
                high = np.max(close[:max(i, 1)])
            pct = (current / high - 1) * 100
            for threshold, s in thresholds:
                if pct >= threshold:
                    score += s
                    break
            details[f"dh{period}"] = round(pct, 1)

        return round(score, 1), details

    # ── 一阶导趋势分 (0-3) ───────────────────────────────────────────

    @staticmethod
    def score_trend(f1_val: float) -> tuple[float, str]:
        """一阶导趋势评分。"""
        if f1_val > 0.3:
            return 3.0, "强涨"
        elif f1_val > 0.1:
            return 2.0, "中涨"
        elif f1_val > 0:
            return 1.0, "弱涨"
        elif f1_val > -0.1:
            return 0.5, "弱跌"
        elif f1_val > -0.3:
            return 0.0, "中跌"
        else:
            return 0.0, "强跌"

    # ── 二阶导曲率分 (0-3) ───────────────────────────────────────────

    @staticmethod
    def score_curvature(
        f2_val: float,
        f2_arr: np.ndarray | None = None,
        idx: int = -1,
    ) -> tuple[float, str]:
        """二阶导曲率评分 = 当前值(0-2) + 拐点检测(0-1)。"""
        # 当前值
        if f2_val > 0.05:
            cur_score = 2.0
            cur_label = "加速上行"
        elif f2_val > 0.01:
            cur_score = 1.5
            cur_label = "减速下跌(蓄力)"
        elif f2_val > -0.01:
            cur_score = 1.0
            cur_label = "平稳"
        elif f2_val > -0.05:
            cur_score = 0.5
            cur_label = "减速上涨"
        else:
            cur_score = 0.0
            cur_label = "加速下跌"

        # 拐点检测
        inflect_score = 0.0
        inflect_label = "无拐点"
        if f2_arr is not None and len(f2_arr) >= 6:
            n = len(f2_arr)
            i = n + idx if idx < 0 else idx
            if i >= 5:
                f2_past = np.mean(f2_arr[i - 5: i])
                f2_now = f2_arr[i]
                if f2_past < 0 and f2_now > 0:
                    inflect_score = 1.0
                    inflect_label = "拐点! f''从负转正"
                elif f2_past < 0 and f2_now > f2_past:
                    inflect_score = 0.5
                    inflect_label = "f''在改善"

        total = min(cur_score + inflect_score, 3.0)
        label = f"{cur_label} | {inflect_label}"
        return round(total, 1), label

    # ── 分钟信号分 (0-2) ─────────────────────────────────────────────

    def score_minute(self, code: str, cutoff_date: str | None = None) -> tuple[float, str, dict]:
        """改进的分钟信号评分。

        基于60min级别多维度信号检测：
        - accel_confirm: f''拐点 + f'>0.05 (71.3%胜率)
        - accel_bull: accel_confirm + 均线多头 (75%胜率)
        - bd_60: 60min底背离
        - top_div_warn: 60min顶背离(减分标记)

        Args:
            code: 纯数字代码
            cutoff_date: 截断日期（YYYY-MM-DD），分钟数据只用到此日期（含），防止前瞻偏差

        Returns:
            (score, label, details)
            score: 0-2分
            details: 信号详情字典
        """
        default_details = {
            "minute_label": "无分钟数据",
            "accel_confirm": False,
            "accel_bull": False,
            "bd_60": False,
            "top_div": False,
            "f1_60min": 0.0,
            "f2_60min": 0.0,
        }

        if self.data_root is None:
            return 0.0, "无数据路径", default_details

        minute_path = self.data_root / "kline" / code / "60min.parquet"
        if not minute_path.exists():
            return 0.0, "无分钟数据", default_details

        try:
            df = pd.read_parquet(minute_path)
            dt_col = "datetime" if "datetime" in df.columns else "date"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)

            # 截断到 cutoff_date（防止回测前瞻偏差）
            if cutoff_date is not None:
                df = df[df[dt_col] <= pd.to_datetime(cutoff_date)].reset_index(drop=True)

            close_60 = df["close"].values.astype(float)
            n = len(close_60)
            if n < self.sg_window + 3:
                return 0.0, "分钟数据不足", default_details

            # ── 导数计算（Kalman 或 Savgol）──
            if self.use_kalman:
                from scripts.analyzers.kalman_derivative_filter import kalman_derivatives
                f1_60, f2_60 = kalman_derivatives(close_60)
            else:
                ln_p = np.log(close_60)
                f1_60 = causal_savgol(ln_p, self.sg_window, self.sg_poly, deriv=1) * 100   # 2026-08-19 P0-1C
                f2_60 = causal_savgol(ln_p, self.sg_window, self.sg_poly, deriv=2) * 10000
                _f = self.sg_window - 1
                if len(f1_60) > _f:
                    f1_60[:_f] = f1_60[_f]
                    f2_60[:_f] = f2_60[_f]

            f1_val = f1_60[-1]
            f2_val = f2_60[-1]

            details = {
                "minute_label": "无信号",
                "accel_confirm": False,
                "accel_bull": False,
                "bd_60": False,
                "top_div": False,
                "f1_60min": round(float(f1_val), 4),
                "f2_60min": round(float(f2_val), 4),
            }

            # ── 1. accel_confirm: f''拐点 + f'>0.05 ──
            # f''拐点: mean(f2[i-3:i]) < 0 且 f2[i] > 0
            lb = 3
            accel_confirm = False
            if n >= lb + 1:
                f2_recent_mean = np.mean(f2_60[-(lb + 1): -1])  # f2[i-3:i]
                f2_current = f2_60[-1]                            # f2[i]
                if f2_recent_mean < 0 and f2_current > 0 and f1_val > 0.05:
                    accel_confirm = True
                    details["accel_confirm"] = True

            # ── 2. accel_bull: accel_confirm + 均线多头 ──
            # MA5 > MA10 > MA20
            accel_bull = False
            if accel_confirm and n >= 20:
                ma5 = np.mean(close_60[-5:])
                ma10 = np.mean(close_60[-10:])
                ma20 = np.mean(close_60[-20:])
                if ma5 > ma10 > ma20:
                    accel_bull = True
                    details["accel_bull"] = True

            # ── 3. MACD 底背离 / 顶背离 ──
            # 底背离用LOW找价格低点, 顶背离用HIGH找价格高点
            # MACD对比用MACD柱(全市场回测: MACD柱54.3% > DIF49.5% > DEA47.6%)
            e12 = df["close"].ewm(span=12, adjust=False).mean()
            e26 = df["close"].ewm(span=26, adjust=False).mean()
            df["DIF"] = e12 - e26
            df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
            df["MACD"] = (df["DIF"] - df["DEA"]) * 2

            rec = df.tail(80).copy()
            rec_r = rec.reset_index(drop=True)
            lows = argrelextrema(rec_r["low"].values, np.less, order=3)[0]
            highs = argrelextrema(rec_r["high"].values, np.greater, order=3)[0]

            bd_60 = False
            top_div = False

            # 底背离：LOW新低 + MACD柱未新低（数据验证: MACD柱胜率54.3% > DIF 49.5%）
            if len(lows) >= 2:
                i1, i2 = lows[-2], lows[-1]
                if (rec_r.iloc[i2]["low"] < rec_r.iloc[i1]["low"]
                        and rec_r.iloc[i2]["MACD"] > rec_r.iloc[i1]["MACD"]):
                    bd_60 = True
                    details["bd_60"] = True

            # 顶背离：HIGH新高 + MACD柱未新高
            if len(highs) >= 2:
                i1, i2 = highs[-2], highs[-1]
                if (rec_r.iloc[i2]["high"] > rec_r.iloc[i1]["high"]
                        and rec_r.iloc[i2]["MACD"] < rec_r.iloc[i1]["MACD"]):
                    top_div = True
                    details["top_div"] = True

            # ── 综合评分 ──
            score = 0.0
            label = "无信号"

            if accel_bull:
                score = 2.0
                label = "60min趋势加速+多头"
            elif accel_confirm:
                score = 1.5
                label = "60min趋势加速"
            elif bd_60:
                score = 1.0
                label = "60min底背离"

            details["minute_label"] = label
            return score, label, details

        except Exception:
            return 0.0, "解析错误", default_details

    # ── MACD柱状图 ─────────────────────────────────────────────────

    @staticmethod
    def _compute_macd_hist(df_daily: pd.DataFrame) -> float:
        """计算日线MACD柱状图当前值（MACD = 2*(DIF-DEA)）。"""
        try:
            close = df_daily["close"].values.astype(float)
            e12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            e26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            dif = e12 - e26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist = (dif - dea) * 2
            return float(macd_hist.iloc[-1])
        except Exception:
            return 0.0

    # ── 涨势延续性指数 CI ───────────────────────────────────────────

    @staticmethod
    def compute_ci(
        f1_val: float,
        f2_val: float,
        f2_delta: float,
        vf1_val: float,
        vol_ratio: float,
        dh180: float,
    ) -> float:
        """涨势延续性指数 (Continuation Index)。

        维度：
          1. f''趋势 (0-1): 加速度是否在收敛
          2. 量能配合 (0-1): 价量关系
          3. 位置空间 (0-1): 距高点距离
        权重: 0.35 / 0.35 / 0.30
        """
        # 维度1: f''趋势
        if f2_delta > 0.01:
            ci_f2 = 1.0   # 加速度扩大
        elif f2_delta > -0.005:
            ci_f2 = 0.8   # 平稳
        elif f2_val > 0.01:
            ci_f2 = 0.5   # 还在加速但减弱
        else:
            ci_f2 = 0.2   # 加速度转负

        # 维度2: 量能配合
        if f1_val > 0:
            if vf1_val > 0.01:
                ci_vol = 1.0    # 量价齐升
            elif vf1_val > -0.01:
                ci_vol = 0.9    # 缩量上涨（筹码锁定）
            elif vol_ratio > 0.7:
                ci_vol = 0.6    # 价涨量平
            else:
                ci_vol = 0.3    # 量能衰竭
        else:
            ci_vol = 0.5  # 下跌中，量能维度中性

        # 维度3: 位置空间
        if dh180 < -40:
            ci_pos = 1.0
        elif dh180 < -20:
            ci_pos = 0.7
        elif dh180 < -10:
            ci_pos = 0.5
        elif dh180 < 0:
            ci_pos = 0.3
        else:
            ci_pos = 0.2

        return round(0.40 * ci_f2 + 0.20 * ci_vol + 0.40 * ci_pos, 3)

    @staticmethod
    def compute_penalty(
        ret20: float,
        volatility: float = 40.0,
        vol_spike: bool = False,
        floor: float = 0.15,
    ) -> float:
        """V2 惩罚函数: 波动率+涨幅双因子sigmoid + floor。

        基于全市场回测(27972个暴涨样本, 严格无未来数据泄漏):
        - 波动率IC=-0.126, 涨幅IC=-0.091
        - 标准ADX IC=+0.004(无效, 已移除)
        - 两因子IC=+0.131 (三因子含ADX IC=+0.109)
        - penalty极低档(0-0.3): 47%到顶 vs 22%继续
        - penalty中档(0.5-0.7): 28%到顶 vs 26%继续
        - penalty高档(0.85+): 21%到顶 vs 26%继续

        双因子独立sigmoid, 等权合成:
          pen = floor + (1-floor) × (0.5×vol_sigmoid + 0.5×ret_sigmoid)
        天量(vol_spike): 额外×0.85
        下限floor=0.15, 不归零

        Args:
            ret20: 20日涨幅(%)
            volatility: 20日年化波动率(%), 默认40(中等)
            vol_spike: 是否天量(2倍均量), 默认False
            floor: 惩罚下限, 默认0.15
        """
        vol_sigmoid = 1.0 / (1.0 + np.exp(0.05 * (volatility - 50)))
        ret_sigmoid = 1.0 / (1.0 + np.exp(0.06 * (ret20 - 45)))

        pen = floor + (1.0 - floor) * (
            0.5 * vol_sigmoid + 0.5 * ret_sigmoid
        )
        if vol_spike:
            pen *= 0.85

        return round(max(pen, floor), 4)

    # ── 信号分类 ──────────────────────────────────────────────────────

    @staticmethod
    def classify_signal(f1_val: float, f2_val: float, above_ma20: bool = True) -> str:
        """根据 f' 和 f'' 分类信号。
        
        主升浪必须 close > MA20，否则降级为底部拐点或观望。
        这是2026-05-26中油工程假主升浪教训的修复。
        """
        if f1_val > 0 and f2_val > 0.01:
            if above_ma20:
                return "🚀 主升浪"
            else:
                return "📊 反弹(假主升浪)"  # f'/f''正但在MA20下方，不是真主升浪
        elif f2_val > 0.01 and f1_val < 0:
            return "🔄 底部拐点"
        elif f1_val > 0 and f2_val < -0.01:
            return "⚠️ 减速见顶"
        elif f1_val < -0.1 and f2_val < -0.01:
            return "🔴 加速下跌"
        else:
            return "➡️ 观望"

    # ── 单只评分 ─────────────────────────────────────────────────────

    def score_stock(self, code: str, df_daily: pd.DataFrame) -> V3Score:
        """对单只股票做 v3 评分。

        Args:
            code: 纯数字代码如 '603881'
            df_daily: 日线 DataFrame，需有 close 列，按日期升序，
                      可选 vol/volume 列（惩罚函数需要）
        """
        result = V3Score()

        close = df_daily["close"].values.astype(float)
        n = len(close)
        if n < 60:
            result.signal = "数据不足"
            return result

        # 导数
        f1, f2 = self.compute_derivatives(close, self.sg_window, self.sg_poly,
                                          use_kalman=self.use_kalman)
        f1_val = f1[-1]
        f2_val = f2[-1]

        # 快通道确认（双窗口系统）
        dual = self.compute_derivatives_dual(close, use_kalman=self.use_kalman)
        f1_fast_val = dual['f1_fast'][-1]
        f2_fast_val = dual['f2_fast'][-1]
        dual_accel_confirm = (f1_val > 0) and (f2_val > 0) and (f1_fast_val > 0) and (f2_fast_val > 0)
        dual_reversal_confirm = (f2_val > 0.01) and (f2_fast_val > 0.01)  # 快慢都确认加速度转正

        # 位置
        ps, pos_details = self.score_position(close)
        # 趋势
        ds, trend_label = self.score_trend(f1_val)
        # 曲率
        cs, curv_label = self.score_curvature(f2_val, f2)
        # 分钟（自动截断到日线最后日期，防止前瞻偏差）
        dt_col = "trade_date" if "trade_date" in df_daily.columns else "date"
        _cutoff = str(df_daily.iloc[-1][dt_col])[:10] if dt_col in df_daily.columns else None
        ms, min_label, min_details = self.score_minute(code, cutoff_date=_cutoff)

        # 顶背离减分（v3.5: 从-0.5加大到-1.5，回测显示顶背离是有效卖出信号）
        if min_details.get("top_div", False):
            ms = max(ms - 1.5, 0.0)

        # CUSUM 融合：验证 f' 是否发生结构性变化，过滤假拐点
        try:
            from scripts.analyzers.cusum_detector import CUSUMDetector
            _cusum_det = CUSUMDetector(h_base=4.0, k=0.5)
            _cusum_fusion = _cusum_det.fuse_with_v3(
                type('V3Proxy', (), {'f1': f1_val, 'f2': f2_val, 'signal': ''})(),
                f1, f2_val,
            )
            cs = max(0.0, min(3.0, cs + _cusum_fusion["curvature_adjustment"]))
            ds = max(0.0, min(3.0, ds + _cusum_fusion["trend_adjustment"]))
            cusum_info = {
                "up": _cusum_fusion["cusum_result"].up_crossing,
                "down": _cusum_fusion["cusum_result"].down_crossing,
                "confidence": _cusum_fusion["cusum_result"].confidence,
                "reason": _cusum_fusion["reason"],
            }
        except Exception:
            cusum_info = {"up": False, "down": False, "confidence": "N/A", "reason": "CUSUM计算失败"}

        raw_total = ps + ds + cs + ms

        # ── PB 数据加载（v3.5: 不再加分，仅保留用于输出参考）──
        pb_rank = 0.5
        pb_bonus = 0.0  # v3.5: 固定为0，PB加分已移除（因果验证确认是动量代理）
        pb_val = np.nan

        dt_col = "trade_date" if "trade_date" in df_daily.columns else "date"
        if self.data_root is not None:
            try:
                last_date = pd.to_datetime(df_daily.iloc[-1][dt_col])
                last_date = pd.to_datetime(df_daily.iloc[-1][dt_col])
                date_str = last_date.strftime("%Y%m%d")
                basic_path = self.data_root / "daily_basic" / f"{date_str}.parquet"
                if basic_path.exists():
                    basic_df = pd.read_parquet(basic_path)
                    ts_code = code + (".SH" if code.startswith("6") else ".SZ")
                    row = basic_df[basic_df["ts_code"] == ts_code]
                    if len(row) > 0:
                        pb_val = row.iloc[0].get("pb", np.nan)
                        if not np.isnan(pb_val) and len(basic_df) > 0:
                            valid_pb = basic_df["pb"].dropna()
                            if len(valid_pb) > 0:
                                pb_rank = (valid_pb < pb_val).sum() / len(valid_pb)
            except Exception:
                pass

        # ── 惩罚函数 ──────────────────────────────────────────────
        # 计算CI所需额外特征
        # 1. f'' delta (近5日加速度变化)
        if n >= 6:
            f2_delta = f2[-1] - np.mean(f2[-6:-1])
        else:
            f2_delta = 0.0

        # 2. 量能一阶导
        has_vol = "vol" in df_daily.columns or "volume" in df_daily.columns
        vf1_val = 0.0
        vol_ratio = 1.0
        if has_vol and n >= self.sg_window:
            vc = "vol" if "vol" in df_daily.columns else "volume"
            vol = df_daily[vc].values.astype(float)
            ln_vol = np.log(vol + 1)
            vf1 = causal_savgol(ln_vol, self.sg_window, self.sg_poly, deriv=1) * 100   # 2026-08-19 P0-1C替换
            _f = self.sg_window - 1
            if len(vf1) > _f:
                vf1[:_f] = vf1[_f]
            vf1_val = vf1[-1]
            vol5 = np.mean(vol[-5:])
            vol20 = np.mean(vol[-20:]) if n >= 20 else vol5
            vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0

        # 3. 20日历史涨幅
        ret20_0 = (close[-1] / close[-21] - 1) * 100 if n >= 21 else 0.0

        # 4. dh180
        dh180 = pos_details.get("dh180", 0.0)

        # 5. MA20 (价格确认)
        ma20 = float(np.mean(close[-20:])) if n >= 20 else close[-1]
        above_ma20 = close[-1] > ma20

        # 6. 极端波动检测（savgol窗口内涨跌停检查）
        extreme_volatility = False
        if n >= 2:
            recent_21_chg = np.abs(np.diff(close[-22:]) / close[-22:-1] * 100) if n >= 22 else np.array([])
            if len(recent_21_chg) > 0:
                extreme_volatility = bool(np.max(recent_21_chg) > 7.0)

        # 计算CI和惩罚
        ci = self.compute_ci(f1_val, f2_val, f2_delta, vf1_val, vol_ratio, dh180)

        # v2 惩罚函数额外参数
        # 6. 波动率(20日年化)
        if n >= 22:
            _c22 = close[-22:].copy()
            daily_rets = np.zeros(21)
            for _i in range(21):
                daily_rets[_i] = (_c22[_i+1] / _c22[_i] - 1)  # 小数, 不是百分比
        else:
            daily_rets = np.array([0.0])
        volatility = float(np.std(daily_rets[-20:]) * np.sqrt(252) * 100) if len(daily_rets) >= 20 else 40.0

        # 7. ADX近似(14日均涨跌幅均值)
        adx = float(np.mean(np.abs(daily_rets[-14:]))) if len(daily_rets) >= 14 else volatility / 20

        # 8. 天量(5日均量 > 2倍20日均量)
        vol_spike = False
        if has_vol and n >= 20:
            vc2 = 'vol' if 'vol' in df_daily.columns else 'volume'
            vol_arr = df_daily[vc2].values.astype(float)
            vol5 = np.mean(vol_arr[-5:])
            vol20m = np.mean(vol_arr[-20:])
            vol_spike = vol5 > vol20m * 2

        penalty = self.compute_penalty(ret20_0, volatility, vol_spike)

        penalized_total = round(raw_total * penalty, 2)

        result.total = penalized_total
        result.raw_total = round(raw_total, 1)
        result.penalty = penalty
        result.ci = ci
        result.position_score = ps
        result.trend_score = ds
        result.curvature_score = cs
        result.minute_score = ms
        result.details = {}  # 初始化，后面会填充
        result.f1 = round(f1_val, 4)
        result.f2 = round(f2_val, 4)
        result.signal = self.classify_signal(f1_val, f2_val, above_ma20)
        result.details = {
            **pos_details,
            "trend_label": trend_label,
            "curvature_label": curv_label,
            "minute_label": min_label,
            "minute_details": min_details,
            "price": close[-1],
            "n_days": n,
            "ret20_0": round(ret20_0, 2),
            "ma20": round(ma20, 2),
            "above_ma20": above_ma20,
            "pb_rank": round(pb_rank, 4),
            "pb_bonus": pb_bonus,
            "pb": round(pb_val, 3) if not np.isnan(pb_val) else None,
            "vol_ratio": round(vol_ratio, 3),
            "macd_hist": round(self._compute_macd_hist(df_daily), 4) if n >= 35 else 0.0,
            "f1_fast": round(f1_fast_val, 4),
            "f2_fast": round(f2_fast_val, 4),
            "dual_accel_confirm": dual_accel_confirm,
            "dual_reversal_confirm": dual_reversal_confirm,
            "volatility": round(volatility, 1),
            "adx": round(adx, 2),
            "vol_spike": vol_spike,
            "extreme_volatility": extreme_volatility,
            "data_date": str(df_daily.iloc[-1][dt_col])[:10] if dt_col in df_daily.columns else None,
            "cusum": cusum_info,
        }

        # 数据时效警告
        if dt_col in df_daily.columns:
            from datetime import datetime, timedelta
            try:
                last_dt = pd.to_datetime(df_daily.iloc[-1][dt_col])
                now = datetime.now()
                if (now - last_dt).days > 3 and now.weekday() < 6:
                    result.details["data_stale"] = True
            except Exception:
                pass

        return result

    # ── HMM 市场状态检测 ────────────────────────────────────────────

    def _load_hmm(self):
        """延迟加载HMM模型。"""
        if self._hmm_model is not None:
            return True
        # 搜索模型路径: data/models/ 或 models/
        candidates = []
        if self.data_root:
            candidates.append(self.data_root.parent / 'models' / 'hmm_3state_sh.pkl')
            candidates.append(self.data_root / 'models' / 'hmm_3state_sh.pkl')
        candidates.append(Path(__file__).parent.parent.parent / 'models' / 'hmm_3state_sh.pkl')

        import pickle
        for p in candidates:
            if p.exists():
                try:
                    with open(p, 'rb') as f:
                        saved = pickle.load(f)
                    self._hmm_model = saved['model']
                    self._hmm_state_names = saved['state_names']
                    return True
                except Exception:
                    continue
        return False

    def get_market_regime(self, idx_close: np.ndarray | None = None) -> dict:
        """检测当前市场状态(趋势/震荡/恐慌)。

        优先用Tushare拉上证指数最新数据（DataStore里可能不到1年）。
        回测: 趋势态趋势加速84.5%胜率, 震荡态70.1%, 恐慌态61.3%。

        Args:
            idx_close: 上证指数收盘价序列。不提供则自动拉取。

        Returns:
            dict: {state: int, name: str, regime_key: str, probs: list}
        """
        default = {'state': -1, 'name': '⚪未知', 'regime_key': 'range', 'probs': []}

        if not self._load_hmm():
            return default

        # 获取上证指数数据 — 优先tushare（完整）, 回退到DataStore
        if idx_close is None:
            idx_close = self._fetch_sh_index()

        if idx_close is None or len(idx_close) < 30:
            return default

        try:
            rets = np.diff(np.log(idx_close.astype(float)))
            vol20 = np.array([np.std(rets[max(0,i-19):i+1]) * np.sqrt(252)
                              for i in range(len(rets))])
            X = np.column_stack([rets, vol20])

            states = self._hmm_model.predict(X)
            current_state = int(states[-1])
            state_name = self._hmm_state_names.get(current_state, '⚪未知')

            # 映射到regime_key
            if '趋势' in state_name or '🟢' in state_name:
                regime_key = 'trend'
            elif '恐慌' in state_name or '🔴' in state_name:
                regime_key = 'panic'
            else:
                regime_key = 'range'

            # 概率
            try:
                probs = self._hmm_model.predict_proba(X)
                current_probs = [round(float(p), 3) for p in probs[-1]]
            except Exception:
                current_probs = []

            return {
                'state': current_state,
                'name': state_name,
                'regime_key': regime_key,
                'probs': current_probs,
            }
        except Exception:
            return default

    def _fetch_sh_index(self) -> np.ndarray | None:
        """拉取上证指数收盘价序列。走DataStore统一管理。"""
        # 方法1: DataStore.get_index_kline()（自动增量）
        if self.data_root is not None:
            try:
                from scripts.core.data_store import DataStore
                ds = DataStore(self.data_root)
                df = ds.get_index_kline('000001SH', auto_update=True)
                if df is not None and len(df) > 100:
                    return df['close'].values.astype(float)
            except Exception:
                pass

        # 方法2: 直接读缓存文件
        if self.data_root is not None:
            cache_path = self.data_root / 'kline' / '000001SH' / 'D.parquet'
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    dt_col = 'trade_date' if 'trade_date' in df.columns else 'date'
                    df[dt_col] = pd.to_datetime(df[dt_col])
                    df = df.sort_values(dt_col).reset_index(drop=True)
                    if len(df) > 100:
                        return df['close'].values.astype(float)
                except Exception:
                    pass

        # 方法3: Tushare在线（最后手段）
        try:
            import os
            token = os.environ.get('TUSHARE_TOKEN')
            if not token:
                from scripts.core.tushare_client import _DEFAULT_TOKEN
                token = _DEFAULT_TOKEN
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()
            df = pro.index_daily(ts_code='000001.SH', start_date='20200101')
            if df is not None and len(df) > 100:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                return df['close'].values.astype(float)
        except Exception:
            pass

        return None

    # ── 双分支筛选 ──────────────────────────────────────────────────

    def score_dual_branch(
        self,
        codes: list[str],
        df_dict: dict[str, pd.DataFrame] | None = None,
        min_score: float = 4.0,
        market_regime: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """三分支筛选：趋势加速 + 底部反转(三级) + 顶部卖出信号。

        v3.5 重构 + HMM市场状态自适应：
        - 🟢趋势态: 趋势加速全权重(84.5%胜率), 底部正常
        - 🟡震荡态: 趋势降权0.85, 底部升权1.1
        - 🔴恐慌态: 趋势大幅降权0.6, 底部升权1.3(只看反转)

        Args:
            codes: 股票代码列表
            df_dict: 可选预加载数据
            min_score: 基础最低分(会被HMM状态调整)
            market_regime: 市场状态dict。不提供则自动检测。

        Returns:
            (accel_df, reversal_df, top_sell_df)
        """
        batch = self.score_batch(codes, df_dict)
        if len(batch) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # ── HMM 市场状态自适应 ──────────────────────────────────
        if market_regime is None:
            market_regime = self.get_market_regime()

        regime_key = market_regime.get('regime_key', 'range')
        regime_name = market_regime.get('name', '⚪未知')
        weights = self.HMM_WEIGHTS.get(regime_key, self.HMM_WEIGHTS['range'])

        # HMM 调整后的最低分
        accel_min_score = weights['accel_min_score']
        reversal_min_ratio = weights['reversal_min_score_ratio']

        # ════════════════════════════════════════════════════════════
        # 分支1: 趋势加速 — HMM自适应
        # ════════════════════════════════════════════════════════════
        accel_mask = (
            (batch["f1"] > 0) &
            (batch["f2"] > 0) &
            (batch["v3_raw"] >= accel_min_score) &  # HMM调整: 趋势态4.0, 震荡5.0, 恐慌6.0
            (batch["above_ma20"] == True)
        )
        if "ret20_0" in batch.columns:
            accel_mask = accel_mask & (batch["ret20_0"] > -10)
        accel_df = batch[accel_mask].copy()

        # HMM权重调整: 给总分乘以市场状态权重
        accel_df["v3_total"] = (accel_df["v3_total"] * weights['accel_weight']).round(2)

        # ── 趋势阶段分类(P0-2, 解决追高悖论, 2026-06-16沃顿教训) ──────
        # f1选"涨得快的"→必然倾向"已涨高的"→与"不追高"矛盾
        # 拆分: 启动期(位置低优先) / 中继期 / 延伸期(已大涨,追高风险)
        def _trend_phase(r20):
            if r20 < 10: return "启动期"      # 20日涨幅<10%, 刚启动位置低
            elif r20 >= 20: return "延伸期"   # 20日涨幅>=20%, 已大涨追高风险
            else: return "中继期"
        accel_df["trend_phase"] = accel_df["ret20_0"].apply(_trend_phase)

        # ── 数据自洽校验(P0-1) ──────────────────────────────────────
        # f1强正(>0.5, 强趋势)但ret20极弱(<5%, 几乎没涨)→ 离线parquet可能失真
        # (如沃顿 f1=+1.336 但 ret20=+4%, 实算+30%, 数据管道问题)
        accel_df["data_suspect"] = (accel_df["f1"] > 0.5) & (accel_df["ret20_0"] < 5)

        # 排序: total分为主, 数据存疑降级。
        # 注(2026-06-17回测修正): trend_phase 不再加排序权重。回测16万样本显示
        # 延伸期(ret20≥20%)未来10日收益+1.85% 显著高于启动期(<10%)+0.66%(p<0.001),
        # 即动量延续强于位置, "启动期优先"假设被推翻。trend_phase 保留为标签
        # (延伸期=追高择时提示), 实际见顶把关交给分钟背离(P1)。
        def accel_sort_key(row):
            score = row['v3_total']  # 已含penalty, 作为主排序依据
            if row.get('data_suspect'): score -= 3.0  # 数据质量问题, 与收益无关
            if row.get('dual_accel_confirm'): score += 2
            md = row.get("minute_details", {})
            if isinstance(md, dict):
                if md.get("accel_bull"): score += 1.5
                elif md.get("accel_confirm"): score += 1.0
            return score
        accel_df["_accel_rank"] = accel_df.apply(accel_sort_key, axis=1)
        accel_df = accel_df.sort_values(
            ["_accel_rank", "v3_total"], ascending=[False, False]
        ).drop(columns=["_accel_rank"]).reset_index(drop=True)

        # 标记HMM状态
        accel_df["market_regime"] = regime_name

        # ════════════════════════════════════════════════════════════
        # 分支2: 底部反转 — 三级确认机制
        # ════════════════════════════════════════════════════════════
        # L1 预警(内部): f''>0 + f'<0 + close<MA20 + dh180<-15
        # L2 关注: f'>=0 + f''>0 + close<MA20 + dh180<-20
        # L3 确认买入: f'>=0 + f''>0.01 + dh180<-30 + macd绿柱 + 放量

        has_macd = "macd_hist" in batch.columns
        has_vol = "vol_ratio" in batch.columns

        # L2 关注（f'从负转正）
        L2_mask = (
            (batch["f1"] >= 0) &
            (batch["f1"] < 0.05) &
            (batch["f2"] > 0.01) &
            (batch["above_ma20"] == False) &
            (batch["dh180"] < -20) &
            (batch["v3_raw"] >= min_score * reversal_min_ratio)  # HMM调整: 趋势0.7, 震荡0.6, 恐慌0.5
        )

        # L3 确认买入（多条件共振，~73%胜率）
        # 回测: f1<0.05(743样本72.4%) vs f1<0.03(435样本72.0%), 放宽更好
        L3_mask = (
            (batch["f1"] >= 0) &
            (batch["f1"] < 0.05) &  # 放宽到0.05（回测确认不影响胜率）
            (batch["f2"] > 0.01) &
            (batch["above_ma20"] == False) &
            (batch["dh180"] <= -25)  # 用<=包含边界
        )
        if has_macd:
            L3_mask = L3_mask & (batch["macd_hist"] < 0)  # MACD还是绿柱
        if has_vol:
            L3_mask = L3_mask & (batch["vol_ratio"] > 0.8)  # 放量确认

        # 合并: L3优先，其余为L2
        L3_codes = set(batch[L3_mask]["code"]) if len(batch[L3_mask]) > 0 else set()
        L2_all = batch[L2_mask].copy()
        L2_all["reversal_level"] = L2_all["code"].apply(
            lambda c: "L3确认买入" if c in L3_codes else "L2关注"
        )
        reversal_df = L2_all.sort_values(
            ["reversal_level", "v3_curvature"], ascending=[True, False]
        ).reset_index(drop=True)
        reversal_df["market_regime"] = regime_name

        # ════════════════════════════════════════════════════════════
        # 分支3: 顶部卖出信号 (69%下跌概率)
        # ════════════════════════════════════════════════════════════
        # 核心: f'>0(还在涨) + f''<0(减速) + macd_hist<0(死叉) + 高位
        top_mask = (
            (batch["f1"] > 0) &
            (batch["f2"] < -0.01) &
            (batch["above_ma20"] == True)
        )
        if has_macd:
            top_mask = top_mask & (batch["macd_hist"] < 0)
        if "ret20_0" in batch.columns:
            top_mask = top_mask & (batch["ret20_0"] > 5)  # 已经涨了不少
        top_df = batch[top_mask].copy()
        top_df = top_df.sort_values("v3_total", ascending=False).reset_index(drop=True)
        top_df["market_regime"] = regime_name

        return accel_df, reversal_df, top_df

    # ── 个股分析模式 ─────────────────────────────────────────────────

    def analyze_stock(self, code: str, df_daily: pd.DataFrame | None = None) -> dict:
        """个股分析模式 — 展示raw分 + 各维度拆解，penalty作为独立提醒。

        与 score_stock 的区别：
        - score_stock: 筛选场景，用 penalty 后的 total 排序过滤
        - analyze_stock: 分析场景，以 raw 为主体，penalty 作为"追高提醒"单独标出

        Returns:
            dict 包含完整分析结果
        """
        if df_daily is None:
            if self.data_root is None:
                return {"error": "无数据路径"}
            path = self.data_root / "kline" / code / "D.parquet"
            if not path.exists():
                return {"error": f"无K线数据: {code}"}
            df_daily = pd.read_parquet(path)
            dt_col = "trade_date" if "trade_date" in df_daily.columns else "date"
            df_daily[dt_col] = pd.to_datetime(df_daily[dt_col])
            df_daily = df_daily.sort_values(dt_col).reset_index(drop=True)

        s = self.score_stock(code, df_daily)
        if s.signal == "数据不足":
            return {"error": "数据不足"}

        # 维度拆解
        dims = {
            "位置距离": {"score": s.position_score, "max": 2.0,
                       "detail": f"dh60={s.details.get('dh60',0):+.1f}% dh180={s.details.get('dh180',0):+.1f}% dh360={s.details.get('dh360',0):+.1f}%"},
            "趋势强度": {"score": s.trend_score, "max": 3.0,
                       "detail": f"f1={s.f1:+.3f} ({s.details.get('trend_label','')})"},
            "加速度":   {"score": s.curvature_score, "max": 3.0,
                       "detail": f"f2={s.f2:+.3f} ({s.details.get('curvature_label','')})"},
            "分钟信号": {"score": s.minute_score, "max": 2.0,
                       "detail": s.details.get('minute_label', '无')},
        }

        # 追高提醒（penalty不是1.0时才有意义）
        chase_warning = None
        if s.penalty < 0.9:
            chase_warning = {
                "level": "⚠️ 追高风险" if s.penalty < 0.7 else "🟡 温和提醒",
                "penalty": round(s.penalty, 3),
                "reason": f"近20日涨幅 {s.details.get('ret20_0', 0):+.1f}%，CI={s.ci:.2f}",
                "impact": f"raw分{s.raw_total} × penalty{s.penalty:.2f} = {s.total}",
            }

        # 信号分类 + 操作建议
        advice = self._get_advice(s)

        # HMM市场状态
        regime = self.get_market_regime()

        return {
            "code": code,
            "price": s.details.get("price", 0),
            "signal": s.signal,
            "raw_total": s.raw_total,
            "penalty_total": s.total,
            "penalty": s.penalty,
            "ci": s.ci,
            "dims": dims,
            "chase_warning": chase_warning,
            "advice": advice,
            "f1": s.f1,
            "f2": s.f2,
            "market_regime": regime.get('name', '⚪未知'),
            "details": s.details,
        }

    @staticmethod
    def _get_advice(s: V3Score) -> str:
        """根据信号类型 + 维度拆解给出操作建议。"""
        f1, f2 = s.f1, s.f2
        md = s.details.get("minute_details", {})
        above_ma20 = s.details.get("above_ma20", False)

        if f1 > 0 and f2 > 0:
            # 主升浪
            parts = ["🚀 趋势加速中"]
            if isinstance(md, dict) and md.get("accel_bull"):
                parts.append("60min趋势加速+多头排列，强势")
            elif isinstance(md, dict) and md.get("accel_confirm"):
                parts.append("60min趋势加速确认")
            if s.penalty < 0.7:
                parts.append("⚠️ 但短期涨幅过大，追高风险极高")
            elif s.penalty < 0.9:
                parts.append("🟡 涨幅偏大，注意仓位")
            else:
                parts.append("✅ 趋势健康")
            return " | ".join(parts)

        elif f1 <= 0 and f2 > 0:
            # 底部拐点
            parts = ["🔄 底部拐点信号"]
            if f1 >= 0:
                parts.append("f'已转正，等MACD金叉+放量确认")
            else:
                parts.append("f'仍负，等f'转正再考虑")
            return " | ".join(parts)

        elif f1 > 0 and f2 < 0:
            # 减速见顶
            parts = ["⚠️ 趋势减速"]
            if isinstance(md, dict) and md.get("top_div"):
                parts.append("60min顶背离！警惕")
            parts.append("密切关注，准备减仓")
            return " | ".join(parts)

        else:
            # 加速下跌
            return "🔴 加速下跌中，回避"

    # ── 批量评分 ─────────────────────────────────────────────────────

    def score_batch(
        self,
        codes: list[str],
        df_dict: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """批量评分。

        Args:
            codes: 代码列表
            df_dict: {code: df_daily}，不提供则从 data_root 读取
        """
        rows = []
        for code in codes:
            try:
                if df_dict and code in df_dict:
                    df = df_dict[code]
                elif self.data_root:
                    path = self.data_root / "kline" / code / "D.parquet"
                    if not path.exists():
                        continue
                    df = pd.read_parquet(path)
                    dt_col = "trade_date" if "trade_date" in df.columns else "date"
                    df[dt_col] = pd.to_datetime(df[dt_col])
                    df = df.sort_values(dt_col).reset_index(drop=True)
                else:
                    continue

                s = self.score_stock(code, df)
                if s.signal == "数据不足":
                    continue

                row_data = {
                    "code": code,
                    "v3_total": s.total,
                    "v3_raw": s.raw_total,
                    "penalty": s.penalty,
                    "ci": s.ci,
                    "v3_position": s.position_score,
                    "v3_trend": s.trend_score,
                    "v3_curvature": s.curvature_score,
                    "v3_minute": s.minute_score,
                    "f1": s.f1,
                    "f2": s.f2,
                    "signal": s.signal,
                    **s.details,
                }
                # 提取 pb_rank 和 pb_bonus 到顶层列
                for k in ("pb_rank", "pb_bonus", "pb"):
                    if k in s.details:
                        row_data[k] = s.details[k]
                rows.append(row_data)
            except Exception:
                continue

        return pd.DataFrame(rows)
