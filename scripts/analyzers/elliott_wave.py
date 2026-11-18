#!/usr/bin/env python3
"""Elliott Wave 波浪分析器 v2.0

核心改进：
1. 支持上涨5浪和下跌5浪（双向）
2. ZigZag模板先看大局，不是一叶障目
3. Fibonacci比例用高斯核评分（连续、平滑）
4. 加入时间比例验证
5. 加入量能验证（推动浪vs调整浪）
6. 交替规则检查（W2和W4形态应不同）

算法流程：
    1. ZigZag 过滤 → 得到有意义的转折点
    2. 从转折点中找候选起止点
    3. 枚举候选组合 + 三大铁律约束剪枝
    4. Fibonacci + 时间 + 量能 综合评分
    5. 输出最优浪型 + 置信度

用法:
    from scripts.analyzers.elliott_wave import ElliottWaveAnalyzer
    analyzer = ElliottWaveAnalyzer()
    result = analyzer.analyze(df, direction='down', start_date='2026-01-12')
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.signal import argrelextrema


# ── Fibonacci 理想比例 ──

# 上涨/下跌5浪中各浪之间的Fibonacci比例关系
# key: 浪间关系, value: 理想比例列表
FIB_IMPULSE_RATIOS = {
    'w2_retrace_w1': [0.382, 0.500, 0.618, 0.786],  # W2回撤W1的比例
    'w3_extend_w1':  [1.000, 1.618, 2.618],          # W3延伸W1的比例
    'w4_retrace_w3': [0.236, 0.382, 0.500],          # W4回撤W3的比例
    'w5_extend_w1':  [0.618, 1.000, 1.618],          # W5延伸W1的比例
}

# 时间比例（浪间持续时间的理想关系）
FIB_TIME_RATIOS = [0.382, 0.500, 0.618, 1.000, 1.618, 2.618]


@dataclass
class Wave:
    """单个浪"""
    label: str           # W1, W2, W3, W4, W5, A, B, C
    start_idx: int
    start_price: float
    end_idx: int
    end_price: float
    start_date: str = ''
    end_date: str = ''

    @property
    def length(self) -> float:
        return abs(self.end_price - self.start_price)

    @property
    def duration(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def pct_change(self) -> float:
        if self.start_price == 0:
            return 0
        return (self.end_price / self.start_price - 1) * 100

    @property
    def direction(self) -> str:
        return 'up' if self.end_price > self.start_price else 'down'

    def to_dict(self) -> dict:
        return {
            'wave': self.label,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'start_price': round(self.start_price, 3),
            'end_price': round(self.end_price, 3),
            'pct_change': round(self.pct_change, 2),
            'duration': self.duration,
            'length': round(self.length, 4),
        }


@dataclass
class WaveResult:
    """波浪分析结果"""
    direction: str                    # 'up' 或 'down'
    waves: List[Wave] = field(default_factory=list)
    fib_score: float = 0.0            # Fibonacci比例评分 (0-1)
    time_score: float = 0.0           # 时间比例评分 (0-1)
    volume_score: float = 0.0         # 量能评分 (0-1)
    iron_rules: List[Tuple[str, bool, str]] = field(default_factory=list)
    total_score: float = 0.0          # 综合评分
    confidence: str = '低'            # 高/中/低
    current_wave: str = ''            # 当前处于哪个浪
    fib_ratios: Dict = field(default_factory=dict)  # 实际比例详情
    exhaustion: Optional[Dict] = None  # W5衰竭信号


class ElliottWaveAnalyzer:
    """Elliott Wave 波浪分析器 v2.0"""

    def __init__(self, zigzag_pct: float = 0.05, min_pivots: int = 5):
        """
        Args:
            zigzag_pct: ZigZag百分比阈值，过滤小于此幅度的波动
            min_pivots: 最少需要多少个转折点才尝试数浪
        """
        self.zigzag_pct = zigzag_pct
        self.min_pivots = min_pivots

    # ── 1. ZigZag 转折点检测 ──

    def zigzag(self, df: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """计算ZigZag转折点。

        算法：
        1. 从第一个点开始，看后续所有点
        2. 当价格反转超过阈值时，确认上一个转折点
        3. 递归处理剩余数据

        Returns:
            [(idx, price, 'high'/'low'), ...]
        """
        high = df['high'].values if 'high' in df.columns else df['最高'].values
        low = df['low'].values if 'low' in df.columns else df['最低'].values
        n = len(df)

        if n < 3:
            return []

        pivots = []
        # 初始方向：看前3根K线
        if high[1] > high[0]:
            direction = 'up'
            pivots.append((0, low[0], 'low'))  # 起点是低
        else:
            direction = 'down'
            pivots.append((0, high[0], 'high'))  # 起点是高

        last_extreme_idx = 0
        last_extreme_price = high[0] if direction == 'up' else low[0]

        for i in range(1, n):
            if direction == 'up':
                # 上升趋势中，更新最高点
                if high[i] >= last_extreme_price:
                    last_extreme_price = high[i]
                    last_extreme_idx = i
                    # 更新最后一个高点
                    if pivots and pivots[-1][2] == 'high':
                        pivots[-1] = (i, high[i], 'high')
                    # else: 继续涨
                else:
                    # 检查是否反转超过阈值
                    retrace = (last_extreme_price - low[i]) / last_extreme_price
                    if retrace >= self.zigzag_pct:
                        # 确认高点，转向
                        if not pivots or pivots[-1][2] != 'high' or pivots[-1][0] != last_extreme_idx:
                            pivots.append((last_extreme_idx, last_extreme_price, 'high'))
                        else:
                            pivots[-1] = (last_extreme_idx, last_extreme_price, 'high')
                        direction = 'down'
                        last_extreme_price = low[i]
                        last_extreme_idx = i
            else:
                # 下降趋势中，更新最低点
                if low[i] <= last_extreme_price:
                    last_extreme_price = low[i]
                    last_extreme_idx = i
                    if pivots and pivots[-1][2] == 'low':
                        pivots[-1] = (i, low[i], 'low')
                else:
                    # 检查是否反转超过阈值
                    retrace = (high[i] - last_extreme_price) / last_extreme_price
                    if retrace >= self.zigzag_pct:
                        if not pivots or pivots[-1][2] != 'low' or pivots[-1][0] != last_extreme_idx:
                            pivots.append((last_extreme_idx, last_extreme_price, 'low'))
                        else:
                            pivots[-1] = (last_extreme_idx, last_extreme_price, 'low')
                        direction = 'up'
                        last_extreme_price = high[i]
                        last_extreme_idx = i

        # 补上最后一个未确认的转折
        if last_extreme_idx > pivots[-1][0]:
            typ = 'high' if direction == 'up' else 'low'
            pivots.append((last_extreme_idx, last_extreme_price, typ))

        # 去重并确保高低交替
        cleaned = [pivots[0]]
        for p in pivots[1:]:
            if p[0] == cleaned[-1][0]:
                continue
            if p[2] == cleaned[-1][2]:
                # 同类型，保留更极端的
                if p[2] == 'high' and p[1] > cleaned[-1][1]:
                    cleaned[-1] = p
                elif p[2] == 'low' and p[1] < cleaned[-1][1]:
                    cleaned[-1] = p
            else:
                cleaned.append(p)

        return cleaned

    # ── 2. Fibonacci 评分（高斯核）──

    @staticmethod
    def fib_gaussian_score(ratio: float, ideals: List[float], sigma: float = 0.08) -> float:
        """用高斯核函数计算比例与理想值的匹配度。

        score = max(exp(-(r - ideal)² / (2σ²)))

        Args:
            ratio: 实际比例
            ideals: 理想比例列表
            sigma: 高斯宽度，越小要求越精确

        Returns:
            0.0 ~ 1.0 之间的匹配度
        """
        scores = [np.exp(-((ratio - ideal) ** 2) / (2 * sigma ** 2)) for ideal in ideals]
        return max(scores)

    def score_fibonacci(self, waves: List[Wave]) -> Tuple[float, Dict]:
        """计算5浪的Fibonacci比例综合评分。

        Returns:
            (score, details_dict)
        """
        if len(waves) < 5:
            return 0.0, {}

        w1, w2, w3, w4, w5 = waves[:5]
        details = {}

        # W2回撤W1
        if w1.length > 0:
            retrace = w2.length / w1.length
            s1 = self.fib_gaussian_score(retrace, FIB_IMPULSE_RATIOS['w2_retrace_w1'])
            details['w2_retrace_w1'] = {'value': retrace, 'score': s1}
        else:
            s1 = 0

        # W3延伸W1
        if w1.length > 0:
            extend = w3.length / w1.length
            s2 = self.fib_gaussian_score(extend, FIB_IMPULSE_RATIOS['w3_extend_w1'])
            details['w3_extend_w1'] = {'value': extend, 'score': s2}
        else:
            s2 = 0

        # W4回撤W3
        if w3.length > 0:
            retrace = w4.length / w3.length
            s3 = self.fib_gaussian_score(retrace, FIB_IMPULSE_RATIOS['w4_retrace_w3'])
            details['w4_retrace_w3'] = {'value': retrace, 'score': s3}
        else:
            s3 = 0

        # W5延伸W1
        if w1.length > 0:
            extend = w5.length / w1.length
            s4 = self.fib_gaussian_score(extend, FIB_IMPULSE_RATIOS['w5_extend_w1'])
            details['w5_extend_w1'] = {'value': extend, 'score': s4}
        else:
            s4 = 0

        # 加权平均：W3比例最重要（35%），W2和W4次之（各25%），W5（15%）
        score = s1 * 0.25 + s2 * 0.35 + s3 * 0.25 + s4 * 0.15
        return score, details

    # ── 3. 时间比例评分 ──

    def score_time(self, waves: List[Wave]) -> float:
        """计算浪间时间比例评分。

        检查相邻浪的持续时间是否接近 Fibonacci 比例。
        """
        if len(waves) < 5:
            return 0.0

        durations = [w.duration for w in waves[:5]]
        scores = []

        # W2时间/W1时间
        if durations[0] > 0:
            r = durations[1] / durations[0]
            scores.append(self.fib_gaussian_score(r, FIB_TIME_RATIOS, sigma=0.15))

        # W3时间/W1时间
        if durations[0] > 0:
            r = durations[2] / durations[0]
            scores.append(self.fib_gaussian_score(r, FIB_TIME_RATIOS, sigma=0.20))

        # W4时间/W2时间（交替：如果W2急W4应该缓，时间比应该>1）
        if durations[1] > 0:
            r = durations[3] / durations[1]
            scores.append(self.fib_gaussian_score(r, FIB_TIME_RATIOS, sigma=0.20))

        # W5时间/W1时间
        if durations[0] > 0:
            r = durations[4] / durations[0]
            scores.append(self.fib_gaussian_score(r, FIB_TIME_RATIOS, sigma=0.20))

        return np.mean(scores) if scores else 0.0

    # ── 4. 量能评分 ──

    def score_volume(self, df: pd.DataFrame, waves: List[Wave]) -> float:
        """推动浪放量 > 调整浪缩量验证。"""
        if len(waves) < 5:
            return 0.0

        vol_col = 'volume' if 'volume' in df.columns else '成交量'
        vol = df[vol_col].values.astype(float)

        impulse_vols = []  # 推动浪均量
        correction_vols = []  # 调整浪均量

        for i, w in enumerate(waves[:5]):
            s, e = int(w.start_idx), int(w.end_idx) + 1
            if e > len(vol):
                e = len(vol)
            avg_vol = np.mean(vol[s:e]) if e > s else 0
            if i in (0, 2, 4):  # W1, W3, W5 推动浪
                impulse_vols.append(avg_vol)
            else:  # W2, W4 调整浪
                correction_vols.append(avg_vol)

        if not impulse_vols or not correction_vols:
            return 0.5

        avg_impulse = np.mean(impulse_vols)
        avg_correction = np.mean(correction_vols)

        if avg_correction == 0:
            return 1.0

        ratio = avg_impulse / avg_correction
        # 推动浪均量应该 > 调整浪均量
        # ratio 1.0 = 无差异, > 1.2 = 明显放量, < 0.8 = 反常
        if ratio >= 1.3:
            return 1.0
        elif ratio >= 1.0:
            return 0.5 + 0.5 * (ratio - 1.0) / 0.3
        elif ratio >= 0.8:
            return 0.3
        else:
            return 0.1  # 反常：调整浪比推动浪还放量

    # ── 5. 三大铁律验证 ──

    def validate_iron_rules(self, waves: List[Wave], direction: str) -> List[Tuple[str, bool, str]]:
        """验证Elliott Wave三大铁律。

        对于上涨5浪：起点低→W1高→W2低→W3高→W4低→W5高
        对于下跌5浪：起点高→W1低→W2高→W3低→W4高→W5低
        """
        results = []
        if len(waves) < 5:
            return [('数据不足', False, '至少需要5个浪')]

        w1, w2, w3, w4, w5 = waves[:5]

        if direction == 'up':
            # 铁律1: W2不跌破W1起点
            rule1 = w2.end_price > w1.start_price
            results.append((
                '铁律1: W2不破起点',
                rule1,
                f'W2底={w2.end_price:.3f} vs 起点={w1.start_price:.3f}'
            ))

            # 铁律2: W3不是1/3/5中最短
            rule2 = w3.length >= w1.length or w3.length >= w5.length
            results.append((
                '铁律2: W3不是最短',
                rule2,
                f'W1={w1.length:.3f} W3={w3.length:.3f} W5={w5.length:.3f}'
            ))

            # 铁律3: W4不进入W1价格区域
            rule3 = w4.end_price > w1.end_price
            results.append((
                '铁律3: W4不进W1区域',
                rule3,
                f'W4底={w4.end_price:.3f} vs W1顶={w1.end_price:.3f}'
            ))
        else:
            # 下跌浪：方向反过来
            # 铁律1: W2不涨过W1起点
            rule1 = w2.end_price < w1.start_price
            results.append((
                '铁律1: W2不破起点(下跌)',
                rule1,
                f'W2顶={w2.end_price:.3f} vs 起点={w1.start_price:.3f}'
            ))

            # 铁律2: W3不是最短
            rule2 = w3.length >= w1.length or w3.length >= w5.length
            results.append((
                '铁律2: W3不是最短',
                rule2,
                f'W1={w1.length:.3f} W3={w3.length:.3f} W5={w5.length:.3f}'
            ))

            # 铁律3: W4不进入W1区域
            rule3 = w4.end_price < w1.end_price
            results.append((
                '铁律3: W4不进W1区域(下跌)',
                rule3,
                f'W4顶={w4.end_price:.3f} vs W1底={w1.end_price:.3f}'
            ))

        return results

    # ── 6. 交替规则 ──

    def check_alternation(self, waves: List[Wave]) -> Tuple[bool, str]:
        """检查W2和W4的交替规则。

        如果W2是急跌（时间短、幅度大），W4应该是缓跌（时间长、幅度小），反之亦然。
        """
        if len(waves) < 5:
            return False, '数据不足'

        w2, w4 = waves[1], waves[3]

        w2_speed = w2.length / max(w2.duration, 1)  # 每天变动幅度
        w4_speed = w4.length / max(w4.duration, 1)

        if w2_speed > w4_speed * 1.2:
            return True, 'W2急调W4缓调 ✓'
        elif w4_speed > w2_speed * 1.2:
            return True, 'W2缓调W4急调 ✓'
        elif abs(w2.duration - w4.duration) > min(w2.duration, w4.duration) * 0.5:
            return True, 'W2/W4时间差异明显 ✓'
        else:
            return False, 'W2/W4形态相似，交替不明显'

    # ── 7. 主分析函数 ──

    def analyze(self, df: pd.DataFrame, direction: str = 'auto',
                start_date: Optional[str] = None,
                start_idx: Optional[int] = None) -> WaveResult:
        """执行完整的波浪分析。

        Args:
            df: K线数据（需要 high/low/close/volume 或 最高/最低/收盘/成交量）
            direction: 'up'(上涨5浪), 'down'(下跌5浪), 'auto'(自动判断)
            start_date: 起始日期（可选，从该日期附近找极值点开始）
            start_idx: 直接指定起始索引（可选）

        Returns:
            WaveResult
        """
        # 标准化列名
        df = self._normalize_columns(df)

        # 确定起点
        if start_idx is None:
            if start_date:
                start_idx = self._find_start(df, start_date, direction)
            else:
                start_idx = 0

        if direction == 'auto':
            direction = self._detect_direction(df, start_idx)

        # 第一步：ZigZag找转折点
        zigzag_pivots = self.zigzag(df)
        if len(zigzag_pivots) < self.min_pivots:
            return WaveResult(direction=direction, confidence='低',
                            waves=[], total_score=0)

        # 第二步：从转折点中搜索最优5浪组合
        best_waves, best_score = self._search_waves(
            df, zigzag_pivots, start_idx, direction
        )

        if not best_waves:
            return WaveResult(direction=direction, confidence='低',
                            waves=[], total_score=0)

        # 2.5步：W4延伸校正 — 检查W4等浪型端点后是否有更极端的值
        best_waves = self.extend_w4(df, best_waves, direction)

        # 2.6步：端点精化 — 将浪型端点校正到OHLC真正极值
        best_waves = self.refine_endpoints(df, best_waves, direction)

        # 第三步：多维度评分
        fib_score, fib_details = self.score_fibonacci(best_waves)
        time_score = self.score_time(best_waves)
        vol_score = self.score_volume(df, best_waves)
        iron_rules = self.validate_iron_rules(best_waves, direction)
        alternation, alt_note = self.check_alternation(best_waves)

        # 综合评分
        total = (fib_score * 0.40 +
                time_score * 0.20 +
                vol_score * 0.15 +
                (1.0 if alternation else 0.3) * 0.10 +
                sum(1 for _, passed, _ in iron_rules if passed) / max(len(iron_rules), 1) * 0.15)

        # 置信度
        if total >= 0.7:
            confidence = '高'
        elif total >= 0.45:
            confidence = '中'
        else:
            confidence = '低'

        # 判断当前浪
        current_wave = self._detect_current_wave(df, best_waves, direction)

        result = WaveResult(
            direction=direction,
            waves=best_waves,
            fib_score=round(fib_score, 3),
            time_score=round(time_score, 3),
            volume_score=round(vol_score, 3),
            iron_rules=iron_rules,
            total_score=round(total, 3),
            confidence=confidence,
            current_wave=current_wave,
            fib_ratios=fib_details,
        )

        return result

    # ── 内部方法 ──

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一列名。"""
        rename = {}
        if 'trade_date' in df.columns:
            rename['trade_date'] = 'date'
        if '日期' in df.columns:
            rename['日期'] = 'date'
        if '最高' in df.columns:
            rename['最高'] = 'high'
        if '最低' in df.columns:
            rename['最低'] = 'low'
        if '收盘' in df.columns:
            rename['收盘'] = 'close'
        if '开盘' in df.columns:
            rename['开盘'] = 'open'
        if 'vol' in df.columns:
            rename['vol'] = 'volume'
        if '成交量' in df.columns:
            rename['成交量'] = 'volume'
        if rename:
            df = df.rename(columns=rename)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _find_start(self, df: pd.DataFrame, start_date: str, direction: str) -> int:
        """在起始日期附近找极值点。"""
        target = pd.to_datetime(start_date)
        window = 15  # 前后15天
        mask = (df['date'] >= target - pd.Timedelta(days=window)) & \
               (df['date'] <= target + pd.Timedelta(days=window))
        subset = df[mask]
        if subset.empty:
            return 0
        if direction == 'down' or direction == 'auto':
            return int(subset['high'].idxmax())
        else:
            return int(subset['low'].idxmin())

    def _detect_direction(self, df: pd.DataFrame, start_idx: int) -> str:
        """自动判断从起点开始的趋势方向。"""
        # 看起点之后30根K线的走势
        end = min(start_idx + 30, len(df) - 1)
        start_price = df.iloc[start_idx]['close']
        end_price = df.iloc[end]['close']
        return 'down' if end_price < start_price else 'up'

    def _search_waves(self, df: pd.DataFrame, pivots: List[Tuple],
                      start_idx: int, direction: str) -> Tuple[List[Wave], float]:
        """在转折点中搜索最优5浪组合。

        核心逻辑：
        - 上涨5浪：起点(低)→W1(高)→W2(低)→W3(高)→W4(低)→W5(高)
        - 下跌5浪：起点(高)→W1(低)→W2(高)→W3(低)→W4(高)→W5(低)
        """
        # 从pivots中找起点之后的转折点
        usable = [(i, p, t) for i, p, t in pivots if i >= start_idx]
        if len(usable) < 5:
            return [], 0.0

        # 分离高低点
        highs = [(i, p) for i, p, t in usable if t == 'high']
        lows = [(i, p) for i, p, t in usable if t == 'low']

        start_price = float(df.iloc[start_idx]['high' if direction == 'down' else 'low'])

        best_waves = None
        best_score = -1.0

        max_cands = 10  # 每级最多候选数

        if direction == 'down':
            # 下跌5浪：起点(高)→W1(低)→W2(高)→W3(低)→W4(高)→W5(低)
            w1_cands = [(i, p) for i, p in lows if p < start_price][:max_cands]

            for w1_idx, w1_price in w1_cands:
                w2_cands = [(i, p) for i, p in highs
                           if i > w1_idx and p > w1_price][:max_cands]

                for w2_idx, w2_price in w2_cands:
                    # 铁律1: W2不破起点
                    if w2_price >= start_price:
                        continue

                    w3_cands = [(i, p) for i, p in lows
                               if i > w2_idx and p < w2_price][:max_cands]

                    for w3_idx, w3_price in w3_cands:
                        # 铁律2: W3不是最短（粗筛）
                        w1_len = abs(w1_price - start_price)
                        w3_len = abs(w3_price - w2_price)
                        if w3_len < w1_len * 0.3:  # 太短直接跳过
                            continue

                        w4_cands = [(i, p) for i, p in highs
                                   if i > w3_idx and p < w2_price  # 铁律3: W4不进W1区域
                                   and p > w3_price][:max_cands]

                        for w4_idx, w4_price in w4_cands:
                            w5_cands = [(i, p) for i, p in lows
                                       if i > w4_idx and p < w4_price][:max_cands]

                            for w5_idx, w5_price in w5_cands:
                                waves = self._build_waves(
                                    df, start_idx, start_price,
                                    [(w1_idx, w1_price), (w2_idx, w2_price),
                                     (w3_idx, w3_price), (w4_idx, w4_price),
                                     (w5_idx, w5_price)],
                                    direction
                                )

                                # 精确评分
                                fib_s, _ = self.score_fibonacci(waves)
                                time_s = self.score_time(waves)
                                vol_s = self.score_volume(df, waves)
                                # 铁律全过加分
                                iron = self.validate_iron_rules(waves, direction)
                                iron_pass = sum(1 for _, p, _ in iron if p) / max(len(iron), 1)

                                score = fib_s * 0.50 + time_s * 0.20 + vol_s * 0.10 + iron_pass * 0.20

                                if score > best_score:
                                    best_score = score
                                    best_waves = waves

        else:
            # 上涨5浪：起点(低)→W1(高)→W2(低)→W3(高)→W4(低)→W5(高)
            w1_cands = [(i, p) for i, p in highs if p > start_price][:max_cands]

            for w1_idx, w1_price in w1_cands:
                w2_cands = [(i, p) for i, p in lows
                           if i > w1_idx and p < w1_price][:max_cands]

                for w2_idx, w2_price in w2_cands:
                    if w2_price <= start_price:
                        continue

                    w3_cands = [(i, p) for i, p in highs
                               if i > w2_idx and p > w2_price][:max_cands]

                    for w3_idx, w3_price in w3_cands:
                        w1_len = abs(w1_price - start_price)
                        w3_len = abs(w3_price - w2_price)
                        if w3_len < w1_len * 0.3:
                            continue

                        w4_cands = [(i, p) for i, p in lows
                                   if i > w3_idx and p > w1_price  # 铁律3
                                   and p < w3_price][:max_cands]

                        for w4_idx, w4_price in w4_cands:
                            w5_cands = [(i, p) for i, p in highs
                                       if i > w4_idx and p > w4_price][:max_cands]

                            for w5_idx, w5_price in w5_cands:
                                waves = self._build_waves(
                                    df, start_idx, start_price,
                                    [(w1_idx, w1_price), (w2_idx, w2_price),
                                     (w3_idx, w3_price), (w4_idx, w4_price),
                                     (w5_idx, w5_price)],
                                    direction
                                )

                                fib_s, _ = self.score_fibonacci(waves)
                                time_s = self.score_time(waves)
                                vol_s = self.score_volume(df, waves)
                                iron = self.validate_iron_rules(waves, direction)
                                iron_pass = sum(1 for _, p, _ in iron if p) / max(len(iron), 1)

                                score = fib_s * 0.50 + time_s * 0.20 + vol_s * 0.10 + iron_pass * 0.20

                                if score > best_score:
                                    best_score = score
                                    best_waves = waves

        return best_waves or [], best_score

    def _build_waves(self, df: pd.DataFrame, start_idx: int, start_price: float,
                     endpoints: List[Tuple[int, float]], direction: str) -> List[Wave]:
        """根据端点构建浪型列表。"""
        waves = []
        prev_idx = start_idx
        prev_price = start_price
        labels = ['W1', 'W2', 'W3', 'W4', 'W5']

        for i, (end_idx, end_price) in enumerate(endpoints):
            if i < len(labels):
                label = labels[i]
            else:
                label = f'W{i+1}'

            start_date = str(df.iloc[prev_idx].get('date', df.iloc[prev_idx].get('日期', '')))[:10]
            end_date = str(df.iloc[end_idx].get('date', df.iloc[end_idx].get('日期', '')))[:10]

            w = Wave(
                label=label,
                start_idx=prev_idx,
                start_price=prev_price,
                end_idx=end_idx,
                end_price=end_price,
                start_date=start_date,
                end_date=end_date,
            )
            waves.append(w)
            prev_idx = end_idx
            prev_price = end_price

        return waves

    def _detect_current_wave(self, df: pd.DataFrame, waves: List[Wave],
                             direction: str) -> str:
        """判断当前价格处于哪个浪中。"""
        if not waves:
            return '未知'

        current_idx = len(df) - 1
        current_price = float(df.iloc[-1]['close'])

        last_wave = waves[-1]
        if current_idx <= last_wave.end_idx:
            return f'{last_wave.label}（进行中）'

        # 在最后一个浪之后
        if last_wave.label == 'W5':
            return 'W5之后的调整'
        elif last_wave.label in ('W3', 'W4'):
            return f'{last_wave.label}之后的延续'
        else:
            return f'{last_wave.label}之后，位置不明'

    # ── 8. W4延伸校正 ──

    def extend_w4(self, df: pd.DataFrame, waves: List[Wave],
                  direction: str) -> List[Wave]:
        """W4延伸校正：检查W4终点后是否还有更极端的点。

        常见问题：ZigZag在W3结束后找的是第一个显著回调点，
        但实际W4可能延伸到更后面（如159227的W4从4/7延伸到5/11）。

        策略：
        - 从当前W4终点向后搜索，看是否有更极端的W4端点
        - 新W4端点后如果还有足够空间给W5，则采用
        - 同样检查W1和W3是否可以延伸
        """
        if len(waves) < 5:
            return waves

        high_col = df['high'].values
        low_col = df['low'].values
        n = len(df)

        new_waves = list(waves)  # shallow copy

        # 校正调整浪（W2和W4）
        # 推动浪（W1/W3/W5）通常ZigZag找的位置够准
        # 调整浪常被ZigZag截短：第一个反弹/回调点不是真正的W2/W4极值
        for wi in [1, 3]:  # W2和W4
            w = new_waves[wi]
            next_w = new_waves[wi + 1]
            cur_end = int(w.end_idx)
            cur_price = w.end_price

            # 确定搜索方向：找什么类型的极值
            if direction == 'down':
                if w.label in ('W1', 'W3', 'W5'):
                    # 推动浪终点找更低的低
                    search_col = low_col
                    is_min = True
                else:
                    # 调整浪终点找更高的高
                    search_col = high_col
                    is_min = False
            else:
                if w.label in ('W1', 'W3', 'W5'):
                    search_col = high_col
                    is_min = False
                else:
                    search_col = low_col
                    is_min = True

            # 在当前端点后搜索更极端的点
            # 搜索范围：到下一个推动浪（W3/W5）的终点
            # 不能只看下一个浪的起点（可能跟W2终点重叠）
            if wi == 1 and len(new_waves) > 2:
                # W2搜索到W3终点（真正的W2高点可能在W3区间内）
                next_impulse_end = int(new_waves[2].end_idx)
                search_end = min(int(w.end_idx) + 80, n, next_impulse_end)
            elif wi == 3 and len(new_waves) > 4:
                # W4搜索到W5终点
                next_impulse_end = int(new_waves[4].end_idx)
                search_end = min(int(w.end_idx) + 60, n, next_impulse_end)
            else:
                search_end = min(int(w.end_idx) + 60, n)
            if search_end <= cur_end + 2:
                continue

            segment = search_col[cur_end:search_end]
            if len(segment) == 0:
                continue

            if is_min:
                extreme_val = float(segment.min())
                if extreme_val >= cur_price:  # 没有更低
                    continue
                extreme_idx = cur_end + int(np.argmin(segment))
            else:
                extreme_val = float(segment.max())
                if extreme_val <= cur_price:  # 没有更高
                    continue
                extreme_idx = cur_end + int(np.argmax(segment))

            # 找到了更极端的点，检查后续浪是否还有空间
            # W4延伸后W5的空间至少要有W3的10%
            remaining_len = abs(extreme_val - cur_price)
            w3_len = new_waves[2].length if len(new_waves) > 2 else 0
            if w.label == 'W4' and w3_len > 0:
                # 确保W5还有足够跌幅空间
                w5_start = extreme_val
                w5_remaining = search_end - extreme_idx
                if w5_remaining < 3:  # W5至少3天
                    continue

            # 更新端点
            new_end_date = str(df.iloc[extreme_idx].get('date', ''))[:10]
            new_waves[wi] = Wave(
                label=w.label,
                start_idx=w.start_idx,
                start_price=w.start_price,
                end_idx=extreme_idx,
                end_price=extreme_val,
                start_date=w.start_date,
                end_date=new_end_date,
            )

            # 前向传递：更新下一浪起点
            nw = new_waves[wi + 1]
            new_waves[wi + 1] = Wave(
                label=nw.label,
                start_idx=extreme_idx,
                start_price=extreme_val,
                end_idx=nw.end_idx,
                end_price=nw.end_price,
                start_date=new_end_date,
                end_date=nw.end_date,
            )

        # W1回扫：检查W1终点到W2终点之间是否有更极端的W1端点
        # 场景：ZigZag在W1处找了个局部低/高点，但真正的W1底/顶在W2反弹前
        if len(new_waves) >= 2:
            w1 = new_waves[0]
            w2 = new_waves[1]
            # 搜索范围：W1检测终点 → W2终点
            s = int(w1.end_idx)
            e = int(w2.end_idx) + 1
            if e > s + 2:
                if direction == 'down':
                    # 下跌浪W1终点找更低
                    seg = low_col[s:e]
                    true_min = float(seg.min())
                    if true_min < w1.end_price:
                        true_idx = s + int(np.argmin(seg))
                        new_date = str(df.iloc[true_idx].get('date', ''))[:10]
                        new_waves[0] = Wave(
                            label='W1', start_idx=w1.start_idx,
                            start_price=w1.start_price,
                            end_idx=true_idx, end_price=true_min,
                            start_date=w1.start_date, end_date=new_date)
                        # 更新W2起点
                        new_waves[1] = Wave(
                            label='W2', start_idx=true_idx,
                            start_price=true_min,
                            end_idx=w2.end_idx, end_price=w2.end_price,
                            start_date=new_date, end_date=w2.end_date)
                else:
                    # 上涨浪W1终点找更高
                    seg = high_col[s:e]
                    true_max = float(seg.max())
                    if true_max > w1.end_price:
                        true_idx = s + int(np.argmax(seg))
                        new_date = str(df.iloc[true_idx].get('date', ''))[:10]
                        new_waves[0] = Wave(
                            label='W1', start_idx=w1.start_idx,
                            start_price=w1.start_price,
                            end_idx=true_idx, end_price=true_max,
                            start_date=w1.start_date, end_date=new_date)
                        new_waves[1] = Wave(
                            label='W2', start_idx=true_idx,
                            start_price=true_max,
                            end_idx=w2.end_idx, end_price=w2.end_price,
                            start_date=new_date, end_date=w2.end_date)

        return new_waves

    # ── 9. 端点精化（OHLC极值校正） ──

    def refine_endpoints(self, df: pd.DataFrame, waves: List[Wave],
                         direction: str) -> List[Wave]:
        """将浪型端点精化到区间内OHLC真正的极值。

        规则：
        - 推动浪(W1,W3,W5)终点：上涨→区间最高价最高点，下跌→区间最低价最低点
        - 调整浪(W2,W4)终点：上涨→区间最低价最低点，下跌→区间最高价最高点
        - 前向传递：精化后自动更新下一浪起点，保持连续性
        """
        high_col = df['high'].values
        low_col = df['low'].values

        refined = []
        prev_idx = waves[0].start_idx if waves else 0
        prev_price = waves[0].start_price if waves else 0
        # 精化起点
        if direction == 'down':
            # 下跌浪起点应该是高点区间内的最高价
            s = max(0, prev_idx - 2)
            e = min(len(high_col), prev_idx + 3)
            prev_price = float(high_col[s:e].max())
            prev_idx = int(s + np.argmax(high_col[s:e]))
        else:
            # 上涨浪起点应该是低点区间内的最低价
            s = max(0, prev_idx - 2)
            e = min(len(low_col), prev_idx + 3)
            prev_price = float(low_col[s:e].min())
            prev_idx = int(s + np.argmin(low_col[s:e]))

        for i, w in enumerate(waves):
            s = prev_idx
            e = int(w.end_idx) + 1
            label = w.label

            # 确定这个浪终点应该找什么极值
            if direction == 'down':
                # 下跌浪：W1低→W2高→W3低→W4高→W5低
                if label in ('W1', 'W3', 'W5'):
                    # 推动浪终点找最低价
                    seg = low_col[s:e]
                    if len(seg) == 0:
                        new_idx, new_price = w.end_idx, w.end_price
                    else:
                        new_idx = s + int(np.argmin(seg))
                        new_price = float(seg.min())
                else:
                    # 调整浪终点找最高价
                    seg = high_col[s:e]
                    if len(seg) == 0:
                        new_idx, new_price = w.end_idx, w.end_price
                    else:
                        new_idx = s + int(np.argmax(seg))
                        new_price = float(seg.max())
            else:
                # 上涨浪：W1高→W2低→W3高→W4低→W5高
                if label in ('W1', 'W3', 'W5'):
                    seg = high_col[s:e]
                    if len(seg) == 0:
                        new_idx, new_price = w.end_idx, w.end_price
                    else:
                        new_idx = s + int(np.argmax(seg))
                        new_price = float(seg.max())
                else:
                    seg = low_col[s:e]
                    if len(seg) == 0:
                        new_idx, new_price = w.end_idx, w.end_price
                    else:
                        new_idx = s + int(np.argmin(seg))
                        new_price = float(seg.min())

            start_date = str(df.iloc[prev_idx].get('date', ''))[:10]
            end_date = str(df.iloc[new_idx].get('date', ''))[:10]

            refined_w = Wave(
                label=label,
                start_idx=prev_idx,
                start_price=prev_price,
                end_idx=new_idx,
                end_price=new_price,
                start_date=start_date,
                end_date=end_date,
            )
            refined.append(refined_w)
            prev_idx = new_idx
            prev_price = new_price

        return refined

    # ── 9. 格式化输出 ──

    @staticmethod
    def format_result(result: WaveResult) -> str:
        """格式化输出分析结果。"""
        lines = []
        lines.append('=' * 65)
        lines.append(f'  Elliott Wave 波浪分析结果')
        lines.append(f'  方向: {"上涨5浪 ↑" if result.direction == "up" else "下跌5浪 ↓"}'
                    f'  置信度: {result.confidence}')
        lines.append('=' * 65)

        # 浪型详情
        lines.append('')
        lines.append('── 浪型结构 ──')
        for w in result.waves:
            arrow = '↑' if w.direction == 'up' else '↓'
            lines.append(
                f'  {w.label}: {w.start_date} → {w.end_date}  '
                f'{w.start_price:.3f} → {w.end_price:.3f}  '
                f'{arrow} {w.pct_change:+.1f}%  {w.duration}天'
            )

        # Fibonacci比例
        if result.fib_ratios:
            lines.append('')
            lines.append('── Fibonacci 比例匹配 ──')
            for name, info in result.fib_ratios.items():
                bar = '█' * int(info['score'] * 20)
                lines.append(
                    f'  {name}: {info["value"]:.3f}  '
                    f'匹配度 {info["score"]:.2f} {bar}'
                )

        # 评分
        lines.append('')
        lines.append('── 多维评分 ──')
        lines.append(f'  Fibonacci比例: {result.fib_score:.3f}')
        lines.append(f'  时间比例:      {result.time_score:.3f}')
        lines.append(f'  量能配合:      {result.volume_score:.3f}')
        lines.append(f'  综合评分:      {result.total_score:.3f}')

        # 铁律
        lines.append('')
        lines.append('── 三大铁律 ──')
        for name, passed, detail in result.iron_rules:
            icon = '✅' if passed else '❌'
            lines.append(f'  {icon} {name}: {detail}')

        # 当前浪
        if result.current_wave:
            lines.append('')
            lines.append(f'── 当前位置: {result.current_wave} ──')

        return '\n'.join(lines)


def analyze_from_tushare(code: str, start_date: str = None,
                         end_date: str = None, direction: str = 'auto',
                         zigzag_pct: float = 0.05) -> WaveResult:
    """便捷函数：从Tushare拉取数据并分析。

    Args:
        code: 股票代码（如 '159227'，自动补全为 159227.SZ）
        start_date: 起始日期
        direction: 'up'/'down'/'auto'
        zigzag_pct: ZigZag阈值

    Usage:
        result = analyze_from_tushare('159227', start_date='20260101', direction='down')
        print(ElliottWaveAnalyzer.format_result(result))
    """
    import tushare as ts

    ts_code = code
    if '.' not in ts_code:
        # 判断是ETF还是股票
        if ts_code.startswith(('1', '5')):
            ts_code = f'{code}.SZ' if ts_code.startswith('1') else f'{code}.SH'
            asset = 'FD'  # ETF
        else:
            ts_code = f'{code}.SZ' if ts_code.startswith(('0', '3')) else f'{code}.SH'
            asset = 'E'  # 股票

    pro = ts.pro_api()
    df = ts.pro_bar(
        ts_code=ts_code, freq='D',
        start_date=start_date.replace('-', '') if start_date else None,
        end_date=end_date.replace('-', '') if end_date else None,
        adj='qfq', asset=asset
    )

    if df is None or df.empty:
        raise ValueError(f'无法获取 {ts_code} 的数据')

    df = df.sort_values('trade_date').reset_index(drop=True)
    df = df.rename(columns={
        'trade_date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'vol': 'volume',
    })
    df['date'] = pd.to_datetime(df['date'])

    analyzer = ElliottWaveAnalyzer(zigzag_pct=zigzag_pct)
    result = analyzer.analyze(df, direction=direction, start_date=start_date)
    return result


if __name__ == '__main__':
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else '159227'
    direction = sys.argv[2] if len(sys.argv) > 2 else 'down'
    start = sys.argv[3] if len(sys.argv) > 3 else '20260101'
    zz = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

    print(f'分析 {code}, 方向={direction}, 起点={start}, ZigZag阈值={zz}')
    result = analyze_from_tushare(code, start_date=start, direction=direction, zigzag_pct=zz)
    print(ElliottWaveAnalyzer.format_result(result))


# ── 10. K线图画图函数 ──

def plot_wave_chart(df: pd.DataFrame, result: 'WaveResult',
                   analyzer: ElliottWaveAnalyzer = None,
                   output_path: str = 'wave_chart.png',
                   title: str = '', dpi: int = 150):
    """画K线图+波浪标注的专业图表。

    核心原则：
    - K线图（蜡烛图）为底层
    - 浪型线连接K线影线的最高/最低点
    - 标签放在高点上方、低点下方
    - Fibonacci水平线
    - 成交量+MACD子图
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.font_manager import FontProperties

    # 中文字体
    _zh_font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    import os
    if os.path.exists(_zh_font_path):
        _zh_fp = FontProperties(fname=_zh_font_path)
    else:
        _zh_fp = None
    plt.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 确保列名标准化
    if 'date' not in df.columns:
        df = analyzer._normalize_columns(df) if analyzer else df

    fig, axes = plt.subplots(3, 1, figsize=(22, 18),
                              gridspec_kw={'height_ratios': [4, 1.2, 1.2]})
    if title:
        fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98,
                     fontproperties=_zh_fp)
    else:
        name = getattr(result, '_name', '')
        fig.suptitle(f'Elliott Wave 分析  评分={result.total_score:.2f}  置信度={result.confidence}',
                     fontsize=22, fontweight='bold', y=0.98,
                     fontproperties=_zh_fp)

    ax_price, ax_vol, ax_macd = axes
    n = len(df)
    x = np.arange(n)

    opens = df['open'].values if 'open' in df.columns else df['close'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    vols = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(n)
    dates = df['date'].values if 'date' in df.columns else np.arange(n)

    # ── K线（蜡烛图）──
    for i in range(n):
        o, c, h, l = opens[i], closes[i], highs[i], lows[i]
        if c >= o:
            color = '#ef5350'  # 红色=涨（中国市场）
            body_color = '#ef5350'
        else:
            color = '#26a69a'  # 绿色=跌
            body_color = '#26a69a'

        # 影线
        ax_price.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
        # 实体
        body_bottom = min(o, c)
        body_height = abs(c - o)
        rect = mpatches.FancyBboxPatch(
            (i - 0.3, body_bottom), 0.6, max(body_height, 0.001),
            boxstyle="square,pad=0",
            facecolor=body_color, edgecolor=color, linewidth=0.5, zorder=3
        )
        ax_price.add_patch(rect)

    # ── 浪型线和标注 ──
    wave_colors = {
        'W1': '#2196F3', 'W2': '#4CAF50', 'W3': '#2196F3',
        'W4': '#4CAF50', 'W5': '#2196F3',
        'A': '#9C27B0', 'B': '#FF9800', 'C': '#F44336'
    }

    for w in result.waves:
        color = wave_colors.get(w.label, '#555')

        # 浪型线
        ax_price.plot(
            [w.start_idx, w.end_idx],
            [w.start_price, w.end_price],
            color=color, linewidth=2.5, alpha=0.9, zorder=5
        )

        # 端点圆点
        ax_price.plot(w.start_idx, w.start_price, 'o',
                      color=color, markersize=7, zorder=6,
                      markeredgecolor='white', markeredgewidth=1)
        ax_price.plot(w.end_idx, w.end_price, 'o',
                      color=color, markersize=7, zorder=6,
                      markeredgecolor='white', markeredgewidth=1)

        # 标签放在浪的中间位置
        mid_x = (w.start_idx + w.end_idx) / 2
        mid_y = (w.start_price + w.end_price) / 2

        # 标签位置：推动浪放在浪方向侧，调整浪放在另一侧
        if w.direction == 'down':
            va = 'top'
            offset_y = -0.01 * (highs.max() - lows.min())
        else:
            va = 'bottom'
            offset_y = 0.01 * (highs.max() - lows.min())

        ax_price.annotate(
            f'{w.label} {w.pct_change:+.1f}% ({w.duration}根)',
            xy=(mid_x, mid_y + offset_y),
            fontsize=10, fontweight='bold',
            color='white',
            fontproperties=_zh_fp,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.92, edgecolor='white', linewidth=0.8),
            ha='center', va=va, zorder=10
        )

        # 关键点位标注（起点和终点价格+日期）
        # 起点
        if w == result.waves[0]:  # 只标第一浪起点
            start_y = w.start_price
            y_pos = start_y + offset_y * 2
            ax_price.annotate(
                f'{w.start_date}\n{w.start_price:.3f}',
                xy=(w.start_idx, start_y),
                fontsize=8, color='#333',
                ha='center',
                va='bottom' if w.direction == 'up' else 'top',
                zorder=10
            )

        # 终点
        end_y = w.end_price
        if w.direction == 'down':
            va_end = 'top'
            ey = end_y - offset_y
        else:
            va_end = 'bottom'
            ey = end_y + offset_y
        ax_price.annotate(
            f'{w.end_date}\n{w.end_price:.3f}',
            xy=(w.end_idx, end_y),
            fontsize=8, color='#333',
            ha='center', va=va_end, zorder=10
        )

    # ── Fibonacci水平线 ──
    if result.waves:
        w0 = result.waves[0]
        all_prices = [w.start_price for w in result.waves] + [w.end_price for w in result.waves]
        total_high = max(all_prices)
        total_low = min(all_prices)
        price_range = total_high - total_low

        # 关键Fibonacci位
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_colors = ['#999', '#FF9800', '#F44336', '#9C27B0', '#2196F3', '#4CAF50', '#999']
        for level, fc in zip(fib_levels, fib_colors):
            price = total_high - price_range * level
            ax_price.axhline(y=price, color=fc, linestyle='--', alpha=0.4, linewidth=0.7)
            ax_price.text(n - 1, price, f' {level:.3f}={price:.3f}',
                          fontsize=7, color=fc, va='center', ha='left')

    ax_price.set_ylabel('价格', fontsize=14, fontweight='bold')
    ax_price.grid(alpha=0.2)
    step = max(1, n // 15)
    ax_price.set_xticks(x[::step])
    ax_price.set_xticklabels([str(d)[:10] for d in dates[::step]],
                              rotation=45, fontsize=10)

    # ── 成交量 ──
    vol_colors = ['#ef5350' if closes[i] >= opens[i] else '#26a69a' for i in range(n)]
    ax_vol.bar(x, vols, color=vol_colors, alpha=0.7, width=0.8)
    ax_vol.set_ylabel('成交量', fontsize=13, fontweight='bold')
    ax_vol.grid(alpha=0.2)
    ax_vol.set_xticks(x[::step])
    ax_vol.set_xticklabels([str(d)[:10] for d in dates[::step]],
                            rotation=45, fontsize=10)

    # ── 评分条 ──
    ax_macd.set_xlim(0, 1)
    scores = {
        'Fibonacci': result.fib_score,
        '时间': result.time_score,
        '量能': result.volume_score,
        '综合': result.total_score,
    }
    y_pos = 0
    for name, val in scores.items():
        color = '#ef5350' if val < 0.4 else '#FF9800' if val < 0.7 else '#4CAF50'
        ax_macd.barh(y_pos, val, height=0.6, color=color, alpha=0.8)
        ax_macd.text(val + 0.02, y_pos, f'{name}: {val:.3f}',
                      va='center', fontsize=13, fontweight='bold',
                      fontproperties=_zh_fp)
        y_pos += 1
    ax_macd.set_xlim(0, 1.2)
    ax_macd.set_ylim(-0.5, len(scores) - 0.5)
    ax_macd.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4)
    ax_macd.set_yticks([])
    ax_macd.set_xlabel('评分', fontsize=13, fontweight='bold')

    # 铁律结果文字
    iron_text = '  '.join(
        ['✅' if p else '❌' for _, p, _ in result.iron_rules]
    )
    ax_macd.text(0.5, -0.15, f'铁律: {iron_text}',
                  transform=ax_macd.transAxes,
                  fontsize=11, fontweight='bold', ha='center', va='top',
                  fontproperties=_zh_fp)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path
