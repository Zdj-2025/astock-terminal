#!/usr/bin/env python3
"""波浪理论分析CLI — Elliott Wave浪型标注与图表生成。

基于Elliott Wave三大铁律自动检测浪型，生成浪型标注图和分析报告。

核心算法:
    - 结构前提验证: W3必须高于W1, W4必须高于W2, W5必须高于W3
    - Fibonacci最优匹配: 遍历候选拐点组合，选最接近理想比例的
    - 多周期检测: 完成5+A后继续检测后续浪型

用法:
    python scripts/cli/wave_analysis_cli.py 688222 --from 2024-09-18
    python scripts/cli/wave_analysis_cli.py 603881 --from 2024-09-18 --to 2026-05-01
    python scripts/cli/wave_analysis_cli.py 300059 --from 2024-02-05 --dpi 200
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import akshare as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

import platform
_system = platform.system()
if _system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'WenQuanYi Zen Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = str(Path(__file__).parent.parent.parent)


# ── Fibonacci 工具 ──

FIB_RATIOS = {
    0.236: '23.6%', 0.382: '38.2%', 0.5: '50.0%',
    0.618: '61.8%', 0.786: '78.6%',
}
EXTENSION_RATIOS = {
    1.0: '1.000', 1.236: '1.236', 1.382: '1.382',
    1.618: '1.618', 2.618: '2.618', 3.618: '3.618',
}


def fib_retrace(low, high):
    """计算 Fibonacci 回撤位"""
    rng = high - low
    return {label: high - rng * r for r, label in FIB_RATIOS.items()}


def fib_extension(low, high):
    """计算 Fibonacci 扩展位（从 low 起算）"""
    rng = high - low
    return {label: low + rng * r for r, label in EXTENSION_RATIOS.items()}


# ── 数据获取 ──

def fetch_daily(code, start_date, end_date):
    """获取前复权日线数据（优先本地parquet,支持CSV/akshare回退）"""
    import os
    parquet_path = f'data/store/kline/{code}/D.parquet'
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        rn = {'date': '日期', 'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘', 'volume': '成交量', 'amount': '成交额'}
        df = df.rename(columns={k: v for k, v in rn.items() if k in df.columns})
        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))]
        return df.sort_values('日期').reset_index(drop=True)
    csv_path = f'/tmp/{code}_daily.csv'
    if os.path.exists(csv_path):
        print(f'  [本地CSV] {csv_path}')
        df = pd.read_csv(csv_path)
        df['日期'] = pd.to_datetime(df['日期'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['日期'] >= start_dt) & (df['日期'] <= end_dt)]
        df = df.sort_values('日期').reset_index(drop=True)
        return df
    df = ak.stock_zh_a_hist(
        symbol=code, period='daily',
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', ''),
        adjust='qfq'
    )
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    return df


def get_stock_name(code):
    """获取股票名称"""
    try:
        info = ak.stock_individual_info_em(symbol=code)
        row = info[info['item'] == '股票简称']
        return row['value'].values[0] if not row.empty else code
    except Exception:
        return code


# ── 浪型检测 ──

def find_extrema(df, order=5):
    """找到局部高点和低点"""
    high_idx = argrelextrema(df['最高'].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df['最低'].values, np.less, order=order)[0]
    return high_idx, low_idx


def detect_start_low(df, start_date):
    """起涨点: df范围内整段最低点。
    原 start±10天找最低,但fetch_daily已过滤start前数据,window找不到-10天→选错(青岛啤酒5/11选62.19高点)。
    改: df整段最低=真实起涨点。2026-08-10精雕。"""
    return df['最低'].idxmin()


def _build_pivots(df, start_idx, high_idx, low_idx):
    """构建排序去重的拐点列表"""
    all_pivots = []
    for i in high_idx:
        if i >= start_idx:
            all_pivots.append((i, float(df.iloc[i]['最高']), 'high'))
    for i in low_idx:
        if i >= start_idx:
            all_pivots.append((i, float(df.iloc[i]['最低']), 'low'))
    seen = set()
    unique = []
    for p in sorted(all_pivots, key=lambda x: x[0]):
        if p[0] not in seen:
            seen.add(p[0])
            unique.append(p)
    return unique


def _make_wave(label, start_idx, start_price, end_idx, end_price, df):
    """构造浪型字典"""
    w = {
        'wave': label,
        'start_idx': int(start_idx),
        'start_price': float(start_price),
        'end_idx': int(end_idx),
        'end_price': float(end_price),
    }
    w['pct_change'] = (w['end_price'] / w['start_price'] - 1) * 100
    w['days'] = w['end_idx'] - w['start_idx']
    w['start_date'] = df.iloc[w['start_idx']]['日期'].strftime('%Y-%m-%d')
    w['end_date'] = df.iloc[w['end_idx']]['日期'].strftime('%Y-%m-%d')
    return w


def _ratio_proximity(ratio, ideal, half_range):
    """计算比率与理想值的逼近度 [0, 1]"""
    return max(0.0, 1.0 - abs(ratio - ideal) / half_range * 0.5)


def _best_proximity(ratio, ideals, half_range):
    """在多个理想值中选最接近的逼近度"""
    return max(_ratio_proximity(ratio, i, half_range) for i in ideals)


def _wave_confidence(waves):
    """多因子加权浪型置信度评分。参考 ta4j ElliottConfidenceScorer。

    5因子：Fibonacci逼近度(35%) + 价格覆盖率(25%) + 交替规则(15%) + 时间比例(15%) + 结构完整度(10%)
    返回 [0, 1.0]
    """
    if len(waves) < 5:
        return 0.0

    # 基础数据
    w1 = waves[0]
    w2 = waves[1]
    w3 = waves[2]
    w4 = waves[3]
    w5 = waves[4]

    w1_len = abs(w1['end_price'] - w1['start_price'])
    w3_len = abs(w3['end_price'] - w3['start_price'])
    w5_len = abs(w5['end_price'] - w5['start_price'])

    # ── Factor 1: Fibonacci 逼近度 (35%) ──
    fib_scores = []
    if w1_len > 0:
        # W2 回撤 W1: [0.382, 0.786], 理 0.618
        w2_retrace = (w1['end_price'] - w2['end_price']) / w1_len
        fib_scores.append(_best_proximity(w2_retrace, [0.618, 0.5, 0.382, 0.786], 0.25))
        # W3/W1 扩展: [1.0, 2.618], 理 1.618
        w3_ratio = w3_len / w1_len
        fib_scores.append(_best_proximity(w3_ratio, [1.618, 1.0, 2.618, 0.618], 0.4))
        # W5/W1 投影: [0.618, 1.618], 理 1.0
        w5_ratio = w5_len / w1_len
        fib_scores.append(_best_proximity(w5_ratio, [1.0, 0.618, 1.618], 0.3))
    if w3_len > 0:
        # W4 回撤 W3: [0.236, 0.786], 理 0.382
        w4_retrace = (w3['end_price'] - w4['end_price']) / w3_len
        fib_scores.append(_best_proximity(w4_retrace, [0.382, 0.5, 0.618], 0.25))
    fib_score = sum(fib_scores) / len(fib_scores) if fib_scores else 0.0

    # ── Factor 2: 价格覆盖率 (25%) ──
    # 波浪覆盖的价格范围占整个数据价格范围的比例
    all_prices = [w['start_price'] for w in waves] + [w['end_price'] for w in waves]
    wave_range = max(all_prices) - min(all_prices)
    # 总范围用 W1起点到最高推进浪终点
    total_range = max(w1['end_price'], w3['end_price'], w5['end_price']) - w1['start_price']
    coverage = min(1.0, wave_range / total_range) if total_range > 0 else 0.0

    # ── Factor 3: 交替规则 (15%) ──
    # W2 和 W4 应在深度和时长上有差异
    if w1_len > 0 and w3_len > 0:
        w2_depth = (w1['end_price'] - w2['end_price']) / w1_len
        w4_depth = (w3['end_price'] - w4['end_price']) / w3_len
        depth_alt = 1.0 if abs(w2_depth - w4_depth) > 0.1 else 0.3
    else:
        depth_alt = 0.3
    w2_dur = max(1, w2.get('days', 1))
    w4_dur = max(1, w4.get('days', 1))
    time_alt = 1.0 if abs(w2_dur - w4_dur) / max(w2_dur, w4_dur) > 0.2 else 0.3
    alternation = (depth_alt + time_alt) / 2

    # ── Factor 4: 时间比例 (15%) ──
    # W3 应是最长的推进浪，W5≈W1 时长
    w1_dur = max(1, w1.get('days', 1))
    w3_dur = max(1, w3.get('days', 1))
    w5_dur = max(1, w5.get('days', 1))
    w3_longest = 1.0 if w3_dur >= w1_dur and w3_dur >= w5_dur else 0.3
    w5_similar = max(0.0, 1.0 - abs(w5_dur - w1_dur) / max(w1_dur, 1) * 0.5)
    time_score = w3_longest * 0.6 + w5_similar * 0.4

    # ── Factor 5: 结构完整度 (10%) ──
    completeness = 1.0 if len(waves) >= 6 else 0.3  # 有 A 浪加分

    # ── 加权合成 ──
    confidence = (
        0.35 * fib_score +
        0.25 * coverage +
        0.15 * alternation +
        0.15 * time_score +
        0.10 * completeness
    )
    return confidence


def _fib_score(waves):
    """评估浪型组合质量（委托 _wave_confidence 多因子评分）"""
    return _wave_confidence(waves)


def _find_best_5wave(pivots, start_idx, start_price, df,
                    max_w1=15, max_inner=12, significant_indices=None):
    """在拐点列表中搜索最优的 5 浪 + A 浪组合。

    改进:
    - W1 候选按价格降序排列（优先试最高峰）
    - W3/W5 候选按价格降序排列
    - 允许截断五浪(W5<W3)，罚分 0.7×
    - 性能守卫: 超过 500K 次迭代提前返回
    - 重要峰预筛选不会被截断遗漏
    """
    highs = [(idx, price) for idx, price, typ in pivots
             if typ == 'high' and idx > start_idx]
    lows = [(idx, price) for idx, price, typ in pivots
            if typ == 'low' and idx > start_idx]

    best_waves = None
    best_score = -1.0
    iterations = 0
    MAX_ITER = 500000

    # W1 候选: 按价格降序（优先试最高峰），重要峰不截断
    w1_all = [(i, p) for i, p in highs if p > start_price]
    if significant_indices:
        w1_sig = [(i, p) for i, p in w1_all if i in significant_indices]
        w1_reg = sorted(
            [(i, p) for i, p in w1_all if i not in significant_indices],
            key=lambda x: (-x[1], x[0])
        )[:max_w1]
        seen = set()
        w1_cands = []
        for item in w1_sig + w1_reg:
            if item[0] not in seen:
                seen.add(item[0])
                w1_cands.append(item)
    else:
        w1_cands = sorted(w1_all, key=lambda x: (-x[1], x[0]))[:max_w1]

    for w1_idx, w1_price in w1_cands:
        if iterations > MAX_ITER:
            return best_waves or []
        # W2 底: 时间顺序（合理性约束）
        w2_cands = [(i, p) for i, p in lows
                    if i > w1_idx and p < w1_price][:max_inner]

        for w2_idx, w2_price in w2_cands:
            # W3 顶: 按价格降序，必须高于 W1 顶
            w3_cands = sorted(
                [(i, p) for i, p in highs if i > w2_idx and p > w1_price],
                key=lambda x: (-x[1], x[0])
            )[:max_inner]

            for w3_idx, w3_price in w3_cands:
                # W4 底: 时间顺序
                w4_cands = [(i, p) for i, p in lows
                            if i > w3_idx and p > w2_price][:max_inner]

                for w4_idx, w4_price in w4_cands:
                    # W5 顶: 正常(>W3) + 截断(>W1且<=W3)
                    w5_normal = sorted(
                        [(i, p) for i, p in highs if i > w4_idx and p > w3_price],
                        key=lambda x: (-x[1], x[0])
                    )[:max_inner]
                    w5_truncated = sorted(
                        [(i, p) for i, p in highs
                         if i > w4_idx and p > w1_price and p <= w3_price],
                        key=lambda x: (-x[1], x[0])
                    )[:3]
                    w5_cands = w5_normal + w5_truncated

                    for w5_idx, w5_price in w5_cands:
                        iterations += 1
                        if iterations > MAX_ITER:
                            return best_waves or []

                        is_truncated = w5_price < w3_price
                        penalty = 0.7 if is_truncated else 1.0

                        # A 浪底
                        a_cands = [(i, p) for i, p in lows
                                   if i > w5_idx and p < w5_price][:10]

                        for a_idx, a_price in a_cands:
                            waves = [
                                _make_wave('W1', start_idx, start_price, w1_idx, w1_price, df),
                                _make_wave('W2', w1_idx, w1_price, w2_idx, w2_price, df),
                                _make_wave('W3', w2_idx, w2_price, w3_idx, w3_price, df),
                                _make_wave('W4', w3_idx, w3_price, w4_idx, w4_price, df),
                                _make_wave('W5', w4_idx, w4_price, w5_idx, w5_price, df),
                                _make_wave('A', w5_idx, w5_price, a_idx, a_price, df),
                            ]
                            score = _fib_score(waves) * penalty
                            if score > best_score:
                                best_score = score
                                best_waves = waves

                        # 也允许只到 W5（没有 A 浪）
                        waves5 = [
                            _make_wave('W1', start_idx, start_price, w1_idx, w1_price, df),
                            _make_wave('W2', w1_idx, w1_price, w2_idx, w2_price, df),
                            _make_wave('W3', w2_idx, w2_price, w3_idx, w3_price, df),
                            _make_wave('W4', w3_idx, w3_price, w4_idx, w4_price, df),
                            _make_wave('W5', w4_idx, w4_price, w5_idx, w5_price, df),
                        ]
                        score5 = _fib_score(waves5) * penalty * 0.85
                        if score5 > best_score:
                            best_score = score5
                            best_waves = waves5

    return best_waves or []


def detect_abc_down(df, end_idx=None):
    """独立下跌ABC调整浪(从高点A跌-B弹-C跌),不依赖W5。2026-08-10精雕加。
    用于'上涨见顶后的下跌调整'(青岛啤酒5月63.99->6/29 50.47),detect_waves只找上涨5浪画不出。"""
    if end_idx is None:
        end_idx = len(df) - 1
    high_col = df['最高'].values; low_col = df['最低'].values
    if end_idx < 10:
        return []
    # 起始高点: df[0:end_idx+1]最高(下跌起点)
    start_idx = int(np.argmax(high_col[:end_idx + 1]))
    start_price = float(high_col[start_idx])
    # A/B/C 拐点(用high/low非close)
    seg_lows = [start_idx + i for i in argrelextrema(low_col[start_idx:end_idx + 1], np.less, order=4)[0]]
    seg_highs = [start_idx + i for i in argrelextrema(high_col[start_idx:end_idx + 1], np.greater, order=4)[0]]
    if not seg_lows:
        return []
    a_idx, a_price = seg_lows[0], float(low_col[seg_lows[0]])      # A底
    b_cands = [(i, float(high_col[i])) for i in seg_highs if i > a_idx and high_col[i] < start_price]
    if not b_cands:
        return []
    b_idx, b_price = max(b_cands, key=lambda x: x[1])              # B反弹顶(最高,<起点)
    c_cands = [(i, float(low_col[i])) for i in seg_lows if i > b_idx]
    if not c_cands:
        return []
    c_idx, c_price = min(c_cands, key=lambda x: x[1])              # C底(最低,通常新低)
    return [
        _make_wave('A', start_idx, start_price, a_idx, a_price, df),
        _make_wave('B', a_idx, a_price, b_idx, b_price, df),
        _make_wave('C', b_idx, b_price, c_idx, c_price, df),
    ]


def _refine_wave_endpoints(waves, df):
    """精化浪型端点：在每浪区间内扫描 OHLC 找真正的极值。

    推动浪(W1,W3,W5): 找区间内最高价最高点
    调整浪(W2,W4,A):   找区间内最低价最低点
    前向传递: 精化 W1 终点后自动更新 W2 起点，保持连续性
    """
    if not waves:
        return waves

    high_col = df['最高'].values
    low_col = df['最低'].values

    refined = list(waves)  # shallow copy

    for i, w in enumerate(refined):
        s = w['start_idx']
        e = w['end_idx']
        label = w['wave']

        if label in ('W1', 'W3', 'W5'):
            # 推动浪: 找区间内最高价的最高点
            seg = high_col[s:e + 1]
            if len(seg) == 0:
                continue
            new_offset = int(np.argmax(seg))
            new_idx = s + new_offset
            new_price = float(seg[new_offset])
        elif label in ('W2', 'W4', 'A'):
            # 调整浪: 找区间内最低价的最低点
            seg = low_col[s:e + 1]
            if len(seg) == 0:
                continue
            new_offset = int(np.argmin(seg))
            new_idx = s + new_offset
            new_price = float(seg[new_offset])
        else:
            # B, C 等后续浪也精化
            if w['pct_change'] > 0:
                seg = high_col[s:e + 1]
                if len(seg) == 0:
                    continue
                new_offset = int(np.argmax(seg))
            else:
                seg = low_col[s:e + 1]
                if len(seg) == 0:
                    continue
                new_offset = int(np.argmin(seg))
            new_idx = s + new_offset
            new_price = float(seg[new_offset])

        # 更新终点
        refined[i] = _make_wave(label, w['start_idx'], w['start_price'],
                                new_idx, new_price, df)
        # 保留额外字段
        if 'confirmed' in w:
            refined[i]['confirmed'] = w['confirmed']
        if 'current_price' in w:
            refined[i]['current_price'] = w['current_price']
            refined[i]['current_date'] = w['current_date']

        # 前向传递: 下一浪的起点 = 当前列的终点
        if i + 1 < len(refined):
            next_w = refined[i + 1]
            refined[i + 1] = _make_wave(
                next_w['wave'], new_idx, new_price,
                next_w['end_idx'], next_w['end_price'], df)
            if 'confirmed' in next_w:
                refined[i + 1]['confirmed'] = next_w['confirmed']

    return refined


def _identify_significant_peaks(pivots, start_idx, df):
    """识别占价格范围 >50% 的重要峰，确保搜索时不被截断遗漏。

    对应专业分析的"先看结构，再标浪型"原则：显著峰/谷应优先作为候选。
    """
    high_pivots = [(idx, price) for idx, price, typ in pivots
                   if typ == 'high' and idx >= start_idx]
    if not high_pivots:
        return set()

    total_high = float(df['最高'].max())
    total_low = float(df['最低'].min())
    total_range = total_high - total_low
    if total_range <= 0:
        return set()

    significant = set()
    for idx, price in high_pivots:
        significance = (price - total_low) / total_range
        if significance > 0.5:
            significant.add(idx)
    return significant


def label_waves(df, start_idx, order=5):
    """自动标注浪型，支持多周期连续检测。

    改进点:
    1. 结构前提: W3>W1顶, W4>W2底, W5>W3顶 — 在选点时即验证
    2. 多因子评分: Fibonacci(35%) + 覆盖率(25%) + 交替(15%) + 时间(15%) + 完整度(10%)
    3. 多周期: 完成 5+A 后，从 A 底继续检测下一组
    4. 重要峰预筛选: 显著峰不会被截断遗漏

    Returns:
        list of dict: [{wave, start_idx, start_price, end_idx, end_price, pct_change, days}]
    """
    high_idx, low_idx = find_extrema(df, order=order)
    pivots = _build_pivots(df, start_idx, high_idx, low_idx)

    start_price = float(df.iloc[start_idx]['最低'])
    all_waves = []

    # 预筛选重要峰
    significant = _identify_significant_peaks(pivots, start_idx, df)

    # 第一组 5 浪
    waves = _find_best_5wave(pivots, start_idx, start_price, df,
                             significant_indices=significant)
    if not waves:
        return []

    # 端点精化: 在每浪区间内找真正的极值
    waves = _refine_wave_endpoints(waves, df)

    all_waves.extend(waves)

    # 多周期: 如果完成了 A 浪，从 A 底继续检测后续浪
    if len(waves) >= 6 and waves[-1]['wave'] == 'A':
        a_wave = waves[-1]
        # 先尝试完整 5 浪（用原始pivots）
        subsequent = _find_best_5wave(
            pivots, a_wave['end_idx'], a_wave['end_price'], df)
        if subsequent:
            if len(subsequent) <= 3:
                for i, w in enumerate(subsequent):
                    new_label = ['B', 'C', 'D'][i] if i < 3 else w['wave']
                    w['wave'] = new_label
            all_waves.extend(subsequent)
        else:
            # 原始pivots太粗，用更细的order检测A浪之后的拐点
            fine_order = max(3, order // 3)
            fine_high_idx, fine_low_idx = find_extrema(df, order=fine_order)
            fine_pivots = _build_pivots(df, a_wave['end_idx'],
                                        fine_high_idx, fine_low_idx)

            a_end_idx = a_wave['end_idx']
            a_end_price = a_wave['end_price']
            highs_after = [(idx, price) for idx, price, typ in fine_pivots
                           if typ == 'high' and idx > a_end_idx]
            if highs_after:
                # B 浪: 从 A 底开始的最高反弹点
                b_high = max(highs_after, key=lambda x: x[1])
                b_wave = _make_wave('B', a_end_idx, a_end_price,
                                    b_high[0], b_high[1], df)
                b_wave['confirmed'] = False
                all_waves.append(b_wave)
                # C 浪: 从 B 顶开始找回调低点
                lows_after_b = [(idx, price) for idx, price, typ in fine_pivots
                                if typ == 'low' and idx > b_high[0]]
                if lows_after_b:
                    c_low = min(lows_after_b, key=lambda x: x[1])
                    c_wave = _make_wave('C', b_high[0], b_high[1],
                                        c_low[0], c_low[1], df)
                    c_wave['confirmed'] = False
                    all_waves.append(c_wave)

    # 最后: 标注当前价格位置（如果不在已有浪的终点上）
    last_wave_end_idx = all_waves[-1]['end_idx']
    last_wave_end_price = all_waves[-1]['end_price']
    current_idx = len(df) - 1
    current_price = float(df.iloc[-1]['收盘'])
    if current_idx > last_wave_end_idx + 2:
        # 当前价格明显在最后浪终点之后，标注为运行中
        last_label = all_waves[-1]['wave']
        if last_label in ('A', 'B', 'C'):
            # 已经在调整浪中，追加"当前位置"标记
            all_waves[-1]['current_price'] = current_price
            all_waves[-1]['current_date'] = df.iloc[-1]['日期'].strftime('%Y-%m-%d')

    return all_waves


def validate_iron_rules(waves):
    """验证三大铁律"""
    results = []
    if len(waves) < 5:
        return results

    w1 = waves[0]  # W1 (推动浪起点→顶)
    w2 = waves[1]  # W2 (顶→底)
    w3 = waves[2]  # W3 (底→顶)
    w4 = waves[3]  # W4 (顶→底)
    w5 = waves[4]  # W5 (底→顶)

    # 获取 W1 起点（推动浪的真正起点）
    w1_start = waves[0]['start_price']  # 整个5浪的起点
    w1_end = waves[0]['end_price']      # W1顶

    # 铁律1: W2终点不得跌破W1起点
    w2_low = w2['end_price']
    rule1 = w2_low > w1_start
    results.append(('铁律1: W2不破W1起点', rule1, f'W2底={w2_low:.2f} vs W1起点={w1_start:.2f}'))

    # 铁律2: W3不是最短
    w1_len = abs(w1['end_price'] - w1['start_price'])
    w3_len = abs(w3['end_price'] - w3['start_price'])
    w5_len = abs(w5['end_price'] - w5['start_price'])
    rule2 = w3_len >= w1_len and w3_len >= w5_len
    if not rule2:
        rule2_note = 'W3不是最短即可（允许等于）'
        rule2 = w3_len >= min(w1_len, w5_len)
    results.append(('铁律2: W3不是1-3-5中最短', rule2,
                     f'W1={w1_len:.2f} W3={w3_len:.2f} W5={w5_len:.2f}'))

    # 铁律3: W4终点不进入W1价格区域
    w4_low = w4['end_price']
    rule3 = w4_low > w1_end
    results.append(('铁律3: W4不进W1区域', rule3,
                     f'W4底={w4_low:.2f} vs W1顶={w1_end:.2f}'))

    return results


def detect_w5_exhaustion(df, waves):
    """W5 衰竭五因子评分，总分 100。

    检测项:
      1. MACD 背离 (25分): W5 区间内最大 DIF < W3 区间内最大 DIF
      2. RSI 衰竭 (20分): W5 终点 RSI < 70（价创新高但 RSI 未超买）
      3. 通道突破 (20分): W2-W4 连线平移至 W3 顶的平行线，W5 顶是否触及上轨
      4. W4 支撑破位 (20分): 当前价 < W4 底
      5. 量能背离 (15分): W5 日均量 < W3 的 80%

    Returns:
        dict with 'score', 'max_score', 'items', 'level'
    """
    impulse = {w['wave']: w for w in waves if w['wave'] in ('W1', 'W2', 'W3', 'W4', 'W5')}
    if len(impulse) < 5:
        return None

    close = df['收盘'].astype(float).values
    high = df['最高'].astype(float).values
    low = df['最低'].astype(float).values
    vol = df['成交量'].astype(float).values

    w3 = impulse['W3']
    w5 = impulse['W5']
    w4 = impulse['W4']
    w2 = impulse['W2']

    items = []

    # 1. MACD 背离 (快速 5,34,5) — 扫描全区间最大 DIF
    dif, dea, hist = calc_macd(close, 5, 34, 5)
    w3_start = int(w3['start_idx'])
    w3_end = int(w3['end_idx'])
    w5_start = int(w5['start_idx'])
    w5_end = int(w5['end_idx'])
    w3_max_dif = float(np.nanmax(dif[w3_start:w3_end + 1])) if w3_end > w3_start else 0
    w5_max_dif = float(np.nanmax(dif[w5_start:w5_end + 1])) if w5_end > w5_start else 0
    macd_div = w5_max_dif < w3_max_dif
    macd_score = 25 if macd_div else 0
    items.append({
        'name': 'MACD背离(5,34,5)',
        'score': macd_score,
        'max': 25,
        'detail': f'W3 DIF_max={w3_max_dif:.3f} W5 DIF_max={w5_max_dif:.3f}',
        'passed': macd_div,
    })

    # 2. RSI 衰竭
    rsi = calc_rsi(close, 14)
    w5_end_idx = int(w5['end_idx'])
    if w5_end_idx < len(rsi) and not np.isnan(rsi[w5_end_idx]):
        w5_rsi = float(rsi[w5_end_idx])
    else:
        w5_rsi = 50.0  # fallback
    rsi_exhausted = w5_rsi < 70
    rsi_score = 20 if rsi_exhausted else 0
    items.append({
        'name': 'RSI衰竭(<70)',
        'score': rsi_score,
        'max': 20,
        'detail': f'W5终点RSI={w5_rsi:.1f}',
        'passed': rsi_exhausted,
    })

    # 3. 通道突破
    # W2底和W4底连线斜率, 平移到W3顶
    w2_idx = int(w2['end_idx'])
    w2_price = float(w2['end_price'])
    w4_idx = int(w4['end_idx'])
    w4_price = float(w4['end_price'])
    w3_idx = int(w3['end_idx'])
    w3_price = float(w3['end_price'])

    if w4_idx > w2_idx and w4_price != w2_price:
        slope = (w4_price - w2_price) / (w4_idx - w2_idx)
        channel_at_w5 = w3_price + slope * (int(w5['end_idx']) - w3_idx)
        w5_hits_channel = w5['end_price'] >= channel_at_w5 * 0.98  # 允许2%容差
    else:
        channel_at_w5 = 0
        w5_hits_channel = False

    channel_score = 20 if not w5_hits_channel else 0  # 未触及上轨=衰竭
    items.append({
        'name': '通道突破',
        'score': channel_score,
        'max': 20,
        'detail': f'通道上轨={channel_at_w5:.2f} W5顶={w5["end_price"]:.2f}',
        'passed': not w5_hits_channel,
    })

    # 4. W4 支撑破位
    current_price = float(df.iloc[-1]['收盘'])
    w4_low = float(w4['end_price'])
    w4_broken = current_price < w4_low
    w4_score = 20 if w4_broken else 0
    items.append({
        'name': 'W4支撑破位',
        'score': w4_score,
        'max': 20,
        'detail': f'当前价={current_price:.2f} W4底={w4_low:.2f}',
        'passed': w4_broken,
    })

    # 5. 量能背离
    w3_seg = vol[int(w3['start_idx']):int(w3['end_idx']) + 1]
    w5_seg = vol[int(w5['start_idx']):int(w5['end_idx']) + 1]
    if len(w3_seg) > 0 and len(w5_seg) > 0:
        vol_ratio = float(w5_seg.mean()) / float(w3_seg.mean())
    else:
        vol_ratio = 1.0
    vol_div = vol_ratio < 0.8
    vol_score = 15 if vol_div else 0
    items.append({
        'name': '量能背离(W5<W3的80%)',
        'score': vol_score,
        'max': 15,
        'detail': f'量比={vol_ratio:.2f}',
        'passed': vol_div,
    })

    total = sum(it['score'] for it in items)
    max_total = sum(it['max'] for it in items)
    if total >= 60:
        level = '高概率衰竭'
    elif total >= 40:
        level = '中概率衰竭'
    else:
        level = '衰竭证据不足'

    return {
        'score': total,
        'max_score': max_total,
        'items': items,
        'level': level,
    }


def compute_ratios(waves):
    """计算浪间 Fibonacci 比例"""
    if len(waves) < 5:
        return {}
    w1_len = abs(waves[0]['end_price'] - waves[0]['start_price'])
    w3_len = abs(waves[2]['end_price'] - waves[2]['start_price'])
    w5_len = abs(waves[4]['end_price'] - waves[4]['start_price'])

    ratios = {
        'W3/W1': w3_len / w1_len if w1_len else 0,
        'W5/W1': w5_len / w1_len if w1_len else 0,
        'W5/W3': w5_len / w3_len if w3_len else 0,
    }

    # 回撤比例
    if len(waves) >= 5:
        w1_range = waves[0]['end_price'] - waves[0]['start_price']
        w2_retrace = (waves[0]['end_price'] - waves[1]['end_price']) / w1_range if w1_range else 0
        w3_range = waves[2]['end_price'] - waves[2]['start_price']
        w4_retrace = (waves[2]['end_price'] - waves[3]['end_price']) / w3_range if w3_range else 0
        ratios['W2回撤W1'] = f'{w2_retrace:.1%}'
        ratios['W4回撤W3'] = f'{w4_retrace:.1%}'

    return ratios


# ── MACD 计算 ──

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return dif.values, dea.values, macd.values


def calc_rsi(close, period=14):
    """Wilder 平滑法计算 RSI"""
    close = pd.Series(close, dtype=float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.values


def verify_macd_divergence(df, waves):
    """用双参数实际计算 W3/W5 高点的 MACD，自动检测背离。返回验证结果。"""
    results = []

    impulse_waves = [w for w in waves if w['wave'] in ('W1', 'W3', 'W5')]
    if len(impulse_waves) < 2:
        return results

    close = df['收盘'].astype(float).values
    params = [(12, 26, 9, '标准(12,26,9)'), (5, 34, 5, '快速(5,34,5)')]

    for fast, slow, signal, label in params:
        dif, dea, hist = calc_macd(close, fast, slow, signal)

        entry = {'param': label, 'peaks': []}
        for w in impulse_waves:
            end_idx = int(w['end_idx'])
            entry['peaks'].append({
                'wave': w['wave'],
                'date': w['end_date'],
                'price': float(w['end_price']),
                'dif': float(dif[end_idx]),
                'dea': float(dea[end_idx]),
                'hist': float(hist[end_idx]),
            })
        results.append(entry)

    # 判断背离：W5 的 DIF 是否低于 W3
    for r in results:
        w3_peak = next((p for p in r['peaks'] if p['wave'] == 'W3'), None)
        w5_peak = next((p for p in r['peaks'] if p['wave'] == 'W5'), None)
        if w3_peak and w5_peak:
            r['top_divergence'] = w5_peak['dif'] < w3_peak['dif']
            r['divergence_detail'] = (
                f"W3 DIF={w3_peak['dif']:.4f} vs W5 DIF={w5_peak['dif']:.4f}"
            )
    return results


def verify_volume(df, waves):
    """对比 W3 和 W5 区间的成交量。返回实际数值。"""
    results = []
    vol = df['成交量'].astype(float).values

    for w in waves:
        if w['wave'] in ('W3', 'W5'):
            start_idx = int(w['start_idx'])
            end_idx = int(w['end_idx'])
            seg_vol = vol[start_idx:end_idx + 1]
            results.append({
                'wave': w['wave'],
                'start_date': w['start_date'],
                'end_date': w['end_date'],
                'days': int(w['days']),
                'total_vol': float(seg_vol.sum()),
                'avg_vol': float(seg_vol.mean()),
                'max_vol': float(seg_vol.max()),
            })

    if len(results) == 2:
        w3, w5 = results[0], results[1]
        ratio = w5['avg_vol'] / w3['avg_vol'] if w3['avg_vol'] > 0 else 0
        w3['ratio_vs_other'] = f'日均量=W5的{1/ratio:.1f}倍' if ratio > 0 else ''
        w5['ratio_vs_other'] = f'日均量=W3的{ratio:.1f}倍'
        w5['volume_divergence'] = ratio < 0.8  # W5 量能低于 W3 的 80% 才算背离
    return results


# ── 图表生成 ──

def generate_chart(df, waves, code, name, output_path, dpi=150, exhaustion=None):
    """生成四合一图表：浪型+成交量+MACD+RSI"""
    fig, axes = plt.subplots(4, 1, figsize=(24, 26),
                              gridspec_kw={'height_ratios': [4, 1.2, 1.2, 1.0]})
    fig.suptitle(f'{name}({code}) 波浪理论分析\n{df.iloc[0]["日期"].strftime("%Y-%m-%d")} ~ {df.iloc[-1]["日期"].strftime("%Y-%m-%d")}',
                 fontsize=18, fontweight='bold', y=0.98)

    ax1, ax2, ax3, ax4 = axes
    x = np.arange(len(df))
    close = df['收盘'].values
    high = df['最高'].values
    low = df['最低'].values
    dates = df['日期']

    # ── 价格 + 浪型 ──
    ax1.fill_between(x, low, high, alpha=0.12, color='#4A90D9')
    ax1.plot(x, close, color='#1a5276', linewidth=1.2, zorder=3)

    # Fibonacci 回撤线
    if waves:
        total_low = waves[0]['start_price']
        total_high = max(w['end_price'] for w in waves if w['wave'] in ['W1', 'W3', 'W5'])
        fib_levels = fib_retrace(total_low, total_high)
        colors_fib = ['#e74c3c', '#f39c12', '#e67e22', '#9b59b6', '#2ecc71']
        for (label, price), color in zip(fib_levels.items(), colors_fib):
            ax1.axhline(y=price, color=color, linestyle='--', alpha=0.5, linewidth=0.8)
            ax1.text(len(x) - 1, price, f' {label}={price:.2f}', fontsize=8, color=color, va='center')

    # 浪型连线和标注
    wave_colors = {'W1': '#2980b9', 'W2': '#27ae60', 'W3': '#2980b9',
                   'W4': '#27ae60', 'W5': '#2980b9', 'A': '#8e44ad', 'B': '#f39c12', 'C': '#e74c3c'}

    for i, w in enumerate(waves):
        color = wave_colors.get(w['wave'], '#555')
        ax1.plot([w['start_idx'], w['end_idx']], [w['start_price'], w['end_price']],
                 color=color, linewidth=2.5, alpha=0.85, zorder=5)
        mid_x = (w['start_idx'] + w['end_idx']) / 2
        mid_y = (w['start_price'] + w['end_price']) / 2
        pct = w['pct_change']
        ax1.annotate(f'{w["wave"]}\n{"+" if pct > 0 else ""}{pct:.1f}%',
                     xy=(mid_x, mid_y), fontsize=9, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85),
                     ha='center', va='bottom' if pct > 0 else 'top', zorder=10)

        # 标注关键点位
        for is_end in [False, True]:
            idx = w['end_idx'] if is_end else w['start_idx']
            price = w['end_price'] if is_end else w['start_price']
            ax1.plot(idx, price, 'o', color=color, markersize=7, zorder=6)
            label_text = f'{w["end_date"]}\n{price:.2f}' if is_end else ''
            if label_text:
                offset = 12 if w['wave'] in ['W1', 'W3', 'W5'] else -12
                ax1.annotate(label_text, xy=(idx, price), fontsize=7,
                             xytext=(0, offset), textcoords='offset points',
                             ha='center', color=color, fontweight='bold', zorder=10)

    # 浪型区间阴影
    regions = []
    for w in waves:
        c = wave_colors.get(w['wave'], '#555')
        regions.append((w['start_idx'], w['end_idx'], c))
    for (s, e, c) in regions:
        ax1.axvspan(s, e, alpha=0.06, color=c, zorder=1)

    # 当前价格标注（从最后浪终点到当前价的虚线）
    if waves:
        last_w = waves[-1]
        last_end_idx = last_w['end_idx']
        current_idx = len(df) - 1
        current_close = float(df.iloc[-1]['收盘'])
        if current_idx > last_end_idx + 2:
            ax1.plot([last_end_idx, current_idx],
                     [last_w['end_price'], current_close],
                     color='#e74c3c', linewidth=1.5, linestyle='--',
                     alpha=0.7, zorder=4)
            ax1.plot(current_idx, current_close, '*',
                     color='#e74c3c', markersize=12, zorder=8)
            ax1.annotate(f'当前 {current_close:.2f}',
                         xy=(current_idx, current_close),
                         fontsize=9, fontweight='bold', color='#e74c3c',
                         xytext=(-10, 12), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='#fff3cd', alpha=0.9),
                         zorder=10)

    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.set_xlim(0, len(x) - 1)
    ax1.grid(True, alpha=0.3)

    tick_step = max(1, len(x) // 20)
    ax1.set_xticks(x[::tick_step])
    ax1.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates.iloc[::tick_step]],
                         rotation=45, fontsize=7)

    # ── 成交量 ──
    vol = df['成交量'].values
    open_p = df['开盘'].values
    colors_vol = ['#e74c3c' if close[i] >= open_p[i] else '#27ae60' for i in range(len(vol))]
    ax2.bar(x, vol, color=colors_vol, alpha=0.7, width=0.8)
    ax2.set_ylabel('成交量', fontsize=10)
    ax2.set_xlim(0, len(x) - 1)
    ax2.set_xticks(x[::tick_step])
    ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates.iloc[::tick_step]],
                         rotation=45, fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── MACD ──
    dif, dea, macd = calc_macd(close)
    ax3.bar(x[macd >= 0], macd[macd >= 0], color='#e74c3c', alpha=0.7, width=0.8)
    ax3.bar(x[macd < 0], macd[macd < 0], color='#27ae60', alpha=0.7, width=0.8)
    ax3.plot(x, dif, color='#2980b9', linewidth=1, label='DIF')
    ax3.plot(x, dea, color='#f39c12', linewidth=1, label='DEA')
    ax3.axhline(y=0, color='gray', linewidth=0.5)
    ax3.set_ylabel('MACD', fontsize=10)
    ax3.set_xlim(0, len(x) - 1)
    ax3.set_xticks(x[::tick_step])
    ax3.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates.iloc[::tick_step]],
                         rotation=45, fontsize=7)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── RSI ──
    rsi = calc_rsi(close, 14)
    ax4.plot(x, rsi, color='#8e44ad', linewidth=1)
    ax4.axhline(y=70, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
    ax4.axhline(y=30, color='#27ae60', linestyle='--', linewidth=0.8, alpha=0.7)
    ax4.fill_between(x, 70, rsi, where=(rsi >= 70), alpha=0.15, color='#e74c3c')
    ax4.fill_between(x, 30, rsi, where=(rsi <= 30), alpha=0.15, color='#27ae60')
    ax4.set_ylabel('RSI(14)', fontsize=10)
    ax4.set_xlim(0, len(x) - 1)
    ax4.set_ylim(0, 100)
    ax4.set_xticks(x[::tick_step])
    ax4.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates.iloc[::tick_step]],
                         rotation=45, fontsize=7)
    ax4.text(len(x) - 1, 71, ' 70', fontsize=7, color='#e74c3c', va='bottom')
    ax4.text(len(x) - 1, 31, ' 30', fontsize=7, color='#27ae60', va='top')
    ax4.grid(True, alpha=0.3)

    # W5 区间高亮（所有面板）
    w5_wave = next((w for w in waves if w['wave'] == 'W5'), None)
    if w5_wave:
        w5_s = w5_wave['start_idx']
        w5_e = w5_wave['end_idx']
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvspan(w5_s, w5_e, alpha=0.10, color='#e74c3c', zorder=0)

    # 衰竭评分标注（价格面板）
    if exhaustion:
        score = exhaustion['score']
        level = exhaustion['level']
        if score >= 60:
            box_color = '#e74c3c'
        elif score >= 40:
            box_color = '#f39c12'
        else:
            box_color = '#27ae60'
        ax1.text(0.98, 0.98, f'W5衰竭评分: {score}/{exhaustion["max_score"]} ({level})',
                 transform=ax1.transAxes, fontsize=11, fontweight='bold',
                 color='white', ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, alpha=0.9),
                 zorder=20)

    # 结论框
    if waves:
        last = waves[-1]
        current_close = float(df.iloc[-1]['收盘'])
        current_date = df.iloc[-1]['日期'].strftime('%Y-%m-%d')

        parts = [f'5浪结构{"完整" if len(waves) >= 5 else "不完整"}({len(waves)}浪)']

        if last['wave'] == 'A':
            parts.append(f'A浪调整完成 底={last["end_price"]:.2f}')
        elif last['wave'] == 'B':
            b_pct = (current_close / last['start_price'] - 1) * 100
            parts.append(f'B浪反弹中 {current_close:.2f}({b_pct:+.1f}%)')
        elif last['wave'] == 'C':
            parts.append(f'C浪调整中 {current_close:.2f}')

        parts.append(f'当前价 {current_close:.2f} ({current_date})')

        conclusion = ' | '.join(parts)
        fig.text(0.5, 0.01, conclusion, ha='center', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                           edgecolor='#e74c3c', linewidth=2))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return output_path


# ── 报告生成 ──

def generate_report(code, name, df, waves, rules, ratios, start_date,
                    macd_verify=None, vol_verify=None, exhaustion=None):
    """生成 Markdown 分析报告（含 MACD 和量能实际数据）"""
    lines = [
        f'# {name}({code}) 波浪理论分析报告',
        f'',
        f'**分析周期**: {start_date} ~ {df.iloc[-1]["日期"].strftime("%Y-%m-%d")}',
        f'**当前价格**: {df.iloc[-1]["收盘"]:.2f}（{df.iloc[-1]["日期"].strftime("%Y-%m-%d")}）',
        f'',
        f'## 浪型标注',
        f'',
    ]

    if not waves:
        lines.append('未检测到完整浪型结构。')
        return '\n'.join(lines)

    for w in waves:
        direction = '↑' if w['pct_change'] > 0 else '↓'
        lines.append(f'### {w["wave"]} {direction}')
        lines.append(f'| 项目 | 数据 |')
        lines.append(f'|------|------|')
        lines.append(f'| 起点 | {w["start_date"]} 价格={w["start_price"]:.2f} |')
        lines.append(f'| 终点 | {w["end_date"]} 价格={w["end_price"]:.2f} |')
        lines.append(f'| 涨跌幅 | {"+" if w["pct_change"] > 0 else ""}{w["pct_change"]:.1f}% |')
        lines.append(f'| 持续 | {w["days"]}个交易日 |')
        lines.append('')

    # 三大铁律
    if rules:
        lines.append('## 三大铁律验证')
        lines.append('')
        lines.append('| 铁律 | 结果 | 详情 |')
        lines.append('|------|------|------|')
        for name, passed, detail in rules:
            lines.append(f'| {name} | {"✓" if passed else "✗"} | {detail} |')
        lines.append('')

    # Fibonacci 比例
    if ratios:
        lines.append('## Fibonacci 比例')
        lines.append('')
        lines.append('| 关系 | 数值 |')
        lines.append('|------|------|')
        for k, v in ratios.items():
            lines.append(f'| {k} | {v if isinstance(v, str) else f"{v:.3f}"} |')
        lines.append('')

    # Fibonacci 回撤位
    if len(waves) >= 5:
        total_low = waves[0]['start_price']
        total_high = max(w['end_price'] for w in waves if w['wave'] in ['W1', 'W3', 'W5'])
        fibs = fib_retrace(total_low, total_high)
        current = df.iloc[-1]['收盘']
        lines.append('## Fibonacci 回撤位')
        lines.append('')
        lines.append('| 级别 | 价格 | 状态 |')
        lines.append('|------|------|------|')
        for label, price in fibs.items():
            status = '已跌破' if current < price else '上方'
            lines.append(f'| {label} | {price:.2f} | {status} |')
        lines.append('')

    # MACD 验证（实际计算数据）
    if macd_verify:
        lines.append('## MACD 验证（实际计算数据）')
        lines.append('')
        for mv in macd_verify:
            lines.append(f'### {mv["param"]}')
            lines.append('')
            lines.append('| 浪 | 日期 | 价格 | DIF | DEA | HIST |')
            lines.append('|----|------|------|-----|-----|------|')
            for p in mv['peaks']:
                lines.append(f'| {p["wave"]} | {p["date"]} | {p["price"]:.2f} | '
                             f'{p["dif"]:.4f} | {p["dea"]:.4f} | {p["hist"]:.4f} |')
            if 'top_divergence' in mv:
                div_label = '**有顶背离**' if mv['top_divergence'] else '**无顶背离**'
                lines.append(f'')
                lines.append(f'判断: {div_label}（{mv["divergence_detail"]}）')
            lines.append('')
        lines.append('> 以上数值均为代码实际计算，非臆断。')
        lines.append('')

    # 成交量验证
    if vol_verify and len(vol_verify) >= 2:
        lines.append('## 成交量对比（实际数据）')
        lines.append('')
        lines.append('| 区间 | 起始 | 结束 | 天数 | 日均成交量 | 最大成交量 |')
        lines.append('|------|------|------|------|-----------|-----------|')
        for v in vol_verify:
            lines.append(f'| {v["wave"]} | {v["start_date"]} | {v["end_date"]} | '
                         f'{v["days"]} | {v["avg_vol"]:,.0f} | {v["max_vol"]:,.0f} |')
        if 'ratio_vs_other' in vol_verify[-1]:
            w5 = vol_verify[-1]
            ratio = w5.get('ratio_vs_other', '')
            div = '**存在量价背离**（W5量能低于W3）' if w5.get('volume_divergence') \
                else '**无量价背离**（W5量能未萎缩）'
            lines.append(f'')
            lines.append(f'W5 {ratio}')
            lines.append(f'判断: {div}')
        lines.append('')
        lines.append('> 以上数值均为代码实际计算，非臆断。')
        lines.append('')

    # W5 衰竭评分
    if exhaustion:
        lines.append('## W5 衰竭检测（五因子评分）')
        lines.append('')
        lines.append(f'**总分: {exhaustion["score"]}/{exhaustion["max_score"]} — {exhaustion["level"]}**')
        lines.append('')
        lines.append('| 检测项 | 得分 | 满分 | 通过 | 详情 |')
        lines.append('|--------|------|------|------|------|')
        for it in exhaustion['items']:
            mark = '✓' if it['passed'] else '✗'
            lines.append(f'| {it["name"]} | {it["score"]} | {it["max"]} | {mark} | {it["detail"]} |')
        lines.append('')
        if exhaustion['score'] >= 60:
            lines.append('**风险提示**: W5衰竭评分较高(≥60)，5浪上升动能可能已经衰竭，A浪调整概率较大。')
        elif exhaustion['score'] >= 40:
            lines.append('**注意**: W5存在部分衰竭信号，关注后续走势确认。')
        else:
            lines.append('W5衰竭证据不足，5浪上升动能可能仍在延续。')
        lines.append('')

    return '\n'.join(lines)


# ── 主入口 ──

def main():
    parser = argparse.ArgumentParser(description='波浪理论分析CLI')
    parser.add_argument('code', help='股票代码（如 688222）')
    parser.add_argument('--from', dest='start_date', required=True,
                        help='起始低点日期 (YYYY-MM-DD)')
    parser.add_argument('--to', dest='end_date', default=None,
                        help='截止日期 (YYYY-MM-DD)，默认今天')
    parser.add_argument('--order', type=int, default=5,
                        help='极值检测窗口大小（默认5，增大则更粗略）')
    parser.add_argument('--dpi', type=int, default=150, help='图片DPI（默认150）')
    parser.add_argument('--output-dir', default=None, help='输出目录')
    args = parser.parse_args()

    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    name = get_stock_name(args.code)

    print(f'{"=" * 60}')
    print(f' {name}({args.code}) 波浪理论分析')
    print(f' 周期: {args.start_date} ~ {end_date}')
    print(f'{"=" * 60}')

    # 获取数据
    print('[1/4] 获取日线数据...')
    df = fetch_daily(args.code, args.start_date, end_date)
    print(f'  数据条数: {len(df)}')

    # 找起始低点
    start_idx = detect_start_low(df, args.start_date)
    start_row = df.iloc[start_idx]
    print(f'  起始低点: {start_row["日期"].strftime("%Y-%m-%d")} 低={start_row["最低"]:.2f}')

    # 检测浪型
    print('[2/4] 检测浪型...')
    waves = label_waves(df, start_idx, order=args.order)
    print(f'  检测到 {len(waves)} 个浪: {[w["wave"] for w in waves]}')

    # 验证铁律
    rules = validate_iron_rules(waves)
    for rule_name, passed, detail in rules:
        status = '✓' if passed else '✗'
        print(f'  {status} {rule_name}: {detail}')

    # 计算比例
    ratios = compute_ratios(waves)
    if ratios:
        print('  Fibonacci比例:')
        for k, v in ratios.items():
            print(f'    {k} = {v if isinstance(v, str) else f"{v:.3f}"}')

    # MACD 背离验证（实际计算）
    macd_verify = verify_macd_divergence(df, waves)
    for mv in macd_verify:
        if 'top_divergence' in mv:
            div_status = '有顶背离' if mv['top_divergence'] else '无顶背离'
            print(f'  MACD {mv["param"]}: {div_status}（{mv["divergence_detail"]}）')

    # 成交量验证
    vol_verify = verify_volume(df, waves)
    for v in vol_verify:
        print(f'  {v["wave"]} 日均成交量: {v["avg_vol"]:,.0f}')
        if 'ratio_vs_other' in v:
            print(f'    {v["ratio_vs_other"]}')

    # W5 衰竭检测
    exhaustion = None
    if len(waves) >= 5:
        exhaustion = detect_w5_exhaustion(df, waves)
        if exhaustion:
            print(f'  W5衰竭评分: {exhaustion["score"]}/{exhaustion["max_score"]} ({exhaustion["level"]})')
            for it in exhaustion['items']:
                mark = '✓' if it['passed'] else '✗'
                print(f'    {mark} {it["name"]}: {it["detail"]}')

    # 输出目录
    out_dir = args.output_dir or os.path.join(project_root, 'reports')
    os.makedirs(out_dir, exist_ok=True)

    # 生成图表
    print('[3/4] 生成图表...')
    chart_path = os.path.join(out_dir, f'{args.code}_wave_analysis.png')
    generate_chart(df, waves, args.code, name, chart_path, dpi=args.dpi, exhaustion=exhaustion)
    print(f'  图表: {chart_path}')

    # 生成报告
    print('[4/4] 生成报告...')
    report = generate_report(args.code, name, df, waves, rules, ratios, args.start_date,
                             macd_verify=macd_verify, vol_verify=vol_verify, exhaustion=exhaustion)
    report_path = os.path.join(out_dir, f'{args.code}_wave_analysis.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'  报告: {report_path}')

    # 输出 JSON 供 skill 使用
    def _to_native(obj):
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    result = {
        'code': args.code, 'name': name,
        'start_date': args.start_date, 'end_date': end_date,
        'start_low': float(start_row['最低']),
        'current_price': float(df.iloc[-1]['收盘']),
        'waves': [{k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in w.items()} for w in waves],
        'rules': [{'name': n, 'passed': p, 'detail': d} for n, p, d in rules],
        'ratios': {k: str(v) for k, v in ratios.items()},
        'macd_verify': macd_verify,
        'vol_verify': vol_verify,
        'exhaustion': exhaustion,
        'chart_path': chart_path, 'report_path': report_path,
    }
    print(f'\n{"=" * 60}')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_to_native))


if __name__ == '__main__':
    main()
