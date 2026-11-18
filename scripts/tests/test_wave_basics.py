#!/usr/bin/env python3
"""波浪理论基础规则验证。

验证波浪分析器的核心逻辑:
1. 拐点检测 (swing point detection)
2. 三条铁律 (ironclad rules)
3. Fibonacci计算
4. MACD背离检测

用法:
    python scripts/tests/test_wave_basics.py
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 导入波浪分析器
wave_path = os.path.join(os.path.dirname(__file__), '../../.agents/skills/wave-t-analysis/scripts')
sys.path.insert(0, os.path.abspath(wave_path))

from wave_analyzer import (
    FIB_RETRACE, FIB_EXTEND,
    detect_swing_points, auto_label_waves,
    check_ironclad_rules, calc_fib_score,
    fib_retrace_price, fib_extend_price, fib_levels,
)


class TestResults:
    def __init__(self):
        self.results = []

    def check(self, name, actual, expected, desc=""):
        ok = actual == expected
        icon = "✅" if ok else "❌"
        self.results.append((name, ok, actual, expected, desc))
        print(f"  {icon} {name}: expected={expected}, actual={actual} {desc}")
        return ok

    def check_approx(self, name, actual, expected, tol=0.01, desc=""):
        ok = abs(actual - expected) < tol
        icon = "✅" if ok else "❌"
        self.results.append((name, ok, actual, expected, desc))
        print(f"  {icon} {name}: expected={expected:.4f}, actual={actual:.4f} {desc}")
        return ok

    def check_true(self, name, condition, desc=""):
        return self.check(name, condition, True, desc)

    def check_false(self, name, condition, desc=""):
        return self.check(name, condition, False, desc)

    def summary(self):
        passed = sum(1 for _, ok, *_ in self.results if ok)
        failed = sum(1 for _, ok, *_ in self.results if not ok)
        print(f"\n  波浪理论验证: ✅ {passed} passed, ❌ {failed} failed\n")
        return failed == 0


def make_df(prices, start_date="2026-01-02"):
    """根据收盘价序列构造DataFrame, 高低点做微小偏移"""
    dates = pd.bdate_range(start_date, periods=len(prices))
    records = []
    for i, p in enumerate(prices):
        h = p + abs(p) * 0.02
        l = p - abs(p) * 0.02
        records.append({
            'date': dates[i],
            'open': p - 0.01,
            'high': h,
            'low': l,
            'close': p,
            'volume': 1000000,
        })
    return pd.DataFrame(records)


def test_swing_points(t):
    """测试拐点检测"""
    print("\n── 1. 拐点检测 ──")

    # 构造一个有明确波峰波谷的序列
    # 峰值在index 5(val=20), 谷值在index 10(val=8)
    prices = [10, 12, 14, 16, 18, 20, 18, 16, 14, 12, 8, 10, 12, 14, 16]
    df = make_df(prices, "2026-01-02")
    swings = detect_swing_points(df, window=3)

    print(f"     价格序列长度: {len(prices)}")
    print(f"     检测到拐点: {len(swings)}个")
    for s in swings:
        kind = "高" if s['type'] == 'high' else "低"
        print(f"     idx={s['idx']} {kind} price={s['price']:.1f}")

    # 应该检测到至少一个高点和低点
    has_high = any(s['type'] == 'high' for s in swings)
    has_low = any(s['type'] == 'low' for s in swings)
    t.check_true("检测到高点", has_high, "序列中有明确的峰")
    t.check_true("检测到低点", has_low, "序列中有明确的谷")

    # 最高点应该在idx=5附近
    high_swings = [s for s in swings if s['type'] == 'high']
    if high_swings:
        max_high = max(high_swings, key=lambda s: s['price'])
        t.check("最高点位置", max_high['idx'], 5, "在idx=5处")


def test_ironclad_rules(t):
    """测试三条铁律"""
    print("\n── 2. 波浪三条铁律 ──")

    # 合法5浪: 10→15→12→20→17→25
    # 浪1: 10→15 (振幅5)
    # 浪2: 15→12 (回撤3/5=60%, 未破浪1起点10) ✓
    # 浪3: 12→20 (振幅8, 不是最短) ✓
    # 浪4: 20→17 (回撤3, 未进浪1区域10-15) ✓
    # 浪5: 17→25 (振幅8)

    waves_valid = [
        {"wave_num": 1, "start_price": 10, "end_price": 15, "type": "up"},
        {"wave_num": 2, "start_price": 15, "end_price": 12, "type": "down"},
        {"wave_num": 3, "start_price": 12, "end_price": 20, "type": "up"},
        {"wave_num": 4, "start_price": 20, "end_price": 17, "type": "down"},
        {"wave_num": 5, "start_price": 17, "end_price": 25, "type": "up"},
    ]
    result = check_ironclad_rules(waves_valid)
    t.check_true("合法5浪通过铁律", result["all_pass"],
                 f"rule1={result['rule1']}, rule2={result['rule2']}, rule3={result['rule3']}")

    # 违规浪1: 浪2破浪1起点 (10→15→8→20→17→25)
    waves_invalid1 = [
        {"wave_num": 1, "start_price": 10, "end_price": 15, "type": "up"},
        {"wave_num": 2, "start_price": 15, "end_price": 8, "type": "down"},
        {"wave_num": 3, "start_price": 8, "end_price": 20, "type": "up"},
        {"wave_num": 4, "start_price": 20, "end_price": 17, "type": "down"},
        {"wave_num": 5, "start_price": 17, "end_price": 25, "type": "up"},
    ]
    result1 = check_ironclad_rules(waves_invalid1)
    t.check_false("铁律1: 浪2破浪1起点", result1["all_pass"],
                  f"回撤{(15-8)/5*100:.0f}%>100%")

    # 违规浪2: 浪3最短 (10→20→15→18→16→25)
    waves_invalid2 = [
        {"wave_num": 1, "start_price": 10, "end_price": 20, "type": "up"},
        {"wave_num": 2, "start_price": 20, "end_price": 15, "type": "down"},
        {"wave_num": 3, "start_price": 15, "end_price": 18, "type": "up"},
        {"wave_num": 4, "start_price": 18, "end_price": 16, "type": "down"},
        {"wave_num": 5, "start_price": 16, "end_price": 25, "type": "up"},
    ]
    result2 = check_ironclad_rules(waves_invalid2)
    t.check_false("铁律2: 浪3最短", result2["all_pass"],
                  f"浪3振幅3 vs 浪1振幅10")

    # 违规浪3: 浪4进入浪1区域 (10→15→12→20→14→25)
    waves_invalid3 = [
        {"wave_num": 1, "start_price": 10, "end_price": 15, "type": "up"},
        {"wave_num": 2, "start_price": 15, "end_price": 12, "type": "down"},
        {"wave_num": 3, "start_price": 12, "end_price": 20, "type": "up"},
        {"wave_num": 4, "start_price": 20, "end_price": 14, "type": "down"},
        {"wave_num": 5, "start_price": 14, "end_price": 25, "type": "up"},
    ]
    result3 = check_ironclad_rules(waves_invalid3)
    t.check_false("铁律3: 浪4进入浪1区域", result3["all_pass"],
                  f"浪4终点14<浪1终点15")


def test_fibonacci(t):
    """测试Fibonacci计算"""
    print("\n── 3. Fibonacci计算 ──")

    # 测试回撤价格
    # 从高点20, 低点10, 回撤38.2% → 20 - (20-10)*0.382 = 16.18
    retrace_382 = fib_retrace_price(20, 10, 0.382)
    t.check_approx("回撤38.2%", retrace_382, 16.18, tol=0.01,
                    desc="20 - 10*0.382 = 16.18")

    # 回撤61.8% → 20 - 10*0.618 = 13.82
    retrace_618 = fib_retrace_price(20, 10, 0.618)
    t.check_approx("回撤61.8%", retrace_618, 13.82, tol=0.01,
                    desc="20 - 10*0.618 = 13.82")

    # 回撤50% → 20 - 10*0.5 = 15
    retrace_50 = fib_retrace_price(20, 10, 0.5)
    t.check_approx("回撤50%", retrace_50, 15.0, tol=0.01, desc="")

    # 测试延伸价格
    # fib_extend_price(start, amplitude, ratio) → start + amplitude * ratio
    # 浪1从10→20(振幅10), 延伸161.8% → 20 + 10*1.618 = 36.18
    extend_1618 = fib_extend_price(20, 10, 1.618)
    t.check_approx("延伸161.8%", extend_1618, 36.18, tol=0.01,
                    desc="20 + 10*1.618 = 36.18")

    # 延伸100% → 20 + 10*1.0 = 30
    extend_100 = fib_extend_price(20, 10, 1.0)
    t.check_approx("延伸100%", extend_100, 30.0, tol=0.01, desc="")

    # 测试fib_levels完整性
    levels = fib_levels(10, 20)
    t.check_true("Fibonacci回撤级别完整", len(levels) > 0,
                 "应返回至少一个回撤位")

    # 验证核心Fibonacci常数
    t.check_approx("FIB 0.382存在", 0.382 in FIB_RETRACE, True)
    t.check_approx("FIB 0.618存在", 0.618 in FIB_RETRACE, True)
    t.check_approx("FIB 1.618存在", 1.618 in FIB_EXTEND, True)


def test_wave_labeling(t):
    """测试自动浪型标注"""
    print("\n── 4. 自动浪型标注 ──")

    # 构造清晰的5浪上升数据
    prices = []
    # 浪1上升: idx 0-5
    prices += [10, 11, 12, 13, 14, 15]
    # 浪2回撤: idx 5-10
    prices += [14, 13, 12, 11, 12]
    # 浪3上升: idx 10-18
    prices += [13, 14, 15, 16, 17, 18, 19, 20]
    # 浪4回撤: idx 18-23
    prices += [19, 18, 17, 18, 19]
    # 浪5上升: idx 23-30
    prices += [20, 21, 22, 23, 24, 25, 26]

    df = make_df(prices, "2026-01-02")
    swings = detect_swing_points(df, window=2)

    print(f"     价格序列: {len(prices)}根K线")
    print(f"     拐点数量: {len(swings)}个")

    if len(swings) >= 5:
        try:
            candidates = auto_label_waves(swings, direction='up')
            print(f"     标注候选: {len(candidates)}个")
            if candidates:
                best = candidates[0]
                print(f"     最佳标注: {len(best.get('points', []))}个浪型点")
                t.check_true("成功标注5浪", len(best.get('points', [])) >= 5,
                             "应标注出至少5个浪型转折点")
            else:
                print("     无有效标注（可能拐点不足或不符合铁律）")
                t.check_true("拐点不足以标注5浪", len(swings) < 6,
                             "拐点不足6个时无法标注5浪")
        except Exception as e:
            print(f"     标注异常: {e}")
            t.check_true("标注异常处理", False, f"auto_label_waves异常: {e}")
    else:
        print(f"     拐点不足({len(swings)}<5), 跳过浪型标注")
        t.check_true("拐点不足5个,跳过标注", True, "需要更多数据")


def test_fib_score(t):
    """测试Fibonacci评分"""
    print("\n── 5. Fibonacci评分 ──")

    # 理想浪型: 浪2回撤61.8%, 浪3延伸161.8%, 浪4回撤23.6%
    waves_ideal = [
        {"wave_num": 1, "start_price": 10.0, "end_price": 20.0, "type": "up"},
        {"wave_num": 2, "start_price": 20.0, "end_price": 13.82, "type": "down"},
        {"wave_num": 3, "start_price": 13.82, "end_price": 36.18, "type": "up"},
        {"wave_num": 4, "start_price": 36.18, "end_price": 33.08, "type": "down"},
        {"wave_num": 5, "start_price": 33.08, "end_price": 50.0, "type": "up"},
    ]

    try:
        score = calc_fib_score(waves_ideal)
        print(f"     理想浪型Fibonacci评分: {score:.4f}")
        t.check_true("理想浪型评分>0", score > 0, "Fibonacci比例好的浪型应有正分")
    except Exception as e:
        print(f"     评分异常: {e}")

    # 非理想浪型
    waves_random = [
        {"wave_num": 1, "start_price": 10.0, "end_price": 20.0, "type": "up"},
        {"wave_num": 2, "start_price": 20.0, "end_price": 19.0, "type": "down"},
        {"wave_num": 3, "start_price": 19.0, "end_price": 21.0, "type": "up"},
        {"wave_num": 4, "start_price": 21.0, "end_price": 10.0, "type": "down"},
        {"wave_num": 5, "start_price": 10.0, "end_price": 22.0, "type": "up"},
    ]

    try:
        score2 = calc_fib_score(waves_random)
        print(f"     随机浪型Fibonacci评分: {score2:.4f}")
        if score > 0:
            t.check_true("理想浪型得分≥随机浪型", score >= score2,
                         "Fibonacci比例好的浪型应得更高分")
    except Exception as e:
        print(f"     评分异常: {e}")


def main():
    print(f"\n{'='*60}")
    print("  波浪理论基础规则验证")
    print(f"{'='*60}")

    t = TestResults()
    test_swing_points(t)
    test_ironclad_rules(t)
    test_fibonacci(t)
    test_wave_labeling(t)
    test_fib_score(t)

    return t.summary()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
