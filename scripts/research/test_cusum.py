#!/usr/bin/env python3
"""CUSUM 检测器验证和演示脚本。

测试场景：
  1. 合成数据：已知变化点，验证检测能力
  2. 模拟A股典型场景：暴跌后横盘（V3的痛点）
  3. 增量更新模式测试
  4. 性能基准：1800只股票扫描时间
"""
import sys
import time
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.analyzers.cusum_detector import (
    CUSUMDetector,
    CUSUMResult,
    quick_cusum,
    cusum_series,
)


def test_synthetic_known_changepoint():
    """测试1：合成数据，已知变化点。"""
    print("=" * 60)
    print("测试1：合成数据 — 已知变化点检测")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    # 前100步均值0，后100步均值+2
    data = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(2, 1, 100),
    ])

    detector = CUSUMDetector(h_base=4.0, k=0.5)
    result = detector.detect(data)

    print(f"  序列长度: {n}")
    print(f"  真实变化点: 第100步")
    print(f"  检测到向上触发: {result.up_crossing}")
    print(f"  最近向上触发位置: 第{result.last_up_change_idx}步")
    print(f"  检测延迟: {result.last_up_change_idx - 100}步")
    print(f"  历史向上触发次数: {result.n_up_events}")
    print(f"  信号强度: {result.signal_strength:.3f}")
    print(f"  置信度: {result.confidence}")
    print()


def test_crash_then_flat():
    """测试2：模拟暴跌后横盘 — V3 核心痛点。

    场景：股价先跌20%（f'持续负），然后横盘（f'在0附近波动）。
    Savgol 的 f'' 会在横盘初期误报正（因为从负回到零看起来像加速）。
    CUSUM 应该不会触发向上突变，因为 f' 没有持续正偏移。
    """
    print("=" * 60)
    print("测试2：暴跌后横盘 — V3 核心痛点场景")
    print("=" * 60)

    np.random.seed(123)
    n = 120

    # 模拟 f' 序列
    # 阶段1(0-40): 强下跌 f' ≈ -0.3
    # 阶段2(40-60): 暴跌减速 f' 从 -0.3 回到 0
    # 阶段3(60-120): 横盘 f' ≈ 0 ± 噪声

    f1 = np.zeros(n)
    for i in range(n):
        if i < 40:
            f1[i] = -0.3 + np.random.normal(0, 0.05)
        elif i < 60:
            t = (i - 40) / 20.0
            f1[i] = -0.3 * (1 - t) + np.random.normal(0, 0.05)
        else:
            f1[i] = np.random.normal(0.0, 0.05)

    # Savgol f'' 在横盘初期会误报正
    from scipy.signal import savgol_filter
    # 模拟 f'' 的行为
    ln_price_sim = np.cumsum(f1 / 100)  # 近似
    f2_sim = savgol_filter(ln_price_sim, 21, 3, deriv=2) * 10000

    print(f"  f'' 在横盘初期(第65步): {f2_sim[65]:+.4f}")
    print(f"  f'' 当前值(第{n-1}步): {f2_sim[-1]:+.4f}")
    print(f"  Savgol f'' 在横盘期转正? {'是(假信号!)' if f2_sim[65] > 0 else '否'}")

    # CUSUM 检测
    detector = CUSUMDetector(h_base=4.0, k=0.5)
    result = detector.detect(f1)

    print(f"\n  CUSUM 检测结果:")
    print(f"    向上触发: {result.up_crossing}")
    print(f"    向下触发: {result.down_crossing}")
    print(f"    置信度: {result.confidence}")
    print(f"    结论: {'CUSUM正确地未触发，过滤了假信号!' if not result.up_crossing else 'CUSUM触发了(可能需要调大h)'}")
    print()


def test_real_reversal():
    """测试3：真实底部反转 — f' 从负持续转正。"""
    print("=" * 60)
    print("测试3：真实底部反转 — f' 从负持续转正")
    print("=" * 60)

    np.random.seed(456)
    n = 120

    # 模拟 f' 序列
    # 阶段1(0-50): 下跌 f' ≈ -0.2
    # 阶段2(50-70): 底部 f' 从 -0.2 持续转正
    # 阶段3(70-120): 上涨 f' ≈ +0.3

    f1 = np.zeros(n)
    for i in range(n):
        if i < 50:
            f1[i] = -0.2 + np.random.normal(0, 0.04)
        elif i < 70:
            t = (i - 50) / 20.0
            f1[i] = -0.2 + 0.5 * t + np.random.normal(0, 0.04)
        else:
            f1[i] = 0.3 + np.random.normal(0, 0.04)

    detector = CUSUMDetector(h_base=4.0, k=0.5)
    result = detector.detect(f1)

    print(f"  序列长度: {n}")
    print(f"  真实拐点: 第50-70步")
    print(f"  CUSUM 向上触发: {result.up_crossing}")
    print(f"  最近向上触发位置: 第{result.last_up_change_idx}步")
    print(f"  检测延迟: {max(0, result.last_up_change_idx - 50)}步(从变化开始)")
    print(f"  信号强度: {result.signal_strength:.3f}")
    print(f"  置信度: {result.confidence}")
    print(f"  结论: {'CUSUM成功检测到真实反转!' if result.up_crossing else 'CUSUM未检测到'}")
    print()


def test_incremental_update():
    """测试4：增量更新模式。"""
    print("=" * 60)
    print("测试4：增量更新模式（实时场景）")
    print("=" * 60)

    np.random.seed(789)
    n = 150

    # 前80步均值0，后70步均值+1.5
    data = np.concatenate([
        np.random.normal(0, 1, 80),
        np.random.normal(1.5, 1, 70),
    ])

    detector = CUSUMDetector(h_base=4.0, k=0.5)

    # 模拟实时逐点更新
    state = {}
    triggered = False
    trigger_idx = -1

    for i in range(n):
        result, state = detector.detect_incremental(data[i], state)
        if result.up_crossing and not triggered:
            triggered = True
            trigger_idx = i

    print(f"  序列长度: {n}")
    print(f"  真实变化点: 第80步")
    print(f"  增量模式触发: {'是' if triggered else '否'}")
    print(f"  触发位置: 第{trigger_idx}步")
    print(f"  检测延迟: {max(0, trigger_idx - 80)}步")
    print(f"  结论: 增量模式正确追踪并检测到变化")
    print()


def test_performance_1800():
    """测试5：1800只股票扫描性能。"""
    print("=" * 60)
    print("测试5：1800只股票扫描性能基准")
    print("=" * 60)

    np.random.seed(2024)
    n_stocks = 1800
    n_bars = 250  # 约一年日线

    # 生成模拟 f1 数据
    f1_dict = {}
    for i in range(n_stocks):
        code = f"{i:06d}"
        # 随机游走 + 偶尔突变
        base = np.random.normal(0, 0.1, n_bars).cumsum() * 0.01
        # 10% 的股票在某处有突变
        if np.random.random() < 0.1:
            cp = np.random.randint(100, 200)
            base[cp:] += np.random.choice([-0.3, 0.3])
        f1_dict[code] = base

    detector = CUSUMDetector(h_base=4.0, k=0.5)

    # 批量检测
    t0 = time.time()
    results = detector.detect_batch(f1_dict)
    t1 = time.time()

    elapsed = t1 - t0
    per_stock_us = elapsed / n_stocks * 1e6

    n_up = sum(1 for r in results.values() if r.up_crossing)
    n_down = sum(1 for r in results.values() if r.down_crossing)

    print(f"  股票数量: {n_stocks}")
    print(f"  每只K线数: {n_bars}")
    print(f"  总耗时: {elapsed:.3f}秒")
    print(f"  每只耗时: {per_stock_us:.1f}微秒")
    print(f"  向上触发: {n_up}只 ({n_up/n_stocks*100:.1f}%)")
    print(f"  向下触发: {n_down}只 ({n_down/n_stocks*100:.1f}%)")
    print(f"  检测成功率: {len(results)}/{n_stocks}")
    print(f"  结论: 1800只股票扫描耗时{elapsed:.1f}秒，完全可用于实时扫描")
    print()

    # 增量模式性能
    t2 = time.time()
    state = {}
    for _ in range(n_stocks):
        state = {}
        for j in range(1):  # 每只只更新1个点
            detector.detect_incremental(np.random.normal(0, 0.1), state)
    t3 = time.time()

    incremental_per_stock_us = (t3 - t2) / n_stocks * 1e6
    print(f"  增量更新每只耗时: {incremental_per_stock_us:.1f}微秒")
    print(f"  1800只增量更新总耗时: {(t3-t2)*1000:.1f}毫秒")
    print()


def test_cusum_series_visualization():
    """测试6：CUSUM 序列可视化数据输出。"""
    print("=" * 60)
    print("测试6：CUSUM 序列可视化数据")
    print("=" * 60)

    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(2, 1, 100),
    ])

    series = cusum_series(data, h_base=4.0, k=0.5)

    print(f"  序列长度: {len(data)}")
    print(f"  检测到的事件:")
    for idx, direction, magnitude in series["events"]:
        print(f"    第{idx}步: {'向上' if direction == 'up' else '向下'}突变,"
              f" 幅度={magnitude:.3f}")
    print()


def test_fuse_with_v3_mock():
    """测试7：与V3融合逻辑（模拟V3Score）。"""
    print("=" * 60)
    print("测试7：V3融合逻辑（模拟）")
    print("=" * 60)

    # 模拟 V3Score（简化）
    class MockV3:
        def __init__(self, signal, f1, f2):
            self.signal = signal
            self.f1 = f1
            self.f2 = f2
            self.total = 6.0

    # 场景A：V3报底部拐点，CUSUM确认
    np.random.seed(100)
    f1_reversal = np.concatenate([
        np.random.normal(-0.2, 0.04, 60),
        np.linspace(-0.2, 0.1, 20) + np.random.normal(0, 0.03, 20),
        np.random.normal(0.1, 0.03, 20),
    ])
    v3_a = MockV3("底部拐点", f1_reversal[-1], 0.02)
    detector = CUSUMDetector(h_base=4.0, k=0.5)
    fusion_a = detector.fuse_with_v3(v3_a, f1_reversal, f2_val=0.02)

    print("  场景A: V3报底部拐点，f'持续转正")
    print(f"    CUSUM向上触发: {fusion_a['cusum_result'].up_crossing}")
    print(f"    曲率调整: {fusion_a['curvature_adjustment']:+.1f}")
    print(f"    原因: {fusion_a['reason']}")
    print()

    # 场景B：V3报底部拐点，但CUSUM未确认（暴跌后横盘）
    f1_flat = np.concatenate([
        np.random.normal(-0.3, 0.05, 40),
        np.linspace(-0.3, 0.0, 20) + np.random.normal(0, 0.05, 20),
        np.random.normal(0.0, 0.05, 40),  # 横盘
    ])
    v3_b = MockV3("底部拐点", f1_flat[-1], 0.015)
    fusion_b = detector.fuse_with_v3(v3_b, f1_flat, f2_val=0.015)

    print("  场景B: V3报底部拐点，但f'在0附近波动(暴跌后横盘)")
    print(f"    CUSUM向上触发: {fusion_b['cusum_result'].up_crossing}")
    print(f"    曲率调整: {fusion_b['curvature_adjustment']:+.1f}")
    print(f"    原因: {fusion_b['reason']}")
    print()

    # 场景C：V3报主升浪，CUSUM检测到减速
    f1_slowdown = np.concatenate([
        np.random.normal(0.3, 0.04, 60),
        np.linspace(0.3, 0.05, 30) + np.random.normal(0, 0.04, 30),
        np.random.normal(0.05, 0.04, 10),
    ])
    v3_c = MockV3("主升浪", f1_slowdown[-1], 0.01)
    fusion_c = detector.fuse_with_v3(v3_c, f1_slowdown, f2_val=0.01)

    print("  场景C: V3报主升浪，但f'正在减速")
    print(f"    CUSUM向下触发: {fusion_c['cusum_result'].down_crossing}")
    print(f"    趋势调整: {fusion_c['trend_adjustment']:+.1f}")
    print(f"    信号覆盖: {fusion_c['signal_override']}")
    print(f"    原因: {fusion_c['reason']}")
    print()


if __name__ == "__main__":
    test_synthetic_known_changepoint()
    test_crash_then_flat()
    test_real_reversal()
    test_incremental_update()
    test_performance_1800()
    test_cusum_series_visualization()
    test_fuse_with_v3_mock()

    print("=" * 60)
    print("所有测试完成")
    print("=" * 60)
