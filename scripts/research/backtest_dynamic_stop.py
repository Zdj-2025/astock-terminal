#!/usr/bin/env python3
"""动态结构止损 vs 固定止损 回测(2026-08-29)。

问题: 用"最近的中期低点(swing low)"做动态止损，是否优于固定止损？
灵感: 熊猫教练BMS(市场结构突破)——跌破中期低点=结构破坏=走

测试设计:
  入场: 金叉信号(MA5上穿MA10+阳线)——已验证有效(t=5.1)
  五种退出策略(同一入场,只比退出):
    A. 固定止损: 入场价-7%
    B. 梯度回撤: 从浮盈峰值回撤15%(现有梯度兑现)
    C. 动态结构v1: 最近已确认的swing low(左右各5根K线确认,无前视)
    D. 动态结构v2: 倒数第二个swing low(更宽松,给波动空间)
    E. MA20收盘破位

  止损上移规则:
    B/C/D: 止损只上移不下移(跟踪型)
    A/E: 不上移(固定型)

  输出: 每笔交易收益率/持有天数/胜率/最大回撤
"""
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
KDIR = 'data/store/kline'
N_SAMPLE = 500
LOOKBACK = 250
SWING_N = 5          # swing low确认窗口(左右各5根)
SWING_BUFFER = 0.005  # 止损放在swing low下方0.5%


def find_swing_lows(lows, n=SWING_N):
    """找所有swing low,返回(位置,价格)列表。无前视:位置p在p+n时才确认。"""
    sls = []
    for i in range(n, len(lows) - n):
        left_ok = all(lows[i] < lows[i-j] for j in range(1, n+1))
        right_ok = all(lows[i] < lows[i+j] for j in range(1, n+1))
        if left_ok and right_ok:
            sls.append((i, lows[i]))
    return sls


def simulate_trade(prices, lows, entry_idx, ma5_series, ma10_series, opens):
    """从entry_idx开始模拟一笔交易,返回5种策略的结果"""
    entry_price = prices[entry_idx]
    results = {}

    # 初始化
    fixed_stop = entry_price * 0.93          # A: -7%
    peak = entry_price                        # B: 峰值跟踪
    trail_stop = float('inf')                 # B: 梯度回撤15%
    # C/D: 预计算swing lows
    swing_lows = find_swing_lows(lows)
    dyn1_stop = None                          # C: 最近swing low
    dyn2_stop = None                          # D: 倒数第二swing low
    ma20 = pd.Series(prices).rolling(20).mean().values

    max_hold = min(entry_idx + 60, len(prices) - 1)  # 最多持有60天

    for t in range(entry_idx + 1, max_hold + 1):
        px = prices[t]
        lo = lows[t]

        # 更新跟踪值
        peak = max(peak, px)
        trail_stop = peak * 0.85 if peak > entry_price else float('inf')

        # 更新swing low止损(无前视:只用在t-SWING_N之前确认的)
        confirmed = [(p, v) for p, v in swing_lows if p + SWING_N <= t]
        if confirmed:
            # C: 最近的swing low(只上移)
            new_d1 = confirmed[-1][1] * (1 - SWING_BUFFER)
            if dyn1_stop is None or new_d1 > dyn1_stop:
                dyn1_stop = new_d1
            # D: 倒数第二个(更宽松)
            if len(confirmed) >= 2:
                new_d2 = confirmed[-2][1] * (1 - SWING_BUFFER)
                if dyn2_stop is None or new_d2 > dyn2_stop:
                    dyn2_stop = new_d2

        # 检查退出
        exit_reason = {}

        # A: 固定止损
        if px <= fixed_stop:
            exit_reason['A_fixed'] = fixed_stop

        # B: 梯度回撤15%
        if trail_stop != float('inf') and px <= trail_stop:
            exit_reason['B_trail15'] = trail_stop

        # C: 动态swing low
        if dyn1_stop is not None and px <= dyn1_stop:
            exit_reason['C_dyn1'] = dyn1_stop

        # D: 动态swing low v2
        if dyn2_stop is not None and px <= dyn2_stop:
            exit_reason['D_dyn2'] = dyn2_stop

        # E: MA20收盘破位
        if not np.isnan(ma20[t]) and px < ma20[t] and prices[t-1] < ma20[t-1]:
            exit_reason['E_ma20'] = px

        # 记录已触发的策略
        for key, exit_px in exit_reason.items():
            if key not in results:
                results[key] = dict(exit_px=exit_px, exit_idx=t,
                                    ret=(exit_px / entry_price - 1) * 100,
                                    hold_days=t - entry_idx)

        # 全部触发则提前结束
        if len(results) >= 5:
            break

    # 未触发的策略=持有到期末
    for key in ['A_fixed', 'B_trail15', 'C_dyn1', 'D_dyn2', 'E_ma20']:
        if key not in results:
            final_px = prices[max_hold]
            results[key] = dict(exit_px=final_px, exit_idx=max_hold,
                                ret=(final_px / entry_price - 1) * 100,
                                hold_days=max_hold - entry_idx)
    return results


def main():
    random.seed(42)
    all_codes = [c for c in os.listdir(KDIR)
                 if c.isdigit() and len(c) == 6 and c[:2] not in ('51', '15', '56', '58')]
    sample = random.sample(all_codes, min(N_SAMPLE, len(all_codes)))

    # 加载数据+找金叉入场点
    trades = []  # (code, entry_idx)
    data = {}
    for c in sample:
        try:
            df = pd.read_parquet(f'{KDIR}/{c}/D.parquet').sort_values('date')
            if len(df) < LOOKBACK + 60:
                continue
            closes = df['close'].astype(float).values[-(LOOKBACK+60):]
            opens = df['open'].astype(float).values[-(LOOKBACK+60):] if 'open' in df.columns else closes.copy()
            lows = df['low'].astype(float).values[-(LOOKBACK+60):] if 'low' in df.columns else closes.copy()
            ma5 = pd.Series(closes).rolling(5).mean().values
            ma10 = pd.Series(closes).rolling(10).mean().values

            # 找金叉入场
            for t in range(30, len(closes) - 30):
                if np.isnan(ma5[t]) or np.isnan(ma10[t]) or np.isnan(ma5[t-1]) or np.isnan(ma10[t-1]):
                    continue
                if ma5[t] > ma10[t] and ma5[t-1] <= ma10[t-1] and closes[t] > opens[t]:
                    trades.append((c, t))
            data[c] = (closes, lows, opens, ma5, ma10)
        except Exception:
            continue

    print(f"样本: {len(data)}只, 金叉入场信号: {len(trades)}笔")

    # 模拟每笔交易
    all_results = {k: [] for k in ['A_fixed', 'B_trail15', 'C_dyn1', 'D_dyn2', 'E_ma20']}
    for code, entry_idx in trades:
        if code not in data:
            continue
        closes, lows, opens, ma5, ma10 = data[code]
        result = simulate_trade(closes, lows, entry_idx, ma5, ma10, opens)
        for key, r in result.items():
            all_results[key].append(r)

    # 汇总输出
    labels = {
        'A_fixed': 'A.固定止损-7%',
        'B_trail15': 'B.梯度回撤15%',
        'C_dyn1': 'C.动态swing(紧)',
        'D_dyn2': 'D.动态swing(松)',
        'E_ma20': 'E.MA20收盘破',
    }
    print(f"\n{'='*70}")
    print(f"动态结构止损 vs 固定止损 | {len(trades)}笔金叉交易 | 最多持有60天")
    print(f"{'='*70}")
    print(f"{'策略':<20}{'均收益':>8}{'中位':>8}{'胜率':>6}{'均持有':>7}{'最大回撤':>9}")
    print('-' * 65)
    for key in ['A_fixed', 'B_trail15', 'C_dyn1', 'D_dyn2', 'E_ma20']:
        rs = all_results[key]
        rets = [r['ret'] for r in rs]
        holds = [r['hold_days'] for r in rs]
        a = np.array(rets)
        wr = (a > 0).mean()
        # 最大回撤 = min return
        print(f"{labels[key]:<20}{a.mean():>+7.2f}%{np.median(a):>+7.2f}%{wr:>6.0%}"
              f"{np.mean(holds):>6.1f}d{a.min():>+8.2f}%")

    # 关键对比
    print(f"\n关键对比(正值=新方法更好):")
    for new_key, old_key, name in [
        ('C_dyn1', 'A_fixed', 'swing紧 vs 固定'),
        ('C_dyn1', 'B_trail15', 'swing紧 vs 梯度15%'),
        ('D_dyn2', 'A_fixed', 'swing松 vs 固定'),
        ('D_dyn2', 'B_trail15', 'swing松 vs 梯度15%'),
        ('D_dyn2', 'E_ma20', 'swing松 vs MA20'),
    ]:
        new_rets = np.array([r['ret'] for r in all_results[new_key]])
        old_rets = np.array([r['ret'] for r in all_results[old_key]])
        diff = new_rets.mean() - old_rets.mean()
        from scipy import stats as st
        t_stat, p_val = st.ttest_rel(new_rets, old_rets)
        print(f"  {name}: {diff:+.2f}% (t={t_stat:.1f}, p={p_val:.3f})")


if __name__ == '__main__':
    main()
