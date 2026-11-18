#!/usr/bin/env python3
"""Swing High止盈回测(2026-08-29)——与swing low止损对称。

问题: 到达前高(swing high)该不该减仓？还是让利润跑到止损？
逻辑: 止损=市场结构破坏(跌破谷);止盈=到达前峰=结构阻力位

五种止盈策略(同一入场=金叉,同一止损=松版swing low):
  A. 纯止损退出: 不主动止盈,让利润跑到swing low破(现有方法)
  B. 前高全出: 到达最近swing high→全仓卖出
  C. 前高半出: 到达swing high→卖50%,剩余跑止损(半仓平衡)
  D. 梯度回撤15%: 峰值回撤15%全出(现有梯度兑现)
  E. 棘轮式: 突破前高后→止损上移到该前高(阻力变支撑),循环

测试关键: A-share反转>动量,到前高"兑现"可能优于"等突破"
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
SWING_N = 5
SWING_BUFFER = 0.005


def _qfq(c):
    c = np.asarray(c, dtype=float)
    rets = np.full(len(c), np.nan)
    rets[1:] = c[1:] / c[:-1] - 1
    cc = c.copy()
    for i in range(len(cc) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            cc[:i + 1] = cc[:i + 1] * (c[i + 1] / c[i])
    return cc


def find_swings(arr, n=SWING_N, find_low=True):
    """找swing low或swing high"""
    swings = []
    for i in range(n, len(arr) - n):
        if find_low:
            ok = all(arr[i] < arr[i-j] for j in range(1, n+1)) and \
                 all(arr[i] < arr[i+j] for j in range(1, n+1))
        else:
            ok = all(arr[i] > arr[i-j] for j in range(1, n+1)) and \
                 all(arr[i] > arr[i+j] for j in range(1, n+1))
        if ok:
            swings.append((i, arr[i]))
    return swings


def simulate(prices, highs, lows, entry_idx):
    """模拟一笔交易,5种止盈策略"""
    entry = prices[entry_idx]
    max_hold = min(entry_idx + 60, len(prices) - 1)

    # 预计算swing lows和highs(用qfq数据)
    q_lows = _qfq(lows)
    q_highs = _qfq(highs)
    swing_lows = find_swings(q_lows, find_low=True)
    swing_highs = find_swings(q_highs, find_low=False)

    # 止损锚: 从最近谷往前找第一个低于它的谷(起涨点)
    stop_anchor = None
    confirmed_lows = [(p, v) for p, v in swing_lows if p + SWING_N <= entry_idx]
    if len(confirmed_lows) >= 2:
        last_low = confirmed_lows[-1]
        for i in range(len(confirmed_lows) - 2, -1, -1):
            if confirmed_lows[i][1] < last_low[1]:
                stop_anchor = confirmed_lows[i][1] * (1 - SWING_BUFFER)
                break
    if stop_anchor is None:
        stop_anchor = entry * 0.93  # 兜底止损

    # 止盈目标: 入场时最近已确认的swing high(高于入场价的)
    confirmed_highs = [(p, v) for p, v in swing_highs
                       if p + SWING_N <= entry_idx and v > entry * 1.03]
    target = confirmed_highs[0][1] if confirmed_highs else None  # 最近的阻力位

    results = {}
    peak = entry

    # 逐日模拟
    for t in range(entry_idx + 1, max_hold + 1):
        px = prices[t]
        hi = highs[t]
        lo = lows[t]
        peak = max(peak, hi)

        # 更新止损(松版swing low,只上移)
        new_lows = [(p, v) for p, v in swing_lows if p + SWING_N <= t]
        if len(new_lows) >= 2:
            nl = new_lows[-1]
            for i in range(len(new_lows) - 2, -1, -1):
                if new_lows[i][1] < nl[1]:
                    new_stop = new_lows[i][1] * (1 - SWING_BUFFER)
                    stop_anchor = max(stop_anchor, new_stop)
                    break

        # 策略A: 纯止损(swing low破)
        if 'A_stop_only' not in results and px <= stop_anchor:
            results['A_stop_only'] = dict(exit=stop_anchor, ret=(stop_anchor/entry-1)*100, days=t-entry_idx)

        # 策略B: 到前高全出
        if 'B_target_all' not in results and target is not None and hi >= target:
            results['B_target_all'] = dict(exit=target, ret=(target/entry-1)*100, days=t-entry_idx)

        # 策略C: 到前高半出+剩余止损(加权)
        if 'C_half_target' not in results:
            if target is not None and hi >= target:
                # 前高半出,剩余继续持有到止损
                half_ret = (target/entry-1)*100
                remaining_stop = stop_anchor
                full_ret = 0.5*half_ret + 0.5*(remaining_stop/entry-1)*100
                results['C_half_target'] = dict(exit=(target+stop_anchor)/2, ret=full_ret, days=t-entry_idx)

        # 策略D: 梯度回撤15%
        trail = peak * 0.85 if peak > entry else None
        if 'D_trail15' not in results and trail is not None and px <= trail:
            results['D_trail15'] = dict(exit=trail, ret=(trail/entry-1)*100, days=t-entry_idx)

        # 策略E: 棘轮式(突破前高→止损上移到前高)
        if 'E_ratchet' not in results:
            # 检查是否突破了新的swing high
            new_highs = [(p, v) for p, v in swing_highs if p + SWING_N <= t]
            for p, v in new_highs:
                if hi > v and v > entry and t > p + SWING_N:
                    # 突破了一个swing high → 止损上移到该high
                    stop_anchor = max(stop_anchor, v * 0.995)
            if px <= stop_anchor:
                results['E_ratchet'] = dict(exit=stop_anchor, ret=(stop_anchor/entry-1)*100, days=t-entry_idx)

        if len(results) >= 5:
            break

    # 未触发的=持有到期末
    for key in ['A_stop_only', 'B_target_all', 'C_half_target', 'D_trail15', 'E_ratchet']:
        if key not in results:
            final = prices[max_hold]
            results[key] = dict(exit=final, ret=(final/entry-1)*100, days=max_hold-entry_idx)

    return results


def main():
    random.seed(42)
    all_codes = [c for c in os.listdir(KDIR)
                 if c.isdigit() and len(c) == 6 and c[:2] not in ('51', '15', '56', '58')]
    sample = random.sample(all_codes, min(N_SAMPLE, len(all_codes)))

    trades = []
    data = {}
    for c in sample:
        try:
            df = pd.read_parquet(f'{KDIR}/{c}/D.parquet').sort_values('date')
            if len(df) < LOOKBACK + 60:
                continue
            closes = df['close'].astype(float).values[-(LOOKBACK+60):]
            highs = df['high'].astype(float).values[-(LOOKBACK+60):]
            lows = df['low'].astype(float).values[-(LOOKBACK+60):]
            opens = df['open'].astype(float).values[-(LOOKBACK+60):]
            ma5 = pd.Series(closes).rolling(5).mean().values
            ma10 = pd.Series(closes).rolling(10).mean().values
            for t in range(30, len(closes) - 30):
                if np.isnan(ma5[t]) or np.isnan(ma10[t]) or np.isnan(ma5[t-1]) or np.isnan(ma10[t-1]):
                    continue
                if ma5[t] > ma10[t] and ma5[t-1] <= ma10[t-1] and closes[t] > opens[t]:
                    trades.append((c, t))
            data[c] = (closes, highs, lows, opens, ma5, ma10)
        except Exception:
            continue

    print(f"样本: {len(data)}只, 金叉入场: {len(trades)}笔")

    all_results = {k: [] for k in ['A_stop_only', 'B_target_all', 'C_half_target', 'D_trail15', 'E_ratchet']}
    for code, entry_idx in trades:
        if code not in data:
            continue
        closes, highs, lows, opens, ma5, ma10 = data[code]
        result = simulate(closes, highs, lows, entry_idx)
        for key, r in result.items():
            all_results[key].append(r)

    labels = {
        'A_stop_only': 'A.纯止损(swing low)',
        'B_target_all': 'B.到前高全出',
        'C_half_target': 'C.前高半出+止损',
        'D_trail15': 'D.梯度回撤15%',
        'E_ratchet': 'E.棘轮式(破前高移止损)',
    }
    print(f"\n{'='*70}")
    print(f"Swing High止盈回测 | {len(trades)}笔 | 前瞻最多60天")
    print(f"{'='*70}")
    print(f"{'策略':<22}{'均收益':>8}{'中位':>8}{'胜率':>6}{'均持有':>7}{'最差':>8}")
    print('-' * 65)
    for key in ['A_stop_only', 'B_target_all', 'C_half_target', 'D_trail15', 'E_ratchet']:
        rs = all_results[key]
        rets = np.array([r['ret'] for r in rs])
        holds = np.array([r['days'] for r in rs])
        print(f"{labels[key]:<22}{rets.mean():>+7.2f}%{np.median(rets):>+7.2f}%"
              f"{(rets>0).mean():>6.0%}{holds.mean():>6.1f}d{rets.min():>+7.2f}%")

    print(f"\n关键配对比较:")
    from scipy import stats as st
    for new_k, old_k, name in [
        ('B_target_all', 'A_stop_only', '前高全出 vs 纯止损'),
        ('C_half_target', 'A_stop_only', '前高半出 vs 纯止损'),
        ('E_ratchet', 'A_stop_only', '棘轮式 vs 纯止损'),
        ('E_ratchet', 'D_trail15', '棘轮式 vs 梯度15%'),
        ('C_half_target', 'D_trail15', '前高半出 vs 梯度15%'),
    ]:
        n_r = np.array([r['ret'] for r in all_results[new_k]])
        o_r = np.array([r['ret'] for r in all_results[old_k]])
        t_stat, p_val = st.ttest_rel(n_r, o_r)
        diff = n_r.mean() - o_r.mean()
        sig = '✅' if p_val < 0.05 else ('⚠️' if p_val < 0.15 else '❌')
        print(f"  {name}: {diff:+.2f}% (t={t_stat:.1f}, p={p_val:.3f}) {sig}")


if __name__ == '__main__':
    main()
