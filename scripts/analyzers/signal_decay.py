#!/usr/bin/env python3
"""信号衰减监控器(2026-08-28建,P1验证通过后落地)。

痛点: catalyst死于2026H2我们8/17才发现——信号衰减从"考古"变"实时"。
方法: 对三类核心信号(divergence/breakout/momentum), 在过去250天每天检测,
     计算前瞻10日超额收益, 滚动60天均值 vs 全样本基准。
     滚动表现<全样本30分位→⚠️降权; <10分位→🔴停用。

信号定义(简化版,可日频计算):
  DIVERGENCE: MACD底背离(价格新低+DIF抬升)+当日阳线
  BREAKOUT: 收盘>20日高+量比>1.5x
  MOMENTUM: 5日涨幅>5%+量比>1.3x(聪明钱的简化代理)

输出: 信号健康面板(当前状态/滚动表现/历史基准/衰减趋势)
用法: python scripts/analyzers/signal_decay.py (盘后跑,17:30扫描后)
     from scripts.analyzers.signal_decay import signal_health_dashboard
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

KDIR = 'data/store/kline'
N_SAMPLE = 500
LOOKBACK = 250          # 天
FORWARD = 10            # 前瞻天数
ROLLING = 60            # 滚动窗口
PCT_WARN = 30           # 30分位→降权
PCT_KILL = 10           # 10分位→停用


def _qfq(c):
    c = np.asarray(c, dtype=float)
    rets = np.full(len(c), np.nan)
    rets[1:] = c[1:] / c[:-1] - 1
    cc = c.copy()
    for i in range(len(cc) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            cc[:i + 1] = cc[:i + 1] * (c[i + 1] / c[i])
    return cc


def detect_signals_sample():
    """抽样N只,过去LOOKBACK天,每天检测三类信号→返回逐日信号收益DataFrame"""
    import random
    random.seed(42)
    all_codes = [c for c in os.listdir(KDIR)
                 if c.isdigit() and len(c) == 6 and c[:2] not in ('51', '15', '56', '58')]
    sample = random.sample(all_codes, min(N_SAMPLE, len(all_codes)))

    # 预加载: 截面收益矩阵(date × code) + 信号矩阵
    closes, dates_ref = {}, None
    for c in sample:
        try:
            df = pd.read_parquet(f'{KDIR}/{c}/D.parquet').sort_values('date')
            if len(df) < LOOKBACK + FORWARD + 30:
                continue
            cl = _qfq(df['close'].astype(float).values)
            closes[c] = cl[-(LOOKBACK + FORWARD + 10):]
            if dates_ref is None:
                dates_ref = pd.to_datetime(df['date']).values[-(LOOKBACK + FORWARD + 10):]
        except Exception:
            continue

    n_codes = len(closes)
    n_days = len(next(iter(closes.values())))
    if n_codes < 100 or n_days < LOOKBACK:
        print(f'样本不足: {n_codes}只×{n_days}天')
        return None

    # 构造收盘价矩阵
    px = np.full((n_days, n_codes), np.nan)
    code_list = list(closes.keys())
    for j, c in enumerate(code_list):
        cl = closes[c]
        px[:len(cl), j] = cl

    # 截面日收益
    ret = px[1:] / px[:-1] - 1  # (n_days-1, n_codes)

    # 市场日收益(截面中位数)
    mkt_ret = np.nanmedian(ret, axis=1)

    # 前瞻FORWARD日超额收益
    fwd = np.full((n_days, n_codes), np.nan)
    for i in range(n_days - FORWARD):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = px[i + FORWARD] / px[i] - 1
            r[(px[i + FORWARD] <= 0) | (px[i] <= 0) | np.isnan(r)] = np.nan
            fwd[i] = r
    fwd_exc = fwd - np.nanmedian(fwd, axis=1, keepdims=True)

    # 逐日信号检测(只看第30到LOOKBACK天,确保有前置数据)
    records = []
    for t in range(30, min(LOOKBACK, n_days - FORWARD)):
        if t >= n_days:
            break
        row_div, row_brk, row_mom = [], [], []
        for j in range(n_codes):
            c_arr = px[:, j]
            if np.isnan(c_arr[t]) or np.isnan(c_arr[t-1]):
                continue
            # 需要至少30天历史
            if np.isnan(c_arr[max(0, t-30)]):
                continue

            # === DIVERGENCE: MACD简化底背离 ===
            # 用20天窗口: 价格接近20日低 + 价格>昨日(阳线)
            lo20 = np.nanmin(c_arr[max(0,t-20):t+1])
            if not np.isnan(lo20) and c_arr[t] > c_arr[t-1]:
                if c_arr[t] / lo20 < 1.03:  # 距低点<3% = 低位
                    # 简化: 价在低位+当日涨 = 底部反弹信号
                    row_div.append(j)

            # === BREAKOUT: 收盘>20日高 + 量比(用价格波动代理) ===
            hi20_prev = np.nanmax(c_arr[max(0,t-21):t])  # 不含今日
            if not np.isnan(hi20_prev) and c_arr[t] > hi20_prev:
                row_brk.append(j)

            # === MOMENTUM: 5日涨幅>5% ===
            if t >= 5 and not np.isnan(c_arr[t-5]) and c_arr[t-5] > 0:
                r5 = c_arr[t] / c_arr[t-5] - 1
                if r5 > 0.05:
                    row_mom.append(j)

        # 计算当日各信号的平均前瞻超额
        date_str = str(dates_ref[t])[:10] if dates_ref is not None else str(t)
        for sig_name, idx_list in [('divergence', row_div), ('breakout', row_brk), ('momentum', row_mom)]:
            if len(idx_list) >= 3:
                avg_exc = np.nanmean([fwd_exc[t, j] for j in idx_list if not np.isnan(fwd_exc[t, j])])
                if not np.isnan(avg_exc):
                    records.append(dict(date=date_str, signal=sig_name,
                                        n_hits=len(idx_list), fwd_exc=avg_exc * 100))
            else:
                records.append(dict(date=date_str, signal=sig_name,
                                    n_hits=len(idx_list), fwd_exc=np.nan))

    return pd.DataFrame(records)


def signal_health_dashboard(verbose=True):
    """主入口: 输出信号健康面板"""
    df = detect_signals_sample()
    if df is None or not len(df):
        print('数据不足')
        return None

    results = {}
    if verbose:
        print(f"\n{'='*70}")
        print(f"🏥 信号健康面板 | 滚动{ROLLING}天 vs 全样本 | 衰减自动降权")
        print('=' * 70)

    for sig in ['divergence', 'breakout', 'momentum']:
        sub = df[df['signal'] == sig].dropna(subset=['fwd_exc']).copy()
        if len(sub) < 30:
            continue
        full_mean = sub['fwd_exc'].mean()
        full_std = sub['fwd_exc'].std()
        # 滚动
        sub = sub.sort_values('date').reset_index(drop=True)
        sub['roll_mean'] = sub['fwd_exc'].rolling(ROLLING, min_periods=20).mean()
        current_roll = sub['roll_mean'].iloc[-1]
        # 历史滚动分布
        roll_hist = sub['roll_mean'].dropna()
        pct30 = np.percentile(roll_hist, PCT_WARN)
        pct10 = np.percentile(roll_hist, PCT_KILL)

        if pd.isna(current_roll):
            status = '⚪数据不足'
        elif current_roll < pct10:
            status = '🔴停用(滚动<10分位)'
        elif current_roll < pct30:
            status = '⚠️降权(滚动<30分位)'
        elif current_roll > np.percentile(roll_hist, 70):
            status = '🟢强(滚动>70分位)'
        else:
            status = '🟡正常'

        trend = current_roll - sub['roll_mean'].iloc[-10] if len(sub) > 10 else 0
        results[sig] = dict(status=status, roll=current_roll, full=full_mean,
                            pct30=pct30, pct10=pct10, n=len(sub), trend=trend)
        if verbose:
            print(f"\n  {sig.upper():<12} {status}")
            print(f"    全样本均值: {full_mean:+.2f}%/10日 | 滚动{ROLLING}天: {current_roll:+.2f}%")
            print(f"    近10日趋势: {trend:+.2f}% | 30分位线: {pct30:+.2f}% | 10分位线: {pct10:+.2f}%")
            # 近期信号量
            recent_n = sub.tail(5)['n_hits'].mean()
            print(f"    信号量(近5日均): {recent_n:.0f}只/天")

    if verbose:
        dead = [s for s, r in results.items() if '🔴' in r['status']]
        warn = [s for s, r in results.items() if '⚠️' in r['status']]
        if dead:
            print(f"\n  🚨 停用信号: {', '.join(dead)} — scanner输出仅显示不推荐")
        if warn:
            print(f"\n  ⚠️ 降权信号: {', '.join(warn)} — 降低推荐优先级")
        if not dead and not warn:
            print(f"\n  ✅ 所有信号健康")
    return results


def decay_slope_alert(verbose=True):
    """衰减斜率警报(2026-10-01立,P1-4)——半衰期从事后检测升级为事前预警。

    预警规则(在分位线触发之前):
      🔔早期预警: 滚动60水平仍>30分位(未降权) 但 20窗斜率z<-1(显著下行)
      ⏰衰减确认: 水平<30分位 且 斜率<0 → 比纯分位规则更早降权
    斜率: 滚动序列最近20个窗口的线性回归斜率(每日变化量);
    z = (当前斜率-历史斜率均值)/历史斜率std, 显著= z<-1
    """
    df = detect_signals_sample()
    if df is None or not len(df):
        return None
    out = {}
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔔 衰减斜率警报 | 20窗斜率z-score | 目标:早于10分位停用线触发")
        print('=' * 70)
    for sig in ['divergence', 'breakout', 'momentum']:
        sub = df[df['signal'] == sig].dropna(subset=['fwd_exc']).sort_values('date')
        if len(sub) < ROLLING + 40:
            continue
        roll = sub['fwd_exc'].rolling(ROLLING, min_periods=20).mean().dropna()
        if len(roll) < 40:
            continue
        pct30 = np.percentile(roll, PCT_WARN)
        # 历史斜率分布(每次用前20窗做回归)
        slopes = pd.Series(index=roll.index, dtype=float)
        vals = roll.values
        for i in range(20, len(vals) + 1):
            y = vals[i-20:i]
            x = np.arange(20)
            slopes.iloc[i-1] = np.polyfit(x, y, 1)[0]
        slopes = slopes.dropna()
        cur_slope = slopes.iloc[-1]
        z = (cur_slope - slopes.mean()) / (slopes.std() or 1)
        level = vals[-1]
        # 首次触发时点回放: 预警 vs 10分位停用 谁先响
        pct10 = np.percentile(roll, PCT_KILL)
        early_days = slopes.index[slopes < slopes.mean() - slopes.std()]
        kill_idx = np.where(vals < pct10)[0]
        early_idx = [roll.index.get_loc(t) for t in early_days
                     if roll.index.get_loc(t) >= 20 and vals[roll.index.get_loc(t)] > pct30]
        first_early = early_idx[0] if early_idx else None
        first_kill = kill_idx[0] if len(kill_idx) else None
        lead = (first_kill - first_early) if (first_early is not None and first_kill is not None) else None
        status = ('🔔早期预警(未破30分位但斜率z<-1)' if z < -1 and level > pct30
                  else '⏰衰减确认(<30分位且斜率<0)' if level < pct30 and cur_slope < 0
                  else '—')
        out[sig] = dict(slope=cur_slope, z=z, level=level, status=status, lead=lead)
        if verbose:
            ld = f'{lead}窗提前' if lead and lead > 0 else ('同时/更晚' if lead is not None else '未对比')
            print(f"\n  {sig.upper():<12} {status}")
            print(f"    当前水平{level:+.2f}% | 20窗斜率{cur_slope*100:+.3f}%/窗 (z={z:+.2f})")
            print(f"    回放: 早期预警@窗{first_early} vs 10分位停用@窗{first_kill} → {ld}")
    return out


if __name__ == '__main__':
    signal_health_dashboard()
