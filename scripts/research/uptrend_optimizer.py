#!/usr/bin/env python3
"""主升浪策略参数调优回测。

目标：找到主升浪最优评分权重/阈值，使高分样本稀缺且胜率最高。

回测框架：1800只×每日采样，穷举不同参数组合。
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from itertools import product

DATA_ROOT = Path("data/store")
STOCK_POOL = Path("data/stock_pool.csv")
SG_WINDOW = 21
SG_POLY = 3


def load_stock_pool():
    df = pd.read_csv(STOCK_POOL, dtype={'code': str})
    return df['code'].astype(str).str.zfill(6).tolist()


def load_kline(code):
    path = DATA_ROOT / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    return df


def score_uptrend(f1, f2, dh180, vol_ratio, above_ma20, params):
    """主升浪评分（参数化版本）。"""
    score = 0.0
    
    # f'权重和阈值
    w_f1 = params['w_f1']
    f1_thresholds = params['f1_thresholds']  # [(threshold, score), ...]
    for thr, s in f1_thresholds:
        if f1 > thr:
            score += s * w_f1
            break
    
    # f''权重和阈值
    w_f2 = params['w_f2']
    f2_thresholds = params['f2_thresholds']
    for thr, s in f2_thresholds:
        if f2 > thr:
            score += s * w_f2
            break
    
    # 放量
    w_vol = params['w_vol']
    vol_thresholds = params['vol_thresholds']
    for thr, s in vol_thresholds:
        if vol_ratio > thr:
            score += s * w_vol
            break
    
    # MA20
    if above_ma20:
        score += params['w_ma20']
    
    # 位置
    w_pos = params['w_pos']
    pos_thresholds = params['pos_thresholds']
    for thr, s in pos_thresholds:
        if dh180 < thr:
            score += s * w_pos
            break
    
    return round(score, 1)


def run():
    print("=" * 70)
    print("主升浪策略参数调优")
    print("=" * 70)
    
    codes = load_stock_pool()
    print(f"股票池: {len(codes)} 只\n")
    print("[1/3] 采集数据...")
    
    records = []
    for i, code in enumerate(codes):
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(codes)}")
        df = load_kline(code)
        if df is None or len(df) < 280:
            continue
        close = df['close'].values.astype(float)
        vc = 'vol' if 'vol' in df.columns else 'volume'
        if vc not in df.columns:
            continue
        volume = df[vc].values.astype(float)
        n = len(close)
        
        ln_p = np.log(close)
        f1 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=1) * 100
        f2 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=2) * 10000
        
        for t in range(250, n - 20):
            if t + 20 >= n:
                break
            c = close[:t+1]
            v = volume[:t+1]
            fwd_ret = (close[t+20] / close[t] - 1) * 100
            
            dh180 = (c[-1] / np.max(c[-180:]) - 1) * 100 if t >= 180 else 0
            vol5 = np.mean(v[-5:])
            vol20 = np.mean(v[-20:]) if len(v) >= 20 else vol5
            vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0
            ma20 = np.mean(c[-20:])
            above_ma20 = c[-1] > ma20
            
            records.append({
                'code': code,
                'f1': f1[t], 'f2': f2[t],
                'dh180': dh180, 'vol_ratio': vol_ratio,
                'above_ma20': above_ma20,
                'fwd_ret20': fwd_ret,
            })
    
    df = pd.DataFrame(records)
    print(f"  总样本: {len(df):,}\n")
    
    # ── [2/3] 参数组合穷举 ──
    print("[2/3] 参数组合穷举...")
    
    # 基础条件：主升浪候选（f'>0 + f''>0 + >MA20）
    base = df[(df['f1'] > 0) & (df['f2'] > 0) & (df['above_ma20'])].copy()
    print(f"  主升浪基础候选: {len(base):,} 样本")
    
    # 固定f'和f''的评分阶梯（IC验证这两个是最强因子）
    # 调整权重分配和放量/位置的影响
    
    param_sets = {
        "V3原始(对照)": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 3), (0.15, 2.5), (0.05, 2), (0, 1), (-999, 0)],
            'w_f2': 1.0, 'f2_thresholds': [(0.05, 2), (0.02, 1.5), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 1.0, 'vol_thresholds': [(1.3, 2), (1.0, 1.5), (0.8, 0.5), (-999, 0)],
            'w_ma20': 1.0, 'w_pos': 1.0,
            'pos_thresholds': [(-15, 2), (-5, 1.5), (0, 1), (5, 0.5), (999, 0)],
            'threshold': 8.0,
        },
        "方案A-放量权重翻倍": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 3), (0.15, 2.5), (0.05, 2), (0, 1), (-999, 0)],
            'w_f2': 1.0, 'f2_thresholds': [(0.05, 2), (0.02, 1.5), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 2.0, 'vol_thresholds': [(1.3, 2), (1.0, 1.5), (0.8, 0.5), (-999, 0)],
            'w_ma20': 1.0, 'w_pos': 1.0,
            'pos_thresholds': [(-15, 2), (-5, 1.5), (0, 1), (5, 0.5), (999, 0)],
            'threshold': 8.0,
        },
        "方案B-f'门槛提高": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 3), (0.2, 2), (0.1, 1), (0, 0.5), (-999, 0)],
            'w_f2': 1.0, 'f2_thresholds': [(0.05, 2), (0.02, 1.5), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 1.0, 'vol_thresholds': [(1.3, 2), (1.0, 1.5), (0.8, 0.5), (-999, 0)],
            'w_ma20': 1.0, 'w_pos': 1.0,
            'pos_thresholds': [(-15, 2), (-5, 1.5), (0, 1), (5, 0.5), (999, 0)],
            'threshold': 7.0,
        },
        "方案C-去掉位置,放量加权": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 4), (0.15, 3), (0.05, 2), (0, 1), (-999, 0)],
            'w_f2': 1.0, 'f2_thresholds': [(0.05, 3), (0.02, 2), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 1.5, 'vol_thresholds': [(1.3, 2), (1.0, 1.5), (0.8, 0.5), (-999, 0)],
            'w_ma20': 1.0, 'w_pos': 0.0,
            'pos_thresholds': [(999, 0)],
            'threshold': 8.0,
        },
        "方案D-极简(f'+f''+放量)": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 4), (0.15, 3), (0.05, 2), (0, 1), (-999, 0)],
            'w_f2': 1.0, 'f2_thresholds': [(0.05, 3), (0.02, 2), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 1.0, 'vol_thresholds': [(1.3, 2), (1.0, 1), (-999, 0)],
            'w_ma20': 0.0, 'w_pos': 0.0,
            'pos_thresholds': [(999, 0)],
            'threshold': 7.0,
        },
        "方案E-f'高门槛+放量强": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 4), (0.2, 3), (0.1, 2), (0, 0.5), (-999, 0)],
            'w_f2': 1.0, 'f2_thresholds': [(0.05, 2), (0.02, 1.5), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 2.0, 'vol_thresholds': [(1.3, 2), (1.0, 1.5), (-999, 0)],
            'w_ma20': 1.0, 'w_pos': 0.5,
            'pos_thresholds': [(-10, 1), (0, 0.5), (999, 0)],
            'threshold': 8.0,
        },
        "方案F-IC加权(f''×1.5)": {
            'w_f1': 1.0, 'f1_thresholds': [(0.3, 3), (0.15, 2.5), (0.05, 2), (0, 1), (-999, 0)],
            'w_f2': 1.5, 'f2_thresholds': [(0.05, 2), (0.02, 1.5), (0.01, 1), (0, 0.5), (-999, 0)],
            'w_vol': 1.0, 'vol_thresholds': [(1.3, 2), (1.0, 1.5), (0.8, 0.5), (-999, 0)],
            'w_ma20': 0.5, 'w_pos': 0.5,
            'pos_thresholds': [(-10, 1), (0, 0.5), (999, 0)],
            'threshold': 8.0,
        },
    }
    
    print(f"\n[3/3] 各方案对比")
    print("=" * 90)
    
    print(f"\n  {'方案':25s} | {'阈值':>4s} | {'高分样本':>7s} | {'胜率':>6s} | {'收益':>8s} | {'盈亏比':>6s} | {'单调性':>6s}")
    print("  " + "-" * 80)
    
    for name, params in param_sets.items():
        threshold = params['threshold']
        
        # 评分
        scores = []
        for _, r in base.iterrows():
            s = score_uptrend(r['f1'], r['f2'], r['dh180'], r['vol_ratio'], r['above_ma20'], params)
            scores.append(s)
        base['_score'] = scores
        
        # 高分组
        high = base[base['_score'] >= threshold]
        if len(high) < 50:
            print(f"  {name:25s} | {threshold:>4.1f} | {len(high):>7,} (不足)")
            continue
        
        wr = (high['fwd_ret20'] > 0).mean() * 100
        ar = high['fwd_ret20'].mean()
        avg_w = high[high['fwd_ret20'] > 0]['fwd_ret20'].mean() if (high['fwd_ret20'] > 0).any() else 0
        avg_l = abs(high[high['fwd_ret20'] <= 0]['fwd_ret20'].mean()) if (high['fwd_ret20'] <= 0).any() else 1
        plr = avg_w / avg_l if avg_l > 0 else 999
        
        # 单调性：分数段胜率是否递增
        bins = [(0, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 10)]
        win_rates = []
        for lo, hi in bins:
            g = base[(base['_score'] >= lo) & (base['_score'] < hi)]
            if len(g) >= 30:
                win_rates.append((g['fwd_ret20'] > 0).mean() * 100)
        
        monotonic = "✅" if all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1)) else "❌"
        
        print(f"  {name:25s} | {threshold:>4.1f} | {len(high):>7,} | {wr:>5.1f}% | {ar:>+7.2f}% | {plr:>5.2f} | {monotonic:>6s}")
    
    # ── 最优方案详细分组 ──
    print(f"\n{'─' * 90}")
    print("各方案分组胜率详细对比")
    print("─" * 90)
    
    for name, params in param_sets.items():
        scores = []
        for _, r in base.iterrows():
            s = score_uptrend(r['f1'], r['f2'], r['dh180'], r['vol_ratio'], r['above_ma20'], params)
            scores.append(s)
        base[f'_s_{name}'] = scores
        
        print(f"\n  {name}")
        print(f"  {'分数段':10s} | {'样本':>7s} | {'胜率':>6s} | {'收益':>8s}")
        for lo, hi in [(0, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 10), (9, 12)]:
            g = base[(base[f'_s_{name}'] >= lo) & (base[f'_s_{name}'] < hi)]
            if len(g) < 20:
                continue
            wr = (g['fwd_ret20'] > 0).mean() * 100
            ar = g['fwd_ret20'].mean()
            print(f"  {lo}-{hi:>2d}       | {len(g):>7,} | {wr:>5.1f}% | {ar:>+7.2f}%")
    
    # ── V3原始对照 ──
    print(f"\n{'─' * 90}")
    print("V3原始主升浪(条件筛选，非评分): f'>0 + f''>0 + >MA20")
    print("─" * 90)
    
    for extra_name, extra_mask in [
        ("基础", pd.Series(True, index=base.index)),
        ("+放量(vol>1.0)", base['vol_ratio'] > 1.0),
        ("+放量(vol>1.2)", base['vol_ratio'] > 1.2),
        ("+f'>0.2", base['f1'] > 0.2),
        ("+f'>0.3", base['f1'] > 0.3),
        ("+放量+f'>0.2", (base['vol_ratio'] > 1.0) & (base['f1'] > 0.2)),
        ("+放量+f'>0.3", (base['vol_ratio'] > 1.0) & (base['f1'] > 0.3)),
        ("+f''>0.03", base['f2'] > 0.03),
        ("+放量+f''>0.03", (base['vol_ratio'] > 1.0) & (base['f2'] > 0.03)),
    ]:
        sub = base[extra_mask]
        if len(sub) < 50:
            continue
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        avg_w = sub[sub['fwd_ret20'] > 0]['fwd_ret20'].mean() if (sub['fwd_ret20'] > 0).any() else 0
        avg_l = abs(sub[sub['fwd_ret20'] <= 0]['fwd_ret20'].mean()) if (sub['fwd_ret20'] <= 0).any() else 1
        plr = avg_w / avg_l if avg_l > 0 else 999
        print(f"  {extra_name:25s} | {len(sub):>7,} | {wr:>5.1f}% | {ar:>+7.2f}% | {plr:>5.2f}")
    
    print("\n" + "=" * 70)
    print("回测完成")


if __name__ == "__main__":
    run()
