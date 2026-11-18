#!/usr/bin/env python3
"""V3维度拆分 + 多策略场景IC回测。

目标：
1. V3每个维度（位置、趋势f'、曲率f''、量能、分钟信号）独立测IC
2. 按5个策略场景分别测IC：底部反转、主升浪、稳定盈利、减速见顶、全市场
3. 按4个市场状态测IC
4. 确定每个场景该用哪些维度的组合

股票池：1800只
采样：每日一个点
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

DATA_ROOT = Path("data/store")
STOCK_POOL = Path("data/stock_pool.csv")
SG_WINDOW = 21
SG_POLY = 3


def load_stock_pool() -> list[str]:
    df = pd.read_csv(STOCK_POOL, dtype={'code': str})
    return df['code'].astype(str).str.zfill(6).tolist()


def load_kline(code: str) -> pd.DataFrame | None:
    path = DATA_ROOT / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    return df


def compute_ic(factor: np.ndarray, ret: np.ndarray) -> float:
    valid = ~(np.isnan(factor) | np.isnan(ret))
    if valid.sum() < 30:
        return np.nan
    ic, _ = spearmanr(factor[valid], ret[valid])
    return ic


def group_winrate(factor: np.ndarray, ret: np.ndarray, n_groups: int = 5) -> list[dict]:
    """分组胜率和收益。"""
    valid = ~(np.isnan(factor) | np.isnan(ret))
    if valid.sum() < 50:
        return []
    f, r = factor[valid], ret[valid]
    try:
        groups = pd.qcut(f, n_groups, labels=False, duplicates='drop')
    except ValueError:
        return []
    results = []
    for g in range(int(groups.max()) + 1):
        mask = groups == g
        if mask.sum() < 10:
            continue
        results.append({
            'n': int(mask.sum()),
            'win_rate': float((r[mask] > 0).mean() * 100),
            'avg_ret': float(r[mask].mean()),
        })
    return results


def run():
    print("=" * 70)
    print("V3维度拆分 × 多策略场景 IC回测")
    print("=" * 70)
    
    codes = load_stock_pool()
    print(f"股票池: {len(codes)} 只\n")
    
    # ── 采集 ──
    print("[1/4] 采集数据...")
    records = []
    for i, code in enumerate(codes):
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(codes)}")
        df = load_kline(code)
        if df is None or len(df) < 250:
            continue
        close = df['close'].values.astype(float)
        vc = 'vol' if 'vol' in df.columns else 'volume'
        if vc not in df.columns:
            continue
        volume = df[vc].values.astype(float)
        n = len(close)
        if n < 280:
            continue
        
        # S-G导数
        ln_p = np.log(close)
        f1 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=1) * 100
        f2 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=2) * 10000
        
        # 量能导数
        ln_vol = np.log(volume + 1)
        vf1 = savgol_filter(ln_vol, SG_WINDOW, SG_POLY, deriv=1) * 100
        
        for t in range(250, n - 20):
            if t + 20 >= n:
                break
            fwd_ret = (close[t+20] / close[t] - 1) * 100
            
            c = close[:t+1]
            v = volume[:t+1]
            
            # 位置距离
            dh60 = (c[-1] / np.max(c[-60:]) - 1) * 100 if t >= 60 else 0
            dh180 = (c[-1] / np.max(c[-180:]) - 1) * 100 if t >= 180 else 0
            dh360 = (c[-1] / np.max(c[-360:]) - 1) * 100 if t >= 360 else 0
            
            # 量比
            vol5 = np.mean(v[-5:])
            vol20 = np.mean(v[-20:]) if len(v) >= 20 else vol5
            vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0
            
            # 20日涨幅
            ret20 = (c[-1] / c[-21] - 1) * 100 if len(c) >= 21 else 0
            
            # 波动率(20日)
            if len(c) >= 22:
                dr = np.diff(c[-22:]) / c[-22:-1]
                volatility = np.std(dr[-20:]) * np.sqrt(252) * 100
            else:
                volatility = 40.0
            
            # MA20
            ma20 = np.mean(c[-20:])
            above_ma20 = c[-1] > ma20
            
            records.append({
                'code': code,
                'f1': f1[t],
                'f2': f2[t],
                'vf1': vf1[t],
                'dh60': dh60,
                'dh180': dh180,
                'dh360': dh360,
                'vol_ratio': vol_ratio,
                'ret20': ret20,
                'volatility': volatility,
                'above_ma20': above_ma20,
                'fwd_ret20': fwd_ret,
            })
    
    df = pd.DataFrame(records)
    print(f"  总样本: {len(df):,}\n")
    
    # ── 因子列表 ──
    factors = {
        'f1': "一阶导f'(趋势方向)",
        'f2': "二阶导f''(加速度)",
        'vf1': "量能一阶导",
        'dh60': "距60日高点",
        'dh180': "距180日高点",
        'dh360': "距360日高点",
        'vol_ratio': "量比(5/20)",
        'ret20': "20日涨幅",
        'volatility': "20日波动率",
    }
    
    # ── 场景定义 ──
    scenarios = {
        '全市场': df,
        '底部拐点': df[(df['f2'] > 0.01) & (df['f1'] < 0) & (df['dh180'] < -15)],
        '底部反转L2': df[(df['f2'] > 0.01) & (df['f1'] >= 0) & (df['dh180'] < -20)],
        '底部反转L3': df[(df['f2'] > 0.01) & (df['f1'] >= 0) & (df['dh180'] < -25) & (df['vol_ratio'] > 1.0)],
        '主升浪': df[(df['f1'] > 0) & (df['f2'] > 0.01) & (df['above_ma20'])],
        '主升浪放量': df[(df['f1'] > 0) & (df['f2'] > 0.01) & (df['vol_ratio'] > 1.0) & (df['above_ma20'])],
        '减速见顶': df[(df['f1'] > 0) & (df['f2'] < -0.01) & (df['ret20'] > 5)],
        '加速下跌': df[(df['f1'] < -0.1) & (df['f2'] < -0.01)],
        '低位震荡': df[(abs(df['f1']) < 0.1) & (abs(df['f2']) < 0.01) & (df['dh180'] < -20)],
    }
    
    # ── [2/4] 各场景IC矩阵 ──
    print("[2/4] 各场景 × 各因子 IC矩阵")
    print("=" * 110)
    
    # 打印表头
    header = f"{'场景':14s} | {'样本':>7s}"
    for col in factors:
        header += f" | {col:>12s}"
    header += f" | {'胜率':>6s} | {'收益':>7s}"
    print(header)
    print("-" * len(header))
    
    ic_matrix = {}
    for name, sub in scenarios.items():
        if len(sub) < 100:
            print(f"  {name:14s} | {len(sub):>7,} (样本不足)")
            continue
        row = {}
        line = f"  {name:12s} | {len(sub):>7,}"
        for col, label in factors.items():
            ic = compute_ic(sub[col].values, sub['fwd_ret20'].values)
            row[col] = ic
            ic_str = f"{ic:+.3f}" if not np.isnan(ic) else "  N/A"
            line += f" | {ic_str:>12s}"
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        line += f" | {wr:>5.1f}% | {ar:>+6.2f}%"
        print(line)
        ic_matrix[name] = row
    
    # ── [3/4] 各因子分组胜率 ──
    print(f"\n[3/4] 关键因子分组胜率（全市场）")
    print("=" * 70)
    
    key_factors = ['f2', 'vol_ratio', 'dh180', 'ret20']
    for col in key_factors:
        label = factors[col]
        grps = group_winrate(df[col].values, df['fwd_ret20'].values, 5)
        if not grps:
            continue
        print(f"\n  {label} (5分位)")
        print(f"  {'分组':8s} {'样本':>8s} {'胜率':>8s} {'平均收益':>10s}")
        for gi, g in enumerate(grps):
            tag = "最低" if gi == 0 else ("最高" if gi == len(grps)-1 else f"Q{gi+1}")
            print(f"  {tag:8s} {g['n']:>8,} {g['win_rate']:>7.1f}% {g['avg_ret']:>+9.2f}%")
    
    # ── [4/4] 维度组合对比 ──
    print(f"\n[4/4] 底部反转：不同维度组合策略对比")
    print("=" * 80)
    
    # 底部反转基础条件
    base = (df['f2'] > 0.01) & (df['dh180'] < -20)
    
    combos = [
        ("基础(f''>0 + dh180<-20)", base),
        ("+f'>=0", base & (df['f1'] >= 0)),
        ("+放量(vol_ratio>1)", base & (df['vol_ratio'] > 1.0)),
        ("+f'>=0 +放量", base & (df['f1'] >= 0) & (df['vol_ratio'] > 1.0)),
        ("+低位(dh180<-30)", base & (df['dh180'] < -30)),
        ("+f'>=0 +低位", base & (df['f1'] >= 0) & (df['dh180'] < -30)),
        ("+f'>=0 +放量 +低位", base & (df['f1'] >= 0) & (df['vol_ratio'] > 1.0) & (df['dh180'] < -30)),
        ("+高波动(vol>40)", base & (df['volatility'] > 40)),
        ("+f'>=0 +放量 +高波动", base & (df['f1'] >= 0) & (df['vol_ratio'] > 1.0) & (df['volatility'] > 40)),
    ]
    
    print(f"\n  {'组合':35s} | {'样本':>7s} | {'胜率':>6s} | {'收益':>8s} | {'盈亏比':>6s}")
    print("  " + "-" * 75)
    
    for name, mask in combos:
        sub = df[mask]
        if len(sub) < 50:
            print(f"  {name:35s} | {len(sub):>7,} (不足)")
            continue
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        avg_win = sub[sub['fwd_ret20'] > 0]['fwd_ret20'].mean() if (sub['fwd_ret20'] > 0).any() else 0
        avg_loss = abs(sub[sub['fwd_ret20'] <= 0]['fwd_ret20'].mean()) if (sub['fwd_ret20'] <= 0).any() else 1
        plr = avg_win / avg_loss if avg_loss > 0 else 999
        print(f"  {name:35s} | {len(sub):>7,} | {wr:>5.1f}% | {ar:>+7.2f}% | {plr:>5.2f}")
    
    # ── 主升浪组合对比 ──
    print(f"\n  主升浪：不同维度组合策略对比")
    print("=" * 80)
    
    base2 = (df['f1'] > 0) & (df['f2'] > 0.01) & (df['above_ma20'])
    
    combos2 = [
        ("基础(f'>0+f''>0+>MA20)", base2),
        ("+放量(vol_ratio>1)", base2 & (df['vol_ratio'] > 1.0)),
        ("+缩量(vol_ratio<1)", base2 & (df['vol_ratio'] < 1.0)),
        ("+低位(dh180<-10)", base2 & (df['dh180'] < -10)),
        ("+放量+低位", base2 & (df['vol_ratio'] > 1.0) & (df['dh180'] < -10)),
        ("+高位(dh180>10)", base2 & (df['dh180'] > 10)),
        ("+放量+高位", base2 & (df['vol_ratio'] > 1.0) & (df['dh180'] > 10)),
        ("+高波动(vol>50)", base2 & (df['volatility'] > 50)),
        ("+低波动(vol<30)", base2 & (df['volatility'] < 30)),
    ]
    
    print(f"\n  {'组合':35s} | {'样本':>7s} | {'胜率':>6s} | {'收益':>8s} | {'盈亏比':>6s}")
    print("  " + "-" * 75)
    
    for name, mask in combos2:
        sub = df[mask]
        if len(sub) < 50:
            print(f"  {name:35s} | {len(sub):>7,} (不足)")
            continue
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        avg_win = sub[sub['fwd_ret20'] > 0]['fwd_ret20'].mean() if (sub['fwd_ret20'] > 0).any() else 0
        avg_loss = abs(sub[sub['fwd_ret20'] <= 0]['fwd_ret20'].mean()) if (sub['fwd_ret20'] <= 0).any() else 1
        plr = avg_win / avg_loss if avg_loss > 0 else 999
        print(f"  {name:35s} | {len(sub):>7,} | {wr:>5.1f}% | {ar:>+7.2f}% | {plr:>5.2f}")
    
    print("\n" + "=" * 70)
    print("回测完成")
    print("=" * 70)


if __name__ == "__main__":
    run()
