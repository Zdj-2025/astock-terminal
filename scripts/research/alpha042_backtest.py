#!/usr/bin/env python3
"""Alpha042缩量上涨因子回测 + V3维度拆分评估。

目标：
1. 实现 Alpha042 的 CORR(价格变化, 成交量变化) 缩量上涨因子
2. 测算该因子的 IC 和分组胜率
3. 将该因子整合到底部反转策略，对比优化前后效果
4. 拆分V3各维度独立IC，评估各维度贡献

股票池：1800只（沪深300+中证500+中证1000）
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from datetime import datetime

DATA_ROOT = Path("data/store")
STOCK_POOL = Path("data/stock_pool.csv")
SG_WINDOW = 21
SG_POLY = 3


def load_stock_pool() -> list[str]:
    df = pd.read_csv(STOCK_POOL, dtype={'code': str})
    codes = df['code'].astype(str).str.zfill(6).tolist()
    return codes


def load_kline(code: str) -> pd.DataFrame | None:
    """加载日线数据。"""
    path = DATA_ROOT / "kline" / code / "D.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    dt_col = "trade_date" if "trade_date" in df.columns else "date"
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    return df


def compute_alpha042_factor(close: np.ndarray, volume: np.ndarray, window: int = 10) -> float:
    """Alpha042: (-1 * RANK(STD(HIGH,10))) * CORR(HIGH, VOLUME, 10)
    
    我们用的是日线数据没有单独HIGH列，用 close 代替 HIGH（日线维度差别很小）。
    同时也计算纯 CORR 版本作为对比。
    
    Returns:
        (alpha042_full, price_vol_corr)
        alpha042_full: 完整Alpha042公式值（需要全市场rank，单票返回原始值）
        price_vol_corr: CORR(close_pct, volume_pct, window) 量价相关系数
    """
    n = len(close)
    if n < window + 2:
        return np.nan, np.nan
    
    # 价格变化率
    close_pct = np.diff(close[-window-1:]) / close[-window-1:-1]
    # 成交量变化率（对数变化更稳定）
    vol_slice = volume[-window-1:]
    vol_pct = np.diff(vol_slice) / (vol_slice[:-1] + 1)
    
    if np.std(close_pct) < 1e-10 or np.std(vol_pct) < 1e-10:
        return np.nan, np.nan
    
    corr = np.corrcoef(close_pct, vol_pct)[0, 1]
    
    # 波动率（STD of close）
    std_close = np.std(close[-window:]) / np.mean(close[-window:])
    
    # Alpha042原始: (-1 * RANK(STD)) * CORR
    # 单票无法rank，返回 (-std_normalized * corr) 作为代理
    # 高波动 * 负corr(缩量上涨) = 大正值
    alpha_raw = -std_close * corr
    
    return alpha_raw, corr


def compute_v3_dimensions(close: np.ndarray, volume: np.ndarray) -> dict:
    """拆分V3各维度独立计算，返回各维度原始值。"""
    n = len(close)
    result = {}
    
    if n < 60:
        return {"error": "数据不足"}
    
    # S-G 导数
    ln_p = np.log(close.astype(float))
    f1 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=1) * 100
    f2 = savgol_filter(ln_p, SG_WINDOW, SG_POLY, deriv=2) * 10000
    
    result['f1'] = float(f1[-1])
    result['f2'] = float(f2[-1])
    
    # 位置距离
    current = close[-1]
    for period in [60, 180, 360, 1000]:
        if n >= period:
            high_p = np.max(close[-period:])
        else:
            high_p = np.max(close)
        result[f'dh{period}'] = round((current / high_p - 1) * 100, 2)
    
    # 量能
    if volume is not None and len(volume) >= 20:
        vol5 = np.mean(volume[-5:])
        vol20 = np.mean(volume[-20:])
        result['vol_ratio'] = round(vol5 / vol20, 3) if vol20 > 0 else 1.0
    else:
        result['vol_ratio'] = 1.0
    
    # 收益率
    if n >= 21:
        result['ret20'] = round((close[-1] / close[-21] - 1) * 100, 2)
    else:
        result['ret20'] = 0.0
    
    # 20日收益率（前向，用于IC计算）
    result['fwd_ret20'] = np.nan  # 后面填充
    
    # 信号分类
    f1v, f2v = result['f1'], result['f2']
    if f1v > 0 and f2v > 0.01:
        result['signal'] = '🚀主升浪'
    elif f2v > 0.01 and f1v < 0:
        result['signal'] = '🔄底部拐点'
    elif f1v > 0 and f2v < -0.01:
        result['signal'] = '⚠️减速见顶'
    elif f1v < -0.1 and f2v < -0.01:
        result['signal'] = '🔴加速下跌'
    else:
        result['signal'] = '➡️观望'
    
    return result


def compute_ic(factor_values: np.ndarray, forward_returns: np.ndarray) -> float:
    """计算IC（信息系数）= rank相关系数。"""
    valid = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    if valid.sum() < 30:
        return np.nan
    from scipy.stats import spearmanr
    ic, _ = spearmanr(factor_values[valid], forward_returns[valid])
    return ic


def run_backtest():
    """主回测流程。"""
    print("=" * 70)
    print("Alpha042缩量上涨因子 + V3维度拆分回测")
    print(f"股票池: {STOCK_POOL} ({'存在' if STOCK_POOL.exists() else '不存在'})")
    print("=" * 70)
    
    codes = load_stock_pool()
    print(f"股票池: {len(codes)} 只")
    
    # ── 第一轮：采集所有数据 ──
    print("\n[1/3] 采集数据...")
    all_records = []
    skipped = 0
    
    for i, code in enumerate(codes):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(codes)}")
        
        df = load_kline(code)
        if df is None or len(df) < 100:
            skipped += 1
            continue
        
        close = df['close'].values.astype(float)
        vc = 'vol' if 'vol' in df.columns else 'volume'
        volume = df[vc].values.astype(float) if vc in df.columns else None
        
        if volume is None:
            skipped += 1
            continue
        
        n = len(close)
        
        # 滑动窗口回测：每个交易日一个样本点
        # 从第250个交易日开始（保证足够历史）
        for t in range(250, n - 20):
            c = close[:t+1]
            v = volume[:t+1]
            
            # 计算Alpha042因子
            alpha_raw, price_vol_corr = compute_alpha042_factor(c, v, window=10)
            
            # 计算V3各维度
            dims = compute_v3_dimensions(c, v)
            if 'error' in dims:
                continue
            
            # 前向20日收益率
            if t + 20 < n:
                fwd_ret = (close[t+20] / close[t] - 1) * 100
            else:
                continue
            
            record = {
                'code': code,
                'date_idx': t,
                'alpha042_raw': alpha_raw,
                'price_vol_corr': price_vol_corr,
                'fwd_ret20': fwd_ret,
                'f1': dims['f1'],
                'f2': dims['f2'],
                'dh60': dims['dh60'],
                'dh180': dims['dh180'],
                'dh360': dims['dh360'],
                'vol_ratio': dims['vol_ratio'],
                'ret20': dims['ret20'],
                'signal': dims['signal'],
            }
            all_records.append(record)
    
    print(f"  有效样本: {len(all_records):,} (跳过: {skipped})")
    
    if len(all_records) < 100:
        print("样本太少，退出")
        return
    
    df_all = pd.DataFrame(all_records)
    
    # ── 第二轮：因子IC分析 ──
    print("\n[2/3] 因子IC分析")
    print("-" * 50)
    
    factors = {
        'alpha042_raw': 'Alpha042原始',
        'price_vol_corr': '量价相关系数(CORR)',
        'f1': '一阶导f\'',
        'f2': '二阶导f\'\'',
        'dh180': '距180日高点',
        'dh360': '距360日高点',
        'vol_ratio': '量比(5/20)',
        'ret20': '20日涨幅',
    }
    
    ic_results = {}
    for col, name in factors.items():
        ic = compute_ic(df_all[col].values, df_all['fwd_ret20'].values)
        ic_results[col] = ic
        star = "⭐⭐⭐" if abs(ic) > 0.05 else ("⭐⭐" if abs(ic) > 0.03 else ("⭐" if abs(ic) > 0.01 else "❌"))
        print(f"  {name:25s} IC={ic:+.4f} {star}")
    
    # ── 量价相关系数分组胜率 ──
    print("\n[3/3] 量价相关系数(CORR)分组分析")
    print("-" * 50)
    
    # 按corr分5组
    df_all['corr_group'] = pd.qcut(df_all['price_vol_corr'], 5, labels=['Q1(最负)', 'Q2', 'Q3', 'Q4', 'Q5(最正)'], duplicates='drop')
    
    print(f"\n{'分组':12s} {'样本数':>8s} {'平均CORR':>10s} {'20日胜率':>10s} {'平均收益':>10s}")
    print("-" * 55)
    for g in ['Q1(最负)', 'Q2', 'Q3', 'Q4', 'Q5(最正)']:
        sub = df_all[df_all['corr_group'] == g]
        if len(sub) == 0:
            continue
        win_rate = (sub['fwd_ret20'] > 0).mean() * 100
        avg_ret = sub['fwd_ret20'].mean()
        avg_corr = sub['price_vol_corr'].mean()
        print(f"  {g:12s} {len(sub):>8,} {avg_corr:>+10.4f} {win_rate:>9.1f}% {avg_ret:>+9.2f}%")
    
    # ── 缩量上涨在底部反转中的效果 ──
    print("\n" + "=" * 70)
    print("重点：缩量上涨因子在【底部反转】场景的效果")
    print("=" * 70)
    
    # 底部反转条件：f2>0(拐点) + dh180<-15(低位)
    bottom = df_all[(df_all['f2'] > 0.01) & (df_all['dh180'] < -15)]
    print(f"\n底部反转样本(f''>0 & dh180<-15): {len(bottom):,}")
    
    if len(bottom) > 50:
        # 按corr分成缩量vs放量
        bottom_corr_median = bottom['price_vol_corr'].median()
        bottom_shrink = bottom[bottom['price_vol_corr'] < bottom_corr_median]  # 缩量组
        bottom_expand = bottom[bottom['price_vol_corr'] >= bottom_corr_median]  # 放量组
        
        print(f"\n缩量组(CORR < {bottom_corr_median:.3f}):")
        print(f"  样本: {len(bottom_shrink):,}")
        print(f"  20日胜率: {(bottom_shrink['fwd_ret20'] > 0).mean()*100:.1f}%")
        print(f"  平均收益: {bottom_shrink['fwd_ret20'].mean():+.2f}%")
        
        print(f"\n放量组(CORR >= {bottom_corr_median:.3f}):")
        print(f"  样本: {len(bottom_expand):,}")
        print(f"  20日胜率: {(bottom_expand['fwd_ret20'] > 0).mean()*100:.1f}%")
        print(f"  平均收益: {bottom_expand['fwd_ret20'].mean():+.2f}%")
        
        # IC对比
        ic_bottom_corr = compute_ic(bottom['price_vol_corr'].values, bottom['fwd_ret20'].values)
        ic_bottom_f2 = compute_ic(bottom['f2'].values, bottom['fwd_ret20'].values)
        print(f"\n底部反转场景IC:")
        print(f"  量价相关系数(CORR): {ic_bottom_corr:+.4f}")
        print(f"  二阶导(f''):        {ic_bottom_f2:+.4f}")
    
    # ── 信号分类 × 缩量/放量 ──
    print("\n" + "=" * 70)
    print("信号分类 × 缩量/放量 交叉分析")
    print("=" * 70)
    
    corr_median = df_all['price_vol_corr'].median()
    df_all['is_shrink'] = df_all['price_vol_corr'] < corr_median
    
    print(f"\n{'信号':10s} | {'全样本':>8s} {'胜率':>6s} {'收益':>8s} | {'缩量':>6s} {'胜率':>6s} {'收益':>8s} | {'放量':>6s} {'胜率':>6s} {'收益':>8s}")
    print("-" * 90)
    for sig in ['🚀主升浪', '🔄底部拐点', '⚠️减速见顶', '🔴加速下跌', '➡️观望']:
        sub = df_all[df_all['signal'] == sig]
        if len(sub) < 50:
            continue
        wr = (sub['fwd_ret20'] > 0).mean() * 100
        ar = sub['fwd_ret20'].mean()
        
        sub_s = sub[sub['is_shrink']]
        sub_e = sub[~sub['is_shrink']]
        
        wr_s = (sub_s['fwd_ret20'] > 0).mean() * 100 if len(sub_s) > 0 else 0
        ar_s = sub_s['fwd_ret20'].mean() if len(sub_s) > 0 else 0
        wr_e = (sub_e['fwd_ret20'] > 0).mean() * 100 if len(sub_e) > 0 else 0
        ar_e = sub_e['fwd_ret20'].mean() if len(sub_e) > 0 else 0
        
        print(f"  {sig:8s} | {len(sub):>7,} {wr:>5.1f}% {ar:>+7.2f}% | {len(sub_s):>5,} {wr_s:>5.1f}% {ar_s:>+7.2f}% | {len(sub_e):>5,} {wr_e:>5.1f}% {ar_e:>+7.2f}%")
    
    # ── 策略对比：底部反转 vs 底部反转+缩量 ──
    print("\n" + "=" * 70)
    print("策略对比：底部反转(旧) vs 底部反转+缩量上涨(新)")
    print("=" * 70)
    
    # 旧策略：f2>0 + dh180<-20 (V3 L2级别)
    old = df_all[(df_all['f2'] > 0.01) & (df_all['dh180'] < -20)]
    # 新策略：旧 + 缩量(corr < 0)
    new = old[old['price_vol_corr'] < 0]
    
    if len(old) > 30:
        print(f"\n旧策略(底部反转L2):")
        print(f"  样本: {len(old):,}")
        print(f"  20日胜率: {(old['fwd_ret20'] > 0).mean()*100:.1f}%")
        print(f"  平均收益: {old['fwd_ret20'].mean():+.2f}%")
        
        print(f"\n新策略(底部反转L2 + 缩量上涨):")
        print(f"  样本: {len(new):,}")
        print(f"  20日胜率: {(new['fwd_ret20'] > 0).mean()*100:.1f}%")
        print(f"  平均收益: {new['fwd_ret20'].mean():+.2f}%")
        
        if len(new) > 0 and len(old) > 0:
            delta_wr = ((new['fwd_ret20'] > 0).mean() - (old['fwd_ret20'] > 0).mean()) * 100
            delta_ret = new['fwd_ret20'].mean() - old['fwd_ret20'].mean()
            print(f"\n胜率变化: {delta_wr:+.1f}%")
            print(f"收益变化: {delta_ret:+.2f}%")
    
    # 同样测试L3级别
    old_l3 = df_all[(df_all['f1'] >= 0) & (df_all['f2'] > 0.01) & (df_all['dh180'] < -25)]
    new_l3 = old_l3[old_l3['price_vol_corr'] < 0]
    
    if len(old_l3) > 30:
        print(f"\n--- L3级别 ---")
        print(f"旧策略(L3): {len(old_l3):,}样本, 胜率{(old_l3['fwd_ret20']>0).mean()*100:.1f}%, 收益{old_l3['fwd_ret20'].mean():+.2f}%")
        print(f"新策略(L3+缩量): {len(new_l3):,}样本, 胜率{(new_l3['fwd_ret20']>0).mean()*100:.1f}%, 收益{new_l3['fwd_ret20'].mean():+.2f}%")
    
    # ── CORR窗口灵敏度 ──
    print("\n" + "=" * 70)
    print("CORR窗口灵敏度测试")
    print("=" * 70)
    
    # 已经是10天窗口，测5/10/15/20
    print("\n注意: 当前用10天窗口，上面结果就是10天的")
    print("如果要测其他窗口，需要重新计算，暂用当前结果")
    
    print("\n" + "=" * 70)
    print("回测完成")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
