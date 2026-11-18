#!/usr/bin/env python3
"""信号累积器回测 — 正确方法论版(2026-09-08).

用户三个核心要求:
1. 模拟连续跑多天(不是只看最新一天)
2. 严格无未来数据(每天T只用T及以前的信息)
3. 横向权重>纵向(独立事件概率相乘 vs 非独立重复确认)

方法:
- 对回测期内每一天T:
  读取T-19到T的扫描记录(20日窗口,无未来)
  计算每只股票截至T的累积分
  记录T的前瞻收益(5/10/20日)
- 按累积分分桶,检验高分是否预测高前瞻收益
- 这是面板数据的滚动窗口回测,非单截面

权重设计:
  composite = horiz^2 × log2(1 + vert_sum)
  横向:平方(指数效应,每多一个独立策略后验跳升一个量级)
  纵向:对数(同一事件重复确认,边际信息递减)
"""
import pandas as pd
import numpy as np
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')

KDIR = 'data/store/kline'
LOOKBACK = 20       # 纵向窗口(交易日)
HOLDS = [5, 10, 20] # 前瞻窗口
COST = 0.0013


def load_index():
    idx = pd.read_parquet('data/store/index/sh000001_long.parquet')
    ic = idx.set_index('d')['c'].astype(float)
    dates = sorted(ic.index.unique())
    return ic, dates


def load(code):
    df = pd.read_parquet(f'{KDIR}/{code}/D.parquet')
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def compute_scores_asof(hist, date_T, dates_all):
    """计算截至date_T的累积分(严格无未来).
    horiz: date_T当天被几个策略命中
    vert_sum: 各策略在[T-19,T]窗口内的加权天数(对数)
    composite: horiz^2 * log2(1+vert_sum)  ← 横向平方×纵向对数
    """
    # 截取无未来的窗口
    idx_T = dates_all.index(date_T) if date_T in dates_all else -1
    if idx_T < 0:
        return pd.DataFrame()
    window_dates = set(dates_all[max(0, idx_T - LOOKBACK + 1):idx_T + 1])
    h = hist[hist['date'].isin(window_dates)].copy()

    if h.empty:
        return pd.DataFrame()

    # 衰减权重
    ordered = sorted(window_dates, reverse=True)
    decay_map = {d: 1.0 if i < 3 else (0.6 if i < 10 else 0.3) for i, d in enumerate(ordered)}
    h['decay'] = h['date'].map(decay_map)

    # 横向: T当天策略数
    today = h[h['date'] == date_T]
    horiz = today.groupby('code')['strategy'].nunique().rename('horiz').reset_index()

    # 纵向: 各策略加权天数
    vert = h.groupby(['code', 'strategy'])['decay'].sum().reset_index()
    vert_sum = vert.groupby('code')['decay'].sum().rename('vert_sum').reset_index()
    # 纵向策略数
    vert_n = vert.groupby('code')['strategy'].nunique().rename('vert_strats').reset_index()

    # 名称
    names = h.groupby('code')['name'].last().reset_index()

    result = horiz.merge(vert_sum, on='code', how='left') \
                   .merge(vert_n, on='code', how='left') \
                   .merge(names, on='code', how='left')
    result['vert_sum'] = result['vert_sum'].fillna(0)
    result['vert_strats'] = result['vert_strats'].fillna(0)

    # 综合分: 横向^2 × log2(1+纵向)  ← 用户确认:横向更强
    result['composite'] = result['horiz'] ** 2 * np.log2(1 + result['vert_sum'])
    # 旧公式(等权)用于对比
    result['composite_old'] = result['horiz'] * result['vert_sum']

    return result


def main():
    ic, dates_all = load_index()
    hist = pd.read_parquet('data/store/scan_history.parquet')
    hist_dates = sorted(hist['date'].unique())
    print(f"扫描历史: {hist_dates[0]}~{hist_dates[-1]} ({len(hist_dates)}天, {len(hist)}条)")

    # 加载全市场价格(forward returns)
    codes = hist['code'].unique()
    prices = {}
    for code in codes:
        try:
            df = load(code)
            prices[code] = df.set_index('d')['close'].astype(float).to_dict()
        except Exception:
            continue

    # 滚动回测: 对每个有扫描记录的日期T
    all_rows = []
    for date_T in hist_dates:
        if date_T not in dates_all:
            continue
        scores = compute_scores_asof(hist, date_T, dates_all)
        if scores.empty:
            continue

        idx_T = dates_all.index(date_T)
        for _, r in scores.iterrows():
            code = r['code']
            px_map = prices.get(code, {})
            if date_T not in px_map:
                continue
            entry = px_map[date_T]
            row = dict(code=code, name=r['name'], date=date_T,
                       horiz=r['horiz'], vert_sum=r['vert_sum'],
                       composite=r['composite'], composite_old=r['composite_old'])
            for H in HOLDS:
                # forward: T日后第H个交易日
                if idx_T + H < len(dates_all):
                    fd = dates_all[idx_T + H]
                    if fd in px_map:
                        row[f'ret{H}'] = (px_map[fd] / entry - 1 - COST) * 100
                    else:
                        row[f'ret{H}'] = np.nan
                else:
                    row[f'ret{H}'] = np.nan
            all_rows.append(row)

    t = pd.DataFrame(all_rows)
    print(f"\n回测面板: {len(t)}行 | 覆盖{t['date'].nunique()}天×{t['code'].nunique()}只")

    # 按横向数分组
    print('\n═══ 按横向策略数分组(新权重核心) ═══')
    print(f"{'横向':>4} {'N':>6} {'10日均':>8} {'10日中位':>8} {'胜率':>5} {'20日均':>8} {'20日胜率':>6}")
    for k in sorted(t['horiz'].unique(), reverse=True):
        g = t[t['horiz'] == k]
        if len(g) < 5:
            continue
        r10 = g['ret10'].dropna(); r20 = g['ret20'].dropna()
        print(f"  {int(k):>2} {len(g):>6} {r10.mean():>+7.2f}% {r10.median():>+7.2f}% "
              f"{(r10>0).mean()*100:>4.0f}% {r20.mean():>+7.2f}% {(r20>0).mean()*100:>5.0f}%")

    # 按综合分分桶
    print('\n═══ 按综合分分桶(新权重: horiz²×log(1+vert)) ═══')
    bins = [0, 1, 3, 6, 10, 100]
    labels = ['0-1', '1-3', '3-6', '6-10', '10+']
    t['bucket'] = pd.cut(t['composite'], bins=bins, labels=labels, right=False)
    print(f"{'分桶':>6} {'N':>6} {'10日均':>8} {'10日中位':>8} {'胜率':>5} {'20日均':>8} {'20日胜率':>6}")
    for b in labels:
        g = t[t['bucket'] == b]
        if len(g) < 5:
            continue
        r10 = g['ret10'].dropna(); r20 = g['ret20'].dropna()
        print(f"  {b:>4} {len(g):>6} {r10.mean():>+7.2f}% {r10.median():>+7.2f}% "
              f"{(r10>0).mean()*100:>4.0f}% {r20.mean():>+7.2f}% {(r20>0).mean()*100:>5.0f}%")

    # 旧权重对比
    print('\n═══ 旧权重对比(horiz×vert等权) ═══')
    t['bucket_old'] = pd.cut(t['composite_old'], bins=[0, 1, 3, 6, 10, 100],
                              labels=labels, right=False)
    for b in labels:
        g = t[t['bucket_old'] == b]
        if len(g) < 5:
            continue
        r10 = g['ret10'].dropna()
        print(f"  {b:>4} N={len(g):>5} 10日均{r10.mean():>+6.2f}% 胜{(r10>0).mean()*100:>3.0f}%")

    # 新旧权重相关性(如果高度相关则区分度不足)
    valid = t[['composite', 'composite_old']].dropna()
    if len(valid) > 10:
        corr = valid['composite'].corr(valid['composite_old'])
        print(f"\n新旧权重相关系数: {corr:.3f}")

    t.to_parquet('data/store/signal_accumulator_backtest.parquet')
    print(f"\n已存 {len(t)} 行 → data/store/signal_accumulator_backtest.parquet")


if __name__ == '__main__':
    main()
