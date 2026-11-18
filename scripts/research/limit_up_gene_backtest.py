#!/usr/bin/env python3
"""连板基因回测 — 找出预测连续涨停的基因组合。
阶段1: 涨停识别 + 基准连板率(本文件)
阶段2: 5大基因特征(预增/主线/超跌/小盘/聪明钱) + 显著性统计
阶段3: 基于验证基因建连板潜力scanner

涨停识别: 用change_pct(主板9.8-10.02%/创业科创19.5-20.05%),排除ST/新股(<60日)/北交所/除权日。
连板: 首板(limit且前日不limit)后续连续limit天数。
作者 2026-08-07"""
import pandas as pd, numpy as np, glob, os
os.chdir('.')

print("=== 连板回测 阶段1: 涨停识别 + 基准连板率 ===\n")

# 1. 过滤ST/北交所
sl = pd.read_parquet('data/store/stock_list.parquet')
st_codes = set(sl.loc[sl['name'].astype(str).str.contains('ST'), 'symbol'])
ind_map = dict(zip(sl['symbol'], sl['industry']))


def is_limit(code, pct):
    """涨停判断: 创业板/科创板20%, 主板10%."""
    if code.startswith('30') or code.startswith('688'):
        return (pct >= 19.5) & (pct <= 20.05)
    return (pct >= 9.8) & (pct <= 10.02)


first_boards = []
n_stocks = 0
for code_dir in glob.glob('data/store/kline/*'):
    code = os.path.basename(code_dir)
    if code in st_codes or code[0] not in ('0', '3', '6'):
        continue
    try:
        df = pd.read_parquet(f'{code_dir}/D.parquet').sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(df) < 120 or 'change_pct' not in df.columns:
        continue
    n_stocks += 1
    df['limit'] = is_limit(code, df['change_pct'].values)
    # 连续涨停段: 段内 streak_len = 该段涨停总数(连板长度)
    df['grp'] = (~df['limit']).cumsum()
    df['streak_len'] = df.groupby('grp')['limit'].transform('sum').astype(int)
    df['prev_limit'] = df['limit'].shift(1).fillna(False)
    df['first'] = df['limit'] & ~df['prev_limit']
    fb_df = df[df['first']]
    for idx, r in fb_df.iterrows():
        if idx < 60:  # 新股过滤(距上市<60日)
            continue
        if r['streak_len'] > 15:  # 异常(新股连板/数据错)跳过
            continue
        first_boards.append((code, r['date'], int(r['streak_len']), ind_map.get(code, '')))

fb = pd.DataFrame(first_boards, columns=['code', 'date', 'streak', 'industry'])
fb['date'] = pd.to_datetime(fb['date'])

print(f"扫描股票: {n_stocks} | 首板样本: {len(fb)}")
print(f"时间范围: {fb['date'].min().date()} ~ {fb['date'].max().date()}")
print(f"\n=== 基准连板率(全样本) ===")
n = len(fb)
for k in [2, 3, 4, 5]:
    cnt = (fb['streak'] >= k).sum()
    print(f"  {k}板+(首板后连{k-1}日): {cnt:>5} = {cnt/n*100:5.1f}%")
print(f"\n  连板长度分布: {dict(fb['streak'].value_counts().sort_index().head(7))}")
fb.to_parquet('data/store/limit_up_first_boards.parquet', index=False)
print(f"\n存档: data/store/limit_up_first_boards.parquet ({len(fb)}条)")
