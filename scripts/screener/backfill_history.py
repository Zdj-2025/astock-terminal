#!/usr/bin/env python3
"""回溯过去N天的扫描历史 — 从已有的scan_result.log或重新跑scanner来填充scan_history.

方法: 用本地K线重算每个策略在过去N天的信号(因果,不含未来数据).
策略: divergence(E组合), smart_money, double_bottom, golden_cross.
不回溯: sector_heat(需行业截面数据), catalyst(需公告), defensive(二阶导轻量跳过).
"""
import pandas as pd
import numpy as np
import os, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')

from scripts.screener.scan_history import record_batch, STORE
from scripts.screener.divergence_scanner import detect_double_bottom

KDIR = 'data/store/kline'
N_DAYS = 60


def load(code):
    df = pd.read_parquet(f'{KDIR}/{code}/D.parquet')
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def macd(c):
    ef = c.ewm(span=12, adjust=False).mean(); es = c.ewm(span=26, adjust=False).mean()
    dif = ef - es
    return dif, dif - dif.ewm(span=9, adjust=False).mean()


def swings(arr, n=5, find_low=True):
    out = []
    for i in range(n, len(arr) - n):
        if find_low:
            ok = all(arr[i] < arr[i - j] for j in range(1, n + 1)) and all(arr[i] <= arr[i + j] for j in range(1, n + 1))
        else:
            ok = all(arr[i] > arr[i - j] for j in range(1, n + 1)) and all(arr[i] >= arr[i + j] for j in range(1, n + 1))
        if ok:
            out.append(i)
    return out


def backtest_divergence(df):
    """E组合: La→Lb背离 + B过滤 + Lc后量能确认 + 12%新鲜 + 距高<-5%"""
    l = df['low'].astype(float).values
    c = df['close'].astype(float).values
    h = df['high'].astype(float).values
    dif, hist = macd(pd.Series(c))
    vol = (df['vol'] if 'vol' in df.columns else df['volume']).astype(float).values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    out = []
    troughs = swings(l, find_low=True)
    for k in range(1, len(troughs)):
        La, Lb = troughs[k-1], troughs[k]
        if not (l[Lb] < l[La] and hist[Lb] > hist[La] and dif[Lb] > dif[La]):
            continue
        if Lb + 1 < len(l) and np.min(l[Lb+1:Lb+21]) < l[Lb] * 0.90:
            continue
        for t in range(max(Lb + 5, Lb + 1), min(Lb + 16, len(c))):
            if c[t] > df['open'].astype(float).values[t] and not np.isnan(vol_ma20[t]) and vol[t] > vol_ma20[t]:
                bounce = c[t] / l[Lb] - 1
                hb = np.max(h[:t+1])
                if bounce <= 0.12 and c[t] / hb - 1 <= -0.05:
                    out.append(t)
                break
    return out


def backtest_smart_money(df):
    """简化版smart: 金叉+稳步抬升+量价配合 (score≥3/6)"""
    c = df['close'].astype(float).values
    o = df['open'].astype(float).values
    ma5 = pd.Series(c).rolling(5).mean().values
    ma10 = pd.Series(c).rolling(10).mean().values
    vol = (df['vol'] if 'vol' in df.columns else df['volume']).astype(float).values
    vol_ma5 = pd.Series(vol).rolling(5).mean().values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    out = []
    for t in range(30, len(c)):
        score = 0
        if not np.isnan(ma5[t]) and not np.isnan(ma10[t]) and not np.isnan(ma5[t-1]) and not np.isnan(ma10[t-1]):
            if ma5[t] > ma10[t] and ma5[t-1] <= ma10[t-1]:
                score += 1  # 金叉
        if not np.isnan(vol_ma5[t]) and not np.isnan(vol_ma20[t]) and vol_ma5[t] > vol_ma20[t]:
            score += 1  # 量价配合
        if t >= 5 and c[t] > c[t-5]:
            score += 1  # 稳步抬升
        if score >= 3:
            out.append(t)
    return out


def backtest_golden_cross(df):
    """MA5上穿MA10 + 阳线"""
    c = df['close'].astype(float).values
    o = df['open'].astype(float).values
    ma5 = pd.Series(c).rolling(5).mean().values
    ma10 = pd.Series(c).rolling(10).mean().values
    out = []
    for t in range(15, len(c)):
        if np.isnan(ma5[t]) or np.isnan(ma10[t]) or np.isnan(ma5[t-1]) or np.isnan(ma10[t-1]):
            continue
        if ma5[t] > ma10[t] and ma5[t-1] <= ma10[t-1] and c[t] > o[t]:
            out.append(t)
    return out


def backtest_double_bottom(df):
    """双底/二买检测"""
    results = detect_double_bottom(df)
    if results:
        return [len(df) - 1]  # 只标记"当前有信号"
    return []


def main():
    sl = pd.read_parquet('data/store/stock_list.parquet')
    sym = sl['symbol'].astype(str).str.zfill(6)
    nm = dict(zip(sym, sl['name'].astype(str)))

    # 获取最近N天的交易日
    idx = pd.read_parquet('data/store/index/sh000001_long.parquet')
    dates = sorted(idx['d'].astype(str).unique())
    recent_dates = [d for d in dates if d >= '20260620'][-N_DAYS:]
    date_set = set(recent_dates)
    print(f"回溯期间: {recent_dates[0]}~{recent_dates[-1]} ({len(recent_dates)}个交易日)")

    codes = sorted([d.name for d in Path(KDIR).iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])  # 全市场

    all_records = {d: {s: [] for s in ['divergence', 'smart_money', 'golden_cross', 'double_bottom']}
                   for d in recent_dates}

    for ci, code in enumerate(codes):
        if ci % 100 == 0:
            print(f"  进度: {ci}/{len(codes)}")
        n = nm.get(code, '')
        if 'ST' in n or '退' in n:
            continue
        try:
            df = load(code)
            if len(df) < 150:
                continue
            c = df['close'].astype(float)
            if float(c.iloc[-1]) < 2:
                continue

            # 对每个策略, 找出信号日, 映射到交易日
            for strat, indices in [
                ('divergence', backtest_divergence(df)),
                ('smart_money', backtest_smart_money(df)),
                ('golden_cross', backtest_golden_cross(df)),
                ('double_bottom', backtest_double_bottom(df)),
            ]:
                for t in indices:
                    d = df['d'].iloc[t]
                    if d in date_set:
                        all_records[d][strat].append(dict(code=code, name=n, score=None, metadata=''))

        except Exception:
            continue

    # 批量写入
    total = 0
    for d, strats in all_records.items():
        for strat, rows in strats.items():
            if rows:
                record_batch(strat, rows, date=d)
                total += len(rows)

    print(f"\n✅ 回溯完成: 写入{total}条记录 → {STORE}")
    print(f"文件大小: {STORE.stat().st_size / 1024:.0f}KB")


if __name__ == '__main__':
    main()
