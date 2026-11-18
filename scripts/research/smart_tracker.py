#!/usr/bin/env python3
"""smart信号持续追踪器 — 每日收盘后运行。

功能：
1. 扫全市场，记录当日 smart_money_score>=4 的股票（含连击天数streak）
2. 自动回填已到期信号的 1/5/10/20 日后续收益
3. 输出滚动统计：成功率、平均回报、超额（vs 信号组同日中位数可后续加）

数据落盘: data/store/smart_signals.parquet
首次运行: 回填最近 seed_days=10 个交易日（约3分钟）
用法: cd stock && PYTHONPATH=. python scripts/research/smart_tracker.py
      加 --seed N 可强制回填最近N个交易日
"""
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from scripts.research.smart_money_backtest import smart_money_score, load

STORE = PROJECT_ROOT / 'data/store/kline'
OUT = PROJECT_ROOT / 'data/store/smart_signals.parquet'
THRESH = 4          # 记录阈值
SEED_DAYS = 10      # 首次运行回填交易日数


def get_trade_dates(n=None):
    """从参考权重股取最近交易日序列"""
    ref = load('600519')
    dates = ref['date'].tolist()
    return dates[-n:] if n else dates


def score_date(df, t):
    return smart_money_score(df, t)[0]


def scan_dates(target_dates):
    """对给定日期列表逐股打分，返回 score>=THRESH 的行。日期统一str(YYYY-MM-DD)比较"""
    tset = {str(pd.Timestamp(d).date()) for d in target_dates}
    rows = []
    for code in sorted(os.listdir(STORE)):
        p = STORE / f'{code}/D.parquet'
        if not p.exists() or not code.startswith(('0', '3', '6')):
            continue
        try:
            df = load(code)
            if df is None or len(df) < 80:
                continue
            d2i = {str(d.date()): i for i, d in enumerate(df['date'])}
            hits = sorted([d2i[d] for d in tset if d in d2i])
            if not hits:
                continue
            for i in hits:
                if i < 60:
                    continue
                s = score_date(df, i)
                if s >= THRESH:
                    # 连击天数：往前数连续>=THRESH的天数
                    streak = 1
                    j = i - 1
                    while j >= 60:
                        try:
                            if smart_money_score(df, j)[0] >= THRESH:
                                streak += 1
                                j -= 1
                            else:
                                break
                        except Exception:
                            break
                    rows.append({'date': str(df['date'].iloc[i].date()),
                                 'code': code, 'score': s, 'streak': streak,
                                 'close': round(float(df['close'].iloc[i]), 2),
                                 'ret_1d': None, 'ret_5d': None,
                                 'ret_10d': None, 'ret_20d': None})
        except Exception:
            continue
    return rows


def backfill(hist):
    """回填到期未算的后续收益（用各自个股K线）"""
    pending = hist[hist['ret_20d'].isna()]
    if not len(pending):
        return 0
    codes = pending['code'].unique()
    filled = 0
    for code in codes:
        df = load(code)
        if df is None:
            continue
        d2i = {str(d.date()): i for i, d in enumerate(df['date'])}
        sub = pending[pending['code'] == code]
        for idx, row in sub.iterrows():
            i = d2i.get(row['date'])
            if i is None:
                continue
            base = df['close'].iloc[i]
            new = {}
            for n, col in [(1, 'ret_1d'), (5, 'ret_5d'), (10, 'ret_10d'), (20, 'ret_20d')]:
                if pd.isna(hist.at[idx, col]) and i + n < len(df):
                    new[col] = round((df['close'].iloc[i + n] / base - 1) * 100, 2)
            if new:
                for k, v in new.items():
                    hist.at[idx, k] = v
                filled += 1
    return filled


def summarize(hist):
    print('\n===== smart信号追踪统计 =====')
    n5 = hist.dropna(subset=['ret_5d'])
    if len(n5):
        print(f"累计信号 {len(hist)} 条 | 已到期(5d) {len(n5)} 条")
        print(f"  5d : 均值{n5['ret_5d'].mean():+.2f}% | 胜率{(n5['ret_5d']>0).mean()*100:.0f}%")
        for col, tag in [('ret_10d','10d'), ('ret_20d','20d')]:
            v = hist.dropna(subset=[col])
            if len(v):
                print(f"  {tag} : 均值{v[col].mean():+.2f}% | 胜率{(v[col]>0).mean()*100:.0f}%")
        # 连击分层
        for lo, hi, tag in [(1,1,'1天'),(2,2,'2天'),(3,3,'3天'),(4,99,'≥4天')]:
            g = n5[(n5['streak']>=lo)&(n5['streak']<=hi)]
            if len(g) >= 10:
                print(f"  连击{tag}: N={len(g)} | 5d均值{g['ret_5d'].mean():+.2f}% | 胜率{(g['ret_5d']>0).mean()*100:.0f}%")
    # 最近3日信号
    latest = sorted(hist['date'].unique())[-3:]
    rec = hist[hist['date'].isin(latest) & (hist['streak']>=2)]
    if len(rec):
        print(f"\n最近3日 连击≥2 信号 {len(rec)} 只:")
        print(rec.sort_values(['streak','score'], ascending=False)
                  [['date','code','score','streak','close']].head(15).to_string(index=False))


def main():
    seed_n = int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else None
    if OUT.exists():
        hist = pd.read_parquet(OUT)
        dates_done = set(hist['date'].unique())
        last_done = max(dates_done)
        ref_dates = get_trade_dates()
        new_dates = [d for d in ref_dates if str(d.date()) > last_done]
        print(f"已有 {len(hist)} 条记录，最后日期 {last_done}，新增 {len(new_dates)} 个交易日待扫")
    else:
        hist = pd.DataFrame(columns=['date','code','score','streak','close',
                                     'ret_1d','ret_5d','ret_10d','ret_20d'])
        n = seed_n or SEED_DAYS
        new_dates = get_trade_dates(n)
        print(f"首次运行，回填最近 {len(new_dates)} 个交易日")

    tset = [d.date() for d in new_dates]
    if tset:
        rows = scan_dates(tset)
        print(f"扫描完成：{len(rows)} 条 score>={THRESH} 信号")
        if rows:
            hist = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
            # 去重
            hist = hist.drop_duplicates(subset=['date','code'], keep='first')

    filled = backfill(hist)
    if filled:
        print(f"回填后续收益 {filled} 条")
    hist.to_parquet(OUT, index=False)
    print(f"落盘 {OUT}（{len(hist)} 行）")
    summarize(hist)


if __name__ == '__main__':
    main()
