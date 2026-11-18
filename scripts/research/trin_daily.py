#!/usr/bin/env python3
"""TRIN/阿姆氏指标 — 每日全市场实算(2026-10-26建, 布鲁斯《第21集》方法论落地)

公式: TRIN = (涨家数/跌家数) ÷ (涨方成交额/跌方成交额)   [chg+amount口径]
解读(布鲁斯框架):
  TRIN < 0.7  = 多头主动进攻(资金放量流入上涨股)
  0.7 ~ 1.1   = 中性
  > 1.1       = 跌方承接(资金流向下跌股)
注意: 大宽度日(>80%)的低TRIN有机械成分(普涨自然放量), 最有效信号在
  低宽度恐慌日(TRIN<0.5=少数上涨股爆量=机构抄底权重)。

用法:
  PYTHONPATH=. python scripts/research/trin_daily.py            # 近10日
  PYTHONPATH=. python scripts/research/trin_daily.py --days 20

数据: data/store/kline/*/D.parquet 全市场(~5000只, 涨跌=收盘vs前收, 金额口径)
2026-10-26核查: 布鲁斯声称10/23 TRIN=0.9/3月23日=0.3, 四种口径(chg/candle × volume/amount)
均无法复现; 我方实算 10/23=0.36(多头进攻) 3/23=0.92(中性)。结论: 框架可用, 他的数字不可信。
"""
import pandas as pd
import os
import sys
import argparse

STORE = 'data/store/kline'

def trin_series(dates, metric='amount'):
    adv = {d: 0 for d in dates}; dec = {d: 0 for d in dates}
    am = {d: 0.0 for d in dates}; dm = {d: 0.0 for d in dates}
    for code in os.listdir(STORE):
        p = f'{STORE}/{code}/D.parquet'
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_parquet(p, columns=['close', metric, 'date'])
        except Exception:
            continue
        if len(df) < 2:
            continue
        df['date'] = df['date'].astype(str)
        df = df.sort_values('date').reset_index(drop=True)
        for i in range(1, len(df)):
            d = df['date'].iloc[i]
            if d not in dates:
                continue
            chg = df['close'].iloc[i] - df['close'].iloc[i - 1]
            m = df[metric].iloc[i]
            if pd.isna(m) or m <= 0:
                continue
            if chg > 0:
                adv[d] += 1; am[d] += m
            elif chg < 0:
                dec[d] += 1; dm[d] += m
    rows = []
    for d in dates:
        if dec[d] == 0 or dm[d] == 0 or adv[d] == 0 or am[d] == 0:
            continue
        t = (adv[d] / dec[d]) / (am[d] / dm[d])
        rows.append({'date': d, 'trin': round(t, 2),
                     'breadth': round(adv[d] / (adv[d] + dec[d]) * 100),
                     'adv_amt_pct': round(am[d] / (am[d] + dm[d]) * 100)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=10)
    args = ap.parse_args()

    # 从新鲜kline取最近N个交易日(index parquet可能过期)
    ref = pd.read_parquet(f'{STORE}/600276/D.parquet', columns=['date'])
    dates = sorted(ref['date'].astype(str).unique())[-args.days:]
    df = trin_series(dates)
    df['tag'] = df['trin'].apply(lambda t: '多头进攻' if t < 0.7 else ('中性' if t <= 1.1 else '跌方承接'))
    print(f'{"日期":<10}{"TRIN":>6}{"宽度%":>6}{"涨方金额%":>8}  解读')
    for _, r in df.iterrows():
        print(f'{r["date"]:<10}{r["trin"]:>6.2f}{r["breadth"]:>6}{r["adv_amt_pct"]:>8}  {r["tag"]}')
    return df

if __name__ == '__main__':
    main()
