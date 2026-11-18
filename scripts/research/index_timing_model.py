#!/usr/bin/env python3
"""大盘变盘倾向模型v0.1(2026-09-05)——三因子打分卡.

因子(全部因果,十年IC+尾部验证):
  S1 粘连极值: 六均线(MA5/10/20/60/120/250)离散度<前半样本10%分位 → +1
     依据: 极端粘连后20日66%上涨/+1.83%(283样本),一杆清观察的尾部验证
  S2 位置反转: 距MA60<-5%→+1(超跌);>+5%→-1(超买)  IC-0.115@20d
  S3 动量反转: 近20日跌幅<-5%→+1;涨幅>+5%→-1     IC+0.05@20d
分数∈[-2,+3]. 用法: 每日输出倾向分,辅助仓位/regime判断——概率非确认.

验证设计: 阈值只用2015-2020定(防lookahead), 后半2021-2026看稳定性.
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

P = 'data/store/index/sh000001_long.parquet'


def load():
    df = pd.read_parquet(P)
    for n in (5, 10, 20, 60, 120, 250):
        df[f'ma{n}'] = df['c'].rolling(n).mean()
    mas = df[[f'ma{n}' for n in (5, 10, 20, 60, 120, 250)]]
    df['粘连'] = mas.std(axis=1) / df['c']
    df['距MA60'] = df['c'] / df['ma60'] - 1
    df['ret20'] = df['c'] / df['c'].shift(20) - 1
    for H in (5, 10, 20):
        df[f'f{H}'] = df['c'].shift(-H) / df['c'] - 1
    return df.dropna(subset=['ma250', '距MA60'])


def score_row(r, q10):
    s = 0
    if r['粘连'] < q10:
        s += 1
    if r['距MA60'] < -0.05:
        s += 1
    elif r['距MA60'] > 0.05:
        s -= 1
    if r['ret20'] < -0.05:
        s += 1
    elif r['ret20'] > 0.05:
        s -= 1
    return s


def main():
    df = load()
    train = df[df['d'] < '20210101']
    test = df[df['d'] >= '20210101']
    q10 = train['粘连'].quantile(0.10)
    print(f'粘连阈值(2015-2020训练段10%分位): {q10:.4f}')

    print('\n═══ 稳定性: 极端粘连的前瞻收益(分半) ═══')
    for name, seg in [('2015-2020', train), ('2021-2026', test)]:
        ext = seg[seg['粘连'] < q10]
        f20 = ext['f20'].dropna()
        print(f'  {name}: N={len(f20)} 20日均{f20.mean()*100:+.2f}% 中位{f20.median()*100:+.2f}% 涨比例{(f20>0).mean()*100:.0f}%')

    df['score'] = df.apply(lambda r: score_row(r, q10), axis=1)
    print('\n═══ 打分卡条件分布(全样本) ═══')
    for s in sorted(df['score'].dropna().unique(), reverse=True):
        g = df[df['score'] == s]
        f = g['f20'].dropna()
        if len(f) < 10:
            continue
        f5m = g['f5'].mean() * 100
        f10m = g['f10'].mean() * 100
        print(f'  分数{s:+d}: N={len(g):>4} | 5日均{f5m:+.2f}% | 10日均{f10m:+.2f}% | '
              f'20日均{f.mean()*100:+.2f}% 涨{(f>0).mean()*100:.0f}%')

    test = df[df['d'] >= '20210101']  # score列加了之后重切
    print('\n═══ 样本外(2021-2026)打分卡 ═══')
    for s in [3, 2, 1, 0, -1]:
        g = test[test['score'] == s]
        f = g['f20'].dropna()
        if len(f) < 10:
            continue
        print(f'  分数{s:+d}: N={len(g):>4} | 20日均{f.mean()*100:+.2f}% 涨{(f>0).mean()*100:.0f}%')

    # 当前
    r = df.iloc[-1]
    print(f'\n═══ 当前({r["d"]}) ═══')
    print(f'  粘连{r["粘连"]:.4f}({"极端" if r["粘连"]<q10 else "正常"}) 距MA60{r["距MA60"]*100:+.1f}% ret20{r["ret20"]*100:+.1f}%')
    print(f'  → 变盘倾向分: {int(r["score"]):+d} (范围-2~+3)')


if __name__ == '__main__':
    main()
