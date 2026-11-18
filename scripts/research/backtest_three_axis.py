#!/usr/bin/env python3
"""三轴配置回测 — 2026-09-12立项(用户批准: 股/ETF/类债三轴目标比例正式立项).

背景: 一杆清(ETF+头部价值)/政事堂(类债+安全)/族计分卡(防御+Tier1为正)三源同指一个方向;
用户实际仓位已先于规则迁移(ETF 56%+类债篮子已建). 本回测回答: 三轴怎么配最优?

诚实边界(先声明):
  ① 轴的beta本质: 中证1000=高beta进攻, 类债篮子=低beta+股息carry, 现金=零.
     "个股战术轴"≈高beta+alpha(我们2026的alpha≈0)→用中证1000代理,承认这个假设.
  ② 月度调仓,无交易成本估计的再平衡成本(类债/ETF佣金极低,可忽略).
  ③ 类债=大秦/长电/国投等权(2024-2026幸存者,有轻微幸存者偏差——三只是今天筛出来的).
     处置: 结论只用于"轴间比例"不用"轴内选股".

对比组合(月度再平衡):
  P1 纯进攻: 100%中证1000
  P2 纯类债: 100%类债篮子
  P3 静态50/50
  P4 静态6/3/1(进攻60%类债30%现金10%)
  P5 regime动态: ret20>+3%→8/1/1;震荡→5/3/2;ret20<-3%→2/5/3(防守切类债)
判定: 动态>静态>单轴 的收益回撤比排序 → 给出目标带提案.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import numpy as np
import pandas as pd


def month_end_series(daily_close):
    s = daily_close.copy()
    s.index = pd.to_datetime(s.index)
    me = s.resample('ME').last()
    return me.dropna()


def main():
    # ① 中证1000: 本地+API补全
    idx = pd.read_parquet('data/store/index/000852_SH.parquet').sort_values('trade_date')
    if str(idx['trade_date'].iloc[-1]) < '20260911':
        from scripts.core.tushare_client import TushareClient
        last = str(idx['trade_date'].iloc[-1])
        t = TushareClient().pro.index_daily(ts_code='000852.SH', start_date=last, end_date='20260911')
        idx = pd.concat([idx, t], ignore_index=True)
    idx['d'] = idx['trade_date'].astype(str)
    idx = idx.drop_duplicates('d').sort_values('d')
    _m = pd.to_numeric(pd.Series(idx['close'].astype(float).values, index=idx['d'].values), errors='coerce').dropna()
    _m = _m[~_m.index.duplicated()].sort_index()
    m1000 = month_end_series(_m)

    # ② 类债篮子: 三只等权月收益
    rets = []
    for code in ['601006', '600900', '600886']:
        k = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
        for c in ['datetime', 'day', 'trade_date']:
            if c in k.columns and 'date' not in k.columns:
                k = k.rename(columns={c: 'date'})
        d = pd.to_datetime(k['date']).dt.strftime('%Y%m%d')
        s = pd.Series(k['close'].astype(float).values, index=d.values).sort_index()
        rets.append(month_end_series(s).pct_change())
    bond_m = pd.concat(rets, axis=1).mean(axis=1)  # 等权月收益
    bond_m = bond_m.reindex(m1000.index)

    # 上证ret20做regime(月末口径)
    sh = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    if str(sh['trade_date'].iloc[-1]) < '20260911':
        last2 = str(sh['trade_date'].iloc[-1])
        t2 = TushareClient().pro.index_daily(ts_code='000001.SH', start_date=last2, end_date='20260911')
        sh = pd.concat([sh, t2], ignore_index=True)
    sh['d'] = sh['trade_date'].astype(str)
    sh = sh.drop_duplicates('d').sort_values('d')
    shc = pd.Series(sh['close'].astype(float).values, index=sh['d'].values)
    shc = pd.to_numeric(shc, errors='coerce').dropna()
    shc = shc[~shc.index.duplicated()].sort_index()
    r20_at = (shc / shc.shift(20) - 1)

    m1000r = m1000.pct_change()
    df = pd.DataFrame({'attack': m1000r, 'bond': bond_m}).dropna()
    # regime: 每月末用当时(非未来)的ret20
    def regime_at(ts):
        d = ts.strftime('%Y%m%d')
        sub = r20_at.loc[:d].dropna()
        if len(sub) < 1:
            return 'shake'
        v = sub.iloc[-1]
        return 'attack' if v > 0.03 else ('defense' if v < -0.03 else 'shake')
    df['regime'] = [regime_at(ts) for ts in df.index]

    ports = {
        'P1 纯进攻1000ETF': (1.0, 0.0, 0.0, False),
        'P2 纯类债篮子': (0.0, 1.0, 0.0, False),
        'P3 静态50/50': (0.5, 0.5, 0.0, False),
        'P4 静态6/3/1': (0.6, 0.3, 0.1, False),
        'P5 regime动态': (None, None, None, True),
    }
    DYN = {'attack': (0.8, 0.1, 0.1), 'shake': (0.5, 0.3, 0.2), 'defense': (0.2, 0.5, 0.3)}
    print(f'三轴配置回测 | {df.index[0].strftime("%Y-%m")}~{df.index[-1].strftime("%Y-%m")} 月度调仓 | N={len(df)}期')
    print(f"期初regime分布: {df['regime'].value_counts().to_dict()}")
    print('═' * 84)
    print(f"{'组合':<16}{'累计':>9}{'年化':>8}{'月胜率':>7}{'最大回撤':>9}{'收益/回撤':>9}")
    rows = {}
    for name, (wa, wb, wc, dyn) in ports.items():
        r = []
        for _, row in df.iterrows():
            a, b, c = DYN[row['regime']] if dyn else (wa, wb, wc)
            r.append(a * row['attack'] + b * row['bond'] + c * 0.0)
        r = pd.Series(r, index=df.index)
        cum = (1 + r).prod() - 1
        ann = (1 + cum) ** (12 / len(r)) - 1
        wealth = (1 + r).cumprod()
        dd = (wealth / wealth.cummax() - 1).min()
        rf = (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0
        rows[name] = (cum, ann, dd, r)
        print(f"{name:<16}{cum*100:>+8.1f}%{ann*100:>+7.1f}%{(r>0).mean()*100:>6.0f}%{dd*100:>+8.1f}%{(ann/abs(dd)) if dd else 0:>9.2f}")
    # 分年
    print('─' * 84)
    for name, (cum, ann, dd, r) in rows.items():
        by = r.groupby(r.index.year).apply(lambda s: ((1 + s).prod() - 1) * 100)
        ys = ' '.join(f"{y}:{v:+.0f}%" for y, v in by.items())
        print(f"{name:<16}分年 {ys}")
    df.to_csv('reports/three_axis_backtest.csv')


if __name__ == '__main__':
    main()
