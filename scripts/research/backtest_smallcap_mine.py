#!/usr/bin/env python3
"""Phase1.1 小市值×排雷回测 — 2026-09-11(9月既定核心,周末系统周执行).

假设: 排雷后的小市值池跑赢全市场等权(小市值溢价×排雷增量).
构造: 每月末调仓, 等权持有;
  组A 小市值原始: total_mv最小五分位
  组B 小市值+排雷: A ∩ (pe_ttm>0盈利) ∩ (价>2) ∩ (非ST/退,当前名单近似)
  组C 全市场等权(基准)
  组D 大市值对照: total_mv最大五分位
区间: 2024-02 → 2026-09(33个月度调仓), 复权K线, 月度收益复合.
诚实边界: ST名单用当前快照近似(历史ST状态不可复原);五查历史点时数据不可得,
  排雷=轻量版(盈利+价格+ST), 完整五查增量留Phase1.2(API预算内不可行 tonight).
判定: B>C 且 B>A(排雷有增量) 且分年稳定 → 进入观察池候选策略;
  仅A>C(小市值裸因子) → 记录不启用(2026H2买入类信号全灭的regime先例).
数据: 月度截面pro.daily_basic(trade_date=)缓存到data/store/research/smallcap_xs/.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import time
import numpy as np
import pandas as pd

XS_DIR = Path('data/store/research/smallcap_xs')
XS_DIR.mkdir(parents=True, exist_ok=True)
STOCK_BENCH = None


def month_end_dates():
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    d = idx['trade_date'].astype(str)
    df = pd.DataFrame({'d': d})
    df['ym'] = df['d'].str[:6]
    me = df.groupby('ym')['d'].max()
    return [x for x in me if '202401' <= x[:6] <= '20260831'[:6]]


def get_xs(date):
    p = XS_DIR / f'{date}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    from scripts.core.tushare_client import TushareClient
    pro = TushareClient().pro
    df = pro.daily_basic(trade_date=date, fields='ts_code,total_mv,pe_ttm,close')
    time.sleep(0.35)
    if df is None or not len(df):
        return None
    df.to_parquet(p, index=False)
    return df


def kline_ret_map():
    """每股: date→close(复权). 供月末收益查取."""
    return None  # 逐股读parquet太慢, 改为按需读


def main():
    sl = pd.read_parquet('data/store/stock_list.parquet')
    sym = sl['ts_code'].astype(str)
    nm = dict(zip(sym, sl['name'].astype(str)))
    st_set = {c for c, n in nm.items() if 'ST' in n or '退' in n}

    dates = month_end_dates()
    print(f'调仓日: {len(dates)}个 ({dates[0]}~{dates[-1]})')

    rows = []
    for di, dt in enumerate(dates[:-1]):
        xs = get_xs(dt)
        nxt = dates[di + 1]
        if xs is None:
            continue
        xs['code'] = xs['ts_code'].astype(str).str[:6]
        xs = xs[~xs['ts_code'].astype(str).str[:2].isin(('68',)) | (xs['ts_code'].astype(str).str[:2] == '68')]
        xs = xs[~xs['ts_code'].str.contains('.BJ')]
        # 月收益: 从kline
        rets = {}
        store = Path('data/store/kline')
        for code in xs['code']:
            kp = store / code / 'D.parquet'
            if not kp.exists():
                continue
            try:
                k = pd.read_parquet(kp, columns=['date', 'close'])
                for c2 in ['datetime', 'day', 'trade_date']:
                    if c2 in k.columns and 'date' not in k.columns:
                        k = k.rename(columns={c2: 'date'})
            except Exception:
                continue
            k['d'] = pd.to_datetime(k['date']).dt.strftime('%Y%m%d')
            k = k.sort_values('d').drop_duplicates('d')
            c0 = k.loc[k['d'] <= dt, 'close']
            c1 = k.loc[k['d'] <= nxt, 'close']
            if len(c0) and len(c1):
                rets[code] = float(c1.iloc[-1]) / float(c0.iloc[-1]) - 1
        r = pd.Series(rets)
        xs = xs[xs['code'].isin(rets)].copy()
        xs['ret'] = xs['code'].map(r)
        valid = xs.dropna(subset=['ret'])
        if len(valid) < 300:
            continue
        valid['mv_q'] = valid['total_mv'].rank(pct=True)
        small = valid[valid['mv_q'] <= 0.2]
        large = valid[valid['mv_q'] >= 0.8]
        small_clean = small[(small['pe_ttm'] > 0) & (small['close'] > 2)
                            & (~small['code'].isin({c[:6] for c in st_set}))]
        rows.append(dict(
            month=dt[:6],
            all=valid['ret'].mean(),
            small=small['ret'].mean() if len(small) else np.nan,
            small_clean=small_clean['ret'].mean() if len(small_clean) else np.nan,
            large=large['ret'].mean() if len(large) else np.nan,
            n_small=len(small), n_clean=len(small_clean),
            limit_up_clip=valid['ret'].clip(-0.25, 0.25).mean(),  # 等权含涨停不可得的保守校验
        ))
        print(f"  {dt[:6]}: N={len(valid)} 小市值{len(small)}只月均{rows[-1]['small']*100:+.1f}% 排雷后{len(small_clean)}只{rows[-1]['small_clean']*100:+.1f}% | 全市场{rows[-1]['all']*100:+.1f}%")

    t = pd.DataFrame(rows).dropna()
    t.to_csv('reports/smallcap_mine_backtest.csv', index=False)
    print('\n' + '═' * 76)
    print(f"组别             累计    年化    月胜率   最大月亏   2024    2025    2026YTD")
    def curve(x):
        c = (1 + x).cumprod().iloc[-1] - 1
        ann = (1 + c) ** (12 / len(x)) - 1
        return c * 100, ann * 100, (x > 0).mean() * 100, x.min() * 100
    for col, label in [('all', 'C 全市场等权'), ('small', 'A 小市值原始'), ('small_clean', 'B 小市值+排雷'), ('large', 'D 大市值对照')]:
        c, ann, wr, worst = curve(t[col])
        by = t.assign(y=t['month'].str[:4]).groupby('y')[col].apply(lambda s: ((1 + s).prod() - 1) * 100)
        y24, y25, y26 = (by.get(str(y), np.nan) for y in (2024, 2025, 2026))
        print(f"{label:<14}{c:>+8.1f}%{ann:>+7.1f}%{wr:>7.0f}%{worst:>+8.1f}%{y24:>+7.1f}%{y25:>+7.1f}%{y26:>+7.1f}%")


if __name__ == '__main__':
    main()
