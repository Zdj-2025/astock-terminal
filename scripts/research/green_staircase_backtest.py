#!/usr/bin/env python3
"""绿珠上梯形态历史回测 — 数据港形态(2026-08-15用户发现)严格验证。

事件定义(每天每只, 全无未来):
  窗口W=5日: 绿柱>=4(close<open) 且 高开>=2 且 重心上移 且 MA5抬升 且 近2日放量(>1.1x)
  位置过滤: 窗口涨幅<4% (涨多的不要, 8/13横截面已证: >=8%组次日-1.45%)
统计: 事件后 1/3/5/10日收益 vs 全市场同日基准, t检验p值, 分组(严格度/量能/窗口涨幅)。

用法: python scripts/research/green_staircase_backtest.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')

W = 5           # 形态窗口
HOLD_LIST = [1, 3, 5, 10]


def load_returns_matrix(kdir):
    """全市场收盘价矩阵(交易日对齐) + 每股OHLC序列"""
    rets, metas = {}, {}
    for code in os.listdir(kdir):
        p = os.path.join(kdir, code, 'D.parquet')
        if not os.path.isfile(p) or not code.isdigit() or len(code) != 6:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        dcol = [c for c in df.columns if 'date' in c.lower()]
        if not dcol:
            continue
        df = df.sort_values(dcol[0]).reset_index(drop=True)
        if len(df) < 60:
            continue
        rets[code] = df
        metas[code] = str(df[dcol[0]].iloc[-1])[:10]
    return rets


def detect_events(df):
    """逐日检测形态事件(全因果: 只用t及之前数据)"""
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    o = df['open'].astype(float).values
    c = df['close'].astype(float).values
    v = df['volume'].astype(float).values
    ma5 = pd.Series(c).rolling(5).mean().values
    events = []
    for t in range(W + 1, len(df) - max(HOLD_LIST)):
        s = t - W + 1                      # 窗口[s..t]
        ow, cw, vw = o[s:t + 1], c[s:t + 1], v[s:t + 1]
        greens = np.sum(cw < ow)
        gaps = np.sum(ow[1:] > cw[:-1]) + (1 if ow[0] > c[s - 1] else 0)
        half = 2
        lift = cw[-half:].mean() > cw[:half].mean()
        ma_up = pd.notna(ma5[t]) and pd.notna(ma5[s]) and ma5[t] > ma5[s]
        vol_amp = vw[-2:].mean() > vw[:-2].mean() * 1.1
        ret_w = (c[t] / c[s - 1] - 1) * 100
        if greens >= 4 and gaps >= 2 and lift and ma_up and vol_amp and ret_w < 4:
            fwd = {h: (c[t + h] / c[t] - 1) * 100 if t + h < len(c) else np.nan for h in HOLD_LIST}
            events.append({'date': str(df[dcol].iloc[t])[:10], 'ret_w': ret_w,
                           'vol_x': vw[-2:].mean() / max(vw[:-2].mean(), 1), **fwd})
    return events


def main():
    kdir = 'data/store/kline'
    files = [d for d in os.listdir(kdir) if d.isdigit() and len(d) == 6
             and os.path.isfile(os.path.join(kdir, d, 'D.parquet'))]
    print(f'扫描{len(files)}只 | 窗口{W}日 绿柱>=4 高开>=2 上移+MA5抬升+放量+窗口<4%')
    all_events, all_base = [], []
    for i, code in enumerate(files):
        try:
            df = pd.read_parquet(os.path.join(kdir, code, 'D.parquet'))
        except Exception:
            continue
        evs = detect_events(df)
        for e in evs:
            e['code'] = code
        all_events.extend(evs)
        if i % 1500 == 0:
            print(f'  进度 {i}/{len(files)} 事件累计{len(all_events)}')
    E = pd.DataFrame(all_events)
    if len(E) == 0:
        print('无事件'); return
    # 基准: 每个事件日, 全市场所有股同持有期收益均值
    print('构建基准(逐事件日全市场均值)...')
    code_frames = {}
    for code in files[:]:
        try:
            df = pd.read_parquet(os.path.join(kdir, code, 'D.parquet'))
            dcol = [c for c in df.columns if 'date' in c.lower()][0]
            df = df.sort_values(dcol[0])
            s = df[[dcol, 'close']].copy()
            s.columns = ['date', 'close']
            code_frames[code] = s.set_index('date')['close'].astype(float)
        except Exception:
            continue
    P = pd.DataFrame(code_frames)          # 日期×股票收盘
    R = P.pct_change() * 100
    for h in HOLD_LIST:
        Rh = R.rolling(h).sum().shift(-h)   # 未来h日收益(标注在起点日)
        col = f'base{h}'
        E_date_idx = E['date'].values
        base_vals = []
        for d in E_date_idx:
            if d in Rh.index:
                base_vals.append(Rh.loc[d].mean())
            else:
                base_vals.append(np.nan)
        E[col] = base_vals
    # 输出
    print(f'\n=== 绿珠上梯回测 | 事件{len(E)}个 ===')
    print(f"{'持有':>4}{'事件均值':>10}{'基准':>8}{'超额':>8}{'胜率':>7}{'t值':>7}{'p值':>10}")
    for h in HOLD_LIST:
        a, b = E[f'fwd{h}' if f'fwd{h}' in E else h].dropna(), E[f'base{h}'].dropna()
        # 对齐
        m = E[[h if h in E.columns else f'fwd{h}', f'base{h}']].dropna()
        ev, bs = m.iloc[:, 0], m.iloc[:, 1]
        excess = ev - bs
        t, p = stats.ttest_1samp(excess, 0)
        print(f"{h:>3}d {ev.mean():>+9.2f}% {bs.mean():>+7.2f}% {excess.mean():>+7.2f}% {(ev>0).mean()*100:>6.0f}% {t:>7.2f} {p:>10.2e}")
    # 分组
    print('\n=== 分组(持有5日超额) ===')
    m5 = E[[5 if 5 in E.columns else 'fwd5', 'base5']].dropna()
    m5.columns = ['ev', 'base']; m5['ex'] = m5.ev - m5.base
    E5 = E.copy(); E5['ex5'] = E5[5 if 5 in E5.columns else 'fwd5'] - E5['base5']
    for name, sub in [('绿柱=4', E5[E5.ret_w.notna() & E5.get('greens', pd.Series(index=E5.index, data=4)).eq(4)] if 'greens' in E5 else E5),
                      ('量能>1.5x', E5[E5.vol_x > 1.5]), ('量能1.1-1.5x', E5[E5.vol_x <= 1.5]),
                      ('窗口<-1%(洗最狠)', E5[E5.ret_w < -1]), ('窗口-1~4%', E5[E5.ret_w >= -1])]:
        s = sub['ex5'].dropna()
        if len(s) > 10:
            t, p = stats.ttest_1samp(s, 0)
            print(f'{name:<18}{len(s):>6}个 超额均值{s.mean():>+7.2f}% 胜率{(s>0).mean()*100:>5.0f}% p={p:.2e}')
    E.to_csv('reports/green_staircase_backtest_events.csv', index=False)
    print('\n事件明细→reports/green_staircase_backtest_events.csv')


if __name__ == '__main__':
    main()
