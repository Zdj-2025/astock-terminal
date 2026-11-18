#!/usr/bin/env python3
"""背离+双底回测(2026-09-02设计,用户主导逻辑).

形态(实测自华胜天成600410):
  La(背离头底) → 下跌(Lb<La新低, hist/DIF抬升=背离) → Lb
  Lb → 大反弹(N=max high, N/Lb≥门槛20%) → 回踩
  Lc ≈ Lb(±7%,更高的低点) → 二底确认 → 触发
红柱积分(用户新思路): Lb→Lc段最大红柱簇积分 > La→Lb段最大红柱簇积分(动量维度吸筹证据)

触发两版:
  早触发: Lc后3根bar低点不破Lc + close∈[Lc*1.04, Lc*1.12] → t+1开盘买
  晚触发: close > N(颈线) → t+1开盘买
止损: close < min(Lb,Lc) 次日出

四臂对照: A纯背离(E组合+12%新鲜) / B背离+双底(20%门槛) / C=B+红柱积分 / D晚触发
参数扫描: 反弹门槛r∈{15%,20%,25%,30%}
验证模式: --verify 在600410复现全部关键数字
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')

START, END = '20240601', '20260901'
HOLDS = [5, 10, 20, 40]
COST = 0.0013
SW = 5                 # swing确认宽度
DIV_MIN, DIV_MAX = 15, 60   # La→Lb间距
DB_MAX = 40            # Lb→Lc间距
BAND = 0.07            # 两底接近带 |Lc/Lb-1|
FRESH_LO, FRESH_HI = 1.04, 1.12   # 早触发反弹带(锚Lc,12%新鲜线上限)
RALLY_BASE = 0.20      # 反弹门槛基准(扫描15/20/25/30%)


def macd(s):
    ef = s.ewm(span=12, adjust=False).mean()
    es = s.ewm(span=26, adjust=False).mean()
    dif = ef - es
    return dif, dif - dif.ewm(span=9, adjust=False).mean()


def swing_lows(lows, n=SW):
    out = []
    for i in range(n, len(lows) - n):
        if all(lows[i] < lows[i - j] for j in range(1, n + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, n + 1)):
            out.append(i)
    return out


def red_clusters(hist):
    """红柱簇: [(start, end, integral)]"""
    runs, i = [], 0
    while i < len(hist):
        if hist[i] > 0:
            j = i
            while j < len(hist) and hist[j] > 0:
                j += 1
            runs.append((i, j - 1, float(hist[i:j].sum())))
            i = j
        else:
            i += 1
    return runs


def detect_patterns(df, rally=0.20):
    """返回结构列表: 每个含La,Lb,Lc,N,红柱积分,早/晚触发日"""
    l = df['low'].astype(float).values
    h = df['high'].astype(float).values
    c = df['close'].astype(float).values
    dif, hist = macd(pd.Series(c))
    dif, hist = dif.values, hist.values
    lows = swing_lows(l)
    if len(lows) < 3:
        return []
    out = []
    for k in range(2, len(lows)):
        La, Lb, Lc = lows[k - 2], lows[k - 1], lows[k]
        if not (Lb - La >= DIV_MIN and Lc - Lb <= DB_MAX):
            continue
        # 背离: Lb新低 + hist/DIF双抬
        if not (l[Lb] < l[La] and hist[Lb] > hist[La] and dif[Lb] > dif[La]):
            continue
        # B过滤: Lb谷后再跌<10%
        if np.min(l[Lb + 1:Lc + 1]) < l[Lb] * 0.90:
            continue
        # 双底: 两底差不多
        if abs(l[Lc] / l[Lb] - 1) > BAND:
            continue
        # 反弹门槛
        N = float(np.max(h[Lb:Lc + 1]))
        if N < l[Lb] * (1 + rally):
            continue
        if l[Lc] >= N:
            continue
        # 红柱积分(最大簇)
        def max_cluster(x0, x1):
            cl = [s for a, b, s in red_clusters(hist[x0:x1 + 1])]
            return max(cl) if cl else 0.0
        red_before = max_cluster(La, Lb)
        red_during = max_cluster(Lb, Lc)
        # 早触发: Lc后3根低点不破Lc + 反弹带
        early = None
        if Lc + 4 < len(c):
            hold3 = all(l[Lc + 1:Lc + 4] > l[Lc])
            if hold3:
                for t in range(Lc + 3, min(Lc + 16, len(c))):
                    if c[t] >= l[Lc] * FRESH_LO and c[t] <= l[Lc] * FRESH_HI:
                        early = t
                        break
        # 晚触发: 突破N
        late = None
        for t in range(Lc + 1, len(c)):
            if c[t] > N:
                late = t
                break
        out.append(dict(La=La, Lb=Lb, Lc=Lc, N=N, red_before=red_before,
                        red_during=red_during, early=early, late=late))
    return out


def load(code):
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    return df.sort_values('d').drop_duplicates('d', keep='last').reset_index(drop=True)


def verify():
    """华胜天成600410全链路数字复现"""
    df = load('600410')
    l = df['low'].astype(float).values
    h = df['high'].astype(float).values
    pats = detect_patterns(df, rally=0.20)
    print(f'600410 检出结构数: {len(pats)}')
    for p in pats:
        if p['Lb'] < len(df) - 60:  # 只详列最近的
            continue
        print(f"\nLa={df.d.iloc[p['La']]} low={l[p['La']]:.2f}")
        print(f"Lb={df.d.iloc[p['Lb']]} low={l[p['Lb']]:.2f} (La→Lb {(l[p['Lb']]/l[p['La']]-1)*100:+.1f}%, {p['Lb']-p['La']}日)")
        print(f"N ={df.d.iloc[p['La']]}~{df.d.iloc[p['Lc']]}区间高点 {p['N']:.2f} (N/Lb {(p['N']/l[p['Lb']]-1)*100:+.1f}%)")
        print(f"Lc={df.d.iloc[p['Lc']]} low={l[p['Lc']]:.2f} (Lc/Lb {(l[p['Lc']]/l[p['Lb']]-1)*100:+.1f}%)")
        print(f"红柱积分: 双底段{p['red_during']:.3f} vs 下跌段{p['red_before']:.3f} → {'✓递增' if p['red_during']>p['red_before'] else '✗'}")
        print(f"早触发日: {df.d.iloc[p['early']] if p['early'] else '无'} | 晚触发日: {df.d.iloc[p['late']] if p['late'] else '未破颈线'}")
        # 人工核对表
        exp = {'La': '20260608@16.89', 'Lb': '20260721@11.95', 'N': '16.40(+37.2%)',
               'Lc': '20260825@12.74(+6.6%)', '红柱': '4.594>1.663', '早触发': '约20260828-31'}
        print('人工核对:', exp)


def main():
    from scipy import stats as st
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    # 补新数据
    if idx['d'].iloc[-1] < '20260820':
        from scripts.core.tushare_client import TushareClient
        t = TushareClient().pro.index_daily(ts_code='000001.SH', start_date=idx['d'].iloc[-1], end_date='20260901')
        idx = pd.concat([idx, t.drop_duplicates('trade_date')], ignore_index=True).drop_duplicates('d').sort_values('d')
    ic = idx.set_index('d')['close'].astype(float)
    dates = idx['d'].tolist()
    dpos = {d: i for i, d in enumerate(dates)}

    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])
    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))

    def exec_trades(rows, label):
        """rows: (code, t_idx_in_df, df)"""
        recs = []
        for code, t, df in rows:
            if t + 1 >= len(df):
                continue
            ed = df.d.iloc[t + 1]
            op = float(df['open'].iloc[t + 1])
            pc = float(df['close'].iloc[t])
            lim = 0.195 if code[0] in ('3', '68') else 0.095
            if op / pc - 1 > lim or ed not in dpos or ed > '20260828':
                continue
            ii = dpos[ed]
            r = dict(code=code, name=nm.get(code, ''), t=df.d.iloc[t], ed=ed, entry=op)
            for H in HOLDS:
                ci = min(t + 1 + H, len(df) - 1)
                px = float(df['close'].iloc[ci])
                jj = min(ii + H, len(dates) - 1)
                r[f'ex{H}'] = ((px / op - 1 - COST) - (ic[dates[jj]] / ic[ed] - 1)) * 100
            recs.append(r)
        return pd.DataFrame(recs, columns=['code', 'name', 't', 'ed', 'entry'] + [f'ex{h}' for h in HOLDS])

    def report(tdf, label):
        if tdf.empty:
            print(f'{label:<30} N=0')
            return
        line = f'{label:<30} N={len(tdf):>4}'
        for H in [10, 20, 40]:
            e = tdf[f'ex{H}']
            line += f' | {H}d中{e.median():+.2f}%/均{e.mean():+.2f}%/胜{(e>0).mean()*100:.0f}%'
        print(line)

    # 参数扫描 + 四臂
    for rally in [0.15, 0.20, 0.25, 0.30]:
        rowsB, rowsC, rowsD = [], [], []
        for code in codes:
            try:
                df = load(code)
            except Exception:
                continue
            if len(df) < 300:
                continue
            if 'ST' in nm.get(code, '') or float(df['close'].iloc[-1]) < 2:
                continue
            c = df['close'].astype(float)
            if c.iloc[-1] / c.rolling(120).max().iloc[-1] - 1 < -0.75:
                continue
            l = df['low'].astype(float).values
            for p in detect_patterns(df, rally):
                Lc = p['Lc']
                trig = p['early']
                if trig is None or not (START <= df.d.iloc[trig] <= END):
                    continue
                rowsB.append((code, trig, df))
                if p['red_during'] > p['red_before']:
                    rowsC.append((code, trig, df))
                if p['late'] is not None and START <= df.d.iloc[p['late']] <= END:
                    rowsD.append((code, p['late'], df))
        tag = f'门槛{rally:.0%}'
        report(exec_trades(rowsB, ''), f'B 背离+双底({tag})')
        report(exec_trades(rowsC, ''), f'C =B+红柱积分({tag})')
        report(exec_trades(rowsD, ''), f'D 晚触发·破颈线({tag})')
        # 红柱增量消融(20%档详列)
        if rally == 0.20:
            b, c_ = exec_trades(rowsB, ''), exec_trades(rowsC, '')
            if not b.empty and not c_.empty:
                m = b.merge(c_[['code', 't']], on=['code', 't'], how='inner', suffixes=('_b', ''))
                only_b = b[~b.set_index(['code', 't']).index.isin(c_.set_index(['code', 't']).index)]
                t1, p1 = st.ttest_ind(c_['ex20'].dropna(), only_b['ex20'].dropna(), equal_var=False)
                print(f'   消融: 红柱通过组N={len(c_)} vs 未通过组N={len(only_b)} | 20d均差{c_.ex20.mean()-only_b.ex20.mean():+.2f}% (p={p1:.3f})')


if __name__ == '__main__':
    if '--verify' in sys.argv:
        verify()
    else:
        main()
