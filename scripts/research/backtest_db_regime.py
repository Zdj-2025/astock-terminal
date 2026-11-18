#!/usr/bin/env python3
"""双底算法×regime条件化回测 — 2026-09-11立(预注册假设:只在坏转好/B反期启用).

背景: 双底算法上线两周两发两负超额(国能日新/华明装备),但样本仅2.
回测族归属: 复用2026-09-02 backtest_div_double_bottom的B臂(背离+双底20%门槛,早触发).

分桶(入场日上证ret20):
  deep_down(<-8%) / down(-8~-3%) / shake(-3~+3%) / up(+3~+8%) / strong_up(>8%)
专项: 坏转好 = ret20<0 且 ret5>0 (下跌中段反弹拐点,B反proxy)

判定标准(R1): 某桶20d超额均值显著为正(t检验p<0.05且N≥30)才考虑regime门启用;
  全桶平庸或全负 → 按既定40日持有期继续观察,不立法.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import pandas as pd
from scipy import stats as st
from scripts.research.backtest_div_double_bottom import load, detect_patterns, HOLDS, COST

START, END = '20240601', '20260901'


def regime_of(ret20, ret5):
    if ret20 < -0.08:
        return 'deep_down'
    if ret20 < -0.03:
        return 'down'
    if ret20 <= 0.03:
        return 'shake'
    if ret20 <= 0.08:
        return 'up'
    return 'strong_up'


def main():
    idx = pd.read_parquet('data/store/index/000001_SH.parquet').sort_values('trade_date')
    idx['d'] = idx['trade_date'].astype(str)
    if idx['d'].iloc[-1] < '20260820':
        from scripts.core.tushare_client import TushareClient
        t = TushareClient().pro.index_daily(ts_code='000001.SH', start_date=idx['d'].iloc[-1], end_date='20260901')
        idx = pd.concat([idx, t.drop_duplicates('trade_date')], ignore_index=True).drop_duplicates('d').sort_values('d')
    ic = idx.set_index('d')['close'].astype(float)
    dates = idx['d'].tolist()
    dpos = {d: i for i, d in enumerate(dates)}
    # 各日regime(用截至前一日的ret20/ret5,避免用当日未来)
    r20 = ic.pct_change(20).shift(1)
    r5 = ic.pct_change(5).shift(1)

    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])
    sl = pd.read_parquet('data/store/stock_list.parquet')
    nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name'].astype(str)))

    recs = []
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
        for p in detect_patterns(df, 0.20):
            trig = p['early']
            if trig is None or not (START <= df.d.iloc[trig] <= END):
                continue
            if trig + 1 >= len(df):
                continue
            ed = df.d.iloc[trig + 1]
            op = float(df['open'].iloc[trig + 1])
            pc = float(df['close'].iloc[trig])
            lim = 0.195 if code[0] in ('3', '68') else 0.095
            if op / pc - 1 > lim or ed not in dpos or ed > '20260828':
                continue
            ii = dpos[ed]
            r = dict(code=code, name=nm.get(code, ''), ed=ed, regime=regime_of(r20.get(ed, 0), r5.get(ed, 0)),
                     turn=bool(r20.get(ed, 0) < 0 and r5.get(ed, 0) > 0))
            for H in HOLDS:
                ci = min(trig + 1 + H, len(df) - 1)
                px = float(df['close'].iloc[ci])
                jj = min(ii + H, len(dates) - 1)
                r[f'ex{H}'] = ((px / op - 1 - COST) - (ic[dates[jj]] / ic[ed] - 1)) * 100
            recs.append(r)
    t = pd.DataFrame(recs)
    if t.empty:
        print('无信号'); return
    print(f'双底(B臂20%门槛)×regime | 信号总数N={len(t)} | {START}~{END} | 超额=个股-上证(入场日→持有期末)')
    print('═' * 88)
    print(f"{'regime':<12}{'N':>5}{'10d中位':>9}{'20d均':>8}{'20d胜率':>8}{'40d均':>8}{'40d胜率':>8}")
    order = ['deep_down', 'down', 'shake', 'up', 'strong_up']
    for g in order:
        sub = t[t['regime'] == g]
        if sub.empty:
            print(f'{g:<12}{0:>5}'); continue
        e20, e40 = sub['ex20'], sub['ex40']
        print(f"{g:<12}{len(sub):>5}{sub['ex10'].median():>+8.2f}%{e20.mean():>+7.2f}%{(e20>0).mean()*100:>7.0f}%{e40.mean():>+7.2f}%{(e40>0).mean()*100:>7.0f}%")
    # 坏转好专项
    turn = t[t['turn']]
    noturn = t[~t['turn']]
    print('─' * 88)
    print(f"坏转好(ret20<0∧ret5>0): N={len(turn)} 20d均{turn['ex20'].mean() if len(turn) else 0:+.2f}%/胜{(turn['ex20']>0).mean()*100 if len(turn) else 0:.0f}% | 40d均{turn['ex40'].mean() if len(turn) else 0:+.2f}%")
    if len(turn) >= 10 and len(noturn) >= 10:
        tstat, p = st.ttest_ind(turn['ex20'].dropna(), noturn['ex20'].dropna(), equal_var=False)
        print(f"坏转好 vs 其他: 20d差{turn['ex20'].mean()-noturn['ex20'].mean():+.2f}% (t={tstat:.2f}, p={p:.4f})")
    # 全样本对照(原结论: 40d中位+1.49%/胜55%)
    print(f"全样本: 20d均{t['ex20'].mean():+.2f}%/胜{(t['ex20']>0).mean()*100:.0f}% | 40d中位{t['ex40'].median():+.2f}%/均{t['ex40'].mean():+.2f}%/胜{(t['ex40']>0).mean()*100:.0f}%")


if __name__ == '__main__':
    main()
