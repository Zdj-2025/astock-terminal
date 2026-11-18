#!/usr/bin/env python3
"""蓄势挤压×点火 回测(2026-08-24,用户提出:北方国际连续红柱+振幅压缩→竞价涨停)。

因子(因果,T日可知):
  蓄势 = MACD红柱连续≥18天 且 窗口内 close max/min < 1.12
点火(T日当日信号,窗口高点用T-1前数据):
  IGN_VOL   放量: vol_T/vol_ma20 > 1.5
  IGN_BREAK 突破: close_T > 压缩窗口最高close(T-1及以前)
  IGN_DRYUP 缩量后首放: min(vr[-5:-1])<0.6 且 vr_T>1.3
对照: 无压缩+同触发(隔离压缩增益) / 压缩无触发(基线) / 全市场抽样(市场基线)
前瞻: T收盘→T+5/T+10/T+20 收盘(未扣成本,结论再减~1%交易费)
除权: qfq回溯调整(|ret|>25%回溯缩放),防假跳空
统计: 均值/中位/胜率/N/t值, 分年稳定性
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

KDIR = 'data/store/kline'
START = '2021-01-01'   # 5年半样本,含牛熊
MIN_STREAK, MAX_AMP = 18, 1.12

def qfq(c):
    rets = np.diff(c, append=np.nan)
    cc = c.copy()
    for i in range(len(cc) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            cc[:i + 1] = cc[:i + 1] * (c[i + 1] / c[i])
    return cc

def run():
    cohorts = {k: {5: [], 10: [], 20: []} for k in
               ['C压缩无火', 'C压缩+放量', 'C压缩+突破', 'C压缩+量破双', 'C压缩+干柴烈火',
                'N无压+放量', 'N无压+突破', 'N无压+量破双', 'MKT市场基线']}
    yearly = {}  # (year, cohort) -> [fwd5...]
    n_files = 0
    for d in os.listdir(KDIR):
        if not d.isdigit() or len(d) != 6 or d[:2] in ('51', '15', '56', '58'):
            continue
        try:
            df = pd.read_parquet(f'{KDIR}/{d}/D.parquet').sort_values('date').reset_index(drop=True)
        except Exception:
            continue
        df = df[df['date'] >= START].reset_index(drop=True)
        if len(df) < 120:
            continue
        c = qfq(df['close'].astype(float).values)
        vcol = 'volume' if 'volume' in df.columns else [x for x in df.columns if 'vol' in x.lower()][0]
        v = df[vcol].values.astype(float)
        s = pd.Series(c)
        dif = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        hist = (dif - dif.ewm(span=9, adjust=False).mean()).values
        # 连续红柱天数(因果): 累计技巧
        red = (hist > 0).astype(int)
        streak = np.zeros(len(red), int)
        for i in range(len(red)):
            streak[i] = streak[i - 1] + 1 if red[i] else 0
        vol_ma20 = pd.Series(v).rolling(20).mean().values
        vr = v / vol_ma20
        fwd = {k: np.full(len(c), np.nan) for k in (5, 10, 20)}
        for k in (5, 10, 20):
            fwd[k][:-k] = c[k:] / c[:-k] - 1

        for T in range(60, len(c) - 20):
            if np.isnan(fwd[5][T]):
                continue
            st = streak[T]
            is_comp = False
            if st >= MIN_STREAK:
                w = c[T - st + 1: T + 1]
                amp = w.max() / w.min()
                if amp < MAX_AMP:
                    is_comp = True
                    whi_prev = c[T - st + 1: T].max()  # T-1及以前的窗口高点
                    ign_vol = (not np.isnan(vr[T])) and vr[T] > 1.5
                    ign_brk = c[T] > whi_prev
                    vr_prev4 = vr[max(0, T - 4):T]
                    ign_dry = (len(vr_prev4) and not np.isnan(vr_prev4).any()
                               and vr_prev4.min() < 0.6 and (not np.isnan(vr[T])) and vr[T] > 1.3)
                    if ign_vol and ign_brk:
                        key = 'C压缩+量破双'
                    elif ign_dry:
                        key = 'C压缩+干柴烈火'
                    elif ign_vol:
                        key = 'C压缩+放量'
                    elif ign_brk:
                        key = 'C压缩+突破'
                    else:
                        key = 'C压缩无火'
                    for k in (5, 10, 20):
                        cohorts[key][k].append(fwd[k][T])
                    yr = str(df['date'].iloc[T])[:4]
                    yearly.setdefault((yr, key), {5: [], 10: [], 20: []})[5].append(fwd[5][T])
            # 无压缩对照(有触发)
            if not is_comp and st >= 3 and not np.isnan(vr[T]):
                w = c[T - min(st, 20) + 1: T + 1]
                if vr[T] > 1.5 and c[T] > w[:-1].max():
                    key = 'N无压+量破双'
                    for k in (5, 10, 20):
                        cohorts[key][k].append(fwd[k][T])
            # 市场基线(每日1/12抽样)
            if T % 12 == 0:
                for k in (5, 10, 20):
                    cohorts['MKT市场基线'][k].append(fwd[k][T])
        n_files += 1
    print(f"样本股票数: {n_files}\n")

    from scipy import stats
    base5 = np.array(cohorts['MKT市场基线'][5]) * 100
    base10 = np.array(cohorts['MKT市场基线'][10]) * 100
    base20 = np.array(cohorts['MKT市场基线'][20]) * 100
    print(f"{'组':<12}{'N':>7} | {'5日超额':>8}{'t':>6}{'胜率':>6} | {'10日超额':>9}{'t':>6} | {'20日超额':>9}{'t':>6}")
    print('-' * 92)
    for key, dat in cohorts.items():
        if key == 'MKT市场基线':
            print(f"{key:<12}{len(base5):>7} | {base5.mean():>7.2f}%{'':>6}{(base5 > 0).mean():>6.0%} | "
                  f"{base10.mean():>8.2f}%{'':>6} | {base20.mean():>8.2f}%")
            continue
        for k, base in [(5, base5), (10, base10), (20, base20)]:
            pass
        r5 = np.array(dat[5]) * 100 - base5.mean()
        r10 = np.array(dat[10]) * 100 - base10.mean()
        r20 = np.array(dat[20]) * 100 - base20.mean()
        t5 = r5.mean() / (r5.std(ddof=1) / np.sqrt(len(r5))) if len(r5) > 30 else np.nan
        t10 = r10.mean() / (r10.std(ddof=1) / np.sqrt(len(r10))) if len(r10) > 30 else np.nan
        t20 = r20.mean() / (r20.std(ddof=1) / np.sqrt(len(r20))) if len(r20) > 30 else np.nan
        print(f"{key:<12}{len(dat[5]):>7} | {r5.mean():>+7.2f}%{t5:>6.1f}{(np.array(dat[5])>0).mean():>6.0%} | "
              f"{r10.mean():>+8.2f}%{t10:>6.1f} | {r20.mean():>+8.2f}%{t20:>6.1f}")

    # 分年稳定性(5日超额)
    print(f"\n--- 分年5日超额%(压缩+量破双 vs 压缩无火) ---")
    years = sorted({y for y, k in yearly if k == 'C压缩+量破双'})
    print(f"{'年份':<6}{'量破双N':>8}{'5日超额':>9} | {'无火N':>7}{'5日超额':>9}")
    for y in years:
        a = np.array(yearly.get((y, 'C压缩+量破双'), {5: []})[5]) * 100
        b = np.array(yearly.get((y, 'C压缩无火'), {5: []})[5]) * 100
        if len(a) < 5:
            continue
        as_ = f"{a.mean() - base5.mean():+.2f}" if len(a) else 'na'
        bs_ = f"{b.mean() - base5.mean():+.2f}" if len(b) else 'na'
        print(f"{y:<6}{len(a):>8}{as_:>9} | {len(b):>7}{bs_:>9}")

if __name__ == '__main__':
    run()
