#!/usr/bin/env python3
"""首板异动池scanner — "吃鱼身不吃鱼头"的大牛股响应系统(2026-11-17立项)

原理(九阳案例背书): 273%大牛股无法预测(7月标记时与49只平庸票无异),
但启动后第一次缩量回踩是合法买点(九阳10/14-15@17.5→+64%)。

两层输出:
  A. 今日新异动(首板/放量突破) — 建池监控
  B. 池内回踩到位 — 生成买点预案(回撤至首波高点的-12%±3%平台+缩量)

买点规则: 启动后回撤10-15%且量能缩至启动前水平 | 止损: 买价-8%或启动平台下沿
用法: cd stock && PYTHONPATH=. python scripts/screener/first_board_scanner.py
数据: data/store/kline (需当日更新)
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

K = 'data/store/kline'
POOL_FILE = 'data/store/first_board_pool.csv'   # 持久池
LIMIT_MAIN = 9.8   # 主板涨停阈值(创业板/科创20cm另算)
LIMIT_GEM = 19.5


def load(code):
    try:
        df = pd.read_parquet(f'{K}/{code}/D.parquet').sort_values('date')
    except Exception:
        return None
    df['date'] = pd.to_datetime(df['date'])
    for col in ['close', 'open', 'high', 'low', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.reset_index(drop=True)


def detect_new_sparks(df, limit_days=10):
    """近limit_days日内出现: 单日涨停 或 放量(≥3x)+涨≥7% 且此前60日无同等异动(首波)"""
    if len(df) < 80:
        return None
    c = df['close'].astype(float); v = df['volume'].astype(float)
    chg = c.pct_change() * 100
    vma60 = v.rolling(60).mean()
    lim = LIMIT_GEM if str(df['code'].iloc[0] if 'code' in df else '').startswith(('3', '68')) else LIMIT_MAIN
    # 简化: 按代码前缀判断涨跌幅制度 — 外部传入更稳, 这里用收盘价判断
    tail = df.tail(limit_days)
    for i in tail.index:
        vr = v.iloc[i] / vma60.iloc[i] if vma60.iloc[i] > 0 else 0
        big = chg.iloc[i] >= 9.8 or (chg.iloc[i] >= 7 and vr >= 3)
        if not big:
            continue
        # "首波": 此前60日无同级别异动
        w = df.loc[max(0, i-60):i-1]
        wc = c.loc[w.index]; wv = v.loc[w.index]
        wchg = (wc / wc.shift(1) - 1) * 100
        wvma = wv.rolling(60).mean()
        prior = ((wchg >= 9.8) | ((wchg >= 7) & (wv / wvma >= 3))).any()
        if not prior:
            peak_after = c.iloc[i:].max()
            peak_px = float(peak_after)
            return dict(spark_date=str(df['date'].iloc[i].date()),
                        spark_close=float(c.iloc[i]), vr=float(vr),
                        peak=float(peak_after), peak_date=str(df['date'].iloc[int(c.iloc[i:].idxmax())].date()),
                        last=float(c.iloc[-1]), last_date=str(df['date'].iloc[-1].date()))
    return None


def pullback_level(info):
    """回踩买价: 首波峰值的-12%(区间-10~-15%)"""
    return dict(zone_lo=round(info['peak'] * 0.85, 2), zone_hi=round(info['peak'] * 0.90, 2),
                stop=round(info['peak'] * 0.80, 2))


def main():
    codes = sorted([d for d in os.listdir(K) if os.path.exists(f'{K}/{d}/D.parquet') and d.startswith(('0', '3', '6'))])
    print(f'=== 首板异动池scanner | {datetime.now().strftime("%Y-%m-%d")} | 扫描{len(codes)}只 ===\n')
    pool = []
    for ci, code in enumerate(codes):
        df = load(code)
        if df is None:
            continue
        df['code'] = code
        info = detect_new_sparks(df)
        if info:
            info['code'] = code
            pool.append(info)
        if (ci + 1) % 1000 == 0:
            print(f'  [{ci+1}/{len(codes)}] 池{len(pool)}只', flush=True)

    P = pd.DataFrame(pool)
    if not len(P):
        print('无新异动')
        return
    P = P.sort_values('spark_date', ascending=False)
    # 名称
    try:
        from scripts.core.tushare_client import TushareClient
        sl = TushareClient().get_stock_list()
        nm = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name']))
        P.insert(1, 'name', P['code'].map(nm).fillna(''))
    except Exception:
        P.insert(1, 'name', '')
    # 回踩区和状态
    zones = P.apply(pullback_level, axis=1, result_type='expand')
    P = pd.concat([P, zones], axis=1)
    P['dd_from_peak'] = ((P['last'] / P['peak'] - 1) * 100).round(1)
    P['state'] = np.where(P['last'] <= P['zone_hi'], '🎯回踩到位', np.where(P['dd_from_peak'] > -5, '⏳高位', '⏬回落中'))
    P.to_csv(POOL_FILE, index=False)

    hit = P[P['state'] == '🎯回踩到位']
    print(f'\n■ 池内共{len(P)}只(近10日首波异动)')
    print(f'\n■ 🎯回踩到位(买点区) {len(hit)}只:')
    if len(hit):
        print(hit[hit.columns[:12]].head(12).to_string(index=False))
    print(f'\n■ 高位/回落中(等回踩) 前12只:')
    cols = ['code', 'name', 'spark_date', 'spark_close', 'peak', 'last', 'dd_from_peak', 'zone_lo', 'zone_hi', 'stop']
    print(P[P['state'] != '🎯回踩到位'][cols].head(12).to_string(index=False))
    print(f'\n池已落盘: {POOL_FILE}')
    print('纪律: 回踩区内还须缩量(量比<1.5)才可入; 止损=zone下沿再-5%; 不追任何涨停')


if __name__ == '__main__':
    main()
