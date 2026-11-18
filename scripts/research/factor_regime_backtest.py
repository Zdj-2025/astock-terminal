#!/usr/bin/env python3
"""质量卡全维度回测(分大盘环境) — 质押/增长/市值/PB/PE在上升/下降/震荡期的效果。

每个维度分3档大盘环境(上证20日涨跌): 上升(>5%) / 震荡(-5%~5%) / 下降(<-5%)
看每个维度在每个环境下的分组收益差异 → 找出"全环境稳定有效"的维度。

用法: cd stock && PYTHONPATH=. python scripts/research/factor_regime_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
import tushare as ts

STORE = Path(__file__).resolve().parent.parent.parent / 'data/store/kline'
COST = 0.003
ORDER = 5
pro = ts.pro_api()


def compute_signals(df):
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150: return None
    close = df['close'].astype(float).ffill()
    low = df['low'].astype(float).ffill() if 'low' in df.columns else close
    op = df['open'].astype(float).ffill() if 'open' in df.columns else close
    vol = df['volume'].astype(float).ffill() if 'volume' in df.columns else pd.Series(1, index=close)
    vol_ma = vol.rolling(20).mean()
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)
    out = pd.DataFrame({'date': df['date'], 'sig': False})
    tr_idx = argrelextrema(low.values, np.less, order=ORDER)[0]
    h_arr = hist.values; lo_arr = low.values
    cl_arr = close.values; op_arr = op.values; vol_arr = vol.values; vm_arr = vol_ma.values
    c_sig = out.columns.get_loc('sig')
    for k in range(1, len(tr_idx)):
        t2 = tr_idx[k]; t1 = tr_idx[k - 1]
        if lo_arr[t2] < lo_arr[t1] and h_arr[t2] > h_arr[t1]:
            for j in range(t2 + 1, min(t2 + ORDER + 1, len(out))):
                if cl_arr[j] > op_arr[j] and vm_arr[j] > 0 and vol_arr[j] > vm_arr[j]:
                    out.iat[j, c_sig] = True; break
    out['ret10'] = (close.shift(-10) / close - 1).values
    out['date_str'] = df['date'].astype(str).str.replace('-', '')
    return out


def load_fundamentals(days=120):
    """取最近N交易日全市场 PE/PB/市值/换手(daily_basic) + 大盘环境(上证)。"""
    ref = pd.read_parquet(STORE / '000001SH' / 'D.parquet')
    for c in ['date', 'day', 'trade_date']:
        if c in ref.columns: ref = ref.rename(columns={c: 'date'})
    ref['date'] = pd.to_datetime(ref['date']); ref = ref.sort_values('date').reset_index(drop=True)
    ref['close'] = pd.to_numeric(ref['close'], errors='coerce')
    ref['idx_ret20'] = ref['close'].pct_change(20)
    ref['date_str'] = ref['date'].dt.strftime('%Y%m%d')
    idx_env = ref[['date_str', 'idx_ret20']].dropna()
    dates = sorted(idx_env['date_str'].unique())[-days:]
    # 大盘环境标签: 上升>5% / 震荡-5~5% / 下降<-5%
    idx_env = idx_env[idx_env['date_str'].isin(dates)].copy()
    idx_env['regime'] = idx_env['idx_ret20'].apply(
        lambda x: '上升' if x > 0.05 else ('下降' if x < -0.05 else '震荡'))
    # daily_basic
    fund_map = {}
    for i, d in enumerate(dates):
        try:
            db = pro.daily_basic(trade_date=d, fields='ts_code,pe_ttm,pb,circ_mv,turnover_rate')
            fund_map[d] = {r['ts_code'][:6]: {'pe': r['pe_ttm'], 'pb': r['pb'], 'mv': r['circ_mv']/1e4 if r['circ_mv'] else None}
                           for _, r in db.iterrows()}
        except Exception: pass
        if (i + 1) % 20 == 0: print(f"  [基本面 {i+1}/{len(dates)}]")
        time.sleep(0.12)
    return fund_map, idx_env


def analyze_factor(snap, col, bins, labels, regime_col='regime'):
    """对某因子分环境×分组统计。"""
    snap['grp'] = pd.cut(snap[col].astype(float), bins=bins, labels=labels)
    print(f"\n{'='*70}")
    print(f"因子: {col}")
    print(f"{'组':<10}{'环境':<6}{'N':>6}{'ret净':>8}{'夏普':>6} | {'全环境':>6}{'N':>6}{'ret净':>8}{'夏普':>6}")
    print('-' * 70)
    for grp in labels:
        row_all = snap[snap.grp == grp]
        parts = []
        for regime in ['上升', '震荡', '下降']:
            g = row_all[row_all[regime_col] == regime]
            n = len(g)
            if n < 30: parts.append(f"  {regime}:N={n}不足"); continue
            r = g['ret10'] - COST
            sh = r.mean() / r.std() * np.sqrt(252 / 10) if r.std() > 0 else 0
            parts.append(f"  {regime}:净{r.mean()*100:+.1f}% 夏普{sh:.2f} N={n}")
        # 全环境
        n_all = len(row_all)
        if n_all >= 30:
            r_all = row_all['ret10'] - COST
            sh_all = r_all.mean() / r_all.std() * np.sqrt(252 / 10) if r_all.std() > 0 else 0
            all_str = f"全部:净{r_all.mean()*100:+.1f}% 夏普{sh_all:.2f} N={n_all}"
        else:
            all_str = f"全部:N={n_all}不足"
        print(f"{grp:<10}")
        for p in parts: print(f"  {p}")
        print(f"  {all_str}")


def main():
    print("质量卡全维度回测(分大盘环境) | 上升/震荡/下降 × 各因子分组")
    fund_map, idx_env = load_fundamentals(250)
    regimes = dict(zip(idx_env['date_str'], idx_env['regime']))
    print(f"  环境分布: {dict(idx_env['regime'].value_counts())}")

    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
    print(f"扫描 {len(codes)} 只...")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = compute_signals(pd.read_parquet(STORE / c / 'D.parquet'))
            if s is not None:
                s = s[s['sig']].copy(); s['code'] = c; snaps.append(s)
        except Exception: pass
        if (i + 1) % 1000 == 0: print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True).dropna(subset=['ret10'])
    # join 基本面 + 环境
    snap['pe'] = snap.apply(lambda r: fund_map.get(r['date_str'], {}).get(r['code'], {}).get('pe', np.nan), axis=1)
    snap['pb'] = snap.apply(lambda r: fund_map.get(r['date_str'], {}).get(r['code'], {}).get('pb', np.nan), axis=1)
    snap['mv'] = snap.apply(lambda r: fund_map.get(r['date_str'], {}).get(r['code'], {}).get('mv', np.nan), axis=1)
    snap['regime'] = snap['date_str'].map(regimes)
    snap = snap.dropna(subset=['pe', 'regime'])
    print(f"信号(有基本面+环境): {len(snap):,}\n")

    # 逐维度分析
    analyze_factor(snap, 'pe', [-9999, 0, 30, 60, 100, 9999], ['亏损', 'PE<30', '30-60', '60-100', '>100'])
    analyze_factor(snap, 'pb', [-1, 2, 5, 10, 9999], ['PB<2', '2-5', '5-10', '>10'])
    analyze_factor(snap, 'mv', [0, 50, 100, 300, 50000], ['<50亿', '50-100亿', '100-300亿', '>300亿'])
    # 市值分组也用更细分(不全给满分,看各档差异)
    analyze_factor(snap, 'mv', [0, 30, 50, 100, 300, 50000], ['<30亿', '30-50', '50-100', '100-300', '>300亿'])

    # 质押(需要单独取,这里跳过——太慢)
    print(f"\n{'='*70}")
    print("质押维度: 需逐只取akshare(慢),本次跳过。建议后续单独回测。")
    print(f"\n{'='*70}")
    print("总结: 看哪个维度在上升/震荡/下降三环境下都稳定有效(夏普一致高)。")
    print("      只在某一环境有效的→不稳定(不可靠)。全环境无效的→从质量卡去掉。")


if __name__ == '__main__':
    main()
