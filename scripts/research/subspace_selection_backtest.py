#!/usr/bin/env python3
"""第4层筛选回测 — 在底背离信号子空间里,比较不同筛选方法(PE/质量/PEG/随机)的选出收益。
直接回答: PE<60会不会筛掉好科技股? 从🥇里选用哪种排序最好?

方法: 全市场底背离confirmed信号 → join PE(最近90交易日daily_basic) → 按PE/质量分组 → 比较ret10扣成本。
用法: cd stock && PYTHONPATH=. python scripts/research/subspace_selection_backtest.py
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
    """底背离confirmed(量能阳线确认)信号 + ret10。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 150:
        return None
    close = df['close'].astype(float).ffill()
    high = df['high'].astype(float).ffill() if 'high' in df.columns else close
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
                    out.iat[j, c_sig] = True
                    break
    out['ret10'] = (close.shift(-10) / close - 1).values
    out['date_str'] = df['date'].astype(str).str.replace('-', '')
    return out


def load_pe_panel(days=90):
    """取最近N个交易日的全市场PE(daily_basic按trade_date)。返回 {date_str: {code: pe_ttm}}。"""
    # 取交易日列表
    ref = pd.read_parquet(STORE / '000001SH' / 'D.parquet')
    for c in ['date', 'day', 'trade_date']:
        if c in ref.columns:
            ref = ref.rename(columns={c: 'date'})
    ref['date'] = pd.to_datetime(ref['date'])
    dates = sorted(ref['date'].dt.strftime('%Y%m%d').unique())[-days:]
    pe_map = {}
    for i, d in enumerate(dates):
        try:
            db = pro.daily_basic(trade_date=d, fields='ts_code,pe_ttm')
            pe_map[d] = {r['ts_code'][:6]: r['pe_ttm'] for _, r in db.iterrows()}
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  [PE {i+1}/{len(dates)}]")
        time.sleep(0.12)
    return pe_map


def main():
    print("第4层筛选回测 | 底背离confirmed信号 × PE分组 → 比较不同筛选方法")
    print("加载PE数据(最近90交易日)...")
    pe_map = load_pe_panel(90)
    print(f"  PE覆盖 {len(pe_map)} 个交易日")

    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
    print(f"扫描 {len(codes)} 只...")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = compute_signals(pd.read_parquet(STORE / c / 'D.parquet'))
            if s is not None:
                s = s[s['sig']].copy()
                s['code'] = c
                snaps.append(s)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")
    snap = pd.concat(snaps, ignore_index=True).dropna(subset=['ret10'])
    # join PE
    snap['pe'] = snap.apply(lambda r: pe_map.get(r['date_str'], {}).get(r['code'], np.nan), axis=1)
    snap = snap.dropna(subset=['pe'])
    print(f"信号(有PE): {len(snap):,}\n")

    # PE分组
    snap['pe_group'] = pd.cut(snap['pe'], bins=[-9999, 0, 20, 40, 60, 100, 9999],
                               labels=['亏损', 'PE<20', '20-40', '40-60', '60-100', '>100'])
    print(f"{'PE组':<10}{'N':>8}{'ret10净':>9}{'胜率':>6}{'夏普':>7}")
    print('-' * 45)
    for grp in ['亏损', 'PE<20', '20-40', '40-60', '60-100', '>100']:
        g = snap[snap.pe_group == grp]
        n = len(g)
        if n < 50:
            print(f"{grp:<10}{n:>8,}  (样本不足)"); continue
        r = g['ret10'] - COST
        wr = (g['ret10'] > COST).mean() * 100
        sh = r.mean() / r.std() * np.sqrt(252 / 10) if r.std() > 0 else 0
        print(f"{grp:<10}{n:>8,}{r.mean()*100:>+8.2f}%{wr:>5.0f}%{sh:>7.2f}")

    # 关键t检验: PE<60 vs PE>=60 (当前质量卡的阈值)
    low_pe = snap[(snap.pe > 0) & (snap.pe < 60)]['ret10'] - COST
    high_pe = snap[snap.pe >= 60]['ret10'] - COST
    if len(low_pe) > 100 and len(high_pe) > 100:
        t, p = stats.ttest_ind(low_pe, high_pe, equal_var=False)
        print(f"\n[t检验 PE<60 vs PE>=60]")
        print(f"  低PE净{low_pe.mean()*100:+.2f}% vs 高PE净{high_pe.mean()*100:+.2f}% t={t:+.2f} p={p:.4f}")
        verdict = '✅低PE显著更优(当前PE<60阈值合理)' if (p < 0.05 and low_pe.mean() > high_pe.mean()) else \
                  ('⚠️高PE反而更好(PE<60阈值筛掉了好票!)' if (p < 0.05 and high_pe.mean() > low_pe.mean()) else '—无显著差异(PE筛选无效)')
        print(f"  {verdict}")

    # 模拟"从TopN选"比较
    print(f"\n[模拟: 从每天信号里用不同方法选Top5的收益]")
    # 按日期分组,每组按不同方法选Top5
    for method, ascending, label in [('pe', True, 'PE最低'), ('pe', False, 'PE最高')]:
        selected = snap.groupby('date_str').apply(
            lambda g: g.nsmallest(5, method) if ascending else g.nlargest(5, method), include_groups=False
        ).reset_index(drop=True)
        if len(selected) > 100:
            r = selected['ret10'] - COST
            print(f"  {label}Top5: 净{r.mean()*100:+.2f}% 胜率{(selected['ret10']>COST).mean()*100:.0f}% 夏普{r.mean()/r.std()*np.sqrt(252/10):.2f} N={len(selected)}")


if __name__ == '__main__':
    main()
