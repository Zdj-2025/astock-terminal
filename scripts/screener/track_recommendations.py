#!/usr/bin/env python3
"""推荐跟踪器 — 推荐闭环(落盘→跟踪→复盘→找问题)的【跟踪】环节。

机制(用户2026-07-09要求"每次推荐都要落盘,跟踪,复盘,找问题"):
  1. 落盘: 每次推荐追加一行到 reports/recommendations.csv
     字段: code,name,rec_date,reason,mainline,regime,buy_point,stop_loss,note
  2. 跟踪: 每日收盘后跑本脚本, 实算每只"推荐日→最新"盈亏 + 止损触发 + 按regime分组
  3. 复盘: 推荐≥5天或批量后, 汇总胜率/规律, 写 reports/recommendation_log.md 复盘#N
  4. 找问题: 诚实记对错(选股/择时/执行), 反哺 scanner + memory

核心规律(复盘#5): 大盘regime是第一维度, 下跌周期推买入=系统性亏损。

用法: cd stock && PYTHONPATH=. python scripts/screener/track_recommendations.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd

STORE = PROJECT_ROOT / 'data/store/kline'
CSV = PROJECT_ROOT / 'reports/recommendations.csv'

from scripts.accounting.rec_outcome import compute_outcomes  # R0: +5/+10/+20 outcome 自动写回


def refresh_outcomes():
    """把已到期的推荐 outcome(+5/+10/+20日)写回 recommendations.csv(幂等)."""
    recs = pd.read_csv(CSV)
    before = recs.copy()
    for col in ['rec_close', 'pnl5', 'pnl10', 'pnl20', 'buy_hit']:
        if col not in recs.columns:
            recs[col] = ''
    recs = compute_outcomes(recs)
    if not recs.equals(before):
        recs.to_csv(CSV, index=False)


def load(code):
    p = STORE / f'{code}/D.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df


def main():
    if not CSV.exists():
        print(f"❌ {CSV} 不存在。先落盘推荐。")
        return
    recs = pd.read_csv(CSV)
    recs['code'] = recs['code'].astype(str).str.zfill(6)
    try:
        refresh_outcomes()   # 先刷新 outcome 列, 再展示
        recs = pd.read_csv(CSV)
        recs['code'] = recs['code'].astype(str).str.zfill(6)
    except Exception as e:
        print(f'(outcome刷新跳过: {e})')

    # 大盘环境
    idx = load('000001SH')
    reg = '?'
    if idx is not None:
        ret20 = idx['close'].iloc[-1] / idx['close'].iloc[-21] - 1
        reg = '上升' if ret20 > 0.05 else ('下降' if ret20 < -0.05 else '震荡')
        print(f"📊 推荐跟踪 | {len(recs)}只推荐 | 上证{idx['close'].iloc[-1]:.0f}({reg},20日{ret20*100:+.1f}%) | 数据{idx['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"{'代码':<7}{'名称':<10}{'推荐日':>11}{'持有':>5}{'推荐价':>8}{'现价':>8}{'收益':>8}  止损/主线/regime")
    print('-' * 92)

    rows = []
    for _, r in recs.iterrows():
        df = load(r['code'])
        if df is None or len(df) < 5:
            print(f"{r['code']:<7}{r['name'][:9]:<10} 无数据"); continue
        rd = pd.to_datetime(r['rec_date'])
        rec_row = df[df['date'] >= rd]
        if rec_row.empty:
            rec_row = df.tail(1)
        rec_close = rec_row.iloc[0]['close']
        cur = df['close'].iloc[-1]
        ret = (cur / rec_close - 1) * 100
        days = (df['date'].iloc[-1] - rec_row.iloc[0]['date']).days
        # 止损触发
        sl_hit = ''
        try:
            sl = float(r['stop_loss'])
            if sl > 0 and cur < sl:
                sl_hit = f'⚠️破止损{sl:.2f}'
        except Exception:
            pass
        rows.append({**r.to_dict(), 'rec_close': rec_close, 'cur': cur, 'ret': ret, 'days': days, 'sl_hit': sl_hit})

    out = pd.DataFrame(rows).sort_values('ret', ascending=False)
    for _, r in out.iterrows():
        ret_s = f"{r['ret']:+.1f}%"
        days_s = f"{r['days']:>3}d"
        flag = r['sl_hit']
        print(f"{r['code']:<7}{r['name'][:9]:<10}{str(r['rec_date'])[:10]:>11}{days_s:>5}{r['rec_close']:>8.2f}{r['cur']:>8.2f}{ret_s:>8}  {flag} {r['mainline']}/{r['regime']}")

    # 汇总 + 按regime分组
    valid = out[out['ret'].notna()]
    if len(valid) > 0:
        wins = (valid['ret'] > 0).sum()
        print(f"\n=== 汇总 ===")
        print(f"推荐{len(valid)}只: 盈{wins} 亏{len(valid)-wins} | 平均{valid['ret'].mean():+.1f}% | 中位{valid['ret'].median():+.1f}%")
        for regime, g in valid.groupby('regime'):
            print(f"  {regime}期推荐{len(g)}只: 平均{g['ret'].mean():+.1f}% | 盈{(g['ret']>0).sum()}")
        sl_trig = out[out['sl_hit'] != '']
        if len(sl_trig) > 0:
            print(f"\n🚨 止损触发{len(sl_trig)}只: {list(sl_trig['name'])}")
        print(f"\n💡 规律: 看regime分组收益差→大盘环境第一维度(复盘#5)。推荐≥5天或批量后写复盘#N到 recommendation_log.md")


if __name__ == '__main__':
    main()
