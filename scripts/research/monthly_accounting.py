#!/usr/bin/env python3
"""月度净值对账 — 2026-10-01立(P3-11,度量优先落地)。

口径诚实规则:
  · 两融账户的"投资者收益"=自有权益(total-leverage)的变动,不是总资产——杠杆放大的是β不是本事
  · era A(无杠杆期)self=total;era B(两融期)self=total-leverage
  · 快照稀疏(git锚定):逐月取当月最后快照;缺月报警示不插值伪造
  · 9/30后补一行"估值锚":持仓×9/30收盘的精确市值(现金未知→只报mv不报equity)
基准: 中证1000ETF(512100,本地qfq)月度收盘,同起点归一。
用法: PYTHONPATH=. python scripts/research/monthly_accounting.py
输出: reports/accounting/净值月报_YYYYMM.md + 净值vs基准.png(覆盖旧图)
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS']


def main():
    snap = pd.read_csv('reports/accounting/capital_snapshots.csv', parse_dates=['asof'])
    snap = snap.dropna(subset=['total']).sort_values('asof')
    snap['equity'] = snap.apply(
        lambda r: r['total'] - r['leverage'] if pd.notna(r['leverage']) and r['leverage'] > 0 else r['total'], axis=1)

    # 估值锚: 9/30持仓×收盘
    try:
        pf = pd.read_excel('config/portfolio.xlsx')
        est_mv = 0.0
        for _, r in pf.iterrows():
            c = str(r['code']).zfill(6)
            try:
                sh = int(r.get('shares', 0) or 0)
            except (ValueError, TypeError):
                continue
            if c == 'nan' or sh <= 0:
                continue
            k = pd.read_parquet(f'data/store/kline/{c}/D.parquet')
            dcol = [x for x in k.columns if 'date' in x.lower()][0]
            est_mv += sh * float(k.sort_values(dcol)['close'].iloc[-1])
        est_date = pd.to_datetime(str(k.sort_values(dcol)[dcol].iloc[-1]), format='mixed')
        snap = pd.concat([snap, pd.DataFrame([{'asof': est_date, 'total': np.nan,
                                               'equity': np.nan, 'source': '估值锚(仅mv)'}])],
                         ignore_index=True).sort_values('asof')
    except Exception as e:
        print(f'估值锚跳过: {e}')

    # 月度: 当月最后快照
    snap['ym'] = snap['asof'].dt.to_period('M')
    monthly = snap.groupby('ym').last().reset_index()

    # 基准: 512100月度
    bench_k = pd.read_parquet('data/store/kline/512100/D.parquet')
    dcol = [x for x in bench_k.columns if 'date' in x.lower()][0]
    bench_k = bench_k.sort_values(dcol)
    bench_k['ym'] = pd.to_datetime(bench_k[dcol], format='%Y%m%d').dt.to_period('M')
    bench_m = bench_k.groupby('ym')['close'].last()

    base_ym = monthly.dropna(subset=['equity']).iloc[0]['ym']
    base_eq = monthly.dropna(subset=['equity']).iloc[0]['equity']
    base_bm = bench_m.get(base_ym, np.nan)

    rows = []
    prev_covered = None
    all_ym = pd.period_range(monthly['ym'].min(), monthly['ym'].max(), freq='M')
    snap_ym_set = set(monthly['ym'])
    for ym in all_ym:
        if ym not in snap_ym_set:
            rows.append((str(ym), None, None, None, '⚠️缺快照', None, None))
            continue
        r = monthly[monthly['ym'] == ym].iloc[0]
        eq, bm = r['equity'], bench_m.get(ym, np.nan)
        if pd.isna(eq):
            rows.append((str(ym), r['source'], None, None, '', None, None))
            continue
        nav = eq / base_eq * 100
        bmi = bm / base_bm * 100 if pd.notna(bm) else np.nan
        rows.append((str(ym), r['source'], eq, nav, '', bmi,
                     nav - bmi if pd.notna(bmi) else np.nan))
    df = pd.DataFrame(rows, columns=['月份', '来源', '自有权益', '净值指数', '备注', '基准指数', '超额'])

    # 曲线(仅有效点)
    valid = df.dropna(subset=['净值指数'])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(range(len(valid)), valid['净值指数'], 'o-', label='自有权益净值', color='#C00000', lw=2)
    vb = valid.dropna(subset=['基准指数'])
    ax.plot([list(valid['月份']).index(m) for m in vb['月份']], vb['基准指数'], 's--',
            label='中证1000ETF', color='#1F4E79', lw=1.5)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid['月份'], rotation=45, fontsize=8)
    ax.axhline(100, color='gray', lw=0.8, ls=':')
    ax.set_title('自有权益净值 vs 中证1000 (基期=100)', fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/accounting/净值vs基准.png', dpi=130)

    # 落md
    ret = valid['净值指数'].iloc[-1] / 100 - 1
    bret = vb['基准指数'].iloc[-1] / 100 - 1 if len(vb) else np.nan
    peak = valid['净值指数'].cummax()
    mdd = (valid['净值指数'] / peak - 1).min()
    md = [f'# 净值月报 · 生成{pd.Timestamp.now().strftime("%Y-%m-%d")}\n',
          f'- 区间: {valid["月份"].iloc[0]} → {valid["月份"].iloc[-1]} | 自有权益{base_eq:,.0f} → {valid["自有权益"].iloc[-1]:,.0f}',
          f'- **净值收益 {ret:+.1%} vs 中证1000 {bret:+.1%}** | 超额 {ret-bret:+.1%} | 净值口径最大回撤 {mdd:.1%}',
          '- 口径: 自有权益=总资产-体外负债(负债不在券商系统,每快照须同记total与debt;借入/归还不算盈亏);缺月不插值\n',
          df.to_string(index=False), '',
          f'最新估值锚: 持仓市值≈{est_mv:,.0f}({est_date.date()}收盘×股数,现金未计入)']
    out = 'reports/accounting/净值月报_最新.md'
    open(out, 'w').write('\n'.join(md))
    print('\n'.join(md))
    print(f'\n已写 {out} + 净值vs基准.png')

if __name__ == '__main__':
    main()
