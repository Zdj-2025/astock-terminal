#!/usr/bin/env python3
"""二阶导因子全面重测 — 用正确因果实现(core/derivatives.py)替代历史错误版本。

2026-08-17 P0-1C连锁重测。背景: 原premarket实现三重错误(pos语义/奇数阶符号/
卷积方向), "因果IC=0.029"结论作废。本脚本三件事:
1. 新正确f2的预测力: 逐日截面Spearman IC(未来1/3/5/10日) + 分块bootstrap CI
2. defensive_scanner标签检验: 🚀主升(f1>0,f2>0)/⚠️减速(f1>0,f2<0)/🔄拐点(f1<0,f2>0)/🔴加速跌(f1<0,f2<0)
   各组未来收益 vs 全样本基准, t检验
3. 新旧实现IC对照: 同数据跑旧错误版, 量化"错误实现误导了多少"

口径: close-to-close(学术口径,注明未含涨停约束); 剔除ST/停牌(volume=0)/NaN导数期。
样本: data/store/kline全市场, 2024-01以来。
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
sys.path.insert(0, '.')
from scripts.core.derivatives import causal_savgol, SG_W, SG_P

START = '2024-01-01'
FWD = [1, 3, 5, 10]


def old_leaky_f2(close):
    """旧错误实现(premarket原版pos=W-1 + c[::-1]卷积) — 仅用于对照量化误导。"""
    from scipy.signal import savgol_coeffs
    ln_p = np.log(np.asarray(close, dtype=float))
    c2 = savgol_coeffs(SG_W, SG_P, deriv=2, pos=SG_W - 1)
    y = np.convolve(ln_p, c2[::-1], 'valid') * 10000
    out = np.full(len(ln_p), np.nan)
    out[SG_W - 1:] = y
    return out


def block_bootstrap_ic(ics, n_boot=1000, block=20, seed=7):
    """IC时序分块bootstrap 95%CI(处理自相关)。"""
    rng = np.random.default_rng(seed)
    ics = np.asarray(ics); ics = ics[~np.isnan(ics)]
    n = len(ics)
    means = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n - block)
            idx.extend(range(s, min(s + block, n)))
        means.append(ics[np.array(idx[:n])].mean())
    return np.percentile(means, [2.5, 97.5])


def main():
    sl = pd.read_parquet('data/store/stock_list.parquet').set_index('symbol')
    st_codes = {c for c in sl.index if 'ST' in str(sl.loc[c, 'name']).upper()}
    kdir = 'data/store/kline'
    codes = [d for d in os.listdir(kdir) if d.isdigit() and len(d) == 6
             and os.path.isfile(f'{kdir}/{d}/D.parquet')]
    print(f'样本: {len(codes)}只 | 导数窗口W={SG_W} | 起点{START}')

    recs = []
    for k, code in enumerate(codes):
        try:
            df = pd.read_parquet(f'{kdir}/{code}/D.parquet')
        except Exception:
            continue
        dcol = [c for c in df.columns if 'date' in c.lower()]
        if not dcol:
            continue
        df = df.sort_values(dcol[0])
        df = df[df[dcol[0]].astype(str) >= START]
        if len(df) < SG_W + 15:
            continue
        c = df['close'].astype(float).values
        v = df['volume'].astype(float).values
        dates = (df[dcol[0]].astype(str).str[:10].str.replace('-', '')).values  # 统一YYYYMMDD
        f1 = causal_savgol(np.log(c), SG_W, SG_P, 1) * 100
        f2 = causal_savgol(np.log(c), SG_W, SG_P, 2) * 10000
        f2_old = old_leaky_f2(c)
        fwd = {h: np.full(len(c), np.nan) for h in FWD}
        for h in FWD:
            fwd[h][:-(h)] = (c[h:] / c[:-h] - 1) * 100
        ok = ~np.isnan(f2) & ~np.isnan(f2_old)
        ok &= v > 0                      # 剔停牌
        if code in st_codes:
            continue                     # 剔ST
        for i in np.where(ok)[0]:
            recs.append((code, dates[i], f1[i], f2[i], f2_old[i],
                         *(fwd[h][i] for h in FWD)))
        if (k + 1) % 1000 == 0:
            print(f'  进度 {k+1}/{len(codes)} 行累计{len(recs)}', flush=True)

    E = pd.DataFrame(recs, columns=['code', 'date', 'f1', 'f2', 'f2_old',
                                    'r1', 'r3', 'r5', 'r10'])
    print(f'\n总观测: {len(E):,} | 日期 {E.date.min()} → {E.date.max()}')

    # ==== 1. IC分析(逐日截面Spearman) ===
    print('\n' + '=' * 72)
    print('【1】IC分析: f2(正确版) vs f2_old(错误版) 逐日截面Spearman IC')
    print('=' * 72)
    print(f"{'因子':<10}{'IC均值':>9}{'IC_std':>8}{'ICIR':>7}{'bootstrap95%CI':>22}{'t':>8}")
    for col in ['f2', 'f2_old']:
        ics = E.groupby('date').apply(
            lambda g: stats.spearmanr(g[col], g['r5'])[0] if len(g) > 30 else np.nan,
            include_groups=False).dropna()
        t = ics.mean() / (ics.std() / np.sqrt(len(ics)))
        lo, hi = block_bootstrap_ic(ics.values)
        print(f"{col:<10}{ics.mean():>+9.4f}{ics.std():>8.4f}"
              f"{ics.mean()/ics.std():>7.2f}  [{lo:+.4f},{hi:+.4f}]{t:>8.2f}")
    for h in [1, 3, 10]:
        ics = E.groupby('date').apply(
            lambda g: stats.spearmanr(g['f2'], g[f'r{h}'])[0] if len(g) > 30 else np.nan,
            include_groups=False).dropna()
        t = ics.mean() / (ics.std() / np.sqrt(len(ics)))
        lo, hi = block_bootstrap_ic(ics.values)
        print(f"f2→r{h:<6}{ics.mean():>+9.4f}{ics.std():>8.4f}"
              f"{ics.mean()/ics.std():>7.2f}  [{lo:+.4f},{hi:+.4f}]{t:>8.2f}")

    # ==== 2. 标签事件检验(defensive_scanner四象限) ===
    print('\n' + '=' * 72)
    print('【2】四象限标签: 未来收益 vs 全样本基准 (defensive_scanner标签基础)')
    print('=' * 72)
    E['label'] = np.select(
        [ (E.f1 > 0) & (E.f2 > 0), (E.f1 > 0) & (E.f2 < 0),
          (E.f1 < 0) & (E.f2 > 0), (E.f1 < 0) & (E.f2 < 0) ],
        ['🚀主升', '⚠️减速', '🔄拐点', '🔴加速跌'], default='?')
    base = E[['r1', 'r3', 'r5', 'r10']].mean()
    print(f"全样本基准: r5={base['r5']:+.3f}%  N={len(E):,}\n")
    print(f"{'标签':<8}{'N':>10}{'占比':>7}{'r1':>8}{'r3':>8}{'r5':>8}{'r10':>8}{'r5超额':>8}{'胜率5d':>8}{'t(超额)':>9}")
    for lab in ['🚀主升', '⚠️减速', '🔄拐点', '🔴加速跌']:
        g = E[E.label == lab]
        if len(g) < 100:
            continue
        ex = g['r5'] - base['r5']
        t = ex.mean() / (ex.std() / np.sqrt(len(ex)))
        print(f"{lab:<8}{len(g):>10,}{len(g)/len(E)*100:>6.1f}%"
              f"{g['r1'].mean():>+8.2f}{g['r3'].mean():>+8.2f}{g['r5'].mean():>+8.2f}"
              f"{g['r10'].mean():>+8.2f}{ex.mean():>+8.2f}{(g['r5']>0).mean()*100:>7.0f}%{t:>9.2f}")

    # 分年度稳定性(🚀主升 vs ⚠️减速的r5差)
    print('\n🚀主升−⚠️减速 的r5差(分年,看稳定性):')
    piv = E[E.label.isin(['🚀主升', '⚠️减速'])].copy()
    piv['year'] = piv['date'].str[:4]
    for y, g in piv.groupby('year'):
        m = g.groupby('label')['r5'].mean()
        d = m['🚀主升'] - m['⚠️减速']
        print(f'  {y}: 🚀{m["🚀主升"]:+.3f}% − ⚠️{m["⚠️减速"]:+.3f}% = {d:+.3f}%')
    # 近3月时效
    print('\n近3月时效(2026-05后):')
    g = E[E.date >= '20260501']
    mb = g['r5'].mean()
    for lab in ['🚀主升', '⚠️减速', '🔄拐点', '🔴加速跌']:
        s = g[g.label == lab]
        if len(s) < 50:
            continue
        print(f'  {lab}: N={len(s):,} r5={s.r5.mean():+.3f}% 超额{s.r5.mean()-mb:+.3f}% 胜率{(s.r5>0).mean()*100:.0f}%')

    # ==== 3. f2五分位分组 ===
    print('\n' + '=' * 72)
    print('【3】f2五分位分组(全样本pooled, r5)')
    print('=' * 72)
    try:
        E['q'] = pd.qcut(E['f2'], 5, labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高'], duplicates='drop')
        for q, g in E.groupby('q', observed=True):
            print(f"  {q}: N={len(g):>9,} r5={g['r5'].mean():+.3f}% 胜率{(g['r5']>0).mean()*100:.0f}%")
    except Exception as e:
        print('  qcut失败:', e)

    E.to_parquet('reports/reverify_derivative_v2_events.parquet', index=False)
    print('\n事件明细→reports/reverify_derivative_v2_events.parquet')


if __name__ == '__main__':
    main()
