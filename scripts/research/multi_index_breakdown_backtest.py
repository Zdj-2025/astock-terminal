#!/usr/bin/env python3
"""多指数同步破前低回测 — 验证"多指数同步破位=真破位/分化=假破位反弹"。

来源: 微信公众号波浪派作者体系(2026-07)。作者判"真破位"靠7大指数是否同步破
      6月低点——"7大指数3个破=危险, 5个破=大概率全破; 仅1-2个破=分化=假破位=反弹"。
      本回测统计验证此主张: 多指数同步破前低, 是否真能预测大盘后续下跌, 且优于单上证。

信号(严格无未来):
  破前低 broken_t: close_t < min(close_{t-N..t-1}), N=20(月度低点)。
                  rolling(N).min() 在t处含t, .shift(1)后变成[t-N..t-1]不含t → 无未来。
  sync_t: 当日7大指数中破前低的个数/7 (0~1)。

7大指数: 上证综指/深证成指/创业板指/上证50/沪深300/中证500/中证1000。

回测:
  - 对每个交易日t算sync_t, 上证forward_ret_h(h=5/10/20)
  - 按sync分5档(0破/1-2破/3-4破/5-6破/7破)看forward_ret
  - Spearman相关(sync, fwd): 显著负=同步越高后续越跌=有效
  - 对照: 单上证破前低(broken_SH=1/0)的预测力, 比信息增量

判读:
  - sync越高→forward越负且单调(p<0.05) → 有效, "同步破位"是真破位信号, 该减仓等深跌
  - 低sync(0-2破)→forward为正 → 假破位反弹, 该左侧买
  - 无单调/p不显著 → 证伪(多指数无额外信息, 作者主张不成立)

用法: cd stock && PYTHONPATH=. python scripts/research/multi_index_breakdown_backtest.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

STORE = PROJECT_ROOT / 'data/store/kline'
N_LOW = 20  # 前低窗口(月度低点)
HOLDS = [5, 10, 20]

INDICES = {
    '000001SH': '上证综指',
    '399001SZ': '深证成指',
    '399006SZ': '创业板指',
    '000016SH': '上证50',
    '000300SH': '沪深300',
    '000905SH': '中证500',
    '000852SH': '中证1000',
}
SH = '000001SH'


def load_index(code):
    p = STORE / code / 'D.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    dc = 'date' if 'date' in df.columns else 'trade_date'
    df[dc] = pd.to_datetime(df[dc])
    df = df.sort_values(dc).reset_index(drop=True)
    return df[[dc, 'close']].rename(columns={dc: 'date', 'close': code})


def broken_series(close: pd.Series, n: int = N_LOW) -> pd.Series:
    """close_t < min(close_{t-n..t-1})。shift(1)确保不含t → 无未来。"""
    prev_min = close.rolling(n, min_periods=n).min().shift(1)
    return (close < prev_min).astype(int)


def main():
    # ── 1. 加载并对齐7大指数 ──
    print(f"📊 多指数同步破前低回测 | {len(INDICES)}大指数 | 前低窗口N={N_LOW}日\n")
    panels = []
    have = {}
    for code, name in INDICES.items():
        df = load_index(code)
        if df is None:
            print(f"  ⚠️ 缺{code}({name}), 跳过")
            continue
        have[code] = name
        panels.append(df)
        print(f"  {code} {name}: {len(df)}行 {df['date'].min().date()}~{df['date'].max().date()}")

    panel = panels[0]
    for d in panels[1:]:
        panel = panel.merge(d, on='date', how='outer')
    panel = panel.sort_values('date').reset_index(drop=True)
    # 仅保留7指数都有数据的交易日(对齐)
    panel = panel.dropna(subset=list(have.keys())).reset_index(drop=True)
    print(f"\n  对齐后公共交易日: {len(panel)}天 ({panel['date'].min().date()}~{panel['date'].max().date()})")

    # ── 2. 算破前低+同步率+forward收益 ──
    brk_cols = []
    for code in have:
        panel[f'brk_{code}'] = broken_series(panel[code])
        brk_cols.append(f'brk_{code}')
    panel['n_break'] = panel[brk_cols].sum(axis=1)
    panel['sync'] = panel['n_break'] / len(have)
    for h in HOLDS:
        panel[f'fwd_{h}'] = panel[SH].shift(-h) / panel[SH] - 1

    bt = panel.dropna(subset=[f'fwd_{h}' for h in HOLDS]).copy()
    bt = bt[bt['sync'].notna() & (panel['n_break'] >= 0)].copy()
    print(f"  可回测样本: {len(bt)}天\n")

    # ── 3. 分档统计 ──
    nb = len(have)
    # 分档: 0破 / 1-2 / 3-4 / 5-6 / 全破(nb)
    def tier(k):
        if k == 0: return '0破(无)'
        if k <= nb // 3: return f'1-{nb//3}破(分化)'
        if k <= nb * 2 // 3: return f'{nb//3+1}-{nb*2//3}破(半同步)'
        if k < nb: return f'{nb*2//3+1}-{nb-1}破(高度同步)'
        return f'{nb}破(全破)'
    bt['tier'] = bt['n_break'].apply(tier)
    tier_order = ['0破(无)', f'1-{nb//3}破(分化)', f'{nb//3+1}-{nb*2//3}破(半同步)',
                  f'{nb*2//3+1}-{nb-1}破(高度同步)', f'{nb}破(全破)']

    print("=" * 78)
    print(f"{'同步档':<22}{'样本':>6}{'占比':>7}{'fwd5d':>9}{'fwd10d':>9}{'fwd20d':>9}{'正收益比':>9}")
    print("-" * 78)
    for t in tier_order:
        sub = bt[bt['tier'] == t]
        if len(sub) < 5:
            print(f"{t:<22}{len(sub):>6}{'—':>7}{'—':>9}{'—':>9}{'—':>9}{'—':>9}")
            continue
        m5, m10, m20 = sub['fwd_5'].mean() * 100, sub['fwd_10'].mean() * 100, sub['fwd_20'].mean() * 100
        pos = (sub['fwd_10'] > 0).mean() * 100
        print(f"{t:<22}{len(sub):>6}{len(sub)/len(bt)*100:>6.1f}%{m5:>+8.2f}%{m10:>+8.2f}%{m20:>+8.2f}%{pos:>+8.1f}%")

    # ── 4. 单调性 + 相关 ──
    print("\n" + "=" * 78)
    print("## 同步率 vs 后续收益 相关性 (Spearman)")
    print("-" * 78)
    for h in HOLDS:
        rho, p = stats.spearmanr(bt['sync'], bt[f'fwd_{h}'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  sync vs fwd_{h}d:  ρ={rho:+.4f}  p={p:.4f} {sig}")

    # ── 5. 对照: 全破 vs 分化 vs 单上证 ──
    print("\n" + "=" * 78)
    print("## 对照: 高同步(≥5破) vs 低同步(0-2破) vs 单上证破前低")
    print("-" * 78)
    hi = bt[bt['n_break'] >= 5]
    lo = bt[bt['n_break'] <= 2]
    sh_only = bt[bt[f'brk_{SH}'] == 1]
    sh_no = bt[bt[f'brk_{SH}'] == 0]
    for label, sub in [('高同步≥5破', hi), ('低同步0-2破', lo),
                       (f'单{SH}破', sh_only), (f'单{SH}未破', sh_no)]:
        if len(sub) < 5:
            print(f"  {label:<14} n={len(sub):>4} (样本不足)")
            continue
        m10 = sub['fwd_10'].mean() * 100
        m20 = sub['fwd_20'].mean() * 100
        pos = (sub['fwd_10'] > 0).mean() * 100
        print(f"  {label:<14} n={len(sub):>4}  fwd10d={m10:>+6.2f}%  fwd20d={m20:>+6.2f}%  正收益比={pos:.0f}%")

    # 高同步 vs 低同步 t检验
    if len(hi) >= 10 and len(lo) >= 10:
        t, p = stats.ttest_ind(hi['fwd_10'], lo['fwd_10'], equal_var=False)
        print(f"\n  t检验(高同步 vs 低同步, fwd10d): t={t:+.2f} p={p:.4f} → {'高同步显著更跌' if (p<0.05 and hi['fwd_10'].mean()<lo['fwd_10'].mean()) else '差异不显著或方向相反'}")

    # 信息增量: 全破 vs 单上证破, 后续谁更跌
    if len(hi) >= 10 and len(sh_only) >= 10:
        print(f"\n  信息增量: 全破组 fwd10d={hi['fwd_10'].mean()*100:+.2f}% vs 单上证破 fwd10d={sh_only['fwd_10'].mean()*100:+.2f}%")
        print(f"  → {'全破更跌, 多指数有额外信息' if hi['fwd_10'].mean() < sh_only['fwd_10'].mean() else '单上证已够, 多指数无增量'}")

    # ── 6. 当前实盘锚: 最近同步率 ──
    print("\n" + "=" * 78)
    print("## 当前实盘锚 (最近5个交易日同步率)")
    print("-" * 78)
    recent = panel.tail(6).copy()
    for _, r in recent.iterrows():
        brk = [c.split('_')[1] for c in brk_cols if r[c] == 1]
        print(f"  {r['date'].strftime('%Y-%m-%d')}  破{int(r['n_break'])}/{nb}({r['sync']*100:.0f}%)  {', '.join(brk) if brk else '无'}")

    # ── 7. 结论 ──
    rho5, p5 = stats.spearmanr(bt['sync'], bt['fwd_5'])
    rho20, p20 = stats.spearmanr(bt['sync'], bt['fwd_20'])
    hi_lo_valid = len(hi) >= 10 and len(lo) >= 10 and hi['fwd_10'].mean() < lo['fwd_10'].mean()
    print("\n" + "=" * 78)
    print("## 结论")
    print("-" * 78)
    if p20 < 0.05 and rho20 < 0 and hi_lo_valid:
        print(f"  ✅ 有效: sync与fwd20d显著负相关(ρ={rho20:+.3f}, p={p20:.4f}), 高同步组更跌。")
        print(f"     → '多指数同步破位=真破位'统计成立, 可作regime增强: ≥5破=减仓等深跌, ≤2破=分化别怕可左侧买。")
    elif p5 < 0.05 and rho5 < 0:
        print(f"  🟡 弱有效: 仅短期(fwd5d)显著负相关(ρ={rho5:+.3f}), 中期不一致。参考价值有限。")
    else:
        print(f"  ❌ 证伪: sync与后续收益无显著单调关系(p20={p20:.4f})。")
        print(f"     → '多指数同步破位'无额外预测力, 作者主张不成立(幸存者偏差/主观记忆)。")


if __name__ == '__main__':
    main()
