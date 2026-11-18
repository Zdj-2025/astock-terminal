#!/usr/bin/env python3
"""卖出信号回测 — 多标准对比(日线3S / 分钟级顶背离 / 组合) + 大盘环境。

验证核心问题: 各种卖出信号触发后, 股票是继续跌(卖对)还是反弹(卖错/洗盘)?
无未来: 信号只用T日及之前数据, 后续收益shift(-N)。

对比的卖出信号:
  A. 3S日线三条件全满(MA5死叉MA10≥3天 + MACD转负 + 破MA20)
  B. 破MA20单条件(简化版)
  C. MACD死叉(DIF<DEA)
  D. MA5死叉MA10
  E. 3S任两条满足(宽松版)
  F. 无信号持有(对照)

分大盘环境: 上升/震荡/下降
用法: cd stock && PYTHONPATH=. python scripts/research/sell_signal_backtest_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np
import pandas as pd
from scipy import stats

STORE = Path(__file__).resolve().parent.parent.parent / 'data/store/kline'
HOLDS = [5, 10, 20]


def load_index():
    """大盘环境(上证20日涨跌)。"""
    for idx in ['000001SH', '000001']:
        p = STORE / idx / 'D.parquet'
        if p.exists():
            df = pd.read_parquet(p)
            for c in ['date', 'day', 'trade_date']:
                if c in df.columns:
                    df = df.rename(columns={c: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['idx_ret20'] = df['close'].pct_change(20)
            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
            df['regime'] = df['idx_ret20'].apply(
                lambda x: '上升' if x > 0.05 else ('下降' if x < -0.05 else '震荡'))
            return df[['date_str', 'regime']].dropna()
    return None


def compute_sell_signals(df):
    """计算各种卖出信号(全部用T日及之前数据, 无未来)。"""
    df = df.sort_values('date').reset_index(drop=True)
    if len(df) < 60:
        return None
    close = df['close'].astype(float).ffill()
    high = df['high'].astype(float).ffill() if 'high' in df.columns else close
    low = df['low'].astype(float).ffill() if 'low' in df.columns else close

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = 2 * (dif - dea)

    out = pd.DataFrame({'date': df['date']})
    out['date_str'] = df['date'].astype(str).str.replace('-', '')

    # A. 3S三条件全满
    cond1 = (ma5 < ma10).rolling(3).sum() >= 3  # MA5死叉MA10≥3天
    cond2 = (hist < 0) | (dif < dea)            # MACD柱转负或DIF死叉DEA
    cond3 = close < ma20                          # 破MA20
    out['sig_3S'] = (cond1 & cond2 & cond3).fillna(False)

    # B. 破MA20单条件
    out['sig_break_ma20'] = (close < ma20).fillna(False)

    # C. MACD死叉(DIF<DEA)
    out['sig_macd_cross'] = (dif < dea).fillna(False)

    # D. MA5死叉MA10
    out['sig_ma5_cross'] = (ma5 < ma10).fillna(False)

    # E. 3S任两条满足(宽松)
    conds = pd.DataFrame({'c1': cond1.fillna(False), 'c2': cond2.fillna(False), 'c3': cond3.fillna(False)})
    out['sig_3S_loose'] = (conds.sum(axis=1) >= 2)

    # F. 无信号(对照) = 所有日期
    out['sig_none'] = True

    # 后续收益(卖出后跌=卖对了)
    for h in HOLDS:
        out[f'ret{h}'] = (close.shift(-h) / close - 1).values

    return out


def main():
    print("卖出信号回测 | 多标准对比(3S/破MA20/MACD死叉/MA5死叉/宽松) × 大盘环境")
    print("核心: 信号触发后继续跌=卖对了; 反弹=卖错了(洗盘)\n")

    idx_env = load_index()
    if idx_env is not None:
        regimes = dict(zip(idx_env['date_str'], idx_env['regime']))
        env_dist = idx_env['regime'].value_counts().to_dict()
        print(f"大盘环境分布: {env_dist}")
    else:
        regimes = {}

    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))])
    print(f"扫描 {len(codes)} 只...")
    snaps = []
    for i, c in enumerate(codes):
        try:
            s = compute_sell_signals(pd.read_parquet(STORE / c / 'D.parquet'))
            if s is not None:
                s['code'] = c
                s['regime'] = s['date_str'].map(regimes)
                snaps.append(s)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]")

    snap = pd.concat(snaps, ignore_index=True).dropna(subset=['ret10'])
    if regimes:
        snap = snap.dropna(subset=['regime'])
    print(f"总快照: {len(snap):,}\n")

    signals = [
        ('无信号(持有对照)', 'sig_none'),
        ('A.3S全满(3条件)', 'sig_3S'),
        ('B.破MA20', 'sig_break_ma20'),
        ('C.MACD死叉', 'sig_macd_cross'),
        ('D.MA5死叉MA10', 'sig_ma5_cross'),
        ('E.3S宽松(任2条)', 'sig_3S_loose'),
    ]

    # 全部环境
    print(f"{'信号':<22}{'N':>8}{'ret5':>8}{'ret10':>8}{'ret20':>8}{'说明':>12}")
    print('-' * 62)
    for nm, col in signals:
        g = snap[snap[col]]
        n = len(g)
        if n < 100:
            print(f"{nm:<20}{n:>8,}  (样本不足)"); continue
        r5 = g['ret5'].mean() * 100
        r10 = g['ret10'].mean() * 100
        r20 = g['ret20'].mean() * 100
        # 卖出后跌=卖对了(负收益好)
        verdict = '✅卖对(续跌)' if r10 < -0.5 else ('⚠️卖错(反弹)' if r10 > 0.5 else '—中性')
        print(f"{nm:<20}{n:>8,}{r5:>+7.2f}%{r10:>+7.2f}%{r20:>+7.2f}%{verdict:>12}")

    # 分大盘环境(只看3S和对照)
    if regimes:
        print(f"\n[分大盘环境] 信号触发后ret10(负=卖对, 正=卖错)")
        print(f"{'信号':<22}{'环境':<6}{'N':>8}{'ret10':>8}{'判定':>12}")
        print('-' * 58)
        for nm, col in [('3S全满', 'sig_3S'), ('破MA20', 'sig_break_ma20'), ('MACD死叉', 'sig_macd_cross')]:
            for regime in ['上升', '震荡', '下降']:
                g = snap[snap[col] & (snap['regime'] == regime)]
                n = len(g)
                if n < 50:
                    print(f"{nm:<20}{regime:<6}{n:>8,}  不足"); continue
                r10 = g['ret10'].mean() * 100
                verdict = '✅卖对' if r10 < -0.5 else ('⚠️卖错' if r10 > 0.5 else '—中性')
                print(f"{nm:<20}{regime:<6}{n:>8,}{r10:>+7.2f}%{verdict:>12}")

    # t检验: 3S信号 vs 对照
    print(f"\n[t检验 3S信号触发后收益 vs 对照(持有)]")
    s_3S = snap.loc[snap.sig_3S, 'ret10'].dropna()
    s_none = snap.loc[snap.sig_none, 'ret10'].dropna()
    if len(s_3S) > 100:
        t, p = stats.ttest_ind(s_3S, s_none, equal_var=False)
        verdict = '✅3S有效(触发后显著更跌=该卖)' if (p < 0.05 and s_3S.mean() < s_none.mean()) else \
                  ('⚠️3S无效(触发后不跌=洗盘下车)' if (p < 0.05 and s_3S.mean() >= s_none.mean()) else '—无显著差异')
        print(f"  3S触发后ret10={s_3S.mean()*100:+.2f}% vs 持有={s_none.mean()*100:+.2f}% t={t:+.2f} p={p:.4f} {verdict} N={len(s_3S)}")

    # 3S不同持有期
    print(f"\n[3S全满信号 不同持有期]")
    g = snap[snap.sig_3S]
    if len(g) >= 100:
        for h in HOLDS:
            r = g[f'ret{h}'].dropna()
            wr_down = (r < 0).mean() * 100  # 继续跌的比例
            print(f"  卖出后{h}日: 均值{r.mean()*100:+.2f}% 继续跌比例{wr_down:.0f}% N={len(r)}")


if __name__ == '__main__':
    main()
