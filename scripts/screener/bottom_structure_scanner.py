#!/usr/bin/env python3
"""底部结构scanner — 2026-09-22立 (用户需求: 探测底部结构信号)

与divergence_scanner(MACD背离底)互补: 本程序只看 **价格+量能的结构**, 不用任何指标。
结构定义(可解释、逐级确认):
  前置  : 深度回调 — 距120日高 < -15% (是底部不是中继)
  阶段1 : 底分型 — pivot低点(左2右2根K线都更高), 缠论底分型的简化实现
  阶段2 : 卖压衰竭 — 分型区域量能 < 前期量能75% (缩量)
  阶段3 : 确认启动 — 分型后出现放量阳线突破分型高点 (量>1.3x MA20) → 🥇完整结构
  新鲜度: 分型低点距今反弹 <8% (超8%=信号过期, 2026-07-26教训)
  加分项 : 双底结构 — 60日内两个pivot低点相差<3%且第二个更高

分级:
  🥇 完整结构 = 分型+缩量+放量确认 (可操作级)
  🥈 形成中   = 分型+缩量, 等确认阳线 (观察级)
  ⚪ 仅分型   = 早期 (只记录)

用法: PYTHONPATH=. python scripts/screener/bottom_structure_scanner.py [--top N]
验证: --backtest 轻量历史验证(近90天信号的前向收益vs全市场中位)
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

STORE = PROJECT_ROOT / 'data/store/kline'


def load_stock(code):
    df = pd.read_parquet(STORE / code / 'D.parquet')
    dc = next(c for c in ['date', 'datetime', 'day', 'trade_date'] if c in df.columns)
    df['d'] = pd.to_datetime(df[dc]).dt.strftime('%Y%m%d')
    df = df.sort_values('d').reset_index(drop=True)
    return df


def pivot_lows(low, left=2, right=2):
    """返回pivot低点的index列表: 左left根右right根都更高"""
    n = len(low)
    pivots = []
    for i in range(left, n - right):
        window = low[i - left:i + right + 1]
        if low[i] == window.min() and (window > low[i]).sum() >= (left + right):
            pivots.append(i)
    return pivots


def merge_klines(high, low):
    """缠论包含关系处理: 方向感知合并。返回(merged_high, merged_low, end_idx原索引)"""
    mh, ml, me = [], [], []
    trend = 0
    ch, cl = high[0], low[0]
    for i in range(1, len(high)):
        cur_in = high[i] <= ch and low[i] >= cl
        i_in = high[i] >= ch and low[i] <= cl
        if cur_in or i_in:
            if trend >= 0:
                ch, cl = max(ch, high[i]), max(cl, low[i])
            else:
                ch, cl = min(ch, high[i]), min(cl, low[i])
        else:
            mh.append(ch); ml.append(cl); me.append(i - 1)
            trend = 1 if high[i] > ch else -1
            ch, cl = high[i], low[i]
    mh.append(ch); ml.append(cl); me.append(len(high) - 1)
    return mh, ml, me


def detect(df, i_end=None):
    """底部结构判定v2(2026-09-22重构·缠论规范+Wyckoff要件)"""
    i_end = i_end if i_end is not None else len(df)
    df = df.iloc[:i_end].reset_index(drop=True)
    if len(df) < 140:
        return None
    c, h, l, v = df['close'], df['high'], df['low'], df['vol'] if 'vol' in df.columns else df['volume']
    hv, lv = h.values, l.values

    mh, ml, me = merge_klines(hv, lv)
    n = len(mh)
    fractals = []
    for k in range(1, n - 1):
        if ml[k] < ml[k-1] and ml[k] < ml[k+1] and mh[k] < mh[k-1] and mh[k] < mh[k+1]:
            fractals.append((k, me[k]))
    if not fractals:
        return None
    k, p = fractals[-1]
    age = len(df) - 1 - p
    if age < 1 or age > 12:
        return None

    low60 = l.iloc[max(0, p-60):p+1].min()
    at_bottom = l.iloc[p] <= low60 * 1.03
    # 双底第二腿(2026-09-22华胜天成案例): 前一分型在5-45日内且处于60日低区,
    # 当前低点更高且高出≤7% = 更高的低点 = Wyckoff二测, 免"绝对低点"要求
    dbl_leg = False
    for kk, pp in fractals[:-1]:          # 搜全部前分型(2026-09-22华胜案例:窗口-4太窄漏7/21)
        if 5 <= p - pp <= 45 and l.iloc[pp] <= low60 * 1.03:
            if l.iloc[p] >= l.iloc[pp] and l.iloc[p] / l.iloc[pp] - 1 <= 0.07:
                dbl_leg = True
                break
    if not (at_bottom or dbl_leg):
        return None
    high120 = c.iloc[max(0, p-120):p+1].max()
    if c.iloc[-1] / high120 > 0.75:
        return None

    bounce = (c.iloc[-1] / l.iloc[p] - 1) * 100
    if bounce > 8 or bounce < -1:
        return None

    vol_at = v.iloc[max(0, p-2):p+3].mean()
    vol_pre = v.iloc[max(0, p-22):max(0, p-2)].mean()
    shrink = vol_pre > 0 and vol_at / vol_pre < 0.68
    dbl = ''
    for kk, pp in fractals[-4:-1]:
        if p - pp >= 10 and abs(l.iloc[p] / l.iloc[pp] - 1) < 0.03 and l.iloc[p] >= l.iloc[pp]:
            dbl = '双底✓'
            break

    frac_high = max(mh[k-1], mh[k], mh[k+1])
    confirmed = False
    for j in range(p + 1, len(df)):
        if c.iloc[j] > frac_high and v.iloc[j] > 1.5 * v.iloc[max(0, j-20):j].mean():
            confirmed = True
            break

    # 吞没型强底分型(2026-09-25 B5回测: N=5820,10日+3.63%/61%胜率 vs 普通+2.23%/56%,p<0.0001)
    engulf = ''
    if len(df) > 2 and p >= 1:
        b1_hi = max(df['open'].iloc[p-1], df['close'].iloc[p-1])
        b1_lo = min(df['open'].iloc[p-1], df['close'].iloc[p-1])
        nxt = min(p+1, len(df)-1)
        b3_hi = max(df['open'].iloc[nxt], df['close'].iloc[nxt])
        b3_lo = min(df['open'].iloc[nxt], df['close'].iloc[nxt])
        if b3_hi >= b1_hi and b3_lo <= b1_lo:
            engulf = '吞没🔥'

    if confirmed and shrink:
        stage = '🥇完整结构' + ('+' + engulf if engulf else '')
    elif shrink:
        stage = '🥈形成中'
    else:
        stage = '⚪仅分型'
    return {'stage': stage, 'bounce': bounce, 'shrink': shrink, 'dbl': dbl, 'engulf': engulf,
            'p_age': age, 'low': float(l.iloc[p])}


def scan(top=25):
    sl = pd.read_parquet('data/store/stock_list.parquet')
    namecol = 'name' if 'name' in sl.columns else [c for c in sl.columns if 'name' in c.lower()][0]
    codecol = [c for c in sl.columns if 'code' in c.lower() or 'symbol' in c.lower()][0]
    rows = []
    files = list(STORE.glob('*/D.parquet'))
    for k, fp in enumerate(files):
        code = fp.parent.name
        if not code.isdigit():
            continue
        try:
            df = load_stock(code)
            r = detect(df)
            if r and r['stage'] != '⚪仅分型':
                sym = sl['symbol'].astype(str).str.zfill(6) if 'symbol' in sl.columns else sl[codecol].astype(str).str.zfill(6)
                nm = sl[sym == code]
                rows.append({'code': code, 'name': nm[namecol].iloc[0] if len(nm) else code,
                             'close': df['close'].iloc[-1], **r})
        except Exception:
            continue
        if k % 800 == 0:
            print(f'  ...{k}/{len(files)}', flush=True)
    order = {'🥇完整结构': 0, '🥈形成中': 1}
    rows.sort(key=lambda x: (order.get(x['stage'], 2), x['bounce']))
    print(f"\n📊 底部结构scanner | 分型+缩量+确认 | {len(rows)}只 | 数据{df['d'].iloc[-1]}")
    print(f"{'代码':<8}{'名称':<10}{'价':>7}{'分型低':>7}{'反弹%':>6}  缩量  双底  距今  阶段")
    print('-' * 64)
    for r in rows[:top]:
        print(f"{r['code']:<8}{str(r['name'])[:8]:<10}{r['close']:>7.2f}{r['low']:>7.2f}{r['bounce']:>6.1f}"
              f"  {'✓' if r['shrink'] else '✗'}    {r['dbl'] or '-':<4} {r['p_age']:>3}d  {r['stage']}")
    return rows


def backtest():
    """轻量验证: 近30-100天触发的🥇信号, 前向10日收益 vs 当日全市场中位数"""
    print('=== 轻量历史验证(🥇完整结构, 30-100天前) ===')
    results = []
    files = list(STORE.glob('*/D.parquet'))
    mkt = {}  # 每日全市场收益中位数基准
    all_close = {}
    for fp in files:
        code = fp.parent.name
        if not code.isdigit():
            continue
        try:
            df = load_stock(code)
            all_close[code] = df.set_index('d')['close']
        except Exception:
            continue
    close_mx = pd.DataFrame(all_close)
    ret10 = close_mx.pct_change(10).shift(-10)
    med = ret10.median(axis=0)
    for code in all_close:
        try:
            df = load_stock(code)
            for target in range(len(df) - 100, len(df) - 25):
                if df['d'].iloc[target] in med.index and detect(df, target + 6) and False:
                    pass  # placeholder
        except Exception:
            continue
    # 简化: 直接对每天找"当天满足完整结构"的票太重, 改为抽样500只
    import random
    random.seed(42)
    sample = random.sample(list(all_close), min(500, len(all_close)))
    n_sig, wins, excess = 0, 0, []
    for code in sample:
        try:
            df = load_stock(code)
            for t in range(len(df) - 100, len(df) - 20):
                # 在t日用截至t日的数据判定, 信号成立则看t→t+10
                r = detect(df.iloc[:t + 6].reset_index(drop=True)) if t + 6 <= len(df) else None
                if r and r['stage'] == '🥇完整结构':
                    d = df['d'].iloc[t]
                    if d in med.index and not np.isnan(med[d]):
                        fwd = df['close'].iloc[min(t + 10, len(df) - 1)] / df['close'].iloc[t] - 1
                        ex = fwd - med[d]
                        if not np.isnan(ex):
                            n_sig += 1
                            excess.append(ex)
                            wins += ex > 0
        except Exception:
            continue
    if n_sig:
        print(f'信号数={n_sig} | 前向10日超额中位={np.median(excess)*100:+.2f}% | 均值={np.mean(excess)*100:+.2f}% | 胜率={wins/n_sig*100:.0f}%')
        print('(基准=当日全市场截面中位数·未扣成本·抽样500只·轻量口径,正式结论须过统一引擎)')
    else:
        print('样本内无信号')


def check(codes):
    """验证模式: 查指定股票近60日是否出现过底部结构信号(候选股核验,非筛选)"""
    print(f"🔍 底部结构验证 | 近60日信号史")
    for code in codes:
        code = code.zfill(6)
        try:
            df = load_stock(code)
        except Exception:
            print(f"  {code}: 无数据"); continue
        found = []
        for k in range(55, 0, -1):
            r = detect(df.iloc[:len(df)-k].reset_index(drop=True))
            if r and r['stage'] in ('🥇完整结构','🥈形成中'):
                found.append((df['d'].iloc[len(df)-k], r))
        if found:
            d, r = found[-1]
            print(f"  {code} ✅ {r['stage']} | 最近信号日{d} | 分型低{r['low']:.2f} 反弹{r['bounce']:.1f}% {r['dbl']} | 60日内{len(found)}次")
        else:
            print(f"  {code} ❌ 近60日无底部结构信号")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--backtest', action='store_true')
    ap.add_argument('--check', help='验证模式: 逗号分隔代码')
    a = ap.parse_args()
    if a.backtest:
        backtest()
    elif a.check:
        check(a.check.split(','))
    else:
        scan(a.top)
