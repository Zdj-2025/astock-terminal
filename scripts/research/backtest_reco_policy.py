#!/usr/bin/env python3
"""推荐政策回测(2026-08-31)——验证反思后的新推荐闸门是否有增量。

背景: 50笔推荐实盘对账: 中位超额-3.6%, 跑赢率30%; 8/13医药批0%跑赢(主题确认日=顶部)。
本回测检验四道闸门(regime/主题冷却/回踩触发)的因果增量 + 旧策略代理(热主题强势票)的负alpha。

五组(累积式):
  A 纯信号: 250日分位<30% + 底背离量能确认(向量化因果版)
  B = A + regime闸(上证t-1收<MA20不推)
  C = B + 主题冷却(行业20日等权涨幅Top5不推)
  D = C + 回踩缩量触发(确认后15bar内 close<=MA5*1.02 且 vol<0.8*vol_ma20 才买)
  E 旧策略代理(反面对照): 每日热行业Top5内 站上MA20 且20日动量最强的3只

执行: t+1开盘买入, 开盘涨停放弃(主板9.5%/创科19.5%), 双边成本0.13%, 超额=个股-上证同窗口。
统计: N/均值/中位/胜率 + 配对t(A内子集) + 分年 + 分regime。
质量层不回测(本地财务无时点数据, 且质量=排雷非alpha)。
幸存者偏差: 本地store无退市票, 如实标注。
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')

START, END = '20250101', '20260828'   # yyyymmdd(与信号日同格式,防字符串比较错位)
HOLDS = [5, 10, 20]
COST = 0.0013          # 双边
PCT_LOW = 0.30         # 低位定义
CONF_WINDOW = 10       # 谷后放量阳线确认窗口
TRIG_WINDOW = 15       # 回踩触发等待窗口
TRIG_MA5 = 1.02        # 贴MA5阈值
TRIG_VOL = 0.8         # 缩量阈值
HOT_TOPN = 5           # 热行业数


def load_index():
    """上证日线, 陈旧则tushare现拉"""
    p = 'data/store/index/000001_SH.parquet'
    df = pd.read_parquet(p)
    df['d'] = df['trade_date'].astype(str)
    df = df.sort_values('d')
    if df['d'].iloc[-1] < '20260820':
        from scripts.core.tushare_client import TushareClient
        pro = TushareClient().pro
        rows = []
        for y in ['20240601', '20250101', '20250801', '20260201', '20260601']:
            pass
        # 直接分段拉
        d0 = df['d'].iloc[-1]
        nxt = d0
        for _ in range(6):
            t = pro.index_daily(ts_code='000001.SH', start_date=nxt, end_date='20260828')
            if t is None or t.empty:
                break
            rows.append(t)
            last = t['trade_date'].max()
            if last >= '20260828' or last == nxt:
                break
            nxt = last
        if rows:
            new = pd.concat(rows).drop_duplicates('trade_date')
            df = pd.concat([df, new], ignore_index=True).drop_duplicates('d', keep='first').sort_values('d')
    df['ma20'] = df['close'].astype(float).rolling(20).mean()
    return df.reset_index(drop=True)


def macd_hist(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif - dea


def causal_signals(df):
    """因果版底背离(对齐生产divergence_detector的E组合):
    谷对=最新两确认谷(失败回退前一对); 价新低+hist抬高+DIF抬高(D过滤);
    谷后再跌>10%失效(B过滤); 谷后10日内放量阳线确认(vol>20日均量); 反弹<8%+距高<-5%(过期)。
    返回信号日索引列表(每天独立评估,全部用截至当日的信息)。"""
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    open_ = df['open'].astype(float).values
    high = df['high'].astype(float).values
    vol = (df['vol'] if 'vol' in df.columns else df['volume']).astype(float).values
    n = len(df)
    hist = macd_hist(pd.Series(close))
    dif = pd.Series(close).ewm(span=12, adjust=False).mean() - pd.Series(close).ewm(span=26, adjust=False).mean()
    hist = hist.values
    dif = dif.values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values

    # swing lows: i是[i-5,i+5]严格最低 → i+5日可知
    troughs = []
    for i in range(5, n - 5):
        if all(low[i] < low[i - j] for j in range(1, 6)) and all(low[i] < low[i + j] for j in range(1, 6)):
            troughs.append(i)
    if len(troughs) < 2:
        return []

    # 每日t作为候选确认日,动态取"截至t已确认"的谷
    out = []
    ti = 0  # trough指针: troughs[ti]为首个未确认谷
    for t in range(n):
        while ti < len(troughs) and troughs[ti] + 5 <= t:
            ti += 1
        conf = troughs[:ti]
        if len(conf) < 2:
            continue
        p2, p1 = conf[-1], conf[-2]
        pairs = [(p1, p2)]
        if len(conf) >= 3:
            pairs.append((conf[-3], conf[-2]))
        for a, b in pairs:
            if not (low[b] < low[a] and hist[b] > hist[a] and dif[b] > dif[a]):
                continue
            if b + 1 <= t and np.min(low[b + 1:t + 1]) < low[b] * 0.90:   # B:谷后再跌>10%
                continue
            if t - b > 10 or t <= b:                                       # C:谷后10日内
                continue
            if not (close[t] > open_[t] and not np.isnan(vol_ma20[t]) and vol[t] > vol_ma20[t]):
                continue
            if close[t] / low[b] - 1 > 0.08:                               # 过期:谷后反弹>8%
                continue
            hb = high[max(0, t - 249):t + 1].max()
            if close[t] / hb - 1 > -0.05:                                  # 过期:距250日高不足5%
                continue
            out.append(t)
            break
    return out


def main():
    rng = np.random.default_rng(42)
    idx = load_index()
    idx_close = idx.set_index('d')['close'].astype(float)
    idx_open = idx.set_index('d')['open'].astype(float)
    idx_ma20 = idx.set_index('d')['ma20']
    dates_all = idx['d'].tolist()
    dpos = {d: i for i, d in enumerate(dates_all)}

    sl = pd.read_parquet('data/store/stock_list.parquet')
    sym = sl['symbol'].astype(str).str.zfill(6)
    ind_map = dict(zip(sym, sl['industry'].astype(str)))
    name_map = dict(zip(sym, sl['name'].astype(str)))

    codes = sorted([d.name for d in Path('data/store/kline').iterdir()
                    if d.is_dir() and (d / 'D.parquet').exists()
                    and d.name.isdigit() and len(d.name) == 6
                    and d.name[:2] not in ('51', '15', '56', '58')])
    print(f'universe: {len(codes)}只 | 窗口 {START}~{END}')

    # ── Pass1: 收集全市场收盘价矩阵(算行业热度) ──
    closes = {}
    parsed = {}
    for code in codes:
        try:
            df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
            for col in ['datetime', 'day', 'trade_date']:
                if col in df.columns and col != 'date':
                    df = df.rename(columns={col: 'date'})
            df['d'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
            df = df.sort_values('d').drop_duplicates('d', keep='last')
            if len(df) < 300:
                continue
            parsed[code] = df.reset_index(drop=True)
            closes[code] = df.set_index('d')['close'].astype(float)
        except Exception:
            continue
    print(f'parsed: {len(parsed)}只')

    # 行业×日 等权20日收益 → 每日热行业Top5
    close_df = pd.DataFrame(closes)
    ret20 = close_df.pct_change(20)
    ind_of = {c: ind_map.get(c, '未知') for c in close_df.columns}
    ind_ret = ret20.T.groupby(ind_of).mean().T  # date × industry
    hot_by_date = {}
    for d, row in ind_ret.iterrows():
        if d < START or pd.isna(row).all():
            continue
        hot_by_date[d] = set(row.nlargest(HOT_TOPN).index)

    # regime: 上证t-1收 < MA20(t-1) → 坏
    bad_regime = set()
    prev = None
    for d in dates_all:
        if prev is not None and not np.isnan(idx_ma20[prev]) and idx_close[prev] < idx_ma20[prev]:
            bad_regime.add(d)
        prev = d

    # ── Pass2: 信号与执行 ──
    trades_A = []   # (code, t_date, entry_date)
    trig_candidates = []  # D组: (code, t_date) 后续判断触发日
    raw_signals = []  # 交叉验证用: 未过滤的原始信号
    for code, df in parsed.items():
        close = df['close'].astype(float)
        if len(df) < 300:
            continue
        nm = name_map.get(code, '')
        # 避雷过滤(与生产divergence_scanner一致): ST/退市/低价<2元
        if 'ST' in nm or '退' in nm or float(close.iloc[-1]) < 2:
            continue
        roll = close.rolling(250)
        lo, hi = roll.min(), roll.max()
        pct = (close - lo) / (hi - lo)
        dist120 = close / close.rolling(120).max() - 1
        ma5 = close.rolling(5).mean()
        vol = (df['vol'] if 'vol' in df.columns else df['volume']).astype(float)
        vol_ma20 = vol.rolling(20).mean()
        dcol = df['d'].tolist()

        for sig_idx in causal_signals(df):
            if sig_idx >= len(dcol) - 1:
                continue
            t = dcol[sig_idx]
            if not (START <= t <= END):
                continue
            raw_signals.append((code, t))
            if sig_idx >= len(df) - 25:   # 前瞻窗口不足,只记录不交易
                continue
            if pd.isna(pct.iloc[sig_idx]) or pct.iloc[sig_idx] >= PCT_LOW:
                continue
            if pd.isna(dist120.iloc[sig_idx]) or dist120.iloc[sig_idx] < -0.75:
                continue
            trades_A.append((code, t))
            # D组触发扫描
            for p in range(sig_idx + 1, min(sig_idx + 1 + TRIG_WINDOW, len(df) - 22)):
                if (close.iloc[p] <= ma5.iloc[p] * TRIG_MA5 and
                        vol.iloc[p] < TRIG_VOL * vol_ma20.iloc[p]):
                    trig_candidates.append((code, t, dcol[p]))
                    break

    # E组: 每日热行业Top5内 20日动量最强3只(站上MA20)
    trades_E = []
    ma20_all = close_df.rolling(20).mean()
    for d in hot_by_date:
        if not (START <= d <= END) or d not in ma20_all.index:
            continue
        pool = [c for c in close_df.columns
                if ind_of[c] in hot_by_date[d]
                and not np.isnan(ret20.at[d, c]) if c in ret20.columns]
        pool = [c for c in pool if not np.isnan(ma20_all.at[d, c])
                and close_df.at[d, c] > ma20_all.at[d, c]]
        if not pool:
            continue
        mom = {c: ret20.at[d, c] for c in pool}
        top3 = sorted(mom, key=mom.get, reverse=True)[:3]
        for c in top3:
            trades_E.append((c, d, d))

    def execute(triples, group):
        """(code, signal_date, buy_decision_date) → t+1开盘买"""
        rows = []
        for code, t, bd in triples:
            df = parsed.get(code)
            if df is None:
                continue
            dcol = df['d'].tolist()
            if bd not in dcol:
                continue
            bi = dcol.index(bd)
            if bi + 1 >= len(dcol) - 1:
                continue
            ed = dcol[bi + 1]  # entry date
            op = float(df['open'].iloc[bi + 1])
            pc = float(df['close'].iloc[bi])
            lim = 0.195 if code[0] in ('3', '68') else 0.095
            if op / pc - 1 > lim:   # 开盘涨停放弃
                continue
            if ed not in dpos:
                continue
            ii = dpos[ed]
            row = dict(code=code, group=group, t=t, ed=ed, entry=op)
            for H in HOLDS:
                j = min(ii + H, len(dates_all) - 1)
                hd = dates_all[j]
                # 个股退出价: entry日后第H个该股交易日收盘
                ci = dcol.index(ed)
                xi = min(ci + H, len(dcol) - 1)
                px = float(df['close'].iloc[xi])
                ret = px / op - 1 - COST
                bench = idx_close[hd] / idx_open[ed] - 1
                row[f'ex{H}'] = (ret - bench) * 100
            rows.append(row)
        return pd.DataFrame(rows)

    dfA = execute([(c, t, t) for c, t in trades_A], 'A')
    dfE = execute(trades_E, 'E')

    # 闸门标记(在A上做子集, 保证配对)
    dfA['hot'] = [ind_of[c] in hot_by_date.get(t, set()) for c, t in zip(dfA.code, dfA.t)]
    dfA['bad'] = [t in bad_regime for t in dfA.t]
    trig_keys = {(c, t) for c, t, p in trig_candidates}
    dfA['trig'] = [(c, t) in trig_keys for c, t in zip(dfA.code, dfA.t)]

    from scipy import stats as st
    print('\n' + '=' * 86)
    print(f'推荐政策回测 | {START}~{END} | 超额=个股(t+1开→H收,扣费0.13%) - 上证同窗口')
    print('=' * 86)

    def report(df, label):
        if df.empty:
            print(f'{label:<28} N=0')
            return None
        line = f'{label:<28} N={len(df):>5}'
        for H in HOLDS:
            e = df[f'ex{H}']
            line += f' | {H}d均{e.mean():+.2f}%/中{e.median():+.2f}%/胜{(e > 0).mean()*100:.0f}%'
        print(line)
        return df

    A = report(dfA, 'A 纯信号(低位+背离确认)')
    B = report(dfA[~dfA.bad], 'B =A+regime闸(好期)')
    C = report(dfA[(~dfA.bad) & (~dfA.hot)], 'C =B+主题冷却(非热门)')
    D = report(dfA[(~dfA.bad) & (~dfA.hot) & dfA.trig], 'D =C+回踩缩量触发')
    report(dfA[dfA.bad], '  [对照]A在坏regime')
    report(dfA[dfA.hot], '  [对照]A在热门行业')
    report(dfE, 'E 旧策略代理(热行业强势3)')

    # 配对t检验(同A池内嵌套对比)
    print('\n配对t检验(超额10日):')
    for lbl, m in [('regime闸: A好期 vs A坏期', (dfA[~dfA.bad], dfA[dfA.bad])),
                   ('主题冷却: 非热门 vs 热门(好期内)', (dfA[(~dfA.bad) & (~dfA.hot)], dfA[(~dfA.bad) & dfA.hot])),
                   ('回踩触发: 触发 vs 未触发(C内)', (dfA[(~dfA.bad) & (~dfA.hot) & dfA.trig], dfA[(~dfA.bad) & (~dfA.hot) & (~dfA.trig)]))]:
        a, b = m[0]['ex10'], m[1]['ex10']
        if len(a) > 5 and len(b) > 5:
            t1, p1 = st.ttest_ind(a, b, equal_var=False)
            print(f'  {lbl}: 差{a.mean()-b.mean():+.2f}% (t={t1:.2f}, p={p1:.4f}) N={len(a)}/{len(b)}'
                  f' {"✅" if p1 < 0.05 else "⚠️" if p1 < 0.15 else "❌"}')
    if not dfE.empty and not dfA.empty:
        t1, p1 = st.ttest_ind(dfE['ex10'], dfA['ex10'], equal_var=False)
        print(f'  E旧代理 vs A纯信号: 差{dfE.ex10.mean()-dfA.ex10.mean():+.2f}% (t={t1:.2f}, p={p1:.4f}) N={len(dfE)}/{len(dfA)}')

    # 分年
    print('\n分年(10日超额中位/胜率/N):')
    for lbl, seg in [('2025', dfA[(dfA.t < '20260101') & (~dfA.bad)]),
                     ('2026H1', dfA[(dfA.t >= '20260101') & (dfA.t < '20260701') & (~dfA.bad)]),
                     ('2026H2', dfA[(dfA.t >= '20260701') & (~dfA.bad)])]:
        if len(seg):
            print(f'  B组 {lbl}: 中位{seg.ex10.median():+.2f}% 胜率{(seg.ex10>0).mean()*100:.0f}% N={len(seg)}')

    # 验证: 用原始信号(不经交易过滤)核对已知案例
    print('\n交叉验证(2026-08下旬已知确认信号,查原始信号池):')
    known = [s for s in raw_signals if s[1] >= '20260815'
             and s[0] in ('603992', '000776', '601108', '600023')]
    for c, t in known:
        print(f'  {c} 信号日{t} ✓')
    if not known:
        aug = sum(1 for _, t in raw_signals if t >= '20260815')
        print(f'  ⚠️ 未检出——raw_signals总数{len(raw_signals)}, 8月下旬{aug}条; 向量化逻辑需校准')
    else:
        aug = sum(1 for _, t in raw_signals if t >= '20260815')
        print(f'  (8月下旬全市场原始信号{aug}条)')

    dfA.to_parquet('data/store/reco_policy_bt.parquet')
    print('\n已存 data/store/reco_policy_bt.parquet')


if __name__ == '__main__':
    main()
