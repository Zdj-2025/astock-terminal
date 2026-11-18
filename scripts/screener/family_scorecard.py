#!/usr/bin/env python3
"""族计分卡 — 推荐按"逻辑族"分组算超额,自动化闭环审计(2026-09-11立).

背景: 9/11手搓计分卡发现——赢家全是两端(突破/防御),输家全是中间(背离/双底抄底),
且9条推荐4条"等回踩"踏空(含+16.5%的招商南油). 手搓不可持续,本脚本自动化.

输出三张表:
  ①族计分卡: 按族聚合(笔数/平均收益/平均超额vs上证/超额胜率)
  ②入场框架分析: entry_style分组——等回踩族的"触发率"(buy_point是否触到)与踏空成本
     踏空成本 = 推荐日→现价收益 - (若10日内触发区被触到则按区间上沿买的模拟收益)
  ③写回: recommendations.csv 增量补 family/entry_style 列(幂等)

用法: cd stock && PYTHONPATH=. python scripts/screener/family_scorecard.py
纪律: 本卡是度量不是建议; 族降权/撤队决策走R1闸(样本不足只记录).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import re

CSV = PROJECT_ROOT / 'reports/recommendations.csv'
STORE = PROJECT_ROOT / 'data/store/kline'

# ── 族分类: 按优先级首个命中(先具体后宽泛;2026-09-11 v2修污染:
#    松霖"质量87背离"曾被quality_pullback误抓→divergence提到其前;quality只认Tier1/六张网) ──
FAMILY_RULES = [
    ('double_bottom',   ['双底算法', '浅双底', '双底']),
    ('breakout',        ['突破真', '突破颈线', '三重底']),
    ('defensive',       ['防御', '公用', '类债', '水电', '火电', '大秦', '电力']),
    ('deep_value',      ['底部抬升', '深价值', '2年底部', '底部区间']),
    ('divergence',      ['背离']),
    ('quality_pullback',['Tier1', '六张网']),
    ('catalyst',        ['催化驱动', '预增>30', '催化']),
    ('smart',           ['smart', '聪明钱']),
]

def classify_family(row):
    """两遍扫描: 先reason+mainline(主判据), 无命中再看note(note常含元数据如'背离族胜率'会污染)."""
    for fields in (('reason', 'mainline'), ('note', 'name')):
        text = ' '.join(str(row.get(c, '')) for c in fields)
        for fam, kws in FAMILY_RULES:
            if any(k in text for k in kws):
                return fam
    return 'other'

def classify_entry(row):
    bp = str(row.get('buy_point', ''))
    if '回踩' in bp and ('等' in bp or re.search(r'\d+\s*MA', bp)):
        return 'wait_pullback'
    if '浅入场' in bp or '次日' in bp:
        return 'shallow_next_day'
    if re.search(r'\d+\.?\d*\s*-\s*\d+\.?\d*', bp):
        return 'zone_limit'
    if '放量' in bp or '破' in bp:
        return 'breakout_trigger'
    if bp in ('不再买入', 'na', 'nan', ''):
        return 'none'
    return 'other'

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
    recs = pd.read_csv(CSV)
    recs['code'] = recs['code'].astype(str).str.zfill(6)

    # 幂等写回 family/entry_style
    changed = False
    if 'family' not in recs.columns:
        recs['family'] = ''
    if 'entry_style' not in recs.columns:
        recs['entry_style'] = ''
    for i, r in recs.iterrows():
        fam, ent = classify_family(r), classify_entry(r)
        if str(recs.at[i, 'family']) != fam or str(recs.at[i, 'entry_style']) != ent:
            recs.at[i, 'family'], recs.at[i, 'entry_style'] = fam, ent
            changed = True
    if changed:
        recs.to_csv(CSV, index=False)

    idx = load('000001SH')
    if idx is None:
        print('❌ 无上证数据'); return
    idx_close_at = lambda d: idx[idx['date'] <= d]['close'].iloc[-1]
    idx_now = idx['close'].iloc[-1]

    rows = []
    for _, r in recs.iterrows():
        df = load(r['code'])
        if df is None:
            continue
        rd = pd.to_datetime(r['rec_date'])
        rec_rows = df[df['date'] >= rd]
        if rec_rows.empty:
            continue
        base = rec_rows.iloc[0]
        cur = df['close'].iloc[-1]
        if not base['close'] or pd.isna(base['close']):
            continue
        ret = cur / base['close'] - 1
        # 超额 vs 上证同窗口
        try:
            ib = idx_close_at(base['date'])
            excess = ret - (idx_now / ib - 1)
        except Exception:
            excess = float('nan')
        # 触发率: buy_point区间上沿10日内是否被触到(等回踩族的踏空度量)
        trigger = None
        m = re.findall(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', str(r.get('buy_point', '')))
        if m:
            hi = max(float(a) for a, b in m)
            win = df[(df['date'] >= base['date'])].head(10)
            trigger = bool((win['low'] <= hi).any())
        rows.append(dict(family=r['family'], entry_style=r['entry_style'],
                         ret=ret * 100, excess=excess * 100, days=(df['date'].iloc[-1] - base['date']).days,
                         zone_touched=trigger))
    t = pd.DataFrame(rows)
    if t.empty:
        print('无数据'); return

    print('═' * 72)
    print(f'📊 族计分卡 | {len(t)}条推荐 | 基准=上证(推荐日→最新) | 数据{idx["date"].iloc[-1].strftime("%Y-%m-%d")}')
    print('═' * 72)
    g = t.groupby('family').agg(n=('ret', 'size'), avg_ret=('ret', 'mean'),
                                avg_excess=('excess', 'mean'),
                                win_excess=('excess', lambda x: (x > 0).mean() * 100)
                                ).sort_values('avg_excess', ascending=False)
    g['avg_ret'] = g['avg_ret'].round(1); g['avg_excess'] = g['avg_excess'].round(1)
    g['win_excess'] = g['win_excess'].round(0).astype(int)
    print(f"{'族':<18}{'笔数':>4}{'平均收益%':>9}{'平均超额%':>9}{'超额胜率%':>9}")
    for fam, r in g.iterrows():
        print(f"{fam:<18}{int(r['n']):>4}{r['avg_ret']:>9.1f}{r['avg_excess']:>9.1f}{r['win_excess']:>9.0f}")

    print('\n── 入场框架分析(踏空度量: 等回踩是否等到) ──')
    e = t[t['zone_touched'].notna()].groupby('entry_style').agg(
        n=('ret', 'size'),
        触发率=('zone_touched', lambda x: x.mean() * 100),
        avg_excess=('excess', 'mean')).round(1)
    print(e.to_string())

    missed = t[(t['zone_touched'] == False) & (t['excess'] > 3)]
    if len(missed):
        print(f'\n⚠️ 踏空且跑赢3%+的推荐: {len(missed)}条 — 入场框架与股性错配样本')

    print('\n💡 度量不是建议: 族降权/撤队走R1闸(小样本只记录);周六审计引用本卡')


if __name__ == '__main__':
    main()
