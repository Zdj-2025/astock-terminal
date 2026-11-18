#!/usr/bin/env python3
"""扫描历史持久化 + 信号累积器 (signal_accumulator)
用户提出的核心洞察: 横向多策略命中 × 纵向多日命中 = 概率累积(贝叶斯更新)

功能:
1. record(): 每个scanner跑完后追加记录到 data/store/scan_history.parquet
2. accumulate(): 读取最近N天历史, 计算每只股票的横向分+纵向分+综合分
3. 输出累积关注榜, 接入daily_scan_all作为第10步

设计:
- 横向分 = 今天被几个不同策略命中 (独立检验数)
- 纵向分 = 同一策略近N天累计命中次数 (对数刻度: log2(1+days))
- 综合分 = 横向分 × Σ(log2(1+纵向天数)) 按策略独立计
- 衰减: 近3日全权重, 4-10日×0.6, 11-20日×0.3
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STORE = PROJECT_ROOT / 'data/store/scan_history.parquet'

STRATEGIES = ['divergence', 'smart_money', 'double_bottom', 'golden_cross',
              'sector_heat', 'catalyst', 'defensive', 'mainline_leader']


def record(strategy, code, name, score=None, metadata=None, date=None):
    """追加一条扫描记录(单条调用,或批量调用record_batch)"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    row = pd.DataFrame([dict(
        date=date, strategy=strategy, code=str(code).zfill(6),
        name=name, score=score,
        metadata=str(metadata) if metadata else ''
    )])
    _append(row)


def record_batch(strategy, rows, date=None):
    """批量追加: rows = [{code,name,score,metadata}, ...]"""
    if not rows:
        return
    if date is None:
        date = datetime.now().strftime('%Y%m%d')  # 紧凑格式(2026-09-09修复:与backfill一致,带杠会被字符串排序挤出窗口)
    df = pd.DataFrame(rows)
    df['date'] = str(date).replace('-', '')
    df['strategy'] = strategy
    df['code'] = df['code'].astype(str).str.zfill(6)
    if 'metadata' not in df.columns:
        df['metadata'] = ''
    _append(df)


def _append(new_rows):
    new_rows = new_rows.copy()
    new_rows['date'] = new_rows['date'].astype(str).str.replace('-', '', regex=False)
    if STORE.exists():
        old = pd.read_parquet(STORE)
        old['date'] = old['date'].astype(str).str.replace('-', '', regex=False)
        df = pd.concat([old, new_rows], ignore_index=True)
    else:
        df = new_rows
    # 去重: 同日同策略同代码只保留最新
    df = df.drop_duplicates(subset=['date', 'strategy', 'code'], keep='last')
    df.to_parquet(STORE, index=False)


def load(n_days=20):
    if not STORE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(STORE)
    df['date'] = df['date'].astype(str).str.replace('-', '', regex=False)  # 防御性归一
    dates = sorted(df['date'].unique(), reverse=True)[:n_days]
    return df[df['date'].isin(dates)].copy()


def accumulate(n_days=20, verbose=True):
    """信号累积: 横向×纵向综合评分"""
    df = load(n_days)
    if df.empty:
        if verbose:
            print("⚠️ 无扫描历史记录")
        return pd.DataFrame()

    today = df['date'].max()
    all_dates = sorted(df['date'].unique(), reverse=True)

    def decay(d):
        """日期衰减: 近3日全权重, 4-10日0.6, 11+日0.3"""
        idx = all_dates.index(d) if d in all_dates else 99
        if idx < 3:
            return 1.0
        elif idx < 10:
            return 0.6
        else:
            return 0.3

    df['decay'] = df['date'].apply(decay)

    # 纵向: 每只股票×策略, 加权天数
    vert = df.groupby(['code', 'strategy']).agg(
        days=('date', 'nunique'),
        decay_days=('decay', 'sum'),
        last_date=('date', 'max')
    ).reset_index()
    vert['vert_score'] = np.log2(1 + vert['decay_days'])

    # 横向: 今天被几个策略命中
    today_hits = df[df['date'] == today].groupby('code')['strategy'].nunique().rename('horiz').reset_index()

    # 综合: 权重加权横向 × 纵向各策略之和 (2026-09-08接入权重引擎)
    vert_sum = vert.groupby('code').agg(
        total_vert=('vert_score', 'sum'),
        strategies=('strategy', 'nunique'),
        max_days=('days', 'max')
    ).reset_index()

    # 先创建today_df
    today_df = df[df['date'] == today].copy()

    # 策略簇修正(2026-09-08 Jaccard分析): golden_cross×smart_money J=0.35
    # 同簇策略同时命中不算独立验证,按簇计数
    STRATEGY_CLUSTER = {
        'divergence': 'bottom',      # 底部结构
        'double_bottom': 'bottom',   # 底部结构(与divergence部分重叠但信号更严)
        'golden_cross': 'trend',     # 趋势(金叉)
        'smart_money': 'trend',      # 趋势(含金叉特征,J=0.35)
        'oversold_reversal': 'bottom', # 超跌(底部族)
        'relative_strength': 'trend',  # 相对强度(趋势族)
        'quality_divergence': 'bottom', # 质量背离(底部族)
        'buyback': 'event',           # 事件族
        'small_cap': 'factor',        # 因子族
        'money_flow_div': 'flow',     # 资金流族
    }
    def _cluster(strat):
        return STRATEGY_CLUSTER.get(strat, 'other')
    today_df['cluster'] = today_df['strategy'].apply(_cluster)
    # 横向按独立簇计数(非同簇策略才算独立命中)
    horiz_clusters = today_df.groupby('code')['cluster'].nunique().rename('horiz').reset_index()

    # 接入权重引擎
    try:
        from scripts.core.strategy_weight_engine import WeightEngine
        _we = WeightEngine()
        _weights = _we.get_weights()
        def _w(strat):
            return _weights.get(strat, 0.05)
    except ImportError:
        _weights = None
        _w = lambda strat: 1.0  # 无权重引擎时退化为等权

    # 横向: 用加权求和替代简单计数
    today_df['w_hit'] = today_df['strategy'].apply(_w)
    horiz_weighted = today_df.groupby('code')['w_hit'].sum().rename('horiz_w').reset_index()
    horiz_count = horiz_clusters  # 按簇计数(修正J=0.35的假独立)
    horiz_final = horiz_count.merge(horiz_weighted, on='code', how='outer')

    result = horiz_final.merge(vert_sum, on='code', how='outer').fillna(0)
    # 综合分: 加权横向² × log(1+纵向) — 权重引擎驱动
    result['composite'] = result['horiz_w'] ** 2 * np.log2(1 + result['total_vert'])

    # 加入名称
    names = df.groupby('code')['name'].last().reset_index()
    result = result.merge(names, on='code', how='left')

    # 当前有持仓的标记
    result = result.sort_values('composite', ascending=False)

    if verbose:
        print(f"\n{'═'*80}")
        print(f"📊 信号累积榜 | 数据{df['date'].min()}~{today} | 覆盖{len(all_dates)}个交易日")
        print(f"{'═'*80}")
        print(f"  记录总数{len(df)} | 独立个股{df['code'].nunique()}只")
        print(f"\n  综合分 = 横向(今日策略数) × Σ纵向(各策略log2加权天数))")
        print(f"\n{'代码':<8}{'名称':<10}{'横向':>4}{'纵向分':>7}{'策略数':>5}{'最大天数':>7}{'综合分':>7}")
        print('─' * 60)
        for _, r in result.head(20).iterrows():
            print(f"  {r['code']:<8}{str(r['name'])[:9]:<10}{int(r['horiz']):>4}{r['total_vert']:>7.1f}"
                  f"{int(r['strategies']):>5}{int(r['max_days']):>7}{r['composite']:>7.1f}")
        print(f"\n  💡 综合分>5=Tier3(主推候选) 3-5=Tier2(重点跟踪) <3=Tier1(观察)")

    return result


def main():
    accumulate()


if __name__ == '__main__':
    main()
