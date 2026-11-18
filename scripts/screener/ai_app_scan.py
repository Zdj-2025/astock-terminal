#!/usr/bin/env python3
"""AI应用软件板块专项扫描 — 基本面哨兵通过后的技术面评分。

用法: PYTHONPATH=. python scripts/screener/ai_app_scan.py
输入: 通过fundamental_sentry的25只AI应用软件股
输出: 三分支(趋势加速/底部反转/顶部卖出) + 全量评分表
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
from scripts.screener.strategy_engine import StrategyEngine
from scripts.screener.multi_timeframe_scorer import MultiTimeframeScorer
from scripts.core.tushare_client import TushareClient

# fundamental_sentry --batch 通过的25只 (A级在前)
PASSED = {
    '002230': 'B', '688111': 'A', '300033': 'A', '600570': 'B', '688318': 'B',
    '300496': 'B', '688088': 'B', '002315': 'A', '300785': 'B', '301171': 'B',
    '300378': 'B', '300687': 'B', '300170': 'B', '688083': 'B', '688507': 'B',
    '301269': 'B', '301095': 'B', '300253': 'B', '002777': 'B', '603039': 'B',
    '300454': 'B', '002415': 'A', '688095': 'B', '603383': 'B', '601360': 'B',
}

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)


def main():
    codes = list(PASSED.keys())

    # 名称映射（代码-名称核实铁律）
    name_map, ind_map = {}, {}
    sl = TushareClient().get_stock_list()
    for _, r in sl.iterrows():
        c = str(r.get('symbol', r.get('ts_code', ''))).split('.')[0]
        name_map[c] = r.get('name', '')
        ind_map[c] = r.get('industry', '')

    scorer = V3DerivativeScorer(data_root=Path('data/store'))
    engine = StrategyEngine(scorer)
    mtf = MultiTimeframeScorer(data_root=Path('data/store'))

    print('=' * 100)
    print('AI应用软件板块 — 25只基本面合格股技术面扫描')
    print('=' * 100)

    accel, reversal, top_sell = scorer.score_dual_branch(codes, min_score=4.0)

    if len(accel) > 0:
        accel = mtf.augment_v3_batch(accel)
    if len(reversal) > 0:
        reversal = mtf.augment_v3_batch(reversal)

    def deco(df):
        if len(df) == 0:
            return df
        df = df.copy()
        df['name'] = df['code'].map(name_map)
        df['industry'] = df['code'].map(ind_map)
        df['fund'] = df['code'].map(PASSED)
        return df

    print('\n【分支1: 趋势加速 🚀】')
    if len(accel):
        cols = [c for c in ['code', 'name', 'industry', 'v3_raw', 'v3_total', 'f1', 'f2',
                            'ret20_0', 'trend_phase', 'above_ma20', 'vol_ratio', 'fund',
                            'mtf_label', 'mtf_conf'] if c in accel.columns]
        print(deco(accel)[cols].to_string(index=False))
    else:
        print('  (无)')

    print('\n【分支2: 底部反转 🔄】')
    if len(reversal):
        cols = [c for c in ['code', 'name', 'industry', 'v3_raw', 'v3_total', 'f1', 'f2',
                            'dh180', 'reversal_level', 'above_ma20', 'vol_ratio', 'fund',
                            'mtf_label', 'mtf_conf'] if c in reversal.columns]
        print(deco(reversal)[cols].to_string(index=False))
    else:
        print('  (无)')

    print('\n【分支3: 顶部卖出 ⚠️（本板块内部提示，不推荐）】')
    if len(top_sell):
        cols = [c for c in ['code', 'name', 'v3_raw', 'f1', 'f2', 'ret20_0', 'fund'] if c in top_sell.columns]
        print(deco(top_sell)[cols].to_string(index=False))
    else:
        print('  (无)')

    # 全量评分表(含不在任何分支的票)
    print('\n【全量评分明细】')
    batch = scorer.score_batch(codes)
    if len(batch):
        batch = batch.copy()
        batch['name'] = batch['code'].map(name_map)
        batch['industry'] = batch['code'].map(ind_map)
        batch['fund'] = batch['code'].map(PASSED)
        cols = [c for c in ['code', 'name', 'industry', 'fund', 'v3_raw', 'v3_total',
                            'f1', 'f2', 'above_ma20', 'ret20_0', 'dh180', 'vol_ratio',
                            'macd_hist', 'signal'] if c in batch.columns]
        print(batch.sort_values('v3_total', ascending=False)[cols].to_string(index=False))

    # 策略引擎v4评分(底反/主升浪F) — 对每只票跑full_evaluation
    print('\n【策略引擎v4 full_evaluation (重点票)】')
    for _, row in batch.sort_values('v3_total', ascending=False).head(12).iterrows():
        code = row['code']
        try:
            df = scorer.get_kline_df(code) if hasattr(scorer, 'get_kline_df') else None
            v3s = scorer.score_stock(code, df)
            full = engine.full_evaluation(v3s)
            br = full.get('bottom_reversal', {})
            up = full.get('uptrend_F', br if isinstance(br, dict) else {})
            print(f"{code} {name_map.get(code, '')}: "
                  f"v3_total={getattr(v3s, 'total', '?')} | "
                  f"底反={br.get('total', br.get('score', '?'))} ({br.get('label', '')}) | "
                  f"主升浪F={up.get('total', up.get('score', '?'))} ({up.get('label', '')})")
        except Exception as e:
            print(f"{code}: 评估失败 {e}")


if __name__ == '__main__':
    main()
