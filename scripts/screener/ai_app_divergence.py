#!/usr/bin/env python3
"""AI应用软件板块补充分析 — 底背离检测 + 市场状态 + 相对强弱排序。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
from scripts.screener.divergence_scanner import detect_double_bottom
from scripts.analyzers.divergence_detector import DivergenceDetector
from scripts.core.tushare_client import TushareClient

PASSED = {
    '002230': 'B', '688111': 'A', '300033': 'A', '600570': 'B', '688318': 'B',
    '300496': 'B', '688088': 'B', '002315': 'A', '300785': 'B', '301171': 'B',
    '300378': 'B', '300687': 'B', '300170': 'B', '688083': 'B', '688507': 'B',
    '301269': 'B', '301095': 'B', '300253': 'B', '002777': 'B', '603039': 'B',
    '300454': 'B', '002415': 'A', '688095': 'B', '603383': 'B', '601360': 'B',
}

STORE = PROJECT_ROOT / 'data/store/kline'

def main():
    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))

    # 市场状态
    scorer = V3DerivativeScorer(data_root=PROJECT_ROOT / 'data/store')
    regime = scorer.get_market_regime()
    print('【市场状态】', regime.get('name'), regime)
    print()

    det = DivergenceDetector()
    rows = []
    for code in PASSED:
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            for col in ['datetime', 'day', 'trade_date']:
                if col in df.columns and col != 'date':
                    df = df.rename(columns={col: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            close = df['close'].astype(float)
            low = df['low'].astype(float)

            sig = det.detect_divergence(df, period='D')
            db = detect_double_bottom(df)

            # 布林位置
            ma20 = close.rolling(20).mean()
            sd20 = close.rolling(20).std()
            boll_pos_v = (close.iloc[-1] - ma20.iloc[-1]) / (2 * sd20.iloc[-1] + 1e-9)

            # MACD
            ef = close.ewm(span=12, adjust=False).mean()
            es = close.ewm(span=26, adjust=False).mean()
            dif = ef - es
            dea = dif.ewm(span=9, adjust=False).mean()
            hist = (dif - dea) * 2
            hist_slope = hist.iloc[-1] - hist.iloc[-3]  # 柱变化

            # 距离近期低点(是否创新低后企稳)
            low60 = low.rolling(60).min().iloc[-1]
            off_low = (close.iloc[-1] / low60 - 1) * 100

            div_txt = ''
            if sig.detected and sig.div_type == 'bottom':
                div_txt = ('✅确认' if sig.confirmed else '🔍未确认') + f"({sig.strength})"
            if db:
                div_txt += ' 🥇双底'

            rows.append({
                'code': code, 'name': name_map.get(code, ''),
                'fund': PASSED[code],
                'close': round(float(close.iloc[-1]), 2),
                'ret20_%': round((close.iloc[-1] / close.iloc[-20] - 1) * 100, 1),
                'off_low60_%': round(off_low, 1),
                'vs_MA20': '>MA20' if close.iloc[-1] > ma20.iloc[-1] else '<MA20',
                'boll_pos': round(float(boll_pos_v), 2),
                'macd_hist': round(float(hist.iloc[-1]), 3),
                'hist_slope': round(float(hist_slope), 3),
                'divergence': div_txt or '—',
            })
        except Exception as e:
            import traceback
            print(f'ERR {code}:')
            traceback.print_exc()
            rows.append({'code': code, 'name': name_map.get(code, ''), 'fund': PASSED[code],
                         'divergence': f'ERR {e}'})

    df = pd.DataFrame(rows)
    # 相对强弱排序: 距低点近(ret20浅)+hist收窄优先
    df['_rs'] = df['ret20_%'].fillna(-99) + df['off_low60_%'].fillna(-99) * 0.5
    df = df.sort_values('_rs', ascending=False).drop(columns='_rs')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print('【AI应用软件 25只 — 企稳度排序(距60日低点/MACD柱变化/底背离)】')
    print(df.to_string(index=False))

    print('\n【底背离命中】')
    hits = df[df['divergence'].str.contains('背离|双底|确认', na=False)]
    if len(hits):
        print(hits[['code', 'name', 'fund', 'close', 'ret20_%', 'divergence']].to_string(index=False))
    else:
        print('  (无底背离信号)')

if __name__ == '__main__':
    main()
