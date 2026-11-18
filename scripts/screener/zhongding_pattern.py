# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

#!/usr/bin/env python3
"""中鼎模式筛选器 — 寻找"5浪走完+突破W4高点"的股票

形态特征（参考中鼎股份5/20-5/21）：
1. 日线5浪下跌走完（或即将走完）
2. 5浪结束后出现初次拉升+回踩
3. 二次拉升突破W4高点

筛选维度：
- 形态匹配度
- 行业/板块
- 市值规模
- 基本面（PE/PB/营收增长）
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from scripts.core.data_store import DataStore
from scripts.analyzers.elliott_wave import ElliottWaveAnalyzer


def check_zhongding_pattern(df: pd.DataFrame, zigzag_pct: float = 0.04) -> Optional[dict]:
    """检查是否符合"中鼎模式"。

    条件：
    1. 检测到5浪下跌结构
    2. W5结束后有新上升浪的迹象
    3. 新上升浪的第二波突破W4高点

    Returns:
        dict with pattern info, or None if no match
    """
    if df is None or len(df) < 60:
        return None

    analyzer = ElliottWaveAnalyzer(zigzag_pct=zigzag_pct)
    result = analyzer.analyze(df, direction='down')

    if not result.waves or len(result.waves) < 5:
        return None

    # 检查是下跌5浪
    if result.direction != 'down':
        return None

    w4 = result.waves[3]  # W4调整浪
    w5 = result.waves[4]  # W5最后一跌

    # W4高点（下跌浪中的反弹高点）
    w4_high = w4.end_price

    # W5结束后的数据
    w5_end_idx = int(w5.end_idx)
    if w5_end_idx >= len(df) - 3:
        return None  # W5太靠近末尾，没有足够后续数据

    # W5结束后的走势
    after_w5 = df.iloc[w5_end_idx:]
    if len(after_w5) < 3:
        return None

    # 找W5结束后的最高点（新高）
    after_high = after_w5['high'].max()
    after_high_idx = after_w5['high'].idxmax()
    after_high_date = str(df.loc[after_high_idx, 'date'])[:10]

    # 判断是否突破W4高点
    breakout = after_high > w4_high
    breakout_pct = (after_high / w4_high - 1) * 100

    # 找W5之后的回踩低点（第一次拉升后的回调）
    # 找到第一个局部高点后的回踩
    after_prices = after_w5['close'].values
    after_lows = after_w5['low'].values

    # 找回踩：在突破W4高点之前是否有回踩
    # 简化：看W5底之后到突破之间的最低点
    w5_bottom_price = w5.end_price

    # 找初次拉升的高点（在最终突破之前的局部高点）
    first_pullback_low = None
    first_rally_high = None

    # 扫描W5后的K线
    for i in range(1, len(after_w5)):
        prev_close = after_prices[i-1]
        curr_close = after_prices[i]

        # 第一次拉升的高点
        if first_rally_high is None:
            if i < len(after_w5) - 1:
                # 找到连续上涨后的最高点
                if after_w5.iloc[i]['high'] > after_w5.iloc[i-1]['high']:
                    continue
                else:
                    first_rally_high = after_w5.iloc[i-1]['high']
            else:
                first_rally_high = after_w5.iloc[i]['high']

        # 拉升后的回踩
        if first_rally_high is not None and first_pullback_low is None:
            if curr_close < prev_close:
                first_pullback_low = after_lows[i]

    # 评分
    score = 0
    reasons = []

    # 1. 铁律通过数
    iron_pass = sum(1 for _, p, _ in result.iron_rules if p)
    if iron_pass >= 2:
        score += 20
        reasons.append(f"铁律{iron_pass}/3")
    
    # 2. 评分
    if result.total_score > 0.6:
        score += 20
        reasons.append(f"评分{result.total_score:.2f}")

    # 3. 突破W4高点
    if breakout:
        score += 30
        if breakout_pct < 10:
            score += 10  # 刚突破更好（没涨太多）
            reasons.append(f"刚突破W4 +{breakout_pct:.1f}%")
        else:
            reasons.append(f"已突破W4 +{breakout_pct:.1f}%")
    else:
        gap = (w4_high - after_high) / w4_high * 100
        if gap < 5:
            score += 15  # 接近突破
            reasons.append(f"接近W4 -{gap:.1f}%")

    # 4. 有回踩形态
    if first_pullback_low is not None:
        score += 10
        reasons.append("有回踩")

    # 5. W5结束后放量（新资金进场）
    w5_end_vol = df.iloc[max(0, w5_end_idx-3):w5_end_idx+1]['volume'].mean()
    after_vol = after_w5['volume'].mean()
    if after_vol > w5_end_vol * 1.5:
        score += 10
        reasons.append(f"放量{(after_vol/w5_end_vol):.1f}倍")

    # 只返回有价值的
    if score < 30:
        return None

    return {
        'score': score,
        'wave_score': result.total_score,
        'iron_rules': iron_pass,
        'direction': 'down',
        'w4_high': w4_high,
        'w5_bottom': w5_bottom_price,
        'after_high': after_high,
        'after_high_date': after_high_date,
        'breakout': breakout,
        'breakout_pct': breakout_pct,
        'reasons': ' | '.join(reasons),
        'waves': [(str(w.start_date)[:10], str(w.end_date)[:10],
                   w.start_price, w.end_price, w.label) for w in result.waves],
    }


def scan_market(store: DataStore, symbol_list: List[str] = None) -> pd.DataFrame:
    """全市场扫描中鼎模式。"""
    if symbol_list is None:
        # 从DataStore中找所有有日线数据的标的
        kline_dir = Path(project_root) / "data" / "store" / "kline"
        if kline_dir.exists():
            symbol_list = [d.name for d in kline_dir.iterdir() if d.is_dir()]
        else:
            print("无本地数据")
            return pd.DataFrame()

    results = []
    total = len(symbol_list)

    for i, symbol in enumerate(symbol_list):
        print(f"  [{i+1}/{total}] {symbol}...", end=" ", flush=True)

        try:
            df = store.get_kline(symbol, period='D', auto_update=False)
            if df is None or df.empty:
                print("无数据")
                continue

            # 只看今年数据
            df = df[df['date'] >= '2026-01-01'].reset_index(drop=True)
            if len(df) < 30:
                print("数据太少")
                continue

            match = check_zhongding_pattern(df)
            if match:
                results.append({
                    'symbol': symbol,
                    'score': match['score'],
                    'wave_score': match['wave_score'],
                    'iron_rules': match['iron_rules'],
                    'w4_high': match['w4_high'],
                    'w5_bottom': match['w5_bottom'],
                    'after_high': match['after_high'],
                    'after_high_date': match['after_high_date'],
                    'breakout': match['breakout'],
                    'breakout_pct': match['breakout_pct'],
                    'reasons': match['reasons'],
                    'waves': match['waves'],
                })
                print(f"✅ 评分{match['score']} {'突破' if match['breakout'] else '接近'}")
            else:
                print("—")
        except Exception as e:
            print(f"错误: {str(e)[:40]}")

    if not results:
        print("\n未找到符合条件的标的")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
    return df_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="中鼎模式筛选器")
    parser.add_argument("--scan", action="store_true", help="全市场扫描")
    parser.add_argument("--symbol", default=None, help="单只标的检查")
    parser.add_argument("--top", type=int, default=20, help="显示前N个结果")
    args = parser.parse_args()

    store = DataStore()

    if args.symbol:
        df = store.get_kline(args.symbol, period='D', auto_update=False)
        if df is not None:
            df = df[df['date'] >= '2026-01-01'].reset_index(drop=True)
            match = check_zhongding_pattern(df)
            if match:
                print(f"\n✅ {args.symbol} 匹配中鼎模式！")
                print(f"  评分: {match['score']}")
                print(f"  W4高点: {match['w4_high']:.2f}")
                print(f"  W5底: {match['w5_bottom']:.2f}")
                print(f"  后续高点: {match['after_high']:.2f} ({match['after_high_date']})")
                print(f"  突破: {'是' if match['breakout'] else '否'} ({match['breakout_pct']:+.1f}%)")
                print(f"  原因: {match['reasons']}")
                print(f"\n  浪型:")
                for sd, ed, sp, ep, label in match['waves']:
                    print(f"    {label}: {sd}→{ed}  {sp:.2f}→{ep:.2f}")
            else:
                print(f"❌ {args.symbol} 不匹配中鼎模式")
    elif args.scan:
        print("🔍 全市场扫描中鼎模式...")
        # 扫描本地有数据的标的
        kline_dir = Path(project_root) / "data" / "store" / "kline"
        symbols = [d.name for d in kline_dir.iterdir() if d.is_dir()]
        print(f"  本地有日线数据的标的: {len(symbols)}")

        df_results = scan_market(store, symbols)

        if not df_results.empty:
            print(f"\n{'='*80}")
            print(f"🎯 中鼎模式筛选结果（共{len(df_results)}只）")
            print(f"{'='*80}")
            for _, row in df_results.head(args.top).iterrows():
                print(f"\n  {row['symbol']}  评分={row['score']}  "
                      f"{'✅突破' if row['breakout'] else '⏳接近'} "
                      f"W4={row['w4_high']:.2f} → {row['after_high']:.2f}({row['after_high_date']})")
                print(f"    {row['reasons']}")
