# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

#!/usr/bin/env python3
"""翻倍股扫描器 — 从全市场中筛选具有翻倍潜力的标的。

核心逻辑：
  1. 上升通道（回归斜率正）
  2. 通道宽度 > 50%（下轨到上轨有足够空间）
  3. 当前位置在下半区（≤40%，还没大涨）
  4. 周线MACD金叉或DIF>0（大级别趋势向上）
  5. 行业催化评分（AI/半导体/商业航天/新能源等热门赛道加分）
  6. 综合排名

用法:
    PYTHONPATH=. python scripts/screener/doubler_screener.py
    PYTHONPATH=. python scripts/screener/doubler_screener.py --top 20
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.signal import argrelextrema, savgol_filter

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.analyzers.channel_analyzer import ChannelAnalyzer
from scripts.core.data_store import DataStore


# ====== 行业催化关键词 ======
CATALYSIS_KEYWORDS = {
    'hot': ['半导体', '芯片', 'AI', '人工智能', '算力', '大模型', '机器人',
            '商业航天', '卫星', '军工', '国防', '新能源', '储能', '光伏',
            '自动驾驶', '无人驾驶', '低空经济', '飞行汽车', '量子',
            '数据要素', '数据港', '信创', '国产替代'],
    'warm': ['消费电子', '医疗器械', '创新药', 'CXO', '汽车', '零部件',
             '有色金属', '稀土', '锂电', '充电桩', '风电', '氢能',
             '工业母机', '数控', '传感器', '物联网', '5G', '通信',
             '券商', '保险', '银行'],
    'cold': ['房地产', '基建', '钢铁', '煤炭', '水泥', '造纸', '纺织',
             '农业', '养殖', '白酒', '啤酒'],
}


def score_catalysis(name: str, industry: str = '') -> float:
    """行业催化评分 0-3。"""
    text = (name + industry).upper()
    for kw in CATALYSIS_KEYWORDS['hot']:
        if kw in text:
            return 3.0
    for kw in CATALYSIS_KEYWORDS['warm']:
        if kw in text:
            return 1.5
    return 0.5  # 默认


def check_weekly_macd(df_weekly: pd.DataFrame) -> dict:
    """周线MACD检查。"""
    if df_weekly is None or len(df_weekly) < 30:
        return {'signal': '无数据', 'score': 0}

    close = df_weekly['close'].values
    # 确保升序
    dates = df_weekly['trade_date'].values if 'trade_date' in df_weekly.columns else (
        df_weekly['date'].values if 'date' in df_weekly.columns else None)
    if dates is not None and str(dates[0]) > str(dates[-1]):
        close = close[::-1]

    s = pd.Series(close)
    ema12 = s.ewm(span=12).mean()
    ema26 = s.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()

    score = 0
    signals = []

    if dif.iloc[-1] > dea.iloc[-1]:
        score += 1
        signals.append('金叉')
    if dif.iloc[-1] > 0:
        score += 1
        signals.append('DIF>0')

    return {'signal': '+'.join(signals) if signals else '死叉', 'score': score}


def check_daily_trend(df_daily: pd.DataFrame) -> dict:
    """日线趋势检查（二阶导）。"""
    if df_daily is None or len(df_daily) < 30:
        return {'signal': '无数据', 'score': 0}

    close = df_daily['close'].values
    dates = df_daily['trade_date'].values if 'trade_date' in df_daily.columns else (
        df_daily['date'].values if 'date' in df_daily.columns else None)
    if dates is not None and str(dates[0]) > str(dates[-1]):
        close = close[::-1]

    ln_p = np.log(close)
    window = min(21, len(close) // 4 * 2 + 1)
    if window < 5:
        return {'signal': '数据不足', 'score': 0}

    from scripts.core.derivatives import causal_savgol  # 2026-09-10: 权威因果, 弃中心化
    f1 = causal_savgol(ln_p, window, 3, deriv=1)
    f2 = causal_savgol(ln_p, window, 3, deriv=2)

    if f1[-1] > 0 and f2[-1] > 0:
        return {'signal': '🚀主升浪', 'score': 2}
    elif f1[-1] < 0 and f2[-1] > 0:
        return {'signal': '🔄底部拐点', 'score': 1}
    elif f1[-1] > 0 and f2[-1] < 0:
        return {'signal': '⚠️减速见顶', 'score': 0}
    else:
        return {'signal': '🔴加速下跌', 'score': -1}


def scan_doubler(top_n: int = 20) -> pd.DataFrame:
    """扫描全市场翻倍股。"""
    store = DataStore()
    analyzer = ChannelAnalyzer()

    # 获取全市场股票列表
    try:
        import tushare as ts
        pro = ts.pro_api()
        stock_list = pro.stock_basic(exchange='', list_status='L',
                                      fields='ts_code,symbol,name,industry')
    except Exception:
        print("无法获取股票列表")
        return pd.DataFrame()

    results = []
    total = len(stock_list)
    print(f"扫描全市场 {total} 只股票...")

    for idx, row in stock_list.iterrows():
        code = row['symbol']
        ts_code = row['ts_code']
        name = row['name']
        industry = row.get('industry', '')

        if (idx + 1) % 500 == 0:
            print(f"  进度: {idx+1}/{total}")

        try:
            df = store.get_kline(code, 'D')
            if df is None or len(df) < 120:
                continue

            # 通道分析
            result = analyzer.find_best_channel(df)
            if not result.has_channel:
                continue

            # 只看上升通道
            if result.direction != 'up':
                continue

            # 通道宽度（下轨到上轨的空间）
            channel_width_pct = (result.upper_now / result.lower_now - 1) * 100

            # 筛选条件
            if channel_width_pct < 30:  # 通道宽度至少30%
                continue
            if result.channel_pct > 50:  # 已经涨过半了
                continue
            if result.win_rate_20d < 60:  # 下轨胜率太低
                continue
            if result.touch_count < 3:  # 触碰次数太少
                continue

            # 周线MACD
            df_w = store.get_kline(code, 'W')
            weekly = check_weekly_macd(df_w)

            # 日线趋势
            daily = check_daily_trend(df)

            # 行业催化
            catalysis = score_catalysis(name, industry)

            # 综合评分
            score = 0
            # 通道宽度（越宽越好）
            score += min(channel_width_pct / 20, 3.0)  # 60%宽度=3分
            # 通道位置（越低越好）
            score += (50 - result.channel_pct) / 10  # 位置0%=5分
            # 下轨胜率
            score += result.win_rate_20d / 100 * 2  # 100%=2分
            # 触碰次数
            score += min(result.touch_count / 5, 1.0)
            # 周线MACD
            score += weekly['score']
            # 日线趋势
            score += daily['score']
            # 行业催化
            score += catalysis

            results.append({
                'code': code,
                'ts_code': ts_code,
                'name': name,
                'industry': industry,
                'channel_width': round(channel_width_pct, 1),
                'channel_pct': round(result.channel_pct, 1),
                'lower': round(result.lower_now, 2),
                'upper': round(result.upper_now, 2),
                'win_rate_20d': round(result.win_rate_20d, 0),
                'touches': result.touch_count,
                'weekly': weekly['signal'],
                'daily_trend': daily['signal'],
                'catalysis': catalysis,
                'score': round(score, 1),
                'signal': result.signal[:10],
            })

        except Exception:
            continue

    if not results:
        print("未找到符合条件的标的")
        return pd.DataFrame()

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('score', ascending=False).head(top_n)
    return df_result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=20, help='输出数量')
    args = parser.parse_args()

    df = scan_doubler(top_n=args.top)
    if len(df) > 0:
        print('\n' + '='*80)
        print('🚀 翻倍股扫描结果')
        print('='*80)
        for _, r in df.iterrows():
            print(f"\n⭐ {r['name']}({r['code']}) | {r['industry']}")
            print(f"   通道宽度{r['channel_width']}% | 位置{r['channel_pct']}% | 下轨{r['lower']}→上轨{r['upper']}")
            print(f"   下轨胜率{r['win_rate_20d']:.0f}% | 触碰{r['touches']}次")
            print(f"   周线:{r['weekly']} | 日线:{r['daily_trend']} | 催化:{r['catalysis']}")
            print(f"   综合评分: {r['score']} | 信号: {r['signal']}")
