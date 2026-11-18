#!/usr/bin/env python3
"""市场情绪温度计 — 涨停池四指标 + 绝对数量维度（2026-09-25建·09-30入daily_scan）。

四指标(数据源: 东财zt_pool):
  涨停数     市场热度(绝对量)
  2板+占比   资金接力意愿 = lb_days≥2 / 总涨停
  最高板     空间高度 = max(lb_days)
  行业分散度 主线明确vs轮动 = hy去重数

分级(2026-09-25定):
  🔥高潮: 2板+>30% 且 最高板≥5
  😊升温: 2板+>20% 且 最高板≥4
  😐中性: 2板+>15%
  ❄️冰点: ≤15%
绝对数量维度(2026-09-30加,交接事项): 涨停<40只 → 降一级
  依据: 9月读数里冰点日涨停47-52只仍算"中性/升温"的口径偏乐观,
  绝对量不足时接力比例易被小基数扭曲(少数游资抱团即可拉高2板+占比)。

与看山行"冰点→发酵→高潮→退潮"四阶段互为验证。
用法:
  PYTHONPATH=. python scripts/screener/sentiment_thermometer.py           # 最新交易日
  PYTHONPATH=. python scripts/screener/sentiment_thermometer.py 20260929  # 指定日
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.zt_pool import zt_pool

LEVELS = ['❄️冰点', '😐中性', '😊升温', '🔥高潮']
WEAK_COUNT = 40  # 涨停<40只降一级


def _latest_trade_date():
    """从上证指数parquet取最新交易日(收盘后数据已更新)"""
    import pandas as pd
    p = PROJECT_ROOT / 'data/store/kline/000001/D.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        dcol = [c for c in df.columns if 'date' in c.lower()]
        if dcol:
            return str(df[dcol[0]].iloc[-1])[:10].replace('-', '')
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d')


def _grade(two_board_pct, max_lb, count):
    """三分位判级 + 绝对量降级"""
    if two_board_pct > 0.30 and max_lb >= 5:
        idx = 3
    elif two_board_pct > 0.20 and max_lb >= 4:
        idx = 2
    elif two_board_pct > 0.15:
        idx = 1
    else:
        idx = 0
    demoted = False
    if count < WEAK_COUNT and idx > 0:
        idx -= 1
        demoted = True
    return LEVELS[idx], demoted


def read_thermometer(date=None):
    """返回 (dict指标, 等级str)。涨停池为空返回 (None, '无数据')。"""
    date = date or _latest_trade_date()
    pool = zt_pool(date)
    if not pool:
        return None, '无数据'

    count = len(pool)
    lbs = [int(p.get('lb_days') or 1) for p in pool]
    two_plus = sum(1 for lb in lbs if lb >= 2)
    two_pct = two_plus / count if count else 0
    max_lb = max(lbs) if lbs else 0
    industries = len({p.get('hy') for p in pool if p.get('hy')})

    level, demoted = _grade(two_pct, max_lb, count)
    return {
        'date': date, 'count': count, 'two_board_pct': two_pct,
        'max_lb': max_lb, 'industries': industries,
        'level': level, 'demoted': demoted,
    }, level


if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else None
    m, level = read_thermometer(date)
    if m is None:
        print(f"🌡️ 情绪温度计 | {date or '最新'} 无涨停池数据(非交易日或接口失败)")
        sys.exit(0)
    demo = f" → 已降级(涨停{m['count']}<{WEAK_COUNT}只,小基数接力失真)" if m['demoted'] else ''
    print(f"🌡️ 市场情绪温度计 | {m['date']}")
    print(f"   涨停{m['count']}只 | 2板+占比{m['two_board_pct']:.0%} | "
          f"最高{m['max_lb']}板 | 行业分散{m['industries']}个")
    print(f"   等级: {m['level']}{demo}")
