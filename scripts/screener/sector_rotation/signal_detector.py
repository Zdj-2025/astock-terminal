"""
科技板块轮动监控系统 - 信号检测器 v3
核心改进：
1. 过热信号：高位+量能衰减+资金流出，不再只看涨幅
2. 新增"强势"信号（高分+多维度共振）
3. 更精细的信号分级
"""
import logging
from typing import Dict

from .scorer import score_sector

logger = logging.getLogger(__name__)

# 信号优先级顺序
SIGNAL_ORDER = {
    "观望": 0, "关注": 1, "潜伏": 2, "启动": 3,
    "强势": 4, "过热": 5, "冷却": 6
}


def detect_signal(score_result: Dict, prev_signal: str = None) -> Dict:
    total = score_result['total_score']
    pv = score_result['price_volume']
    ff = score_result['fund_flow']
    tech = score_result['technicals']
    br = score_result['breadth']

    # 各维度是否偏强
    pv_strong = pv.get('score', 0) >= 20
    fund_positive = ff.get('net_5d', 0) > 0 or ff.get('consecutive_days', 0) >= 2
    tech_strong = tech.get('f_prime', 0) > 0 and tech.get('f_double_prime', 0) > 0
    macd_good = tech.get('macd_state') in ('红柱扩大', '红柱缩短', '即将金叉')

    # 计算共振维度数
    resonance = sum([pv_strong, fund_positive, tech_strong, macd_good])

    signal = "观望"
    emoji = "⚪"
    action = "等待信号"

    # 🔴 冷却：趋势走坏+资金流出+无量能
    if (ff.get('consecutive_out_days', 0) >= 3 and
            tech.get('f_prime', 0) < -0.002 and
            pv.get('max_daily_vol', 0) < 1.2 and
            pv.get('surge_days', 0) == 0):
        signal = "冷却"
        emoji = "🔴"
        action = "离场"

    # 🟠 过热（高位+资金大额流出+缩量）
    elif (pv.get('ret20', 0) > 20 and ff.get('net_5d', 0) < -1 and
              pv.get('max_daily_vol', 0) < 1.3):
        signal = "过热"
        emoji = "🟠"
        action = "注意风险，考虑减仓"

    # 🟣 强势：高分+多维度共振（最可靠信号）
    elif (total >= 72 and resonance >= 3 and tech_strong and fund_positive):
        signal = "强势"
        emoji = "🟣"
        action = "坚定持有"

    # 🔵 启动：总分够 + 技术确认 + 有异动或资金
    elif (total >= 55 and
          (tech_strong or macd_good) and
          (pv.get('surge_days', 0) >= 1 or ff.get('consecutive_days', 0) >= 2)):
        signal = "启动"
        emoji = "🔵"
        action = "建仓"

    # 🟢 潜伏：有异动+技术拐点
    elif (total >= 45 and
          (pv.get('surge_days', 0) >= 1 or pv.get('max_daily_vol', 0) > 1.4) and
          (macd_good or tech.get('f_double_prime', 0) > 0)):
        signal = "潜伏"
        emoji = "🟢"
        action = "重点关注，小仓位试水"

    # 🟠 关注：有异动迹象
    elif (total >= 35 and
          (pv.get('surge_days', 0) >= 1 or pv.get('max_daily_vol', 0) > 1.5 or
           ff.get('consecutive_days', 0) >= 3)):
        signal = "关注"
        emoji = "🟠"
        action = "纳入观察"

    # 信号变化
    change = "→"
    if prev_signal and prev_signal != signal:
        curr_rank = SIGNAL_ORDER.get(signal, 0)
        prev_rank = SIGNAL_ORDER.get(prev_signal, 0)
        if curr_rank > prev_rank and signal in ("关注", "潜伏", "启动", "强势"):
            change = "▲"
        elif signal in ("过热", "冷却") and prev_signal in ("关注", "潜伏", "启动", "强势"):
            change = "▼"
        else:
            change = "↕"

    return {
        "signal": signal,
        "emoji": emoji,
        "action": action,
        "change": change,
        "resonance": resonance,
    }


def scan_all_sectors(all_data: Dict[str, Dict], prev_signals: Dict[str, str] = None) -> list:
    if prev_signals is None:
        prev_signals = {}

    results = []
    for name, data in all_data.items():
        try:
            score_result = score_sector(
                etf_daily=data.get('etf_daily'),
                sector_flow=data.get('sector_moneyflow'),
                stocks_daily=data.get('stocks_daily'),
            )
            signal_result = detect_signal(score_result, prev_signals.get(name))

            results.append({
                "sector": name,
                "total_score": score_result['total_score'],
                "signal": signal_result['signal'],
                "emoji": signal_result['emoji'],
                "action": signal_result['action'],
                "change": signal_result['change'],
                "resonance": signal_result['resonance'],
                "price_volume": score_result['price_volume'],
                "fund_flow": score_result['fund_flow'],
                "technicals": score_result['technicals'],
                "breadth": score_result['breadth'],
            })
        except Exception as e:
            logger.error(f"评分失败 {name}: {e}", exc_info=True)
            results.append({
                "sector": name, "total_score": 0,
                "signal": "错误", "emoji": "❌", "action": str(e), "change": "→",
                "resonance": 0,
                "price_volume": {"score": 0}, "fund_flow": {"score": 0},
                "technicals": {"score": 0}, "breadth": {"score": 0},
            })

    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results
