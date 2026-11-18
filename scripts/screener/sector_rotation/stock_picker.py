"""
科技板块轮动监控系统 - 个股推荐模块
板块信号触发后，在板块内筛选最强个股
复用v3_derivative_scorer的评分体系 + 资金流过滤
"""
import logging
import sys
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from scripts.screener.sector_rotation.config import SECTORS, SECTOR_DEFS

logger = logging.getLogger(__name__)


def _rate_limit():
    import time
    time.sleep(0.35)


def score_stock(code: str, daily: pd.DataFrame, moneyflow: pd.DataFrame) -> Dict:
    """
    个股评分（简化版v3逻辑）
    维度：趋势(30) + 动量(25) + 资金(25) + 位置(20)
    """
    if daily is None or daily.empty or len(daily) < 20:
        return {"score": 0, "signal": "数据不足"}

    close = daily['close'].values
    n = len(close)

    # 1. 趋势（0-30）：f'方向 + MACD
    try:
        from scipy.signal import savgol_filter
        log_close = np.log(close)
        window = min(21, n // 2 * 2 - 1)
        if window < 5:
            window = 5
        smoothed = savgol_filter(log_close, window_length=window, polyorder=3)
        f_prime = np.gradient(smoothed)[-1]
        f_double_prime = np.gradient(np.gradient(smoothed))[-1]
    except:
        f_prime = 0
        f_double_prime = 0

    # f'评分
    if f_prime > 0.005:
        fp_s = 10
    elif f_prime > 0:
        fp_s = 6
    elif f_prime > -0.003:
        fp_s = 3
    else:
        fp_s = 0

    # f''评分
    if f_double_prime > 0:
        fdp_s = 8
    elif f_double_prime > -0.001:
        fdp_s = 4
    else:
        fdp_s = 0

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()

    if dif.iloc[-1] > dea.iloc[-1]:
        macd_s = 12
    elif abs(dif.iloc[-1] - dea.iloc[-1]) / (abs(dea.iloc[-1]) + 0.001) < 0.05:
        macd_s = 6  # 即将金叉
    else:
        macd_s = 0

    trend_score = fp_s + fdp_s + macd_s

    # 2. 动量（0-25）：近期涨跌幅 + 放量
    if n >= 5:
        ret5 = (close[-1] / close[-5] - 1) * 100
    else:
        ret5 = 0
    if n >= 10:
        ret10 = (close[-1] / close[-10] - 1) * 100
    else:
        ret10 = 0

    # 温和上涨最好
    if 2 < ret5 < 10:
        mom_s = 15
    elif 0 < ret5 <= 2:
        mom_s = 10
    elif -3 < ret5 <= 0:
        mom_s = 5
    elif ret5 > 10:
        mom_s = 8  # 涨太多要小心
    else:
        mom_s = 0

    # 放量
    if 'amount' in daily.columns and n >= 10:
        vol_5 = daily['amount'].tail(5).mean()
        vol_10 = daily['amount'].tail(10).mean()
        vol_ratio = vol_5 / vol_10 if vol_10 > 0 else 1
        if vol_ratio > 1.5:
            mom_s += 10
        elif vol_ratio > 1.2:
            mom_s += 6
        elif vol_ratio > 1.0:
            mom_s += 3

    mom_score = min(25, mom_s)

    # 3. 资金（0-25）
    fund_score = 10  # 基准
    if moneyflow is not None and not moneyflow.empty and len(moneyflow) >= 3:
        recent = moneyflow.tail(3)
        net_3d = recent['net_mf_amount'].sum()
        if net_3d > 0:
            fund_score += 10
        elif net_3d < 0:
            fund_score -= 5

        # 超大单方向
        if 'buy_elg_amount' in recent.columns:
            elg_net = (recent['buy_elg_amount'] - recent['sell_elg_amount']).sum()
            if elg_net > 0:
                fund_score += 5

    fund_score = min(25, max(0, fund_score))

    # 4. 位置（0-20）
    if n >= 20:
        ret20 = (close[-1] / close[-20] - 1) * 100
    else:
        ret20 = 0

    if 0 <= ret20 < 5:
        pos_s = 15  # 刚启动
    elif -5 < ret20 < 0:
        pos_s = 12  # 温和回调
    elif 5 <= ret20 < 15:
        pos_s = 8
    elif -10 < ret20 <= -5:
        pos_s = 10
    elif ret20 >= 15:
        pos_s = 3
    else:
        pos_s = 5

    pos_score = min(20, pos_s)

    total = trend_score + mom_score + fund_score + pos_score
    total = min(100, max(0, total))

    # 信号类型
    if f_prime > 0 and f_double_prime > 0:
        sig = "🚀主升浪"
    elif f_prime > 0 and f_double_prime <= 0:
        sig = "⚠️减速"
    elif f_prime <= 0 and f_double_prime > 0:
        sig = "🔄拐点"
    else:
        sig = "🔴下跌"

    return {
        "score": total,
        "signal": sig,
        "f_prime": round(f_prime, 5),
        "f_double_prime": round(f_double_prime, 6),
        "ret5": round(ret5, 2),
        "ret10": round(ret10, 2),
        "ret20": round(ret20, 2),
        "trend": trend_score,
        "momentum": mom_score,
        "fund": fund_score,
        "position": pos_score,
    }


def pick_stocks(sector_name: str, top_n: int = 3) -> List[Dict]:
    """
    在板块内筛选最强个股
    Returns: 按评分排序的个股列表
    """
    sectors = SECTORS if SECTORS else SECTOR_DEFS
    config = sectors.get(sector_name, {})
    stock_codes = list(config.get('stocks', {}).keys())
    if not stock_codes:
        stock_codes = config.get('seed_stocks', [])

    if not stock_codes:
        return []

    pro = ts.pro_api()
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=120)).strftime('%Y%m%d')

    results = []
    for code in stock_codes:
        try:
            _rate_limit()
            daily = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            if daily is None or daily.empty:
                continue
            daily = daily.sort_values('trade_date').reset_index(drop=True)

            _rate_limit()
            mf = pro.moneyflow(ts_code=code, start_date=start_date, end_date=end_date)
            if mf is not None and not mf.empty:
                mf = mf.sort_values('trade_date').reset_index(drop=True)
            else:
                mf = pd.DataFrame()

            s = score_stock(code, daily, mf)

            # 获取股票名称
            name = config.get('stocks', {}).get(code, code)

            results.append({
                "code": code,
                "name": name,
                "score": s['score'],
                "signal": s['signal'],
                "f_prime": s['f_prime'],
                "f_double_prime": s['f_double_prime'],
                "ret5": s['ret5'],
                "ret10": s['ret10'],
                "ret20": s['ret20'],
                "close": daily['close'].iloc[-1],
                "pct_chg": daily['pct_chg'].iloc[-1],
            })
        except Exception as e:
            logger.warning(f"个股评分失败 {code}: {e}")
            continue

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]


def format_stock_picks(picks: List[Dict]) -> str:
    """格式化个股推荐"""
    if not picks:
        return "  无推荐"

    lines = []
    for p in picks:
        lines.append(
            f"  {p['name']:<8}({p['code']}) {p['score']}分 {p['signal']} "
            f"现价{p['close']:.2f} 5日{p['ret5']:+.1f}% 20日{p['ret20']:+.1f}%"
        )
    return "\n".join(lines)
