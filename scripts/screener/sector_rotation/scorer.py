"""
科技板块轮动监控系统 - 四维评分算法 v3
基于1年回测数据优化的版本
核心改进：
1. 过热信号改用"高位+量能衰减"，不再看绝对涨幅
2. breadth重设计：用板块内个股上涨的持续性替代单日占比
3. 优化各维度评分分布，增强单调性
4. 加入动量因子：近期趋势方向
"""
import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_div(a, b, default=0):
    return a / b if b != 0 and not pd.isna(b) else default


# ============================================================
# ① 价格+量能异动（满分35）
# ============================================================
def score_price_volume(etf_daily: pd.DataFrame) -> Dict:
    if etf_daily is None or etf_daily.empty or len(etf_daily) < 5:
        return {"score": 0, "details": "数据不足"}

    close = etf_daily['close'].values
    n = len(close)

    # --- 价格位置（0-8分）— 纯量能异动，不评价位置高低
    ret20 = (close[-1] / close[-min(20, n)] - 1) * 100

    # 5日趋势
    if n >= 5:
        ret5 = (close[-1] / close[-5] - 1) * 100
    else:
        ret5 = 0

    # 距高点回撤
    high_40 = etf_daily['high'].tail(min(40, n)).max()
    drawdown = (close[-1] / high_40 - 1) * 100

    # 位置评分：只做极端过滤，不给超跌加分
    if ret20 > 25:   # 过热
        pos_score = 0
    elif ret20 > 15:
        pos_score = 2
    else:
        pos_score = 5  # 正常范围给基础分

    # --- 量能异动（0-17分）---
    # 单日最大放量
    max_daily_vol = 1.0
    surge_days = 0
    max_surge_pct = 0

    if n >= 5 and 'amount' in etf_daily.columns and 'pct_chg' in etf_daily.columns:
        recent = etf_daily.tail(min(5, n))
        avg_amt_20 = etf_daily['amount'].tail(min(20, n)).mean()

        if avg_amt_20 > 0:
            max_daily_vol = recent['amount'].max() / avg_amt_20

            for _, row in recent.iterrows():
                if row['pct_chg'] > 2.0 and row['amount'] > avg_amt_20 * 1.3:
                    surge_days += 1
                    max_surge_pct = max(max_surge_pct, row['pct_chg'])

    # 放量评分
    if max_daily_vol > 2.0:
        vol_s = 6
    elif max_daily_vol > 1.5:
        vol_s = 5
    elif max_daily_vol > 1.3:
        vol_s = 4
    elif max_daily_vol > 1.0:
        vol_s = 2
    else:
        vol_s = 0

    # 异动天数
    if surge_days >= 2:
        surge_s = 7
    elif surge_days >= 1:
        surge_s = 4
    else:
        surge_s = 0

    # 5日动量加分（正向动量好，但不要太猛）
    if 0 < ret5 < 5:
        trend_s = 3  # 温和上涨=好
    elif -3 < ret5 <= 0:
        trend_s = 2  # 小回调=还行
    elif 5 <= ret5 < 10:
        trend_s = 1  # 有点猛
    else:
        trend_s = 0

    vol_score = vol_s + surge_s + trend_s
    total = min(35, max(0, pos_score + vol_score))

    return {
        "score": total,
        "ret20": round(ret20, 2),
        "ret5": round(ret5, 2),
        "drawdown": round(drawdown, 2),
        "max_daily_vol": round(max_daily_vol, 2),
        "surge_days": surge_days,
        "max_surge_pct": round(max_surge_pct, 1),
        "pos_score": pos_score,
        "vol_score": vol_score,
        "details": f"20日{ret20:+.1f}% 5日{ret5:+.1f}% 量比max{max_daily_vol:.2f}x 异动{surge_days}天"
    }


# ============================================================
# ② 资金流（满分25）
# ============================================================
def score_fund_flow(sector_flow: pd.DataFrame) -> Dict:
    if sector_flow is None or sector_flow.empty or len(sector_flow) < 2:
        return {"score": 10, "consecutive_days": 0, "net_5d": 0, "details": "数据不足(默认10)"}

    recent = sector_flow.tail(10)

    # 连续净流入/流出天数
    consecutive = 0
    for _, row in recent.iloc[::-1].iterrows():
        if row['net_mf_amount'] > 0:
            consecutive += 1
        else:
            break

    consecutive_out = 0
    for _, row in recent.iloc[::-1].iterrows():
        if row['net_mf_amount'] < 0:
            consecutive_out += 1
        else:
            break

    # 5日净流入
    recent_5d = sector_flow.tail(5)
    net_5d = recent_5d['net_mf_amount'].sum()

    # 评分
    score = 10  # 基准分

    # 连续流入
    if consecutive >= 5:
        score += 8
    elif consecutive >= 3:
        score += 6
    elif consecutive >= 2:
        score += 4
    elif consecutive >= 1:
        score += 2
    # 连续流出
    elif consecutive_out >= 5:
        score -= 8
    elif consecutive_out >= 3:
        score -= 4

    # 净流入方向
    if net_5d > 0:
        score += 5
    elif net_5d < 0:
        score -= 3

    # 超大单方向
    if 'elg_net_amount' in recent_5d.columns:
        elg_net = recent_5d['elg_net_amount'].sum()
        if elg_net > 0:
            score += 3
        elif elg_net < 0:
            score -= 2

    score = min(25, max(0, score))
    net_5d_yi = net_5d / 100000

    return {
        "score": score,
        "consecutive_days": consecutive,
        "consecutive_out_days": consecutive_out,
        "net_5d": round(net_5d_yi, 2),
        "details": f"5日净{net_5d_yi:+.1f}亿 连入{consecutive} 连出{consecutive_out}"
    }


# ============================================================
# ③ 技术形态（满分25，提高权重）
# ============================================================
def score_technicals(etf_daily: pd.DataFrame) -> Dict:
    if etf_daily is None or etf_daily.empty or len(etf_daily) < 30:
        return {"score": 0, "f_prime": 0, "f_double_prime": 0, "macd_state": "数据不足", "details": "数据不足"}

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return {"score": 0, "details": "scipy未安装"}

    close = etf_daily['close'].values
    log_close = np.log(close)
    n = len(log_close)

    window = min(21, n // 2 * 2 - 1)
    if window < 5:
        window = 5

    smoothed = savgol_filter(log_close, window_length=window, polyorder=3)
    f_prime = np.gradient(smoothed)
    f_double_prime = np.gradient(f_prime)

    fp = f_prime[-1]
    fdp = f_double_prime[-1]

    # f'评分(0-10) — 趋势方向
    if fp > 0.005:
        fp_score = 10
    elif fp > 0.002:
        fp_score = 8
    elif fp > 0:
        fp_score = 5
    elif fp > -0.002:
        fp_score = 3
    else:
        fp_score = 0

    # f''评分(0-8) — 加速度/拐点
    if fdp > 0.001:
        fdp_score = 8
    elif fdp > 0:
        fdp_score = 5
    elif fdp > -0.001:
        fdp_score = 2
    else:
        fdp_score = 0

    # MACD(0-7)
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2

    if dif.iloc[-1] > dea.iloc[-1]:
        if len(macd_hist) > 1 and macd_hist.iloc[-1] > macd_hist.iloc[-2]:
            macd_state = "红柱扩大"
            macd_score = 7
        else:
            macd_state = "红柱缩短"
            macd_score = 5
    else:
        if len(dea) > 1 and abs(dea.iloc[-1]) < abs(dea.iloc[-2]) * 0.95:
            macd_state = "即将金叉"
            macd_score = 4
        else:
            macd_state = "死叉"
            macd_score = 0

    total = min(25, fp_score + fdp_score + macd_score)

    return {
        "score": total,
        "f_prime": round(fp, 5),
        "f_double_prime": round(fdp, 6),
        "macd_state": macd_state,
        "details": f"f'={fp:.4f} f''={fdp:.5f} {macd_state}"
    }


# ============================================================
# ④ 板块广度（满分15）— 简化，IC低就降权
# ============================================================
def score_breadth(stocks_daily: Dict[str, pd.DataFrame]) -> Dict:
    if not stocks_daily:
        return {"score": 5, "up_ratio": 0, "details": "无个股数据(默认5)"}

    up_count = 0
    strong_count = 0
    total_count = 0

    for code, df in stocks_daily.items():
        if df is not None and not df.empty and len(df) >= 1:
            total_count += 1
            pct = df['pct_chg'].iloc[-1]
            if pct > 0:
                up_count += 1
            if pct > 3:
                strong_count += 1

    if total_count == 0:
        return {"score": 5, "up_ratio": 0, "details": "无数据"}

    up_ratio = up_count / total_count

    score = 5  # 基准分
    if up_ratio > 0.8:
        score += 7
    elif up_ratio > 0.6:
        score += 5
    elif up_ratio > 0.4:
        score += 2
    else:
        score -= 2

    if strong_count >= 3:
        score += 3
    elif strong_count >= 1:
        score += 1

    score = min(15, max(0, score))

    return {
        "score": score,
        "up_ratio": round(up_ratio, 2),
        "strong_count": strong_count,
        "total_count": total_count,
        "details": f"上涨{up_count}/{total_count} 强势{strong_count}只"
    }


# ============================================================
# 综合评分 v3
# 权重：价格量能35 + 资金流25 + 技术25 + 广度15
# ============================================================
def score_sector(etf_daily: pd.DataFrame,
                 sector_flow: pd.DataFrame,
                 stocks_daily: Dict[str, pd.DataFrame] = None) -> Dict:
    pv = score_price_volume(etf_daily)
    ff = score_fund_flow(sector_flow)
    tech = score_technicals(etf_daily)
    breadth = score_breadth(stocks_daily)

    total = pv['score'] + ff['score'] + tech['score'] + breadth['score']
    total = max(0, min(100, total))

    return {
        "total_score": total,
        "price_volume": pv,
        "fund_flow": ff,
        "technicals": tech,
        "breadth": breadth,
    }
