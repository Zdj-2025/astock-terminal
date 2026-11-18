#!/usr/bin/env python3
"""分钟顶背离检测器 — 持仓卖出的重要预警信号(2026-09-09).

原理: 多周期MACD(12,26,9)顶背离 = 价格创新高但DIF峰下降 = 上涨动能衰竭.
双确认规则(用户2026-08-20数据港教学实证): 60min+30min同时顶背离才有效,
单30min信号无效. 15min最灵敏,仅作前哨提示.

新鲜度约束(memory: 背离须查过期): 最后一个DIF峰必须在近N根K线内,
过期的背离不再报警. 峰检测argrelextrema order=5 + 剔除右边界clip假峰:
峰须有ORDER根真实右侧bar才成立, 与divergence_detector._find_peaks的
range(order, len-order)语义一致(见detect_div_at_freq内注释).

数据schema兼容: 个股baostock(date,time) / ETF tushare(datetime) / 腾讯备用(两者).

用法:
  from scripts.analyzers.minute_top_divergence import detect_top_divergence, scan_position
  r = detect_top_divergence('600522')        # 单只全频次
  rows = scan_position()                      # 全持仓扫描
"""
import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
KDIR = os.path.join(PROJECT_ROOT, 'data/store/kline')

# 每频次的新鲜度: 最后DIF峰距今不超过N根(≈2周交易日)
FRESHNESS_BARS = {'60': 24, '30': 48, '15': 96}   # 24根60min≈6交易日, 48根30min≈6交易日
MIN_BARS = 60            # 每频次最少K线数(60min下≈15交易日MACD才稳定)
STRENGTH_MIN = 0.05      # DIF峰降幅>5%才算有效背离(过滤噪声)
ORDER = 5                # DIF峰确认所需的右侧真实bar数(与divergence_detector._find_peaks一致)


def _load_minute(code, freq, live=False):
    """读分钟parquet, 兼容个股(date,time)/ETF(datetime)两种schema. 返回(close序列, 时间标签).

    live=True: 盘中模式 — 从腾讯拉当日最新bar拼接到parquet历史尾部(2026-09-09:
    baostock个股分钟滞后一个交易日,盘中盯盘需要实时). 拼接带除权守卫: 当日腾讯首bar
    开盘价与parquet最后收盘偏离>11% → 疑似当日除权 → 放弃拼接用纯历史(宁陈旧不错乱).
    """
    path = os.path.join(KDIR, code, f'{freq}min.parquet')
    if not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path)
    if len(df) < MIN_BARS:
        return None, None
    close = df['close'].astype(float).reset_index(drop=True)
    if 'datetime' in df.columns:
        label = df['datetime'].astype(str).reset_index(drop=True)
        last_date = str(df['datetime'].iloc[-1])[:10].replace('-', '')
    else:
        label = (df['date'].astype(str) + ' ' + df['time'].astype(str).str[8:10]
                 + ':' + df['time'].astype(str).str[10:12]).reset_index(drop=True)
        last_date = str(df['date'].iloc[-1]).replace('-', '')[:8]

    if live:
        try:
            from scripts.data.minute_data import fetch_tencent_minute
            tdf = fetch_tencent_minute(code, freq, 3)
            if tdf is not None and len(tdf):
                today_rows = tdf[tdf['date'] > last_date]
                if len(today_rows):
                    # 除权守卫: 当日首bar开盘 vs 历史末bar收盘, 偏离>11%疑似除权
                    first_open = float(today_rows['open'].iloc[0])
                    last_close = float(close.iloc[-1])
                    if abs(first_open / last_close - 1) <= 0.11:
                        close = pd.concat([close, today_rows['close'].astype(float)],
                                          ignore_index=True)
                        label = pd.concat([label, today_rows['datetime'].astype(str)],
                                          ignore_index=True)
        except Exception:
            pass  # 盘中叠加失败不影响离线检测
    return close, label


def _macd_dif(close):
    """标准MACD(12,26,9)的DIF线."""
    s = pd.Series(close)
    return (s.ewm(span=12, adjust=False).mean()
            - s.ewm(span=26, adjust=False).mean()).values


def _hist(close):
    """MACD柱(DIF-DEA)."""
    dif = _macd_dif(close)
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    return dif - dea


def detect_div_at_freq(code, freq, live=False):
    """单频次顶背离检测. 返回dict或None(无数据/无背离).

    顶背离定义: 存在相邻两个DIF峰p1<p2(时间序), 满足
      价格: close[p2] > close[p1] (新高)
      DIF:  dif[p2] < dif[p1]   (动能峰下降)
      新鲜: p2 距最新K线 <= FRESHNESS_BARS[freq] 根
      强度: (dif[p1]-dif[p2])/|dif[p1]| >= STRENGTH_MIN
    live=True时叠加腾讯当日实时bar(盘中盯盘用).
    """
    close, label = _load_minute(code, freq, live=live)
    if close is None:
        return None
    c = close.values
    dif = _macd_dif(c)
    n = len(c)
    # 峰检测: scipy argrelextrema 默认 mode='clip' 在右边界用边值补齐, 会让 DIF 峰贴到
    # len-2(右侧仅1根真实bar)就被当成"已确认峰" → 提前3-4根bar误报(2026-09-10审计,
    # 603881/600522 真实回放: 主升中继误报, 价格继续创新高)。修复: 强制峰索引<=n-1-ORDER,
    # 峰须有 ORDER 根真实右侧bar, 与 divergence_detector._find_peaks 的 range(order,len-order)
    # 语义一致。代价是信号滞后~ORDER根(60min≈1天), 但误报消除。
    peaks = argrelextrema(dif, np.greater, order=ORDER)[0]
    peaks = peaks[peaks <= n - 1 - ORDER]
    if len(peaks) < 2:
        return None
    p1, p2 = peaks[-2], peaks[-1]
    bars_ago = n - 1 - p2
    fresh = bars_ago <= FRESHNESS_BARS[freq]
    price_hh = c[p2] > c[p1]
    dif_lh = dif[p2] < dif[p1]
    strength = (dif[p1] - dif[p2]) / (abs(dif[p1]) + 1e-9)
    is_div = price_hh and dif_lh and strength >= STRENGTH_MIN and fresh
    # 红柱状态(辅助): 最近5根hist是否连续缩短(动能衰减)
    h = _hist(c)
    h5 = h[-6:-1] if len(h) >= 6 else h
    fading = bool(len(h5) >= 3 and np.all(np.diff(h5) < 0) and h5[-1] > 0)
    return {
        'code': code, 'freq': freq, 'n': n,
        'is_div': bool(is_div),
        'fresh': bool(fresh), 'price_hh': bool(price_hh), 'dif_lh': bool(dif_lh),
        'strength': round(float(strength), 3),
        'bars_ago': int(bars_ago),
        'peak1_price': round(float(c[p1]), 2), 'peak2_price': round(float(c[p2]), 2),
        'peak1_dif': round(float(dif[p1]), 4), 'peak2_dif': round(float(dif[p2]), 4),
        'peak2_time': str(label.iloc[p2]) if label is not None else '',
        'last_time': str(label.iloc[-1]) if label is not None else '',
        'last_close': round(float(c[-1]), 2),
        'hist_fading': fading,           # 红柱连续衰减(辅助确认)
        'dif_now': round(float(dif[-1]), 4),
        'above_zero': bool(dif[-1] > 0),
    }


def detect_top_divergence(code, freqs=('60', '30', '15'), live=False):
    """单只股票多频次顶背离. 返回 {'code', 'per_freq', 'level'}.

    level分级(双确认规则, 2026-08-20数据港教学: 60+30双确认背离才有效):
      🔴 强预警: 60min+30min同时顶背离
      🟡 观察:   仅单周期顶背离(60或30其一)
      🔵 前哨:   仅15min背离(最灵敏,只提示)
      ⚪ 无
    """
    per_freq = {}
    for f in freqs:
        r = detect_div_at_freq(code, f, live=live)
        if r:
            per_freq[f] = r
    d60 = per_freq.get('60', {}).get('is_div', False)
    d30 = per_freq.get('30', {}).get('is_div', False)
    d15 = per_freq.get('15', {}).get('is_div', False)
    if d60 and d30:
        level = '🔴强预警'
    elif d60 or d30:
        level = '🟡观察'
    elif d15:
        level = '🔵前哨'
    else:
        level = '⚪无'
    return {'code': code, 'per_freq': per_freq, 'level': level,
            'd60': d60, 'd30': d30, 'd15': d15}


def scan_position(codes=None, portfolio='config/portfolio.xlsx', live=False):
    """全持仓扫描. codes为None时读portfolio.xlsx. 返回[dict],按级别排序(🔴在前)."""
    if codes is None:
        try:
            os.chdir(PROJECT_ROOT)
            df = pd.read_excel(portfolio)
            codes = [str(c).zfill(6) for c in df['code']]
        except Exception:
            codes = []
    order = {'🔴强预警': 0, '🟡观察': 1, '🔵前哨': 2, '⚪无': 3}
    results = [detect_top_divergence(c, live=live) for c in codes]
    results.sort(key=lambda r: order.get(r['level'], 9))
    return results


def format_scan_table(results, name_map=None):
    """扫描结果→可读表格字符串."""
    name_map = name_map or {}
    lines = [f"{'级别':<6}{'代码':<9}{'名称':<10}{'60min':<14}{'30min':<14}{'15min':<14}{'红柱衰减'}"]
    lines.append('-' * 78)
    for r in results:
        code = r['code']
        nm = name_map.get(code, '')[:9]
        cells = []
        for f in ('60', '30', '15'):
            d = r['per_freq'].get(f)
            if d is None:
                cells.append('无数据')
            elif d['is_div']:
                cells.append(f"背离{d['bars_ago']}根前")
            elif not d['fresh'] and d['dif_lh'] and d['price_hh']:
                cells.append(f"过期({d['bars_ago']}根)")
            else:
                cells.append('—')
        fade = any(r['per_freq'].get(f, {}).get('hist_fading') for f in ('60', '30', '15'))
        lines.append(f"{r['level']:<6}{code:<9}{nm:<10}{cells[0]:<14}{cells[1]:<14}{cells[2]:<14}{'⚠️' if fade else ''}")
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        code = sys.argv[1].zfill(6)
        r = detect_top_divergence(code)
        print(f"\n{code} 分钟顶背离 | 级别: {r['level']}")
        for f in ('60', '30', '15'):
            d = r['per_freq'].get(f)
            if d is None:
                print(f"  {f}min: 无数据/不足{MIN_BARS}根")
            else:
                flag = '✅背离' if d['is_div'] else '  —'
                print(f"  {f:>3}min: {flag} 强度{d['strength']:+.2f} "
                      f"峰2距今{d['bars_ago']}根 价{d['peak1_price']}→{d['peak2_price']} "
                      f"DIF{d['peak1_dif']}→{d['peak2_dif']}"
                      f"{' 红柱衰减⚠️' if d['hist_fading'] else ''} 数据到{d['last_time']}")
    else:
        results = scan_position()
        print(format_scan_table(results))
