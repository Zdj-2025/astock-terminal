#!/usr/bin/env python3
"""分钟级MACD顶背离检测器 — P1买入择时把关(2026-06-16沃顿教训自动化)。

日线V3选股后, 用60/30分钟背离过滤"短线见顶"标的。
配合 memory `buy_timing_minute_divergence.md` 纪律, 把"日线选股≠短线买点"自动化。

分钟线源: 优先Baostock(无限流), AKShare备选, Tushare最后(限流)。

用法:
    from scripts.core.divergence_detector import check_minute_divergence
    r = check_minute_divergence('000920')  # {has_top_div, m60, m30, hist_decay, detail}
    # CLI: PYTHONPATH=. python scripts/core/divergence_detector.py 000920
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

_client = None


def _get_client():
    global _client
    if _client is None:
        from scripts.core.tushare_client import TushareClient
        _client = TushareClient()
    return _client


def _macd(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def _detect_top_div(df, w=5):
    """单周期顶背离检测。

    顶背离 = 价格波段新高 但 DIF 下降 + HIST红柱连续衰竭。
    Returns: (has_div: bool, detail: str, hist_decay: bool)
    """
    if df is None or len(df) < 30:
        return False, "数据不足", False
    df = df.copy().dropna(subset=['close']).reset_index(drop=True)
    dif, dea, hist = _macd(df['close'])
    df['DIF'], df['DEA'], df['HIST'] = dif, dea, hist
    # 波段高点(窗口w, center)
    df['isH'] = df['high'].rolling(w, center=True).max() == df['high']
    highs = df[df['isH'] & df['DIF'].notna()]
    if len(highs) < 2:
        return False, "波段高点不足", False
    a = highs.iloc[-2]
    b = highs.iloc[-1]
    has_div = bool(b['high'] > a['high'] and b['DIF'] < a['DIF'])
    detail = (f"价新高{b['high']:.2f}>{a['high']:.2f} DIF降{b['DIF']:.4f}<{a['DIF']:.4f}"
              if has_div else f"最新高{b['high']:.2f} DIF{b['DIF']:.4f}")
    # HIST红柱连续衰竭(动能见顶)
    h5 = df['HIST'].tail(5).values
    hist_decay = bool(len(h5) >= 3 and h5[-1] < h5[-2] < h5[-3] and h5[-1] > 0)
    return has_div, detail, hist_decay


def check_minute_divergence(code, client=None):
    """检查60/30分钟顶背离。

    Returns:
        {has_top_div, m60, m30, m60_latest, m30_latest, hist_decay, detail, stale}
        has_top_div=True → 短线见顶风险, 不应追买
        stale=True → 分钟线数据滞后(>2日), 背离结论需在线核实
    """
    c = client or _get_client()
    result = {"has_top_div": False, "m60": "", "m30": "",
              "m60_latest": "", "m30_latest": "",
              "hist_decay": False, "detail": "", "stale": False}
    for freq, key in [("60", "m60"), ("30", "m30")]:
        try:
            df = c.get_kline_minute(code, freq=freq, source='auto')
            latest = ""
            if df is not None and len(df) and 'date' in df.columns:
                latest = pd.to_datetime(df['date']).max().strftime("%Y-%m-%d %H:%M")
            result[f"{key}_latest"] = latest
            has_div, det, decay = _detect_top_div(df)
            result[key] = det
            if decay:
                result["hist_decay"] = True
            if has_div:
                result["has_top_div"] = True
                result["detail"] += f"{freq}min顶背离; "
        except Exception:
            result[key] = "获取失败"
    # 数据新鲜度: 任一周期的最新分钟线 >2天前 → stale
    try:
        from datetime import datetime, timedelta
        now = datetime.now()
        for k in ("m60_latest", "m30_latest"):
            v = result.get(k, "")
            if v:
                lt = datetime.strptime(v[:16], "%Y-%m-%d %H:%M")
                if (now - lt).days > 2:
                    result["stale"] = True
    except Exception:
        pass
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分钟MACD顶背离检测")
    parser.add_argument("code", help="股票代码(纯数字)")
    args = parser.parse_args()
    r = check_minute_divergence(args.code)
    print(f"{args.code} 分钟背离检测:")
    print(f"  60min: {r['m60']}")
    print(f"  30min: {r['m30']}")
    print(f"  HIST衰竭: {r['hist_decay']}")
    print(f"  {'🚫短线见顶(有顶背离)' if r['has_top_div'] else '✅无顶背离'}")
