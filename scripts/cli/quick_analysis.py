"""一键分析入口 — OpenClaw/Claude Code 共用的快速分析工具

自动选择数据源（盘中AKShare/盘后Tushare），读取 watchlist.json 持仓，
输出标准化分析结果。

用法:
    python scripts/cli/quick_analysis.py 603881                  # 快速分析
    python scripts/cli/quick_analysis.py 603881 --mode full      # 完整分析
    python scripts/cli/quick_analysis.py 603881 --mode realtime  # 仅实时行情
    python scripts/cli/quick_analysis.py --all-holdings           # 全部持仓
    python scripts/cli/quick_analysis.py 603881 --json            # JSON输出
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WATCHLIST_PATH = PROJECT_ROOT / "config" / "watchlist.json"


def _is_trading_time() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return 9 <= now.hour <= 15


def _load_holdings() -> dict:
    """从 watchlist.json 加载持仓索引"""
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {h["code"]: h for h in data.get("holdings", [])}


def _data_source_tag() -> tuple:
    """返回 (来源名, 是否盘中)"""
    is_trading = _is_trading_time()
    now = datetime.now()
    tag = f"{now.strftime('%Y-%m-%d %H:%M')} {'AKShare(实时)' if is_trading else 'Tushare(收盘)'}"
    return tag, is_trading


def _get_realtime(symbol: str, is_trading: bool) -> dict:
    """获取实时/最新价格"""
    if is_trading:
        from scripts.core.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        quote = fetcher.get_realtime_quote(symbol)
        if quote:
            return {
                "price": quote.get("price", 0),
                "change_pct": quote.get("change_percent", 0),
                "volume": quote.get("volume", 0),
                "high": quote.get("high", 0),
                "low": quote.get("low", 0),
                "source": "AKShare",
            }

    # 盘后：从 DataStore 取最新K线收盘价
    from scripts.core.data_store import DataStore
    store = DataStore()
    df = store.get_kline(symbol, period="D")
    if df is not None and len(df) > 0:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        chg = (last["close"] / prev["close"] - 1) * 100 if prev["close"] > 0 else 0
        return {
            "price": float(last["close"]),
            "change_pct": round(chg, 2),
            "volume": float(last.get("volume", 0)),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "source": "Tushare",
            "date": str(last["date"]) if "date" in last else "",
        }
    return {"price": 0, "source": "无数据"}


def _quick_score(symbol: str) -> dict:
    """快速4维评分（基于已有K线数据）"""
    from scripts.core.data_store import DataStore
    store = DataStore()
    df = store.get_kline(symbol, period="D")
    if df is None or len(df) < 60:
        return {"ma_score": 0, "macd_score": 0, "note": "数据不足"}

    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float) if "volume" in df.columns else None
    n = len(close)

    # 均线趋势
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:]) if n >= 60 else ma20

    if ma5 > ma10 > ma20 > ma60:
        ma_signal = "多头排列"
        ma_score = 3
    elif ma5 > ma10 > ma20:
        ma_signal = "中多"
        ma_score = 2
    elif ma5 > ma10:
        ma_signal = "短多"
        ma_score = 1
    elif ma5 < ma10 < ma20:
        ma_signal = "空头排列"
        ma_score = -2
    else:
        ma_signal = "粘合"
        ma_score = 0

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9).mean().values
    hist = (dif - dea) * 2

    macd_bullish = dif[-1] > dea[-1]
    macd_cross = dif[-1] > dea[-1] and dif[-2] <= dea[-2] if n > 2 else False
    if macd_cross:
        macd_signal = "金叉"
        macd_score = 2
    elif macd_bullish:
        macd_signal = "多头"
        macd_score = 1
    elif dif[-1] < dea[-1] and dif[-2] >= dea[-2] if n > 2 else False:
        macd_signal = "死叉"
        macd_score = -2
    else:
        macd_signal = "空头"
        macd_score = -1

    # 量能
    vol_score = 0
    vol_signal = ""
    if volume is not None and len(volume) >= 10:
        avg5 = np.mean(volume[-5:])
        avg20 = np.mean(volume[-20:])
        ratio = avg5 / avg20 if avg20 > 0 else 1
        if ratio > 1.5:
            vol_score = 2
            vol_signal = f"放量(×{ratio:.1f})"
        elif ratio > 1.1:
            vol_score = 1
            vol_signal = "温和放量"
        elif ratio < 0.7:
            vol_score = -1
            vol_signal = "缩量"
        else:
            vol_signal = "正常"

    return {
        "ma_score": ma_score,
        "ma_signal": ma_signal,
        "ma_values": {"ma5": round(ma5, 2), "ma10": round(ma10, 2),
                      "ma20": round(ma20, 2), "ma60": round(ma60, 2)},
        "macd_score": macd_score,
        "macd_signal": macd_signal,
        "macd_values": {"dif": round(float(dif[-1]), 3), "dea": round(float(dea[-1]), 3),
                        "hist": round(float(hist[-1]), 3)},
        "vol_score": vol_score,
        "vol_signal": vol_signal,
    }


def analyze_symbol(symbol: str, mode: str = "quick") -> dict:
    """分析单只股票"""
    holdings = _load_holdings()
    info = holdings.get(symbol, {})
    tag, is_trading = _data_source_tag()

    # 实时/最新价格
    rt = _get_realtime(symbol, is_trading)
    price = rt.get("price", 0)
    source = rt.get("source", "unknown")

    result = {
        "code": symbol,
        "name": info.get("name", symbol),
        "data_tag": tag,
        "source": source,
        "is_trading": is_trading,
        "price": price,
        "change_pct": rt.get("change_pct", 0),
        "holdings": {
            "shares": info.get("shares", 0),
            "cost": info.get("cost", 0),
            "stop_loss": info.get("stop_loss"),
            "take_profit": info.get("take_profit"),
            "pnl_pct": round((price / info["cost"] - 1) * 100, 1) if info.get("cost") and price else None,
        },
        "mode": mode,
    }

    # 止损止盈检查
    sl = info.get("stop_loss")
    tp = info.get("take_profit")
    if price and sl:
        result["stop_loss_dist"] = round((sl / price - 1) * 100, 1)
    if price and tp:
        result["take_profit_dist"] = round((tp / price - 1) * 100, 1)

    # 快速评分
    if mode in ("quick", "full"):
        score = _quick_score(symbol)
        result["scores"] = score

    return result


def format_output(result: dict) -> str:
    """格式化输出"""
    lines = []
    code = result["code"]
    name = result["name"]
    tag = result["data_tag"]
    mode = result.get("mode", "quick")

    # 头部
    lines.append(f"[分析] {code} {name} | 数据: {tag} | {'盘中' if result['is_trading'] else '盘后'}")

    # 价格
    price = result["price"]
    chg = result.get("change_pct", 0)
    h = result.get("holdings", {})
    cost = h.get("cost", 0)
    shares = h.get("shares", 0)
    sl = h.get("stop_loss")
    tp = h.get("take_profit")
    pnl = h.get("pnl_pct")

    price_line = f"现价: {price:.3f} | 涨跌: {chg:+.2f}%"
    if cost and shares:
        price_line += f" | 成本: {cost:.3f} | 盈亏: {pnl:+.1f}%"
        if sl:
            sl_dist = result.get("stop_loss_dist", 0)
            if sl_dist < 0:
                price_line += f" | 止损{sl}(↓{abs(sl_dist):.1f}%)"
            else:
                price_line += f" | 止损{sl}(已破↓{abs(sl_dist):.1f}%)"
        if tp:
            tp_dist = result.get("take_profit_dist", 0)
            if tp_dist > 0:
                price_line += f" | 止盈{tp}(↑{abs(tp_dist):.1f}%)"
            else:
                price_line += f" | 止盈{tp}(已到↑{abs(tp_dist):.1f}%)"
    lines.append(price_line)

    # 评分
    scores = result.get("scores")
    if scores:
        ma = scores.get("ma_signal", "")
        macd = scores.get("macd_signal", "")
        vol = scores.get("vol_signal", "")
        lines.append(f"均线: {ma} | MACD: {macd} | 量能: {vol}")

        ma_v = scores.get("ma_values", {})
        if ma_v:
            lines.append(f"  MA5={ma_v.get('ma5',0):.2f} MA10={ma_v.get('ma10',0):.2f} "
                         f"MA20={ma_v.get('ma20',0):.2f} MA60={ma_v.get('ma60',0):.2f}")
        macd_v = scores.get("macd_values", {})
        if macd_v:
            lines.append(f"  DIF={macd_v.get('dif',0):.3f} DEA={macd_v.get('dea',0):.3f} "
                         f"HIST={macd_v.get('hist',0):.3f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="一键分析入口")
    parser.add_argument("symbol", nargs="?", help="股票代码")
    parser.add_argument("--mode", choices=["quick", "full", "realtime"], default="quick")
    parser.add_argument("--all-holdings", action="store_true", help="扫描全部持仓")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    args = parser.parse_args()

    if args.all_holdings:
        holdings = _load_holdings()
        results = []
        for code, info in sorted(holdings.items()):
            r = analyze_symbol(code, mode="quick")
            results.append(r)
            if not args.json:
                print(format_output(r))
                print()
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return

    if not args.symbol:
        parser.print_help()
        return

    result = analyze_symbol(args.symbol, mode=args.mode)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_output(result))


if __name__ == "__main__":
    main()
