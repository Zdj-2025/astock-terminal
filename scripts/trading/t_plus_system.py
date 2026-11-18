#!/usr/bin/env python3
"""做T（日内/短线波段交易）系统 — 多周期共振信号 + 风控 + 波浪识别。

依赖: pandas, numpy, tushare
数据源: TushareClient (从 data/kline_store/ 读取，如无则下载)

用法:
    PYTHONPATH=. python scripts/cli/t_plus_cli.py 603881
    PYTHONPATH=. python scripts/cli/t_plus_cli.py --holdings
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── 项目路径 ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "watchlist.json"

# ── 数据结构 ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    """单个周期的信号"""
    period: str            # "60min" / "15min" / "5min"
    direction: str         # "buy" / "sell" / "neutral"
    strength: int          # 0-5
    reason: str            # 信号原因描述

@dataclass
class TPlusResult:
    """做T分析结果"""
    symbol: str
    name: str = ""
    current_price: float = 0.0
    signal: str = "观望"          # "买入" / "卖出" / "观望"
    signal_strength: int = 0      # 1-5 星
    buy_price: float = 0.0        # 建议买入价
    target_price: float = 0.0     # 目标卖出价
    stop_loss_price: float = 0.0  # 止损价
    details: list[Signal] = field(default_factory=list)
    resonance: str = ""           # 共振描述
    risk_warning: str = ""        # 风险提示
    can_trade: bool = True
    can_trade_reason: str = ""
    position_shares: int = 0      # 建议仓位（股数）
    position_amount: float = 0.0  # 建议仓位（金额）


# ── 技术指标计算 ──────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(
    df: pd.DataFrame,
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算MACD，返回 (dif, dea, hist)"""
    ema_fast = calc_ema(df["close"], fast)
    ema_slow = calc_ema(df["close"], slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    hist = (dif - dea) * 2
    return dif, dea, hist


def calc_kdj(
    df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算KDJ，返回 (K, D, J)"""
    low_n = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_ma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(period, min_periods=1).mean()


# ── 5浪识别（简化模型）─────────────────────────────────────────────────

def identify_5wave(df: pd.DataFrame, lookback: int = 60) -> dict:
    """简化的5浪识别模型。

    找最近的高低点序列，判断当前处于哪一浪。
    返回: {"current_wave": 1-5 or "A"-"C", "description": str}
    """
    if len(df) < 20:
        return {"current_wave": "?", "description": "数据不足"}

    recent = df.tail(min(lookback, len(df))).copy().reset_index(drop=True)
    close = recent["close"]
    high = recent["high"]
    low = recent["low"]

    # 找局部极值点（用窗口扫描）
    order = max(3, len(recent) // 10)
    from scipy.signal import argrelextrema
    high_idx = argrelextrema(high.values, np.greater, order=order)[0]
    low_idx = argrelextrema(low.values, np.less, order=order)[0]

    # 合并高低点，按时间排序
    pivots = []
    for i in high_idx:
        pivots.append({"idx": i, "price": high.iloc[i], "type": "high"})
    for i in low_idx:
        pivots.append({"idx": i, "price": low.iloc[i], "type": "low"})
    pivots.sort(key=lambda x: x["idx"])

    if len(pivots) < 3:
        return {
            "current_wave": "?",
            "description": f"拐点不足({len(pivots)}个)，无法判断浪型",
            "pivots": pivots,
        }

    # 尝试识别上升5浪或下降5浪
    # 上升5浪: 低-高-低-高-低-高-低（共7个点: 1低,1高,2低,3高,4低,5高,A低）
    # 简化: 找最近3个拐点判断当前处于哪一段
    last3 = pivots[-3:]
    last = pivots[-1]
    current_price = close.iloc[-1]

    # 判断趋势方向（最近一段是上升还是下降）
    # 找最近的一个大低点和大高点
    recent_low_idx = 0
    recent_high_idx = 0
    recent_low_price = float("inf")
    recent_high_price = 0.0

    for p in pivots[-6:]:  # 看最近6个拐点
        if p["type"] == "low" and p["price"] < recent_low_price:
            recent_low_price = p["price"]
            recent_low_idx = p["idx"]
        if p["type"] == "high" and p["price"] > recent_high_price:
            recent_high_price = p["price"]
            recent_high_idx = p["idx"]

    # 如果最近低点在最近高点之前 → 当前在上升段
    if recent_low_idx < recent_high_idx:
        # 上升段，判断是第几浪
        if current_price >= recent_high_price * 0.98:
            wave = "5浪末端（冲顶）"
            wave_num = 5
        elif current_price <= recent_low_price * 1.03:
            wave = "2浪/4浪回调末端"
            wave_num = 2
        else:
            mid = (recent_low_price + recent_high_price) / 2
            if current_price > mid:
                wave = "3浪或5浪上升中"
                wave_num = 3
            else:
                wave = "回调中，可能在2浪或4浪"
                wave_num = 4
    else:
        # 下降段（调整浪）
        if current_price <= recent_low_price * 1.02:
            wave = "A浪下跌末端/C浪末端"
            wave_num = -1
        else:
            wave = "调整浪（B浪反弹或C浪下跌中）"
            wave_num = -2

    return {
        "current_wave": wave,
        "wave_num": wave_num,
        "recent_low": round(recent_low_price, 2),
        "recent_high": round(recent_high_price, 2),
        "pivots_count": len(pivots),
        "description": f"{wave} | 近低:{recent_low_price:.2f} 近高:{recent_high_price:.2f}",
    }


# ── TPlusSignal 信号生成器 ─────────────────────────────────────────────

class TPlusSignal:
    """多周期共振信号生成器"""

    def __init__(self, client):
        """
        Args:
            client: TushareClient 实例
        """
        self.client = client

    def _get_kline(self, symbol: str, freq: str, days: int = 30) -> pd.DataFrame:
        """获取K线数据，优先缓存，API失败时用日线降级"""
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        # 1. 先尝试正常获取（缓存优先）
        try:
            df = self.client.get_kline(symbol, freq=freq, start=start, adj="qfq")
            if not df.empty:
                return df
        except Exception:
            pass

        # 2. 尝试直接从缓存读取（包括过期缓存）
        from scripts.core.tushare_client import to_ts_code, _freq_map, _CACHE_DIR, _DiskCache
        ts_code = to_ts_code(symbol)
        mapped_freq = _freq_map(freq)
        cache = _DiskCache(_CACHE_DIR)
        # 尝试多种缓存键
        for end_val in [None, datetime.now().strftime("%Y%m%d"), "None"]:
            for start_val in [start, str(start), "None"]:
                try:
                    p = cache._key("kline", ts_code=ts_code, freq=mapped_freq,
                                    start=start_val, end=end_val, adj="qfq")
                    if p.exists():
                        df = pd.read_parquet(p)
                        if not df.empty:
                            return df
                except Exception:
                    continue
        
        # 3. 暴力搜索缓存目录（匹配 ts_code 中 . 变 _ 的情况）
        try:
            import glob as _glob
            import os as _os
            # ts_code 中的点在文件名中变成下划线: 603881.SH → 603881_SH
            safe_code = ts_code.replace('.', '_')
            # 优先匹配含目标freq的文件
            all_files = _glob.glob(str(_CACHE_DIR / f"kline__*{safe_code}*.parquet"))
            if all_files:
                # 优先选含目标freq的文件
                freq_files = [f for f in all_files if mapped_freq in _os.path.basename(f)]
                search_files = freq_files if freq_files else all_files
                # 按文件修改时间排序，最新的优先
                search_files.sort(key=lambda x: _os.path.getmtime(x), reverse=True)
                for f in search_files[:5]:
                    try:
                        df = pd.read_parquet(f)
                        if not df.empty:
                            return df
                    except Exception:
                        continue
        except Exception:
            pass

        # 4. 最后降级：用日线数据
        try:
            daily_start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            df = self.client.get_kline(symbol, freq="D", start=daily_start, adj="qfq")
            if not df.empty:
                return df
        except Exception:
            pass

        return pd.DataFrame()

    def analyze_60min(self, df: pd.DataFrame) -> Signal:
        """60分钟MACD信号"""
        if len(df) < 30:
            return Signal("60min", "neutral", 0, "数据不足")

        dif, dea, hist = calc_macd(df)

        # 取最近几根K线的指标
        dif_val = dif.iloc[-1]
        dea_val = dea.iloc[-1]
        hist_val = hist.iloc[-1]
        prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0
        prev2_hist = hist.iloc[-3] if len(hist) >= 3 else 0
        prev3_hist = hist.iloc[-4] if len(hist) >= 4 else 0

        reasons = []
        direction = "neutral"
        strength = 0

        # 金叉判断
        if dif_val > dea_val and dif.iloc[-2] <= dea.iloc[-2]:
            direction = "buy"
            strength = 3
            reasons.append("MACD金叉")
        # 死叉判断
        elif dif_val < dea_val and dif.iloc[-2] >= dea.iloc[-2]:
            direction = "sell"
            strength = 3
            reasons.append("MACD死叉")
        # 红柱扩大/缩小
        elif hist_val > 0:
            if hist_val > prev_hist:
                direction = "buy"
                strength = 2
                reasons.append(f"红柱扩大({hist_val:.3f}>{prev_hist:.3f})")
            else:
                direction = "neutral"
                strength = 0
                reasons.append(f"红柱缩短({hist_val:.3f}<{prev_hist:.3f})")
        elif hist_val < 0:
            if hist_val < prev_hist:
                direction = "sell"
                strength = 2
                reasons.append(f"绿柱扩大({hist_val:.3f}<{prev_hist:.3f})")
            else:
                direction = "neutral"
                strength = 1
                reasons.append(f"绿柱缩短({hist_val:.3f}>{prev_hist:.3f})")

        # 连续3根红柱缩短 → 卖出信号
        if (prev3_hist > prev2_hist > prev_hist > hist_val
                and hist_val > 0):
            direction = "sell"
            strength = 3
            reasons.append("红柱连续3根缩短")

        reason_str = "; ".join(reasons) if reasons else "无明显信号"
        return Signal("60min", direction, strength, reason_str)

    def analyze_15min(self, df: pd.DataFrame) -> Signal:
        """15分钟KDJ信号"""
        if len(df) < 20:
            return Signal("15min", "neutral", 0, "数据不足")

        k, d, j = calc_kdj(df)
        j_val = j.iloc[-1]
        k_val = k.iloc[-1]
        d_val = d.iloc[-1]
        prev_j = j.iloc[-2] if len(j) >= 2 else j_val
        prev_k = k.iloc[-2] if len(k) >= 2 else k_val

        reasons = []
        direction = "neutral"
        strength = 0

        # 超卖后拐头（买入）
        if prev_j < 20 and j_val > prev_j:
            direction = "buy"
            strength = 3
            reasons.append(f"KDJ超卖拐头(J:{prev_j:.1f}→{j_val:.1f})")
        elif prev_j < 30 and j_val > prev_j and j_val < 50:
            direction = "buy"
            strength = 2
            reasons.append(f"KDJ低位回升(J:{prev_j:.1f}→{j_val:.1f})")

        # 超买后拐头（卖出）
        if prev_j > 80 and j_val < prev_j:
            direction = "sell"
            strength = 3
            reasons.append(f"KDJ超买拐头(J:{prev_j:.1f}→{j_val:.1f})")
        elif prev_j > 70 and j_val < prev_j and j_val > 50:
            direction = "sell"
            strength = 2
            reasons.append(f"KDJ高位回落(J:{prev_j:.1f}→{j_val:.1f})")

        # J值极端区域
        if j_val < 10:
            reasons.append(f"J值极低({j_val:.1f})")
            if direction != "sell":
                strength = max(strength, 2)
        elif j_val > 100:
            reasons.append(f"J值极高({j_val:.1f})")
            if direction != "buy":
                strength = max(strength, 2)

        reason_str = "; ".join(reasons) if reasons else f"KDJ正常(J:{j_val:.1f},K:{k_val:.1f},D:{d_val:.1f})"
        return Signal("15min", direction, strength, reason_str)

    def analyze_5min(self, df: pd.DataFrame) -> Signal:
        """5分钟波浪 + MA20信号"""
        if len(df) < 40:
            return Signal("5min", "neutral", 0, "数据不足")

        ma20 = calc_ma(df, 20)
        current_price = df["close"].iloc[-1]
        ma20_val = ma20.iloc[-1]
        ma5_val = calc_ma(df, 5).iloc[-1]

        # 波浪识别
        wave_info = identify_5wave(df, lookback=50)
        wave_desc = wave_info.get("description", "")
        wave_num = wave_info.get("wave_num", 0)

        reasons = []
        direction = "neutral"
        strength = 0

        # 价格与MA20关系
        price_vs_ma20 = (current_price - ma20_val) / ma20_val * 100

        if abs(price_vs_ma20) < 1.0 and current_price > ma20_val:
            # 价格在MA20附近企稳
            reasons.append(f"价格回调到MA20附近企稳(偏离{price_vs_ma20:+.2f}%)")
            direction = "buy"
            strength = 2
        elif price_vs_ma20 > 3.0:
            # 价格远离MA20上方
            reasons.append(f"价格远离MA20上方(偏离{price_vs_ma20:+.2f}%)")
            direction = "sell"
            strength = 2
        elif price_vs_ma20 < -2.0:
            # 价格远离MA20下方
            reasons.append(f"价格远离MA20下方(偏离{price_vs_ma20:+.2f}%)")
            direction = "buy"
            strength = 1
        else:
            reasons.append(f"价格围绕MA20波动(偏离{price_vs_ma20:+.2f}%)")

        # 波浪判断
        if wave_num in (2, 4):
            # 2浪/4浪末端 → 买入
            reasons.append(f"波浪:{wave_desc}")
            if direction != "sell":
                strength += 1
                direction = "buy"
        elif wave_num == 5:
            # 5浪末端 → 卖出
            reasons.append(f"波浪:{wave_desc}")
            direction = "sell"
            strength += 1
        elif wave_num == 3:
            # 3浪上升中
            reasons.append(f"波浪:{wave_desc}")
            if direction != "sell":
                strength += 1
        elif wave_num < 0:
            # 调整浪
            reasons.append(f"波浪:{wave_desc}")
        else:
            reasons.append(f"波浪:{wave_desc}")

        reason_str = "; ".join(reasons)
        return Signal("5min", direction, strength, reason_str)


# ── TPlusRiskManager 风控 ─────────────────────────────────────────────

class TPlusRiskManager:
    """做T风控管理器"""

    def __init__(self, total_capital: float = 491000):
        self.total_capital = total_capital
        self.max_daily_amount = 50000      # 5万/天
        self.stop_loss_pct = -0.02         # -2%
        self.max_trades_per_day = 2        # 每天最多2次
        self.max_consecutive_losses = 2    # 连续2次亏损暂停1天

    def check_can_trade(self, today_trades: list[dict]) -> tuple[bool, str]:
        """检查今天是否还能做T。

        Args:
            today_trades: 今日已做交易列表
                [{"time": "09:35", "action": "buy", "price": 22.5,
                  "amount": 2200, "pnl": -200, ...}, ...]
                pnl > 0 表示盈利, pnl < 0 表示亏损

        Returns:
            (can_trade, reason)
        """
        if not today_trades:
            return True, "今日尚未交易，可以操作"

        # 1. 检查交易次数
        trade_count = len(today_trades)
        if trade_count >= self.max_trades_per_day:
            return False, f"今日已交易{trade_count}次，达到上限({self.max_trades_per_day}次)"

        # 2. 检查已用金额
        total_amount = sum(t.get("amount", 0) * t.get("price", 0) for t in today_trades)
        if total_amount >= self.max_daily_amount:
            return False, f"今日已用金额{total_amount:,.0f}元，达到上限({self.max_daily_amount:,.0f}元)"

        # 3. 检查连续亏损（最近几笔交易）
        consecutive_losses = 0
        for t in reversed(today_trades):
            pnl = t.get("pnl", 0)
            if pnl < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= self.max_consecutive_losses:
            return False, f"连续{consecutive_losses}次亏损，暂停1天冷静"

        return True, f"已交易{trade_count}次，剩余{self.max_trades_per_day - trade_count}次机会"

    def calculate_position(self, price: float) -> tuple[int, float]:
        """计算可买股数（5万额度内，100股整数倍）。

        Returns:
            (shares, amount)
        """
        if price <= 0:
            return 0, 0.0
        max_shares = int(self.max_daily_amount / price)
        # 100股整数倍
        shares = (max_shares // 100) * 100
        amount = shares * price
        return shares, round(amount, 2)

    def check_stop_loss(
        self, buy_price: float, current_price: float,
    ) -> tuple[bool, float]:
        """检查是否触发止损。

        Returns:
            (triggered, pnl_pct)
        """
        if buy_price <= 0:
            return False, 0.0
        pnl_pct = (current_price - buy_price) / buy_price
        triggered = pnl_pct <= self.stop_loss_pct
        return triggered, round(pnl_pct * 100, 2)

    def calculate_stop_loss_price(self, buy_price: float) -> float:
        """计算止损价"""
        return round(buy_price * (1 + self.stop_loss_pct), 3)

    def calculate_target_price(self, buy_price: float, target_pct: float = 0.02) -> float:
        """计算目标价（默认+2%）"""
        return round(buy_price * (1 + target_pct), 3)


# ── TPlusAnalyzer 做T分析器 ────────────────────────────────────────────

class TPlusAnalyzer:
    """做T机会综合分析"""

    def __init__(self, client=None):
        """Args:
            client: TushareClient 实例
        """
        if client is None:
            from scripts.core.tushare_client import TushareClient
            client = TushareClient()
        self.client = client
        self.signal_gen = TPlusSignal(client)
        self.risk_mgr = TPlusRiskManager()
        self._load_config()

    def _load_config(self):
        """加载配置"""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception:
            self.config = {"holdings": [], "watchlist": []}

    def _get_stock_name(self, symbol: str) -> str:
        """从配置中获取股票名称"""
        for item in self.config.get("holdings", []) + self.config.get("watchlist", []):
            if item.get("code") == symbol:
                return item.get("name", symbol)
        try:
            info = self.client.get_stock_info(symbol)
            return info.get("name", symbol)
        except Exception:
            return symbol

    def analyze(self, symbol: str) -> TPlusResult:
        """分析一只股票的做T机会"""
        symbol = symbol.strip()
        name = self._get_stock_name(symbol)
        result = TPlusResult(symbol=symbol, name=name)

        # ── 获取K线数据 ──
        # 尝试获取分钟线，API受限时降级到日线
        df_60min = self.signal_gen._get_kline(symbol, "60min", days=30)
        df_15min = self.signal_gen._get_kline(symbol, "15min", days=15)
        df_5min = self.signal_gen._get_kline(symbol, "5min", days=5)

        # 检测是否降级到日线：通过数据时间跨度判断
        # 分钟线的date列通常包含时间(如 2026-05-19 10:30:00)，日线只有日期
        using_daily = False
        for df in [df_60min, df_15min, df_5min]:
            if not df.empty and 'date' in df.columns:
                first_date = str(df['date'].iloc[0])
                # 日线只有日期没有时间
                if ' ' not in first_date and ':' not in first_date:
                    using_daily = True
                    break

        # 当前价格
        if not df_5min.empty:
            result.current_price = round(df_5min["close"].iloc[-1], 3)
        elif not df_15min.empty:
            result.current_price = round(df_15min["close"].iloc[-1], 3)
        elif not df_60min.empty:
            result.current_price = round(df_60min["close"].iloc[-1], 3)
        else:
            result.risk_warning = "无法获取价格数据"
            return result

        # ── 多周期信号分析 ──
        if using_daily:
            # 日线降级模式：用日线MACD代替60min，KDJ代替15min，MA20/波浪代替5min
            sig_60 = self.signal_gen.analyze_60min(df_60min) if not df_60min.empty else Signal("60min", "neutral", 0, "无数据")
            sig_15 = self.signal_gen.analyze_15min(df_15min) if not df_15min.empty else Signal("15min", "neutral", 0, "无数据")
            sig_5 = self.signal_gen.analyze_5min(df_5min) if not df_5min.empty else Signal("5min", "neutral", 0, "无数据")
            # 给信号标注降级提示
            for sig in [sig_60, sig_15, sig_5]:
                if sig.reason != "数据不足" and sig.reason != "无数据":
                    sig.reason = f"[日线降级] {sig.reason}"
        else:
            sig_60 = self.signal_gen.analyze_60min(df_60min) if not df_60min.empty else Signal("60min", "neutral", 0, "无数据")
            sig_15 = self.signal_gen.analyze_15min(df_15min) if not df_15min.empty else Signal("15min", "neutral", 0, "无数据")
            sig_5 = self.signal_gen.analyze_5min(df_5min) if not df_5min.empty else Signal("5min", "neutral", 0, "无数据")

        result.details = [sig_60, sig_15, sig_5]

        # ── 综合信号判断 ──
        buy_score = 0
        sell_score = 0
        for sig in result.details:
            if sig.direction == "buy":
                buy_score += sig.strength
            elif sig.direction == "sell":
                sell_score += sig.strength

        # 共振判断
        buy_periods = sum(1 for s in result.details if s.direction == "buy")
        sell_periods = sum(1 for s in result.details if s.direction == "sell")

        if buy_score >= 5 and buy_periods >= 2:
            result.signal = "买入"
            result.signal_strength = min(5, buy_score)
            result.resonance = f"买入共振({buy_periods}周期一致，总分{buy_score})"
        elif sell_score >= 5 and sell_periods >= 2:
            result.signal = "卖出"
            result.signal_strength = min(5, sell_score)
            result.resonance = f"卖出共振({sell_periods}周期一致，总分{sell_score})"
        elif buy_score > sell_score and buy_score >= 3:
            result.signal = "买入"
            result.signal_strength = min(5, buy_score)
            result.resonance = f"买入偏强(总分{buy_score}，卖出分{sell_score})"
        elif sell_score > buy_score and sell_score >= 3:
            result.signal = "卖出"
            result.signal_strength = min(5, sell_score)
            result.resonance = f"卖出偏强(总分{sell_score}，买入分{buy_score})"
        else:
            result.signal = "观望"
            result.signal_strength = 0
            result.resonance = f"信号不明(买入{buy_score}，卖出{sell_score})"

        # ── 价位计算 ──
        price = result.current_price
        if result.signal == "买入":
            # 买入价：当前价或稍低（MA20附近）
            if len(df_5min) >= 20:
                ma20_val = calc_ma(df_5min, 20).iloc[-1]
                result.buy_price = round(ma20_val, 3)
            else:
                result.buy_price = round(price * 0.995, 3)
            result.target_price = self.risk_mgr.calculate_target_price(result.buy_price, 0.02)
            result.stop_loss_price = self.risk_mgr.calculate_stop_loss_price(result.buy_price)
        elif result.signal == "卖出":
            # 如果已有持仓，建议卖出价
            result.buy_price = 0
            result.target_price = round(price * 1.005, 3)  # 当前价附近
            result.stop_loss_price = round(price * 0.98, 3)
        else:
            result.buy_price = round(price * 0.99, 3)
            result.target_price = round(price * 1.01, 3)
            result.stop_loss_price = round(price * 0.98, 3)

        # ── 仓位计算 ──
        if result.signal == "买入" and result.buy_price > 0:
            shares, amount = self.risk_mgr.calculate_position(result.buy_price)
            result.position_shares = shares
            result.position_amount = amount

        # ── 风险提示 ──
        warnings = []
        if result.signal_strength <= 2:
            warnings.append("信号较弱，建议谨慎")
        if result.signal == "买入" and sell_score >= 3:
            warnings.append("存在反向信号，注意风险")
        if result.signal_strength >= 4:
            warnings.append("⚠️ 信号较强，严格执行止损")

        result.risk_warning = " | ".join(warnings) if warnings else "正常"

        return result

    def analyze_holdings(self) -> list[TPlusResult]:
        """分析所有持仓的做T机会"""
        results = []
        for item in self.config.get("holdings", []):
            code = item.get("code", "")
            if not code or item.get("type") == "etf":
                # ETF做T规则不同，暂跳过
                continue
            try:
                r = self.analyze(code)
                results.append(r)
            except Exception as e:
                results.append(TPlusResult(
                    symbol=code,
                    name=item.get("name", code),
                    risk_warning=f"分析失败: {e}",
                ))
        return results


# ── 报告生成 ──────────────────────────────────────────────────────────

def format_report(result: TPlusResult) -> str:
    """格式化单只股票的做T分析报告"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  📊 {result.name}({result.symbol}) 做T分析报告")
    lines.append("=" * 60)

    # 当前价格
    lines.append(f"\n💰 当前价格: {result.current_price:.3f}")

    # 综合信号
    stars = "⭐" * result.signal_strength if result.signal_strength > 0 else "—"
    signal_emoji = {"买入": "🟢", "卖出": "🔴", "观望": "⚪"}.get(result.signal, "❓")
    lines.append(f"\n{signal_emoji} 综合信号: {result.signal}  {stars}")
    lines.append(f"   共振情况: {result.resonance}")

    # 各周期信号
    lines.append(f"\n📈 多周期信号明细:")
    period_names = {"60min": "60分钟", "15min": "15分钟", "5min": "5分钟"}
    for sig in result.details:
        direction_emoji = {"buy": "🟢", "sell": "🔴", "neutral": "⚪"}.get(sig.direction, "❓")
        direction_cn = {"buy": "买入", "sell": "卖出", "neutral": "中性"}.get(sig.direction, "未知")
        pname = period_names.get(sig.period, sig.period)
        s_stars = "⭐" * sig.strength if sig.strength > 0 else "—"
        lines.append(f"   {pname:6s} {direction_emoji} {direction_cn} {s_stars}")
        lines.append(f"          {sig.reason}")

    # 建议价位
    if result.signal == "买入":
        lines.append(f"\n🎯 建议价位:")
        lines.append(f"   买入价:   {result.buy_price:.3f}")
        lines.append(f"   目标价:   {result.target_price:.3f} (+{(result.target_price/result.buy_price - 1)*100:.2f}%)")
        lines.append(f"   止损价:   {result.stop_loss_price:.3f} ({(result.stop_loss_price/result.buy_price - 1)*100:.2f}%)")
        if result.position_shares > 0:
            lines.append(f"\n📐 建议仓位:")
            lines.append(f"   股数: {result.position_shares} 股 ({result.position_shares // 100}手)")
            lines.append(f"   金额: {result.position_amount:,.0f} 元")
            # 预计盈亏
            target_pnl = (result.target_price - result.buy_price) * result.position_shares
            stop_pnl = (result.stop_loss_price - result.buy_price) * result.position_shares
            lines.append(f"   预计盈利: +{target_pnl:,.0f} 元")
            lines.append(f"   预计亏损: {stop_pnl:,.0f} 元")
    elif result.signal == "卖出":
        lines.append(f"\n🎯 建议操作:")
        lines.append(f"   当前价:   {result.current_price:.3f}")
        lines.append(f"   目标价:   {result.target_price:.3f}")

    # 风险提示
    lines.append(f"\n⚠️  风险提示: {result.risk_warning}")

    # 风控状态
    lines.append(f"\n🛡️  风控检查: {'✅ 可以交易' if result.can_trade else '❌ 暂停交易'}")
    if result.can_trade_reason:
        lines.append(f"   {result.can_trade_reason}")

    # 纪律提醒
    lines.append(f"\n📋 做T纪律:")
    lines.append(f"   ✓ 做T资金: 5万/天（总资金10%）")
    lines.append(f"   ✓ 止损: -2%强制（约1000元）")
    lines.append(f"   ✓ 做T不过夜")
    lines.append(f"   ✓ 每天最多2次")
    lines.append(f"   ✓ 连续2次亏损暂停1天")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_holdings_report(results: list[TPlusResult]) -> str:
    """格式化持仓做T汇总报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("  📊 持仓做T机会汇总")
    lines.append("=" * 70)

    # 表头
    lines.append(f"\n{'代码':<8} {'名称':<8} {'现价':>8} {'信号':<6} {'强度':>6} {'买入价':>8} {'目标价':>8} {'止损价':>8} {'仓位':>8}")
    lines.append("-" * 70)

    for r in results:
        signal_emoji = {"买入": "🟢", "卖出": "🔴", "观望": "⚪"}.get(r.signal, "❓")
        stars = "⭐" * r.signal_strength if r.signal_strength > 0 else "—"
        buy_p = f"{r.buy_price:.3f}" if r.buy_price > 0 else "—"
        target_p = f"{r.target_price:.3f}" if r.target_price > 0 else "—"
        stop_p = f"{r.stop_loss_price:.3f}" if r.stop_loss_price > 0 else "—"
        pos = f"{r.position_shares}股" if r.position_shares > 0 else "—"

        lines.append(
            f"{r.symbol:<8} {r.name:<8} {r.current_price:>8.3f} "
            f"{signal_emoji}{r.signal:<5} {stars:>6} "
            f"{buy_p:>8} {target_p:>8} {stop_p:>8} {pos:>8}"
        )

    lines.append("-" * 70)

    # 有信号的优先列出详情
    actionable = [r for r in results if r.signal in ("买入", "卖出") and r.signal_strength >= 3]
    if actionable:
        lines.append(f"\n🔥 重点关注:")
        for r in actionable:
            emoji = {"买入": "🟢", "卖出": "🔴"}.get(r.signal, "")
            lines.append(f"   {emoji} {r.name}({r.symbol}): {r.signal} {'⭐' * r.signal_strength} — {r.resonance}")

    lines.append("=" * 70)
    return "\n".join(lines)
