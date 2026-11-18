#!/usr/bin/env python3
"""MACD动量选股器 — 基于"MACD连续红柱+底部企稳+即将启动"形态。

实战案例：TCL科技(000100)，2026/6/2买入特征：
  - MACD金叉中，红柱连续35天+
  - 距120日高点-13%（适中位）
  - 从底部恢复+6%
  - 收盘站MA20
  - 量价健康（涨放量跌缩量）
  - 5日涨幅温和

评分体系（满分100）：
  ① MACD趋势确认（25分）
  ② 底部企稳（25分）
  ③ 位置适中（20分）
  ④ 量价健康（15分）
  ⑤ 启动前特征（15分）

用法:
    python scripts/screener/macd_momentum_screener.py                # 默认扫全市场
    python scripts/screener/macd_momentum_screener.py --top 20       # 前20只
    python scripts/screener/macd_momentum_screener.py --code 000100  # 分析单只股票
    python scripts/screener/macd_momentum_screener.py --detail       # 显示每只详情
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from scripts.core.tushare_client import TushareClient, from_ts_code, to_ts_code


# ── MACD计算（numpy递推EMA） ─────────────────────────────────────────

def _calc_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA递推计算"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    result = np.empty_like(series, dtype=float)
    k = 2.0 / (period + 1)
    # 第一个EMA值用SMA
    result[:period] = np.nan
    result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result


def _calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD指标，返回 (dif, dea, hist)"""
    ema_fast = _calc_ema(close, fast)
    ema_slow = _calc_ema(close, slow)
    dif = ema_fast - ema_slow
    # DEA是DIF的EMA
    valid_dif = dif[~np.isnan(dif)]
    if len(valid_dif) < signal:
        dea = np.full_like(dif, np.nan)
        hist = np.full_like(dif, np.nan)
        return dif, dea, hist
    # 计算DEA：从第一个有效DIF开始
    first_valid = np.where(~np.isnan(dif))[0][0]
    dea = np.full_like(dif, np.nan)
    k = 2.0 / (signal + 1)
    dea[first_valid + signal - 1] = np.mean(dif[first_valid:first_valid + signal])
    for i in range(first_valid + signal, len(dif)):
        dea[i] = dif[i] * k + dea[i - 1] * (1 - k)
    hist = (dif - dea) * 2
    return dif, dea, hist


def _calc_ma(series: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    result = np.empty_like(series, dtype=float)
    result[:period - 1] = np.nan
    cumsum = np.cumsum(series)
    result[period - 1] = cumsum[period - 1] / period
    for i in range(period, len(series)):
        result[i] = (cumsum[i] - cumsum[i - period]) / period
    return result


# ── 评分结果 ──────────────────────────────────────────────────────────

@dataclass
class MACDScoreResult:
    """MACD动量评分结果"""
    symbol: str = ""
    name: str = ""
    total_score: float = 0.0

    # 分项分数
    macd_score: float = 0.0       # 满分25
    bottom_score: float = 0.0     # 满分25
    position_score: float = 0.0   # 满分20
    volume_score: float = 0.0     # 满分15
    startup_score: float = 0.0    # 满分15

    # MACD详情
    dif: float = 0.0
    dea: float = 0.0
    hist: float = 0.0
    red_bars: int = 0             # 红柱连续天数
    fast_golden: bool = False     # 快速MACD金叉

    # 底部详情
    pct_from_bottom: float = 0.0  # 从底部恢复百分比
    volume_expand: float = 0.0    # 底部后量能放大倍数

    # 位置详情
    pct_from_high: float = 0.0    # 距120日高点百分比
    above_ma20: bool = False

    # 量价详情
    up_down_vol_ratio: float = 0.0
    vol_increasing: bool = False

    # 启动详情
    pct_5d: float = 0.0
    small_yang_count: int = 0
    volume_ratio_today: float = 0.0

    # 描述
    detail_desc: str = ""


# ── 主类 ──────────────────────────────────────────────────────────────

class MACDMomentumScreener:
    """MACD动量选股器：寻找'MACD连续红柱+底部企稳+即将启动'形态"""

    def __init__(self, client: Optional[TushareClient] = None):
        self.client = client or TushareClient()

    # ── 单只股票评分 ───────────────────────────────────────────────

    def score(self, code: str, daily_df: Optional[pd.DataFrame] = None) -> MACDScoreResult:
        """对单只股票进行MACD动量评分。

        Args:
            code: 股票代码（纯数字）
            daily_df: 可选日线数据，为None则自动获取

        Returns:
            MACDScoreResult
        """
        result = MACDScoreResult(symbol=code)

        # 获取数据
        if daily_df is None:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
            daily_df = self.client.get_kline(code, freq="D", start=start, end=end, adj="qfq")

        if daily_df is None or len(daily_df) < 60:
            return result

        df = daily_df.copy()
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        close = df["close"].astype(float).values
        volume = df["volume"].astype(float).values
        n = len(close)

        if n < 60 or volume[-1] <= 0:
            return result

        # 获取股票名称
        try:
            info = self.client.get_stock_info(code)
            result.name = info.get("name", "")
        except Exception:
            pass

        # ── ① MACD趋势确认（25分） ──
        dif, dea, hist = _calc_macd(close, 12, 26, 9)
        fast_dif, fast_dea, fast_hist = _calc_macd(close, 5, 34, 5)

        macd_score = 0.0

        # 检查MACD金叉（DIF > DEA）
        if n >= 2 and not np.isnan(dif[-1]) and not np.isnan(dea[-1]):
            result.dif = round(float(dif[-1]), 4)
            result.dea = round(float(dea[-1]), 4)
            result.hist = round(float(hist[-1]), 4)

            if dif[-1] > dea[-1]:
                macd_score += 10  # 金叉基础分

                # 红柱连续天数
                red_count = 0
                for i in range(len(hist) - 1, -1, -1):
                    if np.isnan(hist[i]):
                        break
                    if hist[i] > 0:
                        red_count += 1
                    else:
                        break
                result.red_bars = red_count

                # 红柱连续天数评分：每10天+3分，最多+8分
                red_bonus = min(8, (red_count // 10) * 3)
                macd_score += red_bonus

                # 快速MACD(5,34,5)也金叉
                if not np.isnan(fast_dif[-1]) and not np.isnan(fast_dea[-1]):
                    if fast_dif[-1] > fast_dea[-1]:
                        macd_score += 7
                        result.fast_golden = True
            else:
                # 不金叉，直接排除
                result.macd_score = 0
                result.total_score = 0
                return result

        result.macd_score = min(25, macd_score)

        # ── ② 底部企稳（25分） ──
        bottom_score = 0.0

        # 30日低点
        lookback = min(30, n)
        recent_30 = close[-lookback:]
        low_30 = np.min(recent_30)
        low_30_idx = np.argmin(recent_30)
        current = close[-1]
        pct_from_low = (current / low_30 - 1) * 100

        # 距30日低点>5%且回升中
        if pct_from_low > 5:
            bottom_score += 10
            result.pct_from_bottom = round(pct_from_low, 2)

            # 从底部恢复程度评分
            if pct_from_low < 5:
                bottom_score += 10  # 刚起步（更好）
            elif pct_from_low <= 15:
                bottom_score += 8   # 不过度反弹
            # 恢复>15%不给分

            # 底部后量能放大
            if low_30_idx < lookback - 5:
                # 底部前的均量 vs 底部后的均量
                before_bottom = volume[-lookback: -lookback + low_30_idx + 1]
                after_bottom = volume[-lookback + low_30_idx:]
                if len(before_bottom) > 0 and len(after_bottom) > 0:
                    vol_before = np.mean(before_bottom)
                    vol_after = np.mean(after_bottom)
                    if vol_before > 0:
                        vol_expand = vol_after / vol_before
                        result.volume_expand = round(vol_expand, 2)
                        if vol_expand > 1.5:
                            bottom_score += 7

        result.bottom_score = min(25, bottom_score)

        # ── ③ 位置适中（20分） ──
        position_score = 0.0

        # 120日高点
        high_lookback = min(120, n)
        high_120 = np.max(close[-high_lookback:])
        pct_from_high = (current / high_120 - 1) * 100
        result.pct_from_high = round(pct_from_high, 2)

        if -25 <= pct_from_high <= -5:
            position_score += 15  # 回调后适中位
        elif pct_from_high > -5:
            position_score -= 5   # 高位
        elif pct_from_high < -30:
            position_score -= 5   # 弱势
        else:  # -30 to -25
            position_score += 8

        # MA20
        ma20 = _calc_ma(close, 20)
        if not np.isnan(ma20[-1]) and current > ma20[-1]:
            position_score += 5
            result.above_ma20 = True

        result.position_score = min(20, max(0, position_score))

        # ── ④ 量价健康（15分） ──
        vol_score = 0.0

        # 近10日上涨均量/下跌均量
        lookback_vol = min(10, n - 1)
        if lookback_vol >= 5:
            up_vols = []
            down_vols = []
            for i in range(-lookback_vol, 0):
                day_change = close[i] - close[i - 1]
                if day_change > 0:
                    up_vols.append(volume[i])
                elif day_change < 0:
                    down_vols.append(volume[i])

            if up_vols and down_vols:
                avg_up_vol = np.mean(up_vols)
                avg_down_vol = np.mean(down_vols)
                if avg_down_vol > 0:
                    ud_ratio = avg_up_vol / avg_down_vol
                    result.up_down_vol_ratio = round(ud_ratio, 2)
                    if ud_ratio > 2.0:
                        vol_score += 15  # 满分
                    elif ud_ratio > 1.5:
                        vol_score += 10

        # 近5日均量 > 近20日均量（量能递增）
        if n >= 25:
            vol_5 = np.mean(volume[-5:])
            vol_20 = np.mean(volume[-20:])
            if vol_20 > 0 and vol_5 > vol_20:
                vol_score += 3
                result.vol_increasing = True

        result.volume_score = min(15, vol_score)

        # ── ⑤ 启动前特征（15分） ──
        startup_score = 0.0

        # 近5日涨幅
        if n >= 6:
            pct_5d = (close[-1] / close[-6] - 1) * 100
            result.pct_5d = round(pct_5d, 2)
            if -5 <= pct_5d <= 5:
                startup_score += 8

        # 近3日小阳线（涨幅0-3%的阳线>=2天）
        if n >= 4:
            small_yang = 0
            for i in range(-3, 0):
                if i - 1 >= -n:
                    day_ret = (close[i] / close[i - 1] - 1) * 100
                    if 0 < day_ret <= 3:
                        small_yang += 1
            result.small_yang_count = small_yang
            if small_yang >= 2:
                startup_score += 4

        # 当日或前一日放量（量比>1.5）
        if n >= 20:
            vol_ma20 = np.mean(volume[-20:])
            if vol_ma20 > 0:
                # 当日
                vr_today = volume[-1] / vol_ma20
                # 前一日
                vr_yesterday = volume[-2] / vol_ma20 if n >= 2 else 0
                best_vr = max(vr_today, vr_yesterday)
                result.volume_ratio_today = round(best_vr, 2)
                if best_vr > 1.5:
                    startup_score += 3

        result.startup_score = min(15, startup_score)

        # ── 汇总 ──
        result.total_score = round(
            result.macd_score + result.bottom_score +
            result.position_score + result.volume_score +
            result.startup_score, 1
        )

        # 生成描述
        parts = []
        if result.red_bars >= 5:
            parts.append(f"红柱{result.red_bars}天")
        if result.pct_from_bottom > 0:
            parts.append(f"底部恢复{result.pct_from_bottom:.1f}%")
        if result.above_ma20:
            parts.append("MA20站稳")
        if result.up_down_vol_ratio > 1.5:
            parts.append(f"量价比{result.up_down_vol_ratio:.1f}")
        if result.pct_from_high != 0:
            parts.append(f"距高{result.pct_from_high:.0f}%")
        if result.fast_golden:
            parts.append("双MACD共振")
        result.detail_desc = "+".join(parts) if parts else ""

        return result

    # ── 全市场扫描 ─────────────────────────────────────────────────

    def scan(self, top_n: int = 30, date: Optional[str] = None) -> list[MACDScoreResult]:
        """全市场扫描。

        Args:
            top_n: 返回前N只
            date: 截断日期 YYYYMMDD

        Returns:
            按总分排序的MACDScoreResult列表
        """
        print("=" * 70, flush=True)
        print("MACD动量选股器 — 全市场扫描", flush=True)
        print("=" * 70, flush=True)

        start_time = time.time()

        # 1. 获取股票列表
        print("\n[1/3] 获取全A股列表...")
        stocks = self.client.get_stock_list()
        if stocks.empty:
            print("  ✗ 获取股票列表失败")
            return []

        # 2. 基础过滤
        print(f"  原始: {len(stocks)} 只")

        # 排除ST
        stocks = stocks[~stocks["name"].str.contains("ST", na=False)]
        print(f"  排除ST后: {len(stocks)} 只")

        # 排除北交所
        stocks = stocks[~stocks["ts_code"].str.endswith(".BJ")]
        print(f"  排除北交所后: {len(stocks)} 只")

        # 排除次新股（上市<60天）
        if "list_date" in stocks.columns:
            ref_date = date or datetime.now().strftime("%Y%m%d")
            cutoff = (datetime.strptime(ref_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")
            stocks = stocks[stocks["list_date"].notna()]
            stocks = stocks[stocks["list_date"] <= cutoff]
        print(f"  排除次新后: {len(stocks)} 只")

        # 排除停牌（通过daily_basic获取当日有数据的）
        print("\n[2/3] 获取当日行情过滤停牌...")
        trade_date = date or self.client.get_recent_trade_day()
        daily_basic = self.client.get_daily_basic(trade_date=trade_date)

        if not daily_basic.empty:
            active_codes = set(daily_basic["ts_code"].tolist())
            stocks = stocks[stocks["ts_code"].isin(active_codes)]
        print(f"  排除停牌后: {len(stocks)} 只")

        # 3. 逐只评分
        print(f"\n[3/3] 逐只评分中（共{len(stocks)}只）...")

        start_dt = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")
        results = []
        passed = 0
        filtered_out = 0
        total = len(stocks)

        for i, (_, row) in enumerate(stocks.iterrows()):
            ts_code = row["ts_code"]
            code = from_ts_code(ts_code)
            name = row.get("name", code)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"  进度: {i+1}/{total} | "
                      f"通过: {passed} | 排除: {filtered_out} | "
                      f"耗时: {elapsed:.0f}s | ETA: {eta:.0f}s")

            try:
                df = self.client.get_kline(code, freq="D", start=start_dt, end=trade_date, adj="qfq")
                if df is None or len(df) < 60:
                    filtered_out += 1
                    continue

                r = self.score(code, df)
                r.name = name

                # 过滤条件：MACD金叉 + 红柱>=5天
                if r.red_bars < 5:
                    filtered_out += 1
                    continue

                if r.total_score > 0:
                    results.append(r)
                    passed += 1

            except Exception as e:
                filtered_out += 1
                continue

        # 排序
        results.sort(key=lambda x: x.total_score, reverse=True)
        results = results[:top_n]

        elapsed = time.time() - start_time
        print(f"\n  ✓ 扫描完成！通过: {passed}只 | 耗时: {elapsed:.1f}s")
        print(f"  前{min(top_n, len(results))}只候选股:")

        return results

    # ── 格式化报告 ─────────────────────────────────────────────────

    def format_report(self, results: list[MACDScoreResult], detail: bool = False) -> str:
        """格式化输出报告"""
        if not results:
            return "无符合条件的股票。"

        lines = []
        lines.append("")
        lines.append("=" * 110)
        lines.append("MACD动量选股报告 — MACD连续红柱+底部企稳+即将启动")
        lines.append("=" * 110)
        lines.append("")

        # 表头
        header = (
            f"{'排名':>4} | {'代码':<8} | {'名称':<10} | {'总分':>5} | "
            f"{'MACD':>8} | {'底部':>8} | {'位置':>8} | {'量价':>8} | {'启动':>8} | 详情"
        )
        lines.append(header)
        lines.append("-" * 110)

        for i, r in enumerate(results):
            line = (
                f"{i+1:>4} | {r.symbol:<8} | {r.name:<10} | {r.total_score:>5.0f} | "
                f"{r.macd_score:>4.0f}/25 | {r.bottom_score:>4.0f}/25 | "
                f"{r.position_score:>4.0f}/20 | {r.volume_score:>4.0f}/15 | "
                f"{r.startup_score:>4.0f}/15 | {r.detail_desc}"
            )
            lines.append(line)

        lines.append("")

        if detail:
            lines.append("=" * 110)
            lines.append("详细分析")
            lines.append("=" * 110)
            for i, r in enumerate(results):
                lines.append("")
                lines.append(f"【{i+1}】{r.name}({r.symbol}) — 总分 {r.total_score}")
                lines.append(f"  MACD: {r.macd_score:.0f}/25 "
                             f"(DIF={r.dif:.4f}, DEA={r.dea:.4f}, HIST={r.hist:.4f}, "
                             f"红柱{r.red_bars}天, 快速MACD{'金叉' if r.fast_golden else '未金叉'})")
                lines.append(f"  底部: {r.bottom_score:.0f}/25 "
                             f"(距底部+{r.pct_from_bottom:.1f}%, "
                             f"量能放大{r.volume_expand:.1f}倍)")
                lines.append(f"  位置: {r.position_score:.0f}/20 "
                             f"(距120日高{r.pct_from_high:.1f}%, "
                             f"{'站' if r.above_ma20 else '未站'}MA20)")
                lines.append(f"  量价: {r.volume_score:.0f}/15 "
                             f"(涨跌量比{r.up_down_vol_ratio:.2f}, "
                             f"{'量能递增' if r.vol_increasing else '量能不增'})")
                lines.append(f"  启动: {r.startup_score:.0f}/15 "
                             f"(5日涨{r.pct_5d:.1f}%, "
                             f"小阳线{r.small_yang_count}天, "
                             f"量比{r.volume_ratio_today:.2f})")

        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    # 强制stdout行缓冲，确保后台运行也能实时看到输出
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="MACD动量选股器")
    parser.add_argument("--code", type=str, help="分析单只股票（代码如 000100）")
    parser.add_argument("--top", type=int, default=30, help="返回前N只（默认30）")
    parser.add_argument("--detail", action="store_true", help="显示详细分析")
    parser.add_argument("--date", type=str, help="截断日期 YYYYMMDD")
    args = parser.parse_args()

    screener = MACDMomentumScreener()

    if args.code:
        # 单只分析
        print(f"分析 {args.code} ...")
        result = screener.score(args.code)
        print(screener.format_report([result], detail=True))
    else:
        # 全市场扫描
        results = screener.scan(top_n=args.top, date=args.date)
        print(screener.format_report(results, detail=args.detail))


if __name__ == "__main__":
    main()
