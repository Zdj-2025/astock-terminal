"""W3浪主升选股器 — 全市场扫描正在走第三浪的股票。

基于Elliott Wave理论的启发式W3检测，非穷举搜索，适合批量扫描。

算法流程:
    1. 自动找起始低点(anchor low)
    2. 检测W1顶和W2底
    3. 判断W3状态(in_progress/topped/no_signal)
    4. 计算置信度评分(0-100)

用法:
    通过选股子系统调用: python scripts/screener/screener_cli.py --template wave3
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)


class WaveScreener:
    """W3浪主升扫描器。"""

    def __init__(self, client):
        self.client = client

    def scan(
        self,
        candidates: pd.DataFrame,
        lookback_days: int = 500,
        min_confidence: float = 50.0,
        min_w3_gain: float = 10.0,
        top: int = 50,
        as_of_date: str = None,
    ) -> pd.DataFrame:
        """批量扫描候选股，找出正在走W3的股票。

        Args:
            candidates: Layer 1 UniverseFilter输出，需包含 ts_code, symbol, name 列
            lookback_days: K线回溯天数(日历日)，默认500(~2年)
            min_confidence: 最低置信度(0-100)
            min_w3_gain: W3最低涨幅%
            top: 最多返回N只

        Returns:
            DataFrame，包含原始列 + wave信息列
        """
        trade_date = as_of_date or datetime.now().strftime("%Y%m%d")

        # 检查缓存
        cached = self._load_cache(trade_date)
        if cached is not None and len(cached) > 0:
            print(f"  [WaveScreener] 使用缓存({trade_date}): {len(cached)}只")
            filtered = cached[cached['w3_confidence'] >= min_confidence]
            filtered = filtered[filtered['w3_gain_pct'] >= min_w3_gain]
            return filtered.head(top)

        # 预过滤减少K线获取量
        prefiltered = self._prefilter(candidates)
        print(f"  [WaveScreener] 预过滤: {len(candidates)} → {len(prefiltered)}只")

        # 获取K线并检测W3
        start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
        results = []
        total = len(prefiltered)

        for i, row in prefiltered.iterrows():
            symbol = str(row['symbol'])
            ts_code = row['ts_code']
            name = row.get('name', symbol)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [WaveScreener] 扫描进度: {i+1}/{total} "
                      f"({len(results)}只W3候选)")

            try:
                df = self.client.get_kline(
                    symbol, freq="D", start=start_date, end=as_of_date, adj="qfq")
                if df is None or len(df) < 60:
                    continue

                w3_info = self._detect_w3_single(df)
                if w3_info and w3_info['in_w3']:
                    record = row.to_dict()
                    record.update(w3_info)
                    results.append(record)
            except Exception:
                continue

        if not results:
            print(f"  [WaveScreener] 未找到W3候选股")
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('w3_confidence', ascending=False)

        # 缓存
        self._save_cache(result_df, trade_date)

        # 过滤
        filtered = result_df[
            (result_df['w3_confidence'] >= min_confidence) &
            (result_df['w3_gain_pct'] >= min_w3_gain)
        ]
        print(f"  [WaveScreener] W3检测: {total}只 → {len(result_df)}只W3 → "
              f"{len(filtered)}只达标(置信度≥{min_confidence}, 涨幅≥{min_w3_gain}%)")

        return filtered.head(top)

    # ── 预过滤 ──

    def _prefilter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """快速过滤，减少K线获取量。"""
        df = candidates.copy()

        # 盈利公司
        if 'pe_ttm' in df.columns:
            df = df[(df['pe_ttm'] > 0) | df['pe_ttm'].isna()]

        # 活跃交易
        if 'turnover_rate' in df.columns:
            df = df[df['turnover_rate'] >= 1.0]

        # 排除低价股
        if 'close' in df.columns:
            df = df[df['close'] >= 5.0]

        return df.reset_index(drop=True)

    # ── W3检测核心算法 ──

    def _detect_w3_single(self, df: pd.DataFrame) -> dict or None:
        """对单只股票检测W3状态。

        Args:
            df: K线DataFrame，需包含 date, open, high, low, close, volume 列

        Returns:
            dict with W3 info, or None if no W3 signal
        """
        if len(df) < 60:
            return None

        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        dates = df['date'].values

        # Step 1: 找起始低点
        anchor_idx = self._find_anchor_low(low)
        if anchor_idx is None:
            return None

        anchor_price = low[anchor_idx]

        # Step 2: 找W1和W2
        w1_result = self._find_w1_w2(high, low, close, anchor_idx, anchor_price)
        if w1_result is None:
            return None

        w1_idx, w1_price, w2_idx, w2_price = w1_result

        # Step 3: 判断W3状态
        w3_info = self._assess_w3_status(
            high, low, close, dates,
            w2_idx, w2_price, w1_idx, w1_price, anchor_idx, anchor_price)

        if not w3_info or not w3_info['in_w3']:
            return None

        # Step 4: 置信度评分
        confidence = self._compute_confidence(
            df, anchor_idx, anchor_price,
            w1_idx, w1_price, w2_idx, w2_price,
            w3_info)

        w3_info['w3_confidence'] = confidence

        # 添加日期格式化
        anchor_date = pd.Timestamp(dates[anchor_idx])
        w3_info['anchor_date'] = anchor_date.strftime('%Y-%m-%d')
        w3_info['anchor_price'] = float(anchor_price)
        w3_info['w1_end_date'] = pd.Timestamp(dates[w1_idx]).strftime('%Y-%m-%d')
        w3_info['w1_gain_pct'] = (w1_price / anchor_price - 1) * 100
        w3_info['w2_end_date'] = pd.Timestamp(dates[w2_idx]).strftime('%Y-%m-%d')
        w3_info['w2_retrace_pct'] = (w1_price - w2_price) / (w1_price - anchor_price) * 100

        current_close = float(close[-1])
        current_date = pd.Timestamp(dates[-1])
        w3_info['current_price'] = current_close
        w3_info['current_date'] = current_date.strftime('%Y-%m-%d')

        return w3_info

    def _find_anchor_low(self, low: np.ndarray) -> int or None:
        """自动找起始低点。"""
        n = len(low)

        # 在全部K线中找局部极小值
        local_lows = argrelextrema(low, np.less, order=10)[0]
        if len(local_lows) == 0:
            # 退化为全区间最低点
            return int(np.argmin(low))

        # 排除最近30天(太近不算浪型起点)
        cutoff = n - 30
        valid = local_lows[local_lows <= cutoff]
        if len(valid) == 0:
            valid = local_lows

        # 选最低价
        prices_at_lows = low[valid]
        best = valid[np.argmin(prices_at_lows)]

        return int(best)

    def _find_w1_w2(self, high, low, close, start_idx, start_price):
        """检测W1顶和W2底。"""
        n = len(high)

        # W1顶: 从起点向前找第一个显著的局部极大值
        search_end = min(start_idx + 120, n)  # 最多看120天
        seg_high = high[start_idx:search_end]
        local_peaks = argrelextrema(seg_high, np.greater, order=8)[0]
        if len(local_peaks) == 0:
            return None

        # 选第一个涨幅>5%的峰
        for peak_offset in local_peaks:
            peak_idx = start_idx + peak_offset
            peak_price = high[peak_idx]
            gain = (peak_price / start_price - 1)
            if gain > 0.05:
                w1_idx = peak_idx
                w1_price = peak_price
                break
        else:
            return None

        # W2底: 从W1顶向前找第一个局部极小值
        search_end2 = min(w1_idx + 80, n)
        seg_low = low[w1_idx:search_end2]
        local_troughs = argrelextrema(seg_low, np.less, order=6)[0]
        if len(local_troughs) == 0:
            return None

        # 选第一个满足铁律1(W2底>起点价)的谷
        for trough_offset in local_troughs:
            trough_idx = w1_idx + trough_offset
            trough_price = low[trough_idx]
            if trough_price > start_price:
                w2_idx = trough_idx
                w2_price = trough_price
                break
        else:
            # 没有满足铁律的谷，尝试取最低的非违规谷
            if len(local_troughs) > 0:
                trough_idx = w1_idx + local_troughs[0]
                trough_price = low[trough_idx]
                if trough_price > start_price * 0.98:  # 容忍2%偏差
                    w2_idx = trough_idx
                    w2_price = trough_price
                else:
                    return None
            else:
                return None

        # W2回撤检查(应在Fibonacci区间 0.236~0.786)
        w1_range = w1_price - start_price
        if w1_range > 0:
            retrace = (w1_price - w2_price) / w1_range
            if retrace < 0.15 or retrace > 0.95:
                return None

        return (int(w1_idx), float(w1_price), int(w2_idx), float(w2_price))

    def _assess_w3_status(self, high, low, close, dates,
                          w2_idx, w2_price, w1_idx, w1_price,
                          start_idx, start_price):
        """判断W3状态。"""
        n = len(high)

        # W3候选区间: W2底到数据末尾
        w3_seg = high[w2_idx:]
        if len(w3_seg) < 5:
            return None

        w3_high_offset = int(np.argmax(w3_seg))
        w3_high_idx = w2_idx + w3_high_offset
        w3_high_price = float(w3_seg[w3_high_offset])

        # W3必须高于W1顶(结构前提)
        if w3_high_price <= w1_price:
            return None

        # W3涨幅
        w3_gain = (w3_high_price / w2_price - 1) * 100
        if w3_gain < 5:
            return None

        # W3天数
        w3_days = int(w3_high_idx - w2_idx)
        if w3_days < 3:
            return None

        # 当前价
        current_close = float(close[-1])
        current_idx = n - 1
        drop_from_high = (current_close / w3_high_price - 1) * 100

        # 判断W3状态
        if drop_from_high > -10:
            w3_type = 'in_progress'
            in_w3 = True
        else:
            w3_type = 'topped'
            in_w3 = False  # 已到顶，不算"正在走W3"

        return {
            'in_w3': in_w3,
            'w3_type': w3_type,
            'w3_gain_pct': float(w3_gain),
            'w3_days': w3_days,
            'w3_high_idx': w3_high_idx,
            'w3_high_price': w3_high_price,
            'w3_high_date': pd.Timestamp(dates[w3_high_idx]).strftime('%Y-%m-%d'),
            'w3_drop_from_high': float(drop_from_high),
        }

    def _compute_confidence(self, df, anchor_idx, anchor_price,
                            w1_idx, w1_price, w2_idx, w2_price,
                            w3_info):
        """计算W3置信度评分(0-100)。"""
        score = 0.0
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        high = df['high'].values.astype(float)
        n = len(close)

        w1_len = abs(w1_price - anchor_price)
        w3_len = abs(w3_info['w3_high_price'] - w2_price)

        # 1. W3/W1比例 (0-25分)
        if w1_len > 0:
            ratio = w3_len / w1_len
            for ideal in [1.618, 2.618, 1.0, 0.618]:
                deviation = abs(ratio - ideal) / ideal
                if deviation < 0.4:
                    score += 25 * max(0, 1 - deviation / 0.4)
                    break

        # 2. W2回撤质量 (0-20分)
        if w1_len > 0:
            retrace = (w1_price - w2_price) / w1_len
            for ideal in [0.618, 0.5, 0.382]:
                deviation = abs(retrace - ideal)
                if deviation < 0.15:
                    score += 20 * max(0, 1 - deviation / 0.15)
                    break

        # 3. 量能确认 (0-20分)
        w1_vol = volume[anchor_idx:w1_idx + 1]
        w3_vol = volume[w2_idx:w3_info['w3_high_idx'] + 1]
        if len(w1_vol) > 0 and len(w3_vol) > 0 and w1_vol.mean() > 0:
            vol_ratio = w3_vol.mean() / w1_vol.mean()
            if vol_ratio >= 1.5:
                score += 20
            elif vol_ratio >= 1.2:
                score += 15
            elif vol_ratio >= 1.0:
                score += 8

        # 4. 趋势强度 (0-20分)
        if n >= 60:
            ma20 = np.convolve(close[-60:], np.ones(20)/20, mode='valid')[-1]
            ma60 = close[-60:].mean()
            current = close[-1]
            if current > ma20 and current > ma60:
                score += 20
            elif current > ma20 or current > ma60:
                score += 10

        # 5. 动量 (0-15分)
        # MACD红柱
        if n >= 35:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
            dif = ema12 - ema26
            dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
            hist = (dif - dea) * 2

            if dif[-1] > dea[-1]:
                score += 8  # MACD多头
            if hist[-1] > 0:
                score += 4  # 红柱

            # RSI在50-80
            delta = np.diff(close[-20:])
            gain = np.sum(delta[delta > 0])
            loss = -np.sum(delta[delta < 0])
            if loss > 0:
                rs = gain / loss
                rsi = 100 - 100 / (1 + rs)
                if 50 <= rsi <= 80:
                    score += 3

        return round(min(score, 100), 1)

    # ── 缓存 ──

    def _cache_dir(self):
        return Path(project_root) / "data" / "cache" / "wave_screener"

    def _load_cache(self, trade_date: str):
        cache_path = self._cache_dir() / f"{trade_date}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return None

    def _save_cache(self, results: pd.DataFrame, trade_date: str):
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(cache_dir / f"{trade_date}.csv", index=False)

    # ── 报告格式化 ──

    @staticmethod
    def format_wave_report(results: pd.DataFrame, meta: dict) -> str:
        """格式化控制台输出。"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  W3浪主升选股报告  {meta.get('trade_date', '')}")
        lines.append(f"  策略: W3浪主升 | "
                     f"候选池: {meta.get('candidates_count', 0)}只 → "
                     f"W3入选: {len(results)}只")
        lines.append("=" * 70)

        if results.empty:
            lines.append("  未找到符合W3条件的股票。")
            return '\n'.join(lines)

        lines.append("")
        lines.append(f"  {'排名':>4}  {'代码':8}  {'名称':8}  "
                     f"{'W3涨幅':>8}  {'置信度':>6}  {'W3天数':>6}  "
                     f"{'起始日':12}  {'当前价':>8}")
        lines.append("  " + "-" * 62)

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            symbol = str(row.get('symbol', ''))
            name = str(row.get('name', ''))[:6]
            w3_gain = row.get('w3_gain_pct', 0)
            conf = row.get('w3_confidence', 0)
            days = row.get('w3_days', 0)
            anchor = row.get('anchor_date', '')
            price = row.get('current_price', 0)

            lines.append(f"  {rank:4d}  {symbol:8s}  {name:8s}  "
                         f"{w3_gain:+7.1f}%  {conf:6.1f}  {days:6d}  "
                         f"{anchor:12s}  {price:8.2f}")

        lines.append("")
        lines.append("  注意: 以上为量化筛选结果，不构成投资建议。")
        return '\n'.join(lines)
