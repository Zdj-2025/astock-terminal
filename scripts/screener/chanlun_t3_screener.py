#!/usrusr/bin/env python3
"""缠论三买点选股器 — 找出"确认上涨趋势后回调不进中枢"的股票。

缠论第三买点定义：
  股价离开中枢（上涨），回调低点不跌入中枢上沿，确认趋势延续。
  这是最适合"追涨"的买点——风险可控（中枢做支撑），资金效率高。

本选股器的三系统验证：
  1. 缠论：三买点检测（离开中枢+回调不破中枢上沿）
  2. MACD：回调时MACD不翻绿/金叉确认（不深跌）
  3. 量能：回调缩量+反弹放量（健康调整）

用法:
  python scripts/screener/screener_cli.py --template chanlun_t3
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)


class ChanLunT3Screener:
    """缠论三买点扫描器"""

    CACHE_DIR = Path(PROJECT_ROOT) / "data" / "cache" / "chanlun_t3"

    def __init__(self, client):
        self.client = client
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def scan(
        self,
        candidates: pd.DataFrame,
        lookback_days: int = 200,
        min_confidence: float = 50.0,
        top: int = 50,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量扫描候选股，找出缠论三买点。

        Args:
            candidates: Layer 1 UniverseFilter 输出
            lookback_days: K线回溯天数
            min_confidence: 最低组合分(0-100)
            top: 最多返回N只
        """
        trade_date = as_of_date or datetime.now().strftime("%Y%m%d")

        cached = self._load_cache(trade_date)
        if cached is not None and len(cached) > 0:
            print(f"  [T3] 使用缓存({trade_date}): {len(cached)}只")
            filtered = cached[cached["t3_composite"] >= min_confidence]
            return filtered.head(top)

        print(f"\n  [T3] 缠论三买点扫描: {len(candidates)}只候选")

        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)
        ).strftime("%Y%m%d")

        results = []
        total = len(candidates)

        for i, row in candidates.iterrows():
            symbol = str(row.get("symbol", row.get("ts_code", ""))[:6])
            ts_code = row.get("ts_code", "")
            name = row.get("name", symbol)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [T3] 扫描进度: {i+1}/{total} ({len(results)}只三买点候选)")

            try:
                df = self.client.get_kline(
                    symbol, freq="D", start=start_date, end=as_of_date, adj="qfq"
                )
                if df is None or len(df) < 40:
                    continue

                info = self._detect_single(df)
                if info is None:
                    continue

                record = row.to_dict()
                record.update(info)
                results.append(record)
            except Exception:
                continue

        if not results:
            print(f"  [T3] 未找到三买点候选股")
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("t3_composite", ascending=False)

        self._save_cache(result_df, trade_date)

        filtered = result_df[result_df["t3_composite"] >= min_confidence]
        print(f"  [T3] 检测完成: {total}只 → {len(result_df)}只三买点 → "
              f"{len(filtered)}只达标(≥{min_confidence})")

        return filtered.head(top)

    # ── 核心检测 ──

    def _detect_single(self, df: pd.DataFrame) -> Optional[dict]:
        """对单只股票做缠论三买点检测。"""
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        n = len(close)
        if n < 40:
            return None

        # Step 1: 识别中枢
        pivot = self._find_pivot(high, low, close)
        if pivot is None:
            return None

        zs_low, zs_high, zs_start, zs_end = pivot

        # Step 2: 找离开中枢后的高点和回调低点
        leaving = self._find_leaving_and_pullback(
            high, low, close, zs_low, zs_high, zs_end
        )
        if leaving is None:
            return None

        leave_high, leave_high_idx, pullback_low, pullback_low_idx = leaving

        # Step 3: 三买点验证 — 回调低点 > 中枢上沿
        if pullback_low <= zs_high:
            return None  # 跌回中枢，不是三买点

        # Step 4: 三系统评分
        chanlun_score = self._score_chanlun(
            close, high, low, zs_low, zs_high, zs_end,
            leave_high, leave_high_idx, pullback_low, pullback_low_idx
        )
        macd_score = self._score_macd(close, pullback_low_idx)
        volume_score = self._score_volume(volume, pullback_low_idx)

        composite = self._compute_composite(chanlun_score, macd_score, volume_score)

        # 通过条件：三系统各≥25，组合≥50
        if chanlun_score < 25 or macd_score < 25 or volume_score < 25:
            return None
        if composite < 50:
            return None

        current_close = float(close[-1])
        dist_to_pivot = (current_close / zs_high - 1) * 100
        pullback_depth = (leave_high - pullback_low) / (leave_high - zs_high) * 100

        return {
            "t3_chanlun_score": round(chanlun_score, 1),
            "t3_macd_score": round(macd_score, 1),
            "t3_volume_score": round(volume_score, 1),
            "t3_composite": round(composite, 1),
            "t3_zs_range": f"[{zs_low:.2f}, {zs_high:.2f}]",
            "t3_zs_low": round(zs_low, 2),
            "t3_zs_high": round(zs_high, 2),
            "t3_leave_high": round(leave_high, 2),
            "t3_pullback_low": round(pullback_low, 2),
            "t3_pullback_depth_pct": round(pullback_depth, 1),
            "t3_dist_to_zs_high_pct": round(dist_to_pivot, 2),
            "t3_current_price": round(current_close, 2),
            # ── 复盘改进 #001 新增 ──
            "t3_suggested_position_pct": round(self._suggest_position(composite), 3),
            "t3_target_price": round(leave_high * 1.05, 2),  # 目标=离开高点+5%
            "t3_stop_loss_price": round(zs_high * 0.98, 2),  # 止损=中枢上沿-2%
        }

    # ── 中枢识别 ──

    def _find_pivot(self, high, low, close):
        """简化中枢识别：找最近一个价格密集区间。

        算法：
          1. 用局部极值点构建笔
          2. 找3根以上重叠的笔构成中枢
          3. 返回中枢的[ZD, ZG, start_idx, end_idx]
        """
        n = len(close)
        if n < 30:
            return None

        # 找局部高点和低点
        peaks = argrelextrema(high, np.greater, order=5)[0]
        troughs = argrelextrema(low, np.less, order=5)[0]

        if len(peaks) < 3 or len(troughs) < 3:
            return None

        # 合并极值点，按时间排序
        extremes = []
        for idx in peaks:
            extremes.append((idx, "high", high[idx]))
        for idx in troughs:
            extremes.append((idx, "low", low[idx]))
        extremes.sort(key=lambda x: x[0])

        # 构建笔序列（简化：相邻的high-low对）
        strokes = []
        for i in range(len(extremes) - 1):
            idx1, type1, price1 = extremes[i]
            idx2, type2, price2 = extremes[i + 1]
            if type1 != type2:  # 高-低 或 低-高
                strokes.append({
                    "start_idx": idx1,
                    "end_idx": idx2,
                    "high": max(price1, price2),
                    "low": min(price1, price2),
                })

        # 找中枢：3根以上笔的价格有重叠区间
        if len(strokes) < 3:
            return None

        # 从最近的笔开始倒找
        for start in range(len(strokes) - 1, max(1, len(strokes) - 10), -1):
            window = strokes[max(0, start - 4):start + 1]
            if len(window) < 3:
                continue

            # 重叠区间 = min(各笔high) 到 max(各笔low)
            zs_high = min(s["high"] for s in window)
            zs_low = max(s["low"] for s in window)

            # 有效中枢要求 ZD < ZG
            if zs_low >= zs_high:
                continue

            # 中枢不能太靠近当前（至少在5天前形成）
            zs_end_idx = max(s["end_idx"] for s in window)
            if zs_end_idx > n - 5:
                continue

            # 中枢至少包含3根笔
            zs_start_idx = min(s["start_idx"] for s in window)
            return (zs_low, zs_high, zs_start_idx, zs_end_idx)

        return None

    # ── 离开中枢 + 回调 ──

    def _find_leaving_and_pullback(self, high, low, close, zs_low, zs_high, zs_end):
        """找离开中枢后的高点和回调低点。"""
        n = len(close)
        if zs_end >= n - 3:
            return None

        # 离开中枢后的区间
        after = high[zs_end + 1:]
        after_low = low[zs_end + 1:]
        if len(after) < 5:
            return None

        # 离开后的最高点
        leave_high_idx_rel = int(np.argmax(after))
        leave_high = float(after[leave_high_idx_rel])
        leave_high_idx = zs_end + 1 + leave_high_idx_rel

        # 最高点必须在中枢上方
        if leave_high <= zs_high:
            return None

        # 最高点之后的回调低点
        if leave_high_idx >= n - 2:
            return None  # 还没回调

        pullback_range = low[leave_high_idx + 1:]
        if len(pullback_range) < 2:
            return None

        # 找回调最低点（至少要有3天的回调）
        if len(pullback_range) >= 3:
            pullback_low_idx_rel = int(np.argmin(pullback_range))
            pullback_low = float(pullback_range[pullback_low_idx_rel])
            pullback_low_idx = leave_high_idx + 1 + pullback_low_idx_rel
        else:
            return None

        # 回调幅度至少3%
        pullback_pct = (leave_high - pullback_low) / leave_high
        if pullback_pct < 0.03:
            return None

        return (leave_high, leave_high_idx, pullback_low, pullback_low_idx)

    # ── 三系统评分 ──

    def _score_chanlun(self, close, high, low, zs_low, zs_high, zs_end,
                       leave_high, leave_high_idx, pullback_low, pullback_low_idx):
        """缠论三买点评分(0-100)。"""
        score = 0.0

        # 1. 回调不破中枢的强度 (0-40)
        margin = (pullback_low - zs_high) / (leave_high - zs_high)
        if margin > 0.5:   # 回调不到一半
            score += 40
        elif margin > 0.3:
            score += 30
        elif margin > 0.15:
            score += 20
        else:
            score += 10  # 刚好不破中枢

        # 2. 中枢厚度（越厚支撑越强）(0-20)
        zs_width = (zs_high - zs_low) / zs_low * 100
        if 3 <= zs_width <= 15:
            score += 20  # 适中
        elif zs_width < 3:
            score += 10  # 太窄
        else:
            score += 5   # 太宽=震荡

        # 3. 离开中枢的力度 (0-20)
        leave_gain = (leave_high / zs_high - 1) * 100
        if leave_gain > 15:
            score += 20  # 强力离开
        elif leave_gain > 8:
            score += 15
        elif leave_gain > 3:
            score += 10

        # 4. 当前价格位置 (0-20)
        current = float(close[-1])
        if current > leave_high:
            score += 20  # 已创新高
        elif current > pullback_low * 1.02:
            score += 15  # 回调后反弹
        else:
            score += 5

        return round(min(score, 100), 1)

    def _score_macd(self, close, pullback_low_idx):
        """MACD回调状态评分(0-100)。"""
        score = 0.0
        dif, dea, hist = self._calc_macd(close, 12, 26, 9)
        n = len(close)

        # 1. 回调低点时MACD状态 (0-40)
        if pullback_low_idx < n:
            pb_dif = float(dif[pullback_low_idx])
            pb_hist = float(hist[pullback_low_idx])
            if pb_dif > 0 and pb_hist > 0:
                score += 40  # 完美：MACD全在水上
            elif pb_dif > 0:
                score += 30  # DIF在零线上
            elif pb_hist > 0:
                score += 20  # 至少红柱
            else:
                score += 5   # MACD翻绿

        # 2. 当前MACD状态 (0-30)
        if n > 1:
            cur_dif = float(dif[-1])
            cur_hist = float(hist[-1])
            prev_hist = float(hist[-2])
            if cur_dif > 0 and cur_hist > 0 and cur_hist > prev_hist:
                score += 30  # 金叉红柱扩大
            elif cur_dif > 0 and cur_hist > 0:
                score += 20
            elif cur_hist > 0:
                score += 10

        # 3. 近期是否有金叉 (0-30)
        recent_cross = False
        for j in range(max(1, n - 5), n):
            if dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                recent_cross = True
                break
        if recent_cross:
            score += 30
        elif dif[-1] > dea[-1]:
            score += 15

        return round(min(score, 100), 1)

    def _score_volume(self, volume, pullback_low_idx):
        """量能评分(0-100)。"""
        if volume is None or len(volume) < 20:
            return 50.0  # 无数据给中等分

        score = 0.0
        n = len(volume)

        # 1. 回调缩量 (0-40)
        if pullback_low_idx >= 5 and pullback_low_idx < n:
            pb_vol = float(np.mean(volume[pullback_low_idx:pullback_low_idx + 3]))
            prior_vol = float(np.mean(volume[max(0, pullback_low_idx - 10):pullback_low_idx]))
            if prior_vol > 0:
                shrink_ratio = pb_vol / prior_vol
                if shrink_ratio < 0.6:
                    score += 40  # 明显缩量
                elif shrink_ratio < 0.8:
                    score += 30
                elif shrink_ratio < 1.0:
                    score += 15

        # 2. 反弹放量 (0-35)
        recent_vol = float(np.mean(volume[-5:]))
        pb_vol_avg = float(np.mean(volume[max(0, pullback_low_idx - 2):pullback_low_idx + 3]))
        if pb_vol_avg > 0:
            expand_ratio = recent_vol / pb_vol_avg
            if expand_ratio > 1.5:
                score += 35  # 明显放量
            elif expand_ratio > 1.2:
                score += 25
            elif expand_ratio > 1.0:
                score += 10

        # 3. 量价配合度 (0-25)
        vol_5 = float(np.mean(volume[-5:]))
        vol_20 = float(np.mean(volume[-20:]))
        if vol_20 > 0:
            ratio = vol_5 / vol_20
            if ratio > 1.3:
                score += 25  # 近期放量
            elif ratio > 0.8:
                score += 15
            else:
                score += 5

        return round(min(score, 100), 1)

    # ── 组合评分 ──

    def _compute_composite(self, chanlun, macd, volume):
        """三系统加权组合分。"""
        return round(chanlun * 0.40 + macd * 0.35 + volume * 0.25, 1)

    # ── MACD计算 ──

    @staticmethod
    def _calc_macd(close, fast, slow, signal):
        s = pd.Series(close)
        ema_fast = s.ewm(span=fast, adjust=False).mean().values
        ema_slow = s.ewm(span=slow, adjust=False).mean().values
        dif = ema_fast - ema_slow
        dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
        hist = 2 * (dif - dea)
        return dif, dea, hist

    # ── 缓存 ──

    def _load_cache(self, trade_date):
        path = self.CACHE_DIR / f"{trade_date}.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        return None

    def _save_cache(self, df, trade_date):
        path = self.CACHE_DIR / f"{trade_date}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception:
            pass

    # ── 复盘改进 #001 新增 ──

    @staticmethod
    def _suggest_position(composite: float) -> float:
        """根据组合分建议仓位。"""
        if composite >= 80:
            return 0.10
        elif composite >= 65:
            return 0.07
        elif composite >= 50:
            return 0.03
        else:
            return 0.0
