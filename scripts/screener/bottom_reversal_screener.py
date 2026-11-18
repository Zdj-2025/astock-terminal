# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

"""底部反转探测器 — 基于牛股启动前特征研究。

研究发现：2023-2025年18只非ST牛股（+350%~+580%）启动前共性：
  - RSI(14)均值33.6（超卖区）
  - MA空头排列（0%多头排列）
  - MACD死叉（仅18%金叉）
  - 缩量（量比0.86）
  - 距60日低点仅+5.7%
  - PE中位27.6（合理估值）
  - 市值中位33亿（中小盘）

本探测器寻找符合上述"极度弱势但基本面不差"特征的股票，
再加一个"反转信号"过滤：近期出现放量阳线或RSI拐头。

评分体系（0-100）：
  - 超卖程度(0-30): RSI越低分越高
  - 缩量程度(0-20): 量比越低分越高
  - 底部特征(0-25): 距低点越近分越高
  - 基本面(0-15): PE合理、市值适中
  - 反转信号(0-10): 近期出现放量阳线/RSI拐头
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)


class BottomReversalScreener:
    """底部反转探测器"""

    def __init__(self, client):
        self.client = client

    def scan(
        self,
        candidates: pd.DataFrame,
        lookback_days: int = 200,
        min_confidence: float = 40.0,
        top: int = 50,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量扫描底部反转候选股。

        Args:
            candidates: 候选股票DataFrame
            lookback_days: K线回溯天数
            min_confidence: 最低综合分(0-100)
            top: 最多返回N只
            as_of_date: 截断日期(回测用)
        """
        trade_date = as_of_date or datetime.now().strftime("%Y%m%d")

        # 检查缓存
        cached = self._load_cache(trade_date)
        if cached is not None and len(cached) > 0:
            passed = cached[cached["br_score"] >= min_confidence]
            return passed.head(top)

        # 预过滤：排除PE异常、超大市值
        prefiltered = self._prefilter(candidates)
        print(f"  [BottomReversal] 预过滤: {len(candidates)} → {len(prefiltered)}只")

        start_date = (
            datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)
        ).strftime("%Y%m%d")

        results = []
        total = len(prefiltered)

        for i, (_, row) in enumerate(prefiltered.iterrows()):
            if (i + 1) % 50 == 0:
                scored = len([r for r in results if r["br_score"] >= min_confidence])
                print(f"  [BottomReversal] 扫描进度: {i+1}/{total} "
                      f"({scored}只底部反转候选)")

            ts_code = row["ts_code"]
            symbol = ts_code.split(".")[0]

            try:
                df = self.client.get_kline(
                    symbol, freq="D", start=start_date, end=as_of_date, adj="qfq")
                if df is None or len(df) < 60:
                    continue

                if "date" in df.columns and "trade_date" not in df.columns:
                    df = df.rename(columns={"date": "trade_date"})

                df = df.sort_values("trade_date").reset_index(drop=True)
                df["close"] = df["close"].astype(float)
                df["volume"] = df["volume"].astype(float)

                score_result = self._score_stock(df, row)

                if score_result and score_result["br_score"] >= min_confidence:
                    score_result["ts_code"] = ts_code
                    score_result["name"] = row.get("name", symbol)
                    results.append(score_result)

            except Exception:
                continue

        if not results:
            empty = pd.DataFrame()
            self._save_cache(trade_date, empty)
            return empty

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("br_score", ascending=False).reset_index(drop=True)
        self._save_cache(trade_date, result_df)

        passed = result_df[result_df["br_score"] >= min_confidence]
        print(f"  [BottomReversal] 底部反转候选: {len(passed)} 只")
        return passed.head(top)

    def _prefilter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """预过滤：排除明显不符合的"""
        df = candidates.copy()

        # 排除PE为负或过高的（牛股研究中PE中位27.6）
        if "pe_ttm" in df.columns:
            df = df[(df["pe_ttm"] > 0) & (df["pe_ttm"] < 100)]

        # 排除超大盘（牛股研究中市值中位33亿）
        if "total_mv" in df.columns:
            mv_yi = df["total_mv"] / 1e4
            df = df[mv_yi <= 500]

        return df

    def _score_stock(self, df: pd.DataFrame, row: pd.Series) -> Optional[dict]:
        """计算单只股票的底部反转评分"""
        close = df["close"]
        vol = df["volume"]
        n = len(df)
        if n < 60:
            return None

        # ── 1. 超卖程度 (0-30) ──
        rsi = self._calc_rsi(close, 14)
        if rsi is None:
            return None
        rsi_val = rsi.iloc[-1]

        # 牛股研究：RSI均值33.6 → 越低越好（但不是0，0说明数据问题）
        if pd.isna(rsi_val) or rsi_val <= 0:
            return None
        if rsi_val > 50:
            return None  # 不在超卖区，不符合底部特征

        if rsi_val <= 25:
            oversold_score = 30
        elif rsi_val <= 35:
            oversold_score = 25
        elif rsi_val <= 40:
            oversold_score = 18
        elif rsi_val <= 45:
            oversold_score = 10
        else:
            oversold_score = 5

        # ── 2. 缩量程度 (0-20) ──
        if n >= 25:
            vol_5d = vol.iloc[-5:].mean()
            vol_20d = vol.iloc[-25:-5].mean()
            if vol_20d > 0:
                vol_ratio = vol_5d / vol_20d
            else:
                vol_ratio = 1.0
        else:
            vol_ratio = 1.0

        # 牛股研究：量比均值0.86 → 越缩量越好
        if vol_ratio <= 0.5:
            shrink_score = 20
        elif vol_ratio <= 0.7:
            shrink_score = 16
        elif vol_ratio <= 0.9:
            shrink_score = 10
        elif vol_ratio <= 1.0:
            shrink_score = 5
        else:
            shrink_score = 0

        # ── 3. 底部特征 (0-25) ──
        low_60 = close.iloc[-60:].min()
        low_20 = close.iloc[-20:].min()
        current = close.iloc[-1]

        pct_from_60low = (current / low_60 - 1) * 100
        pct_from_20low = (current / low_20 - 1) * 100

        # 牛股研究：距60日低点仅+5.7%，距20日低点+2.4%
        bottom_score = 0
        if pct_from_60low <= 5:
            bottom_score += 15
        elif pct_from_60low <= 10:
            bottom_score += 10
        elif pct_from_60low <= 15:
            bottom_score += 5

        if pct_from_20low <= 3:
            bottom_score += 10
        elif pct_from_20low <= 5:
            bottom_score += 5

        bottom_score = min(25, bottom_score)

        # ── 4. 基本面 (0-15) ──
        fund_score = 0
        pe = float(row.get("pe_ttm", 0)) if pd.notna(row.get("pe_ttm")) else 0
        pb = float(row.get("pb", 0)) if pd.notna(row.get("pb")) else 0
        mv_yi = float(row.get("total_mv", 0)) / 1e4 if pd.notna(row.get("total_mv")) else 0

        # PE 20-60: 牛股研究82%在此区间
        if 15 <= pe <= 60:
            fund_score += 8
        elif 10 <= pe <= 80:
            fund_score += 4

        # 市值20-200亿（牛股中位33亿）
        if 20 <= mv_yi <= 200:
            fund_score += 7
        elif 20 <= mv_yi <= 500:
            fund_score += 3

        fund_score = min(15, fund_score)

        # ── 5. 反转信号 (0-10) ──
        reversal_score = 0

        # 近5日出现放量阳线（成交量>20日均量且涨幅>2%）
        if n >= 25:
            vol_ma20 = vol.iloc[-25:-5].mean()
            for j in range(-5, 0):
                if j >= -len(df):
                    day_close = close.iloc[j]
                    day_open = float(df.iloc[j]["open"]) if "open" in df.columns else day_close
                    day_vol = vol.iloc[j]
                    if (day_close > day_open and
                            (day_close / day_open - 1) > 0.02 and
                            day_vol > vol_ma20 * 1.2):
                        reversal_score = 8
                        break

        # RSI拐头（5日前RSI比现在低，说明在回升）
        if rsi is not None and len(rsi) >= 5:
            rsi_5d_ago = rsi.iloc[-5]
            rsi_now = rsi.iloc[-1]
            if rsi_now > rsi_5d_ago and rsi_5d_ago < 35:
                reversal_score = max(reversal_score, 5)

        reversal_score = min(10, reversal_score)

        # ── 总分 ──
        total = oversold_score + shrink_score + bottom_score + fund_score + reversal_score

        return {
            "br_score": total,
            "br_oversold": oversold_score,
            "br_shrink": shrink_score,
            "br_bottom": bottom_score,
            "br_fundamental": fund_score,
            "br_reversal": reversal_score,
            "br_rsi": round(rsi_val, 1),
            "br_vol_ratio": round(vol_ratio, 2),
            "br_pct_60low": round(pct_from_60low, 1),
            "br_pe": round(pe, 1),
            "br_mv_yi": round(mv_yi, 0),
        }

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> Optional[pd.Series]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── 缓存 ──

    def _cache_path(self, trade_date: str) -> Path:
        d = Path("data/cache/bottom_reversal_screener")
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{trade_date}.csv"

    def _load_cache(self, trade_date: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(trade_date)
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
        return None

    def _save_cache(self, trade_date: str, df: pd.DataFrame) -> None:
        path = self._cache_path(trade_date)
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass
