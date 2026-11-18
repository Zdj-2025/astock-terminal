#!/usr/bin/env python3
"""大盘环境检测 — 数据获取层。

通过 AKShare 获取外部市场数据（美股、美债、北向资金、融资余额），
内部通过 TushareClient 获取A股指数K线。
所有数据带 parquet 缓存，TTL=1天。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

from scripts.core.tushare_client import TushareClient


class MarketDataFetcher:
    """大盘环境数据获取器"""

    CACHE_DIR = Path("data/cache/market_regime")
    TTL_HOURS = 6

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.client = TushareClient()

    # ── A股指数 ──

    def get_index_kline(
        self, ts_code: str, days: int = 120
    ) -> Optional[pd.DataFrame]:
        """获取A股指数K线（上证/深证/创业板）。

        Args:
            ts_code: 如 "000001.SH", "399001.SZ", "399006.SZ"
            days: 回溯天数
        """
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        return self.client.get_index_daily(ts_code=ts_code, start=start, end=end)

    # ── 美股 ──

    def get_sp500(self, days: int = 120) -> Optional[pd.DataFrame]:
        """获取标普500指数K线。

        数据源: tushare index_global(2026-06-19换源)。
        原 ak.index_us_stock_sina 依赖 py_mini_racer 执行加密JS,
        在 py3.13/macOS-ARM 上 dlsym(mr_eval_context): symbol not found 失效。
        """
        cache_key = f"sp500_{days}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
            pro = self.client.pro  # TushareClient.pro property → pro_api
            df = pro.index_global(ts_code="SPX", start_date=start, end_date=end)
            if df is None or df.empty:
                return None
            df["date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            self._save_cache(cache_key, df)
            return df
        except Exception as e:
            print(f"  [DataFetcher] S&P500(tushare)获取失败: {e}")
            return None

    # ── 美债 ──

    def get_us_bond_10y(self, days: int = 120) -> Optional[pd.DataFrame]:
        """获取美国10年期国债收益率"""
        cache_key = f"us_bond_10y_{days}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            df = ak.bond_zh_us_rate()
            if df is None or df.empty:
                return None
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期").tail(days).reset_index(drop=True)
            df = df.rename(columns={
                "日期": "date",
                "美国国债收益率10年": "us_10y",
                "中国国债收益率10年": "cn_10y",
            })
            self._save_cache(cache_key, df)
            return df
        except Exception as e:
            print(f"  [DataFetcher] 美债获取失败: {e}")
            return None

    # ── 北向资金 ──

    def get_northbound_flow(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取北向资金历史流入"""
        cache_key = f"northbound_{days}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
            if df is None or df.empty:
                return None
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期").tail(days).reset_index(drop=True)
            df = df.rename(columns={
                "日期": "date",
                "当日成交净买额": "net_buy",
                "当日资金流入": "flow",
            })
            self._save_cache(cache_key, df)
            return df
        except Exception as e:
            print(f"  [DataFetcher] 北向资金获取失败: {e}")
            return None

    # ── 融资余额 ──

    def get_margin_balance(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取沪市融资余额"""
        cache_key = f"margin_sse_{days}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
            df = ak.stock_margin_sse(start_date=start, end_date=end)
            if df is None or df.empty:
                return None
            df["信用交易日期"] = pd.to_datetime(df["信用交易日期"])
            df = df.sort_values("信用交易日期").tail(days).reset_index(drop=True)
            df = df.rename(columns={
                "信用交易日期": "date",
                "融资余额": "margin_balance",
            })
            self._save_cache(cache_key, df)
            return df
        except Exception as e:
            print(f"  [DataFetcher] 融资余额获取失败: {e}")
            return None

    # ── 涨跌家数 ──

    def get_advance_decline(self) -> Optional[dict]:
        """当日涨跌家数。返回 {'上涨':n,'下跌':n,'平盘':n}。

        优先级(2026-06-19):
          ① 本地全市场parquet统计(最稳, 不依赖网络)
          ② ak.stock_market_activity_legu(备选, 常因网络/接口返回None而失败)
        """
        cache_key = "advance_decline"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # ① 本地parquet统计(最稳)
        try:
            result = self._advance_decline_from_store()
            if result and (result.get("上涨", 0) + result.get("下跌", 0)) > 0:
                self._save_cache(cache_key, result)
                return result
        except Exception as e:
            print(f"  [DataFetcher] 本地涨跌统计失败: {e}")

        # ② akshare legu(备选)
        try:
            df = ak.stock_market_activity_legu()
            if df is None or df.empty:
                return None
            result = {}
            for _, row in df.iterrows():
                val = row["value"]
                if pd.notna(val):
                    val = str(val).replace("%", "").replace(",", "")
                    try:
                        result[row["item"]] = float(val)
                    except ValueError:
                        result[row["item"]] = 0
                else:
                    result[row["item"]] = 0
            self._save_cache(cache_key, result)
            return result
        except Exception as e:
            print(f"  [DataFetcher] 涨跌家数(akshare)获取失败: {e}")
            return None

    def _advance_decline_from_store(self) -> Optional[dict]:
        """从本地 data/store/kline 全市场parquet统计最新交易日涨跌家数。

        用 change_pct 列(>0涨/<0跌/==0平)。不依赖网络, 最稳。
        注: 数据新鲜度取决于 update_all_kline 最近一次运行(可能滞后1-2天)。
        """
        from pathlib import Path

        store = Path("data/store/kline")
        if not store.exists():
            return None
        up = down = flat = 0
        for d in store.iterdir():
            p = d / "D.parquet"
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p, columns=["change_pct"])
                if len(df) == 0:
                    continue
                chg = df["change_pct"].iloc[-1]
                if pd.isna(chg):
                    continue
                if chg > 0:
                    up += 1
                elif chg < 0:
                    down += 1
                else:
                    flat += 1
            except Exception:
                continue
        if up + down + flat == 0:
            return None
        return {"上涨": up, "下跌": down, "平盘": flat}

    # ── 缓存 ──

    def _cache_path(self, key: str) -> Path:
        return self.CACHE_DIR / f"{key}.parquet"

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=self.TTL_HOURS):
                return None
            if key in ("advance_decline",):
                data = json.loads(path.read_text(encoding="utf-8"))
                return data
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, key: str, data) -> None:
        path = self._cache_path(key)
        try:
            if key in ("advance_decline",):
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            else:
                data.to_parquet(path, index=False)
        except Exception:
            pass
