#!/usr/bin/env python3
"""数据骨干 — 共享数据访问、缓存去重、配置管理、事件总线。

所有编排器共享同一个DataBackbone实例，避免重复获取数据。
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient


class _EventBus:
    """模块间通信的事件总线"""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)

    def on(self, event: str, handler: Callable):
        self._listeners[event].append(handler)

    def emit(self, event: str, data: Any = None):
        for handler in self._listeners.get(event, []):
            try:
                handler(data)
            except Exception:
                pass


class DataBackbone:
    """单例数据骨干 — 共享数据访问层"""

    _instance: Optional["DataBackbone"] = None

    def __init__(self):
        self._ts: Optional[TushareClient] = None
        self._cache: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._loaded_at: Dict[str, datetime] = {}
        self.events = _EventBus()
        self._portfolio_df = None
        self._capital_df = None

    @classmethod
    def get(cls) -> "DataBackbone":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @property
    def ts(self) -> TushareClient:
        if self._ts is None:
            self._ts = TushareClient()
        return self._ts

    # ── 配置加载 ──────────────────────────────────────────────────

    def load_config(self, name: str) -> dict:
        if name in self._config:
            return self._config[name]
        path = os.path.join(PROJECT_ROOT, "config", name)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self._config[name] = cfg
        return cfg

    @property
    def risk_rules(self) -> dict:
        return self.load_config("risk_rules.json")

    @property
    def analysis_config(self) -> dict:
        return self.load_config("analysis_config.json")

    @property
    def notification_config(self) -> dict:
        return self.load_config("notification_config.json")

    # ── 持仓/资金 ─────────────────────────────────────────────────

    def load_portfolio(self):
        if self._portfolio_df is not None:
            return self._portfolio_df
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "config", "portfolio.xlsx")
        if not os.path.exists(path):
            return None
        try:
            # 尝试已知sheet名
            for sheet in ['持仓信息', 'Sheet1']:
                try:
                    df = pd.read_excel(path, sheet_name=sheet)
                    if not df.empty:
                        self._portfolio_df = df
                        return df
                except Exception:
                    continue
            # fallback: 第一个sheet
            df = pd.read_excel(path, sheet_name=0)
            self._portfolio_df = df
            return df
        except Exception:
            return None

    def load_capital(self):
        if self._capital_df is not None:
            return self._capital_df
        import pandas as pd
        path = os.path.join(PROJECT_ROOT, "config", "capital.xlsx")
        if not os.path.exists(path):
            return None
        try:
            for sheet in ['资金池', 'Sheet1']:
                try:
                    df = pd.read_excel(path, sheet_name=sheet)
                    if not df.empty:
                        self._capital_df = df
                        return df
                except Exception:
                    continue
            df = pd.read_excel(path, sheet_name=0)
            self._capital_df = df
            return df
        except Exception:
            return None

    def invalidate_portfolio(self):
        self._portfolio_df = None
        self._capital_df = None

    # ── 共享缓存 ──────────────────────────────────────────────────

    def get_cached(self, key: str, ttl_seconds: int = 3600) -> Optional[Any]:
        if key in self._cache:
            elapsed = (datetime.now() - self._loaded_at[key]).total_seconds()
            if elapsed < ttl_seconds:
                return self._cache[key]
        return None

    def set_cached(self, key: str, data: Any):
        self._cache[key] = data
        self._loaded_at[key] = datetime.now()

    def get_market_data(self, trade_date: str) -> Optional[Any]:
        key = f"daily_{trade_date}"
        cached = self.get_cached(key, ttl_seconds=1800)
        if cached is not None:
            return cached
        df = self.ts._call(self.ts._pro.daily, trade_date=trade_date,
                           fields="ts_code,close,pct_chg,vol,amount")
        if df is not None and not df.empty:
            self.set_cached(key, df)
            return df
        return None

    # ── 数据源健康检查 ────────────────────────────────────────────

    def health_check(self) -> dict:
        results = {}
        try:
            df = self.ts.get_trade_cal(start_date="20260101", end_date="20260110")
            results["tushare"] = "ok" if df is not None and not df.empty else "error"
        except Exception as e:
            results["tushare"] = f"error: {e}"

        try:
            from scripts.core.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            results["akshare"] = "ok"
        except Exception:
            results["akshare"] = "error"

        results["config_dir"] = "ok" if os.path.isdir(
            os.path.join(PROJECT_ROOT, "config")) else "missing"
        results["tracker_db"] = "ok" if os.path.exists(
            os.path.join(PROJECT_ROOT, "data", "tracker.db")) else "missing"

        return results

    # ── 交易日历 ──────────────────────────────────────────────────

    def get_latest_trade_date(self) -> str:
        return self.ts.get_recent_trade_day(1)

    def is_trade_day(self, date: str = None) -> bool:
        return self.ts.is_trade_day(date)
