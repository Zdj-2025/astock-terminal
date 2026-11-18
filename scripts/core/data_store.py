#!/usr/bin/env python3
"""公共数据仓库 — 统一数据访问层。

所有消费者（skill/cron/Claude Code/分析脚本）通过此类访问数据。
自动管理缓存、增量更新、并发安全。

设计原则：
  1. 写一次，到处用 — 所有消费者只读，DataStore 负责写入
  2. 增量优先 — 检查 manifest 判断数据新鲜度，只下载缺失部分
  3. 并发安全 — 写临时文件后原子重命名，消费者永远读到一致的数据
  4. manifest 是唯一真相源 — 数据新鲜度只看 manifest

存储结构：
  data/store/
  ├── manifest.json
  ├── kline/{symbol}/{D,W,60min,...}.parquet
  ├── daily_basic/{YYYYMMDD}.parquet
  ├── financial/{symbol}.parquet
  ├── fund_flow/{symbol}.parquet
  ├── concept/{YYYYMMDD}.parquet
  └── realtime/{symbol}.json       (盘中临时, TTL=5分钟)

用法：
  store = DataStore()
  df = store.get_kline("603881", period="D")      # 自动增量更新
  df = store.get_kline("603881", period="W")
  df = store.get_kline("603881", period="60min")
  info = store.get_manifest_summary()               # 查看数据概况
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

# ── 路径常量 ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_STORE_ROOT = _PROJECT_ROOT / "data" / "store"
_MANIFEST_PATH = _STORE_ROOT / "manifest.json"

# 旧数据目录（迁移用）
_OLD_KLINE_STORE = _PROJECT_ROOT / "data" / "kline_store"
_OLD_TUSHARE_CACHE = _PROJECT_ROOT / "data" / "cache" / "tushare"

# ── 周期配置 ──────────────────────────────────────────────────────────

PERIOD_CONFIG = {
    "W":     {"lookback_days": 730, "label": "周线",   "filename": "W.parquet"},
    "D":     {"lookback_days": 365, "label": "日线",   "filename": "D.parquet"},
    "60min": {"lookback_days": 180, "label": "60分钟", "filename": "60min.parquet"},
    "30min": {"lookback_days": 120, "label": "30分钟", "filename": "30min.parquet"},
    "15min": {"lookback_days": 90,  "label": "15分钟", "filename": "15min.parquet"},
    "5min":  {"lookback_days": 60,  "label": "5分钟",  "filename": "5min.parquet"},
}

MINUTE_PERIODS = {"60min", "30min", "15min", "5min"}

# K线数据的 TTL（超过此时间视为需要更新）
_KLINE_TTL = {
    "W": 86400 * 2,       # 周线2天
    "D": 3600 * 4,        # 日线4小时（收盘后更新）
    "60min": 1800,        # 60分钟线30分钟
    "30min": 900,         # 30分钟线15分钟
    "15min": 600,         # 15分钟线10分钟
    "5min": 300,          # 5分钟线5分钟
}

# 其他数据的 TTL
_DAILY_BASIC_TTL = 3600 * 4   # 4小时
_FINANCIAL_TTL = 86400 * 7    # 7天
_FUND_FLOW_TTL = 300          # 5分钟
_CONCEPT_TTL = 86400 * 3      # 3天
_REALTIME_TTL = 300           # 5分钟


# ── Manifest ──────────────────────────────────────────────────────────

class Manifest:
    """数据仓库清单 — 记录所有数据的新鲜度。

    结构：
    {
      "version": 1,
      "created": "2026-05-22T...",
      "last_updated": "2026-05-22T...",
      "kline": {
        "603881": {
          "D":  {"rows": 242, "range": ["2025-05-21","2026-05-21"], "updated": 1779394560.0},
          "W":  {"rows": 103, "range": ["2024-05-21","2026-05-21"], "updated": 1779394560.0}
        }
      },
      "daily_basic": {
        "last_date": "2026-05-21",
        "updated": 1779394560.0
      }
    }
    """

    def __init__(self, path: Path = _MANIFEST_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": 1,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "kline": {},
            "daily_basic": {},
            "financial": {},
            "fund_flow": {},
            "concept": {},
        }

    def _save(self):
        self._data["last_updated"] = datetime.now().isoformat()
        # 原子写入：先写临时文件，再重命名
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    def save(self):
        """外部调用保存（线程安全）。"""
        with self._lock:
            self._save()

    # ── K线 manifest ──

    def get_kline_meta(self, symbol: str, period: str) -> Optional[dict]:
        """获取某只标的某周期的元信息。"""
        return self._data.get("kline", {}).get(symbol, {}).get(period)

    def update_kline_meta(self, symbol: str, period: str, rows: int,
                          date_range: tuple[str, str]):
        """更新K线元信息。"""
        with self._lock:
            if "kline" not in self._data:
                self._data["kline"] = {}
            if symbol not in self._data["kline"]:
                self._data["kline"][symbol] = {}
            self._data["kline"][symbol][period] = {
                "rows": rows,
                "range": list(date_range),
                "updated": time.time(),
            }
            self._save()

    def is_kline_fresh(self, symbol: str, period: str, ttl: Optional[int] = None) -> bool:
        """判断K线数据是否新鲜。"""
        meta = self.get_kline_meta(symbol, period)
        if meta is None:
            return False
        if ttl is None:
            ttl = _KLINE_TTL.get(period, 3600)
        age = time.time() - meta.get("updated", 0)
        return age < ttl

    # ── 其他数据 manifest ──

    def get_meta(self, category: str, key: str = None) -> Optional[dict]:
        """获取某类数据的元信息。"""
        section = self._data.get(category, {})
        if key:
            return section.get(key)
        return section

    def update_meta(self, category: str, key: str, meta: dict):
        """更新某类数据的元信息。"""
        with self._lock:
            if category not in self._data:
                self._data[category] = {}
            self._data[category][key] = {**meta, "updated": time.time()}
            self._save()

    # ── 概览 ──

    def summary(self) -> dict:
        """返回仓库概况。"""
        kline = self._data.get("kline", {})
        symbols = list(kline.keys())
        total_entries = sum(len(v) for v in kline.values())

        # 统计各周期覆盖情况
        period_coverage = {}
        for sym, periods in kline.items():
            for p in periods:
                if p not in period_coverage:
                    period_coverage[p] = 0
                period_coverage[p] += 1

        return {
            "symbols": len(symbols),
            "total_kline_entries": total_entries,
            "period_coverage": period_coverage,
            "last_updated": self._data.get("last_updated"),
        }


# ── DataStore ─────────────────────────────────────────────────────────

class DataStore:
    """公共数据仓库 — 统一数据访问层。

    所有消费者通过此类获取数据。自动判断是否需要增量更新，
    消费者无需关心数据从哪来、怎么拿。
    """

    def __init__(self, store_root: Path = _STORE_ROOT, lazy: bool = True):
        """
        Args:
            store_root: 数据仓库根目录
            lazy: True=需要时才下载; False=初始化时检查全部
        """
        self._root = Path(store_root)
        self._manifest = Manifest(self._root / "manifest.json")
        self._tushare = None  # 延迟初始化
        self._akshare = None  # 延迟初始化

        # 确保目录存在
        for subdir in ("kline", "daily_basic", "financial", "fund_flow",
                        "concept", "realtime"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)

    # ── 延迟加载的数据源 ──

    @property
    def tushare(self):
        """延迟初始化 TushareClient。"""
        if self._tushare is None:
            from scripts.core.tushare_client import TushareClient
            self._tushare = TushareClient()
        return self._tushare

    # ── K线（核心接口）──

    def get_kline(self, symbol: str, period: str = "D",
                  auto_update: bool = True) -> Optional[pd.DataFrame]:
        """获取K线数据。

        自动判断本地数据是否新鲜，过期则增量更新。

        Args:
            symbol: 6位股票/ETF代码
            period: D/W/60min/30min/15min/5min
            auto_update: 是否自动增量更新（默认True）

        Returns:
            DataFrame 或 None
        """
        if period not in PERIOD_CONFIG:
            print(f"  [DataStore] 未知周期: {period}")
            return None

        symbol = self._normalize_symbol(symbol)
        cfg = PERIOD_CONFIG[period]
        path = self._kline_path(symbol, period)

        # 尝试读本地数据
        df = self._read_parquet(path)

        if df is not None and not df.empty:
            if not auto_update:
                return df
            # 检查是否需要增量更新
            if self._manifest.is_kline_fresh(symbol, period):
                return df
            # 增量更新
            updated = self._incremental_update_kline(symbol, period, df)
            return updated if updated is not None else df

        # 本地没有 → 全量下载
        if auto_update:
            return self._full_download_kline(symbol, period)

        return None

    def get_kline_multi(self, symbol: str,
                        periods: List[str] = None) -> Dict[str, pd.DataFrame]:
        """获取多周期K线数据。

        Args:
            symbol: 6位代码
            periods: 周期列表，默认 ["W", "D"]

        Returns:
            {period: DataFrame}
        """
        if periods is None:
            periods = ["W", "D"]

        result = {}
        for p in periods:
            df = self.get_kline(symbol, period=p)
            if df is not None:
                result[p] = df
        return result

    def batch_update_kline(self, symbols: List[str],
                           periods: List[str] = None,
                           force: bool = False) -> dict:
        """批量更新K线数据。

        Args:
            symbols: 股票代码列表
            periods: 周期列表，默认 ["W", "D"]
            force: 是否强制全量重下

        Returns:
            {symbol: {period: {"rows": int, "updated": bool}}}
        """
        if periods is None:
            periods = ["W", "D"]

        results = {}
        for i, symbol in enumerate(symbols):
            symbol = self._normalize_symbol(symbol)
            results[symbol] = {}
            print(f"[{i+1}/{len(symbols)}] {symbol}")
            for p in periods:
                if force:
                    df = self._full_download_kline(symbol, p)
                else:
                    df = self.get_kline(symbol, period=p, auto_update=True)
                if df is not None:
                    results[symbol][p] = {"rows": len(df), "updated": True}
                else:
                    results[symbol][p] = {"rows": 0, "updated": False}
        return results

    # ── 每日行情指标 ──

    def get_daily_basic(self, symbol: str = None,
                        trade_date: str = None) -> Optional[pd.DataFrame]:
        """获取每日行情指标（PE/PB/换手率/市值等）。

        Args:
            symbol: 单只股票代码（可选）
            trade_date: 交易日期 YYYYMMDD（可选，默认最近交易日）
        """
        meta = self._manifest.get_meta("daily_basic", "last_date")
        cache_path = self._root / "daily_basic" / f"{trade_date or 'latest'}.parquet"

        # 检查缓存
        if trade_date and (cache_path).exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < _DAILY_BASIC_TTL:
                df = self._read_parquet(cache_path)
                if df is not None:
                    if symbol:
                        ts_code = self._to_ts_code(symbol)
                        df = df[df["ts_code"] == ts_code]
                    return df

        # 从 Tushare 获取
        try:
            if trade_date is None:
                trade_date = self.tushare.get_recent_trade_day()
            df = self.tushare.get_daily_basic(trade_date=trade_date)
            if df is not None and not df.empty:
                # 缓存全量
                self._write_parquet(df, cache_path)
                self._manifest.update_meta("daily_basic", "last_date", {
                    "last_date": trade_date,
                })
                if symbol:
                    ts_code = self._to_ts_code(symbol)
                    df = df[df["ts_code"] == ts_code]
            return df
        except Exception as e:
            print(f"  [DataStore] daily_basic 获取失败: {e}")
            return None

    # ── 实时行情 ──

    def get_realtime(self, symbol: str) -> Optional[dict]:
        """获取实时行情（盘中有效）。

        使用 TushareClient.get_realtime_quote，结果缓存5分钟。
        """
        symbol = self._normalize_symbol(symbol)
        cache_path = self._root / "realtime" / f"{symbol}.json"

        # 检查缓存
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < _REALTIME_TTL:
                try:
                    with open(cache_path, "r") as f:
                        return json.load(f)
                except Exception:
                    pass

        # 获取实时数据
        try:
            result = self.tushare.get_realtime_quote(symbol)
            if result:
                # 缓存
                tmp = cache_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(result, f, ensure_ascii=False, default=str)
                tmp.replace(cache_path)
            return result
        except Exception as e:
            print(f"  [DataStore] 实时行情获取失败: {e}")
            return None

    def get_realtime_batch(self, symbols: List[str]) -> Dict[str, dict]:
        """批量获取实时行情。"""
        results = {}
        for s in symbols:
            r = self.get_realtime(s)
            if r:
                results[s] = r
        return results

    # ── 财务数据 ──

    def get_financial(self, symbol: str,
                      report_type: str = "income") -> Optional[pd.DataFrame]:
        """获取财务数据。

        Args:
            symbol: 股票代码
            report_type: "income"/"balance_sheet"/"cashflow"
        """
        symbol = self._normalize_symbol(symbol)
        cache_path = self._root / "financial" / f"{symbol}_{report_type}.parquet"
        meta_key = f"{symbol}_{report_type}"

        # 检查 manifest
        meta = self._manifest.get_meta("financial", meta_key)
        if meta and cache_path.exists():
            age = time.time() - meta.get("updated", 0)
            if age < _FINANCIAL_TTL:
                return self._read_parquet(cache_path)

        # 下载
        try:
            ts_code = self._to_ts_code(symbol)
            if report_type == "income":
                df = self.tushare.get_income(ts_code)
            elif report_type == "balance_sheet":
                df = self.tushare.get_balance_sheet(ts_code)
            elif report_type == "cashflow":
                df = self.tushare.get_cashflow(ts_code)
            else:
                return None

            if df is not None and not df.empty:
                self._write_parquet(df, cache_path)
                self._manifest.update_meta("financial", meta_key, {
                    "rows": len(df),
                })
            return df
        except Exception as e:
            print(f"  [DataStore] 财务数据获取失败: {e}")
            return None

    # ── 资金流向 ──

    def get_fund_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取个股资金流向（Tushare moneyflow）。"""
        symbol = self._normalize_symbol(symbol)
        cache_path = self._root / "fund_flow" / f"{symbol}.parquet"
        meta_key = symbol

        meta = self._manifest.get_meta("fund_flow", meta_key)
        if meta and cache_path.exists():
            age = time.time() - meta.get("updated", 0)
            if age < _FUND_FLOW_TTL:
                return self._read_parquet(cache_path)

        try:
            ts_code = self._to_ts_code(symbol)
            df = self.tushare.get_moneyflow(ts_code=ts_code)
            if df is not None and not df.empty:
                self._write_parquet(df, cache_path)
                self._manifest.update_meta("fund_flow", meta_key, {
                    "rows": len(df),
                })
            return df
        except Exception as e:
            print(f"  [DataStore] 资金流向获取失败: {e}")
            return None

    # ── 板块/概念 ──

    def get_concepts(self) -> Optional[pd.DataFrame]:
        """获取概念板块列表。"""
        cache_path = self._root / "concept" / "concepts.parquet"
        meta = self._manifest.get_meta("concept", "concepts")

        if meta and cache_path.exists():
            age = time.time() - meta.get("updated", 0)
            if age < _CONCEPT_TTL:
                return self._read_parquet(cache_path)

        try:
            df = self.tushare.get_concepts()
            if df is not None and not df.empty:
                self._write_parquet(df, cache_path)
                self._manifest.update_meta("concept", "concepts", {
                    "rows": len(df),
                })
            return df
        except Exception as e:
            print(f"  [DataStore] 概念板块获取失败: {e}")
            return None

    # ── 数据迁移 ──

    def migrate_from_old_store(self) -> dict:
        """从旧的 kline_store 迁移数据到新仓库。

        Returns:
            {"migrated": int, "skipped": int, "errors": int}
        """
        result = {"migrated": 0, "skipped": 0, "errors": 0}

        if not _OLD_KLINE_STORE.exists():
            print("  [迁移] 旧 kline_store 不存在，跳过")
            return result

        print("  [迁移] 从 kline_store 迁移到 data/store ...")

        for symbol_dir in _OLD_KLINE_STORE.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name

            # 旧文件名 → 新周期名的映射
            name_map = {
                "daily.parquet": "D",
                "weekly.parquet": "W",
                "60min.parquet": "60min",
                "30min.parquet": "30min",
                "15min.parquet": "15min",
                "5min.parquet": "5min",
            }

            for old_name, period in name_map.items():
                old_path = symbol_dir / old_name
                new_path = self._kline_path(symbol, period)

                if not old_path.exists():
                    continue

                # 新目录已存在且新数据行数 >= 旧数据 → 跳过
                if new_path.exists():
                    try:
                        old_df = pd.read_parquet(old_path)
                        new_df = pd.read_parquet(new_path)
                        if len(new_df) >= len(old_df):
                            result["skipped"] += 1
                            continue
                    except Exception:
                        pass

                # 迁移
                try:
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    df = pd.read_parquet(old_path)
                    # 标准化列名
                    df = self._normalize_kline_columns(df)
                    # 原子写入
                    tmp = new_path.with_suffix(".tmp")
                    df.to_parquet(tmp, index=False)
                    tmp.replace(new_path)

                    # 更新 manifest
                    if "date" in df.columns:
                        dates = pd.to_datetime(df["date"])
                        date_range = (
                            dates.min().strftime("%Y-%m-%d"),
                            dates.max().strftime("%Y-%m-%d"),
                        )
                    else:
                        date_range = ("unknown", "unknown")
                    self._manifest.update_kline_meta(
                        symbol, period, len(df), date_range
                    )
                    result["migrated"] += 1
                    print(f"    ✅ {symbol} {period}: {len(df)}条")
                except Exception as e:
                    result["errors"] += 1
                    print(f"    ❌ {symbol} {period}: {e}")

        print(f"  [迁移] 完成: {result['migrated']}迁移, "
              f"{result['skipped']}跳过, {result['errors']}错误")
        return result

    # ── 概览 ──

    def get_manifest_summary(self) -> dict:
        """获取数据仓库概况。"""
        summary = self._manifest.summary()

        # 补充磁盘使用信息
        total_size = 0
        for p in self._root.rglob("*.parquet"):
            total_size += p.stat().st_size
        for p in self._root.rglob("*.json"):
            total_size += p.stat().st_size
        summary["disk_usage_mb"] = round(total_size / 1024 / 1024, 1)

        return summary

    def print_summary(self):
        """打印数据仓库概况。"""
        s = self.get_manifest_summary()
        print(f"\n{'='*60}")
        print(f"📦 数据仓库概况")
        print(f"{'='*60}")
        print(f"  标的数量: {s['symbols']}")
        print(f"  K线条目: {s['total_kline_entries']}")
        print(f"  磁盘占用: {s['disk_usage_mb']} MB")
        print(f"  最后更新: {s['last_updated']}")
        if s.get("period_coverage"):
            print(f"  周期覆盖:")
            for p, count in sorted(s["period_coverage"].items()):
                print(f"    {p:>6s}: {count} 只标的")
        print(f"{'='*60}\n")

    # ── 内部方法 ──

    def _kline_path(self, symbol: str, period: str) -> Path:
        """获取K线文件路径。"""
        return self._root / "kline" / symbol / PERIOD_CONFIG[period]["filename"]

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化代码格式：去掉后缀，只保留6位数字。"""
        symbol = symbol.strip().upper()
        if "." in symbol:
            symbol = symbol.split(".")[0]
        for prefix in ("SH", "SZ", "BJ"):
            if symbol.startswith(prefix):
                symbol = symbol[2:]
                break
        return symbol

    def _to_ts_code(self, symbol: str) -> str:
        """纯数字 → Tushare格式。"""
        from scripts.core.tushare_client import to_ts_code
        return to_ts_code(symbol)

    # ── 市场指数（上证/深证/创业板）──

    _INDEX_CODES = {
        '000001SH': '000001.SH',  # 上证指数
        '399001SZ': '399001.SZ',  # 深证成指
        '399006SZ': '399006.SZ',  # 创业板指
        '000688SH': '000688.SH',  # 科创50
    }

    def get_index_kline(self, index_key: str = '000001SH',
                         start_date: str = '20200101',
                         auto_update: bool = True) -> Optional[pd.DataFrame]:
        """获取市场指数日线数据，自动增量更新。

        与 get_kline() 的区别：
        - 用 index_daily 接口（不是股票 daily）
        - 默认从2020年起，4年+数据
        - 存储在 kline/{index_key}/ 目录（如 kline/000001SH/）
        - Tushare指数接口无频率限制，放心增量

        Args:
            index_key: 存储key，如 '000001SH'
            start_date: 起始日期 YYYYMMDD
            auto_update: 是否自动增量
        """
        ts_code = self._INDEX_CODES.get(index_key)
        if not ts_code:
            print(f"  [DataStore] 未知指数: {index_key}")
            return None

        path = self._root / 'kline' / index_key / 'D.parquet'

        # 读本地
        df = self._read_parquet(path)
        now = time.time()

        if df is not None and not df.empty and len(df) > 100:
            if not auto_update:
                return df
            # 检查是否需要增量（指数日线TTL=4小时）
            meta = self._manifest.get_kline_meta(index_key, 'D')
            if meta:
                last_update = meta.get('updated', 0)
                if now - last_update < _KLINE_TTL['D']:
                    return df

            # 增量更新
            dt_col = 'trade_date' if 'trade_date' in df.columns else 'date'
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)
            last_date = df[dt_col].iloc[-1].strftime('%Y%m%d')
            today = pd.Timestamp.now().strftime('%Y%m%d')

            if last_date >= today:
                return df  # 已是最新

            try:
                new_df = self.tushare.pro.index_daily(
                    ts_code=ts_code,
                    start_date=(pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y%m%d'),
                    end_date=today,
                )
                if new_df is not None and len(new_df) > 0:
                    new_df['trade_date'] = pd.to_datetime(new_df['trade_date'])
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=['trade_date'], keep='last')
                    df = df.sort_values('trade_date').reset_index(drop=True)
                    self._write_parquet(path, df)
                    dates = pd.to_datetime(df['trade_date'])
                    self._manifest.update_kline_meta(index_key, 'D', len(df),
                        (dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d')))
                    print(f"  [DataStore] 指数{index_key}增量+{len(new_df)}行 → {len(df)}行")
                return df
            except Exception as e:
                print(f"  [DataStore] 指数增量更新失败: {e}")
                return df

        # 全量下载
        try:
            df = self.tushare.pro.index_daily(
                ts_code=ts_code, start_date=start_date
            )
            if df is not None and len(df) > 0:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._write_parquet(path, df)
                dates = pd.to_datetime(df['trade_date'])
                self._manifest.update_kline_meta(index_key, 'D', len(df),
                    (dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d')))
                print(f"  [DataStore] 指数{index_key}全量下载{len(df)}行")
                return df
        except Exception as e:
            print(f"  [DataStore] 指数全量下载失败: {e}")
            return None

    def _read_parquet(self, path: Path) -> Optional[pd.DataFrame]:
        """安全读取 parquet 文件。"""
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _write_parquet(self, df: pd.DataFrame, path: Path):
        """原子写入 parquet 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)

    def _validate_kline(self, df: pd.DataFrame, symbol: str, period: str) -> pd.DataFrame:
        """验证K线数据质量：NaN检查、OHLCV一致性。

        Args:
            df: K线 DataFrame
            symbol: 标的代码（用于日志）
            period: 周期（用于日志）

        Returns:
            清洗后的 DataFrame
        """
        if df is None or df.empty:
            return df

        ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        if not ohlcv_cols:
            return df

        original_len = len(df)
        close_nan = 0

        # 1. NaN 检查
        nan_counts = df[ohlcv_cols].isna().sum()
        if nan_counts.any():
            nan_cols = {c: int(nan_counts[c]) for c in ohlcv_cols if nan_counts[c] > 0}
            print(f"  [DataStore] {symbol} {period} NaN警告: {nan_cols}")

        # 2. 删除 close 为 NaN 的行
        if "close" in df.columns:
            close_nan = int(df["close"].isna().sum())
            if close_nan > 0:
                df = df.dropna(subset=["close"]).reset_index(drop=True)

        # 3. OHLCV 一致性验证
        bad_mask = pd.Series(False, index=df.index)
        if all(c in df.columns for c in ("high", "open")):
            bad_mask |= df["high"] < df["open"]
        if all(c in df.columns for c in ("high", "close")):
            bad_mask |= df["high"] < df["close"]
        if all(c in df.columns for c in ("low", "open")):
            bad_mask |= df["low"] > df["open"]
        if all(c in df.columns for c in ("low", "close")):
            bad_mask |= df["low"] > df["close"]
        if "volume" in df.columns:
            bad_mask |= df["volume"] < 0

        bad_count = bad_mask.sum()
        if bad_count > 0:
            df = df[~bad_mask].reset_index(drop=True)

        # 4. 汇总日志
        total_removed = original_len - len(df)
        if total_removed > 0:
            print(f"  [DataStore] {symbol} {period} 数据清洗: "
                  f"移除 {total_removed} 条异常行 "
                  f"(NaN close={close_nan}, "
                  f"OHLCV不一致={bad_count}), 剩余 {len(df)} 条")

        return df

    def _normalize_kline_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化K线列名。"""
        rename_map = {
            "trade_date": "date",
            "trade_time": "date",
            "pct_chg": "change_pct",
        }
        # 先处理vol/volume冲突：合并两列，优先取非NaN值
        if "vol" in df.columns and "volume" in df.columns:
            # 两列都存在：用vol填充volume的NaN
            df["volume"] = df["volume"].fillna(df["vol"])
            df = df.drop(columns=["vol"])
        elif "vol" in df.columns:
            # 只有vol，改名为volume
            df = df.rename(columns={"vol": "volume"})
        
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date" if "date" in df.columns else df.columns[0]).reset_index(drop=True)
        return df

    def _full_download_kline(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """全量下载K线数据（多数据源自动降级）。

        降级链路: Tushare → AKShare → 返回None
        """
        cfg = PERIOD_CONFIG[period]
        lookback = cfg["lookback_days"]
        start = (datetime.now() - timedelta(days=lookback)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")

        df = None
        source_used = ""

        # ── 数据源1: Tushare ──
        try:
            is_etf = self._is_etf(symbol)
            if is_etf:
                df = self._download_etf_kline(symbol, period, start, end)
            else:
                df = self.tushare.get_kline(
                    symbol, freq=period, start=start, end=end, adj="qfq"
                )
            if df is not None and not df.empty:
                source_used = "Tushare"
        except Exception as e:
            err_msg = str(e)
            if "频率超限" in err_msg or "limit" in err_msg.lower():
                print(f"  [DataStore] {symbol} {period} Tushare限流，降级AKShare...")
            else:
                print(f"  [DataStore] {symbol} {period} Tushare失败({e})，降级AKShare...")

        # ── 数据源2: AKShare ──
        if df is None or df.empty:
            try:
                df = self._download_kline_akshare(symbol, period, start, end)
                if df is not None and not df.empty:
                    source_used = "AKShare"
            except Exception as e:
                print(f"  [DataStore] {symbol} {period} AKShare也失败: {e}")

        # 写入存储
        if df is not None and not df.empty:
            df = self._normalize_kline_columns(df)
            df = self._validate_kline(df, symbol, period)
            path = self._kline_path(symbol, period)
            self._write_parquet(df, path)

            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                date_range = (
                    dates.min().strftime("%Y-%m-%d"),
                    dates.max().strftime("%Y-%m-%d"),
                )
            else:
                date_range = ("unknown", "unknown")
            self._manifest.update_kline_meta(
                symbol, period, len(df), date_range
            )
            print(f"  [DataStore] {symbol} {cfg['label']}: "
                  f"下载 {len(df)}条 [{source_used}]")
            return df

        print(f"  [DataStore] {symbol} {period} 所有数据源均失败")
        return None

    def _incremental_update_kline(self, symbol: str, period: str,
                                   existing: pd.DataFrame) -> Optional[pd.DataFrame]:
        """增量更新K线数据（多数据源自动降级）。"""
        if existing is None or existing.empty:
            return self._full_download_kline(symbol, period)

        if "date" not in existing.columns:
            return None
        last_date = pd.to_datetime(existing["date"].iloc[-1])
        if last_date >= pd.Timestamp.now() - timedelta(hours=1):
            return None

        next_start = (last_date + timedelta(minutes=1)).strftime("%Y%m%d")
        end = datetime.now().strftime("%Y%m%d")

        new_df = None

        # ── 数据源1: Tushare ──
        try:
            is_etf = self._is_etf(symbol)
            if is_etf:
                new_df = self._download_etf_kline(symbol, period, next_start, end)
            else:
                new_df = self.tushare.get_kline(
                    symbol, freq=period, start=next_start, end=end, adj="qfq"
                )
        except Exception as e:
            err_msg = str(e)
            if "频率超限" in err_msg or "limit" in err_msg.lower():
                print(f"  [DataStore] {symbol} {period} Tushare限流，降级AKShare...")
            else:
                print(f"  [DataStore] {symbol} {period} Tushare失败，降级AKShare...")

        # ── 数据源2: AKShare ──
        if new_df is None or (hasattr(new_df, 'empty') and new_df.empty):
            try:
                new_df = self._download_kline_akshare(symbol, period, next_start, end)
            except Exception as e:
                print(f"  [DataStore] {symbol} {period} AKShare增量也失败: {e}")

        if new_df is None or (hasattr(new_df, 'empty') and new_df.empty):
            self._manifest.update_kline_meta(
                symbol, period, len(existing),
                (
                    pd.to_datetime(existing["date"].iloc[0]).strftime("%Y-%m-%d"),
                    pd.to_datetime(existing["date"].iloc[-1]).strftime("%Y-%m-%d"),
                ),
            )
            return None

        new_df = self._normalize_kline_columns(new_df)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
        merged = self._validate_kline(merged, symbol, period)

        new_count = len(merged) - len(existing)
        path = self._kline_path(symbol, period)
        self._write_parquet(merged, path)

        dates = pd.to_datetime(merged["date"])
        self._manifest.update_kline_meta(
            symbol, period, len(merged),
            (dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")),
        )

        if new_count > 0:
            print(f"  [DataStore] {symbol} {PERIOD_CONFIG[period]['label']}: "
                  f"+{new_count}条 (总计 {len(merged)}条)")
        return merged

    def _is_etf(self, symbol: str) -> bool:
        """判断是否为ETF。"""
        return symbol[:2] in ("51", "52", "15", "16", "50", "58")

    def _download_etf_kline(self, symbol: str, period: str,
                            start: str, end: str) -> Optional[pd.DataFrame]:
        """下载ETF K线数据（借用 MultiPeriodDownloader 的逻辑）。"""
        import tushare as ts
        from scripts.core.tushare_client import _freq_map

        ts_code = self._to_ts_code(symbol)
        freq = _freq_map(period)

        # ETF周线：从日线重采样
        if freq == "W":
            df = self._download_etf_kline(symbol, "D", start, end)
            if df is None or df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            weekly = df.resample("W-FRI").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum", "amount": "sum",
            }).dropna(subset=["open"])
            return weekly.reset_index()

        # ETF日线
        if freq == "D":
            self.tushare._throttle()
            s = start.replace("-", "")
            e = end.replace("-", "")
            df = ts.pro_bar(
                ts_code=ts_code, freq="D",
                start_date=s, end_date=e,
                adj="qfq", asset="FD",
            )
            if df is None or df.empty:
                return None
            col_map = {"trade_date": "date", "vol": "volume", "pct_chg": "change_pct"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)

        # ETF分钟线
        s = start if "-" in start else f"{start[:4]}-{start[4:6]}-{start[6:8]} 09:30:00"
        e = end if "-" in end else f"{end[:4]}-{end[4:6]}-{end[6:8]} 15:00:00"
        self.tushare._throttle()
        df = ts.pro_bar(
            ts_code=ts_code, freq=freq,
            start_date=s, end_date=e,
            adj="qfq", asset="FD",
        )
        if df is None or df.empty:
            return None
        col_map = {"trade_date": "date", "trade_time": "date", "vol": "volume", "pct_chg": "change_pct"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ── AKShare 降级数据源 ──

    def _download_kline_akshare(self, symbol: str, period: str,
                                 start: str, end: str) -> Optional[pd.DataFrame]:
        """通过 AKShare 下载K线数据（Tushare 失败时的备用源）。

        支持: D, W, 60min, 30min, 15min, 5min
        ETF和股票都走 stock_zh_a_hist / stock_zh_a_hist_min_em
        """
        import os
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'
        import akshare as ak

        # AKShare 周期映射
        period_map = {
            "D": "daily",
            "W": "weekly",
        }
        min_period_map = {
            "60min": "60",
            "30min": "30",
            "15min": "15",
            "5min": "5",
        }

        # 格式化日期为 YYYYMMDD
        s = start.replace("-", "")[:8]
        e = end.replace("-", "")[:8]

        try:
            if period in period_map:
                # 日线/周线
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period=period_map[period],
                    start_date=s,
                    end_date=e,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    return None
                # 标准化列名
                col_map = {
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount", "涨跌幅": "change_pct",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)

            elif period in min_period_map:
                # 分钟线
                df = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=min_period_map[period],
                    start_date=f"{s[:4]}-{s[4:6]}-{s[6:8]} 09:30:00",
                    end_date=f"{e[:4]}-{e[4:6]}-{e[6:8]} 15:00:00",
                    adjust="qfq",
                )
                if df is None or df.empty:
                    return None
                col_map = {
                    "时间": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "成交额": "amount",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)

            else:
                return None

        except Exception as e:
            print(f"  [DataStore] AKShare下载 {symbol} {period} 失败: {e}")
            return None


# ── CLI 入口 ──────────────────────────────────────────────────────────

def main():
    """命令行工具：数据仓库管理。"""
    import argparse

    parser = argparse.ArgumentParser(description="公共数据仓库管理")
    sub = parser.add_subparsers(dest="command")

    # summary
    sub.add_parser("summary", help="查看数据仓库概况")

    # update
    p_update = sub.add_parser("update", help="更新K线数据")
    p_update.add_argument("--symbol", help="单只标的代码")
    p_update.add_argument("--periods", default="W,D", help="周期，逗号分隔")
    p_update.add_argument("--all-periods", action="store_true", help="所有周期")
    p_update.add_argument("--force", action="store_true", help="强制全量重下")

    # migrate
    sub.add_parser("migrate", help="从旧 kline_store 迁移数据")

    # get
    p_get = sub.add_parser("get", help="获取数据并显示概况")
    p_get.add_argument("symbol", help="标的代码")
    p_get.add_argument("--period", default="D", help="周期")

    args = parser.parse_args()

    store = DataStore()

    if args.command == "summary" or args.command is None:
        store.print_summary()

    elif args.command == "update":
        periods = list(PERIOD_CONFIG.keys()) if args.all_periods else args.periods.split(",")
        if args.symbol:
            symbols = [args.symbol]
        else:
            # 从 watchlist 读取
            from scripts.data.multi_period_downloader import load_watchlist
            items = load_watchlist()
            symbols = [it["code"] for it in items]

        if args.force:
            for s in symbols:
                for p in periods:
                    store._full_download_kline(store._normalize_symbol(s), p)
        else:
            store.batch_update_kline(
                [store._normalize_symbol(s) for s in symbols],
                periods=periods,
            )

    elif args.command == "migrate":
        store.migrate_from_old_store()

    elif args.command == "get":
        symbol = store._normalize_symbol(args.symbol)
        period = args.period
        df = store.get_kline(symbol, period=period)
        if df is not None:
            print(f"\n{symbol} {period}: {len(df)}条")
            print(f"  时间范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            print(f"  列: {list(df.columns)}")
            print(f"\n最近5条:")
            print(df.tail().to_string(index=False))
        else:
            print(f"❌ {symbol} {period} 无数据")


if __name__ == "__main__":
    main()
