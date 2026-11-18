#!/usr/bin/env python3
"""周末批量数据下载器 v1.0

功能：
1. 批量下载所有关注标的的多周期K线（W/D/60min/30min/15min/5min）
2. 下载交易日历
3. 下载技术指标数据（每日行情指标：PE/PB/换手率/市值等）
4. 数据完整性校验
5. 智能限流处理

使用：
    python scripts/core/weekend_downloader.py --all          # 全量下载
    python scripts/core/weekend_downloader.py --kline         # 只下K线
    python scripts/core/weekend_downloader.py --calendar      # 只下交易日历
    python scripts/core/weekend_downloader.py --check         # 只做完整性检查
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from scripts.core.data_store import DataStore, PERIOD_CONFIG

# ── 全部需要下载的标的 ──

# 持仓 + 关注 + 商业航天 + 大盘指数
ALL_SYMBOLS = [
    # 持仓
    "603881",  # 数据港
    "512480",  # 半导体ETF
    "512760",  # 芯片ETF
    "603617",  # 君禾股份
    "002250",  # 联化科技
    "301060",  # 兰卫医学
    # 关注
    "300365",  # 恒华科技
    "600339",  # 中油工程
    "002800",  # 天顺股份
    "600488",  # 津药药业
    "002452",  # 长高电新
    "000682",  # 东方电子
    "600888",  # 新疆众和
    "601678",  # 华泰柏瑞
    "603486",  # 科沃斯
    "000887",  # 中鼎股份
    # 商业航天
    "000733",  # 振华科技
    "300699",  # 光威复材
    "300034",  # 钢研高纳
    "688122",  # 西部超导
    "159227",  # 航天ETF
    "563530",  # 卫星ETF
    # 大盘指数
    "000001",  # 上证指数
    "399001",  # 深证成指
    "399006",  # 创业板指
    "000300",  # 沪深300
]

# K线周期优先级：日线/周线先下（无限流），分钟线后下（1次/小时）
PERIODS_PHASE1 = ["W", "D"]       # 无限流
PERIODS_PHASE2 = ["60min"]        # 1次/小时
PERIODS_PHASE3 = ["30min"]        # 1次/小时
PERIODS_PHASE4 = ["15min"]        # 1次/小时
PERIODS_PHASE5 = ["5min"]         # 1次/小时


def load_watchlist_symbols() -> List[str]:
    """从watchlist.json加载所有标的代码"""
    wl_path = Path(project_root) / "config" / "watchlist.json"
    if not wl_path.exists():
        return ALL_SYMBOLS
    with open(wl_path) as f:
        data = json.load(f)
    codes = set()
    for key in ("holdings", "watchlist", "commercial_aerospace"):
        for item in data.get(key, []):
            code = item.get("code", "")
            if code and code.isdigit():
                codes.add(code)
    # 补充固定列表
    codes.update(ALL_SYMBOLS)
    return sorted(codes)


def download_kline_phase(store: DataStore, symbols: List[str],
                         periods: List[str], phase_name: str,
                         delay_sec: float = 3.0,
                         max_retries: int = 3,
                         retry_wait: int = 3700) -> Dict:
    """下载一个阶段的K线数据。

    Args:
        store: DataStore实例
        symbols: 标的列表
        periods: 周期列表
        phase_name: 阶段名称
        delay_sec: 每次下载间隔（秒）
        max_retries: 最大重试次数
        retry_wait: 限流后等待时间（秒），默认3700≈62分钟

    Returns:
        {symbol: {period: {"rows": int, "status": str}}}
    """
    results = {}
    total = len(symbols) * len(periods)
    done = 0
    failed = []

    for symbol in symbols:
        results[symbol] = {}
        for period in periods:
            done += 1
            cfg = PERIOD_CONFIG.get(period, {})
            label = cfg.get("label", period)
            print(f"  [{done}/{total}] {symbol} {label}({period})...", end=" ", flush=True)

            success = False
            for attempt in range(max_retries):
                try:
                    df = store.get_kline(symbol, period=period, auto_update=True)
                    if df is not None and not df.empty:
                        results[symbol][period] = {
                            "rows": len(df),
                            "range": f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}",
                            "status": "ok"
                        }
                        print(f"✅ {len(df)}条")
                        success = True
                        break
                    else:
                        results[symbol][period] = {"rows": 0, "status": "empty"}
                        print("⚠️ 无数据")
                        success = True
                        break
                except Exception as e:
                    err = str(e)
                    if "频率超限" in err or "limit" in err.lower():
                        if attempt < max_retries - 1:
                            wait_min = retry_wait // 60
                            print(f"⏳ 限流，等{wait_min}分钟...")
                            time.sleep(retry_wait)
                        else:
                            print(f"❌ 限流{max_retries}次，跳过")
                            results[symbol][period] = {"rows": 0, "status": "rate_limited"}
                            failed.append(f"{symbol}_{period}")
                    else:
                        print(f"❌ {err[:50]}")
                        results[symbol][period] = {"rows": 0, "status": f"error: {err[:50]}"}
                        break

            if not success:
                failed.append(f"{symbol}_{period}")

            # 间隔（日线/周线短一些，分钟线长一些）
            if delay_sec > 0:
                time.sleep(delay_sec)

    if failed:
        print(f"\n  ⚠️ {phase_name}失败项: {failed}")
    return results


def download_trade_calendar(store: DataStore) -> Optional[pd.DataFrame]:
    """下载交易日历"""
    print("\n📅 下载交易日历...")
    try:
        import tushare as ts
        pro = ts.pro_api()

        # 获取2020-2027年的交易日历
        df = pro.trade_cal(exchange='SSE', start_date='20200101', end_date='20271231')
        if df is not None and not df.empty:
            # 保存
            cal_path = Path(project_root) / "data" / "store" / "trade_calendar.parquet"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cal_path, index=False)

            open_days = df[df['is_open'] == 1]
            print(f"  ✅ 交易日历: {len(df)}条, 其中交易日{len(open_days)}天")
            print(f"  范围: {df['cal_date'].min()} ~ {df['cal_date'].max()}")
            return df
    except Exception as e:
        print(f"  ❌ 交易日历下载失败: {e}")
    return None


def check_data_integrity(store: DataStore, symbols: List[str],
                         periods: List[str]) -> Dict:
    """检查数据完整性。

    对比交易日历和实际数据，找出缺失的交易日。
    """
    print("\n🔍 数据完整性检查...")
    cal_path = Path(project_root) / "data" / "store" / "trade_calendar.parquet"

    # 加载交易日历
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        trade_dates = set(cal[cal['is_open'] == 1]['cal_date'].astype(str).tolist())
    else:
        print("  ⚠️ 无交易日历，跳过完整性检查")
        return {}

    issues = {}

    for symbol in symbols:
        issues[symbol] = {}
        for period in periods:
            if period not in ("D", "W"):
                continue  # 分钟线不做日期完整性检查

            df = store.get_kline(symbol, period=period, auto_update=False)
            if df is None or df.empty:
                issues[symbol][period] = {"status": "无数据"}
                continue

            # 检查日期范围
            dates = set(df['date'].astype(str).str[:10].tolist())
            start = min(dates).replace("-", "")
            end = max(dates).replace("-", "")

            # 找交易日历中的交易日
            expected = {d for d in trade_dates if start <= d <= end}

            # 对比
            missing = expected - dates
            extra = dates - expected

            status = "✅" if not missing else f"⚠️ 缺{len(missing)}天"
            issues[symbol][period] = {
                "status": status,
                "rows": len(df),
                "range": f"{start} ~ {end}",
                "missing_days": len(missing),
                "extra_days": len(extra),
            }

            if missing and len(missing) <= 20:
                issues[symbol][period]["missing_dates"] = sorted(missing)[:20]

    # 打印摘要
    ok_count = sum(1 for s in issues.values() for p in s.values() if "✅" in str(p.get("status", "")))
    warn_count = sum(1 for s in issues.values() for p in s.values() if "⚠️" in str(p.get("status", "")))
    no_count = sum(1 for s in issues.values() for p in s.values() if p.get("status") == "无数据")

    print(f"\n  完整性报告: ✅{ok_count}  ⚠️{warn_count}  ❌{no_count}")
    for symbol, periods_data in issues.items():
        for period, info in periods_data.items():
            if "⚠️" in str(info.get("status", "")) or info.get("status") == "无数据":
                print(f"    {info['status']} {symbol} {period}: {info.get('range','')} "
                      f"(缺{info.get('missing_days',0)}天)")

    return issues


def main():
    parser = argparse.ArgumentParser(description="周末批量数据下载器")
    parser.add_argument("--all", action="store_true", help="全量下载（K线+日历+检查）")
    parser.add_argument("--kline", action="store_true", help="只下载K线")
    parser.add_argument("--calendar", action="store_true", help="只下载交易日历")
    parser.add_argument("--check", action="store_true", help="只做完整性检查")
    parser.add_argument("--symbols", default=None, help="指定标的，逗号分隔")
    parser.add_argument("--periods", default=None, help="指定周期，逗号分隔")
    parser.add_argument("--fast", action="store_true", help="快速模式（跳过分钟线）")
    args = parser.parse_args()

    if not any([args.all, args.kline, args.calendar, args.check]):
        args.all = True  # 默认全量

    store = DataStore()
    symbols = load_watchlist_symbols() if not args.symbols else args.symbols.split(",")
    print(f"🎯 标的数量: {len(symbols)}")
    print(f"🎯 标的列表: {', '.join(symbols)}")

    start_time = time.time()

    # ── 1. 交易日历 ──
    if args.all or args.calendar:
        download_trade_calendar(store)

    # ── 2. K线数据 ──
    if args.all or args.kline:
        print(f"\n{'='*60}")
        print("📊 Phase 1: 日线 + 周线（无限流）")
        print(f"{'='*60}")
        r1 = download_kline_phase(store, symbols, PERIODS_PHASE1,
                                  "日线+周线", delay_sec=2.0, max_retries=2)

        if not args.fast:
            for phase_periods, phase_name, phase_label in [
                (PERIODS_PHASE2, "Phase 2", "60分钟线"),
                (PERIODS_PHASE3, "Phase 3", "30分钟线"),
                (PERIODS_PHASE4, "Phase 4", "15分钟线"),
                (PERIODS_PHASE5, "Phase 5", "5分钟线"),
            ]:
                print(f"\n{'='*60}")
                print(f"📊 {phase_name}: {phase_label}（1次/小时限流）")
                print(f"{'='*60}")
                # 分钟线限流：每次下载后等62分钟
                r = download_kline_phase(store, symbols, phase_periods,
                                        phase_name, delay_sec=3.0,
                                        max_retries=2, retry_wait=3700)

    # ── 3. 每日行情指标（PE/PB/换手率等）──
    if args.all or args.kline:
        print(f"\n{'='*60}")
        print("📈 下载每日行情指标（PE/PB/换手率/市值等）")
        print(f"{'='*60}")
        for i, symbol in enumerate(symbols):
            print(f"  [{i+1}/{len(symbols)}] {symbol}...", end=" ", flush=True)
            try:
                df = store.get_daily_basic(symbol)
                if df is not None:
                    print(f"✅ {len(df)}条")
                else:
                    print("⚠️ 无数据")
            except Exception as e:
                print(f"❌ {str(e)[:50]}")
            time.sleep(0.5)

    # ── 4. 完整性检查 ──
    if args.all or args.check:
        check_periods = ["W", "D"]
        if not args.fast:
            check_periods += ["60min", "30min", "15min", "5min"]
        check_data_integrity(store, symbols, check_periods)

    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed/60:.1f}分钟")
    print("✅ 完成！")


if __name__ == "__main__":
    main()
