#!/usr/bin/env python3
"""选股器回测引擎 — 时间旅行回测。

在历史截断日运行选股器（只用截断日之前数据），然后看前瞻收益率。
支持多窗口回测，增量缓存，中断续跑。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.core.tushare_client import TushareClient, from_ts_code
from scripts.screener.screener import StockScreener


class ScreenerBacktester:
    """选股器回测引擎"""

    # 评分算法变更时递增此版本号，旧缓存自动失效
    CACHE_VERSION = 3

    def __init__(self, cache_dir: str = "data/cache/screener_backtest"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = TushareClient()

    def run(
        self,
        template: str,
        snapshot_dates: list[str],
        forward_days: list[int] = (5, 10, 20),
        benchmark: str = "000001.SH",
        quick: bool = False,
        top: int = 20,
        **screener_kwargs,
    ) -> dict:
        """执行多窗口回测。

        Args:
            template: 选股模板(wave3/wave3_startup/growth等)
            snapshot_dates: 截断日期列表["20240601", "20240701", ...]
            forward_days: 前瞻天数列表
            benchmark: 基准指数
            quick: 快速模式
            top: 每个窗口最多选N只
            **screener_kwargs: 传给StockScreener.run()的额外参数

        Returns:
            dict: {windows: [...], aggregate: {...}}
        """
        windows = []

        for i, snap_date in enumerate(snapshot_dates):
            print(f"\n{'='*60}")
            print(f"  回测窗口 {i+1}/{len(snapshot_dates)}: {snap_date}")
            print(f"{'='*60}")

            # 检查增量缓存
            cached = self._load_window_cache(template, snap_date)
            if cached is not None:
                print(f"  [缓存命中] {snap_date}")
                windows.append(cached)
                continue

            # resolve到最近交易日
            trade_date = self._resolve_snapshot_date(snap_date)
            if trade_date is None:
                print(f"  ⚠ 无法解析交易日: {snap_date}")
                continue

            print(f"  截断日: {snap_date} → 交易日: {trade_date}")

            # 大盘环境检测
            regime_info = None
            try:
                from scripts.market_regime.regime_detector import MarketRegimeDetector
                regime_detector = MarketRegimeDetector()
                regime_info = regime_detector.detect(as_of_date=trade_date)
                regime_label = {
                    "risk_on": "多头", "neutral": "震荡", "risk_off": "空头",
                }.get(regime_info["regime"], regime_info["regime"])
                print(f"  大盘环境: {regime_label} "
                      f"(环境分{regime_info['score']:+.2f})")
            except Exception:
                pass

            # 运行选股器
            screener = StockScreener()
            result = screener.run(
                template=template,
                quick=quick,
                top=top,
                as_of_date=trade_date,
                **screener_kwargs,
            )

            selected = result["results"]
            meta = result["meta"]

            if selected.empty:
                window_data = {
                    "snapshot_date": snap_date,
                    "trade_date": trade_date,
                    "selected_count": 0,
                    "selected_codes": [],
                    "forward_returns": {},
                    "benchmark_returns": {},
                    "metrics": {},
                }
                self._save_window_cache(template, snap_date, window_data)
                windows.append(window_data)
                continue

            ts_codes = selected["ts_code"].tolist()
            print(f"  选出 {len(ts_codes)} 只")

            # 计算前瞻收益
            forward_returns = self._compute_forward_returns(
                ts_codes, trade_date, forward_days)

            # 基准收益
            benchmark_returns = self._get_benchmark_returns(
                benchmark, trade_date, forward_days)

            # 窗口指标
            metrics = self._compute_window_metrics(
                forward_returns, benchmark_returns, forward_days)

            # 股票名称映射
            name_map = {}
            if "name" in selected.columns:
                name_map = dict(zip(selected["ts_code"], selected["name"]))

            window_data = {
                "snapshot_date": snap_date,
                "trade_date": trade_date,
                "market_regime": regime_info,
                "selected_count": len(ts_codes),
                "selected_codes": ts_codes,
                "selected_names": name_map,
                "forward_returns": forward_returns,
                "benchmark_returns": benchmark_returns,
                "metrics": metrics,
            }

            self._save_window_cache(template, snap_date, window_data)
            windows.append(window_data)

            # 打印窗口摘要
            m = metrics
            fwd_str = "  ".join(
                f"{d}d:{(m['avg_max_returns'].get(d) or 0):+.1f}%"
                for d in forward_days
            )
            close_str = "  ".join(
                f"{d}d:{(m['avg_close_returns'].get(d) or 0):+.1f}%"
                for d in forward_days
            )
            dd_str = "  ".join(
                f"{d}d:{(m['avg_drawdowns'].get(d) or 0):+.1f}%"
                for d in forward_days
            )
            hit_str = "  ".join(
                f"{d}d:{(m['max_hit_rates'].get(d) or 0):.0f}%"
                for d in forward_days
            )
            print(f"  最高: {fwd_str}")
            print(f"  收盘: {close_str}")
            print(f"  回撤: {dd_str}")
            print(f"  触及: {hit_str}")

        # 汇总
        aggregate = self._compute_aggregate(windows, forward_days)

        return {"windows": windows, "aggregate": aggregate}

    def _resolve_snapshot_date(self, date_str: str) -> Optional[str]:
        """将日期解析到最近交易日"""
        try:
            return self.client.get_recent_trade_day(base_date=date_str)
        except Exception:
            return None

    def _compute_forward_returns(
        self,
        ts_codes: list[str],
        snapshot_date: str,
        forward_days: list[int],
    ) -> dict:
        """计算选股后前瞻收益率。

        Returns:
            {ts_code: {5: {"close": X, "max": Y, "drawdown": Z}, ...}, ...}
            - close: 第N日收盘收益率
            - max: 期间最高价收益率（主指标）
            - drawdown: 期间最大回撤（最低价相对买入价，负数）
        """
        results = {}

        for ts_code in ts_codes:
            symbol = from_ts_code(ts_code)
            try:
                max_fwd = max(forward_days)
                fetch_days = int(max_fwd * 1.5) + 60

                start = (
                    datetime.strptime(snapshot_date, "%Y%m%d")
                    + timedelta(days=1)
                ).strftime("%Y%m%d")
                end = (
                    datetime.strptime(snapshot_date, "%Y%m%d")
                    + timedelta(days=fetch_days)
                ).strftime("%Y%m%d")

                df = self.client.get_kline(
                    symbol, freq="D", start=start, end=end, adj="qfq")
                if df is None or len(df) < 2:
                    continue

                entry_price = float(df.iloc[0]["open"])
                highs = df["high"].astype(float) if "high" in df.columns else df["close"].astype(float)
                lows = df["low"].astype(float) if "low" in df.columns else df["close"].astype(float)
                returns = {}

                for days in forward_days:
                    if len(df) > days:
                        exit_price = float(df.iloc[days]["close"])
                        close_ret = round(
                            (exit_price / entry_price - 1) * 100, 2)

                        period_high = highs.iloc[:days + 1].max()
                        max_ret = round(
                            (period_high / entry_price - 1) * 100, 2)

                        period_low = lows.iloc[:days + 1].min()
                        drawdown = round(
                            (period_low / entry_price - 1) * 100, 2)

                        returns[days] = {
                            "close": close_ret, "max": max_ret,
                            "drawdown": drawdown,
                        }
                    else:
                        returns[days] = None

                results[ts_code] = returns

            except Exception:
                continue

        return results

    def _get_benchmark_returns(
        self,
        benchmark: str,
        snapshot_date: str,
        forward_days: list[int],
    ) -> dict:
        """获取基准指数同期收益率"""
        max_fwd = max(forward_days)
        fetch_days = int(max_fwd * 1.8) + 30

        start = (
            datetime.strptime(snapshot_date, "%Y%m%d")
            + timedelta(days=1)
        ).strftime("%Y%m%d")
        end = (
            datetime.strptime(snapshot_date, "%Y%m%d")
            + timedelta(days=fetch_days)
        ).strftime("%Y%m%d")

        try:
            df = self.client.get_index_daily(
                ts_code=benchmark,
                start=start,
                end=end)
            if df is None or len(df) < 2:
                return {d: None for d in forward_days}

            df = df.sort_values("trade_date").reset_index(drop=True)
            entry_price = float(df.iloc[0]["close"])
            returns = {}

            for days in forward_days:
                if len(df) > days:
                    exit_price = float(df.iloc[days]["close"])
                    returns[days] = round(
                        (exit_price / entry_price - 1) * 100, 2)
                else:
                    returns[days] = None

            return returns

        except Exception:
            return {d: None for d in forward_days}

    def _compute_window_metrics(
        self,
        forward_returns: dict,
        benchmark_returns: dict,
        forward_days: list[int],
    ) -> dict:
        """计算单个窗口的回测指标"""
        metrics = {
            "avg_max_returns": {},
            "avg_close_returns": {},
            "avg_drawdowns": {},
            "median_max_returns": {},
            "median_close_returns": {},
            "worst_drawdowns": {},
            "hit_rates": {},
            "max_hit_rates": {},
            "best_max_returns": {},
            "worst_close_returns": {},
            "alphas": {},
            "drawdown_hit_rates": {},
            # 兼容旧格式
            "avg_returns": {},
        }

        for days in forward_days:
            close_vals = []
            max_vals = []
            dd_vals = []
            for ts_code, rets in forward_returns.items():
                v = rets.get(days)
                if v is not None:
                    close_vals.append(v["close"])
                    max_vals.append(v["max"])
                    dd_vals.append(v.get("drawdown", 0))

            if not close_vals:
                for k in metrics:
                    metrics[k][days] = None
                continue

            metrics["avg_max_returns"][days] = round(np.mean(max_vals), 2)
            metrics["avg_close_returns"][days] = round(np.mean(close_vals), 2)
            metrics["avg_drawdowns"][days] = round(np.mean(dd_vals), 2)
            metrics["median_max_returns"][days] = round(np.median(max_vals), 2)
            metrics["median_close_returns"][days] = round(np.median(close_vals), 2)
            metrics["worst_drawdowns"][days] = round(min(dd_vals), 2)
            metrics["hit_rates"][days] = round(
                sum(1 for v in close_vals if v > 0) / len(close_vals) * 100, 1)
            metrics["max_hit_rates"][days] = round(
                sum(1 for v in max_vals if v > 0) / len(max_vals) * 100, 1)
            metrics["drawdown_hit_rates"][days] = round(
                sum(1 for v in dd_vals if v > -5) / len(dd_vals) * 100, 1)
            metrics["best_max_returns"][days] = round(max(max_vals), 2)
            metrics["worst_close_returns"][days] = round(min(close_vals), 2)
            # 兼容
            metrics["avg_returns"][days] = metrics["avg_close_returns"][days]

            bench = benchmark_returns.get(days)
            if bench is not None:
                metrics["alphas"][days] = round(
                    metrics["avg_close_returns"][days] - bench, 2)
            else:
                metrics["alphas"][days] = None

        return metrics

    def _compute_aggregate(
        self, windows: list[dict], forward_days: list[int]
    ) -> dict:
        """跨窗口汇总统计"""
        aggregate = {
            "total_windows": len(windows),
            "valid_windows": 0,
            "avg_max_returns": {},
            "avg_close_returns": {},
            "avg_drawdowns": {},
            "worst_drawdowns": {},
            "avg_hit_rates": {},
            "avg_max_hit_rates": {},
            "avg_alphas": {},
            "consistency": {},
            "close_consistency": {},
        }

        valid = [w for w in windows if w["selected_count"] > 0
                 and w.get("metrics")]
        aggregate["valid_windows"] = len(valid)

        if not valid:
            return aggregate

        for days in forward_days:
            max_rets = [w["metrics"]["avg_max_returns"].get(days)
                        for w in valid
                        if w["metrics"]["avg_max_returns"].get(days) is not None]
            close_rets = [w["metrics"]["avg_close_returns"].get(days)
                          for w in valid
                          if w["metrics"]["avg_close_returns"].get(days) is not None]
            dd_vals = [w["metrics"]["avg_drawdowns"].get(days)
                       for w in valid
                       if w["metrics"]["avg_drawdowns"].get(days) is not None]
            worst_dd = [w["metrics"]["worst_drawdowns"].get(days)
                        for w in valid
                        if w["metrics"]["worst_drawdowns"].get(days) is not None]
            hits = [w["metrics"]["hit_rates"].get(days)
                    for w in valid
                    if w["metrics"]["hit_rates"].get(days) is not None]
            max_hits = [w["metrics"]["max_hit_rates"].get(days)
                        for w in valid
                        if w["metrics"]["max_hit_rates"].get(days) is not None]
            alphas = [w["metrics"]["alphas"].get(days)
                      for w in valid
                      if w["metrics"]["alphas"].get(days) is not None]

            if max_rets:
                aggregate["avg_max_returns"][days] = round(np.mean(max_rets), 2)
                aggregate["consistency"][days] = round(
                    sum(1 for r in max_rets if r > 0) / len(max_rets) * 100, 1)
            if close_rets:
                aggregate["avg_close_returns"][days] = round(np.mean(close_rets), 2)
                aggregate["close_consistency"][days] = round(
                    sum(1 for r in close_rets if r > 0) / len(close_rets) * 100, 1)
            if dd_vals:
                aggregate["avg_drawdowns"][days] = round(np.mean(dd_vals), 2)
            if worst_dd:
                aggregate["worst_drawdowns"][days] = round(min(worst_dd), 2)
            if hits:
                aggregate["avg_hit_rates"][days] = round(np.mean(hits), 1)
            if max_hits:
                aggregate["avg_max_hit_rates"][days] = round(np.mean(max_hits), 1)
            if alphas:
                aggregate["avg_alphas"][days] = round(np.mean(alphas), 2)

        return aggregate

    # ── 增量缓存 ──

    def _window_cache_path(self, template: str, snap_date: str) -> Path:
        return self.cache_dir / f"{template}_{snap_date}_v{self.CACHE_VERSION}.json"

    def _load_window_cache(
        self, template: str, snap_date: str
    ) -> Optional[dict]:
        path = self._window_cache_path(template, snap_date)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # 旧缓存格式迁移: forward_returns值是数字而非dict
                # 返回None强制重新计算
                fwd_rets = data.get("forward_returns", {})
                if fwd_rets:
                    sample = list(fwd_rets.values())
                    if sample:
                        first = list(sample[0].values())[0] if sample[0] else None
                        # 旧格式: 值是数字 或 只有close/max没有drawdown
                        if isinstance(first, (int, float)):
                            return None
                        if isinstance(first, dict) and "drawdown" not in first:
                            return None
                return data
            except Exception:
                return None
        return None

    def _save_window_cache(
        self, template: str, snap_date: str, data: dict
    ) -> None:
        path = self._window_cache_path(template, snap_date)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def clear_cache(self, template: Optional[str] = None) -> int:
        """清除缓存，返回删除数量"""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            if template is None or f.name.startswith(template + "_"):
                f.unlink()
                count += 1
        return count

    # ── 报告 ──

    @staticmethod
    def format_report(results: dict, forward_days: list[int]) -> str:
        """格式化回测报告"""
        lines = []
        windows = results.get("windows", [])
        agg = results.get("aggregate", {})
        total = agg.get("total_windows", 0)

        fwd_labels = [f"{d}T" for d in forward_days]  # T = trading days

        lines.append("=" * 70)
        lines.append("  选股器回测报告")
        lines.append(f"  回测窗口: {total} | 前瞻: {'/'.join(fwd_labels)} (交易日)")
        lines.append("=" * 70)

        for i, w in enumerate(windows, 1):
            sd = w["snapshot_date"]
            td = w["trade_date"]
            n = w["selected_count"]
            m = w.get("metrics", {})

            if not m or not m.get("avg_returns"):
                lines.append(f"\n  窗口{i:2d}  {sd}  选出{n}只  (无数据)")
                continue

            # 大盘环境标注
            regime = w.get("market_regime")
            regime_str = ""
            if regime:
                regime_tag = {
                    "risk_on": "多头", "neutral": "震荡", "risk_off": "空头",
                }.get(regime.get("regime", ""), "")
                regime_str = f"  环境:{regime_tag}{regime.get('score', 0):+.2f}"

            fwd_str = "  ".join(
                f"{d}T:{(m['avg_max_returns'].get(d) or 0):+.1f}%"
                for d in forward_days
            )
            close_str = "  ".join(
                f"{d}T:{(m['avg_close_returns'].get(d) or 0):+.1f}%"
                for d in forward_days
            )
            dd_str = "  ".join(
                f"{d}T:{(m['avg_drawdowns'].get(d) or 0):+.1f}%"
                for d in forward_days
            )
            hit_str = "  ".join(
                f"{d}T:{(m['max_hit_rates'].get(d) or 0):.0f}%"
                for d in forward_days
            )
            lines.append(
                f"\n  窗口{i:2d}  {sd}(交易日{td})  选出{n:2d}只{regime_str}")
            lines.append(f"    最高收益: {fwd_str}")
            lines.append(f"    收盘收益: {close_str}")
            lines.append(f"    最大回撤: {dd_str}")
            lines.append(f"    触及率:   {hit_str}")

            # 个股明细表（单窗口时显示）
            if total == 1 and w.get("forward_returns"):
                names = w.get("selected_names", {})
                fwd_rets = w["forward_returns"]
                codes_ordered = w.get("selected_codes", list(fwd_rets.keys()))

                lines.append("")
                lines.append("  ── 个股明细（▲期间最高 / ▽收盘价 / ↘回撤）──")

                for rank, ts_code in enumerate(codes_ordered, 1):
                    name = names.get(ts_code, from_ts_code(ts_code))[:6]
                    rets = fwd_rets.get(ts_code, {})

                    # 最高收益行
                    row = f"  {rank:4d}  {from_ts_code(ts_code):8}  {name:<8}"
                    for d in forward_days:
                        v = rets.get(d)
                        if v is None:
                            row += f"  {'N/A':>8}"
                        else:
                            sign = "+" if v["max"] >= 0 else ""
                            row += f"  ▲{sign}{v['max']:.1f}%"
                    lines.append(row)

                    # 收盘 + 回撤行
                    row = f"  {'':>4}  {'':8}  {'':8}"
                    for d in forward_days:
                        v = rets.get(d)
                        if v is None:
                            row += f"  {'':>8}"
                        else:
                            dd = v.get("drawdown", 0)
                            row += f"  ▽{v['close']:+.1f}%↘{dd:+.1f}%"
                    lines.append(row)

                lines.append("")

                # 窗口汇总行
                lines.append("  " + "─" * 55)
                summary_rows = [
                    ("平均最高", m.get("avg_max_returns", {}), False),
                    ("平均收盘", m.get("avg_close_returns", {}), False),
                    ("平均回撤", m.get("avg_drawdowns", {}), False),
                    ("最大回撤", m.get("worst_drawdowns", {}), False),
                    ("中位最高", m.get("median_max_returns", {}), False),
                    ("命中率(收盘)", m.get("hit_rates", {}), True),
                    ("触及率(最高)", m.get("max_hit_rates", {}), True),
                ]
                for label, vals, is_pct in summary_rows:
                    row = f"  {'':>4}  {'':8}  {label:<10}"
                    for d in forward_days:
                        v = vals.get(d)
                        if v is None:
                            row += f"  {'N/A':>8}"
                        elif is_pct:
                            row += f"  {v:>6.0f}% "
                        else:
                            sign = "+" if v >= 0 else ""
                            row += f"  {sign}{v:>6.1f}%"
                    lines.append(row)

                # 超额收益
                alphas = m.get("alphas", {})
                if any(v is not None for v in alphas.values()):
                    row = f"  {'':>4}  {'':8}  {'超额α(收盘)':12}"
                    for d in forward_days:
                        v = alphas.get(d)
                        if v is None:
                            row += f"  {'N/A':>8}"
                        else:
                            sign = "+" if v >= 0 else ""
                            row += f"  {sign}{v:>6.1f}%"
                    lines.append(row)

                lines.append(f"  {'':>4}  {'':8}  {'基准':8}  上证指数")

        # 多窗口汇总
        if total > 1 and agg.get("valid_windows", 0) > 0:
            lines.append(f"\n  {'─'*60}")
            lines.append("  汇总")

            avg_max = agg.get("avg_max_returns", {})
            avg_close = agg.get("avg_close_returns", {})
            avg_dd = agg.get("avg_drawdowns", {})
            worst_dd = agg.get("worst_drawdowns", {})
            avg_hit = agg.get("avg_hit_rates", {})
            avg_max_hit = agg.get("avg_max_hit_rates", {})
            avg_alpha = agg.get("avg_alphas", {})
            consistency = agg.get("consistency", {})
            close_consistency = agg.get("close_consistency", {})

            def _fmt_line(label, data, is_pct=False):
                if not data:
                    return None
                if is_pct:
                    return f"  {label:10s}" + "  ".join(
                        f"{d}T:{data.get(d, 0):.0f}%"
                        for d in forward_days if d in data)
                return f"  {label:10s}" + "  ".join(
                    f"{d}T:{data.get(d, 0):+.1f}%"
                    for d in forward_days if d in data)

            line = _fmt_line("平均最高:", avg_max)
            if line:
                lines.append(line)
            line = _fmt_line("平均收盘:", avg_close)
            if line:
                lines.append(line)
            line = _fmt_line("平均回撤:", avg_dd)
            if line:
                lines.append(line)
            line = _fmt_line("最差回撤:", worst_dd)
            if line:
                lines.append(line)
            line = _fmt_line("触及率:", avg_max_hit, is_pct=True)
            if line:
                lines.append(line)
            line = _fmt_line("收盘命中:", avg_hit, is_pct=True)
            if line:
                lines.append(line)
            line = _fmt_line("超额α:", avg_alpha)
            if line:
                lines.append(line)

            if consistency:
                vw = agg.get("valid_windows", 0)
                cons_str = "  ".join(
                    f"{d}T:{consistency.get(d, 0):.0f}%"
                    for d in forward_days if d in consistency)
                lines.append(f"  一致性:    {cons_str}  ({vw}个有效窗口最高收益为正)")
            if close_consistency:
                cons_str = "  ".join(
                    f"{d}T:{close_consistency.get(d, 0):.0f}%"
                    for d in forward_days if d in close_consistency)
                lines.append(f"  收盘一致:  {cons_str}")

        lines.append(f"\n{'='*70}")
        return "\n".join(lines)

    @staticmethod
    def generate_snapshot_dates(
        from_date: str,
        to_date: Optional[str] = None,
        step: str = "monthly",
        count: Optional[int] = None,
    ) -> list[str]:
        """生成截断日期列表。

        Args:
            from_date: 起始日期YYYYMMDD
            to_date: 结束日期，默认今天
            step: monthly/biweekly/weekly
            count: 生成N个（优先于to_date）

        Returns:
            ["20240601", "20240701", ...]
        """
        start = datetime.strptime(from_date, "%Y%m%d")
        end = datetime.strptime(to_date, "%Y%m%d") if to_date else datetime.now()

        if step == "monthly":
            delta = timedelta(days=30)
        elif step == "biweekly":
            delta = timedelta(days=14)
        elif step == "weekly":
            delta = timedelta(days=7)
        else:
            delta = timedelta(days=30)

        dates = []
        current = start
        while current <= end:
            if count is not None and len(dates) >= count:
                break
            dates.append(current.strftime("%Y%m%d"))
            current += delta

        return dates
