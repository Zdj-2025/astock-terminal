#!/usr/bin/env python3
"""卖出信号评分器回测 — 验证SellSignalScorer的预测能力。

核心问题：
  - 评分≥70（强烈卖出）后，股票是否真的后续大跌？
  - 评分<30（可能是洗盘）后，是否真的恢复了？

设计：
  1. 预定义测试案例（真跌+洗盘各≥10个）
  2. 截断数据到指定日期之前，计算卖出评分（无前瞻偏差）
  3. 获取后续N日实际走势，对比评分预测 vs 实际结果
  4. 与两条简单基线规则对比

用法：
  cd .
  PYTHONPATH=. python scripts/research/backtest_sell_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 延迟导入 SellSignalScorer（可能还没写好）──────────────────────────
_ScorerAvailable = False
try:
    from scripts.analyzers.sell_signal_scorer import SellSignalScorer, SellSignalResult
    _ScorerAvailable = True
except ImportError:
    SellSignalScorer = None  # type: ignore
    SellSignalResult = None  # type: ignore

from scripts.core.tushare_client import TushareClient

# ── 常量 ──────────────────────────────────────────────────────────────
FORWARD_DAYS = 10          # 前瞻天数
DROP_THRESHOLD = -5.0      # 真跌定义：后续N日跌幅超过此阈值(%)
RECOVER_THRESHOLD = 0.0    # 恢复定义：后续N日收益>=此阈值(%)
LOOKBACK_DAYS = 250        # 获取多少历史日线用于评分
INDEX_CODE = "000001.SH"   # 上证指数

# ── 测试案例 ──────────────────────────────────────────────────────────
# 每个案例: (code, date, category, description)
#   category: "real_drop"（真跌）或 "washout"（洗盘）

TEST_CASES: list[tuple[str, str, str, str]] = [
    # ═══ 真跌案例（期望评分≥50，后续实际下跌≥5%） ═══
    # --- 2025年9月大跌期间 ---
    ("601012", "2025-09-08", "real_drop", "隆基绿能-高位放量破位下跌"),
    # 300750 宁德时代已移除(后续+19.5%，非真跌)
    ("600519", "2025-09-10", "real_drop", "贵州茅台-趋势走坏大跌前夕"),
    # 002475 立讯精密已移除(后续+45.6%，非真跌)
    ("603259", "2025-09-11", "real_drop", "药明康德-持续下跌途中"),

    # --- 2026年3月大跌期间 ---
    ("688981", "2026-03-05", "real_drop", "中芯国际-高位回落加速"),
    ("002371", "2026-03-06", "real_drop", "北方华创-高位放量长阴"),
    ("300059", "2026-03-09", "real_drop", "东方财富-券商龙头破位"),
    ("601127", "2026-03-10", "real_drop", "赛力斯-高位连续下跌"),
    ("002049", "2026-03-10", "real_drop", "紫光国微-半导体高位补跌"),

    # --- 其他高位放量破位案例 ---
    ("300760", "2025-12-15", "real_drop", "迈瑞医疗-高位放量阴线"),
    # 600036 招商银行已移除(后续+0.5%，未实际下跌)
    ("000858", "2026-01-12", "real_drop", "五粮液-白酒高位补跌"),

    # --- 原标为洗盘，后验确认实际大跌 → 改为真跌 ---
    ("300274", "2026-01-19", "real_drop", "阳光电源-后续-13%实际大跌"),
    ("300124", "2025-10-20", "real_drop", "汇川技术-后续-7%实际下跌"),
    ("688111", "2025-08-04", "real_drop", "金山办公-后续-5.7%实际下跌"),

    # ═══ 洗盘案例（期望评分<30，后续实际恢复上涨） ═══
    # --- 上升趋势中缩量回调后恢复 ---
    ("603881", "2025-07-28", "washout", "数据港-缩量回调后继续上涨"),
    # 688111 金山办公已改为real_drop(后续-5.7%)
    ("002230", "2025-10-13", "washout", "科大讯飞-上涨中继缩量整理"),
    # 300124 汇川技术已改为real_drop(后续-7%)
    ("600588", "2025-11-03", "washout", "用友网络-低位放量启动后缩量回调"),

    # --- 假破MA20后快速收回 ---
    ("688256", "2026-02-09", "washout", "寒武纪-短暂跌破MA20后快速收回"),
    ("002415", "2026-02-16", "washout", "海康威视-下影线回踩MA20后反弹"),
    # 300274 阳光电源已改为real_drop(后续-13%)
    ("601899", "2025-12-08", "washout", "紫金矿业-缩量洗盘后恢复上升"),
    ("002594", "2026-02-23", "washout", "比亚迪-上升趋势中正常回调"),
]


# ── 数据结构 ──────────────────────────────────────────────────────────
@dataclass
class CaseResult:
    """单个案例的回测结果。"""
    code: str
    name: str
    date: str
    category: str          # "real_drop" / "washout"
    description: str

    # 评分器结果
    score: Optional[float] = None
    action: Optional[str] = None
    confidence: Optional[float] = None

    # 基线1: f'<0 + f''<0
    baseline1_sell: Optional[bool] = None

    # 基线2: 收盘破MA20
    baseline2_sell: Optional[bool] = None

    # 实际后续走势
    forward_return: Optional[float] = None   # 后续N日收益率(%)
    max_drawdown: Optional[float] = None      # 后续N日最大回撤(%)
    actual_dropped: Optional[bool] = None     # 是否真跌
    actual_recovered: Optional[bool] = None   # 是否恢复

    # 判定
    scorer_correct: Optional[bool] = None     # 评分器是否判断正确
    baseline1_correct: Optional[bool] = None
    baseline2_correct: Optional[bool] = None

    error: Optional[str] = None


@dataclass
class BacktestSummary:
    """回测汇总统计。"""
    total_cases: int = 0
    valid_cases: int = 0

    # 真跌案例
    real_drop_total: int = 0
    real_drop_hit: int = 0       # 评分≥50且确实跌
    real_drop_miss: int = 0      # 评分<50但真跌了

    # 洗盘案例
    washout_total: int = 0
    washout_hit: int = 0         # 评分<30且确实恢复
    washout_miss: int = 0        # 评分≥30但没恢复

    # 总体
    overall_accuracy: float = 0.0

    # 假阳性/假阴性
    false_positive_rate: float = 0.0   # 评分≥70但没跌
    false_negative_rate: float = 0.0   # 评分<30但真跌

    # 提前天数统计
    advance_days_list: list[float] = field(default_factory=list)

    # 基线对比
    scorer_accuracy: float = 0.0
    baseline1_accuracy: float = 0.0
    baseline2_accuracy: float = 0.0


class SellSignalBacktester:
    """卖出信号评分器回测引擎。"""

    def __init__(self):
        self.ts = TushareClient()
        self.scorer = None
        if _ScorerAvailable:
            try:
                self.scorer = SellSignalScorer()
            except Exception as e:
                print(f"⚠ SellSignalScorer 初始化失败: {e}")
                self.scorer = None

        # 缓存大盘日线
        self._index_cache: dict[str, pd.DataFrame] = {}

    # ── 数据获取 ───────────────────────────────────────────────────

    def _get_daily(self, code: str, end_date: str, lookback: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
        """获取截断到end_date之前的日线数据（无前瞻偏差）。"""
        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.Timedelta(days=lookback * 2)  # 多取一些确保够
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        try:
            df = self.ts.get_kline(code, freq="D", start=start_str, end=end_str, adj="hfq")
        except Exception as e:
            print(f"  ⚠ 获取 {code} 日线失败: {e}")
            return None

        if df is None or df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 严格截断：只用 end_date 当天及之前的数据
        df = df[df["date"] <= end_dt].copy()

        # 转数值
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _get_index_daily(self, end_date: str) -> Optional[pd.DataFrame]:
        """获取上证指数日线（截断）。

        注意：上证指数不能用 get_kline()，必须用 pro.index_daily()。
        """
        if end_date in self._index_cache:
            return self._index_cache[end_date]

        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.Timedelta(days=LOOKBACK_DAYS * 2)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        try:
            raw = self.ts.pro.index_daily(
                ts_code="000001.SH", start_date=start_str, end_date=end_str
            )
        except Exception:
            raw = None

        if raw is None or raw.empty:
            self._index_cache[end_date] = None
            return None

        # 转成统一格式（与 get_kline 一致的列名）
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["trade_date"]),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("vol", 0), errors="coerce"),
        })
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] <= end_dt].copy()

        self._index_cache[end_date] = df
        return df

    def _get_forward_prices(self, code: str, date: str, n_days: int = FORWARD_DAYS) -> Optional[dict]:
        """获取date之后N日的实际走势。"""
        start_dt = pd.Timestamp(date)
        end_dt = start_dt + pd.Timedelta(days=n_days * 2 + 10)  # 多取覆盖节假日
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        try:
            df = self.ts.get_kline(code, freq="D", start=start_str, end=end_str, adj="hfq")
        except Exception:
            return None

        if df is None or df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 只取 date 之后的交易日
        future = df[df["date"] > start_dt].head(n_days + 1)  # 多取一天算max drawdown
        if len(future) < 2:
            return None

        # 入场价：date次日开盘价
        entry = float(future.iloc[0]["open"])
        # 退出价：第N日收盘价
        exit_price = float(future.iloc[-1]["close"])

        if entry <= 0:
            return None

        fwd_return = (exit_price / entry - 1) * 100

        # 最大回撤（从入场到期间最低点）
        lows = future["low"].astype(float).values
        max_dd = (np.min(lows) / entry - 1) * 100

        return {
            "forward_return": round(fwd_return, 2),
            "max_drawdown": round(max_dd, 2),
            "actual_dropped": fwd_return <= DROP_THRESHOLD,
            "actual_recovered": fwd_return >= RECOVER_THRESHOLD,
            "n_trading_days": len(future),
        }

    # ── 基线规则 ───────────────────────────────────────────────────

    def _compute_baselines(self, daily_df: pd.DataFrame) -> dict:
        """计算两条基线规则。

        基线1: f'<0 + f''<0 → 卖出信号
        基线2: 收盘破MA20 → 卖出信号
        """
        result = {"baseline1_sell": None, "baseline2_sell": None}

        if daily_df is None or len(daily_df) < 30:
            return result

        close = daily_df["close"].astype(float).values
        n = len(close)

        # 基线1: Savitzky-Golay 一阶导和二阶导
        # ⚠️ 必须做 shift 防止未来数据暴露（与评分器一致）
        try:
            from scipy.signal import savgol_filter
            ln_price = np.log(close[close > 0])
            if len(ln_price) >= 21:
                window = 21
                f1_raw = savgol_filter(ln_price, window_length=window, polyorder=3, deriv=1)
                f2_raw = savgol_filter(ln_price, window_length=window, polyorder=3, deriv=2)
                # shift 防止未来数据泄漏
                shift = window // 2  # 10
                f1 = np.empty_like(f1_raw)
                f2 = np.empty_like(f2_raw)
                f1[:shift] = f1_raw[:shift]
                f2[:shift] = f2_raw[:shift]
                f1[shift:] = f1_raw[:-shift]
                f2[shift:] = f2_raw[:-shift]
                result["baseline1_sell"] = (f1[-1] < 0) and (f2[-1] < 0)
        except Exception:
            pass

        # 基线2: 收盘破MA20
        try:
            if n >= 20:
                ma20 = np.mean(close[-20:])
                result["baseline2_sell"] = close[-1] < ma20
        except Exception:
            pass

        return result

    # ── 评分器调用 ─────────────────────────────────────────────────

    def _compute_score(self, code: str, daily_df: pd.DataFrame, index_df: Optional[pd.DataFrame],
                       minute_df: Optional[pd.DataFrame] = None) -> Optional[dict]:
        """调用 SellSignalScorer 计算评分。"""
        if self.scorer is None:
            return None

        try:
            res = self.scorer.score(code=code, daily_df=daily_df, index_daily=index_df,
                                     minute_df=minute_df)
            return {
                "score": getattr(res, "score", None),
                "action": getattr(res, "action", None),
                "confidence": getattr(res, "confidence", None),
                "details": getattr(res, "details", {}),
            }
        except Exception as e:
            print(f"  ⚠ 评分器异常: {e}")
            return None

    # ── 获取股票名称 ───────────────────────────────────────────────

    def _get_stock_name(self, code: str) -> str:
        """尝试获取股票名称，失败返回代码。"""
        try:
            ts_code = code if "." in code else f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"
            info = self.ts.pro.stock_basic(ts_code=ts_code, fields="ts_code,name")
            if info is not None and len(info) > 0:
                return info.iloc[0]["name"]
        except Exception:
            pass
        return code

    # ── 60分钟线数据获取 ─────────────────────────────────────────────

    def _get_minute60(self, code: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取60分钟线数据（截断到end_date），多源自动降级。

        优先级: 本地缓存 → Baostock(无限流) → AKShare → Tushare。
        如果全部失败，返回None（评分器会给中间分）。
        """
        end_dt = pd.Timestamp(end_date)
        # 60min数据取最近约40个交易日（约60天自然日）
        start_dt = end_dt - pd.Timedelta(days=90)
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")

        # 纯6位代码
        symbol = code.replace(".SH", "").replace(".SZ", "")

        try:
            df = self.ts.get_kline_minute(
                symbol, freq="60", start=start_str, end=end_str,
                adj="qfq", source="auto",
            )
        except Exception as e:
            print(f"    ⚠ 获取{code} 60min数据失败: {e}")
            return None

        if df is None or df.empty:
            return None

        # 统一列名（可能来自不同数据源）
        if "day" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"day": "date"})
        if "datetime" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"datetime": "date"})

        if "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 严格截断到end_date当天
        df = df[df["date"] <= end_dt + pd.Timedelta(hours=23, minutes=59)].copy()

        # 转数值
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if len(df) < 35:
            print(f"    ⚠ 60min数据仅{len(df)}根K线，不足35根")
            return None

        print(f"    ✅ 60min数据: {len(df)}根K线 ({df['date'].min().date()} ~ {df['date'].max().date()})")
        return df

    # ── 单案例回测 ─────────────────────────────────────────────────

    def run_case(self, code: str, date: str, category: str, description: str = "") -> CaseResult:
        """在指定日期对指定股票计算卖出评分，对比实际走势。"""
        result = CaseResult(
            code=code,
            name=self._get_stock_name(code),
            date=date,
            category=category,
            description=description,
        )

        print(f"  📊 {code} ({result.name}) @ {date} [{category}] — {description}")

        # 1. 获取截断日线
        daily_df = self._get_daily(code, date)
        if daily_df is None or len(daily_df) < 30:
            result.error = f"数据不足: {len(daily_df) if daily_df is not None else 0}条"
            print(f"    ❌ {result.error}")
            return result

        # 2. 获取大盘数据
        index_df = self._get_index_daily(date)

        # 2.5 获取60min数据（AKShare，失败则跳过）
        minute_df = self._get_minute60(code, date)
        if minute_df is not None:
            print(f"    📊 60min数据: {len(minute_df)}根K线")

        # 3. 计算评分器评分
        score_result = self._compute_score(code, daily_df, index_df, minute_df)
        if score_result is not None:
            result.score = score_result["score"]
            result.action = score_result["action"]
            result.confidence = score_result["confidence"]
            print(f"    评分: {result.score} | 操作: {result.action} | 置信度: {result.confidence}")
        else:
            print(f"    ⚠ 评分器不可用，跳过评分")

        # 4. 计算基线
        baselines = self._compute_baselines(daily_df)
        result.baseline1_sell = baselines["baseline1_sell"]
        result.baseline2_sell = baselines["baseline2_sell"]
        print(f"    基线1(f'<0+f''<0): {'卖出' if result.baseline1_sell else '不卖'}")
        print(f"    基线2(破MA20):    {'卖出' if result.baseline2_sell else '不卖'}")

        # 5. 获取后续实际走势
        fwd = self._get_forward_prices(code, date, FORWARD_DAYS)
        if fwd is None:
            result.error = "无法获取后续走势数据"
            print(f"    ❌ {result.error}")
            return result

        result.forward_return = fwd["forward_return"]
        result.max_drawdown = fwd["max_drawdown"]
        result.actual_dropped = fwd["actual_dropped"]
        result.actual_recovered = fwd["actual_recovered"]
        print(f"    后续{fwd['n_trading_days']}日: 收益={result.forward_return:+.2f}% "
              f"最大回撤={result.max_drawdown:+.2f}% "
              f"{'📉真跌' if result.actual_dropped else '📈恢复' if result.actual_recovered else '➡️横盘'}")

        # 6. 判定评分器是否正确
        if result.score is not None:
            if category == "real_drop":
                # 真跌案例：评分≥30算正确预警（v3五维度阈值降低）
                result.scorer_correct = result.score >= 30
            else:
                # 洗盘案例：评分<30算正确识别（没误判为卖出）
                result.scorer_correct = result.score < 30

        # 7. 判定基线是否正确
        if category == "real_drop":
            if result.baseline1_sell is not None:
                result.baseline1_correct = result.baseline1_sell  # 基线说卖 = 正确
            if result.baseline2_sell is not None:
                result.baseline2_correct = result.baseline2_sell
        else:
            if result.baseline1_sell is not None:
                result.baseline1_correct = not result.baseline1_sell  # 基线说不卖 = 正确
            if result.baseline2_sell is not None:
                result.baseline2_correct = not result.baseline2_sell

        return result

    # ── 全量回测 ───────────────────────────────────────────────────

    def run_all(self, cases: Optional[list[tuple[str, str, str, str]]] = None) -> tuple[list[CaseResult], BacktestSummary]:
        """跑所有案例，输出汇总。"""
        if cases is None:
            cases = TEST_CASES

        print("=" * 80)
        print("卖出信号评分器回测")
        print(f"案例数: {len(cases)} | 前瞻天数: {FORWARD_DAYS} | "
              f"真跌阈值: {DROP_THRESHOLD}% | 恢复阈值: {RECOVER_THRESHOLD}%")
        print(f"评分器状态: {'✅ 可用' if self.scorer is not None else '❌ 不可用（仅跑基线）'}")
        print("=" * 80)

        results: list[CaseResult] = []
        for i, (code, date, cat, desc) in enumerate(cases):
            print(f"\n[{i+1}/{len(cases)}]")
            cr = self.run_case(code, date, cat, desc)
            results.append(cr)

        # 汇总统计
        summary = self._compute_summary(results)
        self._print_summary(results, summary)

        return results, summary

    # ── 统计汇总 ───────────────────────────────────────────────────

    def _compute_summary(self, results: list[CaseResult]) -> BacktestSummary:
        """计算汇总统计。"""
        s = BacktestSummary()
        s.total_cases = len(results)

        valid = [r for r in results if r.error is None]
        s.valid_cases = len(valid)

        if not valid:
            return s

        # 真跌案例
        real_drops = [r for r in valid if r.category == "real_drop"]
        s.real_drop_total = len(real_drops)

        for r in real_drops:
            if r.score is not None and r.score >= 30 and r.actual_dropped:
                s.real_drop_hit += 1
            elif r.score is not None and r.score < 30 and r.actual_dropped:
                s.real_drop_miss += 1

        # 洗盘案例
        washouts = [r for r in valid if r.category == "washout"]
        s.washout_total = len(washouts)

        for r in washouts:
            if r.score is not None and r.score < 30 and r.actual_recovered:
                s.washout_hit += 1
            elif r.score is not None and r.score >= 30 and not r.actual_recovered:
                s.washout_miss += 1

        # 总体准确率（评分器判断正确的比例）
        scored = [r for r in valid if r.scorer_correct is not None]
        if scored:
            correct_count = sum(1 for r in scored if r.scorer_correct)
            s.overall_accuracy = correct_count / len(scored)
            s.scorer_accuracy = s.overall_accuracy

        # 假阳性率：评分≥50但后续没跌
        high_score = [r for r in valid if r.score is not None and r.score >= 50]
        if high_score:
            false_pos = sum(1 for r in high_score if not r.actual_dropped)
            s.false_positive_rate = false_pos / len(high_score)

        # 假阴性率：评分<30但后续真跌
        low_score = [r for r in valid if r.score is not None and r.score < 30]
        if low_score:
            false_neg = sum(1 for r in low_score if r.actual_dropped)
            s.false_negative_rate = false_neg / len(low_score)

        # 提前天数（暂时留空，需要逐日扫描才能算）
        # TODO: 如果要算提前天数，需要在date之前逐日扫描首次score>=50的日期

        # 基线准确率
        b1_valid = [r for r in valid if r.baseline1_correct is not None]
        if b1_valid:
            s.baseline1_accuracy = sum(1 for r in b1_valid if r.baseline1_correct) / len(b1_valid)

        b2_valid = [r for r in valid if r.baseline2_correct is not None]
        if b2_valid:
            s.baseline2_accuracy = sum(1 for r in b2_valid if r.baseline2_correct) / len(b2_valid)

        return s

    def _print_summary(self, results: list[CaseResult], s: BacktestSummary):
        """打印汇总报告。"""
        print("\n")
        print("=" * 80)
        print("📊 回测汇总报告")
        print("=" * 80)

        # 逐案例表格
        print("\n── 逐案例结果 ──")
        header = f"{'代码':>8} {'名称':>6} {'日期':>12} {'类别':>8} {'评分':>6} {'基线1':>6} {'基线2':>6} {'后续收益':>8} {'最大回撤':>8} {'实际':>6} {'评分器':>6}"
        print(header)
        print("-" * len(header))

        for r in results:
            cat_label = "真跌" if r.category == "real_drop" else "洗盘"
            score_str = f"{r.score:.0f}" if r.score is not None else "N/A"
            b1_str = "卖出" if r.baseline1_sell else "持有" if r.baseline1_sell is not None else "N/A"
            b2_str = "卖出" if r.baseline2_sell else "持有" if r.baseline2_sell is not None else "N/A"
            fwd_str = f"{r.forward_return:+.2f}%" if r.forward_return is not None else "N/A"
            dd_str = f"{r.max_drawdown:+.2f}%" if r.max_drawdown is not None else "N/A"

            if r.actual_dropped:
                actual_str = "📉跌"
            elif r.actual_recovered:
                actual_str = "📈涨"
            else:
                actual_str = "➡️平"

            if r.error:
                actual_str = "❌"

            if r.scorer_correct is None:
                sc_str = "N/A"
            elif r.scorer_correct:
                sc_str = "✅"
            else:
                sc_str = "❌"

            name_short = r.name[:6] if len(r.name) > 6 else r.name
            print(f"{r.code:>8} {name_short:>6} {r.date:>12} {cat_label:>8} {score_str:>6} "
                  f"{b1_str:>6} {b2_str:>6} {fwd_str:>8} {dd_str:>8} {actual_str:>6} {sc_str:>6}")

        # 汇总统计
        print(f"\n── 汇总统计 (有效案例: {s.valid_cases}/{s.total_cases}) ──")

        if s.real_drop_total > 0:
            hit_rate = s.real_drop_hit / s.real_drop_total * 100 if s.real_drop_total else 0
            print(f"\n  真跌案例 (n={s.real_drop_total}):")
            print(f"    命中率 (评分≥30且确实跌): {s.real_drop_hit}/{s.real_drop_total} = {hit_rate:.1f}%")
            print(f"    漏判   (评分<30但真跌了): {s.real_drop_miss}")

        if s.washout_total > 0:
            hit_rate = s.washout_hit / s.washout_total * 100 if s.washout_total else 0
            print(f"\n  洗盘案例 (n={s.washout_total}):")
            print(f"    命中率 (评分<30且确实恢复): {s.washout_hit}/{s.washout_total} = {hit_rate:.1f}%")
            print(f"    误判   (评分≥30但没恢复):   {s.washout_miss}")

        print(f"\n  总体指标:")
        print(f"    总体准确率:       {s.overall_accuracy:.1%}")
        print(f"    假阳性率(≥50没跌): {s.false_positive_rate:.1%}")
        print(f"    假阴性率(<30真跌): {s.false_negative_rate:.1%}")

        # 基线对比
        print(f"\n── 基线对比 ──")
        print(f"  {'方法':<20} {'准确率':>10}")
        print(f"  {'─'*20} {'─'*10}")
        print(f"  {'SellSignalScorer':<20} {s.scorer_accuracy:>10.1%}")
        print(f"  {'基线1: f′<0+f″<0':<20} {s.baseline1_accuracy:>10.1%}")
        print(f"  {'基线2: 破MA20':<20} {s.baseline2_accuracy:>10.1%}")

        if s.scorer_accuracy > max(s.baseline1_accuracy, s.baseline2_accuracy):
            print(f"\n  ✅ SellSignalScorer 优于所有基线")
        elif s.scorer_accuracy > 0:
            print(f"\n  ⚠️ SellSignalScorer 未显著优于基线")
        else:
            print(f"\n  ❌ SellSignalScorer 不可用或表现不如基线")

        print("\n" + "=" * 80)


# ── 主函数 ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="卖出信号评分器回测")
    parser.add_argument("--cases", type=int, default=None,
                        help="只跑前N个案例（调试用）")
    parser.add_argument("--category", choices=["real_drop", "washout"], default=None,
                        help="只跑指定类别")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.category:
        cases = [c for c in cases if c[2] == args.category]
    if args.cases:
        cases = cases[:args.cases]

    backtester = SellSignalBacktester()
    results, summary = backtester.run_all(cases)

    return results, summary


if __name__ == "__main__":
    main()
