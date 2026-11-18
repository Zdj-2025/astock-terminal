#!/usr/bin/env python3
"""板块扫描器 — 用V3+策略引擎聚合分析板块整体状态。

用法:
  # 扫描中证酒指数成分股
  PYTHONPATH=. python scripts/screener/sector_scanner.py --index 399987

  # 扫描指定股票列表
  PYTHONPATH=. python scripts/screener/sector_scanner.py --codes 000858,600519,002304

  # 用akshare板块代码扫描
  PYTHONPATH=. python scripts/screener/sector_scanner.py --concept BK0477

输出:
  1. 板块总览：信号分布、平均V3、资金流向
  2. 个股排行：V3/策略评分排行
  3. 板块健康度：综合评级(A/B/C/D)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 项目依赖 ──────────────────────────────────────────────
from scripts.core.data_store import DataStore
from scripts.screener.v3_derivative_scorer import V3DerivativeScorer, V3Score
from scripts.screener.strategy_engine import StrategyEngine, StrategyScore

# 尝试导入akshare（部分功能需要）
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class StockResult:
    """单只股票扫描结果。"""
    code: str = ""
    name: str = ""
    close: float = 0.0
    change_pct: float = 0.0
    v3: V3Score = field(default_factory=V3Score)
    bottom_score: StrategyScore = field(default_factory=lambda: StrategyScore())
    uptrend_score: StrategyScore = field(default_factory=lambda: StrategyScore())
    # 距高点
    dh180: float = 0.0
    dh360: float = 0.0
    ret20: float = 0.0


@dataclass
class VolumeRankResult:
    """全市场成交额排行结果。"""
    scan_date: str = ""
    top_stocks: list[dict] = field(default_factory=list)  # [{code, name, amount, change, mkt_cap, sector}]
    sector_heat: dict = field(default_factory=dict)       # {板块名: {count, total_amount, avg_change}}
    sentiment: dict = field(default_factory=dict)          # 市场情绪偏好


@dataclass
class SectorReport:
    """板块扫描报告。"""
    sector_name: str = ""
    scan_date: str = ""
    stock_count: int = 0
    scored_count: int = 0

    # ── 信号分布 ──
    signal_counts: dict = field(default_factory=dict)
    # signal -> [stock_names]

    # ── 聚合指标 ──
    avg_v3: float = 0.0
    median_v3: float = 0.0
    avg_f1: float = 0.0       # 板块平均趋势方向
    avg_f2: float = 0.0       # 板块平均加速度
    avg_dh180: float = 0.0    # 板块平均距180日高点
    avg_ret20: float = 0.0    # 板块平均20日涨幅

    # ── 策略命中 ──
    bottom_hits: int = 0      # 底部反转命中数
    uptrend_hits: int = 0     # 主升浪命中数
    bottom_avg_score: float = 0.0
    uptrend_avg_score: float = 0.0

    # ── 板块健康度 ──
    health_grade: str = ""    # A/B/C/D
    health_score: float = 0.0
    health_details: dict = field(default_factory=dict)

    # ── 成交额维度（新增）──
    total_amount: float = 0.0          # 板块总成交额（亿）
    avg_amount: float = 0.0            # 个股均成交额（亿）
    amount_rank_in_market: int = 0     # 全市场板块成交额排名

    # ── 个股排行 ──
    stocks: list[StockResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# 板块健康度计算
# ═══════════════════════════════════════════════════════════

def compute_health(report: SectorReport) -> None:
    """计算板块健康度评级。

    评分维度(0-100):
      - 动能(30分): avg_f2正比例 + avg_f1正比例
      - 位置(25分): 距高点越低越好(弹性空间大)
      - 策略命中(25分): 底部反转+主升浪命中占比
      - 广度(20分): 正信号占比(主升浪+底部拐点 / 总数)
    """
    n = report.scored_count
    if n == 0:
        report.health_grade = "N/A"
        return

    # 1. 动能分(30分)
    f2_pos_ratio = sum(1 for s in report.stocks if s.v3.f2 > 0) / n
    f1_pos_ratio = sum(1 for s in report.stocks if s.v3.f1 > 0) / n
    momentum_score = (f2_pos_ratio * 20) + (f1_pos_ratio * 10)

    # 2. 位置分(25分): 距高点越低分越高
    avg_dh = report.avg_dh180
    if avg_dh <= -40:
        pos_score = 25
    elif avg_dh <= -30:
        pos_score = 20
    elif avg_dh <= -20:
        pos_score = 15
    elif avg_dh <= -10:
        pos_score = 10
    else:
        pos_score = 5

    # 3. 策略命中分(25分)
    hit_ratio = (report.bottom_hits + report.uptrend_hits) / n
    strategy_score = min(hit_ratio * 50, 25)  # 上限25分

    # 4. 广度分(20分): 正信号占比
    positive_signals = report.signal_counts.get("🚀 主升浪", []) + report.signal_counts.get("🔄 底部拐点", [])
    breadth_score = (len(positive_signals) / n) * 20

    total = momentum_score + pos_score + strategy_score + breadth_score
    report.health_score = round(total, 1)

    # 评级
    if total >= 75:
        report.health_grade = "A"
    elif total >= 55:
        report.health_grade = "B"
    elif total >= 35:
        report.health_grade = "C"
    else:
        report.health_grade = "D"

    report.health_details = {
        "动能分": round(momentum_score, 1),
        "位置分": round(pos_score, 1),
        "策略命中分": round(strategy_score, 1),
        "广度分": round(breadth_score, 1),
        "f2正比例": round(f2_pos_ratio * 100, 1),
        "f1正比例": round(f1_pos_ratio * 100, 1),
    }


# ═══════════════════════════════════════════════════════════
# 股票列表获取
# ═══════════════════════════════════════════════════════════

def get_index_stocks(index_code: str) -> dict[str, str]:
    """从指数代码获取成分股 {code: name}。"""
    import akshare as ak
    # 去掉可能的前缀
    idx = index_code.lstrip("CSI").lstrip("sh").lstrip("sz")
    df = ak.index_stock_cons_csindex(symbol=idx)
    result = {}
    for _, row in df.iterrows():
        code = str(row["成分券代码"]).zfill(6)
        name = str(row["成分券名称"]).strip()
        result[code] = name
    return result


def get_concept_stocks(concept_code: str) -> dict[str, str]:
    """从概念板块代码获取成分股。"""
    import akshare as ak
    df = ak.stock_board_concept_cons_em(symbol=concept_code)
    result = {}
    for _, row in df.iterrows():
        code = str(row["代码"]).zfill(6)
        name = str(row["名称"]).strip()
        result[code] = name
    return result


def get_industry_stocks(industry_name: str) -> dict[str, str]:
    """从行业名称获取成分股。"""
    import akshare as ak
    df = ak.stock_board_industry_cons_em(symbol=industry_name)
    result = {}
    for _, row in df.iterrows():
        code = str(row["代码"]).zfill(6)
        name = str(row["名称"]).strip()
        result[code] = name
    return result


# ═══════════════════════════════════════════════════════════
# 核心扫描逻辑
# ═══════════════════════════════════════════════════════════

def scan_sector(
    codes: dict[str, str],
    sector_name: str = "自定义板块",
    verbose: bool = True,
) -> SectorReport:
    """扫描板块所有成分股，生成板块报告。"""

    store = DataStore()
    v3_scorer = V3DerivativeScorer()
    engine = StrategyEngine(v3_scorer)

    report = SectorReport(
        sector_name=sector_name,
        scan_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        stock_count=len(codes),
    )

    stock_results: list[StockResult] = []

    for code, name in codes.items():
        try:
            df = store.get_kline(code, period="D")
            if df is None or len(df) < 100:
                if verbose:
                    print(f"  ⚠️ {name}({code}): 数据不足")
                continue

            # V3评分
            v3 = v3_scorer.score_stock(code, df)

            # 策略评分
            full = v3_scorer.analyze_stock(code, df) if hasattr(v3_scorer, "analyze_stock") else {}
            # 用策略引擎做完整评估
            eval_result = engine.full_evaluation(v3)
            br = eval_result.get("bottom_reversal", StrategyScore())
            ut = eval_result.get("uptrend", StrategyScore())

            # 价格信息
            close = df["close"].iloc[-1]
            prev_close = df["close"].iloc[-2]
            chg = round((close - prev_close) / prev_close * 100, 2)

            # 距高点
            h180 = df["high"].rolling(180).max().iloc[-1]
            h360 = df["high"].rolling(min(360, len(df))).max().iloc[-1]
            dh180 = round((close / h180 - 1) * 100, 1)
            dh360 = round((close / h360 - 1) * 100, 1)

            # 20日涨幅
            ret20 = round((close / df["close"].iloc[-20] - 1) * 100, 1) if len(df) >= 20 else 0.0

            sr = StockResult(
                code=code, name=name, close=close,
                change_pct=chg, v3=v3,
                bottom_score=br, uptrend_score=ut,
                dh180=dh180, dh360=dh360, ret20=ret20,
            )
            stock_results.append(sr)

            if verbose:
                sig = v3.signal
                v3t = v3.total
                br_hit = "🅑" if br.match else "  "
                ut_hit = "🅤" if ut.match else "  "
                print(f"  {br_hit}{ut_hit} {name:6s}({code}) V3={v3t:.1f} {sig}")

        except Exception as e:
            if verbose:
                print(f"  ❌ {name}({code}): {e}")

    # ── 聚合统计 ──
    report.scored_count = len(stock_results)
    report.stocks = sorted(stock_results, key=lambda s: s.v3.total, reverse=True)

    if stock_results:
        # 信号分布
        for sr in stock_results:
            sig = sr.v3.signal
            if sig not in report.signal_counts:
                report.signal_counts[sig] = []
            report.signal_counts[sig].append(f"{sr.name}({sr.code})")

        # 平均值
        report.avg_v3 = round(np.mean([s.v3.total for s in stock_results]), 2)
        report.median_v3 = round(np.median([s.v3.total for s in stock_results]), 2)
        report.avg_f1 = round(np.mean([s.v3.f1 for s in stock_results]), 4)
        report.avg_f2 = round(np.mean([s.v3.f2 for s in stock_results]), 4)
        report.avg_dh180 = round(np.mean([s.dh180 for s in stock_results]), 1)
        report.avg_ret20 = round(np.mean([s.ret20 for s in stock_results]), 1)

        # 策略命中
        report.bottom_hits = sum(1 for s in stock_results if s.bottom_score.match)
        report.uptrend_hits = sum(1 for s in stock_results if s.uptrend_score.match)
        hits_b = [s for s in stock_results if s.bottom_score.match]
        hits_u = [s for s in stock_results if s.uptrend_score.match]
        report.bottom_avg_score = round(np.mean([s.bottom_score.total for s in hits_b]), 1) if hits_b else 0.0
        report.uptrend_avg_score = round(np.mean([s.uptrend_score.total for s in hits_u]), 1) if hits_u else 0.0

    # 健康度
    compute_health(report)

    return report


# ═══════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════

SIGNAL_ORDER = ["🚀 主升浪", "🔄 底部拐点", "⚠️ 减速见顶", "🔴 加速下跌", "观望"]


def print_report(report: SectorReport) -> None:
    """打印板块扫描报告。"""

    grade_emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "N/A": "⚪"}

    print("\n" + "=" * 70)
    print(f"  📊 板块扫描报告: {report.sector_name}")
    print(f"  📅 扫描时间: {report.scan_date}")
    print(f"  成分股: {report.stock_count}只 | 已评分: {report.scored_count}只")
    print("=" * 70)

    # ── 健康度 ──
    ge = grade_emoji.get(report.health_grade, "⚪")
    print(f"\n  {ge} 板块健康度: {report.health_grade} ({report.health_score}/100)")
    for k, v in report.health_details.items():
        print(f"     {k}: {v}")

    # ── 核心指标 ──
    print(f"\n  📈 核心指标:")
    print(f"     平均V3: {report.avg_v3} | 中位V3: {report.median_v3}")
    print(f"     平均f'(趋势): {'↑' if report.avg_f1 > 0 else '↓'} {report.avg_f1}")
    print(f"     平均f''(加速度): {'↑' if report.avg_f2 > 0 else '↓'} {report.avg_f2}")
    print(f"     平均距180日高: {report.avg_dh180}%")
    print(f"     平均20日涨幅: {report.avg_ret20}%")

    # ── 策略命中 ──
    print(f"\n  🎯 策略命中:")
    print(f"     底部反转: {report.bottom_hits}/{report.scored_count}只命中"
          f"{f' (均分{report.bottom_avg_score})' if report.bottom_hits else ''}")
    print(f"     主升浪: {report.uptrend_hits}/{report.scored_count}只命中"
          f"{f' (均分{report.uptrend_avg_score})' if report.uptrend_hits else ''}")

    # ── 信号分布 ──
    print(f"\n  📊 信号分布:")
    for sig in SIGNAL_ORDER:
        stocks = report.signal_counts.get(sig, [])
        if stocks:
            print(f"     {sig} ({len(stocks)}只):")
            for s in stocks[:8]:
                print(f"        {s}")
            if len(stocks) > 8:
                print(f"        ... 还有{len(stocks)-8}只")

    # ── Top排行 ──
    print(f"\n  🏆 V3评分 Top 10:")
    print(f"     {'排名':>4} {'代码':>8} {'名称':8} {'V3':>5} {'信号':12} {'距高180':>8} {'20日%':>7} {'底部':>5} {'主升浪':>5}")
    print(f"     {'─'*4} {'─'*8} {'─'*8} {'─'*5} {'─'*12} {'─'*8} {'─'*7} {'─'*5} {'─'*5}")
    for i, sr in enumerate(report.stocks[:10], 1):
        br_mark = "✅" if sr.bottom_score.match else ""
        ut_mark = "✅" if sr.uptrend_score.match else ""
        print(f"     {i:>4} {sr.code:>8} {sr.name:8} {sr.v3.total:>5.1f} {sr.v3.signal:12} "
              f"{sr.dh180:>7.1f}% {sr.ret20:>6.1f}% {br_mark:>5} {ut_mark:>5}")

    # ── 板块判断 ──
    print(f"\n  💡 板块判断:")
    judgment = _judge_sector(report)
    for line in judgment:
        print(f"     {line}")

    print("\n" + "=" * 70)


def _judge_sector(report: SectorReport) -> list[str]:
    """基于扫描结果给出板块判断。"""
    lines = []
    n = report.scored_count
    if n == 0:
        return ["数据不足，无法判断"]

    # f2方向
    if report.avg_f2 > 5:
        lines.append("🟢 板块加速度(f'')强势向上 — 整体启动中")
    elif report.avg_f2 > 0:
        lines.append("🟡 板块加速度(f'')微正 — 底部蓄力阶段")
    else:
        lines.append("🔴 板块加速度(f'')为负 — 仍在下探")

    # f1方向
    if report.avg_f1 > 0:
        lines.append("🟢 板块趋势(f')向上 — 多头主导")
    elif report.avg_f1 > -0.5:
        lines.append("🟡 板块趋势(f')接近零轴 — 方向选择中")
    else:
        lines.append("🔴 板块趋势(f')向下 — 空头主导")

    # 策略命中
    hit_ratio = (report.bottom_hits + report.uptrend_hits) / n
    if hit_ratio >= 0.3:
        lines.append(f"🟢 策略命中率高({hit_ratio:.0%}) — 板块级机会")
    elif hit_ratio > 0:
        lines.append(f"🟡 策略命中率一般({hit_ratio:.0%}) — 个别个股机会")
    else:
        lines.append("🔴 无策略命中 — 暂无系统性机会")

    # 位置
    if report.avg_dh180 < -30:
        lines.append(f"🟢 平均距180日高点{report.avg_dh180}% — 位置极低，弹性大")
    elif report.avg_dh180 < -15:
        lines.append(f"🟡 平均距180日高点{report.avg_dh180}% — 位置偏低")
    else:
        lines.append(f"🔴 平均距180日高点{report.avg_dh180}% — 位置偏高")

    # 综合结论
    if report.health_grade == "A":
        lines.append("➡️ 结论：板块整体启动，可以积极布局")
    elif report.health_grade == "B":
        lines.append("➡️ 结论：板块底部蓄力，可逐步建仓")
    elif report.health_grade == "C":
        lines.append("➡️ 结论：板块分化，精选个股")
    else:
        lines.append("➡️ 结论：板块偏弱，观望为主")

    return lines


# ═══════════════════════════════════════════════════════════
# 成交额排行选股法（一杆清方法论融入）
# ═══════════════════════════════════════════════════════════

def get_market_cap_category(mkt_cap: float) -> str:
    """市值分类。"""
    if mkt_cap >= 1000:
        return "大盘"
    elif mkt_cap >= 300:
        return "中盘"
    else:
        return "小盘"


def _batch_get_sectors(codes: list[str]) -> dict[str, str]:
    """批量获取个股所属行业板块。

    策略：从东财行业涨幅榜取前20个行业，查成分股做反向映射。
    这样只查20个行业就能覆盖大部分活跃个股。
    """
    sector_map: dict[str, str] = {}
    if not HAS_AKSHARE:
        return sector_map
    try:
        # 获取行业涨幅榜（含涨跌信息）
        df_ind = ak.stock_board_industry_name_em()
        # 按涨幅绝对值排序，取前30个（覆盖最活跃行业）
        df_ind["abs_chg"] = df_ind["涨跌额"].abs()
        df_ind = df_ind.sort_values("abs_chg", ascending=False).head(30)

        codes_set = set(codes)
        for _, ind_row in df_ind.iterrows():
            ind_name = str(ind_row["板块名称"]).strip()
            try:
                df_cons = ak.stock_board_industry_cons_em(symbol=ind_name)
                for _, c in df_cons.iterrows():
                    c_code = str(c["代码"]).zfill(6)
                    if c_code in codes_set:
                        sector_map[c_code] = ind_name
            except Exception:
                continue
            # 全找到了就退出
            if codes_set.issubset(set(sector_map.keys())):
                break
    except Exception as e:
        print(f"  ⚠️ 板块映射获取失败: {e}")
    return sector_map


def scan_volume_rank(top_n: int = 30) -> VolumeRankResult:
    """全市场成交额排行扫描 — 锁定资金主攻方向。

    方法论来源：一杆清「成交金额排行选股法」
    ① 成交金额排行，看涨跌分布，对照大盘浪型
    ② 归纳板块，找资金最集中板块的浪型趋势
    ③ 判断市场情绪偏好：大/小盘、投资/投机、机构/游资
    """
    import time

    if not HAS_AKSHARE:
        print("❌ 需要 akshare 库")
        return VolumeRankResult()

    print(f"📊 获取全市场成交额排行 Top {top_n}...")

    # 获取全市场实时行情
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"❌ 获取行情失败: {e}")
        return VolumeRankResult()

    # 清洗数据
    df = df[df["成交额"] > 0].copy()
    df = df.sort_values("成交额", ascending=False).head(top_n)

    # 检查可用的列
    amount_col = "成交额"
    change_col = "涨跌幅"
    mkt_cap_col = None
    for col_name in ["总市值", "total_mv"]:
        if col_name in df.columns:
            mkt_cap_col = col_name
            break

    # 尝试用 ak 板块接口批量获取板块映射
    sector_map = _batch_get_sectors(list(df["代码"].values))

    result = VolumeRankResult(
        scan_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

    # 逐只组装
    sector_stocks = {}  # {板块: [stocks]}
    for _, row in df.iterrows():
        code_raw = str(row["代码"]).zfill(6)
        # akshare的代码可能带前缀
        code = code_raw.lstrip("0") if code_raw.startswith("6") else code_raw
        code = code_raw  # 保持原样
        name = str(row["名称"]).strip()
        amount = float(row[amount_col]) / 1e8  # 转亿
        change = float(row[change_col])
        mkt_cap = float(row[mkt_cap_col]) / 1e8 if mkt_cap_col and pd.notna(row.get(mkt_cap_col)) else 0

        # 从缓存获取板块
        sector = sector_map.get(code_raw, "未知")

        stock_info = {
            "code": code,
            "name": name,
            "amount": round(amount, 2),
            "change": round(change, 2),
            "mkt_cap": round(mkt_cap, 0),
            "cap_category": get_market_cap_category(mkt_cap),
            "sector": sector,
        }
        result.top_stocks.append(stock_info)

        # 按板块归类
        if sector not in sector_stocks:
            sector_stocks[sector] = []
        sector_stocks[sector].append(stock_info)

    # 板块热度排名
    for sector, stocks in sorted(sector_stocks.items(), key=lambda x: len(x[1]), reverse=True):
        total_amt = sum(s["amount"] for s in stocks)
        avg_chg = np.mean([s["change"] for s in stocks])
        result.sector_heat[sector] = {
            "count": len(stocks),
            "total_amount": round(total_amt, 2),
            "avg_change": round(avg_chg, 2),
            "stocks": [f"{s['name']}({s['code']})" for s in stocks],
        }

    # 市场情绪偏好分析
    if result.top_stocks:
        n = len(result.top_stocks)
        red_count = sum(1 for s in result.top_stocks if s["change"] > 0)
        green_count = sum(1 for s in result.top_stocks if s["change"] < 0)
        big_cap = sum(1 for s in result.top_stocks if s["cap_category"] == "大盘")
        mid_cap = sum(1 for s in result.top_stocks if s["cap_category"] == "中盘")
        small_cap = sum(1 for s in result.top_stocks if s["cap_category"] == "小盘")
        avg_chg = np.mean([s["change"] for s in result.top_stocks])

        # 投机属性判断：涨幅>5%的小盘股占比
        speculative = sum(1 for s in result.top_stocks if s["cap_category"] == "小盘" and s["change"] > 5)

        result.sentiment = {
            "red_green": f"{red_count}红{green_count}绿",
            "red_ratio": round(red_count / n * 100, 1),
            "cap_preference": f"大盘{big_cap}/中盘{mid_cap}/小盘{small_cap}",
            "avg_change": round(avg_chg, 2),
            "speculative_ratio": round(speculative / n * 100, 1) if n > 0 else 0,
            "dominant_style": "投机" if speculative / n > 0.3 else ("机构" if big_cap / n > 0.5 else "混合"),
        }

    return result


def print_volume_report(result: VolumeRankResult) -> None:
    """打印成交额排行报告。"""
    print("\n" + "=" * 70)
    print(f"  💰 全市场成交额排行报告")
    print(f"  📅 扫描时间: {result.scan_date}")
    print("=" * 70)

    if not result.top_stocks:
        print("  ❌ 无数据")
        return

    # ── Top个股 ──
    print(f"\n  📊 成交额 Top {len(result.top_stocks)}:")
    print(f"     {'排名':>4} {'代码':>8} {'名称':8} {'成交额(亿)':>10} {'涨跌%':>7} {'市值(亿)':>10} {'类型':>5} {'板块'}")
    print(f"     {'─'*4} {'─'*8} {'─'*8} {'─'*10} {'─'*7} {'─'*10} {'─'*5} {'─'*8}")
    for i, s in enumerate(result.top_stocks, 1):
        chg_icon = "🔴" if s["change"] < 0 else "🟢" if s["change"] > 0 else "⚪"
        print(f"     {i:>4} {s['code']:>8} {s['name']:8} {s['amount']:>10.2f} "
              f"{chg_icon}{s['change']:>5.2f}% {s['mkt_cap']:>10.0f} {s['cap_category']:>5} {s['sector']}")

    # ── 板块资金集中度 ──
    if result.sector_heat:
        print(f"\n  🔥 板块资金集中度（按个股数量排序）:")
        for i, (sector, info) in enumerate(result.sector_heat.items(), 1):
            chg = info["avg_change"]
            chg_icon = "🔴" if chg < 0 else "🟢" if chg > 0 else "⚪"
            print(f"     {i}. {sector}: {info['count']}只 | 成交额{info['total_amount']:.1f}亿 | {chg_icon}均涨{chg}%")
            for name in info["stocks"][:5]:
                print(f"        · {name}")
            if len(info["stocks"]) > 5:
                print(f"        ... 还有{len(info['stocks'])-5}只")

    # ── 市场情绪 ──
    if result.sentiment:
        print(f"\n  🎭 市场情绪偏好:")
        print(f"     涨跌分布: {result.sentiment['red_green']} (红盘占比{result.sentiment['red_ratio']}%)")
        print(f"     市值偏好: {result.sentiment['cap_preference']}")
        print(f"     均涨跌幅: {result.sentiment['avg_change']:+.2f}%")
        print(f"     主导风格: {result.sentiment['dominant_style']}" +
              f" (投机股占比{result.sentiment['speculative_ratio']}%)")

    # ── 操作建议 ──
    print(f"\n  💡 资金方向判断:")
    top_sector = list(result.sector_heat.keys())[0] if result.sector_heat else "无"
    if result.sentiment.get("red_ratio", 0) > 60:
        print(f"     🟢 资金活跃偏多，主攻方向: {top_sector}")
        print(f"     → 关注 {top_sector} 板块的龙头个股")
    elif result.sentiment.get("red_ratio", 0) > 40:
        print(f"     🟡 资金分化，局部机会在: {top_sector}")
    else:
        print(f"     🔴 资金偏空，成交集中在少数板块({top_sector})")
        print(f"     → 谨慎操作，等待资金方向明确")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="板块扫描器 — V3+策略引擎聚合分析")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", help="中证指数代码，如 399987(中证酒)")
    group.add_argument("--codes", help="逗号分隔的股票代码，如 000858,600519")
    group.add_argument("--concept", help="东财概念板块名称，如 白酒")
    group.add_argument("--industry", help="东财行业板块名称，如 酿酒行业")
    group.add_argument("--top-volume", nargs="?", const="30", type=int, help="全市场成交额排行(默认Top30)")

    parser.add_argument("--name", default="", help="自定义板块名称")
    parser.add_argument("--quiet", action="store_true", help="静默模式，只输出报告")

    args = parser.parse_args()

    # ── 成交额排行模式 ──
    if args.top_volume is not None:
        result = scan_volume_rank(top_n=args.top_volume)
        print_volume_report(result)
        return

    # 获取成分股
    codes: dict[str, str] = {}
    sector_name = args.name

    if args.index:
        print(f"📂 获取指数 {args.index} 成分股...")
        codes = get_index_stocks(args.index)
        if not sector_name:
            sector_name = f"中证指数{args.index}"

    elif args.codes:
        store = DataStore()
        for code in args.codes.split(","):
            code = code.strip().zfill(6)
            codes[code] = code  # 名称后续填充
        if not sector_name:
            sector_name = "自定义股票池"

    elif args.concept:
        print(f"📂 获取概念板块「{args.concept}」成分股...")
        codes = get_concept_stocks(args.concept)
        if not sector_name:
            sector_name = f"概念-{args.concept}"

    elif args.industry:
        print(f"📂 获取行业板块「{args.industry}」成分股...")
        codes = get_industry_stocks(args.industry)
        if not sector_name:
            sector_name = f"行业-{args.industry}"

    if not codes:
        print("❌ 未获取到成分股")
        sys.exit(1)

    print(f"📋 共 {len(codes)} 只成分股，开始扫描...\n")

    # 扫描
    report = scan_sector(codes, sector_name=sector_name, verbose=not args.quiet)

    # 输出报告
    print_report(report)


if __name__ == "__main__":
    main()
