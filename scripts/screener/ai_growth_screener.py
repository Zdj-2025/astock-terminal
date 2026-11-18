#!/usr/bin/env python3
"""AI产业进攻型选股器。

筛选条件：
1. 周线回踩MA20附近（±5%内），距前高在Fibonacci回撤区间
2. 日线初步形成上涨趋势（W3启动特征），30分钟MACD零轴上方金叉
3. 有外资（北向资金）近期注入
4. 总市值≤200亿
5. 净利润为正
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from scripts.core.tushare_client import TushareClient, from_ts_code


def run_screener():
    client = TushareClient()
    trade_date = datetime.now().strftime("%Y%m%d")
    print(f"AI进攻型选股器 — {trade_date}")

    # ── Phase 1: 基本面过滤（市值≤200亿+净利润为正）──
    print("\n[Phase 1] 基本面过滤...")
    df_basic = _get_fundamental_universe(client)
    print(f"  基本面合格: {len(df_basic)} 只")

    if df_basic.empty:
        print("  无合格股票")
        return pd.DataFrame()

    # ── Phase 2: 北向资金筛选 ──
    print("\n[Phase 2] 北向资金筛选...")
    northbound_codes = _get_northbound_stocks()
    print(f"  北向持仓股: {len(northbound_codes)} 只")
    df_phase2 = df_basic[df_basic["symbol"].isin(northbound_codes)]
    print(f"  基本面+北向交集: {len(df_phase2)} 只")
    if df_phase2.empty:
        df_phase2 = df_basic
        print("  北向交集为空，回退到基本面池")

    # ── Phase 3: 周线技术筛选 ──
    print("\n[Phase 3] 周线技术筛选...")
    weekly_results = []
    total = len(df_phase2)
    for i, row in df_phase2.iterrows():
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{total}")
        try:
            w_score = _check_weekly(client, row["symbol"])
            if w_score is not None:
                record = row.to_dict()
                record.update(w_score)
                weekly_results.append(record)
        except Exception:
            continue

    df_phase3 = pd.DataFrame(weekly_results)
    print(f"  周线合格: {len(df_phase3)} 只")
    if df_phase3.empty:
        print("  无周线合格股")
        return pd.DataFrame()

    # ── Phase 4: 日线趋势+W3特征 ──
    print("\n[Phase 4] 日线趋势筛选...")
    daily_results = []
    for _, row in df_phase3.iterrows():
        try:
            d_score = _check_daily_trend(client, row["symbol"])
            if d_score is not None:
                daily_results.append((row["ts_code"], d_score))
        except Exception:
            continue

    # 合并日线得分（用字典映射，避免iterrows copy问题）
    daily_dict = {tc: s for tc, s in daily_results}
    df_phase3["daily_score"] = df_phase3["ts_code"].map(
        lambda tc: daily_dict.get(tc, {}).get("daily_score", 0))
    df_phase3["daily_stage"] = df_phase3["ts_code"].map(
        lambda tc: daily_dict.get(tc, {}).get("stage", ""))
    df_phase3["daily_ma5"] = df_phase3["ts_code"].map(
        lambda tc: daily_dict.get(tc, {}).get("ma5", 0))
    df_phase3["daily_ma20"] = df_phase3["ts_code"].map(
        lambda tc: daily_dict.get(tc, {}).get("ma20", 0))

    daily_passed = df_phase3[df_phase3["ts_code"].isin(
        [d[0] for d in daily_results if d[1]["daily_score"] >= 40])]
    print(f"  日线趋势合格: {len(daily_passed)} 只")

    if daily_passed.empty:
        daily_passed = df_phase3.sort_values(
            "weekly_score", ascending=False).head(30)

    # ── Phase 5: 30分钟MACD ──
    print("\n[Phase 5] 30分钟MACD筛选...")
    macd_results = []
    for _, row in daily_passed.iterrows():
        try:
            m_score = _check_30min_macd(row["symbol"])
            if m_score is not None:
                macd_results.append((row["ts_code"], m_score))
        except Exception:
            continue

    macd_dict = {tc: s for tc, s in macd_results}

    # ── Phase 6: 综合评分 ──
    print("\n[Phase 6] 综合评分...")
    results = []
    for _, row in daily_passed.iterrows():
        tc = row["ts_code"]
        w_score = row.get("weekly_score", 0)
        d_score = row.get("daily_score", 0)
        m_score = macd_dict.get(tc, {}).get("macd_score", 0)
        has_golden = macd_dict.get(tc, {}).get("golden_cross_above0", False)

        composite = w_score * 0.35 + d_score * 0.35 + m_score * 0.30
        if has_golden:
            composite = min(100, composite + 5)

        record = {
            "ts_code": tc,
            "symbol": row.get("symbol", ""),
            "name": row.get("name", ""),
            "close": row.get("close", 0),
            "total_mv_yi": round(row.get("total_mv", 0) / 1e4, 1),
            "pe_ttm": row.get("pe_ttm", 0),
            "weekly_score": round(w_score, 1),
            "daily_score": round(d_score, 1),
            "macd30_score": round(m_score, 1),
            "composite": round(composite, 1),
            "weekly_dist_ma20": round(row.get("dist_ma20_pct", 0), 2),
            "weekly_fib_retrace": round(row.get("fib_retrace", 0), 2),
            "daily_stage": row.get("daily_stage", ""),
            "macd30_golden": "Yes" if has_golden else "No",
            "northbound": row.get("northbound_change", 0),
        }
        results.append(record)

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("composite", ascending=False)
    df_result = df_result.head(20).reset_index(drop=True)

    print(f"\n  最终入选: {len(df_result)} 只")
    return df_result


# ── Phase 1: 基本面 ──

def _get_fundamental_universe(client: TushareClient) -> pd.DataFrame:
    """总市值≤200亿 + 净利润为正"""
    try:
        trade_day = client.get_recent_trade_day()
    except Exception:
        trade_day = datetime.now().strftime("%Y%m%d")

    df = client.get_daily_basic(trade_date=trade_day)
    # 当天数据未更新时回退到前几个交易日
    if df is None or df.empty:
        for offset in range(1, 5):
            fallback = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
            df = client.get_daily_basic(trade_date=fallback)
            if df is not None and not df.empty:
                trade_day = fallback
                break
    if df is None or df.empty:
        return pd.DataFrame()

    # 市值≤200亿
    df = df[df["total_mv"] <= 200 * 1e4].copy()
    # PE为正（盈利）
    if "pe_ttm" in df.columns:
        df = df[(df["pe_ttm"] > 0) & (df["pe_ttm"] < 200)]
    # 排除北交所
    df = df[~df["ts_code"].str.startswith(("8", "4"))]
    # 添加symbol列
    df["symbol"] = df["ts_code"].apply(lambda x: x.split(".")[0])

    # 补充名称（daily_basic不含name）
    stock_list = client.get_stock_list()
    if not stock_list.empty and "name" in stock_list.columns:
        name_map = dict(zip(stock_list["ts_code"], stock_list["name"]))
        df["name"] = df["ts_code"].map(name_map).fillna("")

    # 排除ST（name合并后才能过滤）
    if "name" in df.columns:
        df = df[~df["name"].str.contains("ST", na=False)]

    return df.reset_index(drop=True)


# ── Phase 2: 北向资金 ──

def _get_northbound_stocks() -> set:
    """获取近期北向资金增持的股票代码集合"""
    try:
        df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
        if df is not None and not df.empty:
            # 增持市值>0的
            increasing = df[df["今日增持估计-市值"].astype(float) > 0]
            return set(increasing["代码"].tolist())
    except Exception:
        pass

    # 备用：用个股资金流
    try:
        df2 = ak.stock_individual_fund_flow_rank(indicator="今日")
        if df2 is not None and not df2.empty:
            # 北向净买入为正
            if "北向资金流入" in df2.columns:
                nb = df2[pd.to_numeric(df2["北向资金流入"], errors="coerce") > 0]
                return set(nb["代码"].tolist())
    except Exception:
        pass

    return set()


# ── Phase 3: 周线技术 ──

def _check_weekly(client: TushareClient, symbol: str) -> dict | None:
    """周线回踩MA20附近 + Fibonacci回撤区间"""
    try:
        df = client.get_kline(symbol, freq="W", start="20250101", adj="qfq")
        if df is None or len(df) < 30:
            return None

        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values

        n = len(closes)
        ma20 = np.mean(closes[-20:])
        current = closes[-1]

        # 条件1：距周线MA20在±5%内
        dist_ma20 = (current / ma20 - 1) * 100
        if abs(dist_ma20) > 8:
            return None

        # 条件2：距前高在Fibonacci回撤区间
        lookback = min(52, n)
        prev_high = np.max(highs[-lookback:-1])
        recent_low = np.min(lows[-lookback:])

        if prev_high <= recent_low:
            return None

        fib_retrace = (prev_high - current) / (prev_high - recent_low)

        # 0.236~0.786 Fibonacci区间
        if fib_retrace < 0.1 or fib_retrace > 0.85:
            return None

        # 评分
        score = 0.0

        # MA20距离评分（越近越好）
        if abs(dist_ma20) <= 2:
            score += 35
        elif abs(dist_ma20) <= 4:
            score += 25
        else:
            score += 15

        # Fibonacci位置评分
        if 0.382 <= fib_retrace <= 0.618:
            score += 35  # 黄金回撤区
        elif 0.236 <= fib_retrace <= 0.786:
            score += 25
        else:
            score += 10

        # 周线趋势：MA5 > MA20加分
        ma5 = np.mean(closes[-5:])
        if ma5 > ma20:
            score += 30
        elif ma5 > ma20 * 0.98:
            score += 15

        return {
            "weekly_score": score,
            "dist_ma20_pct": dist_ma20,
            "fib_retrace": fib_retrace,
            "weekly_ma20": ma20,
            "weekly_high52": prev_high,
        }

    except Exception:
        return None


# ── Phase 4: 日线趋势 ──

def _check_daily_trend(client: TushareClient, symbol: str) -> dict | None:
    """日线初步上涨趋势 + W3启动特征"""
    try:
        start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
        df = client.get_kline(symbol, freq="D", start=start, adj="qfq")
        if df is None or len(df) < 60:
            return None

        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        volumes = df["volume"].astype(float).values

        n = len(closes)
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:]) if n >= 60 else np.mean(closes)

        current = closes[-1]

        score = 0.0
        stage = "unknown"

        # 趋势判断
        if ma5 > ma10 > ma20:
            score += 30
            stage = "uptrend"
        elif ma5 > ma10:
            score += 20
            stage = "turning"
        elif ma5 > ma20:
            score += 10
            stage = "recovering"
        else:
            return None  # 空头排列不选

        # W3启动特征：找近期低点，看是否有上涨突破
        recent_low_idx = np.argmin(lows[-40:])
        recent_low = lows[-40:][recent_low_idx]
        recent_high = np.max(highs[-40:])

        if current > recent_low * 1.05 and current > ma5:
            # 从低点反弹并站稳均线
            bounce = (current / recent_low - 1) * 100
            if 3 <= bounce <= 25:
                score += 25
                stage = "w3_startup"
            elif bounce > 25:
                score += 15
                stage = "w3_mid"
            else:
                score += 10

        # 量价配合
        vol_5 = np.mean(volumes[-5:])
        vol_20 = np.mean(volumes[-20:])
        if vol_20 > 0 and vol_5 > vol_20 * 1.2:
            score += 20
        elif vol_5 > vol_20:
            score += 10

        # MACD日线状态
        dif, dea, hist = _calc_macd(closes, 12, 26, 9)
        if dif[-1] > dea[-1]:
            score += 15
        # 近5日金叉
        for j in range(max(1, n - 5), n):
            if dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                score += 10
                break

        if score < 40:
            return None

        return {
            "daily_score": min(score, 100),
            "stage": stage,
            "ma5": ma5,
            "ma20": ma20,
        }

    except Exception:
        return None


# ── Phase 5: 30分钟MACD ──

def _check_30min_macd(symbol: str) -> dict | None:
    """30分钟MACD零轴上方金叉"""
    try:
        start_dt = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d 09:30:00")
        end_dt = datetime.now().strftime("%Y-%m-%d 15:00:00")
        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="30",
                                        start_date=start_dt,
                                        end_date=end_dt,
                                        adjust="qfq")
        if df is None or len(df) < 30:
            return None

        closes = df["收盘"].astype(float).values
        n = len(closes)

        dif, dea, hist = _calc_macd(closes, 12, 26, 9)

        # 检查零轴上方金叉
        golden_cross_above0 = False
        for j in range(max(1, n - 20), n):
            if dif[j] > dea[j] and dif[j - 1] <= dea[j - 1] and dif[j] > 0:
                golden_cross_above0 = True
                break

        score = 0.0
        # DIF在零轴上方
        if dif[-1] > 0:
            score += 40
        elif dif[-1] > dif[-5] if n >= 5 else True:
            score += 20

        # 金叉
        if golden_cross_above0:
            score += 40
        else:
            # 检查普通金叉
            for j in range(max(1, n - 10), n):
                if dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                    score += 20
                    break

        # 柱状图为正
        if hist[-1] > 0:
            score += 20

        return {
            "macd_score": min(score, 100),
            "golden_cross_above0": golden_cross_above0,
            "dif_last": float(dif[-1]),
            "dea_last": float(dea[-1]),
        }

    except Exception:
        return None


# ── 工具函数 ──

def _calc_macd(close: np.ndarray, fast=12, slow=26, signal=9):
    """计算MACD"""
    s = pd.Series(close, dtype=float)
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif.values, dea.values, hist.values


def format_report(df: pd.DataFrame) -> str:
    """格式化报告"""
    lines = []
    lines.append("# AI产业进攻型选股报告")
    lines.append("")
    lines.append(f"**日期**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("**筛选条件**: 周线MA20附近+Fibonacci回撤 | 日线W3启动趋势 | 30分钟MACD零轴上金叉 | 北向资金注入 | 市值≤200亿 | 盈利")
    lines.append("")
    lines.append("---")
    lines.append("")

    if df.empty:
        lines.append("无符合条件的股票。")
        return "\n".join(lines)

    lines.append("## 综合排名TOP20")
    lines.append("")
    lines.append("| 排名 | 代码 | 名称 | 综合分 | 周线分 | 日线分 | MACD30分 | 收盘价 | 市值(亿) | PE | 距周MA20 | Fib回撤 | 日线阶段 | 30分金叉 |")
    lines.append("|------|------|------|--------|--------|--------|----------|--------|----------|-----|----------|----------|----------|----------|")

    for i, row in df.iterrows():
        code = row.get("symbol", row.get("ts_code", ""))
        lines.append(
            f"| {i+1} | {code} | {row['name']} | "
            f"{row['composite']:.1f} | {row['weekly_score']:.1f} | "
            f"{row['daily_score']:.1f} | {row['macd30_score']:.1f} | "
            f"{row['close']:.2f} | {row['total_mv_yi']} | "
            f"{row.get('pe_ttm', 0):.1f} | "
            f"{row['weekly_dist_ma20']:+.1f}% | "
            f"{row['weekly_fib_retrace']:.2f} | "
            f"{row['daily_stage']} | "
            f"{row['macd30_golden']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 评分说明")
    lines.append("- **周线分(35%权重)**：距MA20越近分越高(35分) + Fibonacci黄金区0.382-0.618(35分) + MA5>MA20趋势(30分)")
    lines.append("- **日线分(35%权重)**：均线多头排列(30分) + W3启动反弹特征(25分) + 量价配合(20分) + MACD金叉(25分)")
    lines.append("- **MACD30分(30%权重)**：DIF>0(40分) + 零轴上方金叉(40分) + 红柱(20分)")
    lines.append("- **综合分** = 周线×35% + 日线×35% + MACD30×30% + 零轴金叉加分5")
    lines.append("")
    lines.append("### 操作建议")
    lines.append("- 综合分≥75：重点关注，等日线回调到MA10附近加仓")
    lines.append("- 综合分60-75：观察，等30分钟MACD零轴金叉确认后入场")
    lines.append("- 综合分<60：备选，暂不操作")
    lines.append("")
    lines.append("*报告由AI进攻型选股器自动生成，不构成投资建议。*")

    return "\n".join(lines)


if __name__ == "__main__":
    df = run_screener()
    report = format_report(df)
    print(report)

    # 保存
    report_dir = Path("reports/choice")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d")
    path = report_dir / f"AI进攻型_{ts}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存: {path}")
