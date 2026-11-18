#!/usr/bin/env python3
"""市场情绪监控CLI — 成交额TOP20 + 板块涨幅TOP20。

用法:
    python scripts/cli/market_sentiment_cli.py                # 今日快照
    python scripts/cli/market_sentiment_cli.py --date 20260512  # 指定日期
    python scripts/cli/market_sentiment_cli.py --top 30         # TOP30
    python scripts/cli/market_sentiment_cli.py --json           # JSON输出
    python scripts/cli/market_sentiment_cli.py --compare 5      # 对比近N日趋势
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scripts.core.tushare_client import TushareClient


def _fmt_amount(val):
    """成交额千元 → 可读格式"""
    if val >= 1e7:
        return f"{val/1e7:.1f}亿"
    elif val >= 1e4:
        return f"{val/1e4:.1f}万"
    else:
        return f"{val:.0f}千"


def get_volume_top20(ts: TushareClient, trade_date: str, top_n: int = 20):
    """获取成交额TOP N个股"""
    df = ts._call(ts._pro.daily,
                  trade_date=trade_date,
                  fields="ts_code,trade_date,open,high,low,close,"
                         "pct_chg,vol,amount")
    if df is None or df.empty:
        return pd.DataFrame()

    # amount 单位千元
    df = df.sort_values("amount", ascending=False).head(top_n).copy()

    # 补充股票名称和行业
    stock_list = ts.get_stock_list()
    if not stock_list.empty:
        merge = stock_list[["ts_code", "name", "industry"]].drop_duplicates("ts_code")
        df = df.merge(merge, on="ts_code", how="left")

    df["name"] = df.get("name", pd.Series([pd.NA]*len(df)))
    df["industry"] = df.get("industry", pd.Series([pd.NA]*len(df)))

    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


def get_concept_top20(ts: TushareClient, trade_date: str, top_n: int = 20):
    """获取概念板块涨幅TOP N"""
    # 获取概念板块列表
    concepts = ts._call(ts._pro.ths_index, src="ts", type="N")
    if concepts is None or concepts.empty:
        return pd.DataFrame()

    # 批量获取概念板块日线 — 串行调用避免限流
    results = []
    codes = concepts["ts_code"].tolist()

    # 分批处理，每批50个
    batch_size = 50
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        for code in batch:
            try:
                df = ts._call(ts._pro.ths_daily,
                              ts_code=code,
                              start_date=trade_date,
                              end_date=trade_date,
                              fields="ts_code,trade_date,open,close,pct_change,vol,amount")
                if df is not None and not df.empty:
                    results.append(df.iloc[0])
            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 合并板块名称
    name_map = dict(zip(concepts["ts_code"], concepts["name"]))
    df["name"] = df["ts_code"].map(name_map)

    df = df.sort_values("pct_change", ascending=False).head(top_n).copy()
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


def get_concept_bottom20(ts: TushareClient, trade_date: str, top_n: int = 20):
    """获取概念板块跌幅TOP N（跌幅最大）"""
    concepts = ts._call(ts._pro.ths_index, src="ts", type="N")
    if concepts is None or concepts.empty:
        return pd.DataFrame()

    results = []
    codes = concepts["ts_code"].tolist()

    batch_size = 50
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        for code in batch:
            try:
                df = ts._call(ts._pro.ths_daily,
                              ts_code=code,
                              start_date=trade_date,
                              end_date=trade_date,
                              fields="ts_code,trade_date,close,pct_change,vol")
                if df is not None and not df.empty:
                    results.append(df.iloc[0])
            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    name_map = dict(zip(concepts["ts_code"], concepts["name"]))
    df["name"] = df["ts_code"].map(name_map)

    df = df.sort_values("pct_change", ascending=True).head(top_n).copy()
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return df


def calc_sentiment(volume_df: pd.DataFrame, concept_df: pd.DataFrame):
    """计算市场情绪指标"""
    indicators = {}

    # === 个股情绪 ===
    if not volume_df.empty:
        pct = volume_df["pct_chg"]
        up_count = (pct > 0).sum()
        down_count = (pct < 0).sum()
        flat_count = (pct == 0).sum()
        avg_gain = pct.mean()
        max_gain = pct.max()
        max_loss = pct.min()

        indicators["volume"] = {
            "up": int(up_count),
            "down": int(down_count),
            "flat": int(flat_count),
            "up_ratio": round(up_count / len(pct), 2),
            "avg_pct": round(avg_gain, 2),
            "max_pct": round(max_gain, 2),
            "min_pct": round(max_loss, 2),
        }

        # 行业集中度
        if "industry" in volume_df.columns:
            ind_counts = volume_df["industry"].value_counts().head(5)
            indicators["volume"]["top_industries"] = {
                ind: int(cnt) for ind, cnt in ind_counts.items()
            }
    else:
        indicators["volume"] = None

    # === 板块情绪 ===
    if not concept_df.empty:
        pct = concept_df["pct_change"]
        up_count = (pct > 0).sum()
        down_count = (pct < 0).sum()
        avg_gain = pct.mean()

        indicators["concept"] = {
            "up": int(up_count),
            "down": int(down_count),
            "avg_pct": round(avg_gain, 2),
        }
    else:
        indicators["concept"] = None

    # === 综合判断 ===
    if indicators["volume"] and indicators["concept"]:
        vol_up = indicators["volume"]["up_ratio"]
        vol_avg = indicators["volume"]["avg_pct"]
        con_avg = indicators["concept"]["avg_pct"]

        if vol_up >= 0.8 and vol_avg > 3:
            mood = "狂热"
            emoji = "🔴🔴"
        elif vol_up >= 0.7 and vol_avg > 2:
            mood = "强势"
            emoji = "🔴"
        elif vol_up >= 0.6 and vol_avg > 1:
            mood = "偏强"
            emoji = "🟠"
        elif vol_up >= 0.5:
            mood = "中性"
            emoji = "🟡"
        elif vol_up >= 0.3:
            mood = "偏弱"
            emoji = "🔵"
        else:
            mood = "弱势"
            emoji = "🔵🔵"

        indicators["verdict"] = {
            "mood": mood,
            "emoji": emoji,
            "vol_up_ratio": vol_up,
            "vol_avg_pct": vol_avg,
            "concept_avg_pct": con_avg,
            "signal": _signal_text(mood, vol_avg),
        }
    else:
        indicators["verdict"] = {"mood": "数据不足", "emoji": "⚪", "signal": ""}

    return indicators


def _signal_text(mood, avg_pct):
    if mood in ("狂热", "强势"):
        return "情绪高涨，注意获利回吐风险"
    elif mood == "偏强":
        return "市场活跃，可适度参与"
    elif mood == "中性":
        return "市场观望，控制仓位"
    elif mood == "偏弱":
        return "情绪转弱，谨慎操作"
    else:
        return "市场低迷，以防守为主"


def check_holdings_alignment(ts: TushareClient, volume_df: pd.DataFrame,
                              concept_df: pd.DataFrame, holdings: dict):
    """检查持仓与市场热点对齐度（精确匹配）

    Args:
        ts: TushareClient实例
        volume_df: 成交额TOP DataFrame
        concept_df: 板块涨幅TOP DataFrame
        holdings: {ts_code: name} 字典
    """
    results = []

    # 构建板块→成分股映射（只在有板块数据时构建）
    concept_stock_map = {}
    if not concept_df.empty:
        concept_codes = concept_df["ts_code"].tolist() if "ts_code" in concept_df.columns else []
        for code in concept_codes:
            try:
                stocks = ts.get_concept_stocks(code)
                if not stocks.empty:
                    concept_stock_map[code] = set(stocks["ts_code"].tolist())
            except Exception:
                continue

    # 构建 ts_code → 持仓名称 的涨跌幅映射
    vol_pct_map = {}
    if not volume_df.empty:
        for _, row in volume_df.iterrows():
            vol_pct_map[row["ts_code"]] = row["pct_chg"]

    for ts_code, name in holdings.items():
        status = {"code": ts_code, "name": name, "in_volume_top": False,
                  "vol_top_pct": None, "related_concepts": []}

        # 检查是否在成交额TOP20
        if ts_code in vol_pct_map:
            status["in_volume_top"] = True
            status["vol_top_pct"] = round(float(vol_pct_map[ts_code]), 2)

        # 检查所属概念板块是否在涨幅TOP20（精确成分股匹配）
        for concept_code, stock_set in concept_stock_map.items():
            if ts_code in stock_set:
                row = concept_df[concept_df["ts_code"] == concept_code]
                if not row.empty:
                    status["related_concepts"].append({
                        "concept": row.iloc[0]["name"],
                        "pct": round(float(row.iloc[0]["pct_change"]), 2)
                    })

        status["related_concepts"] = sorted(
            status["related_concepts"], key=lambda x: x["pct"], reverse=True
        )[:3]

        results.append(status)

    return results


# ── 持久持仓配置 ──────────────────────────────────────────────────

def get_holdings_from_portfolio():
    """从 portfolio.xlsx 读取持仓"""
    import openpyxl
    path = os.path.join(project_root, "config", "portfolio.xlsx")
    if not os.path.exists(path):
        return {}

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    holdings = {}

    header_row = 1
    code_col = name_col = None
    for col in range(1, ws.max_column + 1):
        v = ws.cell(header_row, col).value
        if v and "代码" in str(v):
            code_col = col
        elif v and "名称" in str(v):
            name_col = col

    if not code_col:
        wb.close()
        return {}

    for row in range(2, ws.max_row + 1):
        code = ws.cell(row, code_col).value
        name = ws.cell(row, name_col).value if name_col else ""
        if code and str(code).strip():
            code_str = str(code).strip().replace(".SH", "").replace(".SZ", "")
            suffix = ".SH" if code_str.startswith(("6", "5")) else ".SZ"
            if code_str.startswith("15"):
                suffix = ".SZ"
            ts_code = f"{code_str}{suffix}"
            holdings[ts_code] = name or code_str

    wb.close()
    return holdings


# ── 报告生成 ──────────────────────────────────────────────────────

def generate_report(trade_date: str, volume_df: pd.DataFrame,
                    concept_df: pd.DataFrame, bottom_df: pd.DataFrame,
                    sentiment: dict, holdings_alignment: list,
                    top_n: int = 20) -> str:
    lines = []
    dt = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"

    lines.append(f"# 市场情绪监控 | {dt}")
    lines.append("")
    lines.append(f"> 数据源：Tushare | TOP{top_n} | 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 综合判断
    v = sentiment.get("verdict", {})
    if v:
        lines.append(f"## 综合判断：{v.get('emoji', '')} {v.get('mood', '')}")
        if v.get("signal"):
            lines.append(f"**{v['signal']}**")
        lines.append("")

    # === 成交额TOP20 ===
    lines.append(f"## 一、成交额TOP{top_n}个股")
    lines.append("")
    if volume_df.empty:
        lines.append("*数据获取失败*")
    else:
        sv = sentiment.get("volume", {})
        if sv:
            lines.append(f"涨跌比 {sv['up']}:{sv['down']}:{sv['flat']} "
                        f"(上涨率{sv['up_ratio']*100:.0f}%) | "
                        f"平均涨幅 {sv['avg_pct']:+.2f}% | "
                        f"最大涨幅 {sv['max_pct']:+.2f}%")
            if sv.get("top_industries"):
                top_ind = ", ".join(f"{k}({v}只)" for k, v in sv["top_industries"].items())
                lines.append(f"行业集中：{top_ind}")
            lines.append("")

        lines.append(f"| # | 代码 | 名称 | 收盘 | 涨跌% | 成交额 | 行业 |")
        lines.append("|---|------|------|------|-------|--------|------|")
        for idx, row in volume_df.iterrows():
            pct = row["pct_chg"]
            pct_str = f"{pct:+.2f}"
            name = row.get("name", "")
            industry = row.get("industry", "")
            amount_str = _fmt_amount(row["amount"])
            lines.append(f"| {idx} | {row['ts_code']} | {name} | "
                        f"{row['close']:.2f} | {pct_str} | {amount_str} | {industry} |")
    lines.append("")

    # === 板块涨幅TOP20 ===
    lines.append(f"## 二、概念板块涨幅TOP{top_n}")
    lines.append("")
    if concept_df.empty:
        lines.append("*数据获取失败*")
    else:
        sc = sentiment.get("concept", {})
        if sc:
            lines.append(f"板块平均涨幅 {sc['avg_pct']:+.2f}%")
            lines.append("")

        lines.append(f"| # | 板块 | 涨跌% | 成交量(万手) |")
        lines.append("|---|------|-------|-------------|")
        for idx, row in concept_df.iterrows():
            pct = row["pct_change"]
            pct_str = f"{pct:+.2f}"
            vol_wan = row["vol"] / 10000 if pd.notna(row["vol"]) else 0
            lines.append(f"| {idx} | {row['name']} | {pct_str} | {vol_wan:.1f} |")
    lines.append("")

    # === 板块跌幅TOP20 ===
    if not bottom_df.empty:
        lines.append(f"## 三、概念板块跌幅TOP{top_n}")
        lines.append("")
        lines.append(f"| # | 板块 | 涨跌% |")
        lines.append("|---|------|-------|")
        for idx, row in bottom_df.iterrows():
            pct = row["pct_change"]
            pct_str = f"{pct:+.2f}"
            lines.append(f"| {idx} | {row['name']} | {pct_str} |")
        lines.append("")

    # === 持仓对齐度 ===
    if holdings_alignment:
        lines.append("## 四、持仓与热点对齐度")
        lines.append("")
        lines.append("| 持仓 | 在成交TOP | 涨跌% | 关联热门板块 |")
        lines.append("|------|-----------|-------|-------------|")
        for h in holdings_alignment:
            in_top = "✓" if h["in_volume_top"] else "✗"
            pct_str = f"{h['vol_top_pct']:+.2f}" if h["vol_top_pct"] is not None else "-"
            concepts_str = ""
            if h["related_concepts"]:
                concepts_str = ", ".join(
                    f"{c['concept']}({c['pct']:+.2f}%)" for c in h["related_concepts"]
                )
            lines.append(f"| {h['name']}({h['code']}) | {in_top} | {pct_str} | {concepts_str or '无'} |")
        lines.append("")

    # === 原文方法论提醒 ===
    lines.append("## 五、方法论提醒")
    lines.append("")
    lines.append("> 参考一杆清/台球之门的方法：")
    lines.append("> - 成交额TOP20集体翻绿 → 资金集体出逃信号")
    lines.append("> - TOP20涨幅只有1-2% → 放量滞涨，警惕见顶")
    lines.append("> - 板块快速轮动（每天不同板块领涨）→ 资金无主线，短线为主")
    lines.append("> - 持续集中在AI/半导体 → 主线行情延续，可持有")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="市场情绪监控")
    parser.add_argument("--date", help="交易日期 YYYYMMDD，默认最近交易日")
    parser.add_argument("--top", type=int, default=20, help="TOP N，默认20")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    parser.add_argument("--compare", type=int, help="对比近N日趋势")
    parser.add_argument("--no-holdings", action="store_true", help="跳过持仓对齐检查")
    args = parser.parse_args()

    ts = TushareClient()

    # 确定交易日期
    if args.date:
        trade_date = args.date.replace("-", "")
    else:
        trade_date = ts.get_recent_trade_day(1)

    dt_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    stderr = sys.stderr
    print(f"📊 市场情绪监控 | {dt_str} | TOP{args.top}", file=stderr)
    print("─" * 50, file=stderr)

    # 1. 成交额TOP N
    print(f"⏳ 获取成交额TOP{args.top}...", file=stderr)
    volume_df = get_volume_top20(ts, trade_date, args.top)
    if not volume_df.empty:
        print(f"✓ 获取到 {len(volume_df)} 只个股", file=stderr)
    else:
        print("✗ 成交额数据获取失败", file=stderr)

    # 2. 板块涨幅TOP N
    print(f"⏳ 获取概念板块涨幅TOP{args.top}（需逐一查询，约30-60秒）...", file=stderr)
    concept_df = get_concept_top20(ts, trade_date, args.top)
    if not concept_df.empty:
        print(f"✓ 获取到 {len(concept_df)} 个板块", file=stderr)
    else:
        print("✗ 板块数据获取失败", file=stderr)

    # 3. 板块跌幅TOP N
    print(f"⏳ 获取概念板块跌幅TOP{args.top}...", file=stderr)
    bottom_df = get_concept_bottom20(ts, trade_date, args.top)
    if not bottom_df.empty:
        print(f"✓ 获取到 {len(bottom_df)} 个板块", file=stderr)
    else:
        print("✗ 跌幅数据获取失败", file=stderr)

    # 4. 计算情绪指标
    sentiment = calc_sentiment(volume_df, concept_df)
    v = sentiment.get("verdict", {})
    print(f"\n{'='*50}", file=stderr)
    print(f"综合判断：{v.get('emoji', '')} {v.get('mood', '未知')}", file=stderr)
    if v.get("signal"):
        print(f"信号：{v['signal']}", file=stderr)
    print(f"{'='*50}\n", file=stderr)

    # 5. 持仓对齐度
    holdings_alignment = []
    if not args.no_holdings:
        holdings = get_holdings_from_portfolio()
        if holdings:
            print(f"⏳ 检查 {len(holdings)} 只持仓与热点对齐度...", file=stderr)
            holdings_alignment = check_holdings_alignment(
                ts, volume_df, concept_df, holdings)

    # 6. 多日对比
    compare_data = None
    if args.compare and args.compare > 1:
        print(f"⏳ 对比近{args.compare}日趋势...", file=stderr)
        compare_data = []
        for i in range(1, args.compare + 1):
            d = ts.get_recent_trade_day(i)
            vdf = get_volume_top20(ts, d, args.top)
            if not vdf.empty:
                s = calc_sentiment(vdf, pd.DataFrame())
                compare_data.append({
                    "date": d,
                    "up_ratio": s["volume"]["up_ratio"],
                    "avg_pct": s["volume"]["avg_pct"],
                })

    # 输出
    if args.json:
        output = {
            "trade_date": trade_date,
            "top_n": args.top,
            "volume_top": volume_df.to_dict(orient="records") if not volume_df.empty else [],
            "concept_top": concept_df.to_dict(orient="records") if not concept_df.empty else [],
            "concept_bottom": bottom_df.to_dict(orient="records") if not bottom_df.empty else [],
            "sentiment": sentiment,
            "holdings_alignment": holdings_alignment,
            "compare": compare_data,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    else:
        report = generate_report(trade_date, volume_df, concept_df, bottom_df,
                                 sentiment, holdings_alignment, args.top)
        print(report)

        # 保存报告
        out_dir = os.path.join(project_root, "reports",
                               f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"市场情绪_TOP{args.top}_{trade_date}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📁 报告已保存: {out_path}")


if __name__ == "__main__":
    main()
