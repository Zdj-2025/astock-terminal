#!/usr/bin/env python3
"""ETF-个股双向分析工具。

功能：
  1. 个股→ETF反向映射：给定一组强势股，找出它们集中出现在哪些ETF中
  2. ETF→成分股体检：给定ETF代码，评估其重仓股的技术面健康度

用法：
  # 个股反查ETF（输入多只股票，找出集中度最高的ETF）
  python scripts/etf/etf_stock_mapper.py --stocks 002993,300323,300831,300131,603297

  # ETF成分股体检
  python scripts/etf/etf_stock_mapper.py --etf 512480 --top 10

  # 我的ETF持仓体检
  python scripts/etf/etf_stock_mapper.py --my-etfs
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient


# 示例 ETF 持仓（截至 2026-05-18，模板参考）
MY_ETF_HOLDINGS = {
    "512760": {"name": "芯片ETF", "shares": 20000, "cost": 1.589},
    "159819": {"name": "AI智能ETF", "shares": 10000, "cost": 0.210},
}


class ETFStockMapper:
    """ETF-个股双向分析器"""

    def __init__(self):
        self.client = TushareClient()
        self._portfolio_cache: dict[str, pd.DataFrame] = {}

    def _get_etf_portfolio(self, ts_code: str) -> pd.DataFrame:
        """获取ETF成分股"""
        if ts_code in self._portfolio_cache:
            return self._portfolio_cache[ts_code]

        try:
            # 用 fund_portfolio 接口获取ETF持仓
            # Tushare 实际返回列: ts_code, ann_date, end_date, symbol, mkv, amount, stk_mkv_ratio, stk_float_ratio
            df = self.client.pro.fund_portfolio(ts_code=ts_code)
            if df is not None and not df.empty:
                # 统一列名映射，兼容下游代码
                rename_map = {}
                if "symbol" in df.columns:
                    rename_map["symbol"] = "hold_symbol"
                if "stk_mkv_ratio" in df.columns:
                    rename_map["stk_mkv_ratio"] = "hold_ratio"
                if "amount" in df.columns:
                    rename_map["amount"] = "hold_amount"
                if "mkv" in df.columns:
                    rename_map["mkv"] = "hold_mv"
                if rename_map:
                    df = df.rename(columns=rename_map)
                # 补 hold_name（API 不返回，留空）
                if "hold_name" not in df.columns:
                    df["hold_name"] = ""
                # hold_ratio 为 0 的用 hold_mv 近似
                # 取最新一期
                latest_date = df["end_date"].max()
                df = df[df["end_date"] == latest_date]
                self._portfolio_cache[ts_code] = df
                return df
        except Exception as e:
            print(f"  获取ETF {ts_code} 持仓失败: {e}")
        return pd.DataFrame()

    def _get_etf_list(self) -> pd.DataFrame:
        """获取ETF列表"""
        try:
            df = self.client.pro.fund(
                fund_type="ETF",
                fields="ts_code,name,list_status,list_date",
            )
            if df is not None and not df.empty:
                return df[df["list_status"] == "L"]
        except Exception:
            pass
        return pd.DataFrame()

    def stocks_to_etfs(self, stock_codes: list[str], min_hits: int = 2) -> pd.DataFrame:
        """个股→ETF反向映射：找出这些股票集中出现在哪些ETF中。

        Args:
            stock_codes: 股票代码列表（6位纯数字）
            min_hits: 至少命中几只才算有意义的ETF
        """
        print(f"  📊 个股→ETF反向映射")
        print(f"  输入: {len(stock_codes)} 只股票")
        print()

        # 先转成 ts_code 格式
        ts_codes = set()
        for code in stock_codes:
            if "." not in code:
                # 尝试 .SH 和 .SZ
                ts_codes.add(code + ".SZ")
                ts_codes.add(code + ".SH")
            else:
                ts_codes.add(code)

        # 获取ETF列表
        etf_list = self._get_etf_list()
        if etf_list.empty:
            print("  ❌ 无法获取ETF列表")
            return pd.DataFrame()

        print(f"  扫描 {len(etf_list)} 只ETF...")

        results = []
        checked = 0
        for _, etf_row in etf_list.iterrows():
            etf_code = etf_row["ts_code"]
            etf_name = etf_row["name"]

            # 跳过规模太小的ETF（名字里通常有特殊标识）
            try:
                portfolio = self._get_etf_portfolio(etf_code)
            except Exception:
                continue

            checked += 1
            if checked % 20 == 0:
                print(f"    已扫描 {checked}/{len(etf_list)}...", end="\r")

            if portfolio.empty:
                continue

            # 检查命中
            hits = []
            for _, holding in portfolio.iterrows():
                hold_symbol = str(holding.get("hold_symbol", ""))
                if hold_symbol in ts_codes or hold_symbol[:-3] in [c.split(".")[0] for c in ts_codes]:
                    hold_name = str(holding.get("hold_name", ""))
                    hold_ratio = float(holding.get("hold_ratio", 0))
                    hits.append({
                        "stock_code": hold_symbol,
                        "stock_name": hold_name,
                        "weight": hold_ratio,
                    })

            if len(hits) >= min_hits:
                total_weight = sum(h["weight"] for h in hits)
                results.append({
                    "etf_code": etf_code,
                    "etf_name": etf_name,
                    "hit_count": len(hits),
                    "hit_weight": round(total_weight, 2),
                    "hits": hits,
                })

        print(f"    扫描完成                          ")

        if not results:
            print(f"  未找到命中≥{min_hits}只的ETF")
            return pd.DataFrame()

        df = pd.DataFrame(results).sort_values("hit_count", ascending=False)
        return df.reset_index(drop=True)

    def etf_health_check(self, ts_code: str, top_n: int = 10) -> dict:
        """ETF成分股体检：评估重仓股的技术面健康度。

        Args:
            ts_code: ETF代码（如 512480.SH）
            top_n: 检查前N大重仓股

        Returns:
            健康度报告
        """
        print(f"  📊 ETF成分股体检: {ts_code}")

        portfolio = self._get_etf_portfolio(ts_code)
        if portfolio.empty:
            print(f"  ❌ 无法获取 {ts_code} 持仓")
            return {}

        # 按权重排序取TOP N
        portfolio = portfolio.sort_values("hold_ratio", ascending=False).head(top_n)

        total_weight = portfolio["hold_ratio"].sum()
        print(f"  TOP{top_n} 重仓股权重合计: {total_weight:.1f}%")

        # 逐只技术面评估
        health_scores = []
        for _, row in portfolio.iterrows():
            symbol = str(row["hold_symbol"])
            name = str(row.get("hold_name", ""))
            weight = float(row.get("hold_ratio", 0))

            score = self._quick_health_check(symbol)
            score["name"] = name
            score["weight"] = weight
            health_scores.append(score)

        # 汇总
        if not health_scores:
            return {}

        avg_rsi = np.mean([s.get("rsi", 50) for s in health_scores])
        n_overbought = sum(1 for s in health_scores if s.get("rsi", 50) > 70)
        n_oversold = sum(1 for s in health_scores if s.get("rsi", 50) < 30)
        n_uptrend = sum(1 for s in health_scores if s.get("ma_trend") == "多头")
        n_macd_gold = sum(1 for s in health_scores if s.get("macd") == "金叉")

        # 加权健康分
        weighted_score = sum(
            s.get("health", 50) * s["weight"] for s in health_scores if "weight" in s
        ) / max(total_weight, 1)

        report = {
            "etf_code": ts_code,
            "top_holdings": health_scores,
            "avg_rsi": round(avg_rsi, 1),
            "n_overbought": n_overbought,
            "n_oversold": n_oversold,
            "n_uptrend": n_uptrend,
            "n_macd_gold": n_macd_gold,
            "total_checked": len(health_scores),
            "weighted_health": round(weighted_score, 1),
        }

        return report

    def _quick_health_check(self, ts_code: str) -> dict:
        """快速技术面健康检查"""
        symbol = ts_code.split(".")[0] if "." in ts_code else ts_code
        result = {
            "ts_code": ts_code,
            "rsi": 50,
            "ma_trend": "未知",
            "macd": "未知",
            "boll_position": 50,
            "health": 50,
        }

        try:
            # 获取近120天日线
            end = self.client.get_recent_trade_day()
            start_dt = datetime.strptime(end, "%Y%m%d") - timedelta(days=180)
            start = start_dt.strftime("%Y%m%d")

            df = self.client.get_kline(symbol, freq="D", start=start, end=end)
            if df is None or len(df) < 60:
                return result

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

            # MA
            ma5 = close.rolling(5).mean().iloc[-1]
            ma10 = close.rolling(10).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20

            if ma5 > ma10 > ma20 > ma60:
                result["ma_trend"] = "多头"
            elif ma5 < ma10 < ma20:
                result["ma_trend"] = "空头"
            else:
                result["ma_trend"] = "震荡"

            # RSI(14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / max(loss.iloc[-1], 0.01)
            rsi = 100 - (100 / (1 + rs))
            result["rsi"] = round(rsi, 1)

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            result["macd"] = "金叉" if dif.iloc[-1] > dea.iloc[-1] else "死叉"

            # 布林位置
            mid = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            current = close.iloc[-1]
            if upper.iloc[-1] != lower.iloc[-1]:
                boll_pct = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
            else:
                boll_pct = 50
            result["boll_position"] = round(boll_pct, 0)

            # 健康分（0-100）
            health = 50
            # MA趋势加分
            if result["ma_trend"] == "多头":
                health += 20
            elif result["ma_trend"] == "震荡":
                health += 5
            else:
                health -= 15
            # MACD加分
            if result["macd"] == "金叉":
                health += 15
            else:
                health -= 10
            # RSI加分（50-65最佳区间）
            if 45 <= rsi <= 65:
                health += 15
            elif 30 <= rsi < 45:
                health += 10
            elif 65 < rsi <= 75:
                health += 5
            elif rsi > 75:
                health -= 10
            elif rsi < 30:
                health += 5  # 超卖可能是机会

            result["health"] = round(max(0, min(100, health)), 1)

        except Exception:
            pass

        return result

    @staticmethod
    def format_health_report(report: dict) -> str:
        """格式化ETF体检报告"""
        if not report:
            return "无体检数据"

        lines = []
        lines.append("=" * 65)
        lines.append(f"  📋 ETF成分股体检: {report['etf_code']}")
        lines.append("=" * 65)
        lines.append("")

        # 汇总
        lines.append(f"  综合健康分: {report['weighted_health']}/100")
        lines.append(f"  平均RSI: {report['avg_rsi']} | "
                      f"超买: {report['n_overbought']}只 | "
                      f"超卖: {report['n_oversold']}只")
        lines.append(f"  多头排列: {report['n_uptrend']}/{report['total_checked']}只 | "
                      f"MACD金叉: {report['n_macd_gold']}/{report['total_checked']}只")
        lines.append("")

        # 逐只
        lines.append(f"  {'名称':8} {'权重':>5} {'RSI':>5} {'趋势':6} {'MACD':6} {'布林':>5} {'健康分':>6}")
        lines.append("  " + "-" * 55)
        for s in report.get("top_holdings", []):
            lines.append(f"  {str(s.get('name',''))[:8]:8} "
                          f"{s.get('weight',0):5.1f}% "
                          f"{s.get('rsi',50):5.1f} "
                          f"{str(s.get('ma_trend','')):6} "
                          f"{str(s.get('macd','')):6} "
                          f"{s.get('boll_position',50):4.0f}% "
                          f"{s.get('health',50):6.1f}")

        lines.append("")

        # 评级
        wh = report["weighted_health"]
        if wh >= 70:
            lines.append("  🟢 健康状态：良好，成分股整体偏强")
        elif wh >= 50:
            lines.append("  🟡 健康状态：一般，部分成分股偏弱")
        else:
            lines.append("  🔴 健康状态：偏弱，成分股整体承压")

        return "\n".join(lines)

    @staticmethod
    def format_stock_to_etf(df: pd.DataFrame, stock_codes: list[str]) -> str:
        """格式化个股→ETF映射结果"""
        if df.empty:
            return "未找到集中度较高的ETF"

        lines = []
        lines.append("=" * 65)
        lines.append("  📊 个股→ETF 集中度映射")
        lines.append(f"  输入: {', '.join(stock_codes)}")
        lines.append("=" * 65)
        lines.append("")

        for _, row in df.head(15).iterrows():
            lines.append(f"  📦 {row['etf_code']} {row['etf_name']}")
            lines.append(f"     命中: {row['hit_count']}只 | 权重合计: {row['hit_weight']:.1f}%")
            for hit in row.get("hits", []):
                lines.append(f"     - {hit['stock_code']} {hit['stock_name']} ({hit['weight']:.1f}%)")
            lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ETF-个股双向分析")
    parser.add_argument("--stocks", type=str, help="股票代码，逗号分隔")
    parser.add_argument("--etf", type=str, help="ETF代码，如 512480")
    parser.add_argument("--my-etfs", action="store_true", help="体检我的ETF持仓")
    parser.add_argument("--min-hits", type=int, default=2, help="最小命中数")
    parser.add_argument("--top", type=int, default=10, help="ETF成分股TOP N")
    args = parser.parse_args()

    mapper = ETFStockMapper()

    if args.my_etfs:
        print("\n📋 我的ETF持仓体检")
        for code, info in MY_ETF_HOLDINGS.items():
            # 补全后缀
            if "." not in code:
                full_code = code + ".SH" if code.startswith("5") else code + ".SZ"
            else:
                full_code = code
            report = mapper.etf_health_check(full_code, top_n=args.top)
            print(ETFStockMapper.format_health_report(report))
            print(f"  我的持仓: {info['shares']}股 @ {info['cost']} | 现价需查")
            print()

    elif args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
        result = mapper.stocks_to_etfs(codes, min_hits=args.min_hits)
        print(ETFStockMapper.format_stock_to_etf(result, codes))

    elif args.etf:
        code = args.etf
        if "." not in code:
            full_code = code + ".SH" if code.startswith("5") else code + ".SZ"
        else:
            full_code = code
        report = mapper.etf_health_check(full_code, top_n=args.top)
        print(ETFStockMapper.format_health_report(report))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
