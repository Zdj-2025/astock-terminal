#!/usr/bin/env python3
"""企业深度分析模块。

对筛选后的TOP候选做综合评判：
- 收入/利润趋势分析（近4期）
- 现金流分析（经营/投资/筹资 + FCF）
- 风险检测（商誉/应收/存货/质押/现金流/负债）
- 综合评价
"""
from __future__ import annotations

import numpy as np
from typing import Optional

import pandas as pd

from scripts.core.tushare_client import TushareClient


class CompanyAnalyzer:
    """企业深度分析器"""

    def __init__(self, client: TushareClient):
        self.client = client

    def analyze(self, ts_code: str) -> dict:
        """对单只股票做深度分析"""
        symbol = ts_code.split(".")[0]

        # 基本信息
        info = self.client.get_stock_info(symbol)

        # 财务数据
        fina = self.client.get_financial(ts_code)
        income = self.client.get_income(ts_code)
        balance = self.client.get_balance_sheet(ts_code)
        cashflow = self.client.get_cashflow(ts_code)

        # 风险数据
        pledge = self.client.get_pledge(symbol)
        forecast = self.client.get_forecast(symbol)

        # 收入分析
        revenue = self._analyze_revenue(income, fina)

        # 利润分析
        profit = self._analyze_profit(income, fina, cashflow)

        # 现金流分析
        cf = self._analyze_cashflow(cashflow)

        # 资产负债分析
        balance_analysis = self._analyze_balance(balance)

        # 风险检测
        risks = self._detect_risks(
            fina, balance, cashflow, pledge, income, revenue
        )

        # 亮点
        highlights = self._find_highlights(fina, revenue, profit, cf)

        # 综合评价
        summary = self._make_summary(info, revenue, profit, cf, risks)

        return {
            "ts_code": ts_code,
            "name": info.get("name", ""),
            "industry": info.get("industry", ""),
            "summary": summary,
            "business": f"{info.get('name', '')}（{info.get('industry', '未知')}行业）",
            "revenue": revenue,
            "profit": profit,
            "cashflow": cf,
            "balance": balance_analysis,
            "risks": risks,
            "highlights": highlights,
            "forecast": self._parse_forecast(forecast, ts_code),
        }

    def _analyze_revenue(self, income: pd.DataFrame,
                          fina: pd.DataFrame) -> dict:
        """收入分析"""
        trend = []
        growth_latest = float("nan")

        if not fina.empty:
            periods = fina.head(4)
            for _, row in periods.iterrows():
                rev = self._safe(row.get("revenue"))
                end_date = str(row.get("end_date", ""))
                if rev and not np.isnan(rev):
                    trend.append({
                        "period": end_date,
                        "revenue": round(rev / 1e8, 2),  # 亿元
                    })

            latest = fina.iloc[0]
            growth_latest = self._safe_pct(latest.get("or_yoy"))

        # 判断稳定性
        if len(trend) >= 3:
            vals = [t["revenue"] for t in trend if t["revenue"]]
            if all(v > 0 for v in vals):
                changes = [(vals[i] - vals[i+1]) / vals[i+1]
                           for i in range(len(vals)-1)]
                if all(c >= 0 for c in changes):
                    stability = "稳定增长"
                elif any(c < -0.1 for c in changes):
                    stability = "波动明显"
                else:
                    stability = "小幅波动"
            else:
                stability = "有负值"
        else:
            stability = "数据不足"

        return {
            "trend": trend[:4],
            "growth_latest": growth_latest,
            "stability": stability,
        }

    def _analyze_profit(self, income: pd.DataFrame,
                         fina: pd.DataFrame,
                         cashflow: pd.DataFrame) -> dict:
        """利润分析"""
        trend = []
        margin_latest = float("nan")

        if not fina.empty:
            periods = fina.head(4)
            for _, row in periods.iterrows():
                n_income = self._safe(row.get("n_income"))
                end_date = str(row.get("end_date", ""))
                if n_income and not np.isnan(n_income):
                    trend.append({
                        "period": end_date,
                        "net_profit": round(n_income / 1e8, 2),
                    })

            latest = fina.iloc[0]
            margin_latest = self._safe_pct(latest.get("netprofit_margin"))

        # 盈利质量
        quality = self._assess_profit_quality(fina, cashflow)

        return {
            "trend": trend[:4],
            "margin_latest": margin_latest,
            "quality": quality,
        }

    def _analyze_cashflow(self, cashflow: pd.DataFrame) -> dict:
        """现金流分析"""
        if cashflow.empty:
            return {
                "operating_cf": None,
                "investing_cf": None,
                "financing_cf": None,
                "fcf": None,
                "status": "数据不足",
            }

        latest = cashflow.iloc[0]
        ocf = self._safe(latest.get("n_cashflow_act"))
        icf = self._safe(latest.get("n_cashflow_inv_act"))
        fcf_val = None
        if ocf is not None and icf is not None:
            fcf_val = round((ocf + icf) / 1e8, 2)

        ocf_yi = round(ocf / 1e8, 2) if ocf else None
        icf_yi = round(icf / 1e8, 2) if icf else None

        # 状态判断
        if ocf_yi is not None:
            if ocf_yi > 0 and (fcf_val is not None and fcf_val > 0):
                status = "健康"
            elif ocf_yi > 0:
                status = "一般（投资支出大）"
            else:
                status = "紧张（经营现金流为负）"
        else:
            status = "数据不足"

        return {
            "operating_cf": ocf_yi,
            "investing_cf": icf_yi,
            "fcf": fcf_val,
            "status": status,
        }

    def _analyze_balance(self, balance: pd.DataFrame) -> dict:
        """资产负债关键指标"""
        if balance.empty:
            return {}

        latest = balance.iloc[0]
        result = {}

        goodwill = self._safe(latest.get("goodwill"))
        total_equity = self._safe(latest.get("total_hldr_eqy_exc_min_int"))
        if goodwill and total_equity and total_equity > 0:
            result["goodwill_ratio"] = round(goodwill / total_equity * 100, 1)

        accounts_rec = self._safe(latest.get("accounts_receiv"))
        total_assets = self._safe(latest.get("total_assets"))
        if accounts_rec and total_assets and total_assets > 0:
            result["accounts_rec_ratio"] = round(
                accounts_rec / total_assets * 100, 1)

        cash = self._safe(latest.get("monetary_capital"))
        if cash and total_assets and total_assets > 0:
            result["cash_ratio"] = round(cash / total_assets * 100, 1)

        inventory = self._safe(latest.get("inventory"))
        if inventory and total_assets and total_assets > 0:
            result["inventory_ratio"] = round(
                inventory / total_assets * 100, 1)

        return result

    def _detect_risks(self, fina: pd.DataFrame,
                      balance: pd.DataFrame,
                      cashflow: pd.DataFrame,
                      pledge: pd.DataFrame,
                      income: pd.DataFrame,
                      revenue_data: dict) -> list:
        """风险检测"""
        risks = []

        # 1. 商誉风险
        if not balance.empty:
            latest = balance.iloc[0]
            goodwill = self._safe(latest.get("goodwill"))
            equity = self._safe(latest.get("total_hldr_eqy_exc_min_int"))
            if (goodwill and equity and equity > 0
                    and goodwill / equity > 0.3):
                risks.append({
                    "level": "high",
                    "desc": f"商誉占净资产{goodwill/equity*100:.0f}%，存在减值风险",
                })

        # 2. 质押风险
        if not pledge.empty:
            row = pledge.iloc[0]
            pledge_ratio = self._safe(row.get("pledge_ratio"))
            if pledge_ratio and pledge_ratio > 30:
                risks.append({
                    "level": "high",
                    "desc": f"股权质押比例{pledge_ratio:.0f}%，控制权风险",
                })

        # 3. 现金流质量
        if not fina.empty and not cashflow.empty:
            n_income = self._safe(fina.iloc[0].get("n_income"))
            ocf = self._safe(cashflow.iloc[0].get("n_cashflow_act"))
            if (n_income and n_income > 0 and ocf is not None
                    and ocf / n_income < 0.5):
                risks.append({
                    "level": "medium",
                    "desc": f"经营现金流/净利润={ocf/n_income:.1f}，利润含金量不足",
                })

        # 4. 负债风险
        if not fina.empty:
            debt = self._safe(fina.iloc[0].get("debt_to_assets"))
            if debt and debt > 60:
                risks.append({
                    "level": "medium",
                    "desc": f"资产负债率{debt:.0f}%，偏高",
                })

        # 5. 增速下滑
        if not fina.empty and len(fina) >= 2:
            latest_growth = self._safe(fina.iloc[0].get("or_yoy"))
            prev_growth = self._safe(fina.iloc[1].get("or_yoy"))
            if (latest_growth is not None and prev_growth is not None
                    and not np.isnan(latest_growth)
                    and not np.isnan(prev_growth)
                    and latest_growth < prev_growth - 20):
                risks.append({
                    "level": "medium",
                    "desc": f"营收增速从{prev_growth:.0f}%降至{latest_growth:.0f}%，增长放缓",
                })

        # 6. 利润与现金流背离
        if not fina.empty and not cashflow.empty:
            n_income = self._safe(fina.iloc[0].get("n_income"))
            ocf = self._safe(cashflow.iloc[0].get("n_cashflow_act"))
            if len(fina) >= 2:
                prev_income = self._safe(fina.iloc[1].get("n_income"))
                if (n_income and prev_income and n_income > prev_income
                        and ocf is not None and len(cashflow) >= 2):
                    prev_ocf = self._safe(cashflow.iloc[1].get(
                        "n_cashflow_act"))
                    if prev_ocf and ocf < prev_ocf:
                        risks.append({
                            "level": "low",
                            "desc": "净利润增长但经营现金流下降，关注盈利质量",
                        })

        # 7. 应收账款风险
        if not balance.empty and not fina.empty:
            latest = balance.iloc[0]
            ar = self._safe(latest.get("accounts_receiv"))
            if ar:
                rev_growth = self._safe(fina.iloc[0].get("or_yoy"))
                if (len(balance) >= 2 and rev_growth is not None
                        and not np.isnan(rev_growth)):
                    prev_ar = self._safe(balance.iloc[1].get(
                        "accounts_receiv"))
                    if prev_ar and prev_ar > 0:
                        ar_growth = (ar - prev_ar) / prev_ar * 100
                        if ar_growth > max(rev_growth * 1.5, 30):
                            risks.append({
                                "level": "medium",
                                "desc": f"应收增速({ar_growth:.0f}%)远超营收增速({rev_growth:.0f}%)",
                            })

        return risks

    def _find_highlights(self, fina: pd.DataFrame,
                          revenue: dict, profit: dict,
                          cf: dict) -> list:
        """亮点发现"""
        highlights = []

        # 连续正增长
        if len(revenue["trend"]) >= 3:
            vals = [t["revenue"] for t in revenue["trend"]]
            if all(v > 0 for v in vals):
                changes = [vals[i] > vals[i+1]
                           for i in range(len(vals)-1)]
                if all(changes):
                    highlights.append("连续多期营收正增长")

        # 经营现金流为正
        if cf.get("operating_cf") and cf["operating_cf"] > 0:
            highlights.append("经营现金流为正")
            if cf.get("fcf") and cf["fcf"] > 0:
                highlights.append("自由现金流为正")

        # 高ROE
        if not fina.empty:
            roe = self._safe(fina.iloc[0].get("roe"))
            if roe and roe > 15:
                highlights.append(f"ROE={roe:.1f}%，盈利能力优秀")

        # 低负债
        if not fina.empty:
            debt = self._safe(fina.iloc[0].get("debt_to_assets"))
            if debt and debt < 40:
                highlights.append(f"负债率{debt:.0f}%，财务稳健")

        return highlights

    def _make_summary(self, info: dict, revenue: dict,
                       profit: dict, cf: dict, risks: list) -> str:
        """生成综合评价"""
        parts = []
        name = info.get("name", "")
        industry = info.get("industry", "")

        if industry:
            parts.append(f"{industry}行业")

        growth = revenue.get("growth_latest")
        if growth is not None and not np.isnan(growth):
            if growth > 30:
                parts.append("高增长")
            elif growth > 10:
                parts.append("稳健增长")
            elif growth > 0:
                parts.append("低速增长")
            else:
                parts.append("增长承压")

        cf_status = cf.get("status", "")
        if "健康" in cf_status:
            parts.append("现金流健康")
        elif "紧张" in cf_status:
            parts.append("现金流紧张")

        high_risks = [r for r in risks if r["level"] == "high"]
        if high_risks:
            parts.append(f"存在{len(high_risks)}个高风险项")

        return f"{name}：{'，'.join(parts)}" if parts else f"{name}：数据不足"

    def _assess_profit_quality(self, fina: pd.DataFrame,
                                cashflow: pd.DataFrame) -> str:
        """盈利质量评估"""
        if fina.empty or cashflow.empty:
            return "数据不足"

        n_income = self._safe(fina.iloc[0].get("n_income"))
        ocf = self._safe(cashflow.iloc[0].get("n_cashflow_act"))

        if n_income is None or ocf is None:
            return "数据不足"

        if n_income <= 0:
            return "亏损状态"

        ratio = ocf / n_income if n_income > 0 else 0
        if ratio >= 0.8:
            return "盈利质量高（现金流匹配）"
        elif ratio >= 0.5:
            return "盈利质量一般"
        elif ratio >= 0:
            return "盈利质量偏低（现金流不足）"
        else:
            return "盈利质量差（经营现金流为负）"

    def _parse_forecast(self, forecast: pd.DataFrame,
                         ts_code: str) -> str:
        """解析业绩预告"""
        if forecast.empty:
            return ""
        row = forecast.iloc[0]
        summary = str(row.get("summary", ""))
        type_desc = str(row.get("type", ""))
        if summary and summary != "nan" and summary != "None":
            return f"业绩预告: {summary}"
        return ""

    @staticmethod
    def _safe(val) -> Optional[float]:
        if val is None:
            return None
        try:
            v = float(val)
            return v if not np.isnan(v) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_pct(val) -> Optional[float]:
        v = CompanyAnalyzer._safe(val)
        return round(v, 1) if v is not None else None
