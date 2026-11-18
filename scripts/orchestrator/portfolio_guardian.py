#!/usr/bin/env python3
"""持仓守护 — 持续监控所有持仓，触发可操作警报。

编排: DisciplineChecker + CapitalEfficiencyAnalyzer + DecisionEngine
输出: GuardianReport（警报、持仓状态、效率分布、T+0机会、健康分）
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.orchestrator.data_backbone import DataBackbone


@dataclass
class PositionAlert:
    severity: str          # red / yellow / blue
    code: str
    name: str
    message: str
    action: str            # 立即止损 / 考虑止盈 / 关注 / 无


@dataclass
class PositionStatus:
    code: str
    name: str
    action: str            # buy / sell / hold
    action_text: str       # 买入 / 卖出 / 观望
    rating: str            # 强烈推荐 / 推荐 / 中性 / 谨慎 / 不推荐
    overall_score: float
    stop_loss: float
    take_profit: float
    current_price: float
    pnl_pct: float
    market_value: float
    position_pct: float
    discipline_violations: List[dict] = field(default_factory=list)
    efficiency_grade: str = ""
    efficiency_score: float = 0.0


@dataclass
class GuardianReport:
    trade_date: str
    generated_at: str
    total_capital: float = 0.0
    total_market_value: float = 0.0
    cash: float = 0.0
    position_ratio: float = 0.0
    portfolio_health_score: int = 0
    alerts: List[PositionAlert] = field(default_factory=list)
    positions: List[PositionStatus] = field(default_factory=list)
    efficiency_distribution: Dict[str, int] = field(default_factory=dict)
    t_opportunities: List[dict] = field(default_factory=list)
    target_progress: Dict = field(default_factory=dict)


class PortfolioGuardian:
    """持仓守护编排器"""

    def __init__(self, backbone: Optional[DataBackbone] = None,
                 use_decision_engine: bool = False):
        self.bb = backbone or DataBackbone.get()
        self._use_decision_engine = use_decision_engine

    def run(self, trade_date: str = None) -> GuardianReport:
        """执行完整的持仓守护流程"""
        if not trade_date:
            trade_date = self.bb.get_latest_trade_date()

        report = GuardianReport(
            trade_date=trade_date,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 1. 加载持仓和资金
        portfolio = self.bb.load_portfolio()
        capital = self.bb.load_capital()
        if portfolio is None or portfolio.empty:
            report.portfolio_health_score = 0
            return report

        total_capital = self._get_capital_value(capital, "总资金") or 493138.70
        cash = self._get_capital_value(capital, "可用资金") or 190919.60
        report.total_capital = total_capital
        report.cash = cash

        # 2. 逐只持仓分析
        total_mv = 0.0
        efficiency_dist = {"A": 0, "B": 0, "C": 0, "D": 0}
        total_health = 0

        for _, row in portfolio.iterrows():
            code = str(row.get("股票代码", "")).strip()
            name = str(row.get("股票名称", "")).strip()
            if not code:
                continue

            cost = float(row.get("买入价格", 0) or 0)
            shares = int(row.get("持仓数量", 0) or 0)
            sl = float(row.get("止损价", 0) or 0)
            tp = float(row.get("止盈价", 0) or 0)

            # 获取当前价格
            price = self._get_current_price(code, trade_date)
            if price <= 0:
                continue

            mv = price * shares
            total_mv += mv
            pnl_pct = (price / cost - 1) * 100 if cost > 0 else 0
            pos_pct = mv / total_capital * 100 if total_capital > 0 else 0

            # 纪律检查
            violations = self._check_discipline(code, name, cost, shares, price,
                                                sl, tp, pos_pct, total_capital)

            # 资金效率
            eff = self._calc_efficiency(code, name, cost, price, shares, mv, pnl_pct)

            # 决策信号
            action, rating, score = self._get_decision_signal(code, name)

            # 警报
            alerts = self._generate_alerts(code, name, price, sl, tp, pnl_pct,
                                           pos_pct, violations, eff)

            pos = PositionStatus(
                code=code, name=name, action=action, action_text=action,
                rating=rating, overall_score=score,
                stop_loss=sl, take_profit=tp, current_price=price,
                pnl_pct=round(pnl_pct, 2), market_value=round(mv, 2),
                position_pct=round(pos_pct, 1),
                discipline_violations=violations,
                efficiency_grade=eff.get("grade", ""),
                efficiency_score=eff.get("score", 0),
            )
            report.positions.append(pos)
            report.alerts.extend(alerts)

            grade = eff.get("grade", "D")
            efficiency_dist[grade] = efficiency_dist.get(grade, 0) + 1

            # 每只持仓贡献的健康分
            health = 100
            if violations:
                red = sum(1 for v in violations if v.get("severity") == "red")
                yellow = sum(1 for v in violations if v.get("severity") == "yellow")
                health -= red * 30 + yellow * 10
            if grade == "D":
                health -= 20
            elif grade == "C":
                health -= 10
            total_health += max(0, health)

        report.total_market_value = round(total_mv, 2)
        report.position_ratio = round(total_mv / total_capital * 100, 1) if total_capital > 0 else 0
        report.efficiency_distribution = efficiency_dist
        report.portfolio_health_score = int(total_health / max(len(report.positions), 1))

        # 3. T+0机会（简化版：检查60min MACD金叉）
        report.t_opportunities = self._find_t_opportunities(report.positions)

        # 4. 目标进度
        target = 1_000_000
        report.target_progress = {
            "current": round(total_capital, 2),
            "target": target,
            "progress_pct": round(total_capital / target * 100, 1),
            "gap": round(target - total_capital, 2),
            "monthly_target_pct": self._calc_monthly_target(total_capital, target),
        }

        return report

    def _get_capital_value(self, capital_df, field_name: str) -> Optional[float]:
        if capital_df is None:
            return None
        for _, row in capital_df.iterrows():
            key = str(row.get("项目", row.iloc[0] if len(row) > 0 else ""))
            if field_name in key:
                val = row.get("金额/数值", row.iloc[1] if len(row) > 1 else None)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
        return None

    def _get_current_price(self, code: str, trade_date: str) -> float:
        ts_code = self._to_ts_code(code)
        cached = self.bb.get_cached(f"price_{ts_code}_{trade_date}", 1800)
        if cached is not None:
            return cached

        is_etf = code.startswith(("51", "15", "16", "56"))
        if is_etf:
            df = self.bb.ts.get_fund_daily(ts_code, start=trade_date,
                                           end=trade_date)
        else:
            df = self.bb.ts.get_kline(code, start=trade_date, end=trade_date)

        if df is not None and not df.empty:
            price = float(df.iloc[-1]["close"])
            self.bb.set_cached(f"price_{ts_code}_{trade_date}", price)
            return price

        df_daily = self.bb.get_market_data(trade_date)
        if df_daily is not None and not df_daily.empty:
            match = df_daily[df_daily["ts_code"] == ts_code]
            if not match.empty:
                price = float(match.iloc[0]["close"])
                self.bb.set_cached(f"price_{ts_code}_{trade_date}", price)
                return price
        return 0.0

    def _to_ts_code(self, code: str) -> str:
        code = str(code).strip()
        if "." in code:
            return code
        if code.startswith(("6", "5")):
            return f"{code}.SH"
        return f"{code}.SZ"

    def _check_discipline(self, code, name, cost, shares, price, sl, tp,
                          pos_pct, total_capital) -> List[dict]:
        from scripts.analyzers.discipline_checker import DisciplineChecker
        checker = DisciplineChecker()
        holding = {
            "code": code, "name": name, "cost": cost, "shares": shares,
            "stop_loss": sl, "take_profit": tp, "buy_date": ""
        }
        result = checker.check(holding, price, all_holdings=[], total_capital=total_capital)
        violations = []
        for v in result.get("violations", []):
            violations.append({"severity": "red", **v})
        for w in result.get("warnings", []):
            violations.append({"severity": "yellow", **w})
        return violations

    def _calc_efficiency(self, code, name, cost, price, shares, mv, pnl_pct) -> dict:
        from scripts.analyzers.capital_efficiency import CapitalEfficiencyAnalyzer
        analyzer = CapitalEfficiencyAnalyzer()
        holding = {
            "code": code, "name": name, "cost": cost, "price": price,
            "shares": shares, "market_value": mv, "pnl_pct": pnl_pct,
        }
        result = analyzer.score_position(holding)
        return {
            "score": result.get("score", 0),
            "grade": result.get("grade", "D"),
        }

    def _get_decision_signal(self, code, name) -> tuple:
        if not self._use_decision_engine:
            return ("hold", "中性", 50.0)
        try:
            from scripts.decision.decision_executor import DecisionExecutor
            executor = DecisionExecutor()
            decision = executor.engine.decide(code, name,
                                              skip_macd_resonance=True)
            return (decision.action, decision.rating, decision.overall_score)
        except Exception:
            return ("hold", "中性", 50.0)

    def _generate_alerts(self, code, name, price, sl, tp, pnl_pct, pos_pct,
                         violations, eff) -> List[PositionAlert]:
        alerts = []
        grade = eff.get("grade", "")

        # 红色警报
        if sl > 0 and price <= sl:
            alerts.append(PositionAlert("red", code, name, f"止损触发！{price:.2f}≤{sl:.2f}", "立即止损"))
        if pos_pct > 15:
            alerts.append(PositionAlert("yellow", code, name, f"仓位{pos_pct:.1f}%超15%上限", "考虑减仓或稀释"))

        # 黄色警报
        if tp > 0 and price >= tp:
            alerts.append(PositionAlert("yellow", code, name, f"止盈触发！{price:.2f}≥{tp:.2f}", "考虑止盈"))
        if grade == "D":
            alerts.append(PositionAlert("yellow", code, name, f"资金效率D级", "建议轮动"))
        if pnl_pct > 50 and pnl_pct < 100:
            alerts.append(PositionAlert("blue", code, name, f"盈利{pnl_pct:.1f}%", "设跟踪止损"))
        if pnl_pct >= 100:
            alerts.append(PositionAlert("yellow", code, name, f"盈利{pnl_pct:.1f}%翻倍", "强烈建议减仓锁利"))

        for v in violations:
            sev = v.get("severity", "blue")
            msg = v.get("message", v.get("rule", str(v)))
            alerts.append(PositionAlert(sev, code, name, str(msg)[:80], "关注"))

        return alerts

    def _find_t_opportunities(self, positions: List[PositionStatus]) -> List[dict]:
        """简化版T+0机会扫描：检查持仓是否有短线信号"""
        opportunities = []
        for pos in positions:
            if pos.efficiency_grade in ("A", "B") and pos.pnl_pct > 10:
                opportunities.append({
                    "code": pos.code, "name": pos.name,
                    "reason": f"底仓盈利{pos.pnl_pct:.0f}%，可做T降本",
                    "signal": "需运行 tech_analysis_cli 检查60min MACD",
                })
        return opportunities

    def _calc_monthly_target(self, current: float, target: float) -> float:
        """计算达到目标所需的月均收益率"""
        months_left = 11
        if months_left <= 0 or current <= 0:
            return 0
        ratio = (target / current) ** (1 / months_left) - 1
        return round(ratio * 100, 2)
