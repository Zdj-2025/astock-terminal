#!/usr/bin/env python3
"""策略实验室编排器 — 回测验证 + 493K→100万情景模拟。

串联: SignalBacktester + ScreenerBacktester + 情景模拟器
输出: 策略表现 + 情景分析 + 信号衰减排名
"""
import math
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
class ScenarioResult:
    name: str
    annual_return: float
    monthly_return: float
    months_to_target: int
    probability: float
    strategy: str
    position_style: str


@dataclass
class StrategyLabReport:
    trade_date: str
    scenarios: List[ScenarioResult] = field(default_factory=list)
    signal_ranking: List[Dict] = field(default_factory=list)
    backtest_summary: Dict = field(default_factory=dict)


class StrategyLab:
    """策略实验室编排器"""

    def __init__(self, backbone: Optional[DataBackbone] = None):
        self.bb = backbone or DataBackbone.get()

    def run_scenarios(self, current_capital: float = 493138.70,
                      target: float = 1_000_000,
                      months_left: int = 11) -> List[ScenarioResult]:
        """计算493K→100万多种情景"""
        scenarios = []

        # 场景A：牛市
        scenarios.append(self._calc_scenario(
            "牛市(AI浪潮持续)", current_capital, target, months_left,
            annual=0.60, prob=0.30,
            strategy="重仓AI主线，做T放大收益，金字塔加仓",
            position_style="70-80%仓位，单股可达20%"))

        # 场景B：基准
        scenarios.append(self._calc_scenario(
            "基准(需达标)", current_capital, target, months_left,
            annual=0.51, prob=0.35,
            strategy="保持AI仓位，释放D级资金到新机会，月度再平衡",
            position_style="60-70%仓位，严格遵守15%单股上限"))

        # 场景C：熊市
        scenarios.append(self._calc_scenario(
            "熊市(AI回调)", current_capital, target, months_left,
            annual=0.20, prob=0.25,
            strategy="降低仓位至50%，增加防御ETF，保留现金等机会",
            position_style="40-50%仓位，止损收紧至-5%"))

        # 场景D：最坏
        scenarios.append(self._calc_scenario(
            "最坏(保本)", current_capital, target, months_left,
            annual=0.05, prob=0.10,
            strategy="以守住利润为主，大幅减仓，等下一个大波段",
            position_style="20-30%仓位，只留核心持仓"))

        return scenarios

    def run_signal_validation(self, symbols: List[str] = None) -> List[Dict]:
        """验证各信号维度的预测能力"""
        if not symbols:
            symbols = ["603881", "512480", "512760", "159819", "300058",
                       "603617", "515120", "301060"]

        results = []
        for dim_name, dim_key in [("MA趋势", "ma"), ("MACD", "macd"),
                                   ("量价", "volume"), ("缠论/波浪", "chan_wave")]:
            results.append({
                "dimension": dim_name,
                "weight": {"ma": 30, "macd": 25, "volume": 20, "chan_wave": 25}[dim_key],
                "ic": 0.0,
                "win_rate": 0.0,
                "note": "需要回测数据填充",
            })
        return results

    def _calc_scenario(self, name, capital, target, months_left,
                       annual, prob, strategy, position_style) -> ScenarioResult:
        monthly = (1 + annual) ** (1/12) - 1
        months_needed = math.log(target / capital) / math.log(1 + monthly) if monthly > 0 else 999

        return ScenarioResult(
            name=name,
            annual_return=round(annual * 100, 1),
            monthly_return=round(monthly * 100, 2),
            months_to_target=int(months_needed),
            probability=prob,
            strategy=strategy,
            position_style=position_style,
        )

    def run_full_report(self) -> StrategyLabReport:
        trade_date = self.bb.get_latest_trade_date()
        report = StrategyLabReport(trade_date=trade_date)
        report.scenarios = self.run_scenarios()
        report.signal_ranking = self.run_signal_validation()
        return report
