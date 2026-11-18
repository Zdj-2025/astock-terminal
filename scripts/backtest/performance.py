#!/usr/bin/env python3
"""
绩效计算器 - 计算回测绩效指标

核心指标:
1. 收益指标: 总收益率、年化收益率、超额收益
2. 风险指标: 最大回撤、波动率、下行风险
3. 风险调整收益: 夏普比率、索提诺比率、卡玛比率
4. 交易质量: 胜率、盈亏比、交易频率
5. 基准对比: Alpha、Beta、信息比率
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class PerformanceCalculator:
    """
    绩效计算器

    计算15+核心绩效指标，全面评估策略表现
    """

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化绩效计算器

        参数:
            risk_free_rate: 无风险利率（年化，默认3%）
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252  # 年交易日数

        print("📊 初始化绩效计算器")
        print(f"   无风险利率: {risk_free_rate:.2%}")

    def calculate_all_metrics(
        self,
        portfolio_values: List[float],
        trade_history: List[Dict],
        benchmark_values: Optional[List[float]] = None,
        dates: Optional[List[datetime]] = None
    ) -> Dict:
        """
        计算所有绩效指标

        参数:
            portfolio_values: 组合净值列表
            trade_history: 交易记录列表
            benchmark_values: 基准净值列表（可选）
            dates: 日期列表（可选）

        返回:
            完整的绩效指标字典
        """
        print("\n📊 计算绩效指标...")

        metrics = {}

        # 1. 收益指标
        metrics['returns'] = self._calculate_return_metrics(portfolio_values)

        # 2. 风险指标
        metrics['risk'] = self._calculate_risk_metrics(portfolio_values)

        # 3. 风险调整收益指标
        metrics['risk_adjusted'] = self._calculate_risk_adjusted_metrics(
            portfolio_values
        )

        # 4. 交易质量指标
        if trade_history:
            metrics['trading'] = self._calculate_trading_metrics(trade_history)
        else:
            metrics['trading'] = {}

        # 5. 基准对比
        if benchmark_values:
            metrics['benchmark'] = self._calculate_benchmark_metrics(
                portfolio_values,
                benchmark_values
            )
        else:
            metrics['benchmark'] = {}

        print("✅ 绩效指标计算完成")
        return metrics

    def _calculate_return_metrics(self, portfolio_values: List[float]) -> Dict:
        """计算收益指标"""
        try:
            if len(portfolio_values) < 2:
                return {}

            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]

            # 总收益率
            total_return = (final_value - initial_value) / initial_value

            # 年化收益率
            days = len(portfolio_values)
            years = days / self.trading_days_per_year
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            # 月度收益率
            monthly_returns = self._calculate_monthly_returns(portfolio_values)

            # 日收益率
            daily_returns = self._calculate_daily_returns(portfolio_values)

            # 平均日收益
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0

            # 累计收益率曲线
            cumulative_return_curve = [
                (v - initial_value) / initial_value for v in portfolio_values
            ]

            return {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': annualized_return,
                'annualized_return_pct': annualized_return * 100,
                'avg_daily_return': avg_daily_return,
                'avg_daily_return_pct': avg_daily_return * 100,
                'monthly_returns': monthly_returns,
                'cumulative_return_curve': cumulative_return_curve,
                'positive_days': sum(1 for r in daily_returns if r > 0),
                'negative_days': sum(1 for r in daily_returns if r < 0)
            }

        except Exception as e:
            print(f"   ⚠️  收益指标计算失败: {str(e)[:50]}")
            return {}

    def _calculate_risk_metrics(self, portfolio_values: List[float]) -> Dict:
        """计算风险指标"""
        try:
            if len(portfolio_values) < 2:
                return {}

            # 日收益率
            daily_returns = self._calculate_daily_returns(portfolio_values)

            # 最大回撤
            max_drawdown, max_dd_start, max_dd_end = self._calculate_max_drawdown(
                portfolio_values
            )

            # 波动率（年化）
            volatility = np.std(daily_returns) * np.sqrt(self.trading_days_per_year)

            # 下行风险（只考虑负收益）
            negative_returns = [r for r in daily_returns if r < 0]
            downside_risk = np.std(negative_returns) * np.sqrt(self.trading_days_per_year) if negative_returns else 0

            # VaR（95%置信度）
            var_95 = np.percentile(daily_returns, 5) if daily_returns else 0

            # CVaR（条件风险价值）
            var_threshold = np.percentile(daily_returns, 5) if daily_returns else 0
            cvar = np.mean([r for r in daily_returns if r <= var_threshold]) if daily_returns else 0

            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'max_dd_start': max_dd_start,
                'max_dd_end': max_dd_end,
                'volatility': volatility,
                'volatility_pct': volatility * 100,
                'downside_risk': downside_risk,
                'downside_risk_pct': downside_risk * 100,
                'var_95': var_95,
                'var_95_pct': var_95 * 100,
                'cvar': cvar,
                'cvar_pct': cvar * 100
            }

        except Exception as e:
            print(f"   ⚠️  风险指标计算失败: {str(e)[:50]}")
            return {}

    def _calculate_risk_adjusted_metrics(
        self,
        portfolio_values: List[float]
    ) -> Dict:
        """计算风险调整收益指标"""
        try:
            if len(portfolio_values) < 2:
                return {}

            daily_returns = self._calculate_daily_returns(portfolio_values)

            # 年化收益率
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            years = len(portfolio_values) / self.trading_days_per_year
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            # 波动率
            volatility = np.std(daily_returns) * np.sqrt(self.trading_days_per_year)

            # 夏普比率
            daily_rf = self.risk_free_rate / self.trading_days_per_year
            excess_returns = [r - daily_rf for r in daily_returns]
            sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year)) if excess_returns else 0

            # 索提诺比率（只考虑下行风险）
            negative_returns = [r - daily_rf for r in daily_returns if r < daily_rf]
            downside_std = np.std(negative_returns) if negative_returns else 0.0001
            sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(self.trading_days_per_year)) if excess_returns else 0

            # 卡玛比率（年化收益/最大回撤）
            max_drawdown = self._calculate_max_drawdown(portfolio_values)[0]
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            }

        except Exception as e:
            print(f"   ⚠️  风险调整收益指标计算失败: {str(e)[:50]}")
            return {}

    def _calculate_trading_metrics(self, trade_history: List[Dict]) -> Dict:
        """计算交易质量指标"""
        try:
            if not trade_history:
                return {}

            # 统计买入和卖出交易
            buys = [t for t in trade_history if t.get('quantity', 0) > 0]
            sells = [t for t in trade_history if t.get('quantity', 0) < 0]

            # 计算盈亏（需要配对买卖）
            profits = []
            for i, buy in enumerate(buys):
                if i < len(sells):
                    sell = sells[i]
                    profit = (sell['price'] - buy['price']) * abs(buy['quantity'])
                    profits.append(profit)

            # 胜率
            wins = sum(1 for p in profits if p > 0)
            total_trades = len(profits)
            win_rate = wins / total_trades if total_trades > 0 else 0

            # 盈亏比
            avg_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
            avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 0.0001
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

            # 交易频率
            if trade_history:
                first_date = trade_history[0].get('date')
                last_date = trade_history[-1].get('date')
                if first_date and last_date:
                    days = (last_date - first_date).days if hasattr(first_date, 'days') else len(trade_history)
                    trade_frequency = len(trade_history) / (days / 30) if days > 0 else 0  # 月均交易次数
                else:
                    trade_frequency = 0
            else:
                trade_frequency = 0

            return {
                'total_trades': len(trade_history),
                'buy_trades': len(buys),
                'sell_trades': len(sells),
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_loss_ratio': profit_loss_ratio,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_profit': sum([p for p in profits if p > 0]),
                'total_loss': sum([p for p in profits if p < 0]),
                'trade_frequency_monthly': trade_frequency
            }

        except Exception as e:
            print(f"   ⚠️  交易质量指标计算失败: {str(e)[:50]}")
            return {}

    def _calculate_benchmark_metrics(
        self,
        portfolio_values: List[float],
        benchmark_values: List[float]
    ) -> Dict:
        """计算基准对比指标"""
        try:
            if len(portfolio_values) < 2 or len(benchmark_values) < 2:
                return {}

            # 组合收益率
            portfolio_returns = self._calculate_daily_returns(portfolio_values)
            portfolio_total = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

            # 基准收益率
            benchmark_returns = self._calculate_daily_returns(benchmark_values)
            benchmark_total = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]

            # Alpha（超额收益）
            alpha = portfolio_total - benchmark_total

            # Beta（系统性风险）
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # 信息比率
            excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
            tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year) if excess_returns else 0
            information_ratio = (alpha / tracking_error) if tracking_error > 0 else 0

            return {
                'portfolio_return': portfolio_total,
                'portfolio_return_pct': portfolio_total * 100,
                'benchmark_return': benchmark_total,
                'benchmark_return_pct': benchmark_total * 100,
                'alpha': alpha,
                'alpha_pct': alpha * 100,
                'beta': beta,
                'tracking_error': tracking_error,
                'tracking_error_pct': tracking_error * 100,
                'information_ratio': information_ratio
            }

        except Exception as e:
            print(f"   ⚠️  基准对比指标计算失败: {str(e)[:50]}")
            return {}

    def _calculate_daily_returns(self, values: List[float]) -> List[float]:
        """计算日收益率"""
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        return returns

    def _calculate_monthly_returns(self, values: List[float]) -> List[float]:
        """计算月度收益率（简化版，按20个交易日为一个月）"""
        monthly_returns = []
        for i in range(20, len(values), 20):
            if values[i-20] != 0:
                ret = (values[i] - values[i-20]) / values[i-20]
                monthly_returns.append(ret)
        return monthly_returns

    def _calculate_max_drawdown(
        self,
        values: List[float]
    ) -> Tuple[float, int, int]:
        """
        计算最大回撤

        返回:
            (max_drawdown, start_index, end_index)
        """
        if len(values) < 2:
            return 0.0, 0, 0

        peak = values[0]
        max_dd = 0.0
        max_dd_start = 0
        max_dd_end = 0

        for i in range(len(values)):
            if values[i] > peak:
                peak = values[i]
                peak_idx = i

            dd = (peak - values[i]) / peak if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
                max_dd_end = i
                # 找到开始点
                for j in range(i, -1, -1):
                    if values[j] >= peak:
                        max_dd_start = j
                        break

        return max_dd, max_dd_start, max_dd_end

    def generate_summary_report(self, metrics: Dict) -> str:
        """
        生成摘要报告

        参数:
            metrics: 绩效指标字典

        返回:
            报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("回测绩效摘要")
        report.append("=" * 80)

        # 收益指标
        if 'returns' in metrics and metrics['returns']:
            r = metrics['returns']
            report.append("\n【收益指标】")
            report.append(f"总收益率: {r.get('total_return_pct', 0):.2f}%")
            report.append(f"年化收益率: {r.get('annualized_return_pct', 0):.2f}%")
            report.append(f"平均日收益: {r.get('avg_daily_return_pct', 0):.3f}%")

        # 风险指标
        if 'risk' in metrics and metrics['risk']:
            r = metrics['risk']
            report.append("\n【风险指标】")
            report.append(f"最大回撤: {r.get('max_drawdown_pct', 0):.2f}%")
            report.append(f"年化波动率: {r.get('volatility_pct', 0):.2f}%")
            report.append(f"下行风险: {r.get('downside_risk_pct', 0):.2f}%")

        # 风险调整收益
        if 'risk_adjusted' in metrics and metrics['risk_adjusted']:
            r = metrics['risk_adjusted']
            report.append("\n【风险调整收益】")
            report.append(f"夏普比率: {r.get('sharpe_ratio', 0):.2f}")
            report.append(f"索提诺比率: {r.get('sortino_ratio', 0):.2f}")
            report.append(f"卡玛比率: {r.get('calmar_ratio', 0):.2f}")

        # 交易质量
        if 'trading' in metrics and metrics['trading']:
            t = metrics['trading']
            report.append("\n【交易质量】")
            report.append(f"总交易次数: {t.get('total_trades', 0)} 笔")
            report.append(f"胜率: {t.get('win_rate_pct', 0):.2f}%")
            report.append(f"盈亏比: {t.get('profit_loss_ratio', 0):.2f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试绩效计算器")
    print("=" * 80)

    calculator = PerformanceCalculator()

    # 创建模拟数据
    np.random.seed(42)
    days = 252  # 1年
    initial_value = 1000000
    values = [initial_value]

    for i in range(days):
        daily_return = np.random.normal(0.0005, 0.02)  # 日均0.05%收益，2%波动
        new_value = values[-1] * (1 + daily_return)
        values.append(new_value)

    # 模拟交易记录
    trades = []
    for i in range(0, days, 30):  # 每月一次交易
        trades.append({
            'date': i,
            'quantity': 100,
            'price': values[i] / 10000
        })
        if i + 15 < len(values):
            trades.append({
                'date': i + 15,
                'quantity': -100,
                'price': values[i + 15] / 10000
            })

    # 计算绩效
    metrics = calculator.calculate_all_metrics(values, trades)

    # 生成报告
    report = calculator.generate_summary_report(metrics)
    print(report)

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
