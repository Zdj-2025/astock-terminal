#!/usr/bin/env python3
"""
回测策略执行器

职责:
- 集成技术分析和基本面分析器
- 生成买入/卖出信号
- 计算建议仓位
- 确保不使用未来数据
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import sys

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.analyzers import TechnicalAnalyzer, FundamentalAnalyzer
from scripts.core import PositionCalculator


class BacktestStrategyExecutor:
    """
    回测策略执行器

    使用现有的分析框架生成交易信号

    策略逻辑:
    1. 技术面分析 (40%):
       - MACD金叉/死叉
       - RSI超买/超卖
       - KDJ金叉/死叉
       - 布林带位置

    2. 基本面分析 (60%):
       - 盈利能力 (ROE、毛利率)
       - 成长性 (营收增长、利润增长)
       - 财务安全 (资产负债率)
       - 估值水平 (PE、PB)

    3. 信号生成:
       - 综合评分 > 70: 买入
       - 综合评分 < 40: 卖出
       - 其他: 持有
    """

    def __init__(
        self,
        min_signal_strength: float = 0.6,
        buy_threshold: float = 70.0,
        sell_threshold: float = 40.0
    ):
        """
        初始化策略执行器

        参数:
            min_signal_strength: 最小信号强度
            buy_threshold: 买入阈值
            sell_threshold: 卖出阈值
        """
        self.tech_analyzer = TechnicalAnalyzer('')
        self.fund_analyzer = FundamentalAnalyzer()
        self.position_calc = PositionCalculator()

        self.min_signal_strength = min_signal_strength
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # 缓存财务数据（按季度）
        self.financial_cache = {}

        print(f"🎯 初始化策略执行器")
        print(f"   买入阈值: {buy_threshold}分")
        print(f"   卖出阈值: {sell_threshold}分")
        print(f"   最小信号强度: {min_signal_strength}")

    def generate_signal(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        current_date: datetime,
        current_capital: float
    ) -> Optional[Dict]:
        """
        生成交易信号

        参数:
            symbol: 股票代码
            historical_data: 历史K线数据（截至当前日期）
            current_date: 当前日期
            current_capital: 当前可用资金

        返回:
            信号字典 或 None
        """
        try:
            # 检查数据充足性
            if len(historical_data) < 60:  # 至少60天数据
                return None

            # 1. 技术面分析
            tech_score, tech_signals = self._analyze_technical(historical_data)

            # 2. 基本面分析（使用缓存）
            fund_score, fund_signals = self._analyze_fundamental(symbol, current_date)

            # 3. 综合评分
            overall_score = tech_score * 0.4 + fund_score * 0.6

            # 4. 生成信号
            signal = None

            if overall_score >= self.buy_threshold:
                # 买入信号
                signal_strength = (overall_score - self.buy_threshold) / (100 - self.buy_threshold)
                signal_strength = min(1.0, max(0.0, signal_strength))

                if signal_strength >= self.min_signal_strength:
                    # 计算建议仓位
                    position = self.position_calc.suggest_position(
                        signal_strength=signal_strength,
                        total_capital=current_capital,
                        win_rate=0.55,  # 默认胜率
                        avg_win=0.12,   # 默认平均盈利
                        avg_loss=0.08   # 默认平均亏损
                    )

                    signal = {
                        'symbol': symbol,
                        'signal_type': 'BUY',
                        'strength': signal_strength,
                        'suggested_position': position['suggested_fraction'],
                        'overall_score': overall_score,
                        'tech_score': tech_score,
                        'fund_score': fund_score,
                        'reason': self._generate_buy_reason(tech_signals, fund_signals)
                    }

            elif overall_score <= self.sell_threshold:
                # 卖出信号
                signal_strength = (self.sell_threshold - overall_score) / self.sell_threshold
                signal_strength = min(1.0, max(0.0, signal_strength))

                if signal_strength >= self.min_signal_strength:
                    signal = {
                        'symbol': symbol,
                        'signal_type': 'SELL',
                        'strength': signal_strength,
                        'suggested_position': 0.0,  # 清仓
                        'overall_score': overall_score,
                        'tech_score': tech_score,
                        'fund_score': fund_score,
                        'reason': self._generate_sell_reason(tech_signals, fund_signals)
                    }

            return signal

        except Exception as e:
            print(f"   ⚠️  {symbol} 信号生成失败: {str(e)[:50]}")
            return None

    def _analyze_technical(self, historical_data: pd.DataFrame) -> tuple:
        """
        技术面分析

        返回:
            (score, signals)
        """
        try:
            # 使用现有技术分析器
            tech_result = self.tech_analyzer.analyze_indicators(historical_data)

            if not tech_result:
                return 50.0, {}  # 默认中性评分

            # 计算技术面评分
            score = 50.0  # 基础分

            # MACD
            macd = tech_result.get('macd', {})
            if macd.get('signal') in ['golden_cross', 'buy']:
                score += 15
            elif macd.get('signal') in ['death_cross', 'sell']:
                score -= 15

            # RSI
            rsi = tech_result.get('rsi', {})
            rsi_value = rsi.get('value', 50)
            if rsi_value < 30:
                score += 10  # 超卖
            elif rsi_value > 70:
                score -= 10  # 超买

            # KDJ
            kdj = tech_result.get('kdj', {})
            if kdj.get('signal') == 'golden_cross':
                score += 10
            elif kdj.get('signal') == 'death_cross':
                score -= 10

            # 布林带
            boll = tech_result.get('bollinger', {})
            if '下轨' in boll.get('description', ''):
                score += 5  # 触及下轨，可能反弹
            elif '上轨' in boll.get('description', ''):
                score -= 5  # 触及上轨，可能回调

            # 限制在0-100
            score = max(0, min(100, score))

            return score, tech_result

        except Exception as e:
            return 50.0, {}

    def _analyze_fundamental(self, symbol: str, current_date: datetime) -> tuple:
        """
        基本面分析（使用缓存，避免重复获取）

        返回:
            (score, signals)
        """
        try:
            # 检查缓存
            cache_key = f"{symbol}_{current_date.year}_{current_date.month // 4 + 1}"

            if cache_key in self.financial_cache:
                return self.financial_cache[cache_key]

            # 使用模拟财务数据（实际应用中应该获取真实数据）
            # 这里为了回测的连续性，使用预设的合理值
            fund_data = self._get_mock_financials(symbol)

            # 使用现有基本面分析器
            fund_result = self.fund_analyzer.analyze(symbol, fund_data)

            if not fund_result:
                return 60.0, {}  # 默认略高评分

            # 基本面评分
            score = fund_result.get('overall_score', 60)

            # 缓存结果
            self.financial_cache[cache_key] = (score, fund_result)

            return score, fund_result

        except Exception as e:
            return 60.0, {}

    def _get_mock_financials(self, symbol: str) -> Dict:
        """获取模拟财务数据（用于回测）"""
        # 不同股票使用不同的模拟数据
        base_data = {
            'pe': 20.0 + np.random.uniform(-5, 10),
            'pb': 3.0 + np.random.uniform(-1, 2),
            'roe': 0.12 + np.random.uniform(-0.03, 0.10),
            'roa': 0.08 + np.random.uniform(-0.02, 0.05),
            'gross_margin': 0.30 + np.random.uniform(-0.10, 0.20),
            'net_margin': 0.10 + np.random.uniform(-0.03, 0.08),
            'revenue_growth': 0.10 + np.random.uniform(-0.05, 0.15),
            'profit_growth': 0.12 + np.random.uniform(-0.05, 0.18),
            'debt_ratio': 0.40 + np.random.uniform(-0.10, 0.20),
            'current_ratio': 2.0 + np.random.uniform(-0.5, 1.0),
            'operating_cash_flow': 500000000,
            'receivable_growth': 0.10 + np.random.uniform(-0.05, 0.10),
            'inventory_growth': 0.08 + np.random.uniform(-0.04, 0.08),
            'gross_margin_change': np.random.uniform(-0.02, 0.02),
            'roe_change': np.random.uniform(-0.01, 0.02)
        }

        return base_data

    def _generate_buy_reason(self, tech_signals: Dict, fund_signals: Dict) -> str:
        """生成买入理由"""
        reasons = []

        # 技术面理由
        if tech_signals.get('macd', {}).get('signal') in ['golden_cross', 'buy']:
            reasons.append("MACD金叉")

        if tech_signals.get('kdj', {}).get('signal') == 'golden_cross':
            reasons.append("KDJ金叉")

        rsi_value = tech_signals.get('rsi', {}).get('value', 50)
        if rsi_value < 30:
            reasons.append("RSI超卖")

        # 基本面理由
        fund_score = fund_signals.get('overall_score', 0)
        if fund_score >= 70:
            reasons.append("基本面优秀")

        return ", ".join(reasons) if reasons else "综合评分达标"

    def _generate_sell_reason(self, tech_signals: Dict, fund_signals: Dict) -> str:
        """生成卖出理由"""
        reasons = []

        # 技术面理由
        if tech_signals.get('macd', {}).get('signal') in ['death_cross', 'sell']:
            reasons.append("MACD死叉")

        if tech_signals.get('kdj', {}).get('signal') == 'death_cross':
            reasons.append("KDJ死叉")

        rsi_value = tech_signals.get('rsi', {}).get('value', 50)
        if rsi_value > 70:
            reasons.append("RSI超买")

        # 基本面理由
        fund_score = fund_signals.get('overall_score', 100)
        if fund_score < 50:
            reasons.append("基本面恶化")

        return ", ".join(reasons) if reasons else "综合评分过低"


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试策略执行器")
    print("=" * 80)

    # 创建策略执行器
    executor = BacktestStrategyExecutor()

    # 创建模拟数据
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    close_prices = 35 + np.random.randn(200).cumsum() * 0.5

    test_data = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(200) * 0.01),
        'high': close_prices * (1 + np.abs(np.random.randn(200) * 0.015)),
        'low': close_prices * (1 - np.abs(np.random.randn(200) * 0.015)),
        'close': close_prices,
        'volume': np.random.randint(10000000, 50000000, 200)
    })

    # 测试信号生成
    signal = executor.generate_signal(
        symbol='603881',
        historical_data=test_data,
        current_date=datetime.now(),
        current_capital=1000000
    )

    if signal:
        print(f"\n✅ 信号生成成功:")
        print(f"   类型: {signal['signal_type']}")
        print(f"   强度: {signal['strength']:.2f}")
        print(f"   建议仓位: {signal['suggested_position']:.2%}")
        print(f"   综合评分: {signal['overall_score']:.0f}")
        print(f"   理由: {signal['reason']}")
    else:
        print(f"\n⏸️  无信号（持有）")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
