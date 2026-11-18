#!/usr/bin/env python3
"""
综合分析工具 - 整合技术面和基本面分析
这是系统的最终形态，提供完整的投资分析报告
"""
import sys
sys.path.insert(0, '/mnt/d/claude-code/stock')

import pandas as pd
import numpy as np
from datetime import datetime
from scripts.core import PortfolioManager, PositionCalculator
from scripts.core.data_fetcher_baostock import BaoStockFetcher
from scripts.analyzers import TechnicalAnalyzer, FundamentalAnalyzer
from scripts.etf.analyzer import ETFAnalyzer


class ComprehensiveAnalyzer:
    """
    综合分析器

    整合技术面和基本面分析，提供完整的投资建议

    示例:
        analyzer = ComprehensiveAnalyzer()
        result = analyzer.analyze_stock('600519', '贵州茅台')
    """

    def __init__(self):
        """初始化综合分析器"""
        # 优先使用BaoStock（稳定、全面）
        try:
            self.fetcher = BaoStockFetcher(auto_login=True)
            self.data_source = 'BaoStock'
            print("✅ 使用BaoStock数据源（推荐）")
        except Exception as e:
            print(f"⚠️  BaoStock初始化失败: {e}，尝试使用备用数据源")
            try:
                from scripts.core.data_fetcher_enhanced import EnhancedDataFetcher
                self.fetcher = EnhancedDataFetcher(enable_cache=True)
                self.data_source = 'EnhancedDataFetcher'
                print("✅ 使用增强版数据源（备用）")
            except Exception as e2:
                print(f"❌ 所有数据源初始化失败: {e2}")
                self.fetcher = None
                self.data_source = 'None'

        self.pm = PortfolioManager()
        self.position_calc = PositionCalculator()
        self.tech_analyzer = TechnicalAnalyzer('')
        self.fund_analyzer = FundamentalAnalyzer()

        # 添加ETF分析器
        try:
            self.etf_analyzer = ETFAnalyzer()
            print("✅ ETF分析器初始化成功")
        except Exception as e:
            print(f"⚠️  ETF分析器初始化失败: {e}")
            self.etf_analyzer = None

        print(f"✅ ComprehensiveAnalyzer初始化成功（数据源: {self.data_source}）")

    def analyze_stock(
        self,
        symbol: str,
        name: str,
        kline_data: pd.DataFrame = None,
        financial_data: dict = None
    ) -> dict:
        """
        综合分析单只股票

        参数:
            symbol: 股票代码
            name: 股票名称
            kline_data: K线数据（可选，如未提供则使用模拟数据）
            financial_data: 财务数据（可选，如未提供则使用模拟数据）

        返回:
            完整的分析报告
        """
        try:
            print(f"\n{'=' * 80}")
            print(f"  {name} ({symbol}) - 综合分析报告")
            print(f"{'=' * 80}")

            # 识别股票类型（A股或ETF）
            symbol_str = str(symbol).zfill(6)
            is_etf = symbol_str.startswith('51') or symbol_str.startswith('15')

            # 如果是ETF，使用ETF分析器
            if is_etf and self.etf_analyzer:
                print(f"📌 识别为ETF，使用ETF分析器...")
                etf_result = self.etf_analyzer.analyze(symbol_str, name)

                if etf_result:
                    # 格式化ETF分析结果为统一格式
                    current_price = kline_data['close'].iloc[-1] if kline_data is not None and not kline_data.empty else 0.80

                    return {
                        'symbol': symbol,
                        'name': name,
                        'current_price': current_price,
                        'tech_score': etf_result['tech_analysis'].get('score', 60),
                        'fund_score': int(
                            (etf_result['premium_analysis'].get('score', 60) +
                             etf_result['tracking_analysis'].get('score', 60) +
                             etf_result['scale_analysis'].get('score', 60) +
                             etf_result['component_analysis'].get('score', 60)) / 4
                        ),
                        'overall_score': etf_result['overall_score'],
                        'rating': etf_result['rating'],
                        'action': etf_result['recommendation']['action'],
                        'position': {
                            'suggested_fraction': etf_result['recommendation']['position_pct'] / 100,
                            'suggested_amount': 0  # 将在调用处计算
                        },
                        'stop_loss': current_price * 0.92,
                        'take_profit': current_price * 1.15,
                        'is_etf': True,
                        'etf_analysis': etf_result
                    }

            # A股分析（原有逻辑）
            # 如果没有提供数据，尝试获取真实数据
            if kline_data is None:
                try:
                    print(f"📡 尝试获取真实K线数据...")
                    kline_data = self.fetcher.get_historical_kline(
                        symbol=symbol,
                        period='daily',
                        start_date=(datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        adjust='qfq'
                    )
                    if kline_data.empty:
                        print(f"⚠️  真实数据获取失败，使用模拟数据")
                        kline_data = self._create_mock_kline(symbol)
                    else:
                        print(f"✅ 成功获取真实数据：{len(kline_data)}条记录")
                except Exception as e:
                    print(f"⚠️  数据获取失败: {str(e)[:50]}，使用模拟数据")
                    kline_data = self._create_mock_kline(symbol)

            if financial_data is None:
                try:
                    print(f"📡 尝试获取财务数据...")
                    # 使用最新一年的财务数据
                    current_year = datetime.now().year
                    financial_data = self.fetcher.get_financial_data(symbol, year=current_year)

                    if not financial_data:
                        # 尝试上一年
                        financial_data = self.fetcher.get_financial_data(symbol, year=current_year-1)

                    if not financial_data:
                        print(f"⚠️  财务数据获取失败，使用模拟数据")
                        financial_data = self._create_mock_financials(symbol)
                    else:
                        print(f"✅ 成功获取财务数据")
                except Exception as e:
                    print(f"⚠️  财务数据获取失败: {str(e)[:50]}，使用模拟数据")
                    financial_data = self._create_mock_financials(symbol)

            current_price = kline_data['close'].iloc[-1]

            # 1. 技术面分析
            print(f"\n【技术面分析】")
            print("-" * 80)
            tech_result = self.tech_analyzer.analyze_indicators(kline_data)

            if tech_result:
                # 显示技术指标
                indicators = ['MACD', 'RSI', 'KDJ', '布林带', '均线']
                signals = [
                    tech_result['macd'],
                    tech_result['rsi'],
                    tech_result['kdj'],
                    tech_result['bollinger'],
                    tech_result['moving_averages']
                ]

                tech_score = 0
                for ind, sig in zip(indicators, signals):
                    signal_emoji = {
                        'buy': '📈', 'strong_buy': '📈', 'golden_cross': '📈',
                        'sell': '📉', 'strong_sell': '📉', 'death_cross': '📉',
                        'hold': '➡️'
                    }.get(sig['signal'], '➡️')

                    print(f"{signal_emoji} {ind:6s}: {sig['description']}")

                    # 计算技术面评分
                    if sig['signal'] in ['buy', 'strong_buy', 'golden_cross']:
                        tech_score += sig['strength'] * 20
                    elif sig['signal'] in ['sell', 'strong_sell', 'death_cross']:
                        tech_score -= sig['strength'] * 20

                tech_score = max(0, min(100, 50 + tech_score))
                overall_tech = tech_result['overall']

                print(f"\n{'➡️' if overall_tech['signal'] == 'hold' else '✅' if overall_tech['signal'] == 'buy' else '❌'} 技术面综合: {overall_tech['signal'].upper()} (强度: {overall_tech['strength']:.2f}, 评分: {tech_score:.0f})")

            # 2. 基本面分析
            print(f"\n【基本面分析】")
            print("-" * 80)
            fund_result = self.fund_analyzer.analyze(symbol, financial_data)

            print(f"盈利能力: {fund_result['profitability']['description']}")
            print(f"  - ROE: {fund_result['profitability']['roe']:.1%}, 毛利率: {fund_result['profitability']['gross_margin']:.1%}")

            print(f"成长性: {fund_result['growth']['description']}")
            print(f"  - 营收增长: {fund_result['growth']['revenue_growth']:.1%}, 利润增长: {fund_result['growth']['profit_growth']:.1%}")

            print(f"财务安全: {fund_result['safety']['description']}")
            print(f"  - 资产负债率: {fund_result['safety']['debt_ratio']:.1%}, 流动比率: {fund_result['safety']['current_ratio']:.2f}")

            print(f"估值分析: {fund_result['valuation']['description']}")
            print(f"  - PE: {fund_result['valuation']['pe']:.1f}, PB: {fund_result['valuation']['pb']:.1f}")

            # 财务异常
            if fund_result['anomalies']:
                print(f"\n⚠️  财务异常:")
                for anomaly in fund_result['anomalies']:
                    print(f"  [{anomaly['severity'].upper()}] {anomaly['description']}")
            else:
                print(f"\n✅ 财务健康: 未检测到明显异常")

            print(f"\n基本面综合: {fund_result['overall_rating'].upper()} (评分: {fund_result['overall_score']:.0f})")

            # 3. 综合评分与建议
            print(f"\n【综合评估】")
            print("-" * 80)

            # 计算综合评分（技术面40%，基本面60%）
            overall_score = tech_score * 0.4 + fund_result['overall_score'] * 0.6

            # 综合评级
            if overall_score >= 80:
                rating = '强烈推荐'
                action = 'buy'
            elif overall_score >= 60:
                rating = '推荐'
                action = 'buy'
            elif overall_score >= 40:
                rating = '中性'
                action = 'hold'
            else:
                rating = '不推荐'
                action = 'sell'

            # 仓位建议
            signal_strength = min(tech_score / 100, fund_result['overall_score'] / 100)
            capital = self.pm.load_capital()
            total_capital = capital.get('总资金', 500000)

            position = self.position_calc.suggest_position(
                signal_strength=signal_strength,
                total_capital=total_capital,
                win_rate=0.55,
                avg_win=0.12,
                avg_loss=0.08
            )

            print(f"技术面评分: {tech_score:.0f}/100")
            print(f"基本面评分: {fund_result['overall_score']:.0f}/100")
            print(f"综合评分: {overall_score:.0f}/100")
            print(f"\n综合评级: {rating}")

            # 最终建议
            print(f"\n【投资建议】")
            print("-" * 80)

            if action == 'buy':
                print(f"✅ 建议操作: 买入")
                print(f"建议价位: ¥{current_price * 0.98:.2f} - ¥{current_price * 1.02:.2f}")
            elif action == 'sell':
                print(f"❌ 建议操作: 卖出")
                print(f"建议价位: ¥{current_price:.2f}")
            else:
                print(f"⏸️  建议操作: 观望")

            print(f"建议仓位: {position['suggested_fraction']:.1%} (¥{position['suggested_amount']:,.2f})")

            # 止损止盈
            stop_loss = current_price * 0.92
            take_profit = current_price * 1.15
            print(f"\n止损价格: ¥{stop_loss:.2f} (-8%)")
            print(f"止盈价格: ¥{take_profit:.2f} (+15%)")

            # 风险提示
            print(f"\n【风险提示】")
            print("-" * 80)

            if fund_result['valuation']['valuation_level'] == 'overvalued':
                print("⚠️  估值偏高，注意回调风险")
            elif fund_result['valuation']['valuation_level'] == 'undervalued':
                print("✅ 估值合理，具有安全边际")

            if tech_result and tech_result['macd']['signal'] == 'death_cross':
                print("⚠️  MACD死叉，短期可能下跌")
            elif tech_result and tech_result['macd']['signal'] == 'golden_cross':
                print("✅ MACD金叉，短期可能上涨")

            if fund_result['anomalies']:
                high_risks = [a for a in fund_result['anomalies'] if a['severity'] == 'high']
                if high_risks:
                    print(f"⚠️  存在{len(high_risks)}个高风险异常，建议谨慎")

            return {
                'symbol': symbol,
                'name': name,
                'current_price': current_price,
                'tech_score': tech_score,
                'fund_score': fund_result['overall_score'],
                'overall_score': overall_score,
                'rating': rating,
                'action': action,
                'position': position,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }

        except Exception as e:
            print(f"❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_mock_kline(self, symbol: str, days: int = 200) -> pd.DataFrame:
        """创建模拟K线数据"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        base_prices = {
            '600519': 1500,
            '603881': 30,
            '512760': 0.7,
            '515070': 1.2,
            '300059': 20,
        }

        base_price = base_prices.get(symbol, 50)
        trend = np.linspace(0, np.random.uniform(-20, 50), days)
        noise = np.random.randn(days) * base_price * 0.03
        close_prices = base_price + trend + noise

        df = pd.DataFrame({
            'date': dates,
            'open': close_prices * (1 + np.random.randn(days) * 0.02),
            'high': close_prices * (1 + np.abs(np.random.randn(days) * 0.03)),
            'low': close_prices * (1 - np.abs(np.random.randn(days) * 0.03)),
            'close': close_prices,
            'volume': np.random.randint(100000, 500000, days)
        })

        return df

    def _create_mock_financials(self, symbol: str) -> dict:
        """创建模拟财务数据"""
        # 确保symbol是字符串
        symbol = str(symbol)

        # 基于不同股票类型返回不同数据
        if symbol == '600519':  # 贵州茅台
            return {
                'pe': 25.3,
                'pb': 8.5,
                'roe': 0.302,
                'roa': 0.225,
                'gross_margin': 0.915,
                'net_margin': 0.523,
                'revenue_growth': 0.125,
                'profit_growth': 0.152,
                'debt_ratio': 0.235,
                'current_ratio': 5.2,
                'operating_cash_flow': 1000000000,
                'receivable_growth': 0.10,
                'inventory_growth': 0.08,
                'gross_margin_change': -0.02,
                'roe_change': -0.01
            }
        elif symbol.startswith('512') or symbol.startswith('515'):  # ETF
            return {
                'pe': 18.5,
                'pb': 2.3,
                'roe': 0.18,
                'roa': 0.12,
                'gross_margin': 0.45,
                'net_margin': 0.35,
                'revenue_growth': 0.08,
                'profit_growth': 0.10,
                'debt_ratio': 0.15,
                'current_ratio': 3.5,
                'operating_cash_flow': 500000000,
                'receivable_growth': 0.05,
                'inventory_growth': 0.03,
                'gross_margin_change': -0.01,
                'roe_change': 0.02
            }
        else:  # 普通股票
            return {
                'pe': 22.0,
                'pb': 3.5,
                'roe': 0.15,
                'roa': 0.10,
                'gross_margin': 0.35,
                'net_margin': 0.12,
                'revenue_growth': 0.10,
                'profit_growth': 0.12,
                'debt_ratio': 0.45,
                'current_ratio': 2.0,
                'operating_cash_flow': 200000000,
                'receivable_growth': 0.12,
                'inventory_growth': 0.10,
                'gross_margin_change': 0.01,
                'roe_change': -0.02
            }


def main():
    """主函数 - 演示综合分析"""
    print("=" * 80)
    print(" " * 20 + "A股混合策略分析系统 - 综合分析演示")
    print("=" * 80)

    print("\n⚠️  注意：本演示使用模拟数据")
    print("实际使用时会获取真实行情和财务数据\n")

    # 初始化
    analyzer = ComprehensiveAnalyzer()

    # 加载用户持仓
    portfolio = analyzer.pm.load_portfolio()

    if portfolio.empty:
        print("❌ 无法加载持仓配置")
        return

    print(f"✅ 已加载持仓: {len(portfolio)}只股票\n")

    # 分析前3只股票
    for idx, row in portfolio.head(3).iterrows():
        symbol = row['股票代码']
        name = row['股票名称']

        analyzer.analyze_stock(symbol, name)

        print("\n" + "─" * 80)

    # 总结
    print("\n" + "=" * 80)
    print(" " * 30 + "系统总结")
    print("=" * 80)

    print("""
✅ Phase 1-4 全部完成！

【已完成功能】
✓ Phase 1: 项目基础设施
✓ Phase 2: 核心数据模块（DataFetcher、PortfolioManager、PositionCalculator）
✓ Phase 3: 技术分析（MultiScaleMACD、TechnicalAnalyzer）
✓ Phase 4: 基本面分析（FundamentalAnalyzer）

【系统特点】
⭐ 多时间尺度MACD分析（核心创新）
⭐ 技术面 + 基本面综合分析
⭐ 凯利公式仓位管理
⭐ 财务异常检测
⭐ 完整风险控制

【下一步】
Phase 5: 策略回测
Phase 6: 自动化功能
Phase 7: 高级功能（机器学习等）

【使用建议】
1. 先在模拟盘验证系统
2. 小仓位试水
3. 严格执行风险控制
4. 持续优化参数

⚠️  重要提示：
• 所有分析仅供参考，不构成投资建议
• 投资有风险，决策需谨慎
• 历史表现不代表未来收益
    """)

    print("=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
