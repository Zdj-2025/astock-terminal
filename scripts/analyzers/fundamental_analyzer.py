#!/usr/bin/env python3
"""
基本面分析模块
整合china-stock-analysis skill，提供财务健康度分析和估值计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class FundamentalAnalyzer:
    """
    基本面分析器

    基于价值投资理论分析股票基本面：
    - 盈利能力分析（ROE、ROA、毛利率等）
    - 成长性分析（营收增长、利润增长）
    - 财务安全分析（资产负债率、流动比率）
    - 估值分析（PE、PB、PEG）
    - 财务异常检测

    示例:
        analyzer = FundamentalAnalyzer()
        result = analyzer.analyze('600519', financial_data)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化基本面分析器

        参数:
            config: 配置参数（可选）
        """
        self.config = config or self._default_config()
        print("✅ FundamentalAnalyzer初始化成功")

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # 筛选标准
            'pe_max': 20,              # PE上限
            'pb_max': 5,               # PB上限
            'roe_min': 0.15,           # ROE下限15%
            'revenue_growth_min': 0.10, # 营收增长下限10%
            'profit_growth_min': 0.15,  # 利润增长下限15%
            'debt_ratio_max': 0.60,    # 资产负债率上限60%
            'current_ratio_min': 1.5,   # 流动比率下限

            # 评分权重
            'weights': {
                'profitability': 0.3,   # 盈利能力权重
                'growth': 0.25,         # 成长性权重
                'safety': 0.25,         # 财务安全权重
                'valuation': 0.2        # 估值权重
            }
        }

    def analyze_profitability(self, data: Dict) -> Dict:
        """
        分析盈利能力

        参数:
            data: 财务数据字典

        返回:
            {
                'roe': 净资产收益率,
                'roa': 总资产收益率,
                'gross_margin': 毛利率,
                'net_margin': 净利率,
                'score': 评分（0-100）,
                'rating': 评级
            }
        """
        try:
            # 提取指标
            roe = data.get('roe', 0)
            roa = data.get('roa', roe * 0.7)  # 估算
            gross_margin = data.get('gross_margin', 0)
            net_margin = data.get('net_margin', 0)

            # 计算评分
            score = 0

            # ROE评分（最重要的指标）
            if roe >= 0.20:
                score += 40
            elif roe >= 0.15:
                score += 30
            elif roe >= 0.10:
                score += 20
            else:
                score += 10

            # 毛利率评分
            if gross_margin >= 0.50:
                score += 25
            elif gross_margin >= 0.30:
                score += 20
            elif gross_margin >= 0.20:
                score += 15
            else:
                score += 10

            # 净利率评分
            if net_margin >= 0.20:
                score += 25
            elif net_margin >= 0.10:
                score += 20
            elif net_margin >= 0.05:
                score += 15
            else:
                score += 10

            # ROA加成
            if roa >= 0.10:
                score += 10

            score = min(score, 100)

            # 评级
            if score >= 80:
                rating = 'excellent'
            elif score >= 60:
                rating = 'good'
            elif score >= 40:
                rating = 'fair'
            else:
                rating = 'poor'

            return {
                'roe': roe,
                'roa': roa,
                'gross_margin': gross_margin,
                'net_margin': net_margin,
                'score': score,
                'rating': rating,
                'description': f"盈利能力: {rating.upper()} (评分{score})"
            }

        except Exception as e:
            print(f"❌ 盈利能力分析失败: {e}")
            return {'score': 0, 'rating': 'error'}

    def analyze_growth(self, data: Dict) -> Dict:
        """
        分析成长性

        参数:
            data: 财务数据字典

        返回:
            {
                'revenue_growth': 营收增长率,
                'profit_growth': 利润增长率,
                'score': 评分（0-100）,
                'rating': 评级
            }
        """
        try:
            revenue_growth = data.get('revenue_growth', 0)
            profit_growth = data.get('profit_growth', 0)

            score = 0

            # 营收增长评分
            if revenue_growth >= 0.30:
                score += 30
            elif revenue_growth >= 0.20:
                score += 25
            elif revenue_growth >= 0.10:
                score += 20
            elif revenue_growth >= 0.05:
                score += 10
            else:
                score += 5

            # 利润增长评分
            if profit_growth >= 0.30:
                score += 40
            elif profit_growth >= 0.20:
                score += 30
            elif profit_growth >= 0.15:
                score += 25
            elif profit_growth >= 0.10:
                score += 20
            else:
                score += 10

            # 增长一致性加成
            if revenue_growth > 0 and profit_growth > 0:
                score += 20
            elif revenue_growth > 0 or profit_growth > 0:
                score += 10

            score = min(score, 100)

            # 评级
            if score >= 70:
                rating = 'excellent'
            elif score >= 50:
                rating = 'good'
            elif score >= 30:
                rating = 'fair'
            else:
                rating = 'poor'

            return {
                'revenue_growth': revenue_growth,
                'profit_growth': profit_growth,
                'score': score,
                'rating': rating,
                'description': f"成长性: {rating.upper()} (评分{score})"
            }

        except Exception as e:
            print(f"❌ 成长性分析失败: {e}")
            return {'score': 0, 'rating': 'error'}

    def analyze_safety(self, data: Dict) -> Dict:
        """
        分析财务安全

        参数:
            data: 财务数据字典

        返回:
            {
                'debt_ratio': 资产负债率,
                'current_ratio': 流动比率,
                'score': 评分（0-100）,
                'rating': 评级
            }
        """
        try:
            debt_ratio = data.get('debt_ratio', 0)
            current_ratio = data.get('current_ratio', 1.5)

            score = 0

            # 资产负债率评分（越低越好）
            if debt_ratio <= 0.30:
                score += 40
            elif debt_ratio <= 0.50:
                score += 30
            elif debt_ratio <= 0.60:
                score += 20
            elif debt_ratio <= 0.70:
                score += 10
            else:
                score += 5

            # 流动比率评分
            if current_ratio >= 2.0:
                score += 35
            elif current_ratio >= 1.5:
                score += 25
            elif current_ratio >= 1.0:
                score += 15
            else:
                score += 5

            # 现金流检查（如果有数据）
            cash_flow = data.get('operating_cash_flow', 0)
            if cash_flow > 0:
                score += 25

            score = min(score, 100)

            # 评级
            if score >= 75:
                rating = 'excellent'
            elif score >= 55:
                rating = 'good'
            elif score >= 35:
                rating = 'fair'
            else:
                rating = 'poor'

            return {
                'debt_ratio': debt_ratio,
                'current_ratio': current_ratio,
                'score': score,
                'rating': rating,
                'description': f"财务安全: {rating.upper()} (评分{score})"
            }

        except Exception as e:
            print(f"❌ 财务安全分析失败: {e}")
            return {'score': 0, 'rating': 'error'}

    def analyze_valuation(self, data: Dict, industry_data: Optional[Dict] = None) -> Dict:
        """
        分析估值

        参数:
            data: 财务数据字典
            industry_data: 行业数据（可选）

        返回:
            {
                'pe': 市盈率,
                'pb': 市净率,
                'peg': PEG比率,
                'valuation_level': 估值水平,
                'score': 评分（0-100）,
                'rating': 评级
            }
        """
        try:
            pe = data.get('pe', 0)
            pb = data.get('pb', 0)
            profit_growth = data.get('profit_growth', 0.10)

            # 计算PEG
            peg = pe / (profit_growth * 100) if profit_growth > 0 else float('inf')

            score = 0

            # PE评分
            if pe <= 15:
                score += 35
            elif pe <= 20:
                score += 28
            elif pe <= 30:
                score += 20
            elif pe <= 50:
                score += 10
            else:
                score += 5

            # PB评分
            if pb <= 2:
                score += 30
            elif pb <= 3:
                score += 22
            elif pb <= 5:
                score += 15
            else:
                score += 8

            # PEG评分
            if peg <= 1.0:
                score += 35
            elif peg <= 1.5:
                score += 25
            elif peg <= 2.0:
                score += 15
            else:
                score += 5

            score = min(score, 100)

            # 估值水平
            if score >= 70:
                valuation_level = 'undervalued'
                rating = 'excellent'
            elif score >= 50:
                valuation_level = 'fair'
                rating = 'good'
            else:
                valuation_level = 'overvalued'
                rating = 'poor'

            return {
                'pe': pe,
                'pb': pb,
                'peg': peg,
                'valuation_level': valuation_level,
                'score': score,
                'rating': rating,
                'description': f"估值水平: {valuation_level.upper()} (评分{score})"
            }

        except Exception as e:
            print(f"❌ 估值分析失败: {e}")
            return {'score': 0, 'rating': 'error'}

    def detect_anomalies(self, data: Dict) -> List[Dict]:
        """
        检测财务异常

        参数:
            data: 财务数据字典

        返回:
            异常列表 [
                {
                    'type': 异常类型,
                    'severity': 严重程度（high/medium/low）,
                    'description': 描述
                }
            ]
        """
        anomalies = []

        try:
            # 1. 应收账款异常
            if 'receivable_growth' in data and 'revenue_growth' in data:
                if data['receivable_growth'] > data['revenue_growth'] * 1.5:
                    anomalies.append({
                        'type': 'receivable_growth',
                        'severity': 'medium',
                        'description': f"应收账款增速({data['receivable_growth']:.1%})超过营收增速({data['revenue_growth']:.1%})的1.5倍"
                    })

            # 2. 现金流背离
            if 'profit_growth' in data and 'cash_flow_growth' in data:
                if data['profit_growth'] > 0.15 and data['cash_flow_growth'] < 0:
                    anomalies.append({
                        'type': 'cash_flow_divergence',
                        'severity': 'high',
                        'description': f"利润增长{data['profit_growth']:.1%}但经营现金流下降"
                    })

            # 3. 存货异常
            if 'inventory_growth' in data and 'revenue_growth' in data:
                if data['inventory_growth'] > data['revenue_growth'] * 2:
                    anomalies.append({
                        'type': 'inventory_spike',
                        'severity': 'medium',
                        'description': f"存货增速({data['inventory_growth']:.1%})超过营收增速2倍"
                    })

            # 4. 毛利率异常波动
            if 'gross_margin_change' in data:
                if abs(data['gross_margin_change']) > 0.10:  # 变动超过10%
                    anomalies.append({
                        'type': 'margin_volatility',
                        'severity': 'medium',
                        'description': f"毛利率异常波动{data['gross_margin_change']:+.1%}"
                    })

            # 5. 高负债风险
            if data.get('debt_ratio', 0) > 0.70:
                anomalies.append({
                    'type': 'high_debt',
                    'severity': 'high',
                    'description': f"资产负债率过高: {data['debt_ratio']:.1%}"
                })

            # 6. ROE下滑
            if 'roe_change' in data:
                if data['roe_change'] < -0.05:  # ROE下降超过5%
                    anomalies.append({
                        'type': 'roe_decline',
                        'severity': 'medium',
                        'description': f"ROE显著下降: {data['roe_change']:+.1%}"
                    })

            return anomalies

        except Exception as e:
            print(f"❌ 财务异常检测失败: {e}")
            return []

    def analyze(self, symbol: str, financial_data: Dict) -> Dict:
        """
        综合基本面分析

        参数:
            symbol: 股票代码
            financial_data: 财务数据

        返回:
            {
                'symbol': 股票代码,
                'profitability': {...},
                'growth': {...},
                'safety': {...},
                'valuation': {...},
                'anomalies': [...],
                'overall_score': 综合评分,
                'overall_rating': 综合评级,
                'recommendation': 投资建议
            }
        """
        try:
            # 分析各个维度
            profitability = self.analyze_profitability(financial_data)
            growth = self.analyze_growth(financial_data)
            safety = self.analyze_safety(financial_data)
            valuation = self.analyze_valuation(financial_data)

            # 检测异常
            anomalies = self.detect_anomalies(financial_data)

            # 计算综合评分
            weights = self.config['weights']
            overall_score = (
                profitability['score'] * weights['profitability'] +
                growth['score'] * weights['growth'] +
                safety['score'] * weights['safety'] +
                valuation['score'] * weights['valuation']
            )

            # 异常扣分
            for anomaly in anomalies:
                if anomaly['severity'] == 'high':
                    overall_score -= 10
                elif anomaly['severity'] == 'medium':
                    overall_score -= 5

            overall_score = max(min(overall_score, 100), 0)

            # 综合评级
            if overall_score >= 75:
                overall_rating = 'excellent'
                recommendation = '强烈推荐'
            elif overall_score >= 60:
                overall_rating = 'good'
                recommendation = '推荐关注'
            elif overall_score >= 40:
                overall_rating = 'fair'
                recommendation = '谨慎观察'
            else:
                overall_rating = 'poor'
                recommendation = '不建议投资'

            return {
                'symbol': symbol,
                'profitability': profitability,
                'growth': growth,
                'safety': safety,
                'valuation': valuation,
                'anomalies': anomalies,
                'overall_score': overall_score,
                'overall_rating': overall_rating,
                'recommendation': recommendation,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            print(f"❌ 综合分析失败: {e}")
            return {
                'symbol': symbol,
                'overall_score': 0,
                'overall_rating': 'error',
                'recommendation': '分析失败'
            }

    def screen_stocks(self, stock_list: List[Dict]) -> List[Dict]:
        """
        股票筛选

        参数:
            stock_list: 股票列表 [{'symbol': '600519', 'data': {...}}, ...]

        返回:
            符合条件的股票列表（按评分排序）
        """
        try:
            results = []

            for stock in stock_list:
                symbol = stock['symbol']
                data = stock['data']

                # 分析
                analysis = self.analyze(symbol, data)

                # 筛选条件
                if (analysis['overall_score'] >= 50 and
                    analysis['valuation']['valuation_level'] in ['undervalued', 'fair'] and
                    analysis['profitability']['roe'] >= self.config['roe_min'] and
                    len([a for a in analysis['anomalies'] if a['severity'] == 'high']) == 0):

                    results.append({
                        'symbol': symbol,
                        'score': analysis['overall_score'],
                        'rating': analysis['overall_rating'],
                        'recommendation': analysis['recommendation'],
                        'analysis': analysis
                    })

            # 按评分排序
            results.sort(key=lambda x: x['score'], reverse=True)

            return results

        except Exception as e:
            print(f"❌ 股票筛选失败: {e}")
            return []


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试 FundamentalAnalyzer 模块")
    print("=" * 80)

    # 创建模拟财务数据（贵州茅台）
    mock_data = {
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

    analyzer = FundamentalAnalyzer()

    print("\n【测试1】分析贵州茅台基本面")
    print("-" * 80)

    result = analyzer.analyze('600519', mock_data)

    print(f"\n盈利能力: {result['profitability']['description']}")
    print(f"  - ROE: {result['profitability']['roe']:.1%}")
    print(f"  - 毛利率: {result['profitability']['gross_margin']:.1%}")
    print(f"  - 净利率: {result['profitability']['net_margin']:.1%}")

    print(f"\n成长性: {result['growth']['description']}")
    print(f"  - 营收增长: {result['growth']['revenue_growth']:.1%}")
    print(f"  - 利润增长: {result['growth']['profit_growth']:.1%}")

    print(f"\n财务安全: {result['safety']['description']}")
    print(f"  - 资产负债率: {result['safety']['debt_ratio']:.1%}")
    print(f"  - 流动比率: {result['safety']['current_ratio']:.2f}")

    print(f"\n估值分析: {result['valuation']['description']}")
    print(f"  - PE: {result['valuation']['pe']:.1f}")
    print(f"  - PB: {result['valuation']['pb']:.1f}")
    print(f"  - PEG: {result['valuation']['peg']:.2f}")

    print(f"\n财务异常检测:")
    if result['anomalies']:
        for anomaly in result['anomalies']:
            print(f"  ⚠️  [{anomaly['severity'].upper()}] {anomaly['description']}")
    else:
        print(f"  ✅ 未检测到明显异常")

    print(f"\n综合评分: {result['overall_score']:.0f}/100")
    print(f"综合评级: {result['overall_rating'].upper()}")
    print(f"投资建议: {result['recommendation']}")

    print("\n【测试2】股票筛选")
    print("-" * 80)

    # 创建模拟股票列表
    stock_list = [
        {'symbol': '600519', 'data': mock_data},
        {'symbol': '000858', 'data': {**mock_data, 'pe': 18, 'roe': 0.22, 'debt_ratio': 0.35}},
        {'symbol': '000001', 'data': {**mock_data, 'pe': 8, 'roe': 0.12, 'debt_ratio': 0.95}}
    ]

    screened = analyzer.screen_stocks(stock_list)
    print(f"筛选结果: {len(screened)}只股票符合条件\n")

    for stock in screened:
        print(f"{stock['symbol']}: 评分{stock['score']:.0f} - {stock['recommendation']}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
