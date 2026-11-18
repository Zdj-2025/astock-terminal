#!/usr/bin/env python3
"""
信号聚合器 - 汇总多维度分析信号

职责:
1. 从技术面、基本面、资金面、ETF特有指标收集信号
2. 按配置权重归一化并加权汇总
3. 输出统一的综合评分和信号方向

作者: AI Assistant
日期: 2026-04-07
"""
import json
import os
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass, field

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.analyzers import TechnicalAnalyzer, FundamentalAnalyzer, MultiScaleMACDAnalyzer
from scripts.etf.analyzer import ETFAnalyzer
from scripts.core.data_fetcher_baostock import BaoStockFetcher
from scripts.core.data_fetcher import DataFetcher


@dataclass
class AggregatedSignal:
    """聚合信号结果"""
    symbol: str = ''
    name: str = ''

    # 各维度评分 (0-100)
    tech_score: float = 0.0
    fund_score: float = 0.0
    macd_resonance_score: float = 0.0
    etf_score: float = 0.0

    # 综合评分 (0-100)
    overall_score: float = 0.0

    # 信号方向和强度
    signal: str = 'hold'        # buy / sell / hold
    strength: float = 0.0       # 0-1
    confidence: str = '低'      # 高 / 中 / 低

    # 各维度详情
    tech_detail: Dict = field(default_factory=dict)
    fund_detail: Dict = field(default_factory=dict)
    macd_detail: Dict = field(default_factory=dict)
    etf_detail: Dict = field(default_factory=dict)

    # 基本面异常
    anomalies: List[Dict] = field(default_factory=list)

    # 元信息
    is_etf: bool = False
    data_source: str = ''
    error_messages: List[str] = field(default_factory=list)


class SignalAggregator:
    """
    信号聚合器

    汇总技术面、基本面、多尺度MACD、ETF特有指标，
    按配置权重计算综合评分。
    """

    # 默认权重配置
    DEFAULT_WEIGHTS = {
        'tech_weight': 0.30,         # 技术面权重
        'fund_weight': 0.30,         # 基本面权重
        'macd_resonance_weight': 0.25,  # 多尺度MACD共振权重
        'etf_weight': 0.15,          # ETF特有指标权重（仅ETF）
    }

    # 非ETF权重（无ETF维度时重新分配）
    STOCK_WEIGHTS = {
        'tech_weight': 0.35,
        'fund_weight': 0.35,
        'macd_resonance_weight': 0.30,
    }

    # ETF权重（v2: 每个维度独立，不重复计算）
    ETF_WEIGHTS = {
        'tech_weight': 0.25,              # 技术面（TechnicalAnalyzer独立计算）
        'fund_weight': 0.25,              # 板块趋势（ETFAnalyzer sector_trend）
        'macd_resonance_weight': 0.20,    # MACD共振
        'etf_weight': 0.30,              # 择时信号（ETFAnalyzer timing_signal）
    }

    def __init__(self, config_path: str = 'config/analysis_config.json'):
        """
        初始化信号聚合器

        参数:
            config_path: 分析配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.weights = self._resolve_weights()

        # 初始化分析器（延迟创建，按需使用）
        self._baostock_fetcher = None
        self._akshare_fetcher = None
        self._tech_analyzer = None
        self._fund_analyzer = None
        self._macd_analyzer = None
        self._etf_analyzer = None

        print("📊 初始化信号聚合器")
        etf_w = self.weights.get('etf_weight', 0) * 100
        weights_desc = (f"   权重配置: 技术{self.weights.get('tech_weight', 0)*100:.0f}% "
                        f"基本面{self.weights.get('fund_weight', 0)*100:.0f}% "
                        f"MACD共振{self.weights.get('macd_resonance_weight', 0)*100:.0f}%")
        if 'etf_weight' in self.weights:
            weights_desc += f" ETF{etf_w:.0f}%"
        print(weights_desc)

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _resolve_weights(self) -> Dict:
        """从配置解析权重，缺失时用默认值"""
        signal_cfg = self.config.get('signal_weights', {})
        return {
            'tech_weight': signal_cfg.get('tech_weight', self.DEFAULT_WEIGHTS['tech_weight']),
            'fund_weight': signal_cfg.get('fund_weight', self.DEFAULT_WEIGHTS['fund_weight']),
            'macd_resonance_weight': signal_cfg.get('macd_resonance_weight',
                                                     self.DEFAULT_WEIGHTS['macd_resonance_weight']),
            'etf_weight': signal_cfg.get('etf_weight', self.DEFAULT_WEIGHTS['etf_weight']),
        }

    # ── 分析器属性（懒加载）──

    @property
    def baostock(self):
        if self._baostock_fetcher is None:
            self._baostock_fetcher = BaoStockFetcher()
        return self._baostock_fetcher

    @property
    def akshare(self):
        if self._akshare_fetcher is None:
            self._akshare_fetcher = DataFetcher()
        return self._akshare_fetcher

    @property
    def tech_analyzer(self):
        if self._tech_analyzer is None:
            self._tech_analyzer = TechnicalAnalyzer('')
        return self._tech_analyzer

    @property
    def fund_analyzer(self):
        if self._fund_analyzer is None:
            self._fund_analyzer = FundamentalAnalyzer()
        return self._fund_analyzer

    @property
    def etf_analyzer(self):
        if self._etf_analyzer is None:
            self._etf_analyzer = ETFAnalyzer()
        return self._etf_analyzer

    # ── 主入口 ──

    def aggregate(
        self,
        symbol: str,
        name: str = '',
        kline_data=None,
        financial_data=None,
        skip_macd_resonance: bool = False
    ) -> AggregatedSignal:
        """
        聚合所有维度信号

        参数:
            symbol: 股票代码 (6位)
            name: 股票名称
            kline_data: 预加载的K线数据 (可选)
            financial_data: 预加载的财务数据 (可选)
            skip_macd_resonance: 是否跳过多尺度MACD（30min/5min数据获取可能失败）

        返回:
            AggregatedSignal 聚合信号
        """
        symbol = str(symbol).zfill(6)
        is_etf = symbol.startswith('51') or symbol.startswith('15')

        result = AggregatedSignal(symbol=symbol, name=name, is_etf=is_etf)

        print(f"\n📊 聚合信号: {symbol} {name} {'[ETF]' if is_etf else '[股票]'}")

        # 1. 获取K线数据
        if kline_data is None:
            kline_data = self._fetch_kline(symbol, is_etf)

        if kline_data is None or (hasattr(kline_data, 'empty') and kline_data.empty):
            result.error_messages.append('无法获取K线数据')
            print(f"   ⚠️  无法获取K线数据")
            return result

        # 2. 技术面分析
        self._analyze_technical(result, kline_data)

        # 3. 基本面分析
        if is_etf:
            self._analyze_etf_v2(result, symbol, name, kline_data)
        else:
            self._analyze_fundamental(result, symbol, financial_data)

        # 4. 多尺度MACD共振
        if not skip_macd_resonance:
            self._analyze_macd_resonance(result, symbol)

        # 5. 计算综合评分
        self._calculate_overall_score(result)

        return result

    # ── 数据获取 ──

    def _fetch_kline(self, symbol: str, is_etf: bool):
        """获取K线数据"""
        try:
            if is_etf:
                data = self.akshare.get_historical_kline(symbol, period='daily', days=120)
            else:
                data = self.baostock.get_historical_kline(
                    symbol, period='daily',
                    start_date='2025-01-01', end_date='2026-12-31'
                )
            if data is not None and not data.empty:
                result_data = data
                result_source = 'akshare' if is_etf else 'baostock'
                print(f"   ✅ K线数据: {len(data)}条 ({result_source})")
                return data
        except Exception as e:
            print(f"   ⚠️  K线数据获取失败: {str(e)[:50]}")
        return None

    # ── 各维度分析 ──

    def _analyze_technical(self, result: AggregatedSignal, kline_data):
        """技术面分析"""
        try:
            tech_result = self.tech_analyzer.analyze_indicators(kline_data)
            if tech_result:
                result.tech_detail = tech_result

                # 从overall提取评分
                overall = tech_result.get('overall', {})
                buy_score = overall.get('buy_score', 0)
                sell_score = overall.get('sell_score', 0)

                # 技术评分: buy_score和sell_score的综合
                if buy_score > sell_score:
                    result.tech_score = min(buy_score * 100, 100)
                else:
                    result.tech_score = max(100 - sell_score * 100, 0)

                print(f"   ✅ 技术面: {result.tech_score:.1f}分 "
                      f"(信号: {overall.get('signal', 'N/A')})")
        except Exception as e:
            result.error_messages.append(f'技术分析失败: {str(e)[:30]}')
            print(f"   ⚠️  技术分析失败: {str(e)[:50]}")

    def _analyze_fundamental(self, result: AggregatedSignal, symbol: str,
                             financial_data=None):
        """基本面分析（股票）"""
        try:
            if financial_data is None:
                financial_data = self.akshare.get_financial_data(symbol)

            if financial_data:
                fund_result = self.fund_analyzer.analyze(symbol, financial_data)
                if fund_result:
                    result.fund_detail = fund_result
                    result.fund_score = fund_result.get('overall_score', 50)
                    result.anomalies = fund_result.get('anomalies', [])
                    print(f"   ✅ 基本面: {result.fund_score:.1f}分 "
                          f"({fund_result.get('overall_rating', 'N/A')})")
                    return

            # 无法获取财务数据时使用中性评分
            result.fund_score = 50.0
            print(f"   ⚠️  基本面: 数据不可用，使用中性评分50分")
        except Exception as e:
            result.fund_score = 50.0
            result.error_messages.append(f'基本面分析失败: {str(e)[:30]}')
            print(f"   ⚠️  基本面分析失败: {str(e)[:50]}")

    def _analyze_etf(self, result: AggregatedSignal, symbol: str, name: str):
        """ETF综合分析"""
        try:
            etf_result = self.etf_analyzer.analyze(symbol, name)
            if etf_result:
                result.etf_detail = etf_result
                result.etf_score = etf_result.get('overall_score', 50)
                result.fund_score = etf_result.get('overall_score', 50)  # ETF基本面用整体评分替代

                # 提取技术评分（如果有）
                tech_analysis = etf_result.get('tech_analysis', {})
                result.tech_score = tech_analysis.get('score', 50)

                print(f"   ✅ ETF分析: {result.etf_score:.1f}分 "
                      f"({etf_result.get('rating', 'N/A')})")
                return
        except Exception as e:
            print(f"   ⚠️  ETF分析失败: {str(e)[:50]}")

        result.etf_score = 50.0
        result.fund_score = 50.0

    def _analyze_etf_v2(self, result: AggregatedSignal, symbol: str, name: str,
                        kline_data=None):
        """
        ETF分析v2 - 拆分为独立维度，修复重复计算bug

        v1的问题: overall_score同时赋给fund_score和etf_score，导致同一值被计算两次(45%)
        v2修复:
        - fund_score ← 板块趋势评分（ETF的"基本面"）
        - etf_score ← 择时信号评分（做差价的信号）
        - tech_score ← 保持步骤2中TechnicalAnalyzer的独立结果，不覆盖
        """
        try:
            # 从portfolio获取成本价
            cost_price = 0.0
            try:
                from scripts.core.portfolio_manager import PortfolioManager
                pm = PortfolioManager()
                portfolio = pm.load_portfolio()
                if not portfolio.empty:
                    match = portfolio[portfolio['股票代码'] == symbol]
                    if not match.empty:
                        cost_price = float(match.iloc[0].get('买入价格', 0))
            except Exception:
                pass

            etf_result = self.etf_analyzer.analyze(symbol, name, cost_price=cost_price)
            if etf_result:
                result.etf_detail = etf_result

                # 板块趋势 → 基本面评分槽
                result.fund_score = etf_result.get('core_position_score', 50)

                # 择时信号 → ETF专属评分槽
                result.etf_score = etf_result.get('trading_signal_score', 50)

                # 技术评分 → 保持步骤2中TechnicalAnalyzer的独立结果，不覆盖
                # （修复v1中覆盖tech_score的bug）

                print(f"   ✅ ETF v2: 板块趋势{result.fund_score:.0f} | "
                      f"择时{result.etf_score:.0f} | "
                      f"评级{etf_result.get('rating', 'N/A')}")
                return
        except Exception as e:
            print(f"   ⚠️  ETF v2分析失败: {str(e)[:50]}")

        result.etf_score = 50.0
        result.fund_score = 50.0

    def _analyze_macd_resonance(self, result: AggregatedSignal, symbol: str):
        """多尺度MACD共振分析"""
        try:
            analyzer = MultiScaleMACDAnalyzer(symbol)
            signal = analyzer.generate_signal()
            if signal:
                result.macd_detail = signal

                # 将MACD信号映射为0-100分
                macd_signal = signal.get('signal', 'hold')
                macd_strength = signal.get('strength', 0.5)

                if macd_signal == 'buy':
                    result.macd_resonance_score = 50 + macd_strength * 50
                elif macd_signal == 'sell':
                    result.macd_resonance_score = 50 - macd_strength * 50
                else:
                    result.macd_resonance_score = 50.0

                print(f"   ✅ MACD共振: {result.macd_resonance_score:.1f}分 "
                      f"(信号: {macd_signal}, 强度: {macd_strength:.2f})")
                return
        except Exception as e:
            print(f"   ⚠️  MACD共振分析失败（可能缺少分时数据）: {str(e)[:50]}")

        # MACD共振数据不可用时，用技术面MACD数据替代
        if result.tech_detail and 'macd' in result.tech_detail:
            macd_info = result.tech_detail['macd']
            macd_signal = macd_info.get('signal', 'hold')
            macd_strength = macd_info.get('strength', 0.5)

            if macd_signal in ('buy', 'golden_cross', 'strong_buy'):
                result.macd_resonance_score = 50 + macd_strength * 50
            elif macd_signal in ('sell', 'death_cross', 'strong_sell'):
                result.macd_resonance_score = 50 - macd_strength * 50
            else:
                result.macd_resonance_score = 50.0
            print(f"   ✅ MACD(日线): {result.macd_resonance_score:.1f}分 (回退至日线MACD)")
        else:
            result.macd_resonance_score = 50.0
            print(f"   ⚠️  MACD共振: 数据不可用，使用中性评分50分")

    # ── 综合评分计算 ──

    def _calculate_overall_score(self, result: AggregatedSignal):
        """
        计算综合评分

        评分逻辑:
        - 股票: 技术面 + 基本面 + MACD共振
        - ETF: 技术面 + ETF综合 + MACD共振
        """
        if result.is_etf:
            # ETF权重 v2（每个维度独立，不重复计算）
            w = self.ETF_WEIGHTS
            result.overall_score = (
                result.tech_score * w['tech_weight'] +
                result.fund_score * w['fund_weight'] +
                result.macd_resonance_score * w['macd_resonance_weight'] +
                result.etf_score * w['etf_weight']
            )
        else:
            # 股票权重（无ETF维度，重新分配）
            w = self.STOCK_WEIGHTS
            result.overall_score = (
                result.tech_score * w['tech_weight'] +
                result.fund_score * w['fund_weight'] +
                result.macd_resonance_score * w['macd_resonance_weight']
            )

        # 确定信号方向
        trading_cfg = self.config.get('trading_params', {})
        buy_threshold = trading_cfg.get('buy_threshold', 70)
        sell_threshold = trading_cfg.get('sell_threshold', 40)

        if result.overall_score >= buy_threshold:
            result.signal = 'buy'
        elif result.overall_score <= sell_threshold:
            result.signal = 'sell'
        else:
            result.signal = 'hold'

        # 计算信号强度
        if result.signal == 'buy':
            result.strength = min((result.overall_score - buy_threshold) / (100 - buy_threshold), 1.0)
        elif result.signal == 'sell':
            result.strength = min((sell_threshold - result.overall_score) / sell_threshold, 1.0)
        else:
            result.strength = 0.0

        # 置信度
        if result.strength >= 0.6:
            result.confidence = '高'
        elif result.strength >= 0.3:
            result.confidence = '中'
        else:
            result.confidence = '低'

        print(f"\n   📊 综合评分: {result.overall_score:.1f}分")
        print(f"      信号: {result.signal} | 强度: {result.strength:.2f} | 置信度: {result.confidence}")


# ── 测试 ──

if __name__ == '__main__':
    print("=" * 80)
    print("测试信号聚合器")
    print("=" * 80)

    aggregator = SignalAggregator()

    # 测试1: 普通股票
    print("\n" + "─" * 60)
    result = aggregator.aggregate('603881', '数据港', skip_macd_resonance=True)
    print(f"\n结果: {result.signal} | 评分: {result.overall_score:.1f} | "
          f"技术: {result.tech_score:.1f} | 基本面: {result.fund_score:.1f}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
