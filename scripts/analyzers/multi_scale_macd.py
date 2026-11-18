#!/usr/bin/env python3
"""
多时间尺度MACD分析模块
三档时间尺度：周线(战略层)、日线(战术层)、60分钟(执行层)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.core import DataFetcher


class MultiScaleMACDAnalyzer:
    """
    多时间尺度MACD分析器（三档框架）

    在三个时间维度上分析MACD：
    - 周线级别（战略层）：确定大趋势方向
    - 日线级别（战术层）：确认趋势，识别做T区域
    - 60分钟级别（执行层）：精准T买卖点

    当三个尺度方向一致时，产生共振信号（最强）

    示例:
        analyzer = MultiScaleMACDAnalyzer('600519')
        result = analyzer.analyze_resonance()
        signal = analyzer.generate_signal()
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[Dict] = None,
        fetcher: Optional[DataFetcher] = None
    ):
        """
        初始化多时间尺度MACD分析器

        参数:
            symbol: 股票代码
            config: 配置参数（可选）
            fetcher: 数据获取器（可选）
        """
        self.symbol = symbol
        self.config = config or self._default_config()
        self.fetcher = fetcher or DataFetcher()

        # MACD参数
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)

        # 分析尺度（三档框架：周线/日线/60分钟）
        self.scales = ['weekly', 'daily', '60min']

        # 数据存储
        self.data = {}
        self.macd_data = {}

        print(f"✅ MultiScaleMACDAnalyzer初始化成功: {symbol}")
        print(f"   - MACD参数: {self.fast_period}, {self.slow_period}, {self.signal_period}")
        print(f"   - 分析尺度: {', '.join(self.scales)}")

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'oversold_threshold': -0.5,   # MACD柱状图超卖阈值
            'overbought_threshold': 0.5,  # MACD柱状图超买阈值
            'divergence_lookback': 20      # 背离检测回溯期
        }

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD指标

        参数:
            df: 包含close列的DataFrame

        返回:
            DataFrame包含MACD、Signal、Histogram列
        """
        try:
            # 计算EMA
            ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()

            # MACD线
            macd = ema_fast - ema_slow

            # 信号线
            signal = macd.ewm(span=self.signal_period, adjust=False).mean()

            # 柱状图（A股惯例：2倍DIF-DEA差值）
            histogram = 2 * (macd - signal)

            # 添加到DataFrame
            result = df.copy()
            result['macd'] = macd
            result['signal'] = signal
            result['histogram'] = histogram

            return result

        except Exception as e:
            print(f"❌ MACD计算失败: {e}")
            return df

    def analyze_trend(self, macd_df: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        分析MACD趋势

        参数:
            macd_df: 包含MACD数据的DataFrame
            lookback: 回溯期数

        返回:
            {
                'trend': 'bullish/bearish/neutral',
                'strength': 0-1,
                'macd_value': 当前MACD值,
                'signal_value': 当前信号线值,
                'histogram': 当前柱状图值
            }
        """
        try:
            if len(macd_df) < lookback:
                return {'trend': 'neutral', 'strength': 0}

            # 获取最近数据
            recent = macd_df.tail(lookback)

            # 当前值
            current_macd = recent['macd'].iloc[-1]
            current_signal = recent['signal'].iloc[-1]
            current_histogram = recent['histogram'].iloc[-1]

            # 趋势判断
            # 1. MACD在信号线上方 -> 多头
            # 2. MACD在信号线下方 -> 空头
            # 3. 柱状图变化趋势

            # 计算柱状图变化
            histogram_changes = recent['histogram'].diff().dropna()
            positive_changes = (histogram_changes > 0).sum()
            negative_changes = (histogram_changes < 0).sum()

            # 趋势判断
            if current_macd > current_signal:
                if positive_changes > negative_changes:
                    trend = 'bullish'
                    strength = min(positive_changes / lookback, 1.0)
                else:
                    trend = 'bullish_weak'
                    strength = 0.5
            elif current_macd < current_signal:
                if negative_changes > positive_changes:
                    trend = 'bearish'
                    strength = min(negative_changes / lookback, 1.0)
                else:
                    trend = 'bearish_weak'
                    strength = 0.5
            else:
                trend = 'neutral'
                strength = 0

            return {
                'trend': trend,
                'strength': strength,
                'macd_value': current_macd,
                'signal_value': current_signal,
                'histogram': current_histogram
            }

        except Exception as e:
            print(f"❌ 趋势分析失败: {e}")
            return {'trend': 'neutral', 'strength': 0}

    def detect_divergence(self, macd_df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        检测MACD背离

        顶背离：价格创新高但MACD不创新高（卖出信号）
        底背离：价格创新低但MACD不创新低（买入信号）

        参数:
            macd_df: 包含MACD数据的DataFrame
            lookback: 回溯期数

        返回:
            {
                'top_divergence': bool,
                'bottom_divergence': bool,
                'strength': 0-1,
                'description': 描述
            }
        """
        try:
            if len(macd_df) < lookback:
                return {
                    'top_divergence': False,
                    'bottom_divergence': False,
                    'strength': 0,
                    'description': '数据不足'
                }

            recent = macd_df.tail(lookback)

            # 寻找价格和MACD的峰值/谷值
            # 简化版本：比较最近两个高点/低点

            # 找价格高点和低点
            price_high_idx = recent['high'].idxmax()
            price_low_idx = recent['low'].idxmin()

            # 找MACD高点和低点
            macd_high_idx = recent['macd'].idxmax()
            macd_low_idx = recent['macd'].idxmin()

            # 检测顶背离
            top_divergence = False
            if price_high_idx == recent.index[-1] or \
               (recent['high'].iloc[-1] > recent['high'].iloc[-5]):
                # 价格创新高或接近新高
                if recent['macd'].iloc[-1] < recent.loc[macd_high_idx, 'macd']:
                    # 但MACD没创新高
                    top_divergence = True

            # 检测底背离
            bottom_divergence = False
            if price_low_idx == recent.index[-1] or \
               (recent['low'].iloc[-1] < recent['low'].iloc[-5]):
                # 价格创新低或接近新低
                if recent['macd'].iloc[-1] > recent.loc[macd_low_idx, 'macd']:
                    # 但MACD没创新低
                    bottom_divergence = True

            # 计算强度
            strength = 0
            if top_divergence or bottom_divergence:
                # 根据背离程度计算强度
                if top_divergence:
                    price_diff = (recent['high'].iloc[-1] - recent.loc[price_high_idx, 'high']) / recent.loc[price_high_idx, 'high']
                    macd_diff = (recent.loc[macd_high_idx, 'macd'] - recent['macd'].iloc[-1]) / abs(recent.loc[macd_high_idx, 'macd'])
                    strength = min((price_diff + macd_diff) / 2, 1.0)
                elif bottom_divergence:
                    price_diff = (recent.loc[price_low_idx, 'low'] - recent['low'].iloc[-1]) / recent.loc[price_low_idx, 'low']
                    macd_diff = (recent['macd'].iloc[-1] - recent.loc[macd_low_idx, 'macd']) / abs(recent.loc[macd_low_idx, 'macd'])
                    strength = min((price_diff + macd_diff) / 2, 1.0)

            # 生成描述
            description = ""
            if top_divergence:
                description = f"检测到顶背离：价格创新高但MACD不创新高，强度{strength:.2f}（卖出信号）"
            elif bottom_divergence:
                description = f"检测到底背离：价格创新低但MACD不创新低，强度{strength:.2f}（买入信号）"
            else:
                description = "未检测到明显背离"

            return {
                'top_divergence': top_divergence,
                'bottom_divergence': bottom_divergence,
                'strength': strength,
                'description': description
            }

        except Exception as e:
            print(f"❌ 背离检测失败: {e}")
            return {
                'top_divergence': False,
                'bottom_divergence': False,
                'strength': 0,
                'description': f'检测失败: {e}'
            }

    def analyze_resonance(self) -> Dict:
        """
        分析多时间尺度MACD共振

        返回:
            {
                'daily': {...},
                '30min': {...},
                '5min': {...},
                'resonance': {
                    'strength': 共振强度（0-1）,
                    'scales': 共振的尺度列表,
                    'signal': buy/sell/hold,
                    'description': 描述
                }
            }
        """
        try:
            results = {}

            # 获取并分析每个尺度的数据
            for scale in self.scales:
                # 获取K线数据
                df = self.fetcher.get_historical_kline(
                    self.symbol,
                    period=scale,
                    start_date='20250101'  # 获取最近3个月数据
                )

                if df.empty:
                    print(f"⚠️  无法获取{scale}数据")
                    continue

                # 计算MACD
                macd_df = self.calculate_macd(df)

                # 分析趋势
                trend = self.analyze_trend(macd_df)

                # 检测背离
                divergence = self.detect_divergence(macd_df)

                results[scale] = {
                    'trend': trend,
                    'divergence': divergence,
                    'data_points': len(df)
                }

            # 分析共振
            resonance = self._analyze_resonance_strength(results)

            return {
                **results,
                'resonance': resonance
            }

        except Exception as e:
            print(f"❌ 共振分析失败: {e}")
            return {'resonance': {'strength': 0, 'signal': 'hold'}}

    def _analyze_resonance_strength(self, results: Dict) -> Dict:
        """
        分析共振强度

        参数:
            results: 各尺度的分析结果

        返回:
            {
                'strength': 共振强度（0-1）,
                'scales': 共振的尺度列表,
                'signal': buy/sell/hold,
                'description': 描述
            }
        """
        try:
            # 统计各趋势的尺度数量
            bullish_scales = []
            bearish_scales = []

            for scale, data in results.items():
                if 'trend' not in data:
                    continue

                trend = data['trend']['trend']
                if 'bullish' in trend:
                    bullish_scales.append(scale)
                elif 'bearish' in trend:
                    bearish_scales.append(scale)

            # 判断共振
            if len(bullish_scales) >= 3:
                # 三尺度多头共振
                strength = 1.0
                signal = 'buy'
                description = f"✅ 三尺度多头共振（{', '.join(bullish_scales)}）- 强烈买入信号"

            elif len(bearish_scales) >= 3:
                # 三尺度空头共振
                strength = 1.0
                signal = 'sell'
                description = f"❌ 三尺度空头共振（{', '.join(bearish_scales)}）- 强烈卖出信号"

            elif len(bullish_scales) == 2:
                # 两尺度多头共振
                strength = 0.7
                signal = 'buy'
                description = f"📈 两尺度多头共振（{', '.join(bullish_scales)}）- 中等买入信号"

            elif len(bearish_scales) == 2:
                # 两尺度空头共振
                strength = 0.7
                signal = 'sell'
                description = f"📉 两尺度空头共振（{', '.join(bearish_scales)}）- 中等卖出信号"

            elif len(bullish_scales) == 1:
                # 单尺度多头
                strength = 0.4
                signal = 'hold'
                description = f"➡️  仅{bullish_scales[0]}显示多头 - 信号较弱，建议观望"

            elif len(bearish_scales) == 1:
                # 单尺度空头
                strength = 0.4
                signal = 'hold'
                description = f"➡️  仅{bearish_scales[0]}显示空头 - 信号较弱，建议观望"

            else:
                # 无明显趋势
                strength = 0
                signal = 'hold'
                description = "⏸️  各尺度趋势不明确 - 建议观望"

            # 检查背离信号
            divergence_boost = 0
            for scale, data in results.items():
                if 'divergence' in data:
                    div = data['divergence']
                    if div['bottom_divergence']:
                        divergence_boost = max(divergence_boost, div['strength'])
                        if signal == 'buy':
                            description += f"\n⚡ {scale}检测到底背离，增强买入信号"
                    elif div['top_divergence']:
                        divergence_boost = max(divergence_boost, div['strength'])
                        if signal == 'sell':
                            description += f"\n⚡ {scale}检测到顶背离，增强卖出信号"

            # 调整强度（背离可增强信号）
            strength = min(strength + divergence_boost * 0.3, 1.0)

            return {
                'strength': strength,
                'bullish_scales': bullish_scales,
                'bearish_scales': bearish_scales,
                'signal': signal,
                'description': description
            }

        except Exception as e:
            print(f"❌ 共振强度分析失败: {e}")
            return {
                'strength': 0,
                'signal': 'hold',
                'description': f'分析失败: {e}'
            }

    def generate_signal(self) -> Dict:
        """
        生成综合交易信号

        返回:
            {
                'symbol': 股票代码,
                'signal': buy/sell/hold,
                'strength': 信号强度（0-1）,
                'confidence': 置信度（0-1）,
                'entry_price': 建议入场价（当前价）,
                'stop_loss': 止损价（-8%）,
                'take_profit': 止盈价（+15%）,
                'description': 详细描述
            }
        """
        try:
            # 获取共振分析结果
            analysis = self.analyze_resonance()
            resonance = analysis.get('resonance', {})

            # 获取当前价格
            quote = self.fetcher.get_realtime_quote(self.symbol)
            current_price = quote.get('price', 0) if quote else 0

            # 如果无法获取当前价格，使用日线最新价格
            if current_price == 0 and 'daily' in analysis:
                # 从日线数据获取最新价格
                df = self.fetcher.get_historical_kline(self.symbol, period='daily')
                if not df.empty:
                    current_price = df['close'].iloc[-1]

            # 生成信号
            signal = resonance.get('signal', 'hold')
            strength = resonance.get('strength', 0)

            # 计算置信度（基于共振强度和数据质量）
            confidence = strength * 0.7  # 基础置信度

            # 数据质量加成
            data_quality_bonus = 0
            for scale in self.scales:
                if scale in analysis and analysis[scale].get('data_points', 0) > 0:
                    data_quality_bonus += 0.1

            confidence = min(confidence + data_quality_bonus, 1.0)

            # 计算止损止盈
            if current_price > 0:
                stop_loss = current_price * 0.92  # -8%
                take_profit = current_price * 1.15  # +15%
            else:
                stop_loss = 0
                take_profit = 0

            # 生成详细描述
            description = f"""
股票: {self.symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{resonance.get('description', '无分析结果')}

信号强度: {strength:.2f} (0-1)
置信度: {confidence:.2f} (0-1)

当前价格: ¥{current_price:.2f}
止损价格: ¥{stop_loss:.2f} (-8%)
止盈价格: ¥{take_profit:.2f} (+15%)

建议操作: {signal.upper()}
"""

            return {
                'symbol': self.symbol,
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'description': description,
                'analysis': analysis
            }

        except Exception as e:
            print(f"❌ 信号生成失败: {e}")
            return {
                'symbol': self.symbol,
                'signal': 'hold',
                'strength': 0,
                'confidence': 0,
                'description': f'生成失败: {e}'
            }


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试 MultiScaleMACDAnalyzer 模块")
    print("=" * 80)

    # 测试股票：贵州茅台
    symbol = '600519'

    analyzer = MultiScaleMACDAnalyzer(symbol)

    # 测试1: 分析共振
    print(f"\n【测试1】分析 {symbol} 多时间尺度MACD共振")
    print("-" * 80)
    result = analyzer.analyze_resonance()

    print(f"\n共振分析结果:")
    resonance = result.get('resonance', {})
    print(resonance.get('description', ''))

    # 测试2: 生成交易信号
    print(f"\n【测试2】生成 {symbol} 交易信号")
    print("-" * 80)
    signal = analyzer.generate_signal()
    print(signal['description'])

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
