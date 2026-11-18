#!/usr/bin/env python3
"""
技术分析器模块
整合MACD、RSI、KDJ、布林带、均线等技术指标
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys
sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.core import DataFetcher


class TechnicalAnalyzer:
    """
    技术分析器

    整合多种技术指标：
    - MACD
    - RSI
    - KDJ
    - 布林带
    - 移动均线
    - 成交量

    示例:
        analyzer = TechnicalAnalyzer('600519')
        result = analyzer.analyze()
    """

    def __init__(self, symbol: str, fetcher: Optional[DataFetcher] = None):
        """
        初始化技术分析器

        参数:
            symbol: 股票代码
            fetcher: 数据获取器（可选）
        """
        self.symbol = symbol
        self.fetcher = fetcher or DataFetcher()

        print(f"✅ TechnicalAnalyzer初始化成功: {symbol}")

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算RSI指标

        参数:
            df: 包含close列的DataFrame
            period: RSI周期

        返回:
            DataFrame包含RSI列
        """
        try:
            # 计算价格变化
            delta = df['close'].diff()

            # 分离上涨和下跌 (Wilder EMA)
            alpha = 1.0 / period
            gain = (delta.where(delta > 0, 0)).ewm(alpha=alpha, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=alpha, adjust=False).mean()

            # 计算RS
            rs = gain / loss

            # 计算RSI
            rsi = 100 - (100 / (1 + rs))

            result = df.copy()
            result['rsi'] = rsi

            return result

        except Exception as e:
            print(f"❌ RSI计算失败: {e}")
            return df

    def calculate_kdj(self, df: pd.DataFrame, n: int = 9, m: int = 3) -> pd.DataFrame:
        """
        计算KDJ指标

        参数:
            df: 包含high, low, close列的DataFrame
            n: KDJ参数N
            m: KDJ参数M

        返回:
            DataFrame包含K, D, J列
        """
        try:
            # 计算RSV
            low_n = df['low'].rolling(window=n).min()
            high_n = df['high'].rolling(window=n).max()

            rsv = (df['close'] - low_n) / (high_n - low_n).replace(0, np.nan) * 100

            # 计算K, D, J
            k = rsv.ewm(com=m-1, adjust=False).mean()
            d = k.ewm(com=m-1, adjust=False).mean()
            j = 3 * k - 2 * d

            result = df.copy()
            result['k'] = k
            result['d'] = d
            result['j'] = j

            return result

        except Exception as e:
            print(f"❌ KDJ计算失败: {e}")
            return df

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2
    ) -> pd.DataFrame:
        """
        计算布林带

        参数:
            df: 包含close列的DataFrame
            period: 周期
            std_dev: 标准差倍数

        返回:
            DataFrame包含布林带上轨、中轨、下轨
        """
        try:
            # 中轨（移动平均）
            middle = df['close'].rolling(window=period).mean()

            # 标准差
            std = df['close'].rolling(window=period).std(ddof=0)

            # 上轨和下轨
            upper = middle + std_dev * std
            lower = middle - std_dev * std

            result = df.copy()
            result['boll_middle'] = middle
            result['boll_upper'] = upper
            result['boll_lower'] = lower

            return result

        except Exception as e:
            print(f"❌ 布林带计算失败: {e}")
            return df

    def calculate_ma(self, df: pd.DataFrame, periods: list = [20, 30, 60]) -> pd.DataFrame:
        """
        计算移动均线

        参数:
            df: 包含close列的DataFrame
            periods: 周期列表

        返回:
            DataFrame包含各周期均线
        """
        try:
            result = df.copy()

            for period in periods:
                result[f'ma{period}'] = df['close'].rolling(window=period).mean()

            return result

        except Exception as e:
            print(f"❌ 均线计算失败: {e}")
            return df

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        计算MACD指标

        参数:
            df: 包含close列的DataFrame
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        返回:
            DataFrame包含MACD、Signal、Histogram列
        """
        try:
            # 计算EMA
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

            # MACD线
            macd = ema_fast - ema_slow

            # 信号线
            signal_line = macd.ewm(span=signal, adjust=False).mean()

            # 柱状图（A股惯例：2倍DIF-DEA差值）
            histogram = 2 * (macd - signal_line)

            result = df.copy()
            result['macd'] = macd
            result['signal'] = signal_line
            result['histogram'] = histogram

            return result

        except Exception as e:
            print(f"❌ MACD计算失败: {e}")
            return df

    def analyze_indicators(self, df: pd.DataFrame) -> Dict:
        """
        分析所有技术指标

        参数:
            df: K线数据

        返回:
            各指标的分析结果
        """
        try:
            # 计算所有指标
            df = self.calculate_macd(df)
            df = self.calculate_rsi(df)
            df = self.calculate_kdj(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_ma(df)

            # 获取最新数据
            latest = df.iloc[-1]

            # 分析MACD
            macd_signal = self._analyze_macd(df)

            # 分析RSI
            rsi_signal = self._analyze_rsi(latest['rsi'])

            # 分析KDJ
            kdj_signal = self._analyze_kdj(latest['k'], latest['d'], latest['j'])

            # 分析布林带
            boll_signal = self._analyze_bollinger(
                latest['close'],
                latest['boll_upper'],
                latest['boll_middle'],
                latest['boll_lower']
            )

            # 分析均线
            ma_signal = self._analyze_ma(latest, df)

            # 综合信号
            overall_signal = self._generate_overall_signal(
                macd_signal,
                rsi_signal,
                kdj_signal,
                boll_signal,
                ma_signal
            )

            return {
                'macd': macd_signal,
                'rsi': rsi_signal,
                'kdj': kdj_signal,
                'bollinger': boll_signal,
                'moving_averages': ma_signal,
                'overall': overall_signal,
                'latest_values': {
                    'close': latest['close'],
                    'macd': latest['macd'],
                    'rsi': latest['rsi'],
                    'k': latest['k'],
                    'd': latest['d'],
                    'j': latest['j']
                }
            }

        except Exception as e:
            print(f"❌ 技术指标分析失败: {e}")
            return {}

    def _analyze_macd(self, df: pd.DataFrame) -> Dict:
        """分析MACD信号"""
        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]

            # MACD趋势
            if latest['histogram'] > 0:
                trend = 'bullish'
                if latest['histogram'] > previous['histogram']:
                    signal = 'strong_buy'
                    strength = 0.8
                else:
                    signal = 'buy'
                    strength = 0.6
            elif latest['histogram'] < 0:
                trend = 'bearish'
                if latest['histogram'] < previous['histogram']:
                    signal = 'strong_sell'
                    strength = 0.8
                else:
                    signal = 'sell'
                    strength = 0.6
            else:
                trend = 'neutral'
                signal = 'hold'
                strength = 0.3

            # 检测金叉/死叉
            if previous['macd'] <= previous['signal'] and latest['macd'] > latest['signal']:
                signal = 'golden_cross'
                strength = 0.9
                description = "MACD金叉（买入信号）"
            elif previous['macd'] >= previous['signal'] and latest['macd'] < latest['signal']:
                signal = 'death_cross'
                strength = 0.9
                description = "MACD死叉（卖出信号）"
            else:
                description = f"MACD趋势: {trend}"

            return {
                'signal': signal,
                'strength': strength,
                'trend': trend,
                'description': description,
                'values': {
                    'macd': latest['macd'],
                    'signal': latest['signal'],
                    'histogram': latest['histogram']
                }
            }

        except Exception as e:
            return {'signal': 'hold', 'strength': 0, 'description': f'分析失败: {e}'}

    def _analyze_rsi(self, rsi: float) -> Dict:
        """分析RSI信号"""
        try:
            if rsi > 70:
                signal = 'sell'
                strength = min((rsi - 70) / 30, 1.0)
                description = f"RSI超买 ({rsi:.2f}) - 卖出信号"
            elif rsi < 30:
                signal = 'buy'
                strength = min((30 - rsi) / 30, 1.0)
                description = f"RSI超卖 ({rsi:.2f}) - 买入信号"
            else:
                signal = 'hold'
                strength = 0.3
                description = f"RSI正常 ({rsi:.2f})"

            return {
                'signal': signal,
                'strength': strength,
                'description': description,
                'value': rsi
            }

        except Exception as e:
            return {'signal': 'hold', 'strength': 0, 'description': f'分析失败: {e}'}

    def _analyze_kdj(self, k: float, d: float, j: float) -> Dict:
        """分析KDJ信号"""
        try:
            # KDJ金叉/死叉
            if k > d:
                signal = 'buy'
                strength = min((k - d) / 20, 1.0)
                description = f"KDJ金叉 (K:{k:.2f} > D:{d:.2f}) - 买入信号"
            elif k < d:
                signal = 'sell'
                strength = min((d - k) / 20, 1.0)
                description = f"KDJ死叉 (K:{k:.2f} < D:{d:.2f}) - 卖出信号"
            else:
                signal = 'hold'
                strength = 0.3
                description = f"KDJ持平"

            # J值判断
            if j > 100:
                strength = min(strength + 0.2, 1.0)
                description += f"，J值超买 ({j:.2f})"
            elif j < 0:
                strength = min(strength + 0.2, 1.0)
                description += f"，J值超卖 ({j:.2f})"

            return {
                'signal': signal,
                'strength': strength,
                'description': description,
                'values': {'k': k, 'd': d, 'j': j}
            }

        except Exception as e:
            return {'signal': 'hold', 'strength': 0, 'description': f'分析失败: {e}'}

    def _analyze_bollinger(
        self,
        price: float,
        upper: float,
        middle: float,
        lower: float
    ) -> Dict:
        """分析布林带信号"""
        try:
            if price >= upper:
                signal = 'sell'
                strength = 0.8
                description = f"价格触及布林上轨 ({price:.2f} >= {upper:.2f}) - 卖出信号"
            elif price <= lower:
                signal = 'buy'
                strength = 0.8
                description = f"价格触及布林下轨 ({price:.2f} <= {lower:.2f}) - 买入信号"
            elif price > middle:
                signal = 'hold'
                strength = 0.4
                description = f"价格在布林中轨上方 ({price:.2f} > {middle:.2f})"
            else:
                signal = 'hold'
                strength = 0.4
                description = f"价格在布林中轨下方 ({price:.2f} < {middle:.2f})"

            return {
                'signal': signal,
                'strength': strength,
                'description': description,
                'values': {'upper': upper, 'middle': middle, 'lower': lower}
            }

        except Exception as e:
            return {'signal': 'hold', 'strength': 0, 'description': f'分析失败: {e}'}

    def _analyze_ma(self, latest: pd.Series, df: pd.DataFrame) -> Dict:
        """分析均线信号"""
        try:
            price = latest['close']
            ma20 = latest['ma20']
            ma30 = latest['ma30']
            ma60 = latest['ma60']

            # 多头排列：MA20 > MA30 > MA60
            if ma20 > ma30 > ma60:
                signal = 'buy'
                strength = 0.8
                description = "均线多头排列 - 上涨趋势"
            # 空头排列：MA20 < MA30 < MA60
            elif ma20 < ma30 < ma60:
                signal = 'sell'
                strength = 0.8
                description = "均线空头排列 - 下跌趋势"
            else:
                signal = 'hold'
                strength = 0.4
                description = "均线缠绕 - 趋势不明"

            # 价格与均线关系
            if price > ma20:
                description += f"，价格在MA20上方"
            else:
                description += f"，价格在MA20下方"

            return {
                'signal': signal,
                'strength': strength,
                'description': description,
                'values': {'ma20': ma20, 'ma30': ma30, 'ma60': ma60}
            }

        except Exception as e:
            return {'signal': 'hold', 'strength': 0, 'description': f'分析失败: {e}'}

    def _generate_overall_signal(self, *signals) -> Dict:
        """生成综合信号"""
        try:
            # 权重
            weights = {
                'macd': 0.3,
                'rsi': 0.2,
                'kdj': 0.2,
                'bollinger': 0.15,
                'moving_averages': 0.15
            }

            # 计算加权信号
            buy_score = 0
            sell_score = 0

            for i, (name, signal) in enumerate(zip(weights.keys(), signals)):
                if signal['signal'] == 'buy' or signal['signal'] == 'strong_buy' or signal['signal'] == 'golden_cross':
                    buy_score += weights[name] * signal['strength']
                elif signal['signal'] == 'sell' or signal['signal'] == 'strong_sell' or signal['signal'] == 'death_cross':
                    sell_score += weights[name] * signal['strength']

            # 判断最终信号
            if buy_score > sell_score and buy_score > 0.6:
                overall_signal = 'buy'
                strength = buy_score
            elif sell_score > buy_score and sell_score > 0.6:
                overall_signal = 'sell'
                strength = sell_score
            else:
                overall_signal = 'hold'
                strength = max(buy_score, sell_score)

            return {
                'signal': overall_signal,
                'strength': strength,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'description': f"综合评分: 买入{buy_score:.2f} vs 卖出{sell_score:.2f}"
            }

        except Exception as e:
            return {'signal': 'hold', 'strength': 0, 'description': f'综合分析失败: {e}'}


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试 TechnicalAnalyzer 模块")
    print("=" * 80)

    # 创建模拟数据
    print("\n【创建模拟数据】")
    print("-" * 80)

    dates = pd.date_range(start='2025-01-01', end='2026-04-03', freq='D')
    n = len(dates)

    # 模拟贵州茅台走势
    base_price = 1500
    trend = np.linspace(0, 100, n)  # 上涨趋势
    noise = np.random.randn(n) * 30  # 随机波动
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.randn(n) * 0.02),
        'high': close_prices * (1 + np.abs(np.random.randn(n) * 0.03)),
        'low': close_prices * (1 - np.abs(np.random.randn(n) * 0.03)),
        'close': close_prices,
        'volume': np.random.randint(10000, 50000, n)
    })

    print(f"✅ 创建了 {len(df)} 天的模拟数据")
    print(f"价格范围: ¥{df['close'].min():.2f} - ¥{df['close'].max():.2f}")

    # 分析
    print("\n【技术指标分析】")
    print("-" * 80)

    analyzer = TechnicalAnalyzer('600519')
    result = analyzer.analyze_indicators(df)

    if result:
        print("\n【MACD分析】")
        print(f"  {result['macd']['description']}")
        print(f"  信号: {result['macd']['signal']}, 强度: {result['macd']['strength']:.2f}")

        print("\n【RSI分析】")
        print(f"  {result['rsi']['description']}")
        print(f"  信号: {result['rsi']['signal']}, 强度: {result['rsi']['strength']:.2f}")

        print("\n【KDJ分析】")
        print(f"  {result['kdj']['description']}")
        print(f"  信号: {result['kdj']['signal']}, 强度: {result['kdj']['strength']:.2f}")

        print("\n【布林带分析】")
        print(f"  {result['bollinger']['description']}")
        print(f"  信号: {result['bollinger']['signal']}, 强度: {result['bollinger']['strength']:.2f}")

        print("\n【均线分析】")
        print(f"  {result['moving_averages']['description']}")
        print(f"  信号: {result['moving_averages']['signal']}, 强度: {result['moving_averages']['strength']:.2f}")

        print("\n【综合信号】")
        print(f"  {result['overall']['description']}")
        print(f"  ✅ 最终信号: {result['overall']['signal'].upper()}")
        print(f"  ✅ 信号强度: {result['overall']['strength']:.2f}")

        print("\n【当前指标值】")
        latest = result['latest_values']
        print(f"  收盘价: ¥{latest['close']:.2f}")
        print(f"  MACD: {latest['macd']:.4f}")
        print(f"  RSI: {latest['rsi']:.2f}")
        print(f"  K: {latest['k']:.2f}, D: {latest['d']:.2f}, J: {latest['j']:.2f}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
