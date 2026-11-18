#!/usr/bin/env python3
"""
多时间尺度均线分析模块

支持日线/60分钟/15分钟的均线系统，用于：
- 趋势判断（多头/空头排列）
- 支撑/压力位识别（带强度评分）
- 金叉/死叉检测
- 跨时间尺度信号共振
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.core import DataFetcher


TIMEFRAME_MA_PERIODS = {
    'daily': [5, 10, 20, 60, 120],
    '60min': [20, 60, 120],
    '15min': [5, 10, 20, 60],
}

STRENGTH_WEIGHTS = {
    120: 1.0,
    60: 1.0,
    20: 0.7,
    10: 0.4,
    5: 0.4,
}

DEFAULT_LOOKBACK = {
    'daily': 250,
    '60min': 500,
    '15min': 500,
}


class MAAnalyzer:
    """
    多时间尺度均线分析器

    计算均线值、识别支撑/压力位、检测金叉死叉、判断均线排列状态。

    示例:
        analyzer = MAAnalyzer('603881')
        result = analyzer.analyze(timeframes=['daily', '60min'])
        print(analyzer.format_report(result))
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[Dict] = None,
        fetcher: Optional[DataFetcher] = None,
    ):
        self.symbol = symbol
        self.config = config or {}
        self.fetcher = fetcher or DataFetcher()
        self.ma_periods = self.config.get('ma_periods', TIMEFRAME_MA_PERIODS)
        self.lookback = self.config.get('lookback', DEFAULT_LOOKBACK)

    def analyze(
        self,
        timeframes: Optional[List[str]] = None,
        lookback_override: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """
        多时间尺度MA分析。

        Args:
            timeframes: 默认 ['daily', '60min']
            lookback_override: 覆盖默认回溯长度

        Returns:
            分析结果字典
        """
        if timeframes is None:
            timeframes = ['daily', '60min']

        results = {
            'symbol': self.symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timeframes': {},
        }

        for tf in timeframes:
            lb = (lookback_override or self.lookback).get(tf, DEFAULT_LOOKBACK.get(tf, 250))
            df = self._fetch_data(tf, lb)
            if df is None or len(df) < 10:
                print(f"⚠️  {tf}数据不足，跳过")
                continue
            results['timeframes'][tf] = self.analyze_single_timeframe(df, tf)

        results['signals'] = self.generate_signals(results)
        return results

    def analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        单时间尺度MA分析（核心计算方法）。

        Args:
            df: K线数据，需要 close 列
            timeframe: 'daily' / '60min' / '15min'

        Returns:
            该时间尺度的分析结果
        """
        periods = self.ma_periods.get(timeframe, TIMEFRAME_MA_PERIODS.get(timeframe, [20, 60]))
        max_period = max(periods)

        # 数据不足时只用可用的周期
        available = [p for p in periods if len(df) >= p + 5]
        if not available:
            return {'error': f'数据不足，需要至少{max_period + 5}条，实际{len(df)}条'}

        # 计算MA
        df_ma = self._calc_ma_values(df, available)
        current_price = float(df_ma['close'].iloc[-1])

        ma_values = {}
        for p in available:
            val = df_ma[f'ma{p}'].iloc[-1]
            if not np.isnan(val):
                ma_values[p] = float(val)

        result = {
            'current_price': current_price,
            'data_points': len(df),
            'ma_values': ma_values,
            'alignment': self._determine_alignment(ma_values, available),
            'price_position': self._determine_price_position(current_price, ma_values, available),
            'support_levels': [],
            'resistance_levels': [],
            'crossovers': self._detect_crossovers(df_ma, available),
            'trend_strength': 0.0,
        }

        # 支撑/压力位
        sr = self._calc_support_resistance(current_price, ma_values, available)
        result['support_levels'] = sr['support']
        result['resistance_levels'] = sr['resistance']

        # 趋势强度
        result['trend_strength'] = self._calc_trend_strength(
            result['alignment']['state'],
            result['price_position']['state'],
        )

        # 价格在MA附近时的具体建议
        result['nearest_support'] = self._nearest_level(current_price, sr['support'])
        result['nearest_resistance'] = self._nearest_level(current_price, sr['resistance'])

        return result

    def generate_signals(self, analysis_result: Dict) -> Dict:
        """
        从多时间尺度分析结果生成交易信号。
        """
        tf_results = analysis_result.get('timeframes', {})
        if not tf_results:
            return {'action': 'hold', 'strength': 0, 'buy_signals': [], 'sell_signals': [], 'description': '无数据'}

        buy_signals = []
        sell_signals = []
        buy_score = 0.0
        sell_score = 0.0
        contributing_tfs = {'buy': 0, 'sell': 0}

        for tf, data in tf_results.items():
            if 'error' in data:
                continue

            # 均线排列贡献
            alignment = data.get('alignment', {}).get('state', 'mixed')
            if alignment == 'bullish':
                buy_score += 0.4
                contributing_tfs['buy'] += 1
                buy_signals.append(f"{tf}: 多头排列")
            elif alignment == 'bearish':
                sell_score += 0.4
                contributing_tfs['sell'] += 1
                sell_signals.append(f"{tf}: 空头排列")

            # 价格回弹/受阻
            for sup in data.get('support_levels', []):
                if sup['strength'] >= 0.5:
                    dist = abs(data['current_price'] - sup['value']) / sup['value'] * 100
                    if dist < 1.5:
                        buy_score += sup['strength'] * 0.25
                        buy_signals.append(f"{tf}: 价格在MA{sup['period']}支撑附近(¥{sup['value']:.2f})")

            for res in data.get('resistance_levels', []):
                if res['strength'] >= 0.5:
                    dist = abs(data['current_price'] - res['value']) / res['value'] * 100
                    if dist < 1.5:
                        sell_score += res['strength'] * 0.25
                        sell_signals.append(f"{tf}: 价格在MA{res['period']}压力附近(¥{res['value']:.2f})")

            # 金叉/死叉贡献
            for cross in data.get('crossovers', []):
                recency = max(0, 1.0 - cross['bars_ago'] / 5)
                if cross['type'] == 'golden_cross':
                    buy_score += 0.3 * recency
                    buy_signals.append(f"{tf}: MA{cross['short']}上穿MA{cross['long']}({cross['bars_ago']}根K线前)")
                else:
                    sell_score += 0.3 * recency
                    sell_signals.append(f"{tf}: MA{cross['short']}下穿MA{cross['long']}({cross['bars_ago']}根K线前)")

        # 共振加权
        tf_count = len(tf_results)
        if contributing_tfs['buy'] >= tf_count and tf_count >= 2:
            buy_score *= 1.0  # 全共振
        elif contributing_tfs['buy'] >= 1:
            buy_score *= 0.7
        if contributing_tfs['sell'] >= tf_count and tf_count >= 2:
            sell_score *= 1.0
        elif contributing_tfs['sell'] >= 1:
            sell_score *= 0.7

        action = 'hold'
        strength = 0
        if buy_score > sell_score and buy_score > 0.5:
            action = 'buy'
            strength = min(buy_score, 1.0)
        elif sell_score > buy_score and sell_score > 0.5:
            action = 'sell'
            strength = min(sell_score, 1.0)

        description = f"买入评分{buy_score:.2f} 卖出评分{sell_score:.2f}"
        if action == 'buy':
            description += " → 均线系统偏多"
        elif action == 'sell':
            description += " → 均线系统偏空"
        else:
            description += " → 均线系统中性"

        return {
            'action': action,
            'strength': strength,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'description': description,
        }

    def format_report(self, analysis_result: Dict) -> str:
        """格式化为可读报告。"""
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"📊 MA均线分析报告: {analysis_result['symbol']}")
        lines.append(f"{'='*60}")

        for tf, data in analysis_result.get('timeframes', {}).items():
            if 'error' in data:
                lines.append(f"\n[{tf}] {data['error']}")
                continue

            tf_label = {'daily': '日线', '60min': '60分钟', '15min': '15分钟'}.get(tf, tf)
            lines.append(f"\n── {tf_label}级别 ──")
            lines.append(f"  当前价格: ¥{data['current_price']:.2f}")

            # MA值
            ma_vals = data.get('ma_values', {})
            if ma_vals:
                ma_str = '  '.join([f"MA{p}={v:.2f}" for p, v in sorted(ma_vals.items())])
                lines.append(f"  均线: {ma_str}")

            # 排列
            align = data.get('alignment', {})
            lines.append(f"  排列: {align.get('state', '?')} {align.get('description', '')}")

            # 位置
            pos = data.get('price_position', {})
            lines.append(f"  位置: {pos.get('state', '?')}")

            # 趋势强度
            lines.append(f"  趋势强度: {data.get('trend_strength', 0):.0%}")

            # 支撑位
            for s in data.get('support_levels', []):
                conf = ' [聚集]' if s.get('confluence') else ''
                lines.append(f"  ✅ 支撑: MA{s['period']} ¥{s['value']:.2f} 强度{s['strength']:.2f}{conf}")

            # 压力位
            for r in data.get('resistance_levels', []):
                conf = ' [聚集]' if r.get('confluence') else ''
                lines.append(f"  ❌ 压力: MA{r['period']} ¥{r['value']:.2f} 强度{r['strength']:.2f}{conf}")

            # 交叉
            for c in data.get('crossovers', []):
                label = '金叉' if c['type'] == 'golden_cross' else '死叉'
                lines.append(f"  {'🔺' if c['type'] == 'golden_cross' else '🔻'} {label}: MA{c['short']}×MA{c['long']} ({c['bars_ago']}根K线前)")

        # 信号
        signals = analysis_result.get('signals', {})
        lines.append(f"\n{'─'*60}")
        lines.append(f"📋 综合信号: {signals.get('action', 'hold').upper()} (强度{signals.get('strength', 0):.2f})")
        lines.append(f"   {signals.get('description', '')}")
        for s in signals.get('buy_signals', []):
            lines.append(f"   📈 {s}")
        for s in signals.get('sell_signals', []):
            lines.append(f"   📉 {s}")

        return '\n'.join(lines)

    # ── 内部方法 ──

    def _calc_ma_values(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        result = df.copy()
        for p in periods:
            result[f'ma{p}'] = df['close'].rolling(window=p).mean()
        return result

    def _determine_alignment(self, ma_values: Dict[int, float], periods: List[int]) -> Dict:
        sorted_periods = sorted(periods)
        vals = [ma_values.get(p) for p in sorted_periods]

        if any(v is None for v in vals):
            return {'state': 'mixed', 'description': '数据不完整', 'ordered_periods': sorted_periods}

        is_bullish = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        is_bearish = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

        if is_bullish:
            return {'state': 'bullish', 'description': '多头排列(短>中>长)', 'ordered_periods': sorted_periods}
        elif is_bearish:
            return {'state': 'bearish', 'description': '空头排列(短<中<长)', 'ordered_periods': sorted_periods}
        else:
            return {'state': 'mixed', 'description': '均线交叉/混乱', 'ordered_periods': sorted_periods}

    def _determine_price_position(self, price: float, ma_values: Dict[int, float], periods: List[int]) -> Dict:
        above = [p for p in periods if p in ma_values and ma_values[p] < price]
        below = [p for p in periods if p in ma_values and ma_values[p] > price]

        if not below:
            state = 'above_all'
        elif not above:
            state = 'below_all'
        else:
            state = 'between'

        # 最近MA
        nearest = None
        min_dist = float('inf')
        for p, v in ma_values.items():
            dist = abs(price - v) / v * 100
            if dist < min_dist:
                min_dist = dist
                nearest = {'period': p, 'value': v, 'distance_pct': dist}

        return {
            'state': state,
            'above_mas': sorted(above),
            'below_mas': sorted(below),
            'nearest_ma': nearest,
        }

    def _calc_support_resistance(self, price: float, ma_values: Dict[int, float], periods: List[int]) -> Dict:
        supports = []
        resistances = []

        for p in periods:
            if p not in ma_values:
                continue
            val = ma_values[p]
            if np.isnan(val):
                continue

            base = STRENGTH_WEIGHTS.get(p, 0.5)
            dist_pct = abs(price - val) / val * 100

            if dist_pct <= 1.0:
                dist_factor = 1.0
            elif dist_pct <= 3.0:
                dist_factor = 0.5
            else:
                dist_factor = 0.2

            strength = round(base * dist_factor, 2)

            entry = {'period': p, 'value': round(val, 3), 'strength': strength, 'confluence': False}

            if val < price:
                supports.append(entry)
            elif val > price:
                resistances.append(entry)

        # 聚集效应检测
        self._mark_confluence(supports)
        self._mark_confluence(resistances)

        supports.sort(key=lambda x: x['value'], reverse=True)
        resistances.sort(key=lambda x: x['value'])

        return {'support': supports, 'resistance': resistances}

    def _mark_confluence(self, levels: List[Dict]):
        if len(levels) < 2:
            return
        for i in range(len(levels)):
            count = 1
            for j in range(len(levels)):
                if i != j and abs(levels[i]['value'] - levels[j]['value']) / levels[j]['value'] <= 0.02:
                    count += 1
            if count >= 2:
                levels[i]['confluence'] = True
                levels[i]['strength'] = min(levels[i]['strength'] * 1.3, 1.0)

    def _detect_crossovers(self, df: pd.DataFrame, periods: List[int], lookback: int = 5) -> List[Dict]:
        sorted_periods = sorted(periods)
        crossovers = []
        check_range = min(lookback, len(df) - 1)

        # 检查相邻MA之间的交叉
        for i in range(len(sorted_periods) - 1):
            sp = sorted_periods[i]
            lp = sorted_periods[i + 1]
            col_s = f'ma{sp}'
            col_l = f'ma{lp}'

            if col_s not in df.columns or col_l not in df.columns:
                continue

            for bar in range(1, check_range + 1):
                idx = len(df) - bar
                prev_idx = idx - 1
                if prev_idx < 0:
                    continue

                curr_short = df[col_s].iloc[idx]
                curr_long = df[col_l].iloc[idx]
                prev_short = df[col_s].iloc[prev_idx]
                prev_long = df[col_l].iloc[prev_idx]

                if any(np.isnan(v) for v in [curr_short, curr_long, prev_short, prev_long]):
                    continue

                if prev_short <= prev_long and curr_short > curr_long:
                    crossovers.append({
                        'type': 'golden_cross',
                        'short': sp, 'long': lp,
                        'bars_ago': bar,
                        'short_ma': round(float(curr_short), 3),
                        'long_ma': round(float(curr_long), 3),
                    })
                elif prev_short >= prev_long and curr_short < curr_long:
                    crossovers.append({
                        'type': 'death_cross',
                        'short': sp, 'long': lp,
                        'bars_ago': bar,
                        'short_ma': round(float(curr_short), 3),
                        'long_ma': round(float(curr_long), 3),
                    })

        return crossovers

    def _calc_trend_strength(self, alignment_state: str, position_state: str) -> float:
        if alignment_state == 'bullish' and position_state == 'above_all':
            return 1.0
        elif alignment_state == 'bullish' and position_state == 'between':
            return 0.7
        elif alignment_state == 'bearish' and position_state == 'below_all':
            return 1.0
        elif alignment_state == 'bearish' and position_state == 'between':
            return 0.7
        else:
            return 0.3

    def _nearest_level(self, price: float, levels: List[Dict]) -> Optional[Dict]:
        if not levels:
            return None
        return min(levels, key=lambda x: abs(price - x['value']))

    def _fetch_data(self, timeframe: str, lookback: int) -> Optional[pd.DataFrame]:
        if timeframe in ('60min', '15min'):
            return self._fetch_minute_data(timeframe, lookback)
        try:
            df = self.fetcher.get_historical_kline(
                self.symbol,
                period=timeframe,
                start_date='20250101',
            )
            if df is not None and not df.empty:
                return df.tail(lookback)
            return None
        except Exception as e:
            print(f"⚠️  获取{timeframe}数据失败: {e}")
            return None

    def _fetch_minute_data(self, timeframe: str, lookback: int) -> Optional[pd.DataFrame]:
        ak_period = timeframe.replace('min', '')  # '60min'→'60', '15min'→'15'
        try:
            import akshare as ak
            code = self.symbol.replace('sh.', '').replace('sz.', '')
            # AKShare minute API needs exchange prefix: sh for 6xxxxx, sz for 0xxxxx/3xxxxx
            if not code.startswith(('sh', 'sz')):
                if code.startswith(('6', '9')):
                    code = 'sh' + code
                else:
                    code = 'sz' + code
            df = ak.stock_zh_a_minute(symbol=code, period=ak_period, adjust='qfq')
            if df is None or df.empty:
                return None
            col_map = {'day': 'date', 'open': 'open', 'high': 'high',
                       'low': 'low', 'close': 'close', 'volume': 'volume'}
            df = df.rename(columns=col_map)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df.tail(lookback).reset_index(drop=True)
        except Exception as e:
            print(f"⚠️  获取{timeframe}数据失败: {e}")
            return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MA均线分析')
    parser.add_argument('--symbol', required=True, help='股票代码')
    parser.add_argument('--timeframes', nargs='+', default=['daily', '60min'],
                        choices=['daily', '60min', '15min'], help='分析时间尺度')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')
    args = parser.parse_args()

    analyzer = MAAnalyzer(args.symbol)
    result = analyzer.analyze(timeframes=args.timeframes)

    if args.json:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(analyzer.format_report(result))
