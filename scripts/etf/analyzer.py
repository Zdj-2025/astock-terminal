#!/usr/bin/env python3
"""
ETF分析器 v2 - 专为长期持有AI ETF + 择时做差价设计

核心理念:
- 长期持有AI及相关产业ETF（核心仓位）
- 通过技术指标择时做差价（交易仓位）
- 降本增股，不改变核心持仓方向

分析维度:
1. 板块趋势 (30%) - AI板块整体方向，决定核心仓位增减
2. 择时信号 (30%) - RSI/布林带/MA距离，决定加减仓时机
3. 板块相对强弱 (20%) - AI子板块轮动排名
4. 折溢价 (10%) - IOPV实时折溢价率
5. 流动性 (10%) - 成交额/换手率

输出双层级建议:
- 核心仓位建议: 增持/维持/减持
- 交易仓位建议: 加仓X%/减仓X%/观望

作者: AI Assistant
日期: 2026-04-08
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.etf.data_fetcher import ETFDataFetcher
from scripts.analyzers.technical_analyzer import TechnicalAnalyzer


# AI及相关产业ETF配置
AI_ETF_PORTFOLIO = {
    '512760': {'name': '芯片ETF', 'sector': 'chips'},
    '515070': {'name': 'AI智能', 'sector': 'ai_theme'},
    '512480': {'name': '半导体', 'sector': 'semiconductor'},
    '515700': {'name': '大数据', 'sector': 'big_data'},
    '510050': {'name': '50ETF', 'sector': 'broad_market'},
}


class ETFAnalyzer:
    """
    ETF分析器 v2

    面向长期持有AI ETF + 择时做差价的投资者。
    """

    def __init__(self):
        """初始化"""
        self.fetcher = ETFDataFetcher(enable_cache=True)
        self.tech_analyzer = TechnicalAnalyzer('')
        self.ai_etf_portfolio = AI_ETF_PORTFOLIO

        # 择时参数
        self.timing_params = {
            'rsi_oversold': 25,        # RSI超卖阈值
            'rsi_overbought': 75,      # RSI超买阈值
            'bollinger_lower_k': 2.0,  # 布林带下轨标准差倍数
            'ma_distance_threshold': 0.03,  # MA距离阈值(3%)
            'volume_spike': 2.0,       # 成交量放大倍数
            'cost_discount': 0.90,     # 相对成本价的折扣（加仓触发）
        }

        print("✅ ETFAnalyzer v2 初始化成功（长期持有+择时做差价模式）")

    def analyze(self, symbol: str, name: str = '', cost_price: float = 0.0) -> Optional[Dict]:
        """
        综合分析ETF

        参数:
            symbol: ETF代码
            name: ETF名称
            cost_price: 持仓成本价（0表示未知）

        返回:
            完整分析结果（含双层级建议）
        """
        try:
            symbol = str(symbol).zfill(6)
            etf_info = self.ai_etf_portfolio.get(symbol, {})
            if not name:
                name = etf_info.get('name', symbol)

            print(f"\n📊 ETF分析: {symbol} {name}")

            # 1. 获取数据
            kline_data = self._get_kline(symbol)
            iopv_data = self._get_iopv(symbol)
            nav_data = self._get_nav(symbol)

            if kline_data is None or kline_data.empty:
                print(f"   ❌ 无法获取K线数据，跳过分析")
                return None

            current_price = float(kline_data.iloc[-1]['close'])

            # 2. 板块趋势分析 (30%)
            sector_trend = self._analyze_sector_trend(symbol, kline_data, iopv_data)

            # 3. 择时信号分析 (30%)
            timing_signal = self._analyze_timing_signal(
                symbol, kline_data, current_price, cost_price
            )

            # 4. 板块相对强弱 (20%)
            relative_strength = self._analyze_relative_strength(symbol, kline_data)

            # 5. 折溢价分析 (10%)
            valuation = self._analyze_valuation(current_price, iopv_data, nav_data)

            # 6. 流动性分析 (10%)
            liquidity = self._analyze_liquidity(iopv_data, kline_data)

            # 7. 综合评分
            overall_score = (
                sector_trend['score'] * 0.30 +
                timing_signal['score'] * 0.30 +
                relative_strength['score'] * 0.20 +
                valuation['score'] * 0.10 +
                liquidity['score'] * 0.10
            )

            # 核心仓位评分（基于板块趋势 + 相对强弱）
            core_position_score = (
                sector_trend['score'] * 0.6 +
                relative_strength['score'] * 0.4
            )

            # 交易信号评分（基于择时信号 + 折溢价）
            trading_signal_score = (
                timing_signal['score'] * 0.7 +
                valuation['score'] * 0.3
            )

            # 评级
            if overall_score >= 80:
                rating = '强烈推荐'
            elif overall_score >= 70:
                rating = '推荐'
            elif overall_score >= 60:
                rating = '中性'
            elif overall_score >= 50:
                rating = '谨慎'
            else:
                rating = '不推荐'

            # 8. 双层级建议
            core_advice = self._generate_core_advice(core_position_score, sector_trend)
            trading_advice = self._generate_trading_advice(
                trading_signal_score, timing_signal, current_price, cost_price
            )

            # 打印摘要
            print(f"\n   📊 评分: 综合{overall_score:.0f} | "
                  f"板块趋势{sector_trend['score']:.0f} | "
                  f"择时{timing_signal['score']:.0f} | "
                  f"强弱{relative_strength['score']:.0f}")
            print(f"   📋 核心仓位: {core_advice['action_text']}")
            print(f"   📋 交易仓位: {trading_advice['action_text']}")

            return {
                'symbol': symbol,
                'name': name,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price,
                'cost_price': cost_price,
                'is_etf': True,

                # 分析维度
                'sector_trend': sector_trend,
                'timing_signal': timing_signal,
                'relative_strength': relative_strength,
                'valuation': valuation,
                'liquidity': liquidity,

                # 评分
                'overall_score': overall_score,
                'core_position_score': core_position_score,
                'trading_signal_score': trading_signal_score,
                'rating': rating,

                # 双层级建议
                'core_advice': core_advice,
                'trading_advice': trading_advice,

                # 向后兼容
                'recommendation': {
                    'action': core_advice['action'],
                    'action_text': f"{core_advice['action_text']}；{trading_advice['action_text']}",
                    'position_pct': trading_advice.get('trade_size_pct', 0),
                    'confidence': '高' if overall_score >= 70 else ('中' if overall_score >= 50 else '低'),
                    'reasons': core_advice.get('reasons', []) + trading_advice.get('reasons', [])
                }
            }

        except Exception as e:
            print(f"   ❌ ETF分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ── 数据获取 ──

    def _get_kline(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        try:
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
            df = self.fetcher.get_etf_kline(symbol, start_date=start, end_date=end)
            if df is not None and not df.empty:
                print(f"   ✅ K线: {len(df)}条")
                return df
        except Exception as e:
            print(f"   ⚠️  K线获取失败: {str(e)[:50]}")
        return None

    def _get_iopv(self, symbol: str) -> Optional[Dict]:
        """获取IOPV实时数据"""
        try:
            data = self.fetcher.get_etf_iopv_quote(symbol)
            if data:
                print(f"   ✅ IOPV: 价格={data['price']:.3f} 折溢价={data['premium_rate']:.2f}%")
                return data
        except Exception:
            pass
        print(f"   ⚠️  IOPV不可用（网络问题）")
        return None

    def _get_nav(self, symbol: str) -> Optional[Dict]:
        """获取NAV数据"""
        try:
            info = self.fetcher.get_etf_info(symbol)
            if info and info.get('nav', 0) > 0:
                print(f"   ✅ NAV: {info['nav']:.3f} ({info['nav_date']})")
                return info
        except Exception:
            pass
        return None

    # ── 分析维度 ──

    def _analyze_sector_trend(self, symbol: str, kline_data: pd.DataFrame,
                              iopv_data: Optional[Dict]) -> Dict:
        """
        板块趋势分析 - 决定核心仓位方向

        基于: MA20/MA60趋势 + MACD方向 + 成交量趋势
        """
        try:
            close = kline_data['close']
            result = {'trend': 'sideways', 'strength': 0.5, 'score': 50, 'reasons': []}

            # 1. 移动均线趋势
            if len(close) >= 60:
                ma20 = close.rolling(20).mean()
                ma60 = close.rolling(60).mean()
                latest_close = close.iloc[-1]
                latest_ma20 = ma20.iloc[-1]
                latest_ma60 = ma60.iloc[-1]

                # MA多头排列: price > MA20 > MA60
                if latest_close > latest_ma20 > latest_ma60:
                    result['trend'] = 'up'
                    result['reasons'].append('MA多头排列（价格>MA20>MA60）')
                elif latest_close < latest_ma20 < latest_ma60:
                    result['trend'] = 'down'
                    result['reasons'].append('MA空头排列（价格<MA20<MA60）')

                # MA20斜率
                if len(ma20) >= 5:
                    slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100
                    if slope > 1:
                        result['reasons'].append(f'MA20上升斜率{slope:.1f}%')
                    elif slope < -1:
                        result['reasons'].append(f'MA20下降斜率{slope:.1f}%')

            # 2. MACD方向
            tech_result = self.tech_analyzer.analyze_indicators(kline_data)
            if tech_result:
                macd_signal = tech_result.get('macd', {}).get('signal', 'hold')
                if macd_signal in ('buy', 'golden_cross', 'strong_buy'):
                    result['reasons'].append(f'MACD {macd_signal}')
                elif macd_signal in ('sell', 'death_cross', 'strong_sell'):
                    result['reasons'].append(f'MACD {macd_signal}')

            # 3. 资金流向（来自IOPV数据）
            if iopv_data:
                change_pct = iopv_data.get('change_pct', 0)
                if change_pct > 1:
                    result['reasons'].append(f'当日涨幅+{change_pct:.1f}%')
                elif change_pct < -1:
                    result['reasons'].append(f'当日跌幅{change_pct:.1f}%')

            # 计算评分
            if result['trend'] == 'up':
                result['strength'] = 0.7
                result['score'] = 75
            elif result['trend'] == 'down':
                result['strength'] = 0.3
                result['score'] = 35

            # MACD加分/减分
            if tech_result:
                macd_s = tech_result.get('macd', {}).get('signal', 'hold')
                if macd_s in ('buy', 'golden_cross', 'strong_buy'):
                    result['score'] = min(100, result['score'] + 10)
                elif macd_s in ('sell', 'death_cross', 'strong_sell'):
                    result['score'] = max(0, result['score'] - 10)

            print(f"   ✅ 板块趋势: {result['trend']} | {result['score']:.0f}分")
            return result

        except Exception as e:
            print(f"   ⚠️  板块趋势分析失败: {str(e)[:50]}")
            return {'trend': 'unknown', 'strength': 0.5, 'score': 50, 'reasons': []}

    def _analyze_timing_signal(self, symbol: str, kline_data: pd.DataFrame,
                                current_price: float, cost_price: float) -> Dict:
        """
        择时信号分析 - 决定交易仓位操作

        基于: RSI极端值 + 布林带位置 + MA距离 + 成交量 + 成本价距离
        """
        try:
            result = {'signal': 'wait', 'strength': 0.0, 'score': 50,
                      'indicators': {}, 'reasons': []}
            close = kline_data['close']

            # 1. RSI分析
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1] if not rsi.empty else 50

            result['indicators']['rsi'] = round(latest_rsi, 1)

            if latest_rsi < self.timing_params['rsi_oversold']:
                result['signal'] = 'add'
                result['reasons'].append(f'RSI={latest_rsi:.0f} 超卖，加仓信号')
                result['score'] += 20
            elif latest_rsi > self.timing_params['rsi_overbought']:
                result['signal'] = 'trim'
                result['reasons'].append(f'RSI={latest_rsi:.0f} 超买，减仓信号')
                result['score'] -= 15
            else:
                result['reasons'].append(f'RSI={latest_rsi:.0f} 正常区间')

            # 2. 布林带位置
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std(ddof=0)
            bb_lower = ma20 - self.timing_params['bollinger_lower_k'] * std20
            bb_upper = ma20 + self.timing_params['bollinger_lower_k'] * std20

            if not bb_lower.empty:
                latest_bb_lower = bb_lower.iloc[-1]
                latest_bb_upper = bb_upper.iloc[-1]

                if current_price <= latest_bb_lower:
                    result['signal'] = 'add'
                    result['reasons'].append('触及布林带下轨，加仓信号')
                    result['score'] += 15
                elif current_price >= latest_bb_upper:
                    result['signal'] = 'trim'
                    result['reasons'].append('触及布林带上轨，减仓信号')
                    result['score'] -= 10

            # 3. MA距离 (价格偏离MA20的程度)
            if not ma20.empty and ma20.iloc[-1] > 0:
                ma_dist = (current_price - ma20.iloc[-1]) / ma20.iloc[-1]
                result['indicators']['ma20_distance_pct'] = round(ma_dist * 100, 2)

                if ma_dist < -self.timing_params['ma_distance_threshold']:
                    result['signal'] = 'add'
                    result['reasons'].append(f'低于MA20 {abs(ma_dist)*100:.1f}%，加仓信号')
                    result['score'] += 10
                elif ma_dist > self.timing_params['ma_distance_threshold']:
                    result['reasons'].append(f'高于MA20 {ma_dist*100:.1f}%')
                    result['score'] -= 5

            # 4. 成交量分析
            vol = kline_data['volume']
            if len(vol) >= 20:
                avg_vol = vol.iloc[-20:].mean()
                latest_vol = vol.iloc[-1]
                vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
                result['indicators']['volume_ratio'] = round(vol_ratio, 2)

                if vol_ratio > self.timing_params['volume_spike']:
                    if current_price < ma20.iloc[-1] if not ma20.empty else True:
                        result['reasons'].append(f'放量下跌（量比{vol_ratio:.1f}），恐慌盘机会')
                        result['score'] += 5

            # 5. 成本价距离
            if cost_price > 0:
                pnl_pct = (current_price / cost_price - 1) * 100
                result['indicators']['pnl_vs_cost_pct'] = round(pnl_pct, 2)

                if current_price < cost_price * self.timing_params['cost_discount']:
                    result['signal'] = 'add'
                    result['reasons'].append(f'低于成本价{abs(pnl_pct):.1f}%，摊薄机会')
                    result['score'] += 10

            # 最终评分限制在0-100
            result['score'] = max(0, min(100, result['score']))

            print(f"   ✅ 择时信号: {result['signal']} | {result['score']:.0f}分 "
                  f"(RSI={latest_rsi:.0f})")
            return result

        except Exception as e:
            print(f"   ⚠️  择时分析失败: {str(e)[:50]}")
            return {'signal': 'wait', 'strength': 0.0, 'score': 50,
                    'indicators': {}, 'reasons': []}

    def _analyze_relative_strength(self, symbol: str, kline_data: pd.DataFrame) -> Dict:
        """
        板块相对强弱 - 比较AI子板块轮动

        基于最近5/10/20天动量排名
        """
        try:
            symbols = list(self.ai_etf_portfolio.keys())
            if symbol not in symbols:
                return {'rank': 1, 'total': 1, 'score': 50, 'reasons': []}

            # 计算各ETF动量
            close = kline_data['close']
            mom_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
            mom_10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) >= 11 else 0
            mom_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0

            # 加权动量 (近期的权重更高)
            momentum = mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2

            # 简化排名：根据自身动量评分
            if momentum > 5:
                rank, score = 1, 90
            elif momentum > 2:
                rank, score = 2, 75
            elif momentum > -2:
                rank, score = 3, 55
            elif momentum > -5:
                rank, score = 4, 40
            else:
                rank, score = 5, 25

            total = len(symbols)
            reasons = []
            if momentum > 0:
                reasons.append(f'近5日动量+{mom_5:.1f}%，板块内偏强')
            else:
                reasons.append(f'近5日动量{mom_5:.1f}%，板块内偏弱')

            print(f"   ✅ 相对强弱: 排名{rank}/{total} | {score:.0f}分")
            return {
                'rank': rank,
                'total': total,
                'momentum_5d': round(mom_5, 2),
                'momentum_20d': round(mom_20, 2),
                'score': score,
                'reasons': reasons
            }

        except Exception as e:
            print(f"   ⚠️  相对强弱分析失败: {str(e)[:50]}")
            return {'rank': 0, 'total': 0, 'score': 50, 'reasons': []}

    def _analyze_valuation(self, current_price: float,
                           iopv_data: Optional[Dict], nav_data: Optional[Dict]) -> Dict:
        """
        折溢价分析

        优先使用IOPV实时数据，其次使用NAV
        """
        try:
            premium_rate = 0.0
            nav = 0.0

            if iopv_data and iopv_data.get('iopv', 0) > 0:
                premium_rate = iopv_data.get('premium_rate', 0)
                nav = iopv_data['iopv']
            elif nav_data and nav_data.get('nav', 0) > 0:
                nav = nav_data['nav']
                premium_rate = (current_price - nav) / nav * 100

            if abs(premium_rate) < 0.3:
                score = 90
            elif abs(premium_rate) < 0.5:
                score = 75
            elif abs(premium_rate) < 1.0:
                score = 60
            elif premium_rate < -1.0:
                score = 85  # 折价是买入机会
            else:
                score = 40  # 高溢价需谨慎

            reasons = []
            if premium_rate < -0.5:
                reasons.append(f'折价{abs(premium_rate):.2f}%，买入安全边际高')
            elif premium_rate > 0.5:
                reasons.append(f'溢价{premium_rate:.2f}%，注意成本')
            else:
                reasons.append(f'折溢价正常({premium_rate:+.2f}%)')

            print(f"   ✅ 折溢价: {premium_rate:+.2f}% | {score:.0f}分")
            return {
                'price': current_price,
                'nav': nav,
                'premium_rate': premium_rate,
                'score': score,
                'reasons': reasons
            }

        except Exception as e:
            print(f"   ⚠️  折溢价分析失败: {str(e)[:50]}")
            return {'price': 0, 'nav': 0, 'premium_rate': 0, 'score': 50, 'reasons': []}

    def _analyze_liquidity(self, iopv_data: Optional[Dict],
                           kline_data: pd.DataFrame) -> Dict:
        """流动性分析"""
        try:
            volume = 0
            amount = 0

            if iopv_data:
                volume = iopv_data.get('volume', 0)
                amount = iopv_data.get('amount', 0)

            if volume == 0 and not kline_data.empty:
                volume = int(kline_data.iloc[-1].get('volume', 0))

            if volume > 50000000:
                liquidity = '优秀'
                score = 100
            elif volume > 20000000:
                liquidity = '良好'
                score = 80
            elif volume > 10000000:
                liquidity = '一般'
                score = 60
            elif volume > 0:
                liquidity = '较差'
                score = 40
            else:
                liquidity = '未知'
                score = 50

            print(f"   ✅ 流动性: {liquidity} (成交量{volume/10000:.0f}万) | {score:.0f}分")
            return {'volume': volume, 'amount': amount, 'liquidity': liquidity, 'score': score}

        except Exception as e:
            return {'volume': 0, 'liquidity': '未知', 'score': 50}

    # ── 建议生成 ──

    def _generate_core_advice(self, core_score: float, sector_trend: Dict) -> Dict:
        """
        生成核心仓位建议

        核心仓位是长期持有部分，只在板块趋势明确改变时调整。
        """
        if core_score >= 75:
            action = 'increase'
            action_text = '增持核心仓位'
        elif core_score >= 55:
            action = 'maintain'
            action_text = '维持核心仓位'
        elif core_score >= 40:
            action = 'reduce'
            action_text = '考虑减少核心仓位'
        else:
            action = 'reduce'
            action_text = '减少核心仓位'

        return {
            'action': action,
            'action_text': action_text,
            'target_direction': sector_trend.get('trend', 'sideways'),
            'reasons': sector_trend.get('reasons', [])
        }

    def _generate_trading_advice(self, trading_score: float, timing_signal: Dict,
                                  current_price: float, cost_price: float) -> Dict:
        """
        生成交易仓位建议

        交易仓位是择时做差价的部分，幅度较小（通常2-5%的仓位调整）。
        """
        signal = timing_signal.get('signal', 'wait')

        if signal == 'add' and trading_score >= 60:
            action = 'add'
            trade_size = min(5.0, (trading_score - 50) / 10)  # 0-5%的加仓幅度
            action_text = f'加仓{trade_size:.1f}%做差价'
        elif signal == 'trim' and trading_score <= 40:
            action = 'trim'
            trade_size = min(5.0, (50 - trading_score) / 10)
            action_text = f'减仓{trade_size:.1f}%做差价'
        else:
            action = 'wait'
            trade_size = 0.0
            action_text = '观望，等待择时信号'

        # 计算触发价
        trigger_price = 0.0
        if action == 'add' and cost_price > 0:
            trigger_price = round(cost_price * self.timing_params['cost_discount'], 3)
        elif action == 'add':
            trigger_price = round(current_price * 0.97, 3)  # 低于当前价3%加仓
        elif action == 'trim':
            trigger_price = round(current_price * 1.03, 3)  # 高于当前价3%减仓

        return {
            'action': action,
            'action_text': action_text,
            'trade_size_pct': trade_size,
            'trigger_price': trigger_price,
            'reasons': timing_signal.get('reasons', [])
        }


# ── 测试 ──

if __name__ == '__main__':
    print("=" * 80)
    print("测试 ETFAnalyzer v2")
    print("=" * 80)

    analyzer = ETFAnalyzer()

    # 测试芯片ETF（带成本价）
    result = analyzer.analyze('512760', '芯片ETF', cost_price=0.549)

    if result:
        print(f"\n{'='*60}")
        print(f"📊 {result['name']} ({result['symbol']})")
        print(f"   当前价: ¥{result['current_price']:.3f}")
        print(f"   综合评分: {result['overall_score']:.0f} | {result['rating']}")
        print(f"   核心仓位: {result['core_advice']['action_text']}")
        print(f"   交易仓位: {result['trading_advice']['action_text']}")
        print(f"{'='*60}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
