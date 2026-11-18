#!/usr/bin/env python3
"""
决策引擎 - 基于综合评分生成交易决策

职责:
1. 接收聚合信号，结合组合状态做最终决策
2. 计算止损/止盈价位
3. 用凯利公式确定建议仓位
4. 生成结构化决策输出

作者: AI Assistant
日期: 2026-04-07
"""
import json
import os
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, '/mnt/d/claude-code/stock')

from scripts.decision.signal_aggregator import SignalAggregator, AggregatedSignal
from scripts.core.position_calculator import PositionCalculator
from scripts.core.portfolio_manager import PortfolioManager


@dataclass
class TradeDecision:
    """交易决策结果"""
    symbol: str = ''
    name: str = ''

    # 决策
    action: str = 'hold'           # buy / sell / hold
    action_text: str = '观望'      # 买入 / 卖出 / 观望
    rating: str = '中性'           # 强烈推荐 / 推荐 / 中性 / 谨慎 / 不推荐

    # 评分
    overall_score: float = 0.0
    tech_score: float = 0.0
    fund_score: float = 0.0
    macd_score: float = 0.0
    etf_score: float = 0.0

    # 仓位建议
    position_fraction: float = 0.0   # 建议仓位比例 (0-1)
    position_amount: float = 0.0     # 建议金额
    position_method: str = ''        # 仓位计算方法

    # 价格建议
    current_price: float = 0.0
    stop_loss_price: float = 0.0     # 止损价
    take_profit_price: float = 0.0   # 止盈价
    stop_loss_pct: float = -0.08     # 止损比例
    take_profit_pct: float = 0.15    # 止盈比例

    # 风控
    confidence: str = '低'           # 高 / 中 / 低
    risk_level: str = '中'           # 低 / 中 / 高
    risk_alerts: List[Dict] = field(default_factory=list)

    # 理由
    buy_reasons: List[str] = field(default_factory=list)
    sell_reasons: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)

    # 元信息
    is_etf: bool = False
    decision_time: str = ''
    holding_info: Optional[Dict] = None  # 如果已持仓，包含持仓信息

    # ETF专用字段
    core_advice: Optional[Dict] = None     # 核心仓位建议
    trading_advice: Optional[Dict] = None   # 交易仓位建议
    cost_basis: float = 0.0                 # 成本价
    pnl_vs_cost: float = 0.0               # 相对成本的盈亏比例


class DecisionEngine:
    """
    决策引擎

    接收聚合信号，结合组合状态，生成最终交易决策。
    """

    # 评级阈值
    RATING_THRESHOLDS = {
        'strong_buy': 80,    # 强烈推荐
        'buy': 70,           # 推荐
        'neutral_high': 60,  # 中性偏多
        'neutral': 50,       # 中性
        'cautious': 40,      # 谨慎
        # < 40: 不推荐
    }

    def __init__(
        self,
        config_dir: str = 'config',
        signal_aggregator: Optional[SignalAggregator] = None
    ):
        """
        初始化决策引擎

        参数:
            config_dir: 配置目录
            signal_aggregator: 信号聚合器（可选，不传则自建）
        """
        self.config_dir = config_dir
        self.risk_config = self._load_risk_config()
        self.analysis_config = self._load_analysis_config()

        # 核心组件
        self.aggregator = signal_aggregator or SignalAggregator(
            os.path.join(config_dir, 'analysis_config.json')
        )
        self.position_calc = PositionCalculator(
            os.path.join(config_dir, 'risk_rules.json')
        )
        self.portfolio_mgr = PortfolioManager(config_dir)

        # ── 铁律v2 动态止损止盈参数 ──
        # 止损分档（按仓位大小）
        self.stop_loss_tiers = {
            'light':  -0.10,   # 仓位≤10%: 止损-10%（给更多波动空间）
            'normal': -0.08,   # 仓位10-20%: 标准止损-8%
            'heavy':  -0.05,   # 仓位>20%: 重仓止损-5%（更敏感）
        }
        # 动态止盈（按盈利幅度递进）
        self.profit_protection = [
            {'threshold': 0.15, 'min_sl': 0.00},    # 盈利15-30%: 止损上移至成本价
            {'threshold': 0.30, 'min_sl': 0.10},    # 盈利30-50%: 止损上移至成本+10%
            {'threshold': 0.50, 'min_sl': 0.25},    # 盈利50-100%: 止损上移至成本+25%
            {'threshold': 1.00, 'min_sl': 0.50},    # 盈利>100%: 止损上移至成本+50%
        ]
        # 兼容：默认值（用于无持仓信息的场景）
        self.stop_loss_pct = -0.08
        self.take_profit_pct = 0.15
        self.trailing_trigger = 0.10

        # 交易参数
        self.min_signal_strength = self.analysis_config.get('trading_params', {}).get(
            'min_signal_strength', 0.6
        )

        print("🎯 初始化决策引擎")
        print(f"   止损分档: 轻仓-10% / 标准-8% / 重仓-5%")
        print(f"   动态止盈: 盈利保护机制已启用（铁律v2）")
        print(f"   最小信号强度: {self.min_signal_strength}")

    def _load_risk_config(self) -> Dict:
        """加载风险配置"""
        path = os.path.join(self.config_dir, 'risk_rules.json')
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _load_analysis_config(self) -> Dict:
        """加载分析配置"""
        path = os.path.join(self.config_dir, 'analysis_config.json')
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def decide(
        self,
        symbol: str,
        name: str = '',
        kline_data=None,
        financial_data=None,
        skip_macd_resonance: bool = False
    ) -> TradeDecision:
        """
        对指定标的做出交易决策

        参数:
            symbol: 股票/ETF代码
            name: 名称
            kline_data: 预加载K线数据 (可选)
            financial_data: 预加载财务数据 (可选)
            skip_macd_resonance: 跳过多尺度MACD

        返回:
            TradeDecision 完整交易决策
        """
        symbol = str(symbol).zfill(6)
        is_etf = symbol.startswith('51') or symbol.startswith('15')

        print(f"\n{'='*60}")
        print(f"🎯 交易决策: {symbol} {name} {'[ETF]' if is_etf else '[股票]'}")
        print(f"{'='*60}")

        decision = TradeDecision(
            symbol=symbol,
            name=name,
            is_etf=is_etf,
            decision_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        # Step 1: 聚合信号
        signal = self.aggregator.aggregate(
            symbol, name, kline_data, financial_data,
            skip_macd_resonance=skip_macd_resonance
        )

        # 复制评分
        decision.overall_score = signal.overall_score
        decision.tech_score = signal.tech_score
        decision.fund_score = signal.fund_score
        decision.macd_score = signal.macd_resonance_score
        decision.etf_score = signal.etf_score
        decision.confidence = signal.confidence

        # Step 2: 获取当前价格
        decision.current_price = self._get_current_price(signal, kline_data, symbol, is_etf)

        # Step 3: 检查持仓状态
        decision.holding_info = self._check_holding(symbol)
        decision.risk_alerts = self._check_portfolio_risks(symbol)

        # Step 4: 计算止损止盈
        self._calculate_stop_levels(decision)

        # Step 5: 确定最终决策
        self._make_decision(decision, signal)

        # Step 5.5: ETF专用决策覆盖
        if is_etf:
            self._apply_etf_decision_overlay(decision, signal)

        # Step 6: 计算仓位
        if decision.action == 'buy':
            self._calculate_position(decision, signal)

        # Step 7: 生成理由
        self._generate_reasons(decision, signal)

        return decision

    # ── 辅助方法 ──

    def _get_current_price(self, signal: AggregatedSignal, kline_data, symbol: str,
                           is_etf: bool) -> float:
        """获取当前价格"""
        # 优先从K线数据获取
        if kline_data is not None and hasattr(kline_data, 'iloc'):
            try:
                return float(kline_data.iloc[-1]['close'])
            except Exception:
                pass

        # 从技术分析详情获取
        if signal.tech_detail and 'latest_values' in signal.tech_detail:
            return float(signal.tech_detail['latest_values'].get('close', 0))

        # 从MACD信号获取
        if signal.macd_detail and 'current_price' in signal.macd_detail:
            return float(signal.macd_detail['current_price'])

        return 0.0

    def _check_holding(self, symbol: str) -> Optional[Dict]:
        """检查是否已持有该标的"""
        try:
            portfolio = self.portfolio_mgr.load_portfolio()
            if not portfolio.empty:
                match = portfolio[portfolio['股票代码'] == symbol]
                if not match.empty:
                    row = match.iloc[0]
                    return {
                        'buy_price': float(row.get('买入价格', 0)),
                        'shares': int(row.get('持仓数量', 0)),
                        'stop_loss': float(row.get('止损价', 0)),
                        'take_profit': float(row.get('止盈价', 0)),
                        'position_pct': float(row.get('仓位%', 0)),
                        'type': str(row.get('持仓类型', ''))
                    }
        except Exception:
            pass
        return None

    def _check_portfolio_risks(self, symbol: str) -> List[Dict]:
        """检查组合风险"""
        alerts = []
        try:
            capital = self.portfolio_mgr.load_capital()
            total_capital = capital.get('总资金', 0)
            if total_capital > 0:
                current_exposure = capital.get('当前仓位', 0)
                max_exposure = self.risk_config.get('position_limits', {}).get('max_total', 0.8)

                if current_exposure >= max_exposure * total_capital:
                    alerts.append({
                        'type': 'total_position_limit',
                        'message': f'总仓位已达上限 {current_exposure/total_capital*100:.1f}%'
                    })
        except Exception:
            pass
        return alerts

    def _calculate_stop_levels(self, decision: TradeDecision):
        """计算止损止盈价位 — 铁律v2动态分档"""
        if decision.current_price <= 0:
            return

        price = decision.current_price
        buy_price = 0.0
        position_pct = 0.0  # 仓位占总资金池比例

        if decision.holding_info:
            buy_price = decision.holding_info.get('buy_price', 0)
            position_pct = decision.holding_info.get('position_pct', 0)

        # ── 止损：按仓位分档 ──
        if position_pct > 0.20:
            sl_pct = self.stop_loss_tiers['heavy']   # -5%
        elif position_pct > 0.10:
            sl_pct = self.stop_loss_tiers['normal']   # -8%
        else:
            sl_pct = self.stop_loss_tiers['light']    # -10%

        # 已持仓：止损基于成本价
        if buy_price > 0:
            base_sl = buy_price * (1 + sl_pct)
            # 盈利保护：如果已盈利，止损线不低于保护线
            pnl_pct = (price / buy_price - 1) if buy_price > 0 else 0
            for rule in reversed(self.profit_protection):
                if pnl_pct >= rule['threshold']:
                    protected_sl = buy_price * (1 + rule['min_sl'])
                    base_sl = max(base_sl, protected_sl)
                    break
            decision.stop_loss_price = round(base_sl, 2)
            decision.stop_loss_pct = round((base_sl / price - 1), 4)
        else:
            # 无持仓：止损基于当前价
            decision.stop_loss_price = round(price * (1 + sl_pct), 2)
            decision.stop_loss_pct = sl_pct

        # ── 止盈：动态递进 ──
        if buy_price > 0:
            pnl_pct = (price / buy_price - 1)
            # 已高盈利时，止盈线=当前盈利保护线+缓冲
            if pnl_pct >= 1.0:
                tp_pct = 0.60  # 盈利>100%: 止盈=成本+60%
            elif pnl_pct >= 0.50:
                tp_pct = 0.35  # 盈利50-100%: 止盈=成本+35%
            elif pnl_pct >= 0.30:
                tp_pct = 0.20  # 盈利30-50%: 止盈=成本+20%
            else:
                tp_pct = 0.15  # 默认+15%
            decision.take_profit_price = round(buy_price * (1 + tp_pct), 2)
            decision.take_profit_pct = tp_pct
        else:
            decision.take_profit_price = round(price * 1.15, 2)
            decision.take_profit_pct = 0.15

        # 持仓配置的止损止盈作为下限（不覆盖动态计算的保护线）
        if decision.holding_info:
            if decision.holding_info.get('stop_loss', 0) > 0:
                decision.stop_loss_price = max(
                    decision.stop_loss_price,
                    decision.holding_info['stop_loss']
                )
            if decision.holding_info.get('take_profit', 0) > 0:
                decision.take_profit_price = decision.holding_info['take_profit']

    def _make_decision(self, decision: TradeDecision, signal: AggregatedSignal):
        """确定最终决策"""
        score = decision.overall_score

        # 评级
        if score >= self.RATING_THRESHOLDS['strong_buy']:
            decision.rating = '强烈推荐'
        elif score >= self.RATING_THRESHOLDS['buy']:
            decision.rating = '推荐'
        elif score >= self.RATING_THRESHOLDS['neutral_high']:
            decision.rating = '中性偏多'
        elif score >= self.RATING_THRESHOLDS['neutral']:
            decision.rating = '中性'
        elif score >= self.RATING_THRESHOLDS['cautious']:
            decision.rating = '谨慎'
        else:
            decision.rating = '不推荐'

        # 操作建议
        if signal.signal == 'buy' and signal.strength >= self.min_signal_strength:
            # 检查是否有阻止买入的风险
            if any(a['type'] == 'total_position_limit' for a in decision.risk_alerts):
                decision.action = 'hold'
                decision.action_text = '观望（总仓位限制）'
            else:
                decision.action = 'buy'
                decision.action_text = '买入'
                decision.risk_level = '低' if score >= 80 else ('中' if score >= 70 else '高')

        elif signal.signal == 'sell' and signal.strength >= self.min_signal_strength:
            decision.action = 'sell'
            decision.action_text = '卖出'
            decision.risk_level = '高'

        elif decision.holding_info:
            # 已持仓但信号不强，检查是否需要止损止盈
            buy_price = decision.holding_info.get('buy_price', 0)
            if buy_price > 0 and decision.current_price > 0:
                pnl_pct = (decision.current_price / buy_price - 1)

                if pnl_pct <= decision.stop_loss_pct:
                    decision.action = 'sell'
                    decision.action_text = '止损卖出'
                    decision.risk_level = '高'
                elif pnl_pct >= decision.take_profit_pct:
                    decision.action = 'sell'
                    decision.action_text = '止盈卖出'
                    decision.risk_level = '低'
                else:
                    decision.action = 'hold'
                    decision.action_text = '继续持有'
            else:
                decision.action = 'hold'
                decision.action_text = '继续持有'
        else:
            decision.action = 'hold'
            decision.action_text = '观望'

    def _calculate_position(self, decision: TradeDecision, signal: AggregatedSignal):
        """计算建议仓位"""
        try:
            capital = self.portfolio_mgr.load_capital()
            total_capital = capital.get('总资金', 1000000)
        except Exception:
            total_capital = 1000000

        result = self.position_calc.suggest_position(
            signal_strength=signal.strength,
            total_capital=total_capital
        )

        decision.position_fraction = result.get('suggested_fraction', 0)
        decision.position_amount = result.get('suggested_amount', 0)
        decision.position_method = result.get('method', 'signal_based')

    def _apply_etf_decision_overlay(self, decision: TradeDecision, signal: AggregatedSignal):
        """
        ETF专用决策覆盖 - 将通用buy/sell/hold翻译为核心仓位+交易仓位建议

        长期持有AI ETF的投资者不需要简单的买/卖建议，
        而是需要知道核心仓位是否要调整 + 当前是否适合做差价。
        """
        # 从ETF分析结果中提取双层级建议
        etf_detail = signal.etf_detail or {}

        decision.core_advice = etf_detail.get('core_advice')
        decision.trading_advice = etf_detail.get('trading_advice')

        # 计算成本价盈亏
        if decision.holding_info and decision.current_price > 0:
            buy_price = decision.holding_info.get('buy_price', 0)
            if buy_price > 0:
                decision.cost_basis = buy_price
                decision.pnl_vs_cost = (decision.current_price / buy_price - 1)

        # 覆盖action_text为更细致的建议
        if decision.core_advice and decision.trading_advice:
            core_text = decision.core_advice.get('action_text', '')
            trade_text = decision.trading_advice.get('action_text', '')

            # 组合建议文本
            if decision.action == 'hold':
                decision.action_text = f"{core_text}；{trade_text}"
            elif decision.action == 'buy':
                decision.action_text = f"{core_text}；加仓做差价"
            elif decision.action == 'sell':
                # ETF不轻易"卖出"，改为"减仓做差价"
                decision.action = 'hold'
                decision.action_text = f"维持核心仓位；减仓做差价"

    def _generate_reasons(self, decision: TradeDecision, signal: AggregatedSignal):
        """生成决策理由"""
        reasons = decision.buy_reasons if decision.action == 'buy' else decision.sell_reasons

        # 技术面理由
        if signal.tech_detail:
            overall = signal.tech_detail.get('overall', {})
            macd = signal.tech_detail.get('macd', {})
            rsi = signal.tech_detail.get('rsi', {})
            kdj = signal.tech_detail.get('kdj', {})

            if decision.action == 'buy':
                if macd.get('signal') in ('buy', 'golden_cross', 'strong_buy'):
                    reasons.append(f"MACD {macd.get('description', '金叉信号')}")
                if rsi.get('value', 50) < 30:
                    reasons.append(f"RSI={rsi.get('value', 0):.0f} 超卖区域")
                if kdj.get('signal') == 'buy':
                    reasons.append("KDJ金叉")
            elif decision.action == 'sell':
                if macd.get('signal') in ('sell', 'death_cross', 'strong_sell'):
                    reasons.append(f"MACD {macd.get('description', '死叉信号')}")
                if rsi.get('value', 50) > 70:
                    reasons.append(f"RSI={rsi.get('value', 0):.0f} 超买区域")
                if kdj.get('signal') == 'sell':
                    reasons.append("KDJ死叉")

        # 基本面理由
        if signal.fund_detail and decision.action == 'buy':
            rating = signal.fund_detail.get('overall_rating', '')
            if rating in ('excellent', 'good'):
                reasons.append(f"基本面{rating}，财务健康")

        # MACD共振理由
        if signal.macd_detail:
            resonance = signal.macd_detail.get('analysis', {}).get('resonance', {})
            desc = resonance.get('description', '')
            if desc:
                reasons.append(desc)

        # 风险警告
        if decision.current_price > 0:
            decision.risk_warnings.append(
                f"止损价位: ¥{decision.stop_loss_price:.2f} ({decision.stop_loss_pct*100:.1f}%) [铁律v2动态]"
            )
            decision.risk_warnings.append(
                f"止盈价位: ¥{decision.take_profit_price:.2f} (+{decision.take_profit_pct*100:.1f}%) [铁律v2动态]"
            )

        if signal.anomalies:
            for anomaly in signal.anomalies[:3]:
                severity = anomaly.get('severity', 'low')
                desc = anomaly.get('description', '')
                if severity in ('high', 'medium'):
                    decision.risk_warnings.append(f"⚠️ {desc}")

        # 组合风险
        for alert in decision.risk_alerts:
            decision.risk_warnings.append(alert.get('message', ''))


# ── 测试 ──

if __name__ == '__main__':
    print("=" * 80)
    print("测试决策引擎")
    print("=" * 80)

    engine = DecisionEngine()

    decision = engine.decide('603881', '数据港', skip_macd_resonance=True)

    print(f"\n{'='*60}")
    print(f"📋 决策结果")
    print(f"{'='*60}")
    print(f"   操作: {decision.action_text}")
    print(f"   评级: {decision.rating}")
    print(f"   综合评分: {decision.overall_score:.1f}")
    print(f"   技术: {decision.tech_score:.1f} | 基本面: {decision.fund_score:.1f} | "
          f"MACD: {decision.macd_score:.1f}")
    print(f"   止损: ¥{decision.stop_loss_price:.2f} | 止盈: ¥{decision.take_profit_price:.2f}")
    if decision.buy_reasons:
        print(f"   买入理由: {', '.join(decision.buy_reasons)}")
    if decision.sell_reasons:
        print(f"   卖出理由: {', '.join(decision.sell_reasons)}")
    if decision.risk_warnings:
        print(f"   风险提示: {'; '.join(decision.risk_warnings)}")

    print(f"\n{'='*80}")
    print("测试完成")
    print("=" * 80)
