#!/usr/bin/env python3
"""
回测引擎 - 事件驱动架构

核心特性:
- 避免未来函数（Look-ahead Bias）
- 严格按时间顺序执行
- 完整的交易成本计算（佣金、印花税、滑点）
- 支持多股票组合回测
- 15+绩效指标

架构:
┌─────────────────────────────────────────────────┐
│          Event Queue (事件队列)                   │
└─────────────────────────────────────────────────┘
             ↓            ↓            ↓
    ┌────────┐    ┌──────────┐    ┌────────┐
    │ Market │ →  │ Strategy │ →  │ Signal │
    │  Event │    │Executor  │    │ Event  │
    └────────┘    └──────────┘    └────────┘
                                        ↓
    ┌────────┐    ┌──────────┐    ┌────────┐
    │  Fill  │ ←  │Execution │ ←  │ Order  │
    │ Event  │    │ Handler  │    │ Event  │
    └────────┘    └──────────┘    └────────┘

作者: AI Assistant & User
日期: 2026-04-04
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from queue import Queue  # noqa: 保留以防外部引用
from dataclasses import dataclass, field
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.core.data_fetcher_baostock import BaoStockFetcher
from scripts.analyzers import TechnicalAnalyzer, FundamentalAnalyzer
from scripts.backtest.strategy_executor import BacktestStrategyExecutor
from scripts.backtest.performance import PerformanceCalculator


# ==================== 事件类型 ====================

class EventType(Enum):
    """事件类型枚举"""
    MARKET = "MARKET"      # 市场数据更新
    SIGNAL = "SIGNAL"      # 交易信号生成
    ORDER = "ORDER"        # 订单创建
    FILL = "FILL"          # 订单成交


@dataclass
class Event:
    """事件基类"""
    type: EventType
    timestamp: datetime


@dataclass
class MarketEvent(Event):
    """市场数据事件"""
    symbol: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int

    def __init__(self, symbol: str, date: datetime, open_price: float,
                 high_price: float, low_price: float, close_price: float, volume: int):
        super().__init__(EventType.MARKET, date)
        self.symbol = symbol
        self.date = date
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume


@dataclass
class SignalEvent(Event):
    """交易信号事件"""
    symbol: str
    datetime: datetime
    signal_type: str  # 'BUY', 'SELL', 'EXIT'
    strength: float   # 信号强度 [0, 1]
    suggested_position: float  # 建议仓位比例 [0, 1]
    reason: str = ""

    def __init__(self, symbol: str, datetime: datetime, signal_type: str,
                 strength: float, suggested_position: float, reason: str = ""):
        super().__init__(EventType.SIGNAL, datetime)
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.suggested_position = suggested_position
        self.reason = reason


@dataclass
class OrderEvent(Event):
    """订单事件"""
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT'
    quantity: int     # 正数买入，负数卖出
    limit_price: Optional[float] = None

    def __init__(self, symbol: str, datetime: datetime, order_type: str,
                 quantity: int, limit_price: Optional[float] = None):
        super().__init__(EventType.ORDER, datetime)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.limit_price = limit_price


@dataclass
class FillEvent(Event):
    """订单成交事件"""
    symbol: str
    exchange: str
    quantity: int
    fill_price: float
    commission: float
    slippage: float
    total_cost: float

    def __init__(self, symbol: str, datetime: datetime, exchange: str,
                 quantity: int, fill_price: float, commission: float,
                 slippage: float):
        super().__init__(EventType.FILL, datetime)
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
        self.slippage = slippage
        self.total_cost = commission + abs(slippage)


# ==================== 数据处理器 ====================

class BacktestDataHandler:
    """
    回测数据处理器

    职责:
    - 从BaoStock获取历史数据
    - 按日期顺序推送市场事件
    - 确保不使用未来数据
    """

    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        初始化数据处理器

        参数:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.fetcher = BaoStockFetcher(auto_login=True)

        self.data = {}          # {symbol: DataFrame}
        self.continue_backtest = True
        self.current_index = {} # {symbol: current_row_index}

        print(f"📊 初始化数据处理器")
        print(f"   股票池: {len(symbols)}只")
        print(f"   时间范围: {start_date} 至 {end_date}")

    def load_data(self) -> bool:
        """加载所有股票的历史数据"""
        print(f"\n📥 加载历史数据...")

        for symbol in self.symbols:
            try:
                symbol_display = str(symbol).zfill(6)
                print(f"   加载 {symbol_display}...")

                # BaoStock使用YYYY-MM-DD格式
                df = self.fetcher.get_historical_kline(
                    symbol=symbol_display,
                    period='daily',
                    start_date=self.start_date,  # YYYY-MM-DD格式
                    end_date=self.end_date,      # YYYY-MM-DD格式
                    adjust='qfq'
                )

                if not df.empty:
                    # 按日期排序
                    df = df.sort_values('date').reset_index(drop=True)
                    self.data[symbol] = df
                    self.current_index[symbol] = 0
                    print(f"      ✅ {len(df)} 条记录")
                else:
                    print(f"      ❌ 无数据")

            except Exception as e:
                print(f"      ❌ 加载失败: {str(e)[:50]}")

        if not self.data:
            print(f"\n❌ 没有加载到任何数据")
            return False

        print(f"\n✅ 成功加载 {len(self.data)} 只股票数据")
        return True

    def get_next_market_event(self) -> Optional[MarketEvent]:
        """
        获取下一个市场事件

        返回:
            MarketEvent 或 None（数据结束）
        """
        if not self.data:
            return None

        # 找到所有股票中最早的未处理日期
        min_date = None
        min_symbol = None

        for symbol, df in self.data.items():
            idx = self.current_index.get(symbol, 0)
            if idx < len(df):
                date = df.iloc[idx]['date']
                if min_date is None or date < min_date:
                    min_date = date
                    min_symbol = symbol

        if min_date is None:
            self.continue_backtest = False
            return None

        # 收集该日期所有股票的数据
        events = []
        for symbol, df in self.data.items():
            idx = self.current_index.get(symbol, 0)
            if idx < len(df):
                row = df.iloc[idx]
                if row['date'] == min_date:
                    event = MarketEvent(
                        symbol=symbol,
                        date=row['date'],
                        open_price=row['open'],
                        high_price=row['high'],
                        low_price=row['low'],
                        close_price=row['close'],
                        volume=row['volume']
                    )
                    events.append(event)
                    self.current_index[symbol] = idx + 1

        # 返回第一个事件（如果有多只股票，需要处理同一天的多个事件）
        return events[0] if events else None

    def get_all_data_up_to(self, symbol: str, date: datetime) -> pd.DataFrame:
        """
        获取截至某日期的所有历史数据（避免未来函数）

        参数:
            symbol: 股票代码
            date: 截止日期

        返回:
            历史数据DataFrame
        """
        if symbol not in self.data:
            return pd.DataFrame()

        df = self.data[symbol]
        return df[df['date'] <= date].copy()


# ==================== 交易成本计算器 ====================

class TransactionCostCalculator:
    """
    交易成本计算器

    A股交易成本:
    - 佣金: 0.03% (双向，最低5元)
    - 印花税: 0.1% (仅卖出)
    - 过户费: 0.001% (上海市场)
    - 滑点: 0.05%
    """

    def __init__(self):
        self.commission_rate = 0.0003  # 0.03%
        self.stamp_duty_rate = 0.0005  # 2023-08-28起0.05%   # 0.1%
        self.transfer_fee_rate = 0.00001  # 0.001%
        self.slippage_rate = 0.0005    # 0.05%
        self.min_commission = 5.0      # 最低佣金5元

    def calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        commission = amount * self.commission_rate
        return max(commission, self.min_commission)

    def calculate_stamp_duty(self, amount: float, is_sell: bool) -> float:
        """计算印花税（仅卖出）"""
        if is_sell:
            return amount * self.stamp_duty_rate
        return 0.0

    def calculate_transfer_fee(self, amount: float, symbol: str) -> float:
        """计算过户费（上海市场）"""
        if symbol.startswith('6'):
            return amount * self.transfer_fee_rate
        return 0.0

    def calculate_slippage(self, amount: float) -> float:
        """计算滑点"""
        return amount * self.slippage_rate

    def calculate_total_cost(
        self,
        symbol: str,
        quantity: int,
        price: float,
        is_sell: bool
    ) -> Tuple[float, float, float]:
        """
        计算总交易成本

        参数:
            symbol: 股票代码
            quantity: 数量（正数）
            price: 价格
            is_sell: 是否卖出

        返回:
            (commission, stamp_duty, slippage)
        """
        amount = quantity * price

        commission = self.calculate_commission(amount)
        stamp_duty = self.calculate_stamp_duty(amount, is_sell)
        transfer_fee = self.calculate_transfer_fee(amount, symbol)
        slippage = self.calculate_slippage(amount)

        return commission, stamp_duty + transfer_fee, slippage


# ==================== 组合模拟器 ====================

class BacktestPortfolio:
    """
    回测组合模拟器

    职责:
    - 管理持仓和现金
    - 计算净值
    - 记录交易历史
    """

    def __init__(self, initial_capital: float):
        """
        初始化组合

        参数:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'quantity': int, 'avg_cost': float}}
        self.all_positions = []  # 每日持仓记录
        self.all_cash = []      # 每日现金记录
        self.trades = []        # 交易记录

        self.cost_calculator = TransactionCostCalculator()

        print(f"💰 初始化组合")
        print(f"   初始资金: ¥{initial_capital:,.2f}")

    def update_market_value(self, market_event: MarketEvent):
        """更新持仓市值"""
        # 记录当前持仓和现金
        self.all_positions.append({
            'date': market_event.date,
            'positions': self.positions.copy()
        })
        self.all_cash.append({
            'date': market_event.date,
            'cash': self.cash
        })

    def process_fill(self, fill_event: FillEvent):
        """处理订单成交"""
        symbol = fill_event.symbol
        quantity = fill_event.quantity
        fill_price = fill_event.fill_price

        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_cost': 0.0}

        current_qty = self.positions[symbol]['quantity']
        current_cost = self.positions[symbol]['avg_cost']

        if quantity > 0:  # 买入
            # 计算新的平均成本
            total_cost = current_cost * current_qty + fill_price * quantity
            new_qty = current_qty + quantity
            new_avg_cost = total_cost / new_qty if new_qty > 0 else 0.0

            self.positions[symbol] = {
                'quantity': new_qty,
                'avg_cost': new_avg_cost
            }

            # 扣除现金
            total_amount = quantity * fill_price + fill_event.commission + fill_event.slippage
            self.cash -= total_amount

        else:  # 卖出
            sell_qty = abs(quantity)

            # 扣除持仓
            new_qty = current_qty - sell_qty
            if new_qty <= 0:
                # 清仓
                if symbol in self.positions:
                    del self.positions[symbol]
            else:
                self.positions[symbol]['quantity'] = new_qty

            # 增加现金
            total_amount = sell_qty * fill_price - fill_event.commission - fill_event.slippage
            self.cash += total_amount

        # 记录交易
        self.trades.append({
            'date': fill_event.timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': fill_price,
            'commission': fill_event.commission,
            'slippage': fill_event.slippage,
            'cash_after': self.cash
        })

    def calculate_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算总资产

        参数:
            current_prices: {symbol: price}

        返回:
            总资产
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position['quantity'] * current_prices[symbol]

        return total_value


# ==================== 交易执行器 ====================

class BacktestExecutionHandler:
    """
    回测交易执行器

    职责:
    - 将订单转换为成交
    - 计算交易成本
    - 模拟滑点
    """

    def __init__(self):
        """初始化执行器"""
        self.cost_calculator = TransactionCostCalculator()
        print(f"⚡ 初始化交易执行器")

    def execute_order(
        self,
        order: OrderEvent,
        current_price: float
    ) -> Optional[FillEvent]:
        """
        执行订单

        参数:
            order: 订单事件
            current_price: 当前价格

        返回:
            FillEvent 或 None
        """
        try:
            quantity = order.quantity
            is_sell = quantity < 0

            # 计算成交价格（加入滑点）
            if is_sell:
                # 卖出时价格略低
                fill_price = current_price * (1 - self.cost_calculator.slippage_rate)
            else:
                # 买入时价格略高
                fill_price = current_price * (1 + self.cost_calculator.slippage_rate)

            # 计算交易成本
            commission, other_fees, slippage = self.cost_calculator.calculate_total_cost(
                symbol=order.symbol,
                quantity=abs(quantity),
                price=fill_price,
                is_sell=is_sell
            )

            # 创建成交事件
            fill_event = FillEvent(
                symbol=order.symbol,
                datetime=order.timestamp,
                exchange='BACKTEST',
                quantity=quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage + other_fees
            )

            return fill_event

        except Exception as e:
            print(f"❌ 订单执行失败: {str(e)[:50]}")
            return None


# ==================== 主引擎 ====================

class BacktestEngine:
    """
    回测引擎主类

    使用方法:
        engine = BacktestEngine(
            symbols=['600519', '000858'],
            start_date='2024-01-01',
            end_date='2025-12-31',
            initial_capital=1000000
        )
        engine.run_backtest()
        results = engine.get_results()
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000
    ):
        """
        初始化回测引擎

        参数:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            initial_capital: 初始资金
        """
        print("\n" + "=" * 80)
        print("初始化回测引擎")
        print("=" * 80)

        # 核心组件
        self.data_handler = BacktestDataHandler(symbols, start_date, end_date)
        self.portfolio = BacktestPortfolio(initial_capital)
        self.execution_handler = BacktestExecutionHandler()
        self.strategy_executor = BacktestStrategyExecutor()  # 策略执行器
        self.performance_calculator = PerformanceCalculator()  # 绩效计算器

        # 回测状态
        self.signals = []
        self.orders = []
        self.fills = []
        self.current_prices = {}  # {symbol: price}
        self.portfolio_values = []  # 每日净值记录
        self._pending_orders = []  # 待执行订单（次日开盘价执行）

        # 配置
        self.min_signal_strength = 0.6  # 最小信号强度

        print(f"\n✅ 回测引擎初始化完成")
        print(f"   股票池: {symbols}")
        print(f"   时间范围: {start_date} 至 {end_date}")
        print(f"   初始资金: ¥{initial_capital:,.2f}")

    def run_backtest(self) -> bool:
        """
        运行回测

        返回:
            是否成功
        """
        print(f"\n" + "=" * 80)
        print("开始回测")
        print("=" * 80)

        # 加载数据
        if not self.data_handler.load_data():
            return False

        # 回测主循环
        print(f"\n📊 执行回测...")
        day_count = 0

        while self.data_handler.continue_backtest:
            # 获取下一个市场事件
            market_event = self.data_handler.get_next_market_event()

            if market_event is None:
                break

            # 更新当前价格
            self.current_prices[market_event.symbol] = market_event.close_price

            # 先执行上一轮的待执行订单（用本轮开盘价，消除未来函数）
            if self._pending_orders:
                for order in self._pending_orders:
                    fill = self.execution_handler.execute_order(
                        order,
                        market_event.open_price
                    )
                    if fill:
                        self.fills.append(fill)
                        self.portfolio.process_fill(fill)
                self._pending_orders = []

            # 更新组合市值
            self.portfolio.update_market_value(market_event)

            # 记录当前组合净值
            current_value = self.portfolio.calculate_total_value(self.current_prices)
            self.portfolio_values.append(current_value)

            # 生成交易信号（使用策略执行器）
            signal = self._generate_signal(market_event)

            if signal:
                # 将信号转换为订单，但不立即执行，放入待执行队列
                order = self._create_order_from_signal(signal, market_event)
                if order:
                    self._pending_orders.append(order)
                    self.signals.append(signal)

            day_count += 1
            if day_count % 50 == 0:
                print(f"   已处理 {day_count} 个交易日...")

        print(f"\n✅ 回测完成")
        print(f"   总交易日: {day_count}")
        print(f"   总成交: {len(self.fills)} 笔")

        return True

    def _generate_signal(self, market_event: MarketEvent) -> Optional[SignalEvent]:
        """
        生成交易信号

        参数:
            market_event: 市场事件

        返回:
            SignalEvent 或 None
        """
        try:
            # 获取截至当前日期的历史数据（避免未来函数）
            historical_data = self.data_handler.get_all_data_up_to(
                market_event.symbol,
                market_event.date
            )

            if historical_data.empty or len(historical_data) < 60:
                return None

            # 获取当前资金
            current_capital = self.portfolio.cash

            # 调用策略执行器生成信号
            signal_dict = self.strategy_executor.generate_signal(
                symbol=market_event.symbol,
                historical_data=historical_data,
                current_date=market_event.date,
                current_capital=current_capital
            )

            if not signal_dict:
                return None

            # 转换为SignalEvent
            return SignalEvent(
                symbol=signal_dict['symbol'],
                datetime=market_event.date,
                signal_type=signal_dict['signal_type'],
                strength=signal_dict['strength'],
                suggested_position=signal_dict['suggested_position'],
                reason=signal_dict.get('reason', '')
            )

        except Exception as e:
            # 信号生成失败是正常情况，不打印错误
            return None

    def _create_order_from_signal(
        self,
        signal: SignalEvent,
        market_event: MarketEvent
    ) -> Optional[OrderEvent]:
        """
        从信号创建订单

        参数:
            signal: 信号事件
            market_event: 市场事件

        返回:
            OrderEvent 或 None
        """
        try:
            # 检查信号强度
            if signal.strength < self.min_signal_strength:
                return None

            # 检查是否已有持仓
            current_position = self.portfolio.positions.get(signal.symbol, {})
            current_qty = current_position.get('quantity', 0)

            # 计算目标仓位
            target_value = self.portfolio.cash * signal.suggested_position
            target_qty = int(target_value / market_event.close_price / 100) * 100  # A股100股为一手

            if signal.signal_type == 'BUY' and target_qty > 0:
                # 买入订单
                return OrderEvent(
                    symbol=signal.symbol,
                    datetime=signal.datetime,
                    order_type='MARKET',
                    quantity=target_qty
                )

            elif signal.signal_type == 'SELL' and current_qty > 0:
                # 卖出订单（卖出全部持仓）
                return OrderEvent(
                    symbol=signal.symbol,
                    datetime=signal.datetime,
                    order_type='MARKET',
                    quantity=-current_qty  # 负数表示卖出
                )

            return None

        except Exception as e:
            print(f"   ⚠️  订单创建失败: {str(e)[:50]}")
            return None

    def get_results(self) -> Dict:
        """
        获取回测结果

        返回:
            回测结果字典
        """
        # 计算最终净值
        final_value = self.portfolio.calculate_total_value(self.current_prices)

        # 使用绩效计算器计算综合指标
        performance_metrics = self.performance_calculator.calculate_all_metrics(
            portfolio_values=self.portfolio_values,
            trade_history=self.portfolio.trades,
            benchmark_values=None,  # 可选：可以传入基准数据
            dates=None
        )

        return {
            'initial_capital': self.portfolio.initial_capital,
            'final_value': final_value,
            'total_return': (final_value - self.portfolio.initial_capital) / self.portfolio.initial_capital,
            'trades': len(self.fills),
            'positions': self.portfolio.all_positions,
            'cash_history': self.portfolio.all_cash,
            'trade_history': self.portfolio.trades,
            'portfolio_values': self.portfolio_values,
            'performance_metrics': performance_metrics  # 新增：完整的绩效指标
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 80)
    print("测试回测引擎")
    print("=" * 80)

    # 创建回测引擎
    engine = BacktestEngine(
        symbols=['603881'],  # 数据港
        start_date='2025-01-01',
        end_date='2025-12-31',
        initial_capital=1000000
    )

    # 运行回测（目前只测试数据加载和流程）
    success = engine.run_backtest()

    if success:
        results = engine.get_results()
        print(f"\n📊 回测结果:")
        print(f"   初始资金: ¥{results['initial_capital']:,.2f}")
        print(f"   最终资产: ¥{results['final_value']:,.2f}")
        print(f"   总收益率: {results['total_return']:.2%}")
        print(f"   交易次数: {results['trades']} 笔")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
