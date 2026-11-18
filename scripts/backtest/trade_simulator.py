#!/usr/bin/env python3
"""
A股交易模拟器

T+1规则、佣金、印花税、滑点、涨跌停
"""
import pandas as pd
import numpy as np


class TradeSimulator:
    """模拟A股交易成本和约束"""

    COMMISSION_RATE = 0.00025   # 万2.5佣金
    MIN_COMMISSION = 5.0        # 最低佣金5元
    STAMP_TAX_RATE = 0.001     # 印花税0.1%（仅卖出）
    SLIPPAGE_RATE = 0.001       # 滑点0.1%
    PRICE_LIMIT = 0.10          # 涨跌停10%（主板）
    LOT_SIZE = 100              # 最小交易单位100股

    def __init__(self, initial_capital: float = 100000,
                 commission_rate: float = None,
                 stamp_tax_rate: float = None):
        self.initial_capital = initial_capital
        if commission_rate is not None:
            self.COMMISSION_RATE = commission_rate
        if stamp_tax_rate is not None:
            self.STAMP_TAX_RATE = stamp_tax_rate

        self.cash = initial_capital
        self.position = 0        # 持有股数
        self.avg_cost = 0.0      # 持仓均价
        self.trades = []         # 交易记录
        self.equity_curve = []   # 权益曲线
        self._pending_buy = None # T+1待结算买入

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.avg_cost = 0.0
        self.trades = []
        self.equity_curve = []
        self._pending_buy = None

    def _calc_commission(self, amount: float) -> float:
        return max(amount * self.COMMISSION_RATE, self.MIN_COMMISSION)

    def _calc_slippage(self, price: float, is_buy: bool) -> float:
        slip = price * self.SLIPPAGE_RATE
        return price + slip if is_buy else price - slip

    def can_buy(self, price: float, date_idx: int) -> bool:
        """检查是否可买入（T+1：买入当天不能买入已有仓位的卖出）"""
        fill_price = self._calc_slippage(price, True)
        max_shares = int(self.cash / (fill_price * (1 + self.COMMISSION_RATE)))
        max_shares = max_shares // self.LOT_SIZE * self.LOT_SIZE
        return max_shares >= self.LOT_SIZE

    def buy(self, price: float, date: str, shares: int = None,
            pct: float = None) -> dict:
        """
        买入股票

        Args:
            price: 买入价格
            date: 日期
            shares: 指定股数（与pct二选一）
            pct: 按可用资金的百分比买入
        """
        fill_price = self._calc_slippage(price, True)

        if shares is None:
            if pct is not None:
                budget = self.cash * pct
                shares = int(budget / (fill_price * (1 + self.COMMISSION_RATE)))
            else:
                shares = int(self.cash / (fill_price * (1 + self.COMMISSION_RATE)))
        shares = shares // self.LOT_SIZE * self.LOT_SIZE

        if shares <= 0:
            return {'success': False, 'reason': '资金不足'}

        amount = fill_price * shares
        commission = self._calc_commission(amount)
        total_cost = amount + commission

        if total_cost > self.cash:
            shares = int((self.cash - self.MIN_COMMISSION) / fill_price)
            shares = shares // self.LOT_SIZE * self.LOT_SIZE
            if shares <= 0:
                return {'success': False, 'reason': '资金不足'}
            amount = fill_price * shares
            commission = self._calc_commission(amount)
            total_cost = amount + commission

        # 更新持仓
        old_position = self.position
        old_cost = self.avg_cost
        self.position += shares
        if self.position > 0:
            self.avg_cost = ((old_position * old_cost) + total_cost) / self.position
        self.cash -= total_cost

        trade = {
            'date': date, 'action': 'buy', 'shares': shares,
            'price': fill_price, 'commission': commission,
            'amount': amount, 'cash_after': self.cash,
        }
        self.trades.append(trade)
        return {'success': True, **trade}

    def sell(self, price: float, date: str, shares: int = None,
             pct: float = None) -> dict:
        """
        卖出股票

        注意：T+1规则 — 当天买入的不能当天卖出。
        在回测中通过信号延迟处理（信号当天收盘后产生，次日执行）
        """
        if self.position <= 0:
            return {'success': False, 'reason': '无持仓'}

        if shares is None:
            if pct is not None:
                shares = int(self.position * pct)
            else:
                shares = self.position
        shares = min(shares, self.position)
        shares = shares // self.LOT_SIZE * self.LOT_SIZE
        if shares <= 0:
            shares = min(self.position, self.LOT_SIZE)

        fill_price = self._calc_slippage(price, False)
        amount = fill_price * shares
        commission = self._calc_commission(amount)
        stamp_tax = amount * self.STAMP_TAX_RATE
        total_proceeds = amount - commission - stamp_tax

        pnl = (fill_price - self.avg_cost) * shares - commission - stamp_tax

        self.position -= shares
        if self.position <= 0:
            self.position = 0
            self._last_avg_cost = self.avg_cost  # 保存供pnl_pct计算(P2-5修)
            self.avg_cost = 0
        self.cash += total_proceeds

        trade = {
            'date': date, 'action': 'sell', 'shares': shares,
            'price': fill_price, 'commission': commission,
            'stamp_tax': stamp_tax, 'amount': amount,
            'pnl': pnl, 'pnl_pct': pnl / (getattr(self, '_last_avg_cost', self.avg_cost) * shares) if shares > 0 and getattr(self, '_last_avg_cost', self.avg_cost) > 0 else 0,
            'cash_after': self.cash,
        }
        self.trades.append(trade)
        return {'success': True, **trade}

    def get_equity(self, current_price: float) -> float:
        """当前权益"""
        return self.cash + self.position * current_price

    def record_equity(self, date: str, price: float):
        """记录每日权益"""
        self.equity_curve.append({
            'date': date, 'equity': self.get_equity(price),
            'cash': self.cash, 'position': self.position,
            'price': price,
        })

    def get_equity_series(self) -> pd.Series:
        """获取权益曲线"""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve)
        return df.set_index('date')['equity']

    def get_returns_series(self) -> pd.Series:
        """获取每日收益率"""
        eq = self.get_equity_series()
        if len(eq) < 2:
            return pd.Series(dtype=float)
        return eq.pct_change().dropna()

    def get_trade_returns(self) -> pd.Series:
        """获取每笔交易的收益率"""
        sells = [t for t in self.trades if t['action'] == 'sell' and t.get('success', True)]
        if not sells:
            return pd.Series(dtype=float)
        return pd.Series([t['pnl_pct'] for t in sells])

    def summary(self) -> dict:
        """交易汇总"""
        sells = [t for t in self.trades if t['action'] == 'sell' and t.get('success', True)]
        buys = [t for t in self.trades if t['action'] == 'buy' and t.get('success', True)]
        total_commission = sum(t.get('commission', 0) for t in self.trades)
        total_stamp_tax = sum(t.get('stamp_tax', 0) for t in self.trades)
        total_pnl = sum(t.get('pnl', 0) for t in sells)

        return {
            'total_trades': len(buys) + len(sells),
            'buy_count': len(buys),
            'sell_count': len(sells),
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'total_stamp_tax': total_stamp_tax,
            'total_cost': total_commission + total_stamp_tax,
        }
