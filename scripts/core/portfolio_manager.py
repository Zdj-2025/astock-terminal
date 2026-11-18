#!/usr/bin/env python3
"""
持仓与交易记录管理系统。

解决痛点：
  - portfolio.xlsx 和 trade_log.xlsx 格式不统一
  - 经常搞混持仓数量、成本价
  - 缺少自动化的持仓跟踪

功能：
  1. portfolio_manager — 统一管理持仓CRUD
  2. 从trade_log自动重建portfolio（对账）
  3. 每日持仓快照+盈亏计算
  4. ATR止损/止盈检查集成

核心原则：
  - portfolio.xlsx 是持仓唯一真相（庄哥6/10确认）
  - trade_log.xlsx 是流水记录（每笔交易）
  - 每次交易后两个文件同步更新
  - 庄哥发截图时以截图为准

用法:
    from scripts.core.portfolio_manager import PortfolioManager
    pm = PortfolioManager()
    pm.show()                          # 显示当前持仓
    pm.add_trade(...)                  # 记录交易
    pm.rebuild_from_trades()           # 从流水重建持仓（对账）
    pm.daily_check()                   # 每日持仓检查

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json

PORTFOLIO_PATH = Path('./config/portfolio.xlsx')
TRADE_LOG_PATH = Path('./config/trade_log.xlsx')


@dataclass
class Position:
    """单只股票持仓。"""
    code: str
    name: str
    buy_date: str          # 首次买入日
    avg_cost: float        # 加权平均成本
    shares: int            # 当前持仓股数
    stop_loss: float = 0.0 # 止损价（0=未设）
    take_profit: float = 0.0
    position_type: str = ""  # 核心持仓/卫星持仓
    notes: str = ""
    last_price: float = 0.0  # 最新价（portfolio.xlsx 第11列，旧格式无此列时为0）

    @property
    def value_at_cost(self) -> float:
        return self.avg_cost * self.shares

    def value_at(self, price: float) -> float:
        return price * self.shares

    def pnl_at(self, price: float) -> tuple[float, float]:
        """返回 (盈亏金额, 盈亏比例%) """
        pnl = (price - self.avg_cost) * self.shares
        pnl_pct = (price - self.avg_cost) / self.avg_cost * 100 if self.avg_cost else 0
        return round(pnl, 0), round(pnl_pct, 1)


@dataclass
class Trade:
    """单笔交易记录。"""
    trade_date: str
    code: str
    name: str
    direction: str          # buy / sell
    price: float
    shares: int
    amount: float           # = price × shares
    trade_type: str = "normal"  # normal / t_buy / t_sell
    commission: float = 5.0
    notes: str = ""
    source: str = "manual"


class PortfolioManager:
    """
    持仓与交易记录管理器。

    统一管理 portfolio.xlsx 和 trade_log.xlsx。
    """

    def __init__(
        self,
        portfolio_path: Path = PORTFOLIO_PATH,
        trade_log_path: Path = TRADE_LOG_PATH,
    ):
        self.portfolio_path = portfolio_path
        self.trade_log_path = trade_log_path

    # ── 读持仓 ──────────────────────────────────────────

    def get_positions(self) -> list[Position]:
        """读取当前持仓。"""
        wb = openpyxl.load_workbook(self.portfolio_path)
        ws = wb.active
        positions = []
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
            if not row or not row[0] or row[0] == '——':
                continue
            row = list(row)[:11] + [None] * max(0, 11 - len(row))  # 弹性补齐: 兼容10列旧格式/11列新格式
            code, name, date_val, cost, shares, sl, tp, pct, ptype, note, last_price = row
            if shares and shares > 0:
                # 处理日期格式
                if isinstance(date_val, datetime):
                    date_str = date_val.strftime('%Y-%m-%d')
                elif isinstance(date_val, date):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val or '')

                positions.append(Position(
                    code=str(code).zfill(6) if str(code).isdigit() else str(code),
                    name=name or '',
                    buy_date=date_str,
                    avg_cost=float(cost or 0),
                    shares=int(shares or 0),
                    stop_loss=float(sl or 0),
                    take_profit=float(tp or 0),
                    position_type=ptype or '',
                    notes=note or '',
                    last_price=float(last_price or 0),
                ))
        return positions

    def get_position(self, code: str) -> Optional[Position]:
        """获取单只股票持仓。"""
        code = self._normalize_code(code)
        for p in self.get_positions():
            if p.code == code:
                return p
        return None

    # ── 读交易流水 ──────────────────────────────────────

    def get_trades(self) -> list[Trade]:
        """读取全部交易流水。"""
        wb = openpyxl.load_workbook(self.trade_log_path)
        ws = wb.active
        trades = []
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
            if not row or not row[0]:
                continue
            date_val, code, name, direction, price, shares, amount, ttype, comm, notes, source = row

            # 处理日期
            if isinstance(date_val, datetime):
                date_str = date_val.strftime('%Y-%m-%d')
            elif isinstance(date_val, date):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val or '')

            # 处理代码
            code_str = str(code) if code else ''
            if code_str.replace('.', '').isdigit():
                code_str = code_str.zfill(6)

            trades.append(Trade(
                trade_date=date_str,
                code=code_str,
                name=name or '',
                direction=direction or '',
                price=float(price or 0),
                shares=int(shares or 0),
                amount=float(amount or 0),
                trade_type=ttype or 'normal',
                commission=float(comm or 0),
                notes=notes or '',
                source=source or 'manual',
            ))
        return trades

    # ── 从流水重建持仓（对账） ──────────────────────────

    def rebuild_from_trades(self) -> dict[str, Position]:
        """
        从trade_log逐笔回放，重建持仓。

        返回: {code: Position}
        这是终极对账手段——不管portfolio怎么写，
        以实际交易流水为准。
        """
        trades = self.get_trades()
        positions: dict[str, Position] = {}

        for t in trades:
            code = t.code
            if not code or code == 'nan':
                continue

            # 做T的交易不影响持仓（当天买当天卖）
            # t_buy 后面有对应 t_sell 的不算持仓变化
            if t.trade_type in ('t_buy', 't_sell'):
                continue  # 做T流水不重建

            if code not in positions:
                positions[code] = Position(
                    code=code, name=t.name,
                    buy_date=t.trade_date,
                    avg_cost=t.price,
                    shares=0,
                )

            pos = positions[code]

            if t.direction == 'buy':
                # 加权平均成本
                old_value = pos.avg_cost * pos.shares
                new_value = t.price * t.shares
                total_shares = pos.shares + t.shares
                if total_shares > 0:
                    pos.avg_cost = round((old_value + new_value) / total_shares, 4)
                pos.shares = total_shares
                pos.name = t.name or pos.name

            elif t.direction == 'sell':
                pos.shares -= t.shares
                # 卖出后如果持仓为0，成本不变
                if pos.shares < 0:
                    print(f"  ⚠️ {code} {t.name} 持仓变为{pos.shares}，数据异常")
                # 卖出不改变成本价

        # 过滤掉持仓为0的
        active = {k: v for k, v in positions.items() if v.shares > 0}
        return active

    def reconcile(self) -> dict:
        """
        对账：比较portfolio.xlsx和trade_log重建结果。

        返回差异报告。
        """
        portfolio_positions = {p.code: p for p in self.get_positions()}
        rebuilt = self.rebuild_from_trades()

        all_codes = set(portfolio_positions.keys()) | set(rebuilt.keys())

        diffs = {
            'matched': [],
            'mismatch': [],
            'only_in_portfolio': [],
            'only_in_trades': [],
        }

        for code in sorted(all_codes):
            p_port = portfolio_positions.get(code)
            p_trade = rebuilt.get(code)

            if p_port and p_trade:
                if p_port.shares == p_trade.shares and abs(p_port.avg_cost - p_trade.avg_cost) < 0.01:
                    diffs['matched'].append({
                        'code': code, 'name': p_port.name,
                        'shares': p_port.shares, 'cost': p_port.avg_cost
                    })
                else:
                    diffs['mismatch'].append({
                        'code': code,
                        'name': p_port.name or (p_trade.name if p_trade else ''),
                        'portfolio': {'shares': p_port.shares, 'cost': p_port.avg_cost},
                        'trades': {'shares': p_trade.shares, 'cost': p_trade.avg_cost} if p_trade else None,
                    })
            elif p_port and not p_trade:
                diffs['only_in_portfolio'].append({
                    'code': code, 'name': p_port.name,
                    'shares': p_port.shares, 'cost': p_port.avg_cost
                })
            elif p_trade and not p_port:
                diffs['only_in_trades'].append({
                    'code': code, 'name': p_trade.name,
                    'shares': p_trade.shares, 'cost': p_trade.avg_cost
                })

        return diffs

    # ── 写交易 ──────────────────────────────────────────

    def add_trade(
        self,
        code: str,
        name: str,
        direction: str,
        price: float,
        shares: int,
        trade_type: str = "normal",
        notes: str = "",
        trade_date: str = None,
    ) -> None:
        """
        记录一笔交易，同时更新trade_log和portfolio。

        参数:
            code: 6位代码
            name: 股票名称
            direction: 'buy' / 'sell'
            price: 成交价
            shares: 成交股数
            trade_type: normal / t_buy / t_sell
            notes: 备注
            trade_date: 日期(YYYY-MM-DD)，默认今天
        """
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y-%m-%d')

        code = self._normalize_code(code)
        amount = round(price * shares, 2)

        # 1. 写入trade_log
        self._append_trade_log(
            trade_date, code, name, direction,
            price, shares, amount, trade_type, notes
        )

        # 2. 更新portfolio（做T不改变持仓）
        if trade_type not in ('t_buy', 't_sell'):
            self._update_portfolio(
                code, name, direction, price, shares, trade_date
            )

        print(f"✅ 已记录: {trade_date} {direction.upper()} {code} {name} "
              f"{shares}股@{price} = {amount:,.0f}元")

    def _append_trade_log(self, *args) -> None:
        """追加一行到trade_log。"""
        date_str, code, name, direction, price, shares, amount, ttype, notes = args
        wb = openpyxl.load_workbook(self.trade_log_path)
        ws = wb.active
        ws.append([date_str, code, name, direction, price, shares,
                   amount, ttype, 5, notes, 'manual'])
        wb.save(self.trade_log_path)

    def _update_portfolio(self, code, name, direction, price, shares, date_str) -> None:
        """更新portfolio中的持仓。"""
        wb = openpyxl.load_workbook(self.portfolio_path)
        ws = wb.active

        # 找现有持仓行
        target_row = None
        for row_idx in range(2, ws.max_row + 1):
            cell_code = ws.cell(row=row_idx, column=1).value
            if cell_code and self._normalize_code(str(cell_code)) == code:
                target_row = row_idx
                break

        if direction == 'buy':
            if target_row:
                # 加仓：更新成本和数量
                old_cost = float(ws.cell(row=target_row, column=4).value or 0)
                old_shares = int(ws.cell(row=target_row, column=5).value or 0)
                new_shares = old_shares + shares
                new_cost = round((old_cost * old_shares + price * shares) / new_shares, 4)
                ws.cell(row=target_row, column=4).value = new_cost
                ws.cell(row=target_row, column=5).value = new_shares
            else:
                # 新建仓
                ws.append([code, name, date_str, price, shares, 0, 0,
                          0, '卫星持仓', ''])

        elif direction == 'sell':
            if target_row:
                old_shares = int(ws.cell(row=target_row, column=5).value or 0)
                new_shares = old_shares - shares
                ws.cell(row=target_row, column=5).value = new_shares
                if new_shares <= 0:
                    # 清仓，标记
                    ws.cell(row=target_row, column=10).value = '已清仓'

        wb.save(self.portfolio_path)

    # ── 显示 ────────────────────────────────────────────

    def show(self, prices: dict[str, float] = None) -> str:
        """
        显示持仓概览。

        参数:
            prices: {code: 最新价}，None则只显示成本价
        """
        positions = self.get_positions()
        if not positions:
            return "📭 无持仓"

        lines = [
            f"{'='*60}",
            f"📊 当前持仓 ({len(positions)}只)",
            f"{'='*60}",
            f"{'代码':>8} {'名称':<10} {'股数':>6} {'成本':>8} {'现价':>8} "
            f"{'市值':>10} {'盈亏%':>8}",
            f"{'-'*60}",
        ]

        total_value = 0
        total_cost = 0
        for p in positions:
            price = (prices or {}).get(p.code, p.avg_cost)
            value = p.value_at(price)
            pnl, pnl_pct = p.pnl_at(price)
            total_value += value
            total_cost += p.value_at_cost
            lines.append(
                f"{p.code:>8} {p.name:<10} {p.shares:>6} "
                f"{p.avg_cost:>8.3f} {price:>8.2f} "
                f"{value:>10,.0f} {pnl_pct:>+7.1f}%"
            )

        total_pnl = total_value - total_cost
        total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0
        lines.extend([
            f"{'-'*60}",
            f"{'合计':>8} {'':<10} {'':>6} {'':>8} {'':>8} "
            f"{total_value:>10,.0f} {total_pnl_pct:>+7.1f}%",
        ])
        return "\n".join(lines)

    def show_trades(self, code: str = None, last_n: int = 20) -> str:
        """显示交易流水。"""
        trades = self.get_trades()
        if code:
            code = self._normalize_code(code)
            trades = [t for t in trades if t.code == code]

        trades = trades[-last_n:]
        lines = [
            f"{'='*60}",
            f"📋 交易流水 (最近{len(trades)}条{'，'+code if code else ''})",
            f"{'='*60}",
        ]
        for t in trades:
            emoji = '🟢' if t.direction == 'buy' else '🔴'
            lines.append(
                f"{t.trade_date} {emoji} {t.code} {t.name:<8} "
                f"{t.shares:>6}股@{t.price:.3f} = {t.amount:>10,.0f}元"
                f" [{t.trade_type}]"
            )
        return "\n".join(lines)

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _normalize_code(code: str) -> str:
        """统一代码格式为6位数字字符串。"""
        code = str(code).strip()
        if code.replace('.', '').isdigit():
            return code.zfill(6)
        return code


# ── CLI入口 ──────────────────────────────────────────────

if __name__ == '__main__':
    pm = PortfolioManager()

    print(pm.show())
    print()
    print(pm.show_trades(last_n=15))
    print()

    print("=" * 60)
    print("🔄 对账报告")
    print("=" * 60)
    diffs = pm.reconcile()

    print(f"\n✅ 一致: {len(diffs['matched'])}只")
    for m in diffs['matched']:
        print(f"  {m['code']} {m['name']}: {m['shares']}股@{m['cost']}")

    print(f"\n⚠️ 不一致: {len(diffs['mismatch'])}只")
    for m in diffs['mismatch']:
        print(f"  {m['code']} {m['name']}:")
        print(f"    portfolio: {m['portfolio']['shares']}股@{m['portfolio']['cost']}")
        if m['trades']:
            print(f"    trade_log: {m['trades']['shares']}股@{m['trades']['cost']}")

    print(f"\n📭 portfolio有/trade_log没有: {len(diffs['only_in_portfolio'])}只")
    for m in diffs['only_in_portfolio']:
        print(f"  {m['code']} {m['name']}: {m['shares']}股@{m['cost']}")

    print(f"\n📥 trade_log有/portfolio没有: {len(diffs['only_in_trades'])}只")
    for m in diffs['only_in_trades']:
        print(f"  {m['code']} {m['name']}: {m['shares']}股@{m['cost']}")
