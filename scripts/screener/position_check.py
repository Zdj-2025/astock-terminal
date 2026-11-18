#!/usr/bin/env python3
"""
每日持仓检查 — 海龟融合版。

每天运行一次，输出每只持仓的：
  1. ATR波动率 + 2ATR止损价
  2. 移动止损（持有期间最高价-2ATR）
  3. 10/20日低点退出信号
  4. V3评分趋势（需要外部传入或跳过）
  5. 建议操作（持有/关注/减仓/清仓）

用法：
  PYTHONPATH=. python scripts/screener/position_check.py
  PYTHONPATH=. python scripts/screener/position_check.py --v3  # 含V3评分

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.data_store import DataStore
from scripts.core.atr_calculator import ATRCalculator
from scripts.core.stop_manager import StopManager, Action as StopAction
from scripts.core.trend_exit import TrendExit, ExitAction as ExitA
from scripts.core.portfolio_manager import PortfolioManager
from scripts.core.turtle_position import TurtlePosition


def run_check(
    include_v3: bool = False,
    total_capital: float = None,
    prices_override: dict = None,
) -> str:
    """
    运行持仓检查。

    Args:
        include_v3: 是否计算V3评分（慢）
        total_capital: 总资金（None=从持仓反推）
        prices_override: {code: 价格} 手动指定价格

    Returns:
        格式化的检查报告
    """
    pm = PortfolioManager()
    store = DataStore()
    atr_calc = ATRCalculator()
    sm = StopManager()
    te = TrendExit()

    positions = pm.get_positions()
    if not positions:
        return "📭 无持仓"

    # 总资金：从portfolio反推（持仓成本+假设现金）
    if total_capital is None:
        # 从截图已知：425510
        total_capital = 425510

    tp = TurtlePosition(total_capital=total_capital)

    lines = [
        f"{'='*60}",
        f"🐢 每日持仓检查 — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*60}",
    ]

    # V3评分器（可选）
    scorer = None
    if include_v3:
        try:
            from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
            scorer = V3DerivativeScorer()
        except Exception:
            pass

    summary = {
        'hold': 0, 'watch': 0, 'reduce': 0, 'exit': 0
    }
    total_pnl = 0
    total_value = 0

    for p in positions:
        # 获取K线
        df = store.get_kline(p.code, period='D', auto_update=False)
        if df is None or df.empty:
            lines.append(f"\n⚠️ {p.code} {p.name}: 无K线数据，跳过")
            continue

        # 计算ATR
        atr = atr_calc.calculate(df)
        if atr <= 0:
            lines.append(f"\n⚠️ {p.code} {p.name}: ATR计算失败")
            continue

        # 当前价
        if prices_override and p.code in prices_override:
            current_price = prices_override[p.code]
        else:
            current_price = float(df['close'].iloc[-1])

        # 持有期间最高价（peak）
        # 用近60日最高，排除当日（用前一日收盘前的peak）
        lookback = min(60, len(df))
        if len(df) >= 2:
            peak_price = float(df['close'].iloc[-lookback:-1].max())
            # 如果前一日peak < 当前价，说明还在创新高
            if peak_price < current_price:
                peak_price = current_price  # 创新高时peak=当前价
        else:
            peak_price = current_price

        # 利润垫
        profit_cushion = (current_price - p.avg_cost) / p.avg_cost if p.avg_cost > 0 else 0
        pnl, pnl_pct = p.pnl_at(current_price)
        total_pnl += pnl
        total_value += current_price * p.shares

        # V3评分
        v3_score = 7.0  # 默认值
        v3_tag = ""
        if scorer:
            try:
                result = scorer.score_stock(p.code, df)
                v3_score = result.total
                v3_tag = f" V3={v3_score:.1f}"
            except Exception:
                pass

        # 止损检查
        # 利润垫<0时只用硬止损，不用移动止损（避免建仓初期被洗出）
        effective_peak = peak_price if profit_cushion > 0 else current_price
        stop_r = sm.check_stop(
            code=p.code, entry_price=p.avg_cost, atr=atr,
            current_price=current_price, peak_price=effective_peak,
            v3_score=v3_score, profit_cushion=profit_cushion,
        )

        # 退出信号
        exit_r = te.check_exit(
            code=p.code, kline_df=df, v3_score=v3_score,
            atr=atr, current_price=current_price,
            profit_cushion=profit_cushion,
        )

        # 综合判断
        actions = []
        if stop_r.action == StopAction.EXIT:
            actions.append('🔴止损')
            summary['exit'] += 1
        elif exit_r.action == ExitA.EXIT:
            actions.append('🔴退出')
            summary['exit'] += 1
        elif stop_r.action == StopAction.REDUCE or exit_r.action == ExitA.REDUCE:
            actions.append('🟡减仓')
            summary['reduce'] += 1
        elif stop_r.action == StopAction.WATCH or exit_r.action == ExitA.WATCH:
            actions.append('👀关注')
            summary['watch'] += 1
        else:
            actions.append('✅持有')
            summary['hold'] += 1

        action_str = '/'.join(actions)
        pnl_emoji = '🟢' if pnl > 0 else '🔴'

        # ATR信息
        atr_pct = atr / current_price * 100
        stop_2atr = current_price - 2 * atr

        # 仓位建议
        pos_r = tp.calculate(
            code=p.code, price=current_price, atr=atr,
            v3_score=v3_score, signal_type='uptrend',
            profit_cushion=profit_cushion,
        )

        # 输出
        lines.extend([
            f"",
            f"{'─'*60}",
            f"{pnl_emoji} **{p.code} {p.name}** | {action_str}{v3_tag}",
            f"  持仓: {p.shares:,}股@{p.avg_cost:.3f} → {current_price:.2f} "
            f"| {pnl_pct:+.1f}% ({pnl:+,.0f}元)",
            f"  ATR: {atr:.3f}({atr_pct:.1f}%) | 2ATR止损: {stop_2atr:.2f} "
            f"({(stop_2atr-current_price)/current_price*100:+.1f}%)",
            f"  止损线: {stop_r.current_stop:.2f} | 距离: {stop_r.distance_pct:+.1f}%",
            f"  退出线: {exit_r.channel_low:.2f}({exit_r.channel_period}日低) "
            f"| 距离: {exit_r.distance_to_low:.1f}ATR",
        ])

        # 加仓建议（仅当持有且V3≥7时）
        if v3_score >= 7.0 and profit_cushion >= 0:
            next_add = current_price + 0.5 * atr
            lines.append(
                f"  加仓参考: 下一加仓价{next_add:.2f}(+0.5ATR) "
                f"| 1单位={pos_r.unit_shares}股"
            )

        # 风险提示
        position_value = current_price * p.shares
        position_pct = position_value / total_capital * 100
        max_loss = 2 * atr * p.shares
        max_loss_pct = max_loss / total_capital * 100
        lines.append(
            f"  仓位: {position_value:,.0f}元({position_pct:.1f}%) "
            f"| 最大风险: {max_loss:,.0f}元({max_loss_pct:.1f}%)"
        )

        # 止损/退出原因
        if stop_r.reason:
            lines.append(f"  📝 {stop_r.reason}")
        if exit_r.reason and exit_r.action != ExitA.HOLD:
            lines.append(f"  📝 {exit_r.reason}")

    # 汇总
    lines.extend([
        f"",
        f"{'='*60}",
        f"📊 持仓汇总",
        f"{'='*60}",
        f"总市值: {total_value:,.0f}元 | 浮动盈亏: {total_pnl:+,.0f}元 "
        f"({total_pnl/(total_value-total_pnl)*100:+.1f}%)",
        f"✅持有: {summary['hold']} | 👀关注: {summary['watch']} | "
        f"🟡减仓: {summary['reduce']} | 🔴退出: {summary['exit']}",
    ])

    # 建议操作清单
    if summary['exit'] > 0 or summary['reduce'] > 0:
        lines.append(f"\n⚠️ **需要操作**: 有{summary['exit']+summary['reduce']}只需要减仓/清仓")
    elif summary['watch'] > 0:
        lines.append(f"\n👀 {summary['watch']}只需要密切关注")
    else:
        lines.append(f"\n✅ 全部正常持有，无需操作")

    text = "\n".join(lines)
    return text


def main():
    parser = argparse.ArgumentParser(description="每日持仓检查（海龟融合版）")
    parser.add_argument('--v3', action='store_true', help='计算V3评分（较慢）')
    parser.add_argument('--capital', type=float, default=None, help='总资金')
    args = parser.parse_args()

    report = run_check(include_v3=args.v3, total_capital=args.capital)
    print(report)


if __name__ == '__main__':
    main()
