#!/usr/bin/env python3
"""
海龟交易法融合模块集成测试。

将ATR仓位管理 + 三重止损 + 趋势退出 + 突破检测
串联成一个完整的决策流程。

流程：
  突破检测 → V3评分(外部) → ATR仓位 → 止损设置
  持有期间：止损检查 + 退出信号检查

用法:
    PYTHONPATH=. python scripts/core/turtle_fusion_demo.py

作者: Elon (OpenClaw)
日期: 2026-06-14
"""
from __future__ import annotations

import sys
sys.path.insert(0, '.')

from scripts.core.atr_calculator import ATRCalculator, get_atr_from_store
from scripts.core.turtle_position import TurtlePosition
from scripts.core.stop_manager import StopManager
from scripts.core.trend_exit import TrendExit
from scripts.core.breakout_detector import BreakoutDetector


class TurtleFusionEngine:
    """海龟融合决策引擎 — 一站式接口。"""

    def __init__(self, total_capital: float = 500000):
        self.atr_calc = ATRCalculator()
        self.position = TurtlePosition(total_capital=total_capital)
        self.stop_mgr = StopManager()
        self.exit_mgr = TrendExit()
        self.breakout = BreakoutDetector()

    def analyze(
        self,
        code: str,
        kline_df,
        v3_score: float,
        signal_type: str = 'uptrend',
        mtf_confirm: str = '',
        spring_squeeze: bool = False,
        existing_shares: int = 0,
        existing_avg_cost: float = 0.0,
        peak_price: float = 0.0,
        last_add_price: float = 0.0,
        f2_history: list = None,
    ) -> dict:
        """
        完整分析：入场决策 + 仓位 + 止损 + 退出信号。

        返回:
            {
                'action': 'BUY' / 'HOLD' / 'WATCH' / 'REDUCE' / 'EXIT',
                'breakout': BreakoutResult,
                'position': TurtleResult,
                'stop': StopResult,
                'exit': ExitResult,
                'summary': 可读摘要,
            }
        """
        import pandas as pd
        if kline_df is None or kline_df.empty:
            return {'action': 'SKIP', 'summary': f'{code}: 无K线数据'}

        # 计算ATR
        atr = self.atr_calc.calculate(kline_df)
        if atr <= 0:
            return {'action': 'SKIP', 'summary': f'{code}: ATR计算失败'}

        current_price = float(kline_df['close'].iloc[-1])

        # 利润垫
        if existing_shares > 0 and existing_avg_cost > 0:
            profit_cushion = (current_price - existing_avg_cost) / existing_avg_cost
        else:
            profit_cushion = 0.0

        # 峰值价
        if peak_price <= 0:
            if existing_shares > 0:
                peak_price = max(current_price, existing_avg_cost)
            else:
                peak_price = current_price

        # ── 1. 突破检测 ──
        breakout_r = self.breakout.detect(
            kline_df, v3_score=v3_score, atr=atr
        )

        # ── 2. 已有持仓 → 检查止损和退出 ──
        if existing_shares > 0:
            stop_r = self.stop_mgr.check_stop(
                code=code, entry_price=existing_avg_cost, atr=atr,
                current_price=current_price, peak_price=peak_price,
                v3_score=v3_score, f2_history=f2_history,
                last_add_price=last_add_price,
                profit_cushion=profit_cushion,
            )
            exit_r = self.exit_mgr.check_exit(
                code=code, kline_df=kline_df, v3_score=v3_score,
                atr=atr, current_price=current_price,
                profit_cushion=profit_cushion,
                f2_history=f2_history,
            )

            # 综合判断动作
            from scripts.core.stop_manager import Action as SAct
            from scripts.core.trend_exit import ExitAction as EAct

            if stop_r.action == SAct.EXIT or exit_r.action == EAct.EXIT:
                action = 'EXIT'
            elif stop_r.action == SAct.REDUCE or exit_r.action == EAct.REDUCE:
                action = 'REDUCE'
            elif stop_r.action == SAct.WATCH or exit_r.action == EAct.WATCH:
                action = 'WATCH'
            else:
                action = 'HOLD'

            return {
                'action': action,
                'breakout': breakout_r,
                'stop': stop_r,
                'exit': exit_r,
                'atr': atr,
                'summary': self._format_holdings(
                    code, action, stop_r, exit_r, atr,
                    current_price, profit_cushion
                ),
            }

        # ── 3. 新建仓 → 计算仓位和止损 ──
        pos_r = self.position.calculate(
            code=code, price=current_price, atr=atr,
            v3_score=v3_score, signal_type=signal_type,
            mtf_confirm=mtf_confirm, spring_squeeze=spring_squeeze,
            profit_cushion=0.0,
        )

        return {
            'action': 'BUY' if pos_r.suggested_shares > 0 else 'WATCH',
            'breakout': breakout_r,
            'position': pos_r,
            'atr': atr,
            'summary': self._format_new_buy(
                code, pos_r, breakout_r, atr, current_price
            ),
        }

    def _format_new_buy(self, code, pos, breakout, atr, price):
        lines = [
            f"{'='*55}",
            f"🐢 海龟融合分析 — {code}",
            f"{'='*55}",
            f"📊 基础数据: 价格{price:.2f} | ATR(20){atr:.3f} | 波动率{atr/price*100:.1f}%",
            f"",
            f"{'─'*55}",
            f"📈 突破检测:",
            f"{breakout.summary()}",
            f"",
            f"{'─'*55}",
            f"💰 仓位建议:",
            f"{pos.summary()}",
        ]
        return "\n".join(lines)

    def _format_holdings(self, code, action, stop, exit_r, atr, price, cushion):
        action_emoji = {
            'EXIT': '🔴 清仓', 'REDUCE': '🟡 减仓',
            'WATCH': '👀 关注', 'HOLD': '✅ 持有',
        }.get(action, action)
        lines = [
            f"{'='*55}",
            f"🐢 海龟融合分析 — {code} (持仓)",
            f"{'='*55}",
            f"📊 基础数据: 价格{price:.2f} | ATR{atr:.3f} | 利润垫{cushion*100:+.1f}%",
            f"",
            f"🎯 综合判断: {action_emoji}",
            f"",
            f"{'─'*55}",
            f"🛡️ 止损检查:",
            f"{stop.summary()}",
            f"",
            f"{'─'*55}",
            f"📤 退出信号:",
            f"{exit_r.summary()}",
        ]
        return "\n".join(lines)


# ── CLI演示 ──────────────────────────────────────────────

if __name__ == '__main__':
    from scripts.core.data_store import DataStore

    print("=" * 55)
    print("🐢 海龟融合引擎 — 集成演示")
    print("=" * 55)

    store = DataStore()
    engine = TurtleFusionEngine(total_capital=500000)

    # 测试1: 新建仓分析
    print("\n\n" + "🔹"*27)
    print("场景A: 新建仓分析")
    print("🔹"*27)

    for code in ['002594', '000001', '688981']:
        df = store.get_kline(code, period='D', auto_update=False)
        if df is None or df.empty:
            continue
        result = engine.analyze(
            code=code, kline_df=df,
            v3_score=7.0, signal_type='uptrend',
            mtf_confirm='partial',
        )
        print("\n" + result['summary'])

    # 测试2: 已有持仓检查
    print("\n\n" + "🔹"*27)
    print("场景B: 已有持仓检查")
    print("🔹"*27)

    for code, cost, shares in [('002594', 85.0, 1000), ('000001', 10.5, 20000)]:
        df = store.get_kline(code, period='D', auto_update=False)
        if df is None or df.empty:
            continue
        result = engine.analyze(
            code=code, kline_df=df,
            v3_score=7.0, signal_type='uptrend',
            existing_shares=shares, existing_avg_cost=cost,
        )
        print("\n" + result['summary'])
