#!/usr/bin/env python3
"""
资金效率分析器 - 评估每笔资金的使用效率

核心指标:
1. 单仓效率评分 (0-100): 趋势得分(40%) + 动能得分(30%) + 资金回报率(30%)
2. 效率等级: A(80+)/B(60-79)/C(40-59)/D(<40)
3. 组合效率总览: 按等级分布、加权效率分、死资本占比
4. 轮动建议: 释放D级资金到更高效标的
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


# 效率等级定义
EFFICIENCY_GRADES = {
    'A': {'min': 80, 'label': '优秀', 'color': '🟢', 'action': '继续持有/加仓'},
    'B': {'min': 60, 'label': '良好', 'color': '🟡', 'action': '维持'},
    'C': {'min': 40, 'label': '低效', 'color': '🟠', 'action': '关注，考虑减仓'},
    'D': {'min': 0,  'label': '僵尸', 'color': '🔴', 'action': '强烈建议清仓/减仓'},
}


def _grade(score: float) -> str:
    if score >= 80: return 'A'
    if score >= 60: return 'B'
    if score >= 40: return 'C'
    return 'D'


def _grade_info(grade: str) -> Dict:
    return EFFICIENCY_GRADES[grade]


class CapitalEfficiencyAnalyzer:
    """资金效率分析器"""

    # ── 铁律v2 持有三理由 ──
    HOLD_REASONS = {
        'trend_up': '趋势向上（周线多头+日线MACD>0）',
        'deep_profit': '深利保护（盈利>50%，止损线远在成本之上）',
        'catalyst': '即将催化（7天内有明确事件驱动）',
    }

    def __init__(self):
        pass

    def check_hold_reasons(self, holding: Dict, tech: Optional[Dict] = None) -> Dict:
        """铁律v2持有三理由检查。

        每只持仓必须满足以下至少一条，否则建议退出：
        1. 趋势向上：周线多头 + 日线MACD>0
        2. 深利保护：盈利>50%，止损线远在成本之上
        3. 即将催化：7天内有明确事件驱动

        Returns:
            {pass, reasons: [...], verdict: '持有'/'退出', details}
        """
        reasons = []
        details = []

        pnl_pct = holding.get('pnl_pct', 0)
        cost = holding.get('cost', 0)
        price = holding.get('price', 0)

        # 理由1：趋势向上
        trend_ok = False
        if tech:
            weekly = tech.get('weekly_trend', 'neutral')
            daily_macd = tech.get('daily_macd_hist', 0)
            if weekly in ('bullish',) and daily_macd > 0:
                trend_ok = True
                reasons.append('trend_up')
                details.append(f"周线多头 + 日线MACD红柱({daily_macd:.4f})")
            elif weekly in ('bullish',):
                trend_ok = True
                reasons.append('trend_up')
                details.append(f"周线多头（日线MACD偏弱）")
        elif pnl_pct > -5:  # 无技术数据时，用盈亏辅助判断
            trend_ok = True
            reasons.append('trend_up')
            details.append("无技术数据，以近盈亏辅助判断")

        # 理由2：深利保护
        deep_profit = pnl_pct > 50
        if deep_profit:
            reasons.append('deep_profit')
            protected_sl = cost * 1.25 if cost > 0 else 0
            details.append(f"盈利{pnl_pct:+.1f}%，止损线上移至成本+25%({protected_sl:.2f})")

        # 理由3：即将催化（需要外部提供催化信息）
        catalyst = holding.get('has_catalyst', False)
        catalyst_desc = holding.get('catalyst_desc', '')
        if catalyst:
            reasons.append('catalyst')
            details.append(f"催化事件: {catalyst_desc}")

        passed = len(reasons) >= 1
        verdict = '持有' if passed else '⚠️ 不满足持有理由，建议退出'

        return {
            'code': holding.get('code', ''),
            'name': holding.get('name', ''),
            'pass': passed,
            'reasons': reasons,
            'reason_count': len(reasons),
            'verdict': verdict,
            'details': details,
            'pnl_pct': pnl_pct,
        }

    def check_portfolio_hold_reasons(self, positions: List[Dict], tech_map: Optional[Dict] = None) -> Dict:
        """批量检查组合内所有持仓的持有理由。

        Args:
            positions: 各持仓的 holding dict 列表
            tech_map: {code: {weekly_trend, daily_macd_hist, ...}}
        Returns:
            {total, passed, failed, exit_list, details}
        """
        tech_map = tech_map or {}
        results = []
        exit_list = []

        for h in positions:
            code = h.get('code', '')
            tech = tech_map.get(code)
            check = self.check_hold_reasons(h, tech)
            results.append(check)
            if not check['pass']:
                exit_list.append(check)

        return {
            'total': len(results),
            'passed': len(results) - len(exit_list),
            'failed': len(exit_list),
            'exit_list': exit_list,
            'all_results': results,
        }

    def score_position(self, holding: Dict, tech: Optional[Dict] = None) -> Dict:
        """
        计算单仓资金效率分

        Args:
            holding: {code, name, cost, price, shares, market_value,
                      buy_date, pnl_pct}
            tech: 可选技术指标 {weekly_trend, daily_trend, daily_rsi,
                  daily_macd_hist, volume_ratio}
        Returns:
            {score, grade, trend_score, momentum_score, return_score,
             annualized_return, holding_days, dead_capital, opportunity_cost}
        """
        pnl_pct = holding.get('pnl_pct', 0)
        market_value = holding.get('market_value', 0)

        # 持仓天数
        buy_date = holding.get('buy_date')
        if buy_date:
            if isinstance(buy_date, str):
                buy_date = datetime.strptime(buy_date, '%Y-%m-%d')
            holding_days = max((datetime.now() - buy_date).days, 1)
        else:
            holding_days = 30  # 默认30天

        # === 趋势得分 (0-100, 权重40%) ===
        trend_score = 50  # 默认中性
        if tech:
            ts = 0
            # 周线趋势 (权重50%)
            wt = tech.get('weekly_trend', 'neutral')
            if wt == 'bullish':
                ts += 50
            elif wt == 'bearish':
                ts -= 40
            else:
                ts += 10
            # 日线趋势 (权重30%)
            dt = tech.get('daily_trend', 'neutral')
            if dt == 'bullish':
                ts += 30
            elif dt == 'bearish':
                ts -= 25
            else:
                ts += 5
            # 均线排列 (权重20%)
            ma_align = tech.get('ma_alignment', 'mixed')
            if ma_align == 'bullish':
                ts += 20
            elif ma_align == 'bearish':
                ts -= 20
            trend_score = max(0, min(100, 50 + ts))

        # === 动能得分 (0-100, 权重30%) ===
        momentum_score = 50
        if tech:
            ms = 0
            # MACD柱状线方向
            macd_hist = tech.get('daily_macd_hist_change', 0)
            if macd_hist > 0:
                ms += 25
            elif macd_hist < 0:
                ms -= 20
            # RSI位置
            rsi = tech.get('daily_rsi', 50)
            if 40 <= rsi <= 60:
                ms += 20  # 中性健康
            elif 60 < rsi <= 75:
                ms += 30  # 偏强
            elif 30 <= rsi < 40:
                ms -= 5   # 偏弱但有反弹机会
            elif rsi > 75:
                ms += 15  # 强但注意超买
            elif rsi < 30:
                ms -= 15  # 超卖弱势
            # 成交量
            vol_ratio = tech.get('volume_ratio', 1.0)
            if vol_ratio > 1.5:
                ms += 15  # 放量活跃
            elif vol_ratio < 0.7:
                ms -= 10  # 缩量低迷
            momentum_score = max(0, min(100, 50 + ms))

        # === 资金回报率得分 (0-100, 权重30%) ===
        # 年化回报率
        annualized_return = (pnl_pct / 100 + 1) ** (365 / holding_days) - 1
        annualized_return_pct = annualized_return * 100

        # 映射到0-100分
        # >50%年化→100, 30%→85, 10%→70, 0%→50, -10%→30, -30%→10, <-50%→0
        if annualized_return_pct >= 50:
            return_score = 100
        elif annualized_return_pct >= 0:
            return_score = 50 + annualized_return_pct * 1.0
        elif annualized_return_pct >= -30:
            return_score = 50 + annualized_return_pct * 1.33
        else:
            return_score = max(0, 50 + annualized_return_pct * 0.8)

        # 加权总分
        total_score = (trend_score * 0.4 + momentum_score * 0.3 + return_score * 0.3)
        total_score = round(total_score, 1)

        grade = _grade(total_score)

        # 死资本判定：D级 + 亏损
        is_dead = grade == 'D' and pnl_pct < 0

        return {
            'code': holding.get('code', ''),
            'name': holding.get('name', ''),
            'score': total_score,
            'grade': grade,
            'grade_info': _grade_info(grade),
            'trend_score': round(trend_score, 1),
            'momentum_score': round(momentum_score, 1),
            'return_score': round(return_score, 1),
            'annualized_return_pct': round(annualized_return_pct, 1),
            'holding_days': holding_days,
            'market_value': market_value,
            'pnl_pct': pnl_pct,
            'is_dead_capital': is_dead,
        }

    def score_portfolio(self, positions: List[Dict], total_capital: float,
                        available_cash: float = 0) -> Dict:
        """
        组合效率总览

        Args:
            positions: 各持仓的 score_position 结果列表
            total_capital: 总资金
            available_cash: 可用现金
        Returns:
            {weighted_score, grade_distribution, dead_capital_pct,
             cash_ratio, issues, rotation_advice}
        """
        if not positions:
            return {'error': '无持仓数据'}

        # 按等级分组
        groups = {'A': [], 'B': [], 'C': [], 'D': []}
        for p in positions:
            groups[p['grade']].append(p)

        # 加权效率分
        total_mv = sum(p['market_value'] for p in positions)
        if total_mv > 0:
            weighted_score = sum(
                p['score'] * p['market_value'] for p in positions
            ) / total_mv
        else:
            weighted_score = 0

        # 各等级资金占比
        grade_capital = {}
        for g, ps in groups.items():
            mv = sum(p['market_value'] for p in ps)
            grade_capital[g] = {
                'market_value': mv,
                'pct': mv / total_capital * 100 if total_capital > 0 else 0,
                'positions': ps,
            }

        # 死资本
        dead = [p for p in positions if p['is_dead_capital']]
        dead_mv = sum(p['market_value'] for p in dead)
        dead_pct = dead_mv / total_capital * 100 if total_capital > 0 else 0

        # 现金占比
        cash_ratio = available_cash / total_capital * 100 if total_capital > 0 else 0

        # 问题检测
        issues = []
        if dead_pct > 10:
            issues.append(f'{dead_pct:.1f}%资金在D级亏损仓位（死资本），建议释放')
        if cash_ratio < 15:
            issues.append(f'现金仅{cash_ratio:.1f}%，低于15%安全线，T操作空间不足')
        if cash_ratio > 40:
            issues.append(f'现金{cash_ratio:.1f}%过高，资金利用率偏低')
        c_pct = grade_capital.get('C', {}).get('pct', 0)
        if c_pct > 30:
            issues.append(f'{c_pct:.1f}%资金在C级低效仓位，关注改善')

        # 轮动建议
        rotation = self.generate_rotation_advice(positions, total_capital, dead)

        # 铁律v2 持有三理由检查
        hold_check = self.check_portfolio_hold_reasons(positions)

        # 排序：效率从低到高
        positions_sorted = sorted(positions, key=lambda x: x['score'])

        return {
            'total_capital': total_capital,
            'total_market_value': total_mv,
            'available_cash': available_cash,
            'cash_ratio': round(cash_ratio, 1),
            'position_ratio': round(total_mv / total_capital * 100, 1) if total_capital > 0 else 0,
            'weighted_score': round(weighted_score, 1),
            'grade_distribution': grade_capital,
            'dead_capital': {'market_value': dead_mv, 'pct': round(dead_pct, 1), 'positions': dead},
            'issues': issues,
            'rotation_advice': rotation,
            'hold_reasons_check': hold_check,
            'positions_sorted': positions_sorted,
        }

    def generate_rotation_advice(self, positions: List[Dict],
                                  total_capital: float,
                                  dead: List[Dict]) -> List[Dict]:
        """
        生成轮动建议

        优先释放D级死资本，其次是C级低效仓位
        """
        advice = []

        # D级死资本 → 强烈建议释放
        for p in sorted(dead, key=lambda x: x['market_value'], reverse=True):
            advice.append({
                'priority': 1,
                'action': '止损释放',
                'code': p['code'],
                'name': p['name'],
                'market_value': p['market_value'],
                'reason': f"效率分{p['score']}(D级) | 亏损{p['pnl_pct']:+.1f}% | "
                          f"年化回报{p['annualized_return_pct']:+.1f}% | "
                          f"占用{p['market_value']/total_capital*100:.1f}%资金",
                'target': '观望或等回调加仓高确信标的',
            })

        # C级低效 → 建议关注
        c_positions = [p for p in positions if p['grade'] == 'C' and not p['is_dead_capital']]
        for p in sorted(c_positions, key=lambda x: x['score']):
            advice.append({
                'priority': 2,
                'action': '关注/考虑减仓',
                'code': p['code'],
                'name': p['name'],
                'market_value': p['market_value'],
                'reason': f"效率分{p['score']}(C级) | 盈亏{p['pnl_pct']:+.1f}% | "
                          f"年化回报{p['annualized_return_pct']:+.1f}%",
                'target': '观察趋势，如进一步恶化则减仓',
            })

        # A级加仓建议
        a_positions = [p for p in positions if p['grade'] == 'A']
        if a_positions and dead:
            total_dead_mv = sum(p['market_value'] for p in dead)
            best_a = max(a_positions, key=lambda x: x['score'])
            advice.append({
                'priority': 3,
                'action': '释放资金可加仓',
                'code': best_a['code'],
                'name': best_a['name'],
                'market_value': total_dead_mv,
                'reason': f"效率最高({best_a['score']}分A级) | "
                          f"年化回报{best_a['annualized_return_pct']:+.1f}% | "
                          f"释放的{total_dead_mv:.0f}元可加仓此标的",
                'target': f'注意：单标的不超过总仓位15%({total_capital*0.15:.0f}元)',
            })

        return advice

    def format_position_report(self, result: Dict) -> str:
        """格式化单仓效率报告"""
        g = result['grade_info']
        lines = [
            f"💰 资金效率: {g['color']}{result['grade']}({result['score']}) | "
            f"年化回报: {result['annualized_return_pct']:+.1f}% | "
            f"持仓{result['holding_days']}天",
            f"   趋势{result['trend_score']} + 动能{result['momentum_score']} + "
            f"回报{result['return_score']} | {g['label']} — {g['action']}",
        ]
        if result['is_dead_capital']:
            lines.append(f"   ⚠️ 死资本：占用{result['market_value']:.0f}元，"
                        f"建议释放到更高效率标的")
        return '\n'.join(lines)

    def format_portfolio_report(self, result: Dict) -> str:
        """格式化组合效率周报"""
        lines = [
            "💰 资金效率周报",
            "─" * 50,
            f"总资金: {result['total_capital']:,.0f} | "
            f"已投入: {result['total_market_value']:,.0f} ({result['position_ratio']:.1f}%) | "
            f"现金: {result['available_cash']:,.0f} ({result['cash_ratio']:.1f}%)",
            f"组合加权效率分: {result['weighted_score']:.1f}",
            "",
            "📊 效率分布:",
        ]

        for grade in ['A', 'B', 'C', 'D']:
            gd = result['grade_distribution'].get(grade, {'market_value': 0, 'pct': 0, 'positions': []})
            gi = _grade_info(grade)
            names = '/'.join(p['name'] for p in gd['positions']) or '—'
            suffix = ' ← 最赚钱' if grade == 'A' else (' ← 死资本' if grade == 'D' else '')
            lines.append(
                f"  {gi['color']}{grade}级({gi['min']}+): {names} ── "
                f"{gd['market_value']:,.0f}元 ({gd['pct']:.1f}%){suffix}"
            )

        if result['issues']:
            lines.append("")
            lines.append("⚠️ 效率问题:")
            for issue in result['issues']:
                lines.append(f"  - {issue}")

        # 持有三理由检查结果
        hold_check = result.get('hold_reasons_check', {})
        if hold_check.get('failed', 0) > 0:
            lines.append("")
            lines.append(f"📋 铁律v2 持有理由检查: {hold_check['passed']}/{hold_check['total']}通过")
            for item in hold_check.get('exit_list', []):
                lines.append(f"  🔴 {item['name']}({item['code']}): {item['verdict']}")
                for d in item.get('details', []):
                    lines.append(f"     {d}")

        if result['rotation_advice']:
            lines.append("")
            lines.append("🎯 轮动建议:")
            for i, r in enumerate(result['rotation_advice'], 1):
                lines.append(
                    f"  优先级{r['priority']}: {r['name']}({r['code']}) → "
                    f"{r['action']} | {r['reason']}"
                )
                if r.get('target'):
                    lines.append(f"    → {r['target']}")

        # 铁律v2提醒
        lines.append("")
        lines.append("📌 铁律v2提醒: 每只持仓须满足 趋势向上/深利保护/即将催化 之一")

        return '\n'.join(lines)
