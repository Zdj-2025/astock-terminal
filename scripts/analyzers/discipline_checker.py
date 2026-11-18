#!/usr/bin/env python3
"""
纪律监督检查器

客观检查持仓是否遵守交易纪律，不迎合用户。
检查7项规则，输出红/黄/蓝三色警告。

用法:
    from scripts.analyzers.discipline_checker import DisciplineChecker
    checker = DisciplineChecker()
    report = checker.check(holding, price, analysis_result)
"""


class DisciplineChecker:
    """纪律监督检查器 — 每次分析自动运行。"""

    # 纪律阈值（来自风控规则+克罗策略）
    STOP_LOSS_PCT = -8.0          # 止损线：-8%
    TAKE_PROFIT_PCT = 15.0        # 止盈线：+15%
    TRAILING_STOP_ACTIVATE = 10.0 # 跟踪止损激活：盈利10%
    MAX_SINGLE_POS_PCT = 15.0     # 单仓上限：15%
    MAX_TOTAL_POS_PCT = 80.0      # 总仓位上限：80%
    MAX_LOSS_PER_TRADE_CAPITAL = 10.0  # 单笔亏损≤本金10%
    MAX_LOSS_PER_TRADE_TOTAL = 2.5     # 单笔亏损≤总资金2.5%
    COOL_DOWN_AFTER_LOSSES = 3    # 连续亏损N次后冷却

    def check(self, holding: dict, price: float, analysis_result: dict = None,
              all_holdings: list = None, total_capital: float = 0) -> dict:
        """
        检查纪律执行情况。

        Args:
            holding: 单只持仓 {code, name, cost, shares, stop_loss, take_profit, buy_date}
            price: 当前价格
            analysis_result: ComprehensiveAnalyzer.analyze() 的结果（用于趋势判断）
            all_holdings: 全部持仓列表（用于总仓位计算）
            total_capital: 总资金

        Returns:
            {violations: [...], warnings: [...], info: [...], score: 0-100, summary: str}
        """
        violations = []  # 红色 — 必须立即行动
        warnings = []    # 黄色 — 需要关注
        info = []        # 蓝色 — 建议性

        cost = holding.get('cost', 0)
        shares = holding.get('shares', 0)
        stop_loss = holding.get('stop_loss', 0)
        code = holding.get('code', '')
        name = holding.get('name', '')

        if cost <= 0 or shares <= 0 or price <= 0:
            return {'violations': [], 'warnings': [], 'info': [],
                    'score': 100, 'summary': '数据不完整，无法检查'}

        pnl_pct = (price / cost - 1) * 100
        market_value = price * shares
        loss_amount = (cost - price) * shares if price < cost else 0

        # ── 检查1: 止损是否到位 ──
        # 用配置中的止损价，如果没配则用-8%规则
        sl_price = stop_loss if stop_loss > 0 else cost * (1 + self.STOP_LOSS_PCT / 100)
        if price <= sl_price:
            violations.append({
                'rule': '止损到位',
                'detail': f'现价{price:.2f} ≤ 止损价{sl_price:.2f}（亏{pnl_pct:+.1f}%）',
                'action': '立即执行止损，不犹豫',
            })

        # ── 检查2: 单笔亏损风险 ──
        if total_capital > 0 and loss_amount > 0:
            loss_pct_capital = loss_amount / total_capital * 100
            loss_pct_position = abs(pnl_pct)
            if loss_pct_capital > self.MAX_LOSS_PER_TRADE_TOTAL:
                violations.append({
                    'rule': '单笔亏损超总资金2.5%',
                    'detail': f'亏损{loss_amount:,.0f}元 = 总资金{loss_pct_capital:.1f}%（阈值2.5%）',
                    'action': '已超过克罗风险上限，考虑减仓',
                })
            if loss_pct_position > self.MAX_LOSS_PER_TRADE_CAPITAL:
                violations.append({
                    'rule': '单笔亏损超本金10%',
                    'detail': f'亏损{loss_pct_position:.1f}%（阈值10%）',
                    'action': '超过单笔本金10%红线，必须止损',
                })

        # ── 检查3: 亏损仓位是否在加仓 ──
        # 注：此检查需要外部提供交易记录，此处仅基于持仓状态提示
        if pnl_pct < -3 and analysis_result:
            sig4 = analysis_result.get('signal_4d', {})
            if sig4 and sig4.get('composite', 0) < 0.2:
                warnings.append({
                    'rule': '亏损仓位+趋势不明确',
                    'detail': f'当前亏{pnl_pct:+.1f}%，四维信号={sig4.get("verdict", "?")}',
                    'action': '禁止加仓。如需补仓须等四维信号≥+0.5',
                })

        # ── 检查4: 单仓是否超限 ──
        if total_capital > 0:
            pos_pct = market_value / total_capital * 100
            if pos_pct > self.MAX_SINGLE_POS_PCT:
                warnings.append({
                    'rule': '单仓超限',
                    'detail': f'{name}仓位{pos_pct:.1f}%（上限{self.MAX_SINGLE_POS_PCT}%）',
                    'action': '考虑减仓至限制以内',
                })

        # ── 检查5: 总仓位是否超限 ──
        if all_holdings and total_capital > 0:
            total_mv = sum(h.get('price', 0) * h.get('shares', 0) for h in all_holdings)
            total_pos_pct = total_mv / total_capital * 100
            if total_pos_pct > self.MAX_TOTAL_POS_PCT:
                warnings.append({
                    'rule': '总仓位超限',
                    'detail': f'总仓位{total_pos_pct:.1f}%（上限{self.MAX_TOTAL_POS_PCT}%）',
                    'action': '考虑减仓释放资金',
                })

        # ── 检查6: 跟踪止损是否应启用 ──
        if pnl_pct >= self.TRAILING_STOP_ACTIVATE:
            trailing_sl = cost  # 盈利>10%，止损至少提到成本价
            if price > cost and stop_loss > 0 and stop_loss < cost:
                info.append({
                    'rule': '跟踪止损建议',
                    'detail': f'已盈利{pnl_pct:+.1f}%（>{self.TRAILING_STOP_ACTIVATE}%）',
                    'action': f'建议将止损从{stop_loss:.2f}提至{cost:.2f}（成本价保本）',
                })
            elif stop_loss <= 0 or stop_loss < cost:
                info.append({
                    'rule': '跟踪止损建议',
                    'detail': f'已盈利{pnl_pct:+.1f}%，未设跟踪止损',
                    'action': f'建议设止损={cost:.2f}保本，或{price*0.97:.2f}锁利',
                })

        # ── 检查7: 趋势不明确时是否持有/加仓 ──
        if analysis_result:
            sig4 = analysis_result.get('signal_4d', {})
            if sig4:
                composite = sig4.get('composite', 0)
                if abs(composite) < 0.2 and pnl_pct < -5:
                    warnings.append({
                        'rule': '趋势不明+深度亏损',
                        'detail': f'四维信号={sig4.get("verdict", "?")}（{composite:+.3f}），亏{pnl_pct:+.1f}%',
                        'action': '趋势不明朗时持有亏损仓风险大，考虑减仓',
                    })

        # ── 计算纪律分数 ──
        score = 100
        score -= len(violations) * 30  # 红色每条扣30分
        score -= len(warnings) * 10    # 黄色每条扣10分
        score = max(0, score)

        # ── 总结 ──
        if violations:
            summary = f'🚨 {len(violations)}项违规！必须立即处理'
        elif warnings:
            summary = f'⚠️ {len(warnings)}项警告，需要关注'
        elif info:
            summary = f'ℹ️ {len(info)}条建议'
        else:
            summary = '✅ 纪律检查通过'

        return {
            'violations': violations,
            'warnings': warnings,
            'info': info,
            'score': score,
            'summary': summary,
        }

    @staticmethod
    def format_report(check_result: dict) -> str:
        """格式化纪律检查报告。"""
        lines = []
        lines.append(f"\n{'='*65}")
        lines.append(f"🔍 纪律监督报告: {check_result.get('summary', '')}")
        lines.append(f"   纪律分数: {check_result.get('score', 0)}/100")
        lines.append(f"{'='*65}")

        for v in check_result.get('violations', []):
            lines.append(f"  🚨 [{v['rule']}] {v['detail']}")
            lines.append(f"     → {v['action']}")

        for w in check_result.get('warnings', []):
            lines.append(f"  ⚠️ [{w['rule']}] {w['detail']}")
            lines.append(f"     → {w['action']}")

        for i in check_result.get('info', []):
            lines.append(f"  ℹ️ [{i['rule']}] {i['detail']}")
            lines.append(f"     → {i['action']}")

        if not check_result.get('violations') and not check_result.get('warnings'):
            lines.append(f"  ✅ 无违规无警告")

        return '\n'.join(lines)
