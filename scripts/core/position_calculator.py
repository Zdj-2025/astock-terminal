#!/usr/bin/env python3
"""
仓位计算模块
基于凯利公式和其他方法计算最优仓位
"""
import numpy as np
from typing import Dict, Optional
import json


class PositionCalculator:
    """
    仓位计算器

    提供多种仓位计算方法：
    - 凯利公式
    - 固定比例
    - 风险平价

    示例:
        calc = PositionCalculator()
        position = calc.kelly_criterion(win_rate=0.6, avg_win=0.15, avg_loss=0.08)
    """

    def __init__(self, config_file: str = 'config/risk_rules.json'):
        """
        初始化仓位计算器

        参数:
            config_file: 风险配置文件
        """
        self.config = self._load_config(config_file)

        # 默认参数
        self.max_single_position = 0.15  # 单股最大15%
        self.max_total_position = 0.80   # 总仓位最大80%
        self.kelly_fraction = 0.25        # 使用1/4凯利

        # 从配置文件加载
        if self.config:
            position_limits = self.config.get('position_limits', {})
            self.max_single_position = position_limits.get('max_single_position', 0.15)
            self.max_total_position = position_limits.get('max_total_position', 0.80)

            kelly_config = self.config.get('kelly_criterion', {})
            self.kelly_fraction = kelly_config.get('fraction', 0.25)

        print("✅ PositionCalculator初始化成功")
        print(f"   - 单股最大仓位: {self.max_single_position:.1%}")
        print(f"   - 总仓位上限: {self.max_total_position:.1%}")
        print(f"   - 凯利系数: {self.kelly_fraction}")

    def _load_config(self, config_file: str) -> Optional[Dict]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  未加载配置文件: {e}")
            return None

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        total_capital: Optional[float] = None
    ) -> Dict:
        """
        凯利公式计算最优仓位

        凯利公式: f* = (p × b - q) / b
        其中:
        - f* = 最优仓位比例
        - p = 胜率
        - b = 赔率（平均盈利/平均亏损）
        - q = 1 - p（败率）

        参数:
            win_rate: 胜率（0-1）
            avg_win: 平均盈利（小数，如0.15表示15%）
            avg_loss: 平均亏损（小数，如0.08表示8%）
            total_capital: 总资金（可选，用于计算具体金额）

        返回:
            {
                'kelly_fraction': 凯利仓位（0-1）,
                'actual_fraction': 实际建议仓位（0-1）,
                'position_amount': 建议仓位金额（如果提供总资金）,
                'explanation': 解释说明
            }
        """
        try:
            # 参数验证
            if not (0 < win_rate < 1):
                raise ValueError(f"胜率应在0-1之间，当前: {win_rate}")
            if avg_win <= 0:
                raise ValueError(f"平均盈利应大于0，当前: {avg_win}")
            if avg_loss <= 0:
                raise ValueError(f"平均亏损应大于0，当前: {avg_loss}")

            # 计算赔率
            odds = avg_win / avg_loss

            # 凯利公式
            q = 1 - win_rate
            kelly = (win_rate * odds - q) / odds

            # 确保在合理范围内
            kelly = max(0, min(kelly, 1))  # 限制在0-1之间

            # 应用凯利系数（如1/4凯利）
            actual_fraction = kelly * self.kelly_fraction

            # 应用最大仓位限制
            actual_fraction = min(actual_fraction, self.max_single_position)

            # 计算金额
            position_amount = None
            if total_capital:
                position_amount = total_capital * actual_fraction

            # 生成解释
            explanation = self._generate_kelly_explanation(
                win_rate, odds, kelly, actual_fraction
            )

            result = {
                'kelly_fraction': kelly,
                'actual_fraction': actual_fraction,
                'position_amount': position_amount,
                'explanation': explanation,
                'params': {
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'odds': odds,
                    'kelly_coefficient': self.kelly_fraction
                }
            }

            return result

        except Exception as e:
            print(f"❌ 凯利公式计算失败: {e}")
            return {
                'kelly_fraction': 0,
                'actual_fraction': 0,
                'position_amount': 0,
                'explanation': f'计算失败: {e}',
                'error': str(e)
            }

    def _generate_kelly_explanation(
        self,
        win_rate: float,
        odds: float,
        kelly: float,
        actual: float
    ) -> str:
        """生成凯利公式解释"""
        explanation = f"""
凯利公式计算结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入参数：
  • 胜率: {win_rate:.1%}
  • 赔率: {odds:.2f} (平均盈利/平均亏损)
  • 凯利系数: {self.kelly_fraction}

计算结果：
  • 凯利仓位: {kelly:.1%}
  • 实际建议: {actual:.1%} (应用{self.kelly_fraction}凯利系数)

风险等级：
"""

        if actual > 0.10:
            explanation += "  ⚠️  高风险 - 建议仓位较大\n"
        elif actual > 0.05:
            explanation += "  ✅ 中等风险 - 适中仓位\n"
        else:
            explanation += "  🛡️  低风险 - 保守仓位\n"

        explanation += f"""
注意事项：
  • 凯利公式假设收益符合正态分布
  • 实际市场存在肥尾效应，建议保守
  • 已应用最大仓位限制: {self.max_single_position:.1%}
"""
        return explanation

    def fixed_fraction(
        self,
        fraction: float,
        total_capital: float,
        risk_level: str = 'medium'
    ) -> Dict:
        """
        固定比例仓位

        参数:
            fraction: 仓位比例（0-1）
            total_capital: 总资金
            risk_level: 风险等级（conservative/medium/aggressive）

        返回:
            {
                'fraction': 实际仓位比例,
                'amount': 仓位金额
            }
        """
        try:
            # 根据风险等级调整
            risk_multipliers = {
                'conservative': 0.7,
                'medium': 1.0,
                'aggressive': 1.3
            }

            multiplier = risk_multipliers.get(risk_level, 1.0)
            adjusted_fraction = fraction * multiplier

            # 应用最大仓位限制
            adjusted_fraction = min(adjusted_fraction, self.max_single_position)

            amount = total_capital * adjusted_fraction

            return {
                'fraction': adjusted_fraction,
                'amount': amount,
                'risk_level': risk_level,
                'multiplier': multiplier
            }

        except Exception as e:
            print(f"❌ 固定比例计算失败: {e}")
            return {'fraction': 0, 'amount': 0}

    def risk_parity(
        self,
        volatilities: Dict[str, float],
        total_capital: float
    ) -> Dict[str, float]:
        """
        风险平价仓位分配

        根据资产的波动率分配仓位，使每个资产的风险贡献相等

        参数:
            volatilities: 各资产的波动率 {symbol: volatility}
            total_capital: 总资金

        返回:
            各资产的仓位分配 {symbol: position_amount}
        """
        try:
            if not volatilities:
                return {}

            # 计算风险倒数
            inv_vols = {sym: 1/vol for sym, vol in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())

            # 分配权重
            weights = {sym: inv_vol/total_inv_vol for sym, inv_vol in inv_vols.items()}

            # 计算仓位金额
            positions = {sym: total_capital * weight for sym, weight in weights.items()}

            return positions

        except Exception as e:
            print(f"❌ 风险平价计算失败: {e}")
            return {}

    def suggest_position(
        self,
        signal_strength: float,
        total_capital: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> Dict:
        """
        综合建议仓位

        根据信号强度和凯利公式综合建议仓位

        参数:
            signal_strength: 信号强度（0-1）
            total_capital: 总资金
            win_rate: 胜率（可选）
            avg_win: 平均盈利（可选）
            avg_loss: 平均亏损（可选）

        返回:
            {
                'suggested_fraction': 建议仓位比例,
                'suggested_amount': 建议仓位金额,
                'method': 使用的方法,
                'explanation': 解释说明
            }
        """
        try:
            # 如果提供了历史数据，使用凯利公式
            if win_rate and avg_win and avg_loss:
                kelly_result = self.kelly_criterion(
                    win_rate, avg_win, avg_loss, total_capital
                )
                kelly_fraction = kelly_result['actual_fraction']

                # 根据信号强度调整
                adjusted_fraction = kelly_fraction * signal_strength
                method = 'kelly_adjusted'

            else:
                # 使用固定比例（基于信号强度）
                base_fraction = self.max_single_position * signal_strength * 0.5
                adjusted_fraction = base_fraction
                method = 'signal_based'

            # 应用最大仓位限制
            adjusted_fraction = min(adjusted_fraction, self.max_single_position)

            amount = total_capital * adjusted_fraction

            # 生成解释
            explanation = f"""
仓位建议：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
信号强度: {signal_strength:.2f}
计算方法: {method}
建议仓位: {adjusted_fraction:.1%} (¥{amount:,.2f})

风险提示：
  • 当前建议仓位: {adjusted_fraction:.1%}
  • 单股最大限制: {self.max_single_position:.1%}
  • 总仓位上限: {self.max_total_position:.1%}
"""

            return {
                'suggested_fraction': adjusted_fraction,
                'suggested_amount': amount,
                'method': method,
                'explanation': explanation
            }

        except Exception as e:
            print(f"❌ 仓位建议失败: {e}")
            return {
                'suggested_fraction': 0,
                'suggested_amount': 0,
                'method': 'error',
                'explanation': f'计算失败: {e}'
            }


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 PositionCalculator 模块")
    print("=" * 60)

    calc = PositionCalculator()

    # 测试1: 凯利公式
    print("\n【测试1】凯利公式计算")
    print("-" * 60)
    result = calc.kelly_criterion(
        win_rate=0.6,
        avg_win=0.15,
        avg_loss=0.08,
        total_capital=500000
    )
    print(result['explanation'])
    print(f"建议仓位金额: ¥{result['position_amount']:,.2f}")

    # 测试2: 固定比例
    print("\n【测试2】固定比例仓位")
    print("-" * 60)
    result = calc.fixed_fraction(
        fraction=0.10,
        total_capital=500000,
        risk_level='medium'
    )
    print(f"仓位比例: {result['fraction']:.1%}")
    print(f"仓位金额: ¥{result['amount']:,.2f}")
    print(f"风险等级: {result['risk_level']}")

    # 测试3: 综合建议
    print("\n【测试3】综合仓位建议")
    print("-" * 60)
    result = calc.suggest_position(
        signal_strength=0.75,
        total_capital=500000,
        win_rate=0.55,
        avg_win=0.12,
        avg_loss=0.08
    )
    print(result['explanation'])

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
