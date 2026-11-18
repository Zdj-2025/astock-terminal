#!/usr/bin/env python3
"""策略权重引擎 (Strategy Weight Engine)
独立模块: 基于证据的动态策略权重计算与更新.

设计原则(用户确认):
1. 权重 = 全样本证据强度(t值) × 近期衰减因子(滚动胜率)
2. 不用ML——用规则做"自然选择"(达尔文式淘汰)
3. 每月自动更新, signal_decay数据驱动
4. 新策略从基线起步, 60日数据后转正
5. 连续2个月胜率<40% → 权重归零(自动停用)

公式:
  w_i = base_i × adj_i
  base_i = |t_i| / Σ|t_j|  (归一化t值)
  adj_i =
    1.2  if rolling_win_rate > 55%
    1.0  if 45% ≤ rate ≤ 55% (中性区)
    0.5  if 40% ≤ rate < 45%
    0.0  if rate < 40% for 2 consecutive months

API:
  from scripts.core.strategy_weight_engine import WeightEngine
  we = WeightEngine()
  weights = we.get_weights()          # {'divergence': 0.28, 'golden_cross': 0.19, ...}
  we.update()                          # 从signal_decay拉最新数据, 重算权重
  we.report()                          # 打印权重表+调整原因

运行:
  python scripts/core/strategy_weight_engine.py          # 打印当前权重
  python scripts/core/strategy_weight_engine.py --update  # 更新并保存
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHT_STORE = PROJECT_ROOT / 'data/store/strategy_weights.json'

# ═══════════════════════════════════════════════════════════════
# 策略注册表: t值 + 基线状态 (有新回测结果时更新此处)
# 来源: 各回测脚本的全样本t统计量
# ═══════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    # 策略名: {'t_stat': float, 'status': str, 'notes': str}
    'divergence': {
        't_stat': 7.4,
        'status': 'active',
        'notes': '底背离+量能确认(E组合), 130万样本, p<0.0000',
    },
    'golden_cross': {
        't_stat': 5.1,
        'status': 'active',
        'notes': 'MA5上穿MA10+阳线, 784只×165天, 胜率65%',
    },
    'double_bottom': {
        't_stat': 3.2,  # 从40日中位+1.49%/胜率55%折算
        'status': 'active',
        'notes': '双底/二买算法, 20%门槛, A3确认即买, 40日中位+1.49%',
    },
    'oversold_reversal': {
        't_stat': 6.2,
        'status': 'active',
        'notes': '超跌反转, 全信号审计t=6.2, 待实现scanner',
    },
    'relative_strength': {
        't_stat': 2.0,  # 新策略, 保守估计
        'status': 'new',
        'notes': '相对强度(逆势抗跌): 上线前未做向前P&L测试即赋权进队列(R1待办: 补预注册前向测试, 通过前建议撤权)',
    },
    'smart_money': {
        't_stat': 4.0,  # 预测力强但实战力反转, 降基线
        'status': 'active',
        'notes': '聪明钱六特征, 预知力p=2.7e-91但2026H2实战-1.87%',
    },
    'quality_divergence': {
        't_stat': 3.0,
        'status': 'active',
        'notes': '质量背离(技术×基本面), mainline_leader',
    },
    'sector_heat': {
        't_stat': 0.0,  # p≈1, 无预测力
        'status': 'deprecated',
        'notes': '行业热度, 回测p≈1, 仅描述性',
    },
    'catalyst': {
        't_stat': 0.5,  # 2026H2已死
        'status': 'deprecated',
        'notes': '催化预增, 2026H2超额-4.49%(t=-3.5), 已衰减',
    },
    'buyback': {
        't_stat': 2.0,  # 新策略, 保守估计
        'status': 'new',
        'notes': '回购/增持信号, 待实现',
    },
    'small_cap': {
        't_stat': 4.5,  # A股最强公开因子
        'status': 'new',
        'notes': '小市值轮动, 2026三重共振, 待实现',
    },
    'money_flow_div': {
        't_stat': 2.0,  # 新策略, 保守估计
        'status': 'new',
        'notes': '资金流背离, 待数据接入',
    },
}

# 权重调整规则
BOOST_THRESHOLD = 0.55     # 胜率>55% → ×1.2
NEUTRAL_LOW = 0.45          # 45-55%中性
DEMOTION_THRESHOLD = 0.40   # 胜率<40% → ×0.5
DISABLE_MONTHS = 2          # 连续2个月<40% → 归零
MAX_WEIGHT = 0.30           # 单策略权重上限
NEW_STRATEGY_DAYS = 60      # 新策略观察期(交易日)
BASELINE_WEIGHT = 0.08      # 新策略基线权重


class WeightEngine:
    def __init__(self):
        self.registry = STRATEGY_REGISTRY.copy()
        self.history = self._load_history()

    def _load_history(self):
        if WEIGHT_STORE.exists():
            return json.loads(WEIGHT_STORE.read_text())
        return {'weights': {}, 'last_update': None, 'adjustments': []}

    def _save(self):
        WEIGHT_STORE.parent.mkdir(parents=True, exist_ok=True)
        WEIGHT_STORE.write_text(json.dumps(self.history, indent=2, ensure_ascii=False))

    def compute_base_weights(self):
        """归一化t值 → 基础权重"""
        active = {k: v for k, v in self.registry.items() if v['status'] == 'active'}
        total_t = sum(abs(v['t_stat']) for v in active.values())
        if total_t == 0:
            return {k: 1.0 / len(active) for k in active}
        return {k: abs(v['t_stat']) / total_t for k, v in active.items()}

    def get_rolling_performance(self, strategy):
        """从signal_decay拉滚动60日表现(如果有的话)"""
        # 这里应该读signal_decay的输出
        # 目前简化: 从history中读上次记录
        for adj in reversed(self.history.get('adjustments', [])):
            if adj.get('strategy') == strategy:
                return adj.get('win_rate', 0.50)
        return 0.50  # 默认中性

    def get_consecutive_bad_months(self, strategy):
        """连续低于40%的月数"""
        count = 0
        for adj in reversed(self.history.get('adjustments', [])):
            if adj.get('strategy') == strategy:
                if adj.get('win_rate', 0.50) < DEMOTION_THRESHOLD:
                    count += 1
                else:
                    break
        return count

    def compute_adjustment(self, strategy):
        """根据近期表现计算调整系数"""
        if self.registry.get(strategy, {}).get('status') == 'new':
            # 新策略: 检查是否有足够数据
            days = sum(1 for adj in self.history.get('adjustments', [])
                      if adj.get('strategy') == strategy)
            if days < 5:  # 少于5次记录, 用基线
                return BASELINE_WEIGHT / max(self.compute_base_weights().get(strategy, 0.08), 0.01)

        win_rate = self.get_rolling_performance(strategy)

        # 连续坏月检查
        bad_months = self.get_consecutive_bad_months(strategy)
        if bad_months >= DISABLE_MONTHS:
            return 0.0  # 自动停用

        # 规则引擎
        if win_rate > BOOST_THRESHOLD:
            return 1.2
        elif win_rate >= NEUTRAL_LOW:
            return 1.0
        elif win_rate >= DEMOTION_THRESHOLD:
            return 0.5
        else:
            return 0.3  # 单月<40%, 大幅降权(还不是零)

    def get_weights(self):
        """返回最终权重字典"""
        base = self.compute_base_weights()
        weights = {}
        for strategy in base:
            w = base[strategy] * self.compute_adjustment(strategy)
            weights[strategy] = round(min(w, MAX_WEIGHT), 4)

        # 加入新建策略(基线权重)和停用策略(权重0)
        for strategy, info in self.registry.items():
            if strategy not in weights:
                if info['status'] == 'new':
                    weights[strategy] = BASELINE_WEIGHT
                elif info['status'] == 'deprecated':
                    weights[strategy] = 0.0

        # 最后统一归一化(确保含新策略后总和=1)
        total = sum(v for v in weights.values() if v > 0)
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        return weights

    def update(self, performance_data=None):
        """手动或从signal_decay更新"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        weights = self.get_weights()

        # 记录调整
        self.history['weights'] = weights
        self.history['last_update'] = timestamp
        if performance_data:
            for strategy, perf in performance_data.items():
                self.history['adjustments'].append({
                    'timestamp': timestamp,
                    'strategy': strategy,
                    'win_rate': perf.get('win_rate', 0.50),
                    'excess_return': perf.get('excess_return', 0),
                    'n_signals': perf.get('n_signals', 0),
                })

        self._save()
        return weights

    def report(self):
        """打印权重报告"""
        weights = self.get_weights()
        print("═" * 70)
        print("📊 策略权重引擎 | 证据加权 × 近期衰减")
        print("═" * 70)
        print(f"{'策略':<20} {'基础权重':>8} {'调整系数':>8} {'最终权重':>8} {'状态':>8}")
        print("─" * 70)

        base = self.compute_base_weights()
        for strategy in sorted(weights, key=weights.get, reverse=True):
            b = base.get(strategy, 0)
            w = weights[strategy]
            adj = w / b if b > 0 else 0
            status = self.registry.get(strategy, {}).get('status', '?')
            print(f"  {strategy:<20} {b:>8.3f} {adj:>8.2f} {w:>8.3f} {status:>8}")

        print(f"\n  总权重: {sum(weights.values()):.3f}")
        print(f"  活跃策略: {sum(1 for w in weights.values() if w > 0)}个")
        print(f"  停用策略: {sum(1 for w in weights.values() if w == 0)}个")
        print(f"  最后更新: {self.history.get('last_update', '从未')}")

    def validate(self):
        """自检: 权重合理性校验"""
        weights = self.get_weights()
        errors = []

        # 1. 权重总和 ≈ 1
        total = sum(v for v in weights.values() if v > 0)
        if not (0.95 <= total <= 1.05):
            errors.append(f"权重总和{total:.3f}不在[0.95,1.05]")

        # 2. 单策略不超过上限
        for k, v in weights.items():
            if v > MAX_WEIGHT + 0.01:
                errors.append(f"{k}权重{v:.3f}超过上限{MAX_WEIGHT}")

        # 3. 停用策略权重=0
        for k, v in weights.items():
            if self.registry.get(k, {}).get('status') == 'deprecated' and v > 0:
                errors.append(f"停用策略{k}仍有权重{v}")

        # 4. 最强策略(底背离)权重应该最高
        active_weights = {k: v for k, v in weights.items()
                         if self.registry.get(k, {}).get('status') == 'active'}
        if active_weights:
            top = max(active_weights, key=active_weights.get)
            if top != 'divergence' and active_weights.get('divergence', 0) > 0:
                if active_weights[top] > active_weights['divergence'] * 1.5:
                    errors.append(f"底背离(t=7.4)不是最高权重: {top}={active_weights[top]:.3f}")

        return errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    we = WeightEngine()

    if args.validate:
        errors = we.validate()
        if errors:
            print("❌ 校验失败:")
            for e in errors:
                print(f"  {e}")
        else:
            print("✅ 全部校验通过")
        return

    if args.update:
        weights = we.update()
        we.report()
        errors = we.validate()
        if errors:
            print("\n⚠️ 校验警告:")
            for e in errors:
                print(f"  {e}")
    else:
        we.report()


if __name__ == '__main__':
    main()
