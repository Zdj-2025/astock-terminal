#!/usr/bin/env python3
"""
回测绩效指标

Sharpe, Sortino, max drawdown, win rate, profit factor, IC, alpha vs buy-hold
"""
import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """总收益率"""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return equity.iloc[-1] / equity.iloc[0] - 1


def annualized_return(total_ret: float, trading_days: int) -> float:
    """年化收益率"""
    if trading_days <= 0:
        return 0.0
    return (1 + total_ret) ** (252 / trading_days) - 1


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Sharpe ratio (年化)"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free) ** (1/252) - 1
    excess = returns - daily_rf
    return excess.mean() / excess.std() * np.sqrt(252)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Sortino ratio (只惩罚下行波动)"""
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free) ** (1/252) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if excess.mean() > 0 else 0.0
    return excess.mean() / downside.std() * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（返回负数）"""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def calmar_ratio(total_ret: float, mdd: float, trading_days: int) -> float:
    """Calmar ratio = 年化收益 / 最大回撤"""
    if mdd == 0:
        return 0.0
    ann = annualized_return(total_ret, trading_days)
    return ann / abs(mdd)


def win_rate(returns: pd.Series) -> float:
    """胜率"""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def profit_factor(returns: pd.Series) -> float:
    """盈亏比 = 总盈利 / 总亏损"""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def avg_win_loss_ratio(returns: pd.Series) -> float:
    """平均盈利 / 平均亏损"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return wins.mean() / abs(losses.mean())


def information_coefficient(signals: pd.Series, forward_returns: pd.Series) -> float:
    """
    信息系数 (IC): 信号分数与后续收益的相关系数
    衡量信号的预测能力
    """
    valid = signals.notna() & forward_returns.notna()
    if valid.sum() < 10:
        return 0.0
    return signals[valid].corr(forward_returns[valid])


def ic_decay(signals: pd.Series, returns_df: pd.DataFrame, periods: list = None) -> pd.Series:
    """
    IC衰减：信号对不同期限收益的预测能力
    returns_df columns: 'fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d'
    """
    if periods is None:
        periods = [1, 3, 5, 10, 20]
    cols = [f'fwd_{p}d' for p in periods if f'fwd_{p}d' in returns_df.columns]
    result = {}
    for c in cols:
        valid = signals.notna() & returns_df[c].notna()
        if valid.sum() >= 10:
            result[c] = signals[valid].corr(returns_df.loc[valid, c])
        else:
            result[c] = 0.0
    return pd.Series(result)


def alpha_vs_buyhold(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Alpha = 策略年化收益 - 基准年化收益"""
    strat_ann = annualized_return(total_return(
        (1 + strategy_returns).cumprod()), len(strategy_returns))
    bench_ann = annualized_return(total_return(
        (1 + benchmark_returns).cumprod()), len(benchmark_returns))
    return strat_ann - bench_ann


def signal_accuracy_by_level(signal_level: pd.Series, forward_return: pd.Series) -> pd.DataFrame:
    """
    按信号级别统计胜率和平均收益
    signal_level: '做多', '偏多', '中性', '偏空', '做空'
    """
    levels = ['做多', '偏多', '中性', '偏空', '做空']
    rows = []
    for lvl in levels:
        mask = signal_level == lvl
        n = mask.sum()
        if n == 0:
            rows.append({'信号': lvl, '次数': 0, '胜率': 0, '平均收益': 0, '中位收益': 0})
            continue
        r = forward_return[mask]
        wr = (r > 0).mean()
        rows.append({
            '信号': lvl,
            '次数': int(n),
            '胜率': f'{wr:.1%}',
            '平均收益': f'{r.mean():.2%}',
            '中位收益': f'{r.median():.2%}',
        })
    return pd.DataFrame(rows)


def format_backtest_report(results: dict) -> str:
    """格式化回测报告"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"📊 回测报告: {results.get('symbol', '')} {results.get('name', '')}")
    lines.append(f"   期间: {results.get('start', '')} ~ {results.get('end', '')}")
    lines.append(f"   交易天数: {results.get('trading_days', 0)}")
    lines.append(f"{'='*70}")

    # 策略 vs 买入持有
    strat = results.get('strategy', {})
    bh = results.get('buy_hold', {})
    lines.append(f"\n{'─'*70}")
    lines.append(f"  策略绩效 vs 买入持有")
    lines.append(f"{'─'*70}")
    lines.append(f"  {'指标':<16} {'策略':>12} {'买入持有':>12} {'差异':>12}")
    lines.append(f"  {'─'*52}")

    metrics_pairs = [
        ('总收益', 'total_return', '.2%'),
        ('年化收益', 'annualized_return', '.2%'),
        ('最大回撤', 'max_drawdown', '.2%'),
        ('Sharpe', 'sharpe', '.3f'),
        ('胜率', 'win_rate', '.1%'),
        ('盈亏比', 'profit_factor', '.2f'),
    ]

    for label, key, fmt in metrics_pairs:
        sv = strat.get(key, 0)
        bv = bh.get(key, 0)
        if isinstance(sv, str):
            sv = float(sv.replace('%', '')) / 100 if '%' in str(sv) else float(sv)
        if isinstance(bv, str):
            bv = float(bv.replace('%', '')) / 100 if '%' in str(bv) else float(bv)
        diff = sv - bv
        lines.append(f"  {label:<16} {sv:>12{fmt}} {bv:>12{fmt}} {diff:>+12{fmt}}")

    # Alpha
    alpha = results.get('alpha', 0)
    lines.append(f"  {'Alpha':<16} {alpha:>+12.2%}")

    # IC
    ic = results.get('ic', 0)
    lines.append(f"  {'IC(信息系数)':<16} {ic:>+12.4f}")

    # 信号级别胜率
    if 'signal_accuracy' in results:
        lines.append(f"\n{'─'*70}")
        lines.append(f"  各信号级别表现")
        lines.append(f"{'─'*70}")
        for row in results['signal_accuracy']:
            lines.append(
                f"  {row['信号']:<6} | 次数:{row['次数']:>5} | "
                f"胜率:{row['胜率']:>6} | 平均收益:{row['平均收益']:>8} | "
                f"中位收益:{row['中位收益']:>8}"
            )

    # 单维度贡献
    if 'dimension_analysis' in results:
        lines.append(f"\n{'─'*70}")
        lines.append(f"  单维度预测力 (IC)")
        lines.append(f"{'─'*70}")
        for dim, data in results['dimension_analysis'].items():
            ic_val = data.get('ic', 0)
            wr = data.get('win_rate', 0)
            lines.append(f"  {dim:<12} | IC: {ic:>+.4f} | 胜率: {wr:.1%}")

    # OBV对比
    if 'obv_comparison' in results:
        obv = results['obv_comparison']
        lines.append(f"\n{'─'*70}")
        lines.append(f"  OBV贡献分析")
        lines.append(f"{'─'*70}")
        lines.append(f"  含OBV IC: {obv.get('with_obv_ic', 0):+.4f} | "
                      f"胜率: {obv.get('with_obv_wr', 0):.1%}")
        lines.append(f"  去OBV IC: {obv.get('without_obv_ic', 0):+.4f} | "
                      f"胜率: {obv.get('without_obv_wr', 0):.1%}")
        diff = obv.get('without_obv_ic', 0) - obv.get('with_obv_ic', 0)
        verdict = 'OBV有害，建议剔除' if diff > 0.01 else \
                  'OBV有益，建议保留' if diff < -0.01 else 'OBV贡献可忽略'
        lines.append(f"  结论: {verdict}")

    return '\n'.join(lines)
