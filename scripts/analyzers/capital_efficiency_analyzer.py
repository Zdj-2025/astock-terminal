#!/usr/bin/env python3
"""资金使用效率分析器 — FIFO配对买→卖,算每笔效率+凯利公式+最优参数回测。
2026-08-12建。回答:每次买多少/持多久/止损止盈/速度最优。

指标:
  daily_pnl   日均盈利(元/天)=盈亏/持有天数
  pct         收益率%=(卖价/买价-1)×100
  capital_days 资金占用(元·天)=买价×股数×天数
  efficiency  年化效率%=盈亏/资金占用×365×100

用法: python scripts/analyzers/capital_efficiency_analyzer.py"""
import pandas as pd, numpy as np
from datetime import datetime
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')


def load_trades():
    """读trade_log(容错坏行),标准化买卖方向"""
    df = pd.read_csv('config/trade_log.csv', on_bad_lines='skip')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'code']).copy()
    df['code'] = df['code'].astype(str).str.strip()
    df = df[df['code'].str.len() >= 5].copy()  # 过滤空/短code(取现等)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').abs()  # 卖出负→正
    df = df.dropna(subset=['price', 'shares'])
    df['is_buy'] = df['action'].astype(str).str.contains('买', na=False) & ~df['action'].astype(str).str.contains('取', na=False)
    df['is_sell'] = df['action'].astype(str).str.contains('卖', na=False)
    df = df[df['is_buy'] | df['is_sell']].copy()
    return df.sort_values('date').reset_index(drop=True)


def fifo_match(trades):
    """FIFO配对买→卖,返回每笔配对结果"""
    results = []
    for code, g in trades.groupby('code'):
        name = str(g['name'].iloc[0]) if pd.notna(g['name'].iloc[0]) else code
        buy_queue = []
        for _, r in g.iterrows():
            if r['is_buy']:
                buy_queue.append({'date': r['date'], 'price': r['price'], 'shares': r['shares']})
            elif r['is_sell']:
                sell_shares = r['shares']
                sell_price = r['price']
                sell_date = r['date']
                while sell_shares > 0.5 and buy_queue:
                    buy = buy_queue[0]
                    matched = min(sell_shares, buy['shares'])
                    pnl = (sell_price - buy['price']) * matched
                    days = max(1, (sell_date - buy['date']).days)
                    cap_days = buy['price'] * matched * days
                    results.append({
                        'code': code, 'name': name,
                        'buy_date': buy['date'], 'sell_date': sell_date,
                        'shares': matched, 'buy_price': buy['price'], 'sell_price': sell_price,
                        'pnl': pnl, 'days': days,
                        'daily_pnl': pnl / days,
                        'pct': (sell_price / buy['price'] - 1) * 100,
                        'efficiency': pnl / cap_days * 365 * 100 if cap_days > 0 else 0,
                    })
                    buy['shares'] -= matched
                    sell_shares -= matched
                    if buy['shares'] <= 0.5:
                        buy_queue.pop(0)
    return pd.DataFrame(results)


def main():
    trades = load_trades()
    m = fifo_match(trades)
    if len(m) == 0:
        print('无配对记录'); return
    m = m.sort_values('daily_pnl', ascending=False)
    print(f"=== 资金效率分析(FIFO配对) | {len(m)}笔配对 ===\n")
    print(f"{'名称':<10}{'买→卖':>22}{'天':>4}{'盈亏':>8}{'日均':>8}{'收益%':>7}{'年化%':>7}")
    print('-' * 75)
    for _, r in m.iterrows():
        period = f"{str(r['buy_date'].date())[5:]}→{str(r['sell_date'].date())[5:]}"
        print(f"{r['name']:<10}{period:>22}{r['days']:>3}d{r['pnl']:>+8.0f}{r['daily_pnl']:>+8.0f}{r['pct']:>+6.1f}%{r['efficiency']:>+7.0f}")
    # 汇总
    wins = m[m['pnl'] > 0]
    losses = m[m['pnl'] < 0]
    print(f"\n=== 汇总 ===")
    print(f"总笔数: {len(m)} | 赚{len(wins)} 亏{len(losses)} | 总盈亏: {m['pnl'].sum():+.0f}")
    print(f"平均持有: {m['days'].mean():.0f}天 | 平均日均: {m['daily_pnl'].mean():+.0f}/天")
    if len(wins) > 0 and len(losses) > 0:
        wr = len(wins) / len(m) * 100
        aw, al = wins['pnl'].mean(), abs(losses['pnl'].mean())
        ratio = aw / al if al > 0 else 0
        print(f"胜率: {wr:.0f}% | 盈亏比: {ratio:.2f} (均盈{aw:.0f}/均亏{al:.0f})")
        p = wr / 100
        kelly = max(0, p - (1 - p) / ratio) if ratio > 0 else 0
        print(f"凯利公式: f*={kelly*100:.1f}% (半凯利{kelly*50:.1f}%) | 建议单股风险{kelly*50*400000/100:.0f}元")
    # 持有期效率对比
    print(f"\n=== 持有期效率对比 ===")
    for d in [3, 7, 10, 14, 30]:
        sub = m[m['days'] <= d]
        if len(sub) > 0:
            print(f"  ≤{d}天: {len(sub)}笔 平均日均{sub['daily_pnl'].mean():+.0f}/天 胜率{len(sub[sub['pnl']>0])/len(sub)*100:.0f}%")
    # 止损效率
    print(f"\n=== 止损效率对比 ===")
    for s in [-5, -8, -10, -15]:
        sub = m[m['pct'] <= s]
        if len(sub) > 0:
            print(f"  收益≤{s}%: {len(sub)}笔 平均亏{sub['pnl'].mean():+.0f} 持有{sub['days'].mean():.0f}天")


if __name__ == '__main__':
    main()
