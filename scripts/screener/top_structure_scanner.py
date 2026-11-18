#!/usr/bin/env python3
"""顶部结构scanner — 2026-09-22立 (用户需求: 持仓卖出提醒信号)

定位: 与bottom_structure_scanner互为镜像。**不是筛选器,是持仓验证器**——
选出的股票查底部结构加信心; 拿着的股票查顶部结构收警示。

结构定义:
  前置  : 分型点须是60日高点3%以内(显著高点, 不是反弹途中的小丘)
  阶段1 : 顶分型 — pivot高点(左2右2根K线都更低)
  🟡警惕 : 高位顶分型形成, 未破位
  🔴确认 : 收盘跌破分型区间低点 → 派发确认 = 卖出警示
  新鲜度: 距分型高点回落<12%(跌更多=信号过期, 该早走的已经走完)
  加分  : 双顶标注(60日内两pivot高点差<3%且后者更低)

⚠️ 纪律: 本信号是"提醒"不是"指令"——卖出仍走3S+止损线+移动止损五维
   (卖出五维检查铁律: 仓位<10%不因单因子减/日线红柱加速不卖/均线多头不卖)

用法:
  PYTHONPATH=. python scripts/screener/top_structure_scanner.py --portfolio  # 持仓警示
  PYTHONPATH=. python scripts/screener/top_structure_scanner.py --check 600522,000100
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
from scripts.screener.bottom_structure_scanner import load_stock

STORE = PROJECT_ROOT / 'data/store/kline'


def pivot_highs(high, left=2, right=2):
    n = len(high)
    pivots = []
    for i in range(left, n - right):
        w = high[i - left:i + right + 1]
        if high[i] == w.max() and (w < high[i]).sum() >= (left + right):
            pivots.append(i)
    return pivots


def detect_top(df):
    """返回 None 或 {'stage','drawdown','age','dbl','pivot_high','fractal_low'}"""
    if len(df) < 140:
        return None
    c, h, l = df['close'], df['high'], df['low']
    pivots = pivot_highs(h.values)
    if not pivots:
        return None
    p = pivots[-1]
    # 显著高点: 60日高3%以内
    hi60 = h.iloc[max(0, p - 60):p + 1].max()
    if h.iloc[p] < hi60 * 0.97:
        return None
    age = len(df) - 1 - p
    if age < 2 or age > 15:
        return None
    drawdown = (c.iloc[-1] / h.iloc[p] - 1) * 100
    if drawdown < -12:                       # 过期: 已经跌完了
        return None
    if drawdown > 1:                         # 价格超过分型高点=分型已被新高否定(2026-09-22国科天成校准)
        return None
    # 破位: 收盘跌破分型区间低点
    fractal_low = l.iloc[max(0, p - 2):p + 3].min()
    broken = c.iloc[-1] < fractal_low
    # 双顶
    dbl = ''
    for q in pivots[-4:-1]:
        if p - q >= 10 and abs(h.iloc[p] / h.iloc[q] - 1) < 0.03 and h.iloc[p] <= h.iloc[q]:
            dbl = '双顶⚠️'
            break
    stage = '🔴顶部确认(卖出警示)' if broken else '🟡高位顶分型(警惕)'
    return {'stage': stage, 'drawdown': drawdown, 'age': age, 'dbl': dbl,
            'pivot_high': float(h.iloc[p]), 'fractal_low': float(fractal_low)}


def check_codes(codes, label=''):
    print(f"🔺 顶部结构检查{label}")
    n_warn = 0
    for code in codes:
        code = str(code).zfill(6)
        try:
            df = load_stock(code)
        except Exception:
            continue
        r = detect_top(df)
        if r:
            n_warn += 1
            print(f"  {code} {r['stage']} | 分型高{r['pivot_high']:.2f} 回落{r['drawdown']:.1f}% "
                  f"破位线{r['fractal_low']:.2f} {r['dbl']} | 数据{df['d'].iloc[-1]}")
    if n_warn == 0:
        print('  全部持仓无顶部结构信号 ✓')
    return n_warn


def portfolio():
    pf = pd.read_excel('config/portfolio.xlsx')
    codes = [str(r['code']).zfill(6) for _, r in pf.iterrows()
             if str(r['code']).zfill(6) not in ('nan', '000000') and _safe(r)]
    check_codes(codes, ' | 持仓卖出警示层')


def _safe(r):
    try:
        return int(r['shares']) > 0
    except (ValueError, TypeError):
        return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--portfolio', action='store_true')
    ap.add_argument('--check', help='逗号分隔代码')
    a = ap.parse_args()
    if a.portfolio:
        portfolio()
    elif a.check:
        check_codes(a.check.split(','))
    else:
        ap.print_help()
