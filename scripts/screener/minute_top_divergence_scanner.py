#!/usr/bin/env python3
"""持仓分钟顶背离scanner — 卖出预警专用(2026-09-09).

背景: 用户要求"股票进入持仓就拉分钟线,关注60/30/15分钟顶背离,作为卖出重要信号".
双确认规则(2026-08-20数据港教学): 60+30min同时背离才有效,单30min不构成卖出理由.
卖出五维检查防机械报警: 日线红柱加速不卖/均线多头不卖 — 本scanner输出带日线上下文.

级别:
  🔴强预警 60+30同时顶背离 → 结合日线状态给操作建议
  🟡观察   单周期(60或30)背离
  🔵前哨   仅15min背离(最灵敏,只提示)
  ⚪无

数据: data/store/kline/{code}/{15,30,60}min.parquet(每日minute_data --update刷新,
ETF走腾讯备用源). 盘中(--live)自动叠加腾讯当日实时bar(带除权守卫).

告警落盘 reports/minute_divergence_log.csv 供日后复盘信号准确率(学习闭环).

用法: cd stock && PYTHONPATH=. python scripts/screener/minute_top_divergence_scanner.py [--live] [--code 600522]
"""
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from scripts.analyzers.minute_top_divergence import detect_top_divergence, scan_position

LOG_PATH = PROJECT_ROOT / 'reports/minute_divergence_log.csv'


def daily_context(code):
    """日线上下文(防机械报警): 收盘vs MA20 + 日线MACD红柱状态. 返回dict."""
    import numpy as np
    path = PROJECT_ROOT / f'data/store/kline/{code}/D.parquet'
    if not path.exists():
        return {'ma20': None, 'below_ma20': None, 'hist_state': '无数据'}
    df = pd.read_parquet(path)
    for col in ('datetime', 'day', 'trade_date'):
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    c = df['close'].astype(float).reset_index(drop=True)
    if len(c) < 35:
        return {'ma20': None, 'below_ma20': None, 'hist_state': '数据不足'}
    ma20 = float(c.tail(20).mean())
    below = bool(c.iloc[-1] < ma20)
    s = pd.Series(c)
    dif = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean())
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea).values
    h5 = hist[-6:-1]
    if len(h5) >= 3 and h5[-1] > 0:
        if np.all(np.diff(h5) < 0):
            state = '红柱衰减⚠️'
        elif np.all(np.diff(h5) > 0):
            state = '红柱加速🚀(动能仍在)'
        else:
            state = '红柱震荡'
    else:
        state = '绿柱'
    return {'ma20': round(ma20, 2), 'below_ma20': below, 'hist_state': state}


def advice_for(level, ctx):
    """操作建议(卖出五维检查精神: 背离是维度之一,不是全部)."""
    if level == '🔴强预警':
        if ctx.get('below_ma20') or '衰减' in str(ctx.get('hist_state')):
            return '🔴减仓预警: 双周期背离+日线转弱,按3S规则执行'
        return '⚠️双周期背离但日线仍强: 不机械卖,挂紧移动止损/减仓1/3锁利润'
    if level == '🟡观察':
        return '观察: 单周期背离,等第二周期确认或日线转弱'
    if level == '🔵前哨':
        return '仅15min背离(前哨),不动手'
    return '—'


def load_portfolio():
    df = pd.read_excel('config/portfolio.xlsx')
    df['code'] = df['code'].astype(str).str.split('.').str[0].str.zfill(6)
    return df


def main():
    args = sys.argv[1:]
    live = '--live' in args
    code_arg = None
    if '--code' in args:
        code_arg = args[args.index('--code') + 1].zfill(6)

    pf = load_portfolio()
    name_map = dict(zip(pf['code'], pf['name'].astype(str)))

    if code_arg:
        results = [detect_top_divergence(code_arg, live=live)]
        name_map = {code_arg: name_map.get(code_arg, '')}
    else:
        results = scan_position(live=live)

    mode = '盘中实时(腾讯当日bar已叠加)' if live else '离线(parquet历史)'
    print(f"📉 持仓分钟顶背离scanner | 60/30/15min MACD(12,26,9) | {mode}")
    print(f"   双确认规则: 60+30同时背离=🔴(数据港教学实证); 15min仅前哨")
    print(f"   持仓: {len(results)}只\n")

    alerts = []
    print(f"{'级别':<7}{'代码':<8}{'名称':<9}{'60min':<13}{'30min':<13}{'15min':<13}{'日线破MA20':<10}{'日线红柱'}")
    print('-' * 92)
    for r in results:
        code = r['code']
        cells = []
        for f in ('60', '30', '15'):
            d = r['per_freq'].get(f)
            if d is None:
                cells.append('无数据')
            elif d['is_div']:
                cells.append(f"背离{d['bars_ago']}根前")
            else:
                cells.append('—')
        ctx = daily_context(code)
        below = ('是' if ctx.get('below_ma20') else '否') if ctx.get('below_ma20') is not None else '?'
        print(f"{r['level']:<7}{code:<8}{name_map.get(code, '')[:8]:<9}"
              f"{cells[0]:<13}{cells[1]:<13}{cells[2]:<13}{below:<10}{ctx.get('hist_state', '')}")
        if r['level'] != '⚪无':
            alerts.append({'level': r['level'], 'code': code, 'name': name_map.get(code, ''),
                           'cells': cells, 'ctx': ctx, 'r': r})

    if alerts:
        print(f"\n--- 操作建议(结合日线上下文,防机械报警) ---")
        for a in alerts:
            print(f"  {a['level']} {a['name']}({a['code']}): {advice_for(a['level'], a['ctx'])}")
    else:
        print(f"\n✅ 全部持仓无分钟顶背离")

    # 告警落盘(学习闭环: 日后复盘信号准确率)
    if alerts:
        today = datetime.now().strftime('%Y-%m-%d')
        rows = []
        for a in alerts:
            rows.append({
                'date': today, 'code': a['code'], 'name': a['name'], 'level': a['level'],
                'd60': a['r']['d60'], 'd30': a['r']['d30'], 'd15': a['r']['d15'],
                'daily_below_ma20': a['ctx'].get('below_ma20'),
                'daily_hist': a['ctx'].get('hist_state'),
                'advice': advice_for(a['level'], a['ctx']),
            })
        new = pd.DataFrame(rows)
        if LOG_PATH.exists():
            old = pd.read_csv(LOG_PATH)
            old = old[~((old['date'] == today) & (old['code'].isin(new['code'])))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(LOG_PATH, index=False)
        print(f"\n  告警已落盘 {LOG_PATH.relative_to(PROJECT_ROOT)}({len(rows)}条,供复盘)")


if __name__ == '__main__':
    main()
