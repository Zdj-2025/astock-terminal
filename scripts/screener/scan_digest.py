#!/usr/bin/env python3
"""扫描总结器+焦点清单(2026-08-27建,用户需求:结论直给/多角度收敛为焦点)。

输入: /tmp/scan_result.log(daily_scan_all输出,格式自控)
逻辑: 多信号共振优先(divergence✅×mainline🥇≥70×breakout★真×smart4),
     叠加位置约束(分位/距高点)与主线背景(仅描述不预测——339周检验追热点-0.08%/周)
输出: ①一段话结论 ②方向(看/避) ③焦点清单≤5只(代码/名称/信号数/触发条件)

用法: python scripts/screener/scan_digest.py  (在daily_scan_all后跑)
"""
import os
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

LOG = '/tmp/scan_result.log'


def parse_log():
    txt = open(LOG, encoding='utf-8', errors='ignore').read()
    signals = defaultdict(dict)  # code -> {div, main, brk, smart, defense, name, ind}
    # divergence确认
    for m in re.finditer(r'^(\d{6})\s+(\S+)\s+[\d.]+\s+-\d+%.{0,40}?✅确认', txt, re.M):
        signals[m.group(1)]['div'] = True; signals[m.group(1)]['name'] = m.group(2)
    # 主线龙头(🥇行: 代码 名称 行业 价 质量 PE 质押 档位)
    for m in re.finditer(r'^(\d{6})\s+(\S+)\s+(\S+)\s+[\d.]+\s+(\d+)\s+.*🥇', txt, re.M):
        signals[m.group(1)].update(main=int(m.group(4)), name=m.group(2), ind=m.group(3))
    # 突破★真
    for m in re.finditer(r'^(\d{6})\s+(\S+)\s+(\S+)\s+[\d.]+\s+[\d.]+x\s+-\d+%\s+(\d+)%.*★真', txt, re.M):
        signals[m.group(1)].setdefault('name', m.group(2))
        signals[m.group(1)]['brk'] = int(m.group(4)); signals[m.group(1)]['ind'] = m.group(3)
    # smart4
    for m in re.finditer(r'◆(\d{6})(\S+)\s+(\S+)\s+[\d.]+\s+4\s+(\d+)', txt):
        signals[m.group(1)].setdefault('name', m.group(2))
        signals[m.group(1)]['smart'] = 4; signals[m.group(1)]['ind'] = m.group(3)
        signals[m.group(1)]['pe'] = int(m.group(4))
    # 防御🚀
    for m in re.finditer(r'^(\d{6})\s+(\S+)\s+(\S+).{0,60}?🚀主升浪', txt, re.M):
        signals[m.group(1)].setdefault('name', m.group(2)); signals[m.group(1)]['rocket'] = True
    # 行业热度
    heat_top = re.findall(r'^(.{2,8}?)\s{2,}\+[\d.]+%\s+[+\-][\d.]+%\s+[+\-][\d.]+%.*\d+\s*$', txt, re.M)[:5]
    return signals, heat_top


def main():
    if not os.path.exists(LOG):
        print('无扫描日志,先跑daily_scan_all')
        return
    signals, heat_top = parse_log()
    sl = pd.read_parquet('data/store/stock_list.parquet')
    sl['code'] = sl['symbol'].astype(str).str.zfill(6)

    # 共振计数 + 位置过滤
    rows = []
    for code, s in signals.items():
        n_sig = sum([bool(s.get('div')), bool(s.get('main') and s['main'] >= 70),
                     bool(s.get('brk') and s['brk'] >= 55), bool(s.get('smart') == 4), bool(s.get('rocket'))])
        if n_sig >= 2:
            try:
                df = pd.read_parquet(f'data/store/kline/{code}/D.parquet').sort_values('date')
                c = df['close'].astype(float)
                pct = (c.tail(250) < c.iloc[-1]).mean()
                ma20 = c.tail(20).mean()
                hi20 = c.tail(20).max()
                name = s.get('name') or sl.loc[sl['code'] == code, 'name'].values[0]
                rows.append(dict(code=code, name=name, ind=s.get('ind', ''), n=n_sig,
                                 div=s.get('div', False), main=s.get('main'), brk=s.get('brk'),
                                 smart=s.get('smart'), pe=s.get('pe'),
                                 px=c.iloc[-1], pct=pct, vsMA20=c.iloc[-1]/ma20-1, hi20=hi20))
            except Exception:
                continue
    rows.sort(key=lambda r: (-r['n'], r['pct']))
    # 焦点=共振数×低位优先,取5
    focus = rows[:5]

    print(f"\n{'='*70}")
    print("📌 扫描总结 | 多角度收敛后的焦点(资金有限,只看这些)")
    print('=' * 70)
    print(f"\n【一句话结论】共{len(rows)}只多信号共振股" +
          (f",焦点{len(focus)}只" if focus else ";今日无达标共振——观望") +
          ("。热点榜仅作背景(追热点下周超额-0.08%,无预测力)。" if heat_top else ""))

    if heat_top:
        print(f"\n【背景·热点榜(描述非预测)】{' / '.join(heat_top)}")

    if focus:
        print(f"\n【焦点清单】(共振信号数↓ + 低位优先)")
        print(f"{'代码':<8}{'名称':<10}{'行业':<10}{'共振':>4}{'分位':>5}{'vsMA20':>8}  信号明细")
        for r in focus:
            tags = []
            if r['div']: tags.append('背离✅')
            if r['main']: tags.append(f"主线{r['main']}")
            if r['brk']: tags.append(f"突破{r['brk']}%")
            if r['smart']: tags.append('smart4')
            print(f"{r['code']:<8}{r['name'][:9]:<10}{str(r['ind'])[:8]:<10}{r['n']:>4}{r['pct']:>5.0%}"
                  f"{r['vsMA20']:>+7.1%}  {'+'.join(tags)}")
        print("\n触发纪律: 焦点≠买入。各自等触发(回踩缩量/放量破20日高/闸门开),三闸门优先。")


if __name__ == '__main__':
    main()
