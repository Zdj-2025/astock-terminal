#!/usr/bin/env python3
"""组合级回撤总闸设计与回测 — 2026-10-04 P2
数据现实: 无日频净值历史 → trade_log+腾讯qfq重构(锚2026-06-16快照)
输出: 日频净值/回撤序列 + 各阈值(-6/-8/-10/-12%)触发点及触发后20日表现 → 阈值拍板数据
"""
import csv, json, urllib.request
from collections import defaultdict
import pandas as pd

BASE = '.'
PY = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={},day,2026-06-01,,320,qfq'

def tx_code(c):
    return ('sh' if c.startswith(('6','5')) else 'sz') + c

def fetch(c):
    try:
        req = urllib.request.Request(PY.format(tx_code(c)), headers={'User-Agent':'Mozilla/5.0'})
        d = json.load(urllib.request.urlopen(req, timeout=15))
        node = d['data'][tx_code(c)]
        rows = node.get('qfqday') or node.get('day') or []
        return {r[0].replace('-',''): float(r[2]) for r in rows}   # date->close
    except Exception:
        return {}

ANCHOR = dict(date='20260616', total=433992.14, cash=125018.24, mv=308973.40)

# ---------- 1. 解析trade_log (6/16起) ----------
events = defaultdict(list)   # date -> [(code,action,amount)]
with open(f'{BASE}/config/trade_log.csv') as f:
    r = csv.reader(f); next(r)
    for ln in r:
        if len(ln) < 7: continue
        date, code, action, amount = ln[0].replace('-',''), ln[1], ln[3], ln[6]
        if action not in ('买','卖','BUY'): continue
        try: amount = float(amount)
        except: continue
        if date < ANCHOR['date']: continue
        events[date].append((code, '买' if action=='BUY' else action, amount))

# ---------- 2. 重构持仓 ----------
# 6/16锚: 持仓 = 锚mv 按当日持仓分布? 无历史持仓明细 → 用"锚后第一笔出现的标的+trade_log回溯6/16前仓位"近似:
# 简化: 6/16 起所有 trade_log 标的中, 在6/16已持有者不可知 → 改用 现金流法:
# total(t) = total(t-1) + 持仓市值变动; 持仓市值变动需持仓明细。
# 折衷: 用「锚后仓位重构」: 6/16持仓 = {} 起步会错。改为锚定 9/30 持仓反推亦复杂。
# 最终方案: 双向重构——先把全期(5/19起)事件累计得到9月末持仓, 反推6/16持仓 = 9月末持仓回放6/16后的卖出/买入。
kl = {}
trades_all = []
with open(f'{BASE}/config/trade_log.csv') as f:
    r = csv.reader(f); next(r)
    for ln in r:
        if len(ln) < 7: continue
        date, code, name, action, price, shares = ln[0].replace('-',''), ln[1], ln[2], ln[3], ln[4], ln[5]
        if action not in ('买','卖','BUY'): continue
        try: price, shares = float(price), int(float(shares))
        except: continue
        trades_all.append(dict(date=date, code=code, action='买' if action=='BUY' else action, price=price, shares=shares))
trades_all.sort(key=lambda x: x['date'])

# 6/16持仓反推: 全期正持仓(5/19-9/24) → 减去6/16后的净变化
hold = defaultdict(int)
for t in trades_all:
    hold[t['code']] += t['shares'] if t['action']=='买' else -t['shares']
for t in [x for x in trades_all if x['date'] >= ANCHOR['date']]:
    hold[t['code']] -= t['shares'] if t['action']=='买' else -t['shares']
hold = {c: s for c, s in hold.items() if s > 0}

# 锚现金: 无法精确分配, 用 mv 按比例归一到锚: scale = ANCHOR.mv / Σ(hold×6/16close)
need = set(hold) | {t['code'] for t in trades_all if t['date'] >= ANCHOR['date']}
for c in need: kl[c] = fetch(c)

mv0 = sum(hold[c] * kl[c].get(ANCHOR['date'], 0) for c in hold)
scale = ANCHOR['mv'] / mv0 if mv0 > 0 else 1.0   # qfq价格与实际持仓市值的校准系数
hold = {c: s*scale for c, s in hold.items()}

# ---------- 3. 日频净值 ----------
all_dates = sorted({d for c in kl.values() for d in c if d >= ANCHOR['date']})
cash = ANCHOR['cash']
curve = []
for d in all_dates:
    for (c, act, amt) in events.get(d, []):
        cash -= amt if act=='买' else -amt
    mv = sum(s * kl[c].get(d, list(kl[c].values())[-1] if kl[c] else 0) for c, s in hold.items())
    curve.append(dict(date=d, total=cash+mv, cash=cash, mv=mv))
cdf = pd.DataFrame(curve)
cdf['peak'] = cdf['total'].cummax()
cdf['dd'] = (cdf['total']/cdf['peak'] - 1) * 100

# ---------- 4. 闸门触发点分析 ----------
print(f"日频净值: {cdf.date.min()} → {cdf.date.max()} ({len(cdf)}个交易日)")
print(f"期末 {cdf.total.iloc[-1]:,.0f} (锚 {ANCHOR['total']:,.0f}), 区间收益 {cdf.total.iloc[-1]/ANCHOR['total']*100-100:+.1f}%")
print(f"最大回撤 {cdf.dd.min():.1f}%\n")
print(f"{'阈值':>5} {'触发次数':>6} 触发日(触发后20日总资产变动)")
idx_list = cdf.index.tolist()
for gate in (-6, -8, -10, -12):
    fired, last_fire = [], -999
    for i in cdf.index:
        if cdf.dd[i] <= gate and i - last_fire > 20:   # 触发后20日冷却
            fired.append(i); last_fire = i
    lines = []
    for i in fired:
        fwd = cdf.total.iloc[i:i+21]
        chg = fwd.iloc[-1]/fwd.iloc[0]*100-100 if len(fwd)>5 else float('nan')
        lines.append(f"{cdf.date[i]}(后20日{chg:+.1f}%)")
    print(f"{gate:>4}% {len(fired):>6}次 {' '.join(lines) if lines else '(从未触发)'}")

cdf.to_csv(f'{BASE}/reports/accounting/daily_equity_reconstructed.csv', index=False)
print("\n净值序列→ reports/accounting/daily_equity_reconstructed.csv")
