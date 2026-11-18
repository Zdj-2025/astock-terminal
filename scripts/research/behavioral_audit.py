#!/usr/bin/env python3
"""行为金融自我审计 — 2026-10-04 P1立项
把用户 trade_log 当回测对象: 处置效应/减仓位置/买点质量/持有期结构。
数据: config/trade_log.csv (121笔, 2026-05~09) + kline_store日线(不足腾讯补)
输出: 控制台 + reports/behavioral_audit_20261004.md
"""
import csv, json, urllib.request, datetime as dt
from collections import defaultdict
import pandas as pd

BASE = '.'
PY = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={},day,2025-06-01,,320,qfq'

def tx_code(code):
    return ('sh' if code.startswith(('6', '5')) else ('sz' if code.startswith(('0','1','3')) else 'bj')) + code

def fetch_kline(code, limit=320):
    """腾讯日线, 返回date->(o,h,l,c)"""
    url = PY.format(tx_code(code), '')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        d = json.load(urllib.request.urlopen(req, timeout=15))
        node = d['data'][tx_code(code)]
        rows = node.get('qfqday') or node.get('day') or []
        out = {}
        for r in rows:
            # [date, open, close, high, low, vol, ...]
            out[r[0].replace('-','')] = (float(r[1]), float(r[3]), float(r[4]), float(r[2]))
        return out
    except Exception as e:
        print(f'  ! {code} 腾讯失败: {e}')
        return {}

def load_local(code):
    try:
        df = pd.read_parquet(f'{BASE}/data/kline_store/{code}/daily.parquet')
        return {str(d.date() if hasattr(d,'date') else d)[:10].replace('-',''): (r.open, r.high, r.low, r.close)
                for r in df.itertuples()}
    except Exception:
        return {}

def get_kline(code):
    k = load_local(code)
    if len(k) < 100 or max(k) < '20260901':   # 不新鲜→腾讯全量
        k2 = fetch_kline(code)
        if len(k2) > len(k): k = k2
    return k

# ---------- 1. 解析 trade_log ----------
trades = []
with open(f'{BASE}/config/trade_log.csv') as f:
    r = csv.reader(f); next(r)
    for ln in r:
        if len(ln) < 7: continue
        date, code, name, action, price, shares = ln[0], ln[1], ln[2], ln[3], ln[4], ln[5]
        if action not in ('买', '卖', 'BUY'): continue
        try: price, shares = float(price), int(float(shares))
        except: continue
        trades.append(dict(date=date, code=code, name=name, action=action if action!='BUY' else '买',
                           price=price, shares=shares))
trades.sort(key=lambda x: x['date'])
print(f'有效买卖记录: {len(trades)} 笔 ({trades[0]["date"]} → {trades[-1]["date"]})')

# ---------- 2. 持仓轨迹 + 事件采样 ----------
positions = defaultdict(lambda: dict(shares=0, cost=0.0, first_buy=None))
sells, buys = [], []
kl_cache = {}
for t in trades:
    code = t['code']
    if code not in kl_cache: kl_cache[code] = get_kline(code)
    kl = kl_cache[code]
    kline_dates = sorted(kl)
    # 事件日及之后的K线索引
    pos = positions[code]
    if t['action'] == '买':
        total_cost = pos['cost'] * pos['shares'] + t['price'] * t['shares']
        pos['shares'] += t['shares']
        pos['cost'] = total_cost / pos['shares']
        if pos['first_buy'] is None: pos['first_buy'] = t['date']
        buys.append(dict(t=t, kl=kl, dates=kline_dates))
    else:  # 卖
        if pos['shares'] <= 0:
            sells.append(dict(t=t, cost=None, pnl_pct=None, hold_days=None, kl=kl, dates=kline_dates)); continue
        pnl_pct = (t['price'] / pos['cost'] - 1) * 100
        first = pos['first_buy'] or t['date']
        # 持有交易日数
        seg = [d for d in kline_dates if first.replace('-','') <= d <= t['date'].replace('-','')]
        hold_n = len(seg)
        sells.append(dict(t=t, cost=pos['cost'], pnl_pct=pnl_pct, hold_days=hold_n, kl=kl, dates=kline_dates))
        pos['shares'] -= t['shares']
        if pos['shares'] <= 0: pos['shares'], pos['cost'], pos['first_buy'] = 0, 0.0, None

# ---------- 3. 行为统计 ----------
def fwd_extreme(ev, field='high', n=20, mode='max'):
    """事件后n日内极值(相对卖价/买价)"""
    t, kl, dates = ev['t'], ev['kl'], ev['dates']
    d0 = t['date'].replace('-','')
    after = [d for d in dates if d > d0][:n]
    if len(after) < 5: return None
    vals = [kl[d][1 if field=='high' else 3] for d in after]
    return max(vals) if mode=='max' else min(vals)

report = ['# 行为金融自我审计 · 2026-10-04', f'> 数据: trade_log {len(trades)}笔有效买卖, 2026-05-19→09-24, {len(kl_cache)}标的; K线本地+腾讯qfq实时补', '']

# 3.1 处置效应: 卖出时的盈亏状态
valid_sells = [s for s in sells if s['pnl_pct'] is not None]
win_sells = [s for s in valid_sells if s['pnl_pct'] > 0]
loss_sells = [s for s in valid_sells if s['pnl_pct'] <= 0]
avg_hold_win = sum(s['hold_days'] or 0 for s in win_sells)/max(1,len(win_sells))
avg_hold_loss = sum(s['hold_days'] or 0 for s in loss_sells)/max(1,len(loss_sells))
report += ['## 1. 处置效应(卖出时点盈亏结构)', '',
  f'- 可配对卖出: {len(valid_sells)} 笔; **盈利状态卖出 {len(win_sells)} ({len(win_sells)/max(1,len(valid_sells))*100:.0f}%) / 亏损状态卖出 {len(loss_sells)} ({len(loss_sells)/max(1,len(valid_sells))*100:.0f}%)**',
  f'- 盈利单平均持有 **{avg_hold_win:.1f}** 个交易日 vs 亏损单平均持有 **{avg_hold_loss:.1f}** 个交易日',
  f'- 解读: 亏损单持有更久 = 处置效应(不愿认赔); 盈利单持有更久 = 反处置(让利润奔跑)。差距 {avg_hold_loss-avg_hold_win:+.1f} 天', '']

# 3.2 减仓位置质量: 卖价 vs 之后20日最高
sell_quality = []
for s in valid_sells:
    hi = fwd_extreme(s, 'high', 20, 'max')
    if hi: sell_quality.append(dict(name=s['t']['name'], date=s['t']['date'], px=s['t']['price'],
                                    hi=hi, ratio=s['t']['price']/hi*100, pnl=s['pnl_pct']))
good = [q for q in sell_quality if q['ratio'] >= 95]     # 卖在后续高点95%以上
mid = [q for q in sell_quality if 85 <= q['ratio'] < 95]
bad = [q for q in sell_quality if q['ratio'] < 85]       # 卖飞>15%
report += ['## 2. 减仓位置质量(卖价 / 之后20日最高)', '',
  f'- 样本 {len(sell_quality)} 笔: **卖得好(≥95%) {len(good)} / 中性(85-95%) {len(mid)} / 卖飞(<85%) {len(bad)}**',
  f'- 平均卖在后续高点的 **{sum(q["ratio"] for q in sell_quality)/max(1,len(sell_quality)):.1f}%**', '']
if bad:
    report += ['卖飞清单(按缺口排序):']
    for q in sorted(bad, key=lambda x: x['ratio'])[:8]:
        report.append(f'  - {q["date"]} {q["name"]}: 卖{q["px"]:.2f} vs 后高{q["hi"]:.2f} (卖在其{q["ratio"]:.0f}%, 浮盈{q["pnl"]:+.1f}%)')
    report.append('')

# 3.3 买点质量: 买价 vs 之后10日均价
buy_q = []
for b in buys:
    t, kl, dates = b['t'], b['kl'], b['dates']
    d0 = t['date'].replace('-','')
    after = [d for d in dates if d > d0][:10]
    if len(after) < 5: continue
    avg_c = sum(kl[d][3] for d in after)/len(after)
    lo = min(kl[d][2] for d in after)
    buy_q.append(dict(name=t['name'], date=t['date'], px=t['price'], avgc=avg_c, lo=lo,
                      prem=(t['price']/avg_c-1)*100))
below = [q for q in buy_q if q['prem'] < 0]
report += ['## 3. 买点质量(买价 vs 之后10日收盘均值)', '',
  f'- 样本 {len(buy_q)} 笔: 买入价低于后10日均价的 **{len(below)} 笔 ({len(below)/max(1,len(buy_q))*100:.0f}%)** — 即"买完短期就涨"的比例',
  f'- 平均溢价: **{sum(q["prem"] for q in buy_q)/max(1,len(buy_q)):+.1f}%** (负=买得便宜)", "']
worst = sorted(buy_q, key=lambda x: -x['prem'])[:5]
if worst:
    report += ['追高清单(买价高于后10日均价最多):']
    for q in worst:
        report.append(f'  - {q["date"]} {q["name"]}: 买{q["px"]:.2f}, 溢价{q["prem"]:+.1f}%')
    report.append('')

# 3.4 已实现盈亏结构(按标的汇总)
realized = defaultdict(lambda: dict(pnl=0.0, name=''))
for s in valid_sells:
    c = s['t']['code']
    realized[c]['pnl'] += (s['t']['price'] - (s['cost'] or 0)) * s['t']['shares']
    realized[c]['name'] = s['t']['name']
total = sum(v['pnl'] for v in realized.values())
winr = [v for v in realized.values() if v['pnl'] > 0]
report += ['## 4. 已实现盈亏结构(2026-05~09, 按标的)', '',
  f'- 标的数 {len(realized)}, 合计已实现 **{total:+,.0f} 元**; 盈利标的 {len(winr)} / 亏损 {len(realized)-len(winr)}',
  f'- TOP5盈亏:']
for c, v in sorted(realized.items(), key=lambda kv: -kv[1]['pnl'])[:5]:
    report.append(f'  - {v["name"]}({c}): {v["pnl"]:+,.0f}')
for c, v in sorted(realized.items(), key=lambda kv: kv[1]['pnl'])[:5]:
    report.append(f'  - {v["name"]}({c}): {v["pnl"]:+,.0f}')
report.append('')

# 3.5 数据局限声明
report += ['## 5. 局限', '- trade_log 无时间戳→早盘/尾盘决策质量无法分析',
  '- 9月最后交易日9/24后的卖出样本后视窗口不足20日(≥5日纳入)', '- realized 用加权成本近似, 分红/配售事件未计入现金流', '']

rep = '\n'.join(report)
print(rep)
open(f'{BASE}/reports/behavioral_audit_20261004.md','w').write(rep)
print(f'\n已写 reports/behavioral_audit_20261004.md')
