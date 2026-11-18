#!/usr/bin/env python3
"""盘前持仓检查 v3(多维·不标签化): 读portfolio(买入依据+买入日期+止损线) + 多维技术 + 持N天 + 卖出触发。
设计(2026-08-05,吸取"别简单化"教训): 持仓评估是多维综合(买入依据+技术+止损规则),
不打单维🔴/🚀判断标签——避免2维技术否定5维买入依据(如中航刚买就🔴的矛盾)。
- 买入依据: portfolio notes(买入时定,盘前读) + 持N天(买入时间,刚买技术弱正常)
- 技术多维: f1(中期21日因果) + ret5(短期) + 3S(卖出三条件) + 盈亏/MA20/量比/底背离(只数值,不打标签)
- 卖出触发: 止损线破(<=stop_loss) + 3S满(MA5<MA10+MACD死叉/柱负+破MA20) + 垫厚移动止损
二阶导: 因果左导数W=21,弱因子IC=0.029只数值参考,不作判断。减速见顶已作废(p=0.90)。"""
import pandas as pd, numpy as np
from scipy.signal import argrelextrema
from datetime import datetime
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
sys.path.insert(0, os.getcwd())
from scripts.core.derivatives import compute_derivatives  # 唯一权威实现(2026-08-17:旧本地版pos语义三重错误已废,见memory scipy_savgol_pos_semantics)

print(f"=== 盘前持仓检查 v3(多维·不标签化) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# 分钟数据新鲜度哨兵(2026-08-17 P0-1B): 8/5静默失败9交易日教训制度化
_sent = 'data/store/kline/_last_minute_update'
if os.path.exists(_sent):
    _t = open(_sent).read().strip()
    try:
        _last = datetime.strptime(_t[:19], '%Y-%m-%d %H:%M:%S')
        _age = (datetime.now() - _last).days
        if _age >= 1:
            print(f"⚠️ 分钟数据陈旧: 上次更新{_t[:16]}(≥{_age}天前) — 18:00分钟cron可能失败,分钟背离分析不可靠\n")
    except Exception:
        pass


# 读portfolio(权威持仓: code/name/cost/shares/buy_date/stop_loss/notes)
pf = pd.read_excel('config/portfolio.xlsx')
HOLDINGS = []
for _, r in pf.iterrows():
    code = str(r['code']).zfill(6)
    bd = str(r.get('buy_date', ''))[:10]
    HOLDINGS.append((code, str(r['name']), float(r['avg_cost']), int(r['shares']),
                     bd, float(r.get('stop_loss', 0) or 0), str(r.get('notes', ''))))
SG_W, SG_P = 21, 3


def qfq_close(df):
    df = df.sort_values('date').reset_index(drop=True)
    c = df['close'].values.astype(float).copy()
    rets = df['close'].pct_change().values
    for i in range(len(c) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            c[:i + 1] = c[:i + 1] * (c[i + 1] / c[i])
    return c


def macd_dif(close):
    s = pd.Series(close)
    return (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).values


def compute_derivatives_local_removed():
    """2026-08-17: 本地实现已删除(pos=W-1实为窗口左端+奇数阶符号反+convolve方向错,三重错误;
    x²常数f''验证掩盖错位). 权威实现→scripts/core/derivatives.py(顶部import)."""
    raise NotImplementedError('use scripts.core.derivatives.compute_derivatives')


def bottom_div(close, dif, lookback=60):
    rc, rd = close[-lookback:], dif[-lookback:]
    tr = argrelextrema(rd, np.less, order=5)[0]
    if len(tr) < 2:
        return False, 0.0
    t1, t2 = tr[-2], tr[-1]
    if rc[t2] < rc[t1] and rd[t2] > rd[t1]:
        bounce = close[-1] / rc[t2] - 1
        drawdown = close[-1] / max(rc) - 1
        if bounce > 0.08 or drawdown > -0.05:
            return False, 0.0
        return True, (rd[t2] - rd[t1]) / (abs(rd[t1]) + 1e-9)
    return False, 0.0


rows = []
# invalidation_lines.csv覆盖(权威线,2026-10-18修: 程序旧线bug——中天31.50事件)
try:
    import csv as _csv
    with open('config/invalidation_lines.csv') as _f:
        for _r in _csv.reader(_f):
            if len(_r) >= 4 and _r[0].isdigit():
                for _i, _h in enumerate(HOLDINGS):
                    if _h[0] == int(_r[0]):
                        _v = _r[2]
                        try:
                            HOLDINGS[_i] = _h[:5] + (float(_v),) + _h[6:]
                        except ValueError:
                            pass  # 文本线(主动兑现等)跳过, 保留原值
except Exception as _e:
    print(f'⚠️ invalidation_lines读取失败: {_e}')

for code, name, cost, shares, buy_date, stop_loss, note in HOLDINGS:
    try:
        df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    except Exception:
        continue
    df = df.sort_values('date').reset_index(drop=True)
    qc = qfq_close(df)
    # 持N天(买入时间,刚买技术弱正常)
    try:
        hold_days = (datetime.now() - datetime.strptime(buy_date, '%Y-%m-%d')).days
    except Exception:
        hold_days = -1
    price = float(qc[-1])
    s = pd.Series(qc)
    ma5 = float(s.tail(5).mean()); ma10 = float(s.tail(10).mean()); ma20 = float(s.tail(20).mean())
    vol_col = 'vol' if 'vol' in df.columns else 'volume'
    vol = df[vol_col].values.astype(float)
    vol_ratio = float(vol[-1] / pd.Series(vol).tail(20).mean())
    ret5 = (qc[-1] / qc[-5] - 1) * 100 if len(qc) >= 5 else 0
    dif = macd_dif(qc)
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    hist = dif - dea
    is_div, div_st = bottom_div(qc, dif)
    f1, f2 = compute_derivatives(qc)
    pnl = (price / cost - 1) * 100 if cost > 0 else 0
    below = price < ma20
    # 3S卖出三条件(回测验证硬规则): MA5<MA10 + MACD死叉/柱负 + 破MA20
    s3_cnt = sum([ma5 < ma10, (dif[-1] < dea[-1]) or (hist[-1] < 0), below])
    # 卖出触发: 止损线破 / 3S满 / 垫厚移动止损
    sl_break = stop_loss > 0 and price <= stop_loss
    sl_dist = (price / stop_loss - 1) * 100 if stop_loss > 0 else None
    thick = pnl > 30  # 垫厚>+30%,移动止损锁利
    rows.append(dict(code=code, name=name, cost=cost, price=price, hold=hold_days, pnl=pnl,
                     ma20=ma20, below=below, vr=vol_ratio, ret5=ret5, div=is_div, div_st=div_st,
                     s3_cnt=s3_cnt, f1=f1, f2=f2, stop_loss=stop_loss, sl_break=sl_break,
                     sl_dist=sl_dist, thick=thick, note=note))

print(f"{'代码':<8}{'名称':<10}{'持':>4}{'盈亏':>7}{'f1':>6}{'ret5':>7}{'量比':>5}{'3S':>6}  买入依据(简) → 卖出触发")
print('-' * 115)
for r in rows:
    f1s = f"{r['f1']:+.1f}" if not np.isnan(r['f1']) else ' NaN'
    # 卖出触发描述(多维,不标签化)
    if r['sl_break']:
        trig = f"🔴止损破{r['stop_loss']:.2f}!"
    elif r['stop_loss'] > 0:
        trig = f"止损{r['stop_loss']:.2f}(距{r['sl_dist']:+.1f}%)"
    elif r['thick']:
        trig = f"垫厚+{r['pnl']:.0f}%移动止损"
    else:
        trig = '无止损线'
    if r['s3_cnt'] == 3:
        trig += ' 3S满⚠️'
    nt = r['note'][:22].replace('\n', ' ')
    print(f"{r['code']:<8}{r['name']:<10}{r['hold']:>3}d{r['pnl']:>+6.1f}%{f1s:>6}{r['ret5']:>+6.1f}%{r['vr']:>5.2f}  {r['s3_cnt']}/3  {nt:<22} → {trig}")

# 卖出触发汇总(硬规则,非标签)
sl_brk = [r for r in rows if r['sl_break']]
s3full = [r for r in rows if r['s3_cnt'] == 3]
below = [r for r in rows if r['below']]
divs = [r for r in rows if r['div']]
thicks = [r for r in rows if r['thick']]
print(f"\n--- 卖出触发(硬规则) ---")
if sl_brk:
    print(f"🔴 止损线破 {len(sl_brk)} 只: {', '.join(r['code'] for r in sl_brk)} — 次日必执行")
if s3full:
    print(f"⚠️ 3S满 {len(s3full)} 只: {', '.join(r['code'] for r in s3full)} — 三条件全满(MA5<MA10+MACD死叉/柱负+破MA20),按规则减(宽基/垫厚纪律除外)")
if thicks:
    print(f"💰 垫厚移动止损 {len(thicks)} 只: {', '.join(r['name'] for r in thicks)} — +30%以上锁利,破前高回撤/MA20减")
print(f"\n--- 技术参考(只数值,不判好坏) ---")
print(f"破MA20 {len(below)} 只: {', '.join(r['code'] for r in below) if below else '无'}")
print(f"底背离 {len(divs)} 只: {', '.join('{}(str{:.2f})'.format(r['code'], r['div_st']) for r in divs) if divs else '无'}")
print(f"\n💡 持仓去留=多维综合(买入依据变否+技术数值+卖出触发),不靠单标签;刚买(持N天少)技术弱正常(B反一日没扭转21日趋势)")
print(f"💡 f1中期/ret5短期/3S卖出规则: 卖出靠3S满+止损线破+移动止损;二阶导f1/f2弱因子只数值参考(IC=0.029)")

# === 🛡️盈利保护检查(profit_guard每日自动算应设止损,2026-08-07集成) ===
# 个股按铁律v2分档(盈利≥15%→成本/≥30%→+10%/≥50%→+25%); 宽基ETF用再评估线,不采纳算法-10%
import sys as _sys
if os.getcwd() not in _sys.path:
    _sys.path.insert(0, os.getcwd())
try:
    from scripts.analyzers.profit_guard import ProfitGuard
    _g = ProfitGuard()
    _pf2 = pd.read_excel('config/portfolio.xlsx')
    _pos = []
    for r in rows:
        if r['cost'] <= 0:  # 成本artifact(半导体ETF)跳过
            continue
        _row = _pf2[_pf2['code'].astype(str).str.zfill(6) == r['code']]
        _pp = float(_row.iloc[0]['position_pct']) / 100 if len(_row) else 0
        _pos.append({'code': r['code'], 'name': r['name'], 'cost': r['cost'], 'price': r['price'],
                     'shares': 0, 'pnl_pct': r['pnl'], 'position_pct': _pp, 'stop_loss': r['stop_loss']})
    _res = _g.check_portfolio(_pos)
    _isetf = lambda c: str(c)[:2] in ('51', '15', '56', '58')
    _upd = [u for u in _res['sl_updates'] if not _isetf(u['code']) and abs(u['new_sl'] - u['old_sl']) > 0.02]
    _etf = [u for u in _res['sl_updates'] if _isetf(u['code'])]
    print(f"\n--- 🛡️盈利保护(profit_guard每日自动算止损) ---")
    if _upd:
        print(f"📐 个股止损需更新 {len(_upd)} 只(按算法采纳):")
        for u in _upd:
            print(f"   {u['name']}({u['code']}): {u['old_sl']:.2f}→{u['new_sl']:.2f} | {u['reason']}")
    else:
        print(f"✅ 个股止损已按算法到位(无大幅需调整)")
    if _etf:
        print(f"⏭️ 宽基ETF算法-10%不采纳(用再评估线/不砍地板): " + ', '.join(f"{u['name']}{u['new_sl']:.2f}" for u in _etf))
    _miss = [r for r in rows if r['cost'] > 0 and r['stop_loss'] <= 0 and not _isetf(r['code'])]
    if _miss:
        print(f"🚨 个股漏设止损 {len(_miss)} 只: " + ', '.join(r['name'] for r in _miss))
    else:
        print(f"✅ 个股止损线齐全(无漏设)")
    print(f"💡 铁律v2:盈利≥15%→成本/≥30%→+10%/≥50%→+25%; <15%仓位分档-5/-8/-10%; 宽基用再评估线不砍地板")
except Exception as _e:
    print(f"\n⚠️ profit_guard检查跳过: {_e}")

# --- 策略层: 因子暴露 + 仓位建议(2026-08-20,回答"配多少"而非"选什么") ---
try:
    from scripts.analyzers.factor_exposure import analyze_factor_exposure
    analyze_factor_exposure()
except Exception as _e:
    print(f"\n⚠️ 因子暴露跳过: {_e}")

try:
    from scripts.analyzers.position_strategy import analyze_position_strategy, policy_tilt_shadow
    r = analyze_position_strategy()
    policy_tilt_shadow(r)  # P3影子模式:每日落盘90天对答案(2026-09-11立),不参与决策
except Exception as _e:
    print(f"\n⚠️ 仓位策略跳过: {_e}")

# --- 动态结构止损(2026-08-29: swing low追踪替代手动拍脑袋,回测t=5.8) ---
try:
    from scripts.analyzers.structural_stop import scan_portfolio_stops
    scan_portfolio_stops(verbose=True, update_portfolio=False)
except Exception as _e:
    print(f"\n⚠️ 结构止损跳过: {_e}")

# --- 三闸门(2026-08-24 8月复盘落地: 顶部禁买令/因子35%上限/梯度兑现) ---
try:
    from scripts.analyzers.buy_gates import check_buy_gates
    check_buy_gates()
except Exception as _e:
    print(f"\n⚠️ 三闸门跳过: {_e}")

# --- 供应链/大客户监控(2026-08-24: 数据港-阿里传导漏诊教训) ---
try:
    from scripts.data.supply_chain_monitor import check_supply_chain, scan_order_signals
    check_supply_chain()
    scan_order_signals()
except Exception as _e:
    print(f"\n⚠️ 供应链监控跳过: {_e}")

# --- 持仓分钟顶背离监控(2026-09-09用户需求: 60/30/15min顶背离=卖出重要信号) ---
try:
    from scripts.screener.minute_top_divergence_scanner import main as _mtd_main
    import sys as _sys
    _argv_backup = _sys.argv[:]
    _sys.argv = ['minute_top_divergence_scanner.py']  # 默认离线模式(盘前无当日bar)
    _mtd_main()
    _sys.argv = _argv_backup
except Exception as _e:
    print(f"\n⚠️ 分钟顶背离监控跳过: {_e}")

# --- 筹码阻力地图(2026-09-10利弗莫尔"最小阻力线"量化: 近90日成交额分布/主分水岭/获利套牢比) ---
try:
    from scripts.analyzers.volume_profile import resistance_map
    resistance_map()
except Exception as _e:
    print(f"\n⚠️ 筹码阻力地图跳过: {_e}")

# ════════════════════════════════════════════════════════════════════
# R3 单页晨报(2026-09-11立) — stdout洪水→一页决策清单
# 结构: 大盘一行 | 必执行(收盘破线) | 监控线表 | 低吸候选≤3 | 现金一行
# ════════════════════════════════════════════════════════════════════
try:
    def _r3_brief():
        import pandas as pd
        from pathlib import Path
        root = Path('.')
        # 大盘
        from scripts.analyzers.position_strategy import get_sh_index
        idx = get_sh_index()
        ic = idx['close'].astype(float)
        r20 = (ic.iloc[-1] / ic.iloc[-21] - 1) * 100
        ma20 = ic.tail(20).mean()
        reg = '进攻' if r20 > 3 else ('防守' if r20 < -3 else '震荡')
        print('\n' + '▓' * 66)
        dcol = [c for c in idx.columns if 'date' in c.lower()][0]
        print(f'▓ 单页晨报 {str(idx[dcol].iloc[-1])[:10]} | 上证{ic.iloc[-1]:.0f} {reg}(ret20 {r20:+.1f}%) vs MA20 {ma20:.0f}')
        print(f'▓ ⚠️数据口径: 上证={str(idx[dcol].iloc[-1])[:10]}, 个股价格=本地parquet最近收盘(可能滞后1交易日), 盘中请用腾讯实时')
        # 必执行 + 监控线
        pf = pd.read_excel(root / 'config/portfolio.xlsx')
        pf['code'] = pf['code'].astype(str).str.zfill(6)
        must, watch = [], []
        for _, r in pf.iterrows():
            kp = root / f'data/store/kline/{r["code"]}/D.parquet'
            if not kp.exists():
                continue
            k = pd.read_parquet(kp, columns=['date', 'close'])
            for _c in ['datetime', 'day', 'trade_date']:
                if _c in k.columns and 'date' not in k.columns:
                    k = k.rename(columns={_c: 'date'})
            px = float(k['close'].astype(float).iloc[-1])
            try:
                sl = float(r['stop_loss'])
            except Exception:
                sl = 0
            if sl > 0 and px < sl:
                must.append(f"{r['name']}({r['code']}) {int(r['shares'])}股 收{px:.2f}<线{sl:.2f}")
            elif sl > 0:
                watch.append((px / sl - 1, f"{r['name']} {px:.2f}→线{sl:.2f}({(px/sl-1)*100:+.1f}%)"))
        print(f"▓ ①今日必执行(收盘破线): {'; '.join(must) if must else '无 ✓'}")
        watch.sort()
        if watch:
            print(f"▓ ②监控线最近3: {' | '.join(w[1] for w in watch[:3])}")
        # 低吸候选≤3: watchlist active带区间且贴近现价
        try:
            wl = pd.read_csv(root / 'config/watchlist.csv', dtype=str)
            act = wl[wl['note'].str.contains('active', na=False)]
            import re
            cands = []
            for _, w in act.iterrows():
                m = re.findall(r'区([\d.]+)-([\d.]+)', str(w['note']))
                if not m:
                    continue
                lo, hi = float(m[-1][0]), float(m[-1][1])
                kp = root / f'data/store/kline/{w["code"].zfill(6)}/D.parquet'
                if not kp.exists():
                    continue
                k = pd.read_parquet(kp, columns=['close'])
                px = float(k['close'].astype(float).iloc[-1])
                dist = (lo / px - 1) * 100
                if -2 <= dist <= 6:
                    cands.append((abs(dist), f"{w['name']} {w['code'].zfill(6)} 区{lo}-{hi}(现{px:.2f},距{dist:+.1f}%)"))
            cands.sort()
            print(f"▓ ③低吸候选(区距现价-2~+6%): {' | '.join(c[1] for c in cands[:3]) if cands else '无贴区者,全部等跌'}")
        except Exception:
            pass
        # 现金一行(引用上文详表)
        # 三轴配置(2026-09-12立项: 股/ETF/类债,回测P5动态收益回撤比6.5)
        try:
            BOND = {'601006', '600900', '600886'}
            etf_v = stk_v = bond_v = 0.0
            for _, r in pf.iterrows():
                kp = root / f'data/store/kline/{r["code"]}/D.parquet'
                if not kp.exists():
                    continue
                k = pd.read_parquet(kp, columns=['close'])
                mv = float(k['close'].astype(float).iloc[-1]) * float(r['shares'])
                if r['code'][:2] in ('51', '15', '56', '58'):
                    etf_v += mv
                elif r['code'] in BOND:
                    bond_v += mv
                else:
                    stk_v += mv
            tot = etf_v + stk_v + bond_v
            if tot > 0:
                print(f"▓ ④三轴(持仓内): ETF {etf_v/tot*100:.0f}% | 个股 {stk_v/tot*100:.0f}% | 类债 {bond_v/tot*100:.0f}%"
                      f"  ← P5动态目标: attack 80/10/10, shake 50/30/20, defense 20/50/30(攻/债/现)")
        except Exception:
            pass
        print(f"▓ ⑤政策tilt影子: 见上文🌑行")
        try:  # 外部变量:美债10年(2026-09-24立·节前尾盘杀跌真因)
            import subprocess as _sp
            _out = _sp.run(['bash','-c',
                "curl -sL 'https://api.frankfurter.dev/v1/latest?base=USD&symbols=CNY' 2>/dev/null | head -1"],
                capture_output=True,text=True,timeout=8).stdout
            # frankfurter给汇率不给美债; 降级为标注提醒
            print(f'▓ ⓾美债10Y: 需人工核(目标: >5.10⚠️ / >5.25🔴 — 节前尾盘杀跌真因=5年期破5%)')
        except Exception:
            pass
        try:  # ⑧赢家兑现建议(2026-09-24·卖单对称铺设修复; 2026-10-01加漏档⚠️三闸门审计落地)
            _winners = [(r['name'], r['code'], r['price'], r['cost'], r['pnl'])
                        for r in rows if r['pnl'] > 30]
            # 漏档检测: 浮盈>30% 且 (从未兑现 或 现价距上次兑现价又涨>15%) → 闸门3漏档
            # 2026-10-01: trade_log历史行字段数不齐(9~11),逐行文本解析容错,不用pd.read_csv
            def _missed_gate(cd, px):
                last_sell_px = None
                try:
                    with open('config/trade_log.csv', encoding='utf-8') as f:
                        for ln in f:
                            parts = ln.strip().split(',')
                            if len(parts) >= 5 and parts[1].zfill(6) == cd and parts[3] == '卖':
                                try:
                                    last_sell_px = float(parts[4])
                                except ValueError:
                                    continue
                except FileNotFoundError:
                    return ''
                if last_sell_px is None:
                    return '漏档⚠️从未兑现'
                if px / last_sell_px - 1 > 0.15:
                    return f'漏档⚠️距上次兑现@{last_sell_px}又涨{(px/last_sell_px-1)*100:.0f}%'
                return ''
            if _winners:
                print(f"▓ ⑧赢家兑现候选(浮盈>30%·回踩期应同时挂卖单):")
                for nm, cd, px, ct, pnl in _winners[:4]:
                    _tp = round(px * 1.03, 2)  # 高3%挂单
                    _flag = _missed_gate(cd, px)
                    print(f"   {nm}({cd}) 浮盈{pnl:+.0f}% → 建议: {_tp}减1/3或1/4 {'['+_flag+']' if _flag else ''}")
            else:
                print(f"▓ ⑧赢家兑现: 无浮盈>30%持仓")
        except Exception as _e8:
            pass
        try:  # ⑥财报季历(2026-09-17接入,Q3窗口) — 招数清单#5
            from scripts.data.earnings_calendar import compact
            print(f'▓ ⑥{compact()}')
        except Exception as _e2:
            print(f'▓ ⑥财报季历: 跳过({_e2})')
        try:  # ⑦低吸候选底部结构背书(2026-09-22·用户设计:筛选后验证增信)
            from scripts.screener.bottom_structure_scanner import load_stock as _ls, detect as _bd
            wl = pd.read_csv('config/watchlist.csv', engine='python', on_bad_lines='skip')
            _tags = []
            for _, wr in wl.head(4).iterrows():
                try:
                    _df = _ls(str(wr['code']).zfill(6))
                    _hit = any(_bd(_df.iloc[:len(_df)-k].reset_index(drop=True)) and
                               _bd(_df.iloc[:len(_df)-k].reset_index(drop=True))['stage'] != '⚪仅分型'
                               for k in range(1, 21))
                    _tags.append(f"{wr['name']}{'✅' if _hit else '❌'}")
                except Exception:
                    continue
            if _tags:
                print(f"▓ ⑦底部结构背书(watchlist前4): {' | '.join(_tags)}")
        except Exception as _e3:
            print(f'▓ ⑦底部结构背书: 跳过({_e3})')
        # ⑨挂单三态核实(2026-11-14审计#5落地: 记忆里的单≠券商的单; 11/13对账证伪教训)
        # 状态默认❌未挂,由用户晨间一句话改成✅已挂/💰成交 — "应已成交"一词废除
        try:
            # 原始行匹配(历史schema混杂:老行action=卖出,新行date开头错位,列名不可靠)
            import datetime as _dtm
            _cutoff = (_dtm.date.today() - _dtm.timedelta(days=7)).isoformat()
            _lines = [l for l in (root / 'config/pre_authorized_orders.csv').read_text(encoding='utf-8').splitlines()
                      if l and not l.startswith('code') and (',买,' in l or ',卖,' in l or ',卖出,' in l)
                      and l[4:5] == '-' and l[:10] >= _cutoff]
            if _lines:
                print(f'▓ ⑨挂单三态核实(默认❌未挂,开盘前逐单回"挂/成交/撤"):')
                for _l in _lines[-5:]:
                    _p = _l.split(',')
                    print(f'   ❌ {_p[2]}({_p[1]}) {_p[3]}@{_p[4]} x{_p[5]}')
        except Exception as _e4:
            print(f'▓ ⑨挂单三态: 跳过({_e4})')
        print('▓' * 66)
    _r3_brief()
except Exception as _e:
    print(f"\n⚠️ 单页晨报跳过: {_e}")
