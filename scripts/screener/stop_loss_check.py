#!/usr/bin/env python3
"""持仓止损检查器 — 破位即走(纪律) + 底背离过滤(不砍地板)。

读portfolio.xlsx持仓, 算MA20+成本-8%+移动止损 + 底背离+缩量过滤。
关键: 破MA20时若底背离+缩量=可能见底, 不机械砍(联化教训, 砍地板亏)。

止损规则(加背离过滤):
  亏损+破MA20+无背离/放量 = 🔴趋势止损(破位即走)
  破成本-8%+无背离/放量  = 🔴限亏止损
  亏损破位+底背离+缩量   = 🟠见底迹象(别砍地板, 持有观察)
  盈利多+破MA20         = 🟡移动止损(保护利润)
  否则                   = 🟢持有

复盘#6: 长华-9.9%没止损(教训) vs 联化底背离+缩量(砍地板亏) → 止损要看技术, 不能纯机械。

用法: cd stock && PYTHONPATH=. python scripts/screener/stop_loss_check.py
"""
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from scripts.analyzers.divergence_detector import DivergenceDetector

# 目录认环境变量（与 web/backend/settings.py 同一套约定：STOCK_*_DIR 可重定向）
STORE = Path(os.environ.get("STOCK_DATA_DIR", str(PROJECT_ROOT / "data"))) / "store/kline"
PORT = Path(os.environ.get("STOCK_CONFIG_DIR", str(PROJECT_ROOT / "config"))) / "portfolio.xlsx"
DET = DivergenceDetector()


def check_stop_loss():
    if not PORT.exists():   # 尚未配置持仓（如刚 clone 零数据起服务）→ 无止损检查
        return []
    port = pd.read_excel(PORT)
    port['code'] = port['code'].astype(str).str.zfill(6)
    rows = []
    for _, r in port.iterrows():
        code = r['code']; cost = float(r['avg_cost']); sh = int(r['shares'])
        if sh == 0:
            continue
        p = STORE / f'{code}/D.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        dc = [c for c in ['date', 'trade_date'] if c in df.columns][0]
        df = df.sort_values(dc)
        c = df['close'].astype(float)
        if len(c) < 20:
            continue
        cur = float(c.iloc[-1])
        ma20 = float(c.rolling(20).mean().iloc[-1])
        pnl = (cur / cost - 1) * 100
        cost_loss = cost * 0.92  # -8%限亏
        # 量能(近5日/20日) — 处理NaN: 有效值不足则无法确认缩量(机器人ETF教训: vol缺失致误判🔴)
        vol = df['volume'].astype(float) if 'volume' in df.columns else None
        v_ratio = 1.0; vol_ok = False
        if vol is not None and len(vol) >= 20:
            recent = vol.iloc[-5:].dropna(); base = vol.iloc[-20:].dropna()
            if len(recent) >= 3 and len(base) >= 10 and base.mean() > 0:
                v_ratio = float(recent.mean() / base.mean()); vol_ok = True
        shrink = vol_ok and v_ratio < 0.9  # 缩量(仅数据充足时判)
        # 底背离(只认strength≥0.5的强背离, 过滤弱信号)
        try:
            sig = DET.detect_divergence(df.tail(60), period='D')
            div = bool(sig.detected and sig.div_type == 'bottom' and getattr(sig, 'strength', 0) >= 0.5)
        except Exception:
            div = False
        see_bottom = div and shrink  # 底背离+缩量=强见底
        # 止损判断(背离过滤: 有强底背离就不机械砍地板, 联化/航天/机器人ETF教训)
        if cur < ma20 and pnl < 0:
            if see_bottom:
                trigger = f'🟠破MA20但底背离+缩量(强见底,别砍地板)'
            elif div:  # 强底背离但量能未确认/缺失 → 倾向见底持有
                tag = f'量比{v_ratio:.2f}未缩量,观察' if vol_ok else '量能数据缺失,倾向见底'
                trigger = f'🟠破MA20但强底背离({tag})'
            else:
                trigger = f'🔴破MA20({ma20:.2f})趋势止损'
        elif cur < cost_loss:
            if see_bottom:
                trigger = f'🟠破成本-8%但底背离+缩量(见底迹象)'
            elif div:
                trigger = f'🟠破成本-8%但强底背离(量能缺失,倾向见底)'
            else:
                trigger = f'🔴破成本-8%({cost_loss:.2f})限亏止损'
        elif cur < ma20 and pnl > 20:
            trigger = f'🟡破MA20({ma20:.2f})移动止损(保护利润)'
        else:
            trigger = '🟢持有'
        rows.append({'code': code, 'name': r['name'], 'cost': cost, 'cur': cur, 'pnl': pnl,
                     'ma20': ma20, 'v_ratio': v_ratio, 'div': div, 'trigger': trigger, 'val': cur * sh})
    return rows


def main():
    rows = check_stop_loss()
    red = [r for r in rows if '🔴' in r['trigger']]
    yellow = [r for r in rows if '🟡' in r['trigger']]
    orange = [r for r in rows if '🟠' in r['trigger']]
    print(f"📊 持仓止损检查 | 破位即走 + 底背离过滤(不砍地板) | 数据基于最新parquet")
    print(f"   {'🚨'+str(len(red))+'只🔴真止损(无背离/放量,破位即走)' if red else '✅无🔴真止损'} | {len(orange)}只🟠见底(底背离+缩量,别砍) | {len(yellow)}只🟡移动止损\n")
    print(f"{'代码':<7}{'名称':<10}{'成本':>7}{'现价':>7}{'盈亏':>7}{'MA20':>7}{'量比':>6}  状态")
    print('-' * 80)
    for r in sorted(rows, key=lambda x: -x['pnl']):
        print(f"{r['code']:<7}{str(r['name'])[:9]:<10}{r['cost']:>7.2f}{r['cur']:>7.2f}{r['pnl']:>+6.1f}%{r['ma20']:>7.2f}{r['v_ratio']:>5.2f}x  {r['trigger']}")
    if orange:
        print(f"\n🟠 见底迹象{len(orange)}只(底背离+缩量) → 别砍地板, 持有等反弹(联化教训, 砍了+6.6%反弹就亏):")
        for r in orange:
            print(f"   {r['name']}({r['code']}) 现价{r['cur']:.2f} {r['trigger']}")
    if red:
        print(f"\n🚨 真止损{len(red)}只(无背离/放量,破位即走):")
        for r in red:
            print(f"   {r['name']}({r['code']}) 现价{r['cur']:.2f} {r['trigger']}")
    print(f"\n💡 🔴真止损→次日执行; 🟠见底→持有等反弹(底背离+缩量别砍地板); 🟡移动止损→减半保护利润")


if __name__ == '__main__':
    main()
