#!/usr/bin/env python3
"""采集后核查 — 2026-09-30立（用户规矩: 采集完要回测,特别是持仓和关注列表）。

三个教训的直接产物:
  ① 9/29超导parquet停在9/28,全量扫描用过期数据报假止损(数据新鲜度必须先于扫描)
  ② 9/30深科技破线,豁免条款(底背离str≥0.5)要显式实算,不能停留在预案文字里
  ③ watchlist买区在自由文本备注里,收盘后没有机器自动对照

流程位置: daily_scan_all第一位(17:00采集后、全扫描前)。任何一项不通过→大字警告。

检查内容:
  1. 数据新鲜度: 持仓+watchlist活跃股的parquet末日期 == 基准交易日(上证指数parquet)
  2. 失效线状态: config/invalidation_lines.csv 逐条对照最新收盘(破线→标注+背离豁免待办)
  3. watchlist区距: 最新收盘+备注原文(买区判断供人工)
用法:
  PYTHONPATH=. python scripts/research/post_collection_check.py [--fix]
  --fix: 对滞后标的自动从tushare补当日K线
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)
import pandas as pd
import numpy as np

BENCH = 'data/store/kline/000001/D.parquet'   # 上证指数=交易日基准


def _last_date(code):
    p = f'data/store/kline/{code}/D.parquet'
    if not os.path.exists(p):
        return None, None
    df = pd.read_parquet(p)
    dcol = [c for c in df.columns if 'date' in c.lower()]
    if not dcol:
        return None, None
    df = df.dropna(subset=['close']).sort_values(dcol[0])
    return str(df[dcol[0]].iloc[-1]), float(df['close'].iloc[-1])


def fix_from_tushare(code, date):
    """对滞后标的补一根日K(tushare daily)。code纯数字。"""
    try:
        from scripts.core.tushare_client import TushareClient
        pro = TushareClient().pro
        suffix = '.SH' if code[0] in '569' else ('.SZ' if code[0] in '03' else '.BJ')
        if code.startswith(('159','15')): suffix = '.SZ'
        if code.startswith(('51','56','58')): suffix = '.SH'
        d = pro.daily(ts_code=code+suffix, start_date=date, end_date=date)
        if d is None or len(d) == 0:
            return False
        p = f'data/store/kline/{code}/D.parquet'
        df = pd.read_parquet(p)
        if 'ts_code' not in df.columns:
            df['ts_code'] = code+suffix
        r = d.iloc[0]
        new = pd.DataFrame([{'date': str(r['trade_date']), 'open': float(r['open']),
                             'high': float(r['high']), 'low': float(r['low']),
                             'close': float(r['close']), 'volume': float(r.get('vol') or 0),  # 手:对齐parquet原口径(2026-10-01修,勿×100)
                             'amount': float('nan'),
                             'ts_code': code+suffix}])
        df = pd.concat([df.dropna(subset=['close']), new], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in df.columns if 'date' in c.lower()][0], keep='last')
        df.to_parquet(p)
        return True
    except Exception as e:
        print(f'    补数失败{code}: {e}')
        return False


def main(fix=False):
    bench_date, _ = _last_date('000001')
    print(f'🔬 采集后核查 | 基准交易日={bench_date} (上证parquet)')
    problems = []

    # ── 1. 新鲜度: 持仓+watchlist ──
    universe = []
    pf = pd.read_excel('config/portfolio.xlsx')
    for _, r in pf.iterrows():
        code = str(r['code']).zfill(6)
        try:
            if code != 'nan' and int(r.get('shares', 0) or 0) > 0:
                universe.append((code, str(r['name']), '持仓'))
        except (ValueError, TypeError):
            continue
    try:
        wl = pd.read_csv('config/watchlist.csv', dtype={'code': str})
        wl['code'] = wl['code'].astype(str).str.zfill(6)
        for _, r in wl.tail(30).iterrows():
            st = str(r.get('status', ''))
            if 'expired' in st or '成交' in st:
                continue
            if (r['code'], str(r['name']), '持仓') not in universe and r['code'] != 'nan':
                universe.append((r['code'], str(r['name']), '关注'))
    except Exception:
        pass

    stale = []
    seen = set()
    print(f'\n[1] 数据新鲜度 ({len(universe)}只: 持仓+关注)')
    for code, name, tag in universe:
        if code in seen or code == 'nan':
            continue
        seen.add(code)
        d, c = _last_date(code)
        if d is None:
            print(f'  ⚠️ {name}({code}) 无parquet')
            continue
        if d != bench_date:
            stale.append((code, name, d))
    if stale:
        for code, name, d in stale:
            print(f'  🚨 {name}({code}) 停在{d} ≠ {bench_date}')
            if fix:
                ok = fix_from_tushare(code, bench_date)
                print(f'     → {"已补✓" if ok else "补数失败,须人工"}')
        problems.append(f'{len(stale)}只数据滞后')
    else:
        print('  ✅ 全部新鲜')

    # ── 2. 失效线 ──
    print('\n[2] 失效线状态 (config/invalidation_lines.csv)')
    try:
        lines = pd.read_csv('config/invalidation_lines.csv', dtype={'code': str})
        lines['code'] = lines['code'].str.zfill(6)
        broke = []
        for _, r in lines.iterrows():
            d, c = _last_date(r['code'])
            if c is None:
                continue
            gap = (c / r['line'] - 1) * 100
            mark = '✓守住' if c >= r['line'] else '🚨破线'
            if c < r['line']:
                broke.append((r['name'], r['code']))
            print(f'  {mark} {r["name"]:8s} 收{c:<8.3f} vs 线{r["line"]:<8.3f} ({gap:+.1f}%) [{d}]')
        if broke:
            print(f'  ⚠️ 破线豁免待办(7/15条款): ' + '、'.join(f'{n}({c})' for n, c in broke))
            print(f'     命令: PYTHONPATH=. python scripts/screener/divergence_scanner.py 或手工实算日线底背离str')
            problems.append(f'{len(broke)}只破线待豁免判定')
    except FileNotFoundError:
        print('  (无invalidation_lines.csv,跳过)')

    # ── 3. watchlist区距速览(2026-10-01升级:读结构化zones CSV自动算距) ──
    print('\n[3] 三批名单区距(结构化zones,收盘价自动对照)')
    try:
        zones = pd.read_csv('config/watchlist_zones.csv', dtype={'code': str})
        zones['code'] = zones['code'].str.zfill(6)
        for _, z in zones.iterrows():
            if str(z.get('status', 'active')) != 'active':
                continue
            d, c = _last_date(z['code'])
            if c is None:
                print(f'  批{z["batch"]} {z["name"]:8s}({z["code"]}) 无parquet')
                continue
            if pd.notna(z.get('zone_low')) and pd.notna(z.get('zone_high')):
                lo, hi = float(z['zone_low']), float(z['zone_high'])
                if c < lo:
                    pos = f'距区低{lo and (lo/c-1)*100:+.1f}%⬆️'
                elif c > hi:
                    pos = f'出区上方{(c/hi-1)*100:+.1f}%'
                else:
                    pos = '🎯区内'
                print(f'  批{z["batch"]} {z["name"]:8s}({z["code"]}) 收{c:<8.3f} 区[{lo}-{hi}] {pos} 止损{z["stop"]}')
            else:
                print(f'  批{z["batch"]} {z["name"]:8s}({z["code"]}) 收{c:<8.3f} 【{z["trigger_note"]}】止损{z["stop"]}')
    except FileNotFoundError:
        print('  (无watchlist_zones.csv,回退自由文本模式)')
        for code, name, tag in universe:
            if tag != '关注':
                continue
            d, c = _last_date(code)
            if c:
                print(f'  {name:8s}({code}) 收{c:<8.2f} [{d}]')

    print('\n' + ('🚨🚨 核查未通过: ' + '; '.join(problems) + ' —— 先修数据再扫描!' if problems
                  else '✅ 核查通过,可以跑全量扫描'))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main(fix='--fix' in sys.argv))
