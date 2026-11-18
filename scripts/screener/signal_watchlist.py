#!/usr/bin/env python3
"""信号验证清单管理器。

长期跟踪扫描信号的实战表现，验证策略有效性。

数据结构: data/verification/watchlist.csv
  - code, name, industry
  - scan_date, scan_price
  - strategy: 使用的策略名(如 v3_channel_backtest)
  - v3_score, total_score, ch_pct, b_signal, divergence
  - tag: 信号标签(趋势加速/底部反转-关注/底部反转-确认/顶部卖出/通道底部)
  - status: watching/confirmed/expired/closed
  - notes

每日跟踪: data/verification/tracking/
  - {date}.csv: 当日所有信号标的的涨跌情况

用法:
    python scripts/screener/signal_watchlist.py add    # 从最新回测结果添加高分信号
    python scripts/screener/signal_watchlist.py track   # 每日跟踪
    python scripts/screener/signal_watchlist.py report   # 输出验证报告
"""
from __future__ import annotations
import sys
import os
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT / 'data' / 'verification'
WATCHLIST = DATA_DIR / 'watchlist.csv'
TRACKING_DIR = DATA_DIR / 'tracking'


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)


def load_watchlist() -> pd.DataFrame:
    if WATCHLIST.exists():
        return pd.read_csv(WATCHLIST)
    return pd.DataFrame(columns=[
        'code', 'name', 'industry',
        'scan_date', 'scan_price',
        'strategy', 'v3_score', 'total_score', 'ch_pct',
        'b_signal', 'divergence',
        'tag', 'status', 'notes',
    ])


def save_watchlist(df: pd.DataFrame):
    ensure_dirs()
    df.to_csv(WATCHLIST, index=False)


def get_realtime_prices(codes: list[str]) -> dict:
    """批量获取实时价格（腾讯接口）。"""
    prefix_map = {'6': 'sh', '0': 'sz', '3': 'sz', '5': 'sh'}
    items = []
    for code in codes:
        c = str(code).zfill(6)
        p = prefix_map.get(c[0], 'sh')
        items.append(f'{p}{c}')
    url = 'https://qt.gtimg.cn/q=' + ','.join(items)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode('gbk')
        prices = {}
        for line in data.strip().split(';'):
            if '~' not in line.strip():
                continue
            parts = line.strip().split('~')
            if len(parts) < 35 or not parts[2] or not parts[3]:
                continue
            code = str(parts[2]).zfill(6)
            prices[code] = {
                'name': parts[1],
                'price': float(parts[3]),
                'pct': float(parts[32]),
                'high': float(parts[33]),
                'low': float(parts[34]),
            }
        return prices
    except Exception:
        return {}


def add_from_backtest():
    """从最新回测结果中添加高分信号到验证清单。"""
    ensure_dirs()
    backtest_file = PROJECT / 'data' / 'v3_channel_backtest.csv'
    if not backtest_file.exists():
        print("回测文件不存在，先运行 v3_channel_backtest.py")
        return

    df_bt = pd.read_csv(backtest_file)
    today = datetime.now().strftime('%Y-%m-%d')

    # 加载股票列表（名称+行业）
    stock_list_file = PROJECT / 'data' / 'store' / 'stock_list.parquet'
    name_map = {}
    industry_map = {}
    if stock_list_file.exists():
        sl = pd.read_parquet(stock_list_file)
        name_map = dict(zip(sl['symbol'], sl['name']))
        industry_map = dict(zip(sl['symbol'], sl['industry']))

    wl = load_watchlist()
    existing = set(zip(wl['code'], wl['scan_date'])) if len(wl) > 0 else set()

    # 策略1: V3≥2 + 通道极底部(0-20%) + B级 — 最佳组合
    mask_best = (df_bt['v3'] >= 2.0) & (df_bt['ch_pct'] < 20) & df_bt['b_signal']
    # 策略2: 通道极底部 + B级
    mask_channel_b = (df_bt['ch_pct'] < 20) & df_bt['b_signal'] & ~mask_best
    # 策略3: V3≥2 + 通道极底部（大样本验证）
    mask_v3_ch = (df_bt['v3'] >= 2.0) & (df_bt['ch_pct'] < 20) & ~mask_best & ~mask_channel_b

    strategies = [
        ('V3≥2+通道底部+B级', mask_best, '底部反转-确认买入'),
        ('通道底部+B级', mask_channel_b, '底部反转-关注'),
        ('V3≥2+通道底部', mask_v3_ch, '通道底部'),
    ]

    added = 0
    for strat_name, mask, tag in strategies:
        sub = df_bt[mask]
        # 按融合分排序，取最新的信号
        sub = sub.sort_values('total', ascending=False)
        # 去重：每个code只取最高分
        seen_codes = set()
        for _, row in sub.iterrows():
            code_str = str(row['code']).zfill(6)
            if code_str in seen_codes or code_str.startswith(('51', '15')):
                continue
            seen_codes.add(code_str)
            if (code_str, today) in existing:
                continue

            entry = {
                'code': code_str,
                'name': name_map.get(code_str, code_str),
                'industry': industry_map.get(code_str, ''),
                'scan_date': today,
                'scan_price': 0,  # 将在track时填充
                'strategy': strat_name,
                'v3_score': round(row['v3'], 2),
                'total_score': round(row['total'], 2),
                'ch_pct': round(row['ch_pct'], 1) if not np.isnan(row['ch_pct']) else '',
                'b_signal': bool(row['b_signal']),
                'divergence': bool(row['divergence']),
                'tag': tag,
                'status': 'watching',
                'notes': '',
            }
            wl = pd.concat([wl, pd.DataFrame([entry])], ignore_index=True)
            existing.add((code_str, today))
            added += 1

            if added >= 30:  # 限制每次最多30只
                break
        if added >= 30:
            break

    # 填充实时价格
    if added > 0:
        codes = wl[wl['scan_date'] == today]['code'].tolist()
        prices = get_realtime_prices(codes)
        for i, row in wl[wl['scan_date'] == today].iterrows():
            if row['scan_price'] == 0 and row['code'] in prices:
                wl.at[i, 'scan_price'] = prices[row['code']]['price']

    save_watchlist(wl)
    print(f"添加了 {added} 只标的到验证清单")
    print(f"验证清单总数: {len(wl)}")


def daily_track():
    """每日跟踪验证清单中的标的。"""
    ensure_dirs()
    wl = load_watchlist()
    if len(wl) == 0:
        print("验证清单为空")
        return

    today = datetime.now().strftime('%Y-%m-%d')

    # 获取实时价格
    active = wl[wl['status'].isin(['watching', 'confirmed'])]
    if len(active) == 0:
        print("没有活跃的验证标的")
        return

    codes = active['code'].tolist()
    prices = get_realtime_prices(codes)

    tracking = []
    for _, row in active.iterrows():
        code = str(row['code']).zfill(6)
        if code not in prices:
            continue
        p = prices[code]
        scan_price = row['scan_price']
        if scan_price <= 0:
            scan_price = p['price']
            wl.loc[wl['code'] == code, 'scan_price'] = scan_price

        ret_pct = (p['price'] / scan_price - 1) * 100
        days_since = (datetime.now() - datetime.strptime(row['scan_date'], '%Y-%m-%d')).days

        tracking.append({
            'date': today,
            'code': code,
            'name': p.get('name', row['name']),
            'price': p['price'],
            'pct_today': p['pct'],
            'scan_price': scan_price,
            'scan_date': row['scan_date'],
            'ret_from_scan': round(ret_pct, 2),
            'days_since_scan': days_since,
            'strategy': row['strategy'],
            'tag': row['tag'],
            'v3_score': row['v3_score'],
            'ch_pct': row['ch_pct'],
        })

    if tracking:
        track_file = TRACKING_DIR / f'{today}.csv'
        pd.DataFrame(tracking).to_csv(track_file, index=False)
        print(f"跟踪数据已保存: {track_file} ({len(tracking)}只)")

    save_watchlist(wl)

    # 输出简要报告
    if tracking:
        df_t = pd.DataFrame(tracking)
        print(f"\n📊 验证清单跟踪 {today}")
        print("=" * 80)
        for _, r in df_t.iterrows():
            arrow = '📈' if r['ret_from_scan'] > 0 else ('📉' if r['ret_from_scan'] < -3 else '➡️')
            print(f"  {arrow} {r['name']}({r['code']}) | {r['price']} | "
                  f"自扫描{r['days_since_scan']}日 {r['ret_from_scan']:+.1f}% | {r['tag']}")


def generate_report():
    """生成验证报告：策略胜率统计。"""
    tracking_files = sorted(TRACKING_DIR.glob('*.csv'))
    if not tracking_files:
        print("无跟踪数据，先运行 daily_track")
        return

    all_tracking = pd.concat([pd.read_csv(f) for f in tracking_files], ignore_index=True)
    wl = load_watchlist()

    print("=" * 70)
    print("📊 信号验证报告")
    print("=" * 70)

    # 按策略分组
    for strategy in all_tracking['strategy'].unique():
        sub = all_tracking[all_tracking['strategy'] == strategy]
        # 取最新一天的数据
        latest_date = sub['date'].max()
        latest = sub[sub['date'] == latest_date]

        # 按持有天数分档
        for days_min, days_max, label in [(0, 5, '5日内'), (5, 11, '5-10日'), (11, 21, '10-20日'), (21, 999, '20日+')]:
            bucket = latest[(latest['days_since_scan'] >= days_min) & (latest['days_since_scan'] < days_max)]
            if len(bucket) == 0:
                continue
            win = (bucket['ret_from_scan'] > 0).sum()
            wr = win / len(bucket) * 100
            avg = bucket['ret_from_scan'].mean()
            print(f"  {strategy} | {label}: {len(bucket)}只 | 胜率{wr:.0f}% | 平均{avg:+.1f}%")
        print()

    # 标签分组
    print("── 按信号标签 ──")
    for tag in all_tracking['tag'].unique():
        sub = all_tracking[all_tracking['tag'] == tag]
        latest = sub[sub['date'] == sub['date'].max()]
        win = (latest['ret_from_scan'] > 0).sum()
        wr = win / len(latest) * 100 if len(latest) > 0 else 0
        avg = latest['ret_from_scan'].mean()
        print(f"  {tag}: {len(latest)}只 | 胜率{wr:.0f}% | 平均{avg:+.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='信号验证清单管理')
    parser.add_argument('action', choices=['add', 'track', 'report'], help='操作')
    args = parser.parse_args()

    if args.action == 'add':
        add_from_backtest()
    elif args.action == 'track':
        daily_track()
    elif args.action == 'report':
        generate_report()
