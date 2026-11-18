#!/usr/bin/env python3
"""持仓做T信号器 — 2026-09-17立 (招数清单#4a)

背景: 用户做T战史(中天+222/轮·1000ETF+528/轮·半导体ETF做到负成本-0.143),全靠盘感。
本脚本把盘感数字化: 盯盘时段(9:30-11:30/13:00-15:00)自动扫持仓,满足条件推T买/T卖。

信号规则(保守,宁缺勿滥):
  触发门槛: 日内振幅 = (高-低)/昨收 > 3%
  T买(低吸接回): 现价回到开盘价附近(≤开盘*1.003) 且 距日内高点回落>1.5%
                  加分: 量比<1(缩量回落=卖压弱)
  T卖(冲高减):   现价>昨收*1.02 且 贴日内高点(距高<0.3%)
  垫子标注: 浮盈>50% = "提款机"优先做T对象; 浮亏标的禁T卖(避免把止损做成T)

用法:
  PYTHONPATH=. python scripts/screener/intraday_t_signal.py            # 盘中跑(有信号)
  PYTHONPATH=. python scripts/screener/intraday_t_signal.py --selftest # 字段自检(盘后)
数据: 腾讯实时(qt.gtimg.cn); 字段映射经parquet自检, 见 --selftest
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os_chdir = None
import os
os.chdir(PROJECT_ROOT)  # config/相对路径锚定

STORE = PROJECT_ROOT / 'data/store/kline'
# 腾讯字段位(f3现价已验证/f32涨跌幅已验证; f33高/f34低需selftest确认)
F_CUR, F_PREV, F_OPEN = 3, 4, 5
F_HIGH, F_LOW, F_VR = 33, 34, 49


def _prefix(code):
    code = str(code).zfill(6)
    return ('sh' if code[0] in '569' or code[:2] in ('51', '56', '58') else 'sz') + code


def fetch_quotes(codes):
    q = ','.join(_prefix(c) for c in codes)
    out = subprocess.run(['bash', '-c',
                          f"curl -s 'http://qt.gtimg.cn/q={q}' | iconv -f gbk -t utf8"],
                         capture_output=True, text=True, timeout=15).stdout
    res = {}
    for line in out.split('\n'):
        if '=' not in line:
            continue
        f = line.split('=')[1].strip('";').split('~')
        if len(f) > 50:
            code = f[2].zfill(6)
            try:
                res[code] = {
                    'name': f[1], 'cur': float(f[F_CUR]), 'prev': float(f[F_PREV]),
                    'open': float(f[F_OPEN]), 'high': float(f[F_HIGH]), 'low': float(f[F_LOW]),
                    'vol_ratio': float(f[F_VR]) if f[F_VR] else None,
                }
            except (ValueError, IndexError):
                continue
    return res


def load_positions():
    """持仓(shares>0, 含ETF——ETF是做T主力)"""
    out = []
    try:
        pf = pd.read_excel('config/portfolio.xlsx')
        for _, r in pf.iterrows():
            try:
                if int(r['shares']) > 0:
                    out.append({'code': str(r['code']).zfill(6), 'name': str(r['name']),
                                'shares': int(r['shares']), 'cost': float(r['avg_cost'])})
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f'⚠️ 读持仓失败: {e}')
    return out


def is_trading_hours(now=None):
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return (930 <= hm <= 1130) or (1300 <= hm <= 1500)


def selftest():
    """字段自检: 腾讯f33/f34(高低) vs parquet最后交易日高低"""
    print('=== 字段自检: 腾讯高/低 vs parquet ===')
    pos = load_positions()
    if not pos:
        print('无持仓');  return
    codes = [p['code'] for p in pos][:6]
    quotes = fetch_quotes(codes)
    ok = 0
    for c in codes:
        p = STORE / c / 'D.parquet'
        if not p.exists() or c not in quotes:
            print(f'{c}: 无parquet或行情,跳过');  continue
        df = pd.read_parquet(p)
        dc = next(x for x in ['date', 'datetime', 'day', 'trade_date'] if x in df.columns)
        df['d'] = pd.to_datetime(df[dc]).dt.strftime('%Y%m%d')
        last = df.sort_values('d').iloc[-1]
        q = quotes[c]
        match = abs(float(last['high']) - q['high']) < 0.01 and abs(float(last['low']) - q['low']) < 0.01
        ok += match
        print(f"{c} {q['name'][:6]:<8} 腾讯高{q['high']:.2f}/低{q['low']:.2f} "
              f"vs parquet高{float(last['high']):.2f}/低{float(last['low']):.2f} "
              f"{'✅' if match else '❌需修字段'} (parquet日{last['d']})")
    print(f'\n通过 {ok}/{len(codes)} — 全✅则字段映射正确')


def main():
    now = datetime.now()
    live = is_trading_hours(now)
    pos = load_positions()
    if not pos:
        return
    print(f"{'🔴盘中实时' if live else '⚪非盘中(信号无效,仅列候选)'} | 持仓做T信号器 @ {now:%H:%M}")

    quotes = fetch_quotes([p['code'] for p in pos])
    sigs = []
    print(f"\n{'代码':<8}{'名称':<10}{'现价':>7}{'振幅%':>6}{'距高%':>7}{'量比':>5}  垫子   状态")
    print('-' * 66)
    for p in pos:
        q = quotes.get(p['code'])
        if not q or not q['prev']:
            continue
        amp = (q['high'] - q['low']) / q['prev'] * 100
        from_high = (q['cur'] - q['high']) / q['prev'] * 100   # 负=在下 方
        cushion = (q['cur'] / p['cost'] - 1) * 100 if p['cost'] else 0
        vr = q['vol_ratio'] if q['vol_ratio'] else 0
        tag = ''
        if live and amp > 3:
            shrink = '·缩量✓' if 0 < vr < 1 else ''
            # T买: 回到开盘附近 + 从高点回落>1.5%
            if q['cur'] <= q['open'] * 1.003 and (q['high'] - q['cur']) / q['prev'] > 0.015:
                tag = f'🟢T买候选{shrink}(回落至开盘{q["open"]:.2f}附近)'
            # T卖: 涨超2% + 贴日高 + 有垫子(浮亏不做T卖,那是止损不是T)
            elif q['cur'] >= q['prev'] * 1.02 and (q['high'] - q['cur']) / q['high'] < 0.003 and cushion > 0:
                tag = '🔴T卖候选(贴日高,冲高>2%)'
        elif not live and amp > 3:
            tag = '(复盘:今日振幅达标)'
        cushion_s = f'{cushing:+.0f}%' if (cushing := cushion) is not None else ''
        mark = '💰提款机' if cushion > 50 else ('⚠️浮亏禁T卖' if cushion < 0 else '')
        print(f"{p['code']:<8}{q['name'][:8]:<10}{q['cur']:>7.2f}{amp:>6.1f}{from_high:>7.1f}{vr:>5.1f}"
              f" {cushion_s:>6} {mark} {tag}")
        if tag and live:
            sigs.append((p['code'], q['name'], tag))

    if live and sigs:
        print(f'\n⚡ {len(sigs)}个信号 — T纪律: 先卖后买等量·单轮目标≥1%·当日必平仓')
        try:
            log = pd.DataFrame([{'time': now.strftime('%Y-%m-%d %H:%M'), 'code': c, 'name': n, 'signal': s}
                                for c, n, s in sigs])
            log.to_csv('reports/t_signal_log.csv', mode='a', header=not Path('reports/t_signal_log.csv').exists(),
                       index=False)
        except Exception:
            pass
    elif live:
        print('\n(无信号 — 振幅<3%或位置不符,不动就是最好的T)')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main()
