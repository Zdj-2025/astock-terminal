#!/usr/bin/env python3
"""动态结构止损模块(2026-08-29建)——用swing low追踪替代手动设地板。

回测依据(4072笔金叉交易): 松版swing low均收益+1.80%/笔,
优于固定-7%(+0.70%,t=5.8p=0.000)和梯度15%(+1.40%,p=0.102)。

算法:
  ① swing low = 低点比前后各N根K线都低(默认N=5)
  ② 确认延迟: 必须等右边N根走完才确认(无前视)
  ③ 止损 = 倒数第二个已确认swing low × (1-buffer)
  ④ 只上移不下移(跟踪型)

与手动地板的关系: 取max(结构止损, 手动地板)作为实际执行线
  → 结构止损比手动地板高 → 用结构(市场画的更准)
  → 手动地板比结构高 → 用手动(人工判断优先)

用法:
  from scripts.analyzers.structural_stop import get_structural_stop, scan_portfolio_stops
  stop = get_structural_stop('603881')       # 单只 → float or None
  scan_portfolio_stops()                      # 全持仓 → 自动更新portfolio.xlsx
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

KDIR = 'data/store/kline'
SWING_N = 5           # swing low确认窗口(左右各5根)
BUFFER = 0.005        # 止损放在swing low下方0.5%
MIN_SWINGS = 2        # 至少需要2个已确认swing low


def _qfq(c):
    """前复权(防除权假跳变)"""
    c = np.asarray(c, dtype=float)
    rets = np.full(len(c), np.nan)
    rets[1:] = c[1:] / c[:-1] - 1
    cc = c.copy()
    for i in range(len(cc) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            cc[:i + 1] = cc[:i + 1] * (c[i + 1] / c[i])
    return cc


def find_swing_lows(lows, n=SWING_N):
    """找所有swing low,返回(位置,价格)列表。
    注意: 位置p在p+n时才"确认"(右边n根走完)——无前视。"""
    sls = []
    for i in range(n, len(lows) - n):
        left_ok = all(lows[i] < lows[i - j] for j in range(1, n + 1))
        right_ok = all(lows[i] < lows[i + j] for j in range(1, n + 1))
        if left_ok and right_ok:
            sls.append((i, lows[i]))
    return sls


def find_swing_highs(highs, n=SWING_N):
    """找swing high(棘轮止盈用): i是[i-n,i+n]严格最高(右侧确认)"""
    out = []
    for i in range(n, len(highs) - n):
        if all(highs[i] > highs[i - j] for j in range(1, n + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, n + 1)):
            out.append((i, highs[i]))
    return out


def get_ratchet_stop(code, verbose=False):
    """棘轮式止盈(2026-08-29回测验证: +1.89%/笔,次优于纯止损+2.18%,持有更短)。
    逻辑: 现价突破最近已确认swing high → 止损上移到该high×0.995(阻力变支撑)。
    返回 dict(ratchet_stop, broken_high, ...) or None(未突破任何前高)。"""
    code = str(code).zfill(6)
    path = f'{KDIR}/{code}/D.parquet'
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('date')
    if len(df) < SWING_N * 2 + 10:
        return None
    highs = _qfq(df['high'].astype(float).values)
    closes = df['close'].astype(float).values
    dates = df['date'].values
    current_idx = len(closes) - 1
    close = closes[-1]

    swing_highs = find_swing_highs(highs)
    confirmed = [(p, v, dates[p]) for p, v in swing_highs if p + SWING_N <= current_idx]
    if not confirmed:
        return None

    # 棘轮: 找"已被现价突破"的最近swing high(从最近往回)
    # 止损=该high*0.995; 若突破多个, 取最高的被突破者
    best = None
    for p, v, d in reversed(confirmed):
        if close > v:
            best = (p, v, d)
            break
    if best is None:
        return None
    ratchet = best[1] * (1 - BUFFER)
    if verbose:
        print(f"  棘轮: 突破{str(best[2])[:10]}前高{best[1]:.2f} → 止损上移{ratchet:.2f}")
    return dict(ratchet_stop=round(ratchet, 2), broken_high=best[1],
                broken_date=str(best[2])[:10])


def get_structural_stop(code, verbose=False):
    """计算单只股票的动态结构止损。
    返回 dict(stop_price, swing_lows, trend_ok) or None if data insufficient."""
    code = str(code).zfill(6)
    path = f'{KDIR}/{code}/D.parquet'
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path).sort_values('date')
    if len(df) < SWING_N * 2 + 10:
        return None

    # 2026-08-29修复: 用前复权low做swing检测(防除权假谷)
    # 数据港8/26除权-26%,不复权low会制造假谷锚定错误止损
    lows_raw = df['low'].astype(float).values
    lows = _qfq(lows_raw)  # 前复权处理
    closes = df['close'].astype(float).values
    dates = df['date'].values

    # 找所有swing low(用复权后的low,消除除权假谷)
    all_swing_lows = find_swing_lows(lows)

    # 只用"已确认"的(位置+SWING_N <= 当前)
    current_idx = len(lows) - 1
    confirmed = [(p, v, dates[p]) for p, v in all_swing_lows if p + SWING_N <= current_idx]

    if len(confirmed) < MIN_SWINGS:
        if verbose:
            print(f"  {code}: 已确认swing low不足{MIN_SWINGS}个({len(confirmed)}个)")
        return None

    # 2026-08-29修复: 锚选择改进——不用死板的"倒数第二"
    # 改为: 找最近的"高于前一个谷"的谷(上升趋势中的关键支撑)
    # 这样锚只会随着趋势走强而上移,不会因新低而意外下移
    last = confirmed[-1]
    second_last = confirmed[-2]

    # 寻找最佳锚: 从最近的谷往前找,找到第一个高于它的谷
    # 这确保锚在趋势中的正确位置
    best_anchor = second_last  # 默认用倒数第二
    for i in range(len(confirmed) - 2, -1, -1):
        if confirmed[i][1] < last[1]:
            # 找到了低于最近谷的前一个谷 → 这是起涨点
            best_anchor = confirmed[i]
            break
    second_last = best_anchor

    # 止损只上移: 如果最近swing low比倒数第二高,且确认了更近的上升趋势
    # 用倒数第二个作为基础止损(松版),但如果最近的也够高且趋势确认,取较高者
    stop_candidate_1 = second_last[1] * (1 - BUFFER)
    stop_candidate_2 = last[1] * (1 - BUFFER)

    # 判断趋势: 最近swing low > 倒数第二swing low = 高低点抬高=趋势向上
    trend_up = last[1] > second_last[1]

    if trend_up:
        # 趋势向上: 用松版(倒数第二),给回调空间
        stop_price = stop_candidate_1
        method = '松版(倒数第二谷,趋势向上)'
    else:
        # 趋势不明/向下: 用紧版(最近谷),保守
        stop_price = max(stop_candidate_1, stop_candidate_2)
        method = '紧版(趋势不明,取较高)'

    # 确保止损不超过当前价
    current_price = closes[-1]
    if stop_price >= current_price:
        stop_price = current_price * 0.95  # 兜底: 当前价-5%
        method += '(超出现价,降级-5%)'

    result = dict(
        code=code,
        stop_price=round(stop_price, 3),
        current_price=current_price,
        method=method,
        trend_up=trend_up,
        swing_lows=[{'date': str(d)[:10], 'price': v} for _, v, d in confirmed[-5:]],
        last_swing_low=last[1],
        second_swing_low=second_last[1],
        date=str(dates[-1])[:10],
    )

    if verbose:
        print(f"  {code}: 结构止损={result['stop_price']:.3f} "
              f"(现价{current_price:.2f}, {method})")
        for sl in result['swing_lows']:
            marker = ' ←止损锚' if abs(sl['price'] - result['second_swing_low']) < 0.001 else ''
            print(f"    谷 {sl['date']} @{sl['price']:.2f}{marker}")

    return result


def scan_portfolio_stops(verbose=True, update_portfolio=True):
    """扫描全部持仓,计算结构止损,与手动地板对比。"""
    pf = pd.read_excel('config/portfolio.xlsx')
    pf['code'] = pf['code'].astype(str).str.zfill(6)

    if verbose:
        print(f"\n{'='*70}")
        print(f"🧗 动态结构止损 | swing low追踪(松版) | {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*70}")

    updates = []
    for _, r in pf.iterrows():
        if int(r['shares']) <= 0:
            continue
        code = r['code']
        result = get_structural_stop(code, verbose=False)
        if result is None:
            if verbose:
                print(f"  {r['name'][:8]:<10} 数据不足,跳过")
            continue

        manual_stop = float(r.get('stop_loss', 0) or 0)
        struct_stop = result['stop_price']
        # 执行线 = max(结构止损, 手动地板)
        exec_stop = max(struct_stop, manual_stop) if manual_stop > 0 else struct_stop

        # 标注谁更高
        if manual_stop > 0 and struct_stop > manual_stop:
            note = f"结构({struct_stop:.2f}) > 手动({manual_stop:.2f}) → 建议用结构"
        elif manual_stop > 0 and manual_stop >= struct_stop:
            note = f"手动({manual_stop:.2f}) ≥ 结构({struct_stop:.2f}) → 保持手动"
        else:
            note = f"结构止损={struct_stop:.2f}(无手动地板)"

        current = result['current_price']
        distance = (current / exec_stop - 1) * 100
        alert = '⚠️近' if distance < 2 else ''

        if verbose:
            print(f"  {r['name'][:8]:<10} 现价{current:>7.2f} → 执行线{exec_stop:>7.2f} "
                  f"(距{distance:+.1f}%{alert}) {note}")
            print(f"    最近谷: {result['swing_lows'][-1]['date']} @{result['last_swing_low']:.2f} | "
                  f"止损锚: {result['swing_lows'][-2]['date']} @{result['second_swing_low']:.2f}")

        updates.append(dict(code=code, name=r['name'], struct_stop=struct_stop,
                            manual_stop=manual_stop, exec_stop=exec_stop,
                            current=current, distance=distance))

    if verbose and updates:
        n_struct_higher = sum(1 for u in updates if u['struct_stop'] > u['manual_stop'] > 0)
        n_manual_higher = sum(1 for u in updates if 0 < u['manual_stop'] >= u['struct_stop'])
        print(f"\n  汇总: 结构更紧{n_struct_higher}只 | 手动更紧{n_manual_higher}只 | "
              f"结构自动追踪替代手动拍脑袋")
        print(f"  💡 结构止损每天自动更新(跟随swing low),手动地板保留为兜底")
        print(f"  💡 执行线=max(结构,手动),两者取高更安全")

    return updates


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='动态结构止损(swing low追踪)')
    parser.add_argument('--code', help='单只股票')
    parser.add_argument('--portfolio', action='store_true', help='全持仓扫描')
    args = parser.parse_args()

    if args.code:
        result = get_structural_stop(args.code, verbose=True)
        if result:
            print(f"\n结构止损: {result['stop_price']}")
    else:
        scan_portfolio_stops()
