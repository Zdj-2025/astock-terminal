#!/usr/bin/env python3
"""三闸门(2026-08-24 8月复盘落地:7%→1%的学费换来的规则)。

教训链: 8/17-18指数顶部区(3982/3990,21天时间窗)新买76k三只全亏 → 顶部禁买令
       AI+医药63%因子集中吃满轮动回撤 → 因子35%硬上限买入门
       数据港+84%→+53%坐电梯,垫厚线(成本+25%)永远碰不到 → 梯度兑现

三闸门:
①顶部禁买令: 指数处于"近20日高点±5%未确认区"(收盘既没创新高也没回撤超5%)
   → 新仓冻结。突破创新高=确认可买; 回撤>5%=充分回调可买。
   8/17-18(距高点0.1-0.3%)会被拦 ✓ 今天(-2.8%)也被拦 ✓
②因子买入门: 单因子暴露>35%时,该因子新仓禁止(只许买缺口因子)
③梯度兑现: 浮盈峰值≥+50%的垫厚仓,从浮盈高点回撤15%→减1/3;20%→再减1/3
   (比五维的"红柱衰减+破MA20+双背离"早触发、少回吐;数据港8/24会在~26.0触发第一批)

用法: premarket_position_check.py末尾自动调用 check_buy_gates()
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

TOP_ZONE_PCT = 5.0     # ①顶部区: 距20日高点±5%未确认
FACTOR_CAP = 0.35      # ②因子上限
LADDER_BASE = 50.0     # ③垫厚门槛: 浮盈峰值≥+50%
LADDER_D1, LADDER_D2 = 15.0, 20.0  # 回撤15%减1/3, 20%再减1/3


def _qfq(c):
    """回溯qfq(防除权假跳变)。⚠️必须用pct_change(比率)不是np.diff(价差)——
    2026-08-25教训: np.diff对26元股跌2毛5就被当-25%暴跌,全序列被乱缩,
    梯度兑现信号因此哑火(峰值30.20被缩成27.76)。"""
    c = np.asarray(c, dtype=float)
    rets = np.full(len(c), np.nan)
    rets[1:] = c[1:] / c[:-1] - 1   # 比率,不是价差
    cc = c.copy()
    for i in range(len(cc) - 2, -1, -1):
        if not np.isnan(rets[i + 1]) and rets[i + 1] < -0.25:
            cc[:i + 1] = cc[:i + 1] * (c[i + 1] / c[i])
    return cc


def gate_top_freeze(verbose=True):
    """①顶部禁买令: 返回 (frozen, detail)"""
    try:
        idx = pd.read_parquet('data/store/kline/000001SH/D.parquet').sort_values('date')
        c = idx['close'].astype(float).values
        px, hi20 = c[-1], c[-21:].max()
        dist = (px / hi20 - 1) * 100
        if dist >= 0:
            frozen, why = False, f'收盘{px:.0f}创20日新高(突破确认),可买'
        elif dist < -TOP_ZONE_PCT:
            frozen, why = False, f'距20日高点{dist:+.1f}%已充分回调(>{TOP_ZONE_PCT}%),可买'
        else:
            frozen, why = True, f'顶部未确认区: 距20日高点{dist:+.1f}%(±{TOP_ZONE_PCT}%内)→新仓冻结'
        if verbose:
            print(f"  {'🚫' if frozen else '✅'} 顶部禁买令: {why}")
        return frozen, why
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 顶部禁买令跳过: {str(e)[:40]}")
        return None, str(e)


def gate_factor_cap(verbose=True):
    """②因子35%买入门: 返回 (blocked_factors, exposure_dict)"""
    try:
        from scripts.analyzers.factor_exposure import analyze_factor_exposure
        fe = analyze_factor_exposure(verbose=False)
        blocked, lines = [], []
        if fe and isinstance(fe.get('factors'), dict):
            for fac, info in fe['factors'].items():
                pct = info['pct'] if isinstance(info, dict) else info
                if pct > FACTOR_CAP:
                    blocked.append(fac)
                    lines.append(f"{fac}{pct:.0%}")
        if verbose:
            if blocked:
                print(f"  🚫 因子买入门: {'/'.join(lines)} 超{FACTOR_CAP:.0%}上限→这些因子新仓禁止")
            else:
                print(f"  ✅ 因子买入门: 无因子超{FACTOR_CAP:.0%}上限")
        return blocked, fe
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 因子门跳过: {str(e)[:40]}")
        return None, None


def ladder_take_profit(verbose=True):
    """③梯度兑现: 峰值≥+50%的仓位,回撤15%/20%减1/3。返回信号列表"""
    signals = []
    try:
        pf = pd.read_excel('config/portfolio.xlsx')
        pf['code'] = pf['code'].astype(str).str.zfill(6)
        for _, r in pf.iterrows():
            code, cost, shares = r['code'], float(r['avg_cost']), int(r['shares'])
            if cost <= 0 or shares <= 0:
                continue
            bd = str(r.get('buy_date', ''))[:10]
            try:
                df = pd.read_parquet(f'data/store/kline/{code}/D.parquet').sort_values('date')
            except Exception:
                continue
            if len(bd) == 10:
                df = df[df['date'] >= bd]
            if not len(df):
                continue
            c = _qfq(df['close'].astype(float).values)
            peak_px = c.max()
            peak_pnl = (peak_px / cost - 1) * 100
            if peak_pnl < LADDER_BASE:
                continue
            px_now = c[-1]
            dd = (px_now / peak_px - 1) * 100  # 从浮盈高点回撤(负)
            if dd <= -LADDER_D2:
                sig = f'回撤{-dd:.0f}%≥{LADDER_D2:.0f}%→累计应减2/3'
                signals.append((r['name'], code, px_now, peak_pnl, dd, sig))
            elif dd <= -LADDER_D1:
                sig = f'回撤{-dd:.0f}%≥{LADDER_D1:.0f}%→应减1/3'
                signals.append((r['name'], code, px_now, peak_pnl, dd, sig))
        if verbose:
            if signals:
                print(f"  🔔 梯度兑现信号 {len(signals)} 只(峰值≥+{LADDER_BASE:.0f}%仓):")
                for nm, code, px, pp, dd, sig in signals:
                    print(f"     {nm}({code}) 现价{px:.2f} 峰值+{pp:.0f}% {sig}")
            else:
                print(f"  ✅ 梯度兑现: 无触发(峰值≥+{LADDER_BASE:.0f}%且回撤≥{LADDER_D1:.0f}%才响)")
        # 2026-11-12机制落地(待修#1): 信号触发→自动生成预授权卖单行,晨报可见,用户确认后挂
        # 教训: 11/11合合100元想卖没动手,规则亮灯13天无执行机制,灯亮≠执行
        try:
            _emit_ladder_preauth(signals)
        except Exception as _e:
            if verbose:
                print(f"  ⚠️ 梯度兑现预授权生成失败: {str(_e)[:40]}")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 梯度兑现跳过: {str(e)[:40]}")
    return signals


def _emit_ladder_preauth(signals):
    """梯度兑现信号→pre_authorized_orders.csv自动行(去重:同代码已有未过期梯度兑现行则跳过)"""
    import datetime as _dt
    from pathlib import Path as _P
    f = _P('config/pre_authorized_orders.csv')
    today = _dt.date.today().isoformat()
    existing = f.read_text(encoding='utf-8') if f.exists() else ''
    added = []
    for nm, code, px, pp, dd, sig in signals:
        # 去重: 该代码已有梯度兑现相关行(手动或自动)且日期>=7天前
        dup = False
        for line in existing.splitlines():
            if code in line:
                try:
                    ld = line.split(',')[0]
                    if len(ld) == 10 and ld >= (_dt.date.today() - _dt.timedelta(days=7)).isoformat():
                        dup = True
                except Exception:
                    pass
        if dup:
            continue
        frac = '2/3' if '2/3' in sig else '1/3'
        row = f"{today},{code},{nm},卖,{px:.2f}开盘附近,-,{frac}仓位·梯度兑现自动生成待确认·峰值+{pp:.0f}%{sig}·用户确认后挂单·可否决"
        with open(f, 'a', encoding='utf-8') as fh:
            fh.write(row + '\n')
        existing += row + '\n'
        added.append(f"{nm}({code})")
    if added:
        print(f"  📋 梯度兑现已自动生成预授权卖单: {' | '.join(added)} → pre_authorized_orders.csv(晨报确认,可否决)")


def check_buy_gates():
    """premarket末尾调用: 三闸门总检。返回 dict"""
    print(f"\n{'='*70}")
    print("🚧 三闸门 | 8月复盘落地(顶部禁买令/因子35%上限/梯度兑现)")
    print('=' * 70)
    frozen, why = gate_top_freeze()
    blocked, _ = gate_factor_cap()
    signals = ladder_take_profit()
    can_buy = not frozen
    print(f"\n  结论: {'❌今日新仓冻结' if frozen else '✅市场层面可买'}"
          + (f" | 禁买因子: {'/'.join(blocked)}" if blocked else '')
          + (f" | 现金线另查" ))
    return dict(top_frozen=frozen, blocked_factors=blocked, ladder_signals=signals)


if __name__ == '__main__':
    check_buy_gates()
