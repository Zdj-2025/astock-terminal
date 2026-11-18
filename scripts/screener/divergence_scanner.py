#!/usr/bin/env python3
"""背离买点scanner — 找当前满足"量能确认底背离+低位"的可复现买点。

回测验证(macd_norm_divergence_backtest.py + divergence_enhance_backtest.py, 130万样本, 扣0.3%成本):
- 标准底背离[centered]: 夏普2.36 ❌未来函数假象, 不可复现
- 标准底背离[右侧死等]: 夏普0.25 ❌扣成本无效(低于基准0.38)
- 量能阳线确认底背离: 夏普0.71, p=0.0000 ✅可复现有效
- 量能确认+低位(距120高<-15%): 夏普0.87, 胜率57% ✅最佳组合

信号(全部右侧确认, 无未来, 可实时复现):
- 底背离: 价格新低 + MACD柱未新低(下跌动能衰竭)
- confirmed(量能阳线确认): 谷后当前bar放量阳线=启动确认, 消除centered未来函数
- 低位: 距120日高<-15%(深位反弹空间大)

定位: 均值回归买点(超跌+动能衰竭+启动确认), 与breakout_scanner(趋势突破)互补。
注意: divergence_detector的_ema曾因NaN bug致背离检测全失效, 2026-06-27已修复。
用法: cd stock && PYTHONPATH=. python scripts/screener/divergence_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.core.tushare_client import TushareClient
from scripts.analyzers.divergence_detector import DivergenceDetector

STORE = PROJECT_ROOT / 'data/store/kline'


def detect_double_bottom(df, rally=0.20):
    """背离+双底结构(2026-09-02回测验证: 早触发40d中位+1.49%/胜55%/均值+3.15%)。
    三谷: La(背离头底)→Lb(新低,hist/DIF抬升)→Lc(≈Lb±7%更高的低点);
    中间反弹N≥Lb*(1+rally); 早触发=Lc后3根不破+close∈[Lc*1.04,Lc*1.12]。
    晚触发(破颈线)已证伪(-4%~-12%中位)不实现; 红柱积分无增量不实现(共线性消融证实)。
    返回最近触发的结构dict或None(触发日须在近5个交易日内,供每日扫描可操作性)。"""
    import numpy as np
    l = df['low'].astype(float).values
    h = df['high'].astype(float).values
    c = df['close'].astype(float).values
    s = pd.Series(c)
    ef = s.ewm(span=12, adjust=False).mean(); es = s.ewm(span=26, adjust=False).mean()
    dif = ef - es; hist = dif - dif.ewm(span=9, adjust=False).mean()
    dif, hist = dif.values, hist.values
    n = 5
    lows = [i for i in range(n, len(l) - n)
            if all(l[i] < l[i - j] for j in range(1, n + 1)) and all(l[i] < l[i + j] for j in range(1, n + 1))]
    if len(lows) < 3:
        return None
    best = None
    for k in range(2, len(lows)):
        La, Lb, Lc = lows[k - 2], lows[k - 1], lows[k]
        if not (15 <= Lb - La <= 60 and Lc - Lb <= 40):
            continue
        if not (l[Lb] < l[La] and hist[Lb] > hist[La] and dif[Lb] > dif[La]):
            continue
        if Lb + 1 < Lc and np.min(l[Lb + 1:Lc + 1]) < l[Lb] * 0.90:
            continue
        band = l[Lc] / l[Lb] - 1
        if not (0 <= band <= 0.30):
            continue   # 2026-09-03回测: 深带二买(7~30%)中位+2.01%/胜58%且2022熊市+4.6%唯一正,浅带双底保留;>30%放弃
        N = float(np.max(h[Lb:Lc + 1]))
        if N < l[Lb] * (1 + rally) or l[Lc] >= N:
            continue
        # 2026-09-02买点矩阵回测: A3确认即买(中位+3.61%/胜57%)优于A1反弹带(+3.21%/54%)
        early = None
        if Lc + 4 < len(c) and all(l[Lc + 1:Lc + 4] >= l[Lc]):  # 相等回踩=不破(9/3欢瑞案例)
            early = Lc + 3   # Lc后3根不破→第4根确认即触发; 实盘买入价≤Lc*1.10防追高
        if early is not None and early >= len(c) - 5:   # 触发在近5日内=可操作
            best = dict(Lb=l[Lb], Lc=l[Lc], N=N,
                        stop=min(l[Lb], l[Lc]),
                        trig_date=str(df['date'].iloc[early])[:10],
                        rally_pct=(N / l[Lb] - 1) * 100,
                        tier='深二买' if band > 0.07 else '浅双底')
    return best


def scan_divergence():
    c = TushareClient()
    sl = c.get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))
    ind_map = dict(zip(sym, sl['industry']))

    det = DivergenceDetector()
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    hits = []
    for code in codes:
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            for col in ['datetime', 'day', 'trade_date']:
                if col in df.columns and col != 'date':
                    df = df.rename(columns={col: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) < 150:
                continue
            sig = det.detect_divergence(df, period='D')
            if not sig.detected or sig.div_type != 'bottom':
                continue
            close = df['close'].astype(float)
            hi120 = close.rolling(120).max().iloc[-1]
            dist120 = (close.iloc[-1] / hi120 - 1) * 100
            # 避雷过滤(实战铁律): ST/退市/低价面值(<2元)/暴跌(距120高<-75%)的背离不可靠
            nm_raw = name_map.get(code, '')
            if isinstance(nm_raw, str) and ('ST' in nm_raw or '退' in nm_raw):
                continue
            if float(close.iloc[-1]) < 2 or dist120 < -75:
                continue
            nm = nm_raw if isinstance(nm_raw, str) else ''
            ind = ind_map.get(code, '')
            ind = ind if isinstance(ind, str) else ''
            db = detect_double_bottom(df)
            hits.append({
                'code': code, 'name': nm, 'industry': ind,
                'price': float(close.iloc[-1]),
                'strength': sig.strength,
                'dist120': dist120,
                'confirmed': sig.confirmed,           # 量能阳线确认(可复现)
                'low_pos': dist120 < -15,             # 低位(回测最佳组合)
                'db': db,                              # 背离+双底结构(2026-09-02新增)
                'date': str(df['date'].iloc[-1])[:10],
            })
        except Exception:
            continue
    return hits


def main():
    hits = scan_divergence()
    # 换手率/量比标注(150天回测验证: 换手>10%+量比>5=分歧看跌避雷; 量比<2=缩量抗跌加分)
    try:
        import tushare as ts
        pro = ts.pro_api()
        latest = max(h['date'] for h in hits).replace('-', '') if hits else None
        db = {}
        if latest:
            d = pro.daily_basic(trade_date=latest, fields='ts_code,turnover_rate,volume_ratio')
            db = {r['ts_code'][:6]: (float(r['turnover_rate']), float(r['volume_ratio'])) for _, r in d.iterrows()}
    except Exception:
        db = {}
    for h in hits:
        tr, vr = db.get(h['code'], (None, None))
        h['turnover'] = tr; h['vratio'] = vr
        if tr is not None and vr is not None:
            if tr > 10 and vr > 5:
                h['vol_tag'] = '⚠️分歧避雷'
            elif vr < 2:
                h['vol_tag'] = '✅缩量'
            else:
                h['vol_tag'] = ''
        else:
            h['vol_tag'] = ''
    # 排序: confirmed优先 > strength高 (2026-08-28: 低位从强制改为标注,
    # 回测证明背离无低位t=7.4远强于背离+低位t=3.1,过度过滤)
    hits.sort(key=lambda h: (not h['confirmed'], -h['strength']))
    conf = [h for h in hits if h['confirmed']]
    conf_low = [h for h in conf if h['low_pos']]
    dbl = [h for h in hits if h.get('db')]
    print(f"📊 背离买点scanner | 底背离 + 量能阳线确认 (低位为标注非门槛,2026-08-28放宽)")
    print(f"   底背离: {len(hits)}只 | ✅量能确认: {len(conf)}只 | └其中低位: {len(conf_low)}只 | 🥇背离+双底: {len(dbl)}只")
    print(f"   回测: 背离无低位t=7.4>背离+低位t=3.1 → 低位降级为参考信息\n")
    if dbl:
        print(f"🥇 背离+双底(2026-09-02上线): 早触发近5日内 | 40日预期(中位+1.5%/胜55%),止损=双底低点")
        print(f"   🏷️regime标注(2026-09-11回测): down桶(上证ret20<-3%)20d超额+9.79%/胜62% vs shake桶-2.52%/胜38%(p=0.042,N=16)——")
        print(f"   shake期触发的双底标⚠️观察不推荐;down桶标注仅供参考,90天样本外验证后才升级为闸门")
        for h in dbl:
            d = h['db']
            print(f"  {h['code']} {h['name'][:9]:<10} [{d.get('tier','')}] 价{h['price']:>7.2f} 触发{d['trig_date']} "
                  f"反弹{d['rally_pct']:+.0f}% 止损{d['stop']:.2f} 颈线{d['N']:.2f}")
        print()
    print(f"{'代码':<7}{'名称':<10}{'价':>7}{'距120高':>8}{'换手%':>6}{'量比':>5}  状态")
    print('-' * 72)
    for h in hits[:30]:
        if h['confirmed']:
            st = '✅确认' + ('·低位★' if h['low_pos'] else '')
        else:
            st = '🔍未确认'
        if h.get('db'):
            st += ' 🥇双底'
        if h['vol_tag']:
            st += ' ' + h['vol_tag']
        trs = f"{h['turnover']:.0f}" if h['turnover'] is not None else '-'
        vrs = f"{h['vratio']:.1f}" if h['vratio'] is not None else '-'
        print(f"{h['code']:<7}{h['name'][:9]:<10}{h['price']:>7.2f}{h['dist120']:>+7.0f}%{trs:>6}{vrs:>5}  {st}")
    print(f"\n💡 ✅确认=首选(低位★为加分非门槛); 🥇双底=背离+双底结构触发(40日持有预期,前2-3周浮亏常态); ⚠️分歧避雷=换手>10%+量比>5")
    # 写入信号累积器(2026-09-09接入:横向×纵向累积的原料)
    try:
        from scripts.screener.scan_history import record_batch
        record_batch('divergence', hits)
        print(f"已写入信号累积器: divergence {len(hits)}只")
    except Exception as _e:
        print(f"⚠️ 累积器写入跳过: {_e}")


if __name__ == '__main__':
    main()
