#!/usr/bin/env python3
"""全量综合策略扫描 — V3+通道+信号+行业催化 融合评分。

输出Top30，带完整标签和逐只分析数据。

⚠️ 防未来数据: savgol shift(window//2), 只用历史数据
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import argrelextrema, savgol_filter

PROJECT = Path(__file__).parent.parent.parent
SG_W = 21
SG_P = 3

# 行业催化关键词
HOT_KW = ['半导体','芯片','AI','人工智能','算力','大模型','机器人','航天','军工','国防',
           '新能源','储能','光伏','自动驾驶','低空','量子','数据','信创','国产替代',
           '软件','元器件','通信','IT设备','电气自动化','汽车','有色','锂电','充电桩',
           '风电','氢能','工业4.0','数控','传感器','5G','物联网','消费电子']
WARM_KW = ['医药','创新药','银行','券商','保险','食品','白酒','家电','建材','化工',
           '环保','物流','农业','纺织','传媒','教育','地产']


def safe_savgol(ln_p):
    w = min(SG_W, len(ln_p) // 4 * 2 + 1)
    if w < 5:
        return np.zeros(len(ln_p)), np.zeros(len(ln_p))
    from scripts.core.derivatives import causal_savgol  # 2026-09-10: 权威因果(pos=0右端), 弃中心化+shift滞后
    return causal_savgol(ln_p, w, SG_P, deriv=1), causal_savgol(ln_p, w, SG_P, deriv=2)


def _sma(series, n, m):
    """SMA(A, n, m): EMA-style with alpha=m/n."""
    alpha = m / n
    result = np.empty_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def score_stock(code, name, industry, close, low, high, opens, volume):
    """综合评分单只股票。返回dict或None。"""
    n = len(close)
    if n < 120:
        return None

    ln_p = np.log(close)
    f1, f2 = safe_savgol(ln_p)
    price = close[-1]

    # ── 1. V3核心(0-4.0) ──
    # 趋势分(0-4): f'方向 + f''加速度
    f1v, f2v = f1[-1], f2[-1]
    f2_rising = f2[-1] > f2[-3] if n >= 3 else False

    # ⚠️ 修复漏洞4: f'阈值提高到0.002，过滤噪声
    F1_THRESHOLD = 0.002

    if f1v > 0.05 and f2v > 0 and f2_rising:
        trend = 4.0; signal = '🚀上涨加速'
    elif f1v > 0.05 and f2v > 0:
        trend = 3.5; signal = '🚀上涨加速'
    elif f1v > F1_THRESHOLD and f2v > 0:
        trend = 3.0; signal = '🚀上涨加速'
    elif f1v > F1_THRESHOLD and f2v <= 0:
        trend = 2.0; signal = '⚠️上涨减速'
    elif f1v > 0 and f2v > 0:
        trend = 2.5; signal = '🔄底部拐点'  # 弱趋势+加速=拐点
    elif f1v > 0:
        trend = 1.5; signal = '⚠️上涨减速'
    elif f2v > 0 and f2_rising:
        trend = 1.0; signal = '🔄拐点向上'
    elif f2v > 0:
        trend = 0.5; signal = '📉下跌减速'
    else:
        trend = 0.0; signal = '🔴下跌加速'

    # 位置分(0-4): 距多周期高点
    lb180 = min(180, n)
    lb360 = min(360, n)
    lb1000 = min(1000, n)
    dh180 = (price / np.max(high[-lb180:]) - 1) * 100
    dh360 = (price / np.max(high[-lb360:]) - 1) * 100
    dh1000 = (price / np.max(high[-lb1000:]) - 1) * 100

    pos = 0
    if dh180 < -30: pos += 1.5
    elif dh180 < -20: pos += 1.0
    elif dh180 < -10: pos += 0.5
    if dh360 < -30: pos += 1.0
    elif dh360 < -20: pos += 0.5
    if dh1000 < -30: pos += 1.0
    elif dh1000 < -20: pos += 0.5
    pos = min(pos, 4.0)

    # 量能分(0-2)
    vol_score = 0
    if volume is not None and n >= 20:
        avg_vol = np.mean(volume[-20:])
        if avg_vol > 0:
            vr = volume[-1] / avg_vol
            if vr > 1.5: vol_score = 2.0
            elif vr > 1.2: vol_score = 1.5
            elif vr > 1.0: vol_score = 1.0
            else: vol_score = 0.5

    v3 = 0.40 * trend + 0.40 * pos + 0.20 * vol_score

    # ── 2. 通道位置(布林) ──
    ma20 = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    upper_b = ma20 + 2 * std20
    lower_b = ma20 - 2 * std20
    boll_pct = (price - lower_b) / (upper_b - lower_b) * 100 if upper_b > lower_b else 50

    # ── 预计算MACD（提前到B级信号之前）──
    s = pd.Series(close)
    e12 = s.ewm(span=12).mean().values
    e26 = s.ewm(span=26).mean().values
    dif = e12 - e26
    dea = pd.Series(dif).ewm(span=9).mean().values
    hist = 2 * (dif - dea)

    # ── 3. B级信号 ──
    f2_turn = any(f2[-k] > 0 and f2[-k-1] <= 0 for k in range(1, min(4, n-1)))
    is_yang = close[-1] > opens[-1]

    # ⚠️ 修复漏洞1: DIF拐头向上（连续3天DIF上升）
    dif_rising = dif[-1] > dif[-2] and dif[-2] > dif[-3] if n >= 3 else False

    # ⚠️ 修复漏洞2: 均线趋势过滤（MA5开始拐头）
    ma5_now = np.mean(close[-5:])
    ma5_prev = np.mean(close[-6:-1]) if n >= 6 else ma5_now
    ma5_turning = ma5_now > ma5_prev  # MA5拐头向上

    b_signal = boll_pct < 35 and f2_turn and is_yang and dif_rising and ma5_turning

    # ── 4. 底背离 ──
    divergence = False
    if n > 40:
        lows_idx = argrelextrema(low, np.less, order=10)[0]
        if len(lows_idx) >= 2:
            l1, l2 = lows_idx[-2], lows_idx[-1]
            if low[l2] < low[l1] and dif[l2] > dif[l1]:
                divergence = True

    # ── 5. 行业催化(0-3) ──
    text = name + industry
    catalysis = 0.5
    for kw in HOT_KW:
        if kw in text:
            catalysis = 3.0
            break
    if catalysis < 3:
        for kw in WARM_KW:
            if kw in text:
                catalysis = 1.5
                break

    # ── 6. Alpha109: 波动率比(短/长) IC=+0.085 ──
    a109 = np.nan
    if n >= 30:
        hl = high - low
        a109_short = _sma(hl, 10, 2)
        a109_long = _sma(a109_short, 10, 2)
        a109 = a109_short[-1] / a109_long[-1] if a109_long[-1] > 0 else 1.0

    # ── 7. Alpha40: 涨跌量比 IC=-0.033 ──
    a40 = np.nan
    if n >= 27 and volume is not None:
        up_vol = sum(volume[j] for j in range(n - 26, n) if close[j] > close[j - 1])
        dn_vol = sum(volume[j] for j in range(n - 26, n) if close[j] <= close[j - 1])
        a40 = up_vol / dn_vol * 100 if dn_vol > 0 else 200

    # ── 8. 波动率压缩状态 ──
    squeeze = False
    a109_expanding = False
    if not np.isnan(a109):
        if a109 < 0.9:
            squeeze = True  # 波动率压缩中
        elif a109 > 1.05:
            a109_expanding = True  # 波动率开始扩张

    # ── 9. 融合评分 ──
    bonus = 0
    if b_signal: bonus += 1.5  # B级信号加分
    if boll_pct < 20: bonus += 0.5  # 通道极底部
    if boll_pct < 35: bonus += 0.3  # 通道下半
    if divergence: bonus += 0.5  # 底背离
    if catalysis >= 3: bonus += 0.5  # 热门行业
    if boll_pct > 85: bonus -= 1.0  # 通道顶部扣分
    if boll_pct > 70: bonus -= 0.5  # 通道上半扣分

    # Alpha109: 波动率扩张加分(最强因子 IC=+0.085)
    if a109_expanding: bonus += 1.0  # 波动率扩张中
    if squeeze: bonus -= 0.3  # 压缩中，还没启动
    # 从压缩刚突破: a109从<0.9升到>1.0 (爆发前夜)
    if n >= 5 and not np.isnan(a109):
        a109_prev = _sma(high[-10:] - low[-10:], 10, 2)
        a109_prev_long = _sma(a109_prev, 10, 2)
        # 简化: 当前扩张但5天前压缩
        if a109 > 1.0 and a109_long[-1] < a109_short[-1]:
            # 短期>长期=正在扩张
            if not squeeze:
                bonus += 0.5  # 波动率拐点(从压缩转向扩张)

    # Alpha40: 涨跌量比低=超跌确认(底部反弹)
    if not np.isnan(a40) and a40 < 80:
        bonus += 0.5  # 下跌放量>上涨放量 = 超跌

    total = v3 + bonus

    # ── 7. MA位置 ──
    ma5 = np.mean(close[-5:])
    ma60 = np.mean(close[-60:]) if n >= 60 else ma20
    ma120 = np.mean(close[-120:]) if n >= 120 else ma60

    # ── 8. RSI ──
    delta = np.diff(close[-15:])
    gains = delta[delta > 0]
    losses = -delta[delta < 0]
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1
    rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100

    # ── 9. MACD日线（已预计算） ──
    macd_state = '金叉' if dif[-1] > dea[-1] else '死叉'
    macd_state += '+DIF>0' if dif[-1] > 0 else '+DIF<0'

    # ── 10. 信号标签 ──
    # ⚠️ 修复漏洞5: 趋势加速需布林>40%（排除底部假信号）
    # ⚠️ 修复漏洞6: 顶部卖出需f''<0确认（避免卖飞）

    if b_signal and boll_pct < 20:
        tag = '底部反转-确认买入'
    elif boll_pct < 30 and f2_turn and ma5_turning:
        tag = '底部反转-关注'
    elif boll_pct > 85 and f2v < 0:  # 顶部+减速确认
        tag = '顶部卖出'
    elif boll_pct > 85:  # 顶部但还在涨
        tag = '高位预警'
    elif boll_pct > 70 and f2v < 0:
        tag = '减仓预警'
    elif signal == '🚀上涨加速' and boll_pct > 40:  # 真加速（不在底部）
        tag = '趋势加速-持有'
    elif signal == '🚀上涨加速':  # 底部刚启动
        tag = '底部启动'
    elif signal == '⚠️上涨减速':
        tag = '上涨减速-警惕'
    else:
        tag = '观望'

    return {
        'code': code, 'name': name, 'industry': industry,
        'price': round(price, 2),
        'v3': round(v3, 2), 'bonus': round(bonus, 1), 'total': round(total, 2),
        'trend': round(trend, 1), 'pos': round(pos, 1), 'vol_score': round(vol_score, 1),
        'signal': signal, 'tag': tag,
        'boll_pct': round(boll_pct, 0), 'b_signal': b_signal, 'divergence': divergence,
        'catalysis': catalysis,
        'dh180': round(dh180, 1), 'dh360': round(dh360, 1), 'dh1000': round(dh1000, 1),
        'f1': round(f1v, 5), 'f2': round(f2v, 5),
        'rsi': round(rsi, 1), 'macd': macd_state,
        'a109': round(a109, 3) if not np.isnan(a109) else '',
        'a40': round(a40, 0) if not np.isnan(a40) else '',
        'squeeze': squeeze,
        'ma_pos': f'MA5{"↑" if price>ma5 else "↓"} MA60{"↑" if price>ma60 else "↓"} MA120{"↑" if price>ma120 else "↓"}',
        'vol_ratio': round(volume[-1]/np.mean(volume[-20:]), 2) if volume is not None and n>=20 else 1.0,
    }


def main():
    data_dir = PROJECT / 'data' / 'store' / 'kline'
    stock_list = pd.read_parquet(PROJECT / 'data' / 'store' / 'stock_list.parquet')
    name_map = dict(zip(stock_list['symbol'], stock_list['name']))
    industry_map = dict(zip(stock_list['symbol'], stock_list['industry']))

    # 默认股票池: 沪深300+中证500+中证1000
    pool_file = PROJECT / 'data' / 'stock_pool.csv'
    pool_codes = None
    if pool_file.exists():
        pool_df = pd.read_csv(pool_file)
        pool_codes = set(pool_df['code'].astype(str).tolist())
        print(f'股票池: {len(pool_codes)}只')
    else:
        print('未找到stock_pool.csv, 扫描全市场')

    results = []
    total = 0

    for code_dir in sorted(data_dir.iterdir()):
        if not code_dir.is_dir(): continue
        code = code_dir.name
        if len(code) != 6 or not code.isdigit(): continue
        if code.startswith('688'): continue  # 跳过科创板
        if pool_codes is not None and code not in pool_codes: continue  # 只跑股票池

        f = code_dir / 'D.parquet'
        if not f.exists(): continue
        total += 1

        try:
            df = pd.read_parquet(f)
            if len(df) < 140: continue

            # 兼容列名
            date_col = 'trade_date' if 'trade_date' in df.columns else ('date' if 'date' in df.columns else None)
            if date_col is None: continue

            dates = df[date_col].values
            close = df['close'].values.astype(float)
            low = df['low'].values.astype(float)
            high = df['high'].values.astype(float)
            opens = df['open'].values.astype(float)
            volume = df['vol'].values.astype(float) if 'vol' in df.columns else (
                df['volume'].values.astype(float) if 'volume' in df.columns else None)

            if str(dates[0]) > str(dates[-1]):
                close=close[::-1]; low=low[::-1]; high=high[::-1]
                opens=opens[::-1]; dates=dates[::-1]
                if volume is not None: volume=volume[::-1]

            name = name_map.get(code, code)
            industry = industry_map.get(code, '')

            r = score_stock(code, name, industry, close, low, high, opens, volume)
            if r is not None:
                results.append(r)
        except Exception:
            continue

    print(f"扫描完成: {len(results)}/{total}只")
    df_r = pd.DataFrame(results).sort_values('total', ascending=False)

    # ====== 按类别输出Top ======

    # 保存完整数据
    out = PROJECT / 'data' / 'full_scan_results.csv'
    df_r.to_csv(out, index=False)

    # ── 底部反转-确认买入 ──
    print("\n" + "=" * 80)
    print("🟢 底部反转-确认买入（通道极底部+B级信号+f''拐点+阳线）")
    print("=" * 80)
    sub = df_r[df_r['tag'] == '底部反转-确认买入'].head(10)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"\n{i}. {r['name']}({r['code']}) | {r['industry']} | {r['price']}元")
        print(f"   融合={r['total']} V3={r['v3']} 催化={'🔥' if r['catalysis']>=3 else '🟡' if r['catalysis']>=1.5 else '⚪'}")
        print(f"   {r['signal']} | 布林{r['boll_pct']}% | {r['macd']} | RSI={r['rsi']}")
        print(f"   距高: 180日{r['dh180']:.1f}% 360日{r['dh360']:.1f}% 1000日{r['dh1000']:.1f}%")
        print(f"   f'={r['f1']:.5f} f''={r['f2']:.5f} 量比={r['vol_ratio']} {r['ma_pos']}")

    # ── 底部反转-关注 ──
    print("\n" + "=" * 80)
    print("🟡 底部反转-关注（通道下半+f''拐点）")
    print("=" * 80)
    sub = df_r[df_r['tag'] == '底部反转-关注'].head(10)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"\n{i}. {r['name']}({r['code']}) | {r['industry']} | {r['price']}元")
        print(f"   融合={r['total']} V3={r['v3']} | {r['signal']} | 布林{r['boll_pct']}%")

    # ── 趋势加速-持有 ──
    print("\n" + "=" * 80)
    print("🚀 趋势加速-持有（f'>0, f''>0，主升浪中）")
    print("=" * 80)
    sub = df_r[df_r['tag'] == '趋势加速-持有'].head(10)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"\n{i}. {r['name']}({r['code']}) | {r['industry']} | {r['price']}元")
        print(f"   融合={r['total']} V3={r['v3']} 催化={'🔥' if r['catalysis']>=3 else '🟡' if r['catalysis']>=1.5 else '⚪'}")
        print(f"   {r['signal']} | 布林{r['boll_pct']}% | {r['macd']} | RSI={r['rsi']}")
        print(f"   距高: 180日{r['dh180']:.1f}% | f'={r['f1']:.5f} f''={r['f2']:.5f}")

    # ── 顶部卖出 ──
    print("\n" + "=" * 80)
    print("🔴 顶部卖出（布林>85%，通道极顶部）")
    print("=" * 80)
    sub = df_r[df_r['tag'] == '顶部卖出'].head(10)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"\n{i}. {r['name']}({r['code']}) | {r['industry']} | {r['price']}元")
        print(f"   融合={r['total']} V3={r['v3']} | {r['signal']} | 布林{r['boll_pct']}% | RSI={r['rsi']}")

    # ── 减仓预警 ──
    print("\n" + "=" * 80)
    print("⚠️ 减仓预警（布林>70%+减速）")
    print("=" * 80)
    sub = df_r[df_r['tag'] == '减仓预警'].head(10)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        print(f"\n{i}. {r['name']}({r['code']}) | {r['industry']} | {r['price']}元")
        print(f"   融合={r['total']} | {r['signal']} | 布林{r['boll_pct']}% | RSI={r['rsi']}")

    # ── 分数分布统计 ──
    print("\n" + "=" * 80)
    print("📊 分数分布 & 各标签统计")
    print("=" * 80)
    print(f"\n分数分位数:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  {q*100:.0f}%: {df_r['total'].quantile(q):.2f}")
    print(f"\n各标签数量:")
    for tag in df_r['tag'].value_counts().index:
        sub = df_r[df_r['tag'] == tag]
        print(f"  {tag}: {len(sub)}只 | 平均分{sub['total'].mean():.1f}")


if __name__ == '__main__':
    main()
