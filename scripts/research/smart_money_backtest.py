#!/usr/bin/env python3
"""聪明钱痕迹信号回测 — 验证"技术指标能否预知业绩预增公告"。

起因(浪潮000977案例): 7/8中报预增公告前21天(6/17)底背离就触发, 前13天(6/25)四信号共振。
用户洞察: 聪明钱提前知道利好→埋伏→留下技术痕迹→scanner捕中→反向预知利好。

算法 smart_money_score: 6特征(底部/放量异动/底背离/金叉/量价配合/稳步抬升), score 0-6。
回测: 实验组(预增公告前N天窗口) vs 对照组(全市场随机) 触发率 + 卡方p + 特征贡献。
防泄漏: detect_divergence 按 df.iloc[:t+1] 切片, MA/量价纯历史。

判读: 实验组触发率 >> 对照组 且 p<0.01 → 信号有效, 可做"利好预知scanner"。
用法: cd stock && PYTHONPATH=. python scripts/research/smart_money_backtest.py
"""
import sys, random
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from scipy import stats
from scripts.analyzers.divergence_detector import DivergenceDetector

STORE = PROJECT_ROOT / 'data/store/kline'
FORECAST = PROJECT_ROOT / 'data/store/forecast/forecast_all.parquet'
DET = DivergenceDetector()


def load(code):
    p = STORE / f'{code}/D.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date':
            df = df.rename(columns={col: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for c in ['close', 'open', 'high', 'low', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def smart_money_score(df, t):
    """截至t日(iloc[:t+1])的聪明钱痕迹评分0-6 + 各特征。全部无未来。"""
    sub = df.iloc[:t + 1]
    if len(sub) < 30:
        return 0, {}
    close = sub['close'].astype(float).ffill()
    vol = sub['volume'].astype(float).ffill() if 'volume' in sub.columns else pd.Series(1, index=close)
    f = {}
    # 1. 底部区域
    ma20 = close.rolling(20).mean().iloc[-1]
    f['底部'] = bool(close.iloc[-1] / ma20 < 0.92)
    # 2. 放量异动(近5日有放量阳线)
    vol_ma = vol.rolling(20).mean()
    chg = close.pct_change()
    if len(sub) >= 5:
        cond = (vol.iloc[-5:].values > vol_ma.iloc[-5:].values * 1.3) & (chg.iloc[-5:].values > 0.04)
        f['放量异动'] = bool(cond.any())
    else:
        f['放量异动'] = False
    # 3. 底背离(截至t切片, 无未来)
    try:
        sig = DET.detect_divergence(sub.tail(60), period='D')
        f['底背离'] = bool(sig.detected and sig.div_type == 'bottom')
    except Exception:
        f['底背离'] = False
    # 4. MA金叉
    ma5 = close.rolling(5).mean().iloc[-1]
    f['金叉'] = bool(ma5 > ma20)
    # 5. 量价配合(近10日涨日均量/跌日均量)
    if len(sub) >= 10:
        d10 = close.diff().iloc[-10:]
        up = vol.iloc[-10:][d10 > 0].mean()
        dn = vol.iloc[-10:][d10 < 0].mean()
        f['量价配合'] = bool((up / dn > 1.3) if (dn is not None and dn > 0 and not pd.isna(dn)) else False)
    else:
        f['量价配合'] = False
    # 6. 稳步抬升(近10日涨幅)
    if len(sub) >= 11:
        f['稳步抬升'] = bool((close.iloc[-1] / close.iloc[-11] - 1) > 0.03)
    else:
        f['稳步抬升'] = False
    score = int(sum(f.values()))
    return score, f


def window_max_score(code, ann_date, points=(1, 5, 10, 15, 20)):
    """公告前N天(几个时间点)的max score + 该时点特征。ann_date不含。"""
    df = load(code)
    if df is None:
        return None, None
    ann = pd.to_datetime(ann_date)
    idx = df.index[df['date'] >= ann]
    if len(idx) == 0:
        return None, None
    ai = idx[0]
    if ai < 30:
        return None, None
    best_s = 0
    best_f = {}
    for back in points:
        t = ai - back
        if t < 30:
            continue
        s, f = smart_money_score(df, t)
        if s > best_s:
            best_s = s
            best_f = f
    return best_s, best_f


def single_case_wave():
    """浪潮000977单case验证: 6/17-6/25应score>=3。"""
    print("=" * 64)
    print("单case验证: 浪潮000977 (公告7/8, 聪明钱痕迹应6/17起)")
    df = load('000977')
    if df is None:
        print("  无数据"); return
    for d_str in ['2026-06-15', '2026-06-17', '2026-06-22', '2026-06-25', '2026-06-30', '2026-07-06']:
        d = pd.to_datetime(d_str)
        idx = df.index[df['date'] >= d]
        if len(idx) == 0:
            continue
        t = idx[0]
        s, f = smart_money_score(df, t)
        trig = ''.join([k[0] for k, v in f.items() if v])
        mark = '✓' if s >= 3 else ' '
        print(f"  {mark} {d_str}: score={s} 触发[{trig}]")


def main():
    random.seed(42)
    single_case_wave()
    if not FORECAST.exists():
        print(f"\n❌ {FORECAST} 不存在, 先跑 scripts/data/fetch_forecast.py")
        return
    fc = pd.read_parquet(FORECAST)
    events = fc[(fc['type'] == '预增') & (fc['p_change_min'] > 30)].copy()
    events['code'] = events['ts_code'].str[:6]
    # 去重(同股多次取最近)
    events = events.sort_values('ann_date').drop_duplicates('code', keep='last')
    print(f"\n{'=' * 64}\n全样本回测: 实验组{len(events)}只预增公告(p_change>30, 去重)")

    # 实验组
    exp_scores = []
    exp_feats = {}
    for i, (_, e) in enumerate(events.iterrows()):
        ms, mf = window_max_score(e['code'], e['ann_date'])
        if ms is not None:
            exp_scores.append(ms)
            for k, v in mf.items():
                exp_feats[k] = exp_feats.get(k, 0) + (1 if v else 0)
        if (i + 1) % 200 == 0:
            print(f"  实验组[{i + 1}/{len(events)}]", flush=True)
    exp_scores = np.array(exp_scores)
    exp_trig = (exp_scores >= 3).mean() * 100
    print(f"实验组: {len(exp_scores)}事件, score≥3触发率{exp_trig:.1f}%, 均分{exp_scores.mean():.2f}")

    # 对照组A: 全市场随机股-日
    all_codes = [d.name for d in STORE.iterdir() if (d / 'D.parquet').exists() and d.name.startswith(('0', '3', '6'))]
    sample = random.sample(all_codes, min(800, len(all_codes)))
    ctrl_scores = []
    for code in sample:
        df = load(code)
        if df is None or len(df) < 60:
            continue
        t = random.randint(30, len(df) - 1)
        s, _ = smart_money_score(df, t)
        ctrl_scores.append(s)
    ctrl_scores = np.array(ctrl_scores)
    ctrl_trig = (ctrl_scores >= 3).mean() * 100
    print(f"对照组A(全市场随机{len(ctrl_scores)}): score≥3触发率{ctrl_trig:.1f}%, 均分{ctrl_scores.mean():.2f}")

    # 卡方检验
    table = np.array([[(exp_scores >= 3).sum(), (exp_scores < 3).sum()],
                      [(ctrl_scores >= 3).sum(), (ctrl_scores < 3).sum()]])
    chi2, p, _, _ = stats.chi2_contingency(table)
    lift = exp_trig / ctrl_trig if ctrl_trig > 0 else float('inf')
    print(f"\n{'=' * 64}\n=== 结论 ===")
    print(f"实验组触发率 {exp_trig:.1f}% vs 对照组 {ctrl_trig:.1f}% | lift={lift:.2f}x | 卡方p={p:.4g}")
    valid = exp_trig > ctrl_trig * 1.5 and p < 0.01
    print(f"{'✅有效' if valid else '❌证伪/弱'}: ", end='')
    print("聪明钱痕迹能预知利好→可做scanner" if valid else "痕迹无显著预知力(浪潮可能是巧合)")

    print(f"\n=== 各特征贡献(实验组该特征触发率) ===")
    for k in ['底部', '放量异动', '底背离', '金叉', '量价配合', '稳步抬升']:
        print(f"  {k}: {exp_feats.get(k, 0) / len(exp_scores) * 100:.1f}%")

    print(f"\n=== score分布(实验组 vs 对照组) ===")
    for s in range(7):
        print(f"  score={s}: 实验{(exp_scores == s).mean() * 100:5.1f}% vs 对照{(ctrl_scores == s).mean() * 100:5.1f}%")


if __name__ == '__main__':
    main()
