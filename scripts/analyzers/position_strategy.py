#!/usr/bin/env python3
"""仓位策略层 — 回答"当前环境下最优仓位是多少"（2026-08-20建）。

背景: 现有系统只回答"买哪只"+"现金≥15%下限"，从不回答"该不该84%仓位"。
本模块用三因子regime模型给出策略级仓位建议。

三因子模型:
  趋势(40%): 上证 vs MA60(长期)/MA20(中期) → 进攻/中性/防守
  宽度(35%): 抽样500只算%above_MA20 → >60%=宽(进攻), 40-60%=中性, <40%=窄(防守)
  量能(25%): 上证5日均量 vs 20日均量 → 放量/缩量

综合评分 → 推荐仓位区间 → 与当前仓位对比 → 超配/低配告警

用法:
  from scripts.analyzers.position_strategy import analyze_position_strategy
  result = analyze_position_strategy()  # dict + 打印
"""
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# 权重
W_TREND = 0.40
W_BREADTH = 0.35
W_VOLUME = 0.25

# 仓位区间
POSITION_BANDS = {
    '进攻': (0.60, 0.85),   # score > 0.5
    '中性': (0.40, 0.60),   # 0 <= score <= 0.5
    '防守': (0.20, 0.40),   # score < 0
}


def get_sh_index():
    """读上证指数日线"""
    path = 'data/store/kline/000001SH/D.parquet'
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    df = df.sort_values(dcol)
    return df


def trend_score(df):
    """趋势因子: 上证 vs MA60/MA20"""
    c = df['close'].astype(float)
    px = c.iloc[-1]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]

    if px > ma60:
        score = 1  # 长期趋势向上
        label = f"进攻(收{px:.0f}>MA60={ma60:.0f})"
    elif px < ma20:
        score = -1  # 中短期破位
        label = f"防守(收{px:.0f}<MA20={ma20:.0f})"
    else:
        score = 0  # 中性区间(MA20<px<MA60)
        label = f"中性(MA20={ma20:.0f}<收{px:.0f}<MA60={ma60:.0f})"

    return score, label, px, ma20, ma60


def breadth_score(sample_size=500):
    """宽度因子: 抽样个股%above_MA20"""
    kdir = 'data/store/kline'
    codes = [d for d in os.listdir(kdir) if d.isdigit() and len(d) == 6
             and os.path.isfile(f'{kdir}/{d}/D.parquet')]
    random.seed(42)
    sample = random.sample(codes, min(sample_size, len(codes)))

    above = 0
    total = 0
    for code in sample:
        try:
            df = pd.read_parquet(f'{kdir}/{code}/D.parquet')
            dcol = [c for c in df.columns if 'date' in c.lower()]
            if not dcol:
                continue
            df = df.sort_values(dcol[0])
            c = df['close'].astype(float)
            if len(c) < 21:
                continue
            ma20 = c.rolling(20).mean().iloc[-1]
            if pd.notna(ma20) and ma20 > 0:
                total += 1
                if c.iloc[-1] > ma20:
                    above += 1
        except Exception:
            continue

    pct = above / total if total > 0 else 0.5

    if pct > 0.60:
        score = 1
        label = f"进攻({pct:.0%}在MA20上)"
    elif pct < 0.40:
        score = -1
        label = f"防守(仅{pct:.0%}在MA20上)"
    else:
        score = 0
        label = f"中性({pct:.0%}在MA20上)"

    return score, label, pct


def volume_score(df):
    """量能因子: 5日均量 vs 20日均量"""
    v = df['volume'].astype(float) if 'volume' in df.columns else df['vol'].astype(float)
    vol5 = v.rolling(5).mean().iloc[-1]
    vol20 = v.rolling(20).mean().iloc[-1]

    if pd.isna(vol5) or pd.isna(vol20) or vol20 == 0:
        return 0, "数据不足"

    ratio = vol5 / vol20
    if ratio > 1.1:
        score = 1
        label = f"放量(5日均量/20日={ratio:.2f})"
    elif ratio < 0.85:
        score = -1
        label = f"缩量(5日均量/20日={ratio:.2f})"
    else:
        score = 0
        label = f"平稳(量比{ratio:.2f})"

    return score, label, ratio


def analyze_position_strategy(capital_path='config/capital.xlsx', verbose=True):
    """主入口: 三因子regime → 推荐仓位 → 与当前仓位对比"""
    # 1. 读上证
    df = get_sh_index()
    if df is None:
        if verbose:
            print("⚠️ 无上证数据")
        return None

    # 2. 三因子评分
    t_score, t_label, px, ma20, ma60 = trend_score(df)
    b_score, b_label, b_pct = breadth_score()
    v_score, v_label, v_ratio = volume_score(df)

    # 3. 综合评分
    total = t_score * W_TREND + b_score * W_BREADTH + v_score * W_VOLUME

    if total > 0.5:
        regime = '进攻'
    elif total < 0:
        regime = '防守'
    else:
        regime = '中性'

    lo, hi = POSITION_BANDS[regime]

    # 4. 当前仓位
    current_pos = None
    try:
        cap = pd.read_excel(capital_path)
        for _, r in cap.iterrows():
            if '当前仓位' in str(r.get('项目', '')):
                val = r['金额/数值']
                current_pos = float(val) if pd.notna(val) and val != '' else None
                break
    except Exception:
        pass

    # 5. 输出
    result = {
        'trend': {'score': t_score, 'label': t_label, 'weight': W_TREND},
        'breadth': {'score': b_score, 'label': b_label, 'weight': W_BREADTH},
        'volume': {'score': v_score, 'label': v_label, 'weight': W_VOLUME},
        'total_score': total,
        'regime': regime,
        'recommended_range': (lo, hi),
        'current_position': current_pos,
        'delta': None,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"⚖️ 仓位策略 | 三因子regime模型")
        print(f"{'='*70}\n")
        print(f"  趋势({W_TREND:.0%}): {t_label:<30} → {t_score:+.0f} × {W_TREND:.2f} = {t_score*W_TREND:+.2f}")
        print(f"  宽度({W_BREADTH:.0%}): {b_label:<30} → {b_score:+.0f} × {W_BREADTH:.2f} = {b_score*W_BREADTH:+.2f}")
        print(f"  量能({W_VOLUME:.0%}): {v_label:<30} → {v_score:+.0f} × {W_VOLUME:.2f} = {v_score*W_VOLUME:+.2f}")
        print(f"  {'─'*50}")
        print(f"  综合: {total:+.2f} → {regime} | 推荐仓位 {lo:.0%}~{hi:.0%}")

        if current_pos is not None:
            result['delta'] = current_pos - (lo + hi) / 2
            if current_pos > hi:
                over = current_pos - hi
                print(f"\n  当前仓位: {current_pos:.1%} ⚠️超配{over:.1%}")
                print(f"  建议: 减持至{hi:.0%}以下")
                # 配合因子暴露建议
                try:
                    from scripts.analyzers.factor_exposure import analyze_factor_exposure
                    fe = analyze_factor_exposure(verbose=False)
                    if fe and fe['max_factor'][1] > 0.35:
                        print(f"  减仓方向: 优先减{fe['max_factor'][0]}因子(当前{fe['max_factor'][1]:.0%})")
                except Exception:
                    pass
            elif current_pos < lo:
                under = lo - current_pos
                print(f"\n  当前仓位: {current_pos:.1%} | 低配{under:.1%}(有加仓空间)")
            else:
                print(f"\n  当前仓位: {current_pos:.1%} ✅在推荐区间内")

    return result


# ── 政策tilt影子模式(2026-09-11立,P3) ────────────────────────────────
# 边界(用户拍板): 政策分析永不直接生成买卖信号 → tilt只做±1档修正建议的"影子",
# 每日落盘但**不影响任何真实输出决策**,90天对答案(2026-12-10)后才决定是否给真权重.
# 输入: config/policy_tilt.csv (date,tilt_score∈[-2,+2],rationale,evidence_level,verify_by)
# 证据分级: 仅L3+可进配置层; 当前L2 → 只影子.
BAND_ORDER = ['防守', '中性', '进攻']


def policy_tilt_shadow(base_result=None, verbose=True):
    import csv
    from datetime import datetime
    tilt_path = PROJECT_ROOT / 'config/policy_tilt.csv'
    shadow_path = PROJECT_ROOT / 'reports/policy_tilt_shadow.csv'
    if base_result is None:
        base_result = analyze_position_strategy(verbose=False)
    if base_result is None or not tilt_path.exists():
        return None
    with open(tilt_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    latest = rows[-1]
    tilt = int(latest['tilt_score'])
    # 影子regime = 基础regime在band序列上移tilt档(钳位)
    try:
        bi = BAND_ORDER.index(base_result['regime'])
    except ValueError:
        return None
    si = max(0, min(len(BAND_ORDER) - 1, bi + tilt))
    s_regime = BAND_ORDER[si]
    s_lo, s_hi = POSITION_BANDS[s_regime]
    cur = base_result.get('current_position')
    would = ''
    if cur is not None:
        if cur > s_hi:
            would = f'影子会说: 减持至{s_hi:.0%}以下'
        elif cur < s_lo:
            would = f'影子会说: 仓位偏低,加仓空间{s_lo:.0%}~{s_hi:.0%}'
        else:
            would = f'影子会说: 仓位在{s_lo:.0%}~{s_hi:.0%}内,OK'
    exists = shadow_path.exists()
    with open(shadow_path, 'a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['date', 'tilt', 'evidence', 'base_regime', 'base_range',
                        'shadow_regime', 'shadow_range', 'current_pos', 'would_say'])
        w.writerow([datetime.now().strftime('%Y-%m-%d'), tilt, latest['evidence_level'],
                    base_result['regime'],
                    f"{base_result['recommended_range'][0]:.0%}~{base_result['recommended_range'][1]:.0%}",
                    s_regime, f'{s_lo:.0%}~{s_hi:.0%}',
                    f'{cur:.1%}' if cur is not None else '',
                    would])
    if verbose:
        print(f"\n  🌑 政策tilt影子({latest['evidence_level']},tilt{tilt:+d}): {base_result['regime']}→{s_regime}"
              f" {s_lo:.0%}~{s_hi:.0%} | {would}")
        print(f"     ⚠️ 影子模式: 仅记录不参与决策,对答案日{latest['verify_by']}")
    return dict(tilt=tilt, shadow_regime=s_regime, would_say=would)


if __name__ == '__main__':
    r = analyze_position_strategy()
    policy_tilt_shadow(r)
