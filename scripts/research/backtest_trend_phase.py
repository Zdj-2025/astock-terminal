#!/usr/bin/env python3
"""趋势阶段(启动/延伸)回测 — 验证P0-2的trend_phase权重假设(2026-06-16)。

假设: 趋势加速候选中, 启动期(ret20<10%)未来收益 > 延伸期(ret20>=20%)。
若显著(p<0.05), 则daily_scan的"启动期+1.5/延伸期-1.0"加权合理, 否则需调权重。

用法: PYTHONPATH=. python scripts/research/backtest_trend_phase.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

STORE = PROJECT_ROOT / "data" / "store" / "kline"
LOOKBACK_DAYS = 120   # 回测窗口(交易日)
HOLD = 10            # 持有日数


def compute_f1_f2(close):
    """近似V3的f1(趋势强度)/f2(加速度)。方向与V3一致(f1>0=上涨加速)。"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    f1 = dif / close * 10           # 归一化趋势强度
    f2 = f1.diff(5)                 # 加速度(5日变化)
    return f1, f2


def main():
    codes = sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])
    print(f"回测 {len(codes)}只 | 窗口{LOOKBACK_DAYS}日 | 持有{HOLD}日")

    records = []
    for i, code in enumerate(codes):
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) < 150:
                continue
            close = df['close']
            df['MA20'] = close.rolling(20).mean()
            df['ret20'] = close.pct_change(20) * 100
            df['f1'], df['f2'] = compute_f1_f2(close)
            df['future_ret'] = close.pct_change(HOLD).shift(-HOLD) * 100
            recent = df.dropna(subset=['f1', 'f2', 'ret20', 'future_ret', 'MA20']).tail(LOOKBACK_DAYS)
            # 趋势加速条件(近似daily_scan的accel_mask): f1>0, f2>0, above_ma20
            accel = recent[(recent['f1'] > 0) & (recent['f2'] > 0) & (recent['close'] > recent['MA20'])]
            for _, r in accel.iterrows():
                r20 = r['ret20']
                phase = '启动期' if r20 < 10 else ('延伸期' if r20 >= 20 else '中继期')
                records.append({'date': r['date'], 'code': code, 'phase': phase, 'future_ret': r['future_ret']})
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}]", flush=True)

    df = pd.DataFrame(records)
    print(f"\n趋势加速样本总数: {len(df)}")
    print("\n各阶段未来收益:")
    print(df.groupby('phase')['future_ret'].agg(['count', 'mean', 'median']).round(2))

    # 启动期 vs 延伸期 显著性(Mann-Whitney, 单边:启动期>延伸期)
    startup = df[df.phase == '启动期']['future_ret']
    extension = df[df.phase == '延伸期']['future_ret']
    if len(startup) > 10 and len(extension) > 10:
        from scipy import stats
        u, p = stats.mannwhitneyu(startup, extension, alternative='greater')
        print(f"\n启动期 vs 延伸期 未来{HOLD}日收益:")
        print(f"  启动期: 均值{startup.mean():+.2f}% (n={len(startup)})")
        print(f"  延伸期: 均值{extension.mean():+.2f}% (n={len(extension)})")
        print(f"  差值: {startup.mean()-extension.mean():+.2f}%")
        print(f"  Mann-Whitney p={p:.4f}")
        if p < 0.05 and startup.mean() > extension.mean():
            print("  ✅ 启动期显著优于延伸期 → +1.5/-1.0加权合理")
        elif startup.mean() > extension.mean():
            print(f"  🟡 启动期略优但不显著(p={p:.4f}) → 权重可保留但偏保守")
        else:
            print("  ❌ 延伸期反而更优 → 权重方向错误,需重新评估")


if __name__ == "__main__":
    main()
