# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

#!/usr/bin/env python3
"""弹簧起涨扫描器 v3 — 回测验证版。

回测验证结论（1-5月全量滚动，82只信号，10轮扫描）：
  1. 红柱100%是硬门槛：<100%红柱1日胜率仅17%（vs 100%红柱69%）
  2. 最佳持有期=1日（夏普比0.16，5日0.04，10日-0.03）
  3. 止损-3%有效：62%触发止损，避免亏损从-3%扩大到-6%+
  4. 未止损的票表现优秀：1日84%胜率，5日74%胜率
  5. 1月最强(85%胜率)，3-4月最弱(0-25%)，需大盘择时配合
  6. 评分≥50分对胜率无明显区分力，红柱100%是唯一硬条件

v2→v3 修改清单：
  - 红柱阈值从70%提高到100%（硬门槛）
  - 移除v3评分依赖（v3在回测中无区分力）
  - 压缩比完全移除（与结果负相关）
  - 简化评分为：红柱100% + 量比 + 方向确认
  - 加入推荐持有期和止损提示
  - 大盘择时标记（需外部提供）

用法：
  PYTHONPATH=. python scripts/screener/spring_scanner.py
  PYTHONPATH=. python scripts/screener/spring_scanner.py --top 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calc_macd_red_ratio(close: pd.Series, fast=12, slow=26, signal=9, lookback=20) -> float:
    """MACD红柱占比 = 近lookback根K线中红柱的比例"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    recent = macd.tail(lookback)
    if len(recent) == 0:
        return 0
    return (recent > 0).sum() / len(recent)


def calc_vol_surge(volume: pd.Series, lookback=20) -> float:
    """量比 = 最近5日均量 / 20日均量"""
    if len(volume) < lookback + 5:
        return 0
    recent_5 = volume.tail(5).mean()
    avg_20 = volume.tail(lookback).mean()
    if avg_20 == 0:
        return 0
    return recent_5 / avg_20


def check_red_fading(close: pd.Series) -> bool:
    """检查MACD红柱是否在衰竭（近3根 vs 前3根，衰减>40%）"""
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    if len(macd) < 8:
        return False
    recent_3 = macd.iloc[-3:].values
    prev_3 = macd.iloc[-6:-3].values
    if all(r > 0 for r in recent_3) and all(r > 0 for r in prev_3):
        avg_r = recent_3.mean()
        avg_p = prev_3.mean()
        if avg_p > 0 and avg_r < avg_p * 0.6:
            return True
    return False


def scan_spring_stocks(
    data_root: str = 'data/store',
    max_ret20: float = 10.0,
    max_ret5: float = 3.0,
    min_vol_surge: float = 1.3,
    top_n: int = 20,
    require_above_ma20: bool = True,
) -> pd.DataFrame:
    """扫描弹簧起涨股票 v3（回测验证版）。

    核心筛选条件（仅保留回测验证有效的）：
      - MACD红柱100%（<100%胜率仅17%，硬门槛）
      - 量比 > 1.3
      - close > MA20（方向确认）
      - ret5 < 3%（防止追高）
      - ret20 < 10%（起涨初期）
    """
    store = Path(data_root)

    codes = sorted([
        d.name for d in (store / 'kline').iterdir()
        if d.is_dir() and (d / 'D.parquet').exists()
    ])

    results = []

    for code in codes:
        try:
            dp = store / 'kline' / code / 'D.parquet'
            df = pd.read_parquet(dp)
            if len(df) < 60:
                continue

            # 标准化日期
            if 'trade_date' in df.columns:
                df = df.sort_values('trade_date').reset_index(drop=True)
            elif 'datetime' in df.columns:
                df = df.sort_values('datetime').reset_index(drop=True)
            elif 'date' in df.columns:
                df = df.sort_values('date').reset_index(drop=True)
            else:
                continue

            close = df['close']
            vol_col = 'vol' if 'vol' in df.columns else ('volume' if 'volume' in df.columns else None)
            if vol_col is None:
                continue
            volume = df[vol_col]
            high = df['high']
            low = df['low']

            if len(close) < 60:
                continue

            # ===== 硬筛选（回测验证有效） =====

            # 1. MACD红柱100%（硬门槛！<100%胜率17%）
            red_ratio = calc_macd_red_ratio(close, lookback=20)
            if red_ratio < 1.0:  # 必须满格
                continue

            # 2. 红柱衰竭检测
            red_fading = check_red_fading(close)

            # 3. 量比
            vol_surge = calc_vol_surge(volume)
            if pd.isna(vol_surge) or vol_surge < min_vol_surge:
                continue

            # 4. 涨幅约束
            ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
            ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0
            if ret20 > max_ret20 or ret5 > max_ret5:
                continue

            # 5. 方向确认
            ma20 = close.rolling(20).mean()
            current_ma20 = ma20.iloc[-1]
            if pd.isna(current_ma20):
                continue
            above_ma20 = close.iloc[-1] > current_ma20
            if require_above_ma20 and not above_ma20:
                continue

            # ===== 简化评分（回测验证：评分对胜率无区分力，红柱100%是核心） =====
            score = 0

            # 量能（唯一有区分力的维度）
            if vol_surge > 2.0:
                score += 40
            elif vol_surge > 1.7:
                score += 30
            elif vol_surge > 1.5:
                score += 20
            elif vol_surge > 1.3:
                score += 10

            # 红柱衰竭惩罚
            if red_fading:
                score -= 20

            # 方向确认
            if above_ma20:
                score += 10

            # ret5最优区间
            if 0 < ret5 <= 2:
                score += 10

            ma20_dist = (close.iloc[-1] / current_ma20 - 1) * 100

            results.append({
                'code': code,
                'close': round(close.iloc[-1], 2),
                'score': score,
                'macd_red_pct': round(red_ratio * 100, 1),
                'vol_surge': round(vol_surge, 2) if not pd.isna(vol_surge) else 0,
                'ret5': round(ret5, 2),
                'ret20': round(ret20, 2),
                'above_ma20': above_ma20,
                'ma20_dist': round(ma20_dist, 1),
                'red_fading': red_fading,
                'high_20d': round(high.tail(20).max(), 2),
                'low_20d': round(low.tail(20).min(), 2),
            })

        except Exception:
            continue

    if not results:
        print('📭 无弹簧起涨信号')
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False).head(top_n)
    return df_results


def main():
    parser = argparse.ArgumentParser(description='弹簧起涨扫描器 v3 — 回测验证版')
    parser.add_argument('--max-ret20', type=float, default=10.0, help='最大20日涨幅%%')
    parser.add_argument('--max-ret5', type=float, default=3.0, help='最大5日涨幅%%')
    parser.add_argument('--min-vol', type=float, default=1.3, help='最低量比')
    parser.add_argument('--top', type=int, default=20, help='输出Top N')
    parser.add_argument('--data-root', default='data/store')
    parser.add_argument('--no-ma20', action='store_true', help='不要求close>MA20')
    args = parser.parse_args()

    print(f'🔍 弹簧起涨扫描 v3（红柱=100% | 量比>{args.min_vol} | ret5<{args.max_ret5}%）')
    print(f'   📊 回测验证：红柱100% 1日胜率69% | 推荐：次日卖出 + 止损-3%')
    print()

    df = scan_spring_stocks(
        data_root=args.data_root,
        max_ret20=args.max_ret20,
        max_ret5=args.max_ret5,
        min_vol_surge=args.min_vol,
        top_n=args.top,
        require_above_ma20=not args.no_ma20,
    )

    if df.empty:
        return

    print(f'🎯 弹簧起涨信号: {len(df)}只\n')
    print(f'{"#":>3} {"代码":>8} {"评分":>4} {"量比":>5} {"5日%":>6} {"20日%":>6} {"MA20":>6} {"现价":>7}')
    print('-' * 55)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        ma20_flag = '✅' if r['above_ma20'] else '❌'
        fade = ' ⚠️衰竭' if r['red_fading'] else ''
        print(f'{i:>3} {r["code"]:>8} {r["score"]:>4.0f} {r["vol_surge"]:>4.1f}x '
              f'{r["ret5"]:>+5.1f}% {r["ret20"]:>+5.1f}% '
              f'{ma20_flag}{r["ma20_dist"]:>+4.1f}% {r["close"]:>7.2f}{fade}')
        print(f'     20日区间=[{r["low_20d"]:.2f}, {r["high_20d"]:.2f}] | 止损={r["close"]*0.97:.2f}')

    print(f'\n💡 操作建议：')
    print(f'   - 持有1日（次日收盘卖），回测胜率69%')
    print(f'   - 严格止损-3%（回测62%触发止损，有效控制亏损）')
    print(f'   - 红柱衰竭标记⚠️的票谨慎')


if __name__ == '__main__':
    main()
