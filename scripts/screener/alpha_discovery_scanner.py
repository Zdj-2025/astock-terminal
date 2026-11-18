#!/usr/bin/env python3
"""发现型选股 — 找别人没发现的机会(2026-06-21)。

核心逻辑: 主力在散户没察觉时收集筹码(股东户数环比↓>10%, ICIR=-1.0062验证),
股价还没动(距60日高点低, 低位未涨) → 别人没发现的早期机会。

与龙头pipeline(共识/beta)互补: 这是非共识alpha来源。
- 筹码集中(holder_chg_pct<-10%): 主力潜伏, 散户看不到(只季报披露)
- 低位未涨(距60日高点<-5%): 主力收集完还没拉升(若已大涨=已被发现, 排除)
- 避雷: 排除ST/低价<2元(面值退市风险)

数据: 复用 research holder 研究(_cache_metrics.parquet, 2026Q1最新) + data/store/kline日线。
时效: 股东户数因子预测力仅1季度, 2026Q1信号4/29披露, 7月底前有效, 注意衰减。

用法: cd stock && PYTHONPATH=. python scripts/screener/alpha_discovery_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.core.tushare_client import TushareClient

METRICS_CACHE = PROJECT_ROOT / 'scripts/research/output/_cache_metrics.parquet'
STORE = PROJECT_ROOT / 'data/store/kline'
HOLDER_THRESHOLD = -10.0   # 股东户数环比变化%(减少>10%=筹码集中信号)
DIST_HI_THRESHOLD = -10.0  # 距60日高点%(<-10%=真低位, 排除大涨后小回调如德福)
STABLE_RET20_MIN = -5.0    # 近20日涨幅下限(>-5%=跌透企稳, 排除下跌中继)
STABLE_RET20_MAX = 20.0    # 近20日涨幅上限(<20%=没大涨过, 排除已拉升被发现)
LOW_PRICE = 2.0            # 低价面值退市风险阈值


def get_holder_concentration():
    """取每股最新季度的股东户数变化%(metrics缓存, 已算好holder_chg_pct)。"""
    m = pd.read_parquet(METRICS_CACHE)
    m['quarter'] = m['quarter'].astype(str)
    latest_q = m['quarter'].max()
    latest = m[m['quarter'] == latest_q].copy()
    latest['code'] = latest['ts_code'].str[:6]  # 000725.SZ -> 000725
    return latest, latest_q


def get_price_position(codes):
    """用日线算每只股距60日高点位置(低位未涨判断) + 近20日涨幅。"""
    rows = []
    for code in codes:
        p = STORE / code / 'D.parquet'
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=['date', 'close'])
            if len(df) < 60:
                continue
            df = df.sort_values('date').reset_index(drop=True)
            close = float(df['close'].iloc[-1])
            hi60 = float(df['close'].iloc[-60:].max())
            ret20 = (close / float(df['close'].iloc[-21]) - 1) * 100 if len(df) >= 21 else 0
            rows.append({
                'code': code, 'close': close,
                'dist_hi60': (close / hi60 - 1) * 100,
                'ret20': ret20,
                'date': str(df['date'].iloc[-1])[:10],
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def main():
    sl = TushareClient().get_stock_list()
    sym = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sym, sl['name']))

    latest, latest_q = get_holder_concentration()
    print(f"📊 发现型选股 | 筹码集中(holder↓) + 低位未涨 + 避雷")
    print(f"   股东户数报告期: {latest_q}(披露~4/29, 因子窗口7月底前有效)\n")

    # ① 筹码集中
    conc = latest[latest['holder_chg_pct'] < HOLDER_THRESHOLD].copy()
    print(f"【① 筹码集中(股东户数环比<{HOLDER_THRESHOLD}%)=主力潜伏】{len(conc)}只")

    # ② 低位未涨(日线)
    price_df = get_price_position(conc['code'].tolist())
    if price_df.empty:
        print("  (无日线数据)"); return
    merged = conc.merge(price_df, on='code', how='inner')
    low = merged[merged['dist_hi60'] < DIST_HI_THRESHOLD].copy()
    print(f"【② +真低位(距60日高点<{DIST_HI_THRESHOLD}%)】{len(low)}只")
    # ②b 企稳: 近20日没大跌(跌透企稳)也没大涨(没被发现)
    stable = low[(low['ret20'] >= STABLE_RET20_MIN) & (low['ret20'] <= STABLE_RET20_MAX)].copy()
    print(f"【②b +企稳(ret20 {STABLE_RET20_MIN}%~{STABLE_RET20_MAX}%, 跌透未大涨)】{len(stable)}只")

    # ③ 避雷: ST + 低价
    stable['name'] = stable['code'].map(name_map).fillna('')
    stable['circ_mv_yi'] = stable['avg_circ_mv'] / 10000.0  # 万元→亿元
    clean = stable[~stable['name'].str.contains('ST', na=False)].copy()
    n_before = len(clean)
    clean = clean[clean['close'] >= LOW_PRICE].copy()
    print(f"【③ 避雷(排除ST/低价<{LOW_PRICE}元)】剔除{n_before - len(clean)}只 → {len(clean)}只")

    # 排序: 筹码集中度优先(holder_chg_pct最负), 次低位
    clean = clean.sort_values(['holder_chg_pct', 'dist_hi60']).head(25)

    print(f"\n【✅ 发现型机会(别人没发现: 筹码集中+低位未涨+避雷)】Top{len(clean)}:")
    print(f"{'代码':<7}{'名称':<10}{'价':>7}{'距60高':>8}{'ret20':>8}{'holder变化':>11}{'流通市值':>9}{'日线':>12}")
    for _, r in clean.iterrows():
        print(f"{r['code']:<7}{str(r['name'])[:9]:<10}{r['close']:>7.2f}"
              f"{r['dist_hi60']:>+7.1f}%{r['ret20']:>+7.1f}%{r['holder_chg_pct']:>+10.1f}%"
              f"{r['circ_mv_yi']:>8.0f}亿{r['date']:>12}")
    print(f"\n💡 逻辑: holder↓(主力收集) + 距60高低位(还没拉升) = 别人没发现的早期机会")
    print(f"⚠️ 时效: 2026Q1信号7月底前有效; 买入前仍需查公告催化+分钟背离(6维度自检)")


if __name__ == '__main__':
    main()
