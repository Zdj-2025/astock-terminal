# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

#!/usr/bin/env python3
"""TCL模式扫描器+回测 — 趋势加速回踩中继(2026-06-17, TCL科技6/1推荐复盘提炼)。

模式: 拉升→回踩MA20/前低→MACD金叉→再涨。比底部反转有动量, 比追高安全。
TCL 6/1: V3主升浪 f1+0.614 f2+4.220 回踩MA20金叉 → 推荐6/1后最高+17%。

═══ 回测严格无未来 ═══
- 信号指标全用 rolling/ewm(无center): 每行T的值只用T及之前数据
- is_tcl 只用当前行指标(不跨行偷未来)
- future_ret = close.shift(-hold)/close - 1: 用T之后价格, 但仅"衡量"信号表现,
  信号判定时绝无未来信息。这是合法的(回测就是用未来收益验证历史信号)。

用法:
    PYTHONPATH=. python scripts/screener/tcl_pattern.py --backtest       # 回测验证
    PYTHONPATH=. python scripts/screener/tcl_pattern.py --backtest --hold 5,10,20
    PYTHONPATH=. python scripts/screener/tcl_pattern.py --scan            # 当前扫描
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

STORE = PROJECT_ROOT / "data" / "store" / "kline"


def compute_indicators(df):
    """算无未来指标。rolling/ewm均无center → 每行T只用T及之前数据。"""
    df = df.sort_values('date').reset_index(drop=True).copy()
    close = df['close']
    df['MA5'] = close.rolling(5).mean()
    df['MA10'] = close.rolling(10).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA60'] = close.rolling(60).mean()
    ef = close.ewm(span=12, adjust=False).mean()
    es = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = ef - es
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    # V3真实f1/f2: savgol对ln(price)求导, 内部已shift防look-ahead(与V3 scorer一致, 无未来)
    from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
    _f1, _f2 = V3DerivativeScorer.compute_derivatives(close.values)
    df['f1'] = _f1
    df['f2'] = _f2
    df['hi60'] = df['high'].rolling(60).max()  # 60日高(无未来)
    df['dh60'] = close / df['hi60'] - 1
    return df


def tcl_mask(df):
    """TCL模式向量化的信号mask(无未来, 只用当前行)。"""
    c = df['close']
    return (
        (df['f1'] > 0.3) &                     # 趋势加速(强涨)
        (df['f2'] > 0) &                        # 加速上行
        (c > df['MA20']) &                      # above MA20
        (c / df['MA20'] >= 0.97) & (c / df['MA20'] <= 1.05) &  # 回踩MA20附近
        (c / df['MA5'] <= 1.0) &                # 不超MA5(回踩中, 未冲高)
        (df['DIF'] > df['DEA']) &               # MACD金叉
        (df['dh60'] >= -0.30) & (df['dh60'] <= -0.05)  # 位置中等(距60日高-5%~-30%)
    )


def load_codes():
    return sorted([d.name for d in STORE.iterdir() if (d / 'D.parquet').exists()])


def backtest(hold_list=(5, 10, 20)):
    """回测(无未来): 逐日TCL信号, 各持有期收益。"""
    codes = load_codes()
    print(f"回测 {len(codes)}只 | 持有{hold_list}日 | 严格无未来(信号用T及之前, 收益用T+hold衡量)")
    max_hold = max(hold_list)
    sig_rows = []
    for i, code in enumerate(codes):
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            df['date'] = pd.to_datetime(df['date'])
            df = compute_indicators(df)
            df['signal'] = tcl_mask(df)
            # 各持有期未来收益(仅衡量, 信号判定无未来)
            for h in hold_list:
                df[f'ret{h}'] = df['close'].shift(-h) / df['close'] - 1
            sig = df[df['signal']]
            for _, r in sig.iterrows():
                row = {'date': r['date'], 'code': code, 'price': r['close']}
                for h in hold_list:
                    row[f'ret{h}'] = r[f'ret{h}']
                sig_rows.append(row)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(codes)}] 信号累计{len(sig_rows)}", flush=True)

    rec = pd.DataFrame(sig_rows)
    if rec.empty:
        print("\n无信号(条件可能过严)"); return
    print(f"\n{'='*60}\nTCL模式回测结果 | 信号数 {len(rec)}\n{'='*60}")
    from scipy import stats
    for h in hold_list:
        col = f'ret{h}'
        s = rec[col].dropna()
        if len(s) < 5:
            continue
        win = (s > 0).mean() * 100
        t, p = stats.ttest_1samp(s, 0)
        mean, se = s.mean(), s.std() / len(s) ** 0.5
        print(f"持有{h:>2}日: 均值{s.mean()*100:+6.2f}% 胜率{win:5.1f}% n={len(s):4d} "
              f"| 95%CI[{mean-1.96*se:+.2%},{mean+1.96*se:+.2%}] p={p:.4f} {'✅' if p<0.05 else '❌'}")
    rec['year'] = rec['date'].dt.year
    print("\n按年(持有10日):")
    print(rec.groupby('year')['ret10'].agg(['count', 'mean']).round(3))
    rec.to_csv(PROJECT_ROOT / 'reports' / 'tcl_pattern_backtest.csv', index=False)
    print(f"\n信号明细 → reports/tcl_pattern_backtest.csv")


def scan():
    """当前扫描: 最新行满足TCL模式的股。"""
    codes = load_codes()
    print(f"扫描 {len(codes)}只 | 找当前TCL模式(趋势加速回踩中继)")
    # 名称(带名称输出, 用户偏好)
    name_map = {}
    try:
        from scripts.core.tushare_client import TushareClient
        _sl = TushareClient().get_stock_list()
        for _, _r in _sl.iterrows():
            name_map[str(_r.get('symbol', _r.get('ts_code', ''))).split('.')[0]] = _r.get('name', '')
    except Exception:
        pass
    hits = []
    for code in codes:
        try:
            df = pd.read_parquet(STORE / code / 'D.parquet')
            df['date'] = pd.to_datetime(df['date'])
            df = compute_indicators(df)
            if tcl_mask(df).iloc[-1]:
                last = df.iloc[-1]
                hits.append({'code': code, 'name': name_map.get(code, ''),
                             'date': last['date'].date(), 'price': last['close'],
                             'f1': round(last['f1'], 3), 'dh60': f"{last['dh60']*100:+.1f}%",
                             'vsMA20': f"{last['close']/last['MA20']*100-100:+.1f}%"})
        except Exception:
            pass
    hits = sorted(hits, key=lambda x: x['f1'], reverse=True)
    print(f"\n=== 当前TCL模式候选: {len(hits)}只 ===")
    for h in hits[:25]:
        print(f"  {h['code']} {h['name']:<8} 价{h['price']:.2f} f1={h['f1']:+.2f} dh60={h['dh60']} vsMA20={h['vsMA20']}")
    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCL模式(趋势加速回踩中继)扫描+回测")
    parser.add_argument("--backtest", action="store_true", help="回测验证(无未来)")
    parser.add_argument("--scan", action="store_true", help="当前扫描")
    parser.add_argument("--hold", default="5,10,20", help="持有期, 逗号分隔")
    args = parser.parse_args()
    if args.backtest:
        holds = tuple(int(x) for x in args.hold.split(','))
        backtest(holds)
    elif args.scan:
        scan()
    else:
        parser.print_help()
