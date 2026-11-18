#!/usr/bin/env python3
"""波浪理论分析CLI v2.1 — 基于 ElliottWaveAnalyzer v2.1 + DataStore

自动检测浪型，支持多周期（周线/日线/60min/30min/15min/5min），K线蜡烛图画图。
数据获取统一走 DataStore（Tushare→AKShare自动降级）。

用法:
    # 最简：只给代码，日线
    python scripts/cli/wave_v2_cli.py 159227

    # 指定周期
    python scripts/cli/wave_v2_cli.py 159227 --period 60min
    python scripts/cli/wave_v2_cli.py 603881 --period W

    # 指定方向和起点
    python scripts/cli/wave_v2_cli.py 159227 --direction down --wave-start 2026-01-12

    # 指定ZigZag阈值
    python scripts/cli/wave_v2_cli.py 300699 --zz 0.04

    # 批量多周期
    python scripts/cli/wave_v2_cli.py 159227 --periods D,60min,30min

    # JSON输出
    python scripts/cli/wave_v2_cli.py 159227 --json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from scripts.analyzers.elliott_wave import ElliottWaveAnalyzer, plot_wave_chart
from scripts.core.data_store import DataStore

# ── 周期参数配置 ──
# 不同周期的ZigZag默认阈值和lookback不同
PERIOD_DEFAULTS = {
    "W":     {"zz": 0.06, "lookback": 730, "label": "周线"},
    "D":     {"zz": 0.04, "lookback": 365, "label": "日线"},
    "60min": {"zz": 0.03, "lookback": 180, "label": "60分钟"},
    "30min": {"zz": 0.025, "lookback": 120, "label": "30分钟"},
    "15min": {"zz": 0.02, "lookback": 90,  "label": "15分钟"},
    "5min":  {"zz": 0.015, "lookback": 60,  "label": "5分钟"},
}

VALID_PERIODS = list(PERIOD_DEFAULTS.keys())


def get_data(symbol, period, store=None):
    """从DataStore获取K线数据（多级fallback）。

    Fallback链路:
    1. DataStore（本地缓存 → Tushare → AKShare）
    2. Tushare直接（修复ETF后缀问题）
    3. AKShare Sina（ETF日线备用）
    """
    if store is None:
        store = DataStore()

    # 1. DataStore
    df = store.get_kline(symbol, period=period, auto_update=True)
    if df is not None and not df.empty:
        return df

    # 2. Tushare直接下载（绕过DataStore的ETF后缀bug）
    df = _fetch_tushare_direct(symbol, period)
    if df is not None and not df.empty:
        return df

    # 3. AKShare Sina（仅ETF日线）
    if period == 'D' and store._is_etf(symbol):
        df = _fetch_sina_etf(symbol)
        if df is not None and not df.empty:
            return df

    raise ValueError(f'无法获取 {symbol} {period} 数据（所有数据源均失败）')


def _fetch_tushare_direct(symbol, period):
    """Tushare直接下载，修复ETF后缀问题，分钟线加throttle。"""
    import tushare as ts
    import time
    from scripts.core.tushare_client import _freq_map
    from datetime import datetime, timedelta

    # ETF后缀修正
    if symbol.startswith(('15', '16')):
        ts_code = f'{symbol}.SZ'
        asset = 'FD'
    elif symbol.startswith(('51', '52', '50', '58')):
        ts_code = f'{symbol}.SH'
        asset = 'FD'
    else:
        from scripts.core.tushare_client import to_ts_code
        ts_code = to_ts_code(symbol)
        asset = 'E'

    freq = _freq_map(period)
    cfg = PERIOD_DEFAULTS.get(period, PERIOD_DEFAULTS["D"])
    lookback = cfg["lookback"]
    start = (datetime.now() - timedelta(days=lookback)).strftime('%Y%m%d')
    end = datetime.now().strftime('%Y%m%d')

    # 分钟线需要throttle（Tushare限制1次/小时）
    if period in ('60min', '30min', '15min', '5min'):
        time.sleep(2)  # 给一点缓冲

    try:
        if period == 'W' and asset == 'FD':
            # ETF周线：从日线重采样
            df_d = ts.pro_bar(ts_code=ts_code, freq='D',
                              start_date=start, end_date=end,
                              adj='qfq', asset=asset)
            if df_d is not None and not df_d.empty:
                df_d = df_d.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df_d['date'] = pd.to_datetime(df_d['date'])
                df_d = df_d.set_index('date')
                df_d = df_d.resample('W').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum',
                }).dropna(subset=['open'])
                return df_d.reset_index()
            return None

        df = ts.pro_bar(
            ts_code=ts_code, freq=freq,
            start_date=start, end_date=end,
            adj='qfq', asset=asset
        )
        if df is not None and not df.empty:
            col_map = {'trade_date': 'date', 'trade_time': 'date',
                       'vol': 'volume', 'pct_chg': 'change_pct'}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        err = str(e)
        if '频率超限' in err:
            print(f'  Tushare {period}分钟线频率限制（1次/小时），稍后重试')
        else:
            print(f'  Tushare直接下载失败: {e}')
    return None


def _fetch_sina_etf(symbol):
    """通过AKShare Sina接口获取ETF日线（非前复权，备用源）。"""
    import os
    os.environ['NO_PROXY'] = '*'
    os.environ['no_proxy'] = '*'
    try:
        import akshare as ak
        # Sina用sz/sh前缀
        prefix = 'sz' if symbol.startswith(('15', '16')) else 'sh'
        df = ak.fund_etf_hist_sina(symbol=f'{prefix}{symbol}')
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f'  Sina ETF日线失败: {e}')
    return None


def run_analysis(symbol, period, direction='auto', wave_start=None,
                 zz=None, output=None, dpi=150, json_output=False,
                 quiet=False, store=None):
    """运行单次波浪分析。

    Args:
        symbol: 6位代码
        period: W/D/60min/30min/15min/5min
        direction: up/down/auto
        wave_start: 浪型起始日期
        zz: ZigZag阈值（None=使用周期默认值）
        output: 图片输出路径
        store: DataStore实例（可复用）
    """
    cfg = PERIOD_DEFAULTS.get(period, PERIOD_DEFAULTS["D"])
    if zz is None:
        zz = cfg["zz"]

    # 获取数据
    df = get_data(symbol, period, store)

    if len(df) < 30:
        if not quiet:
            print(f'数据太少({len(df)}条)，至少需要30条')
        return None

    # 分析
    analyzer = ElliottWaveAnalyzer(zigzag_pct=zz)
    result = analyzer.analyze(
        df,
        direction=direction,
        start_date=wave_start,
    )

    if not result.waves:
        if not quiet:
            print(f'未检测到完整浪型结构（{cfg["label"]} zz={zz}）')
            print(f'建议：调整 --zz 参数或指定 --wave-start')
        return None

    # 格式化输出
    if not quiet:
        print(f'\n{"="*65}')
        print(f'  {symbol} {cfg["label"]}({period}) 波浪分析')
        print(f'{"="*65}')
        print(ElliottWaveAnalyzer.format_result(result))

    # 画图
    if output is None:
        output_dir = os.path.expanduser('~/.openclaw/workspace/output')
        os.makedirs(output_dir, exist_ok=True)
        suffix = '' if period == 'D' else f'_{period}'
        output = os.path.join(output_dir, f'wave_{symbol}{suffix}.png')

    plot_wave_chart(df, result, analyzer, output,
                    title=f'{symbol} {cfg["label"]}({period})',
                    dpi=dpi)

    if not quiet:
        print(f'\n📊 图表: {output}')

    # JSON输出
    if json_output and result:
        out = {
            'symbol': symbol,
            'period': period,
            'period_label': cfg['label'],
            'direction': result.direction,
            'confidence': result.confidence,
            'total_score': result.total_score,
            'fib_score': result.fib_score,
            'time_score': result.time_score,
            'volume_score': result.volume_score,
            'waves': [w.to_dict() for w in result.waves],
            'iron_rules': [{'name': n, 'passed': p, 'detail': d}
                           for n, p, d in result.iron_rules],
            'current_wave': result.current_wave,
            'chart_path': output,
            'data_rows': len(df),
            'date_range': f'{df["date"].iloc[0]} ~ {df["date"].iloc[-1]}',
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(
        description='波浪理论分析CLI v2.1（多周期支持）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s 159227                          # 日线分析
  %(prog)s 159227 --period 60min           # 60分钟线
  %(prog)s 159227 --periods D,60min,30min  # 批量多周期
  %(prog)s 159227 --direction down --wave-start 2026-01-12
  %(prog)s 603881 --period W --zz 0.06     # 周线，大阈值
        ''')
    parser.add_argument('code', help='股票/ETF代码（如 159227, 603881）')
    parser.add_argument('--period', default='D', choices=VALID_PERIODS,
                        help='K线周期（默认D=日线）')
    parser.add_argument('--periods', default=None,
                        help='批量多周期，逗号分隔（如 D,60min,30min）')
    parser.add_argument('--direction', choices=['up', 'down', 'auto'], default='auto',
                        help='浪型方向（默认auto自动判断）')
    parser.add_argument('--wave-start', default=None,
                        help='数浪起始日期 (YYYY-MM-DD)')
    parser.add_argument('--zz', type=float, default=None,
                        help='ZigZag阈值（默认按周期自动选取）')
    parser.add_argument('-o', '--output', default=None,
                        help='输出图片路径')
    parser.add_argument('--dpi', type=int, default=150, help='图片DPI')
    parser.add_argument('--json', action='store_true', help='输出JSON结果')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    args = parser.parse_args()

    # 标准化代码
    symbol = args.code.strip()
    store = DataStore()

    # 批量多周期
    if args.periods:
        periods = [p.strip() for p in args.periods.split(',')]
        results = {}
        for p in periods:
            if p not in VALID_PERIODS:
                print(f'跳过未知周期: {p}')
                continue
            if not args.quiet:
                print(f'\n>>> 分析 {symbol} {PERIOD_DEFAULTS[p]["label"]}({p})...')
            r = run_analysis(
                symbol, p,
                direction=args.direction,
                wave_start=args.wave_start,
                zz=args.zz,
                output=args.output,
                dpi=args.dpi,
                json_output=args.json,
                quiet=args.quiet,
                store=store,
            )
            results[p] = r
        return

    # 单周期
    run_analysis(
        symbol, args.period,
        direction=args.direction,
        wave_start=args.wave_start,
        zz=args.zz,
        output=args.output,
        dpi=args.dpi,
        json_output=args.json,
        quiet=args.quiet,
        store=store,
    )


if __name__ == '__main__':
    main()
