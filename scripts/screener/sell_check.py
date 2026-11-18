#!/usr/bin/env python3
"""减仓规则检查器(3S+顶背离双确认) — 让脚本做方向判断,不靠模型直觉。

减仓逻辑(卖出):
  日线 3S 三条件(趋势反转) + 分钟 顶背离(见顶) 双确认才"可减"。
  冲突(3S满足但分钟无顶背离) → 倾向观望(CLAUDE.md约束3)。

方向铁律(不能反):
  减仓/卖出 → 查顶背离(价格新高+MACD不新高=见顶=涨不动=卖)
  买入      → 查底背离(价格新低+MACD不新低=触底=跌不动=买)

3S铁律: ①MA5死叉MA10≥3天 ②MACD柱转负/死叉 ③破MA20 同时满足。
核心: 让利润奔跑,卖出靠规则(趋势破坏+见顶)不靠直觉(涨多就减)。

用法: cd stock && PYTHONPATH=. python scripts/screener/sell_check.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
import baostock as bs
from scripts.core.tushare_client import TushareClient
from scripts.analyzers.divergence_detector import DivergenceDetector

HOLDINGS = [
    ('000100', 'TCL科技'), ('603881', '数据港'), ('002250', '联化科技'),
    ('301518', '长华化学'), ('300016', '北陆药业'), ('300404', '博济医药'),
    ('512480', '半导体ETF'), ('510500', '500ETF'), ('562500', '机器人ETF'),
    ('301110', '青木科技'), ('300455', '航天智装'),
]
_tc = TushareClient(); _det = DivergenceDetector()


def mkt(c): return 'sh' if c[:1] in ('6', '5', '9') else 'sz'


def check_3S(code):
    """日线3S三条件(趋势反转)。用TushareClient日线(避免baostock session冲突)。"""
    try:
        df = _tc.get_kline(code, freq='D', start='20260301', end='20260701')
    except Exception:
        return None
    if df is None or df.empty: return None
    for col in ['datetime', 'day', 'trade_date']:
        if col in df.columns and col != 'date': df = df.rename(columns={col: 'date'})
    if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if 'close' not in df.columns: return None
    c = df['close'].astype(float)
    if len(c) < 30: return None
    ma5 = c.rolling(5).mean(); ma10 = c.rolling(10).mean(); ma20 = c.rolling(20).mean()
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean(); hist = 2 * (dif - dea)
    cond1 = bool((ma5 < ma10).iloc[-3:].sum() >= 3)
    cond2 = bool(hist.iloc[-1] < 0 or dif.iloc[-1] < dea.iloc[-1])
    cond3 = bool(c.iloc[-1] < ma20.iloc[-1])
    triggers = sum([cond1, cond2, cond3])
    return {'price': float(c.iloc[-1]), 'ma20': float(ma20.iloc[-1]), 'cond1': cond1,
            'cond2': cond2, 'cond3': cond3, 'triggers': triggers, 'sell': triggers == 3}


def check_top_div(code):
    """60/30分钟顶背离(减仓看顶背离=见顶)。返回 (是否顶背离, 周期, strength)。"""
    for f in ['60', '30']:
        try:
            df = _tc.get_kline_minute(code, freq=f, start='20260601', end='20260701', source='baostock')
            if df is None or df.empty:
                df = _tc.get_kline_minute(code, freq=f, start='20260601', end='20260701', source='auto')
        except Exception:
            continue
        if df is None or df.empty or len(df) < 30: continue
        df = df.loc[:, ~df.columns.duplicated()]
        for col in ['datetime', 'day', 'time']:
            if col in df.columns: df = df.drop(columns=[col])
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        sig = _det.detect_divergence(df, period=f'{f}min', lookback=120)  # 分钟级120根窗(2026-08-19:内部先算后切,EMA预热)
        if sig.detected and sig.div_type == 'top':
            return True, f'{f}min', sig.strength
    return False, '-', 0


def get_market_regime():
    """大盘环境(上证20日涨跌)。回测验证: 3S上升期有效(-0.72%),震荡/下降期无效(+0.81%/+4.18%)。"""
    try:
        idx = pd.read_parquet(Path(__file__).resolve().parent.parent.parent / 'data/store/kline/000001SH/D.parquet')
        for c in ['date', 'day', 'trade_date']:
            if c in idx.columns: idx = idx.rename(columns={c: 'date'})
        idx['date'] = pd.to_datetime(idx['date']); idx = idx.sort_values('date').reset_index(drop=True)
        idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
        ret20 = (idx['close'].iloc[-1] / idx['close'].iloc[-21] - 1)
        if ret20 > 0.05:
            return '上升', ret20
        elif ret20 < -0.05:
            return '下降', ret20
        else:
            return '震荡', ret20
    except Exception:
        return '未知', 0


def main():
    regime, ret20 = get_market_regime()
    print("📊 减仓检查器(3S+顶背离双确认+大盘环境) | 减仓须: 日线3S三条件 + 分钟顶背离(见顶)")
    print("   方向铁律: 减仓看顶背离(见顶=卖), 买入看底背离(触底=买)。核心: 让利润奔跑。")
    print(f"   大盘环境: {regime}(上证20日{ret20*100:+.1f}%)", end='')
    if regime == '上升':
        print(" → 3S减仓信号有效(回测-0.72%)")
    elif regime == '下降':
        print(" → ⚠️3S在此环境是底部信号(回测+4.18%),不卖!")
    else:
        print(" → 3S在震荡期偏弱(回测+0.81%),谨慎")
    print()
    print(f"{'代码':<8}{'名称':<10}{'3S触发':>6}{'顶背离':>8}  综合判定")
    print('-' * 70)
    for code, nm in HOLDINGS:
        r5 = check_3S(code)
        if r5 is None:
            print(f"{code:<8}{nm:<10} 日线数据不足"); continue
        top, tf, tstr = check_top_div(code)
        t5 = f"{r5['triggers']}/3"
        td = f"⚠️{tf}顶背离" if top else "无"
        if r5['sell'] and top:
            if regime == '上升':
                verdict = '🔴可减仓(3S+顶背离+上升期 三确认)'
            elif regime == '下降':
                verdict = '❌不卖!(下降期3S=底部信号,回测+4.18%)'
            else:
                verdict = '🟠观望(震荡期3S偏弱,回测+0.81%)'
        elif r5['sell'] and not top:
            if regime == '上升':
                verdict = '🟠接近(上升期3S满但无顶背离确认)'
            elif regime == '下降':
                verdict = '❌不卖!(下降期3S满=底部,别割)'
            else:
                verdict = '🟠观望(震荡期,不急减)'
        elif not r5['sell'] and top:
            verdict = '⚠️注意(分钟顶背离但3S未满,盯)'
        elif r5['triggers'] >= 2:
            verdict = '🟡3S接近(2/3),观察'
        else:
            verdict = '🟢持有(趋势没坏,让利润奔跑)'
        print(f"{code:<8}{nm:<10}{t5:>6}{td:>8}  {verdict}")
    print("\n💡 3S减仓只在上升期有效(回测-0.72%)。下降期3S触发=底部信号(回测+4.18%),不卖!")
    print("   让利润奔跑=移动止损(破MA20/趋势线再走)。能拿住也是本事。")


if __name__ == '__main__':
    main()
