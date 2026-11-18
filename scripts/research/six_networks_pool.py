#!/usr/bin/env python3
"""六张网空白方向候选池(2026-08-30)。

背景: HALO(华尔街重资产)+六张网(中国版7万亿)同源逻辑;组合空白=电网/物流/管网水/通信
(算力已有数据港)。本脚本只做客观计算:位置+估值+成长,不追高,给回踩条件。

输出: 网络|名称|价|250日分位|vsMA20|距20日高|PE|股息|扣非增速|毛利率|ROE
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, '.')

# ═══ 候选池: 六张网空白四网,央国企/成熟龙头优先 ═══
CANDIDATES = {
    # code: (名称, 网络)
    '600406': ('国电南瑞', '电网'),
    '600089': ('特变电工', '电网'),
    '002028': ('思源电气', '电网'),
    '600312': ('平高电气', '电网'),
    '601985': ('中国核电', '电网'),
    '600900': ('长江电力', '电网'),
    '601006': ('大秦铁路', '物流'),
    '601816': ('京沪高铁', '物流'),
    '601919': ('中远海控', '物流'),
    '603871': ('嘉友国际', '物流'),
    '002372': ('伟星新材', '管网水'),
    '002457': ('青龙管业', '管网水'),
    '000598': ('兴蓉环境', '管网水'),
    '601158': ('重庆水务', '管网水'),
    '601139': ('深圳燃气', '管网水'),
    '600941': ('中国移动', '通信'),
    '601728': ('中国电信', '通信'),
    '600522': ('中天科技', '通信'),
    '600487': ('亨通光电', '通信'),
    '600905': ('三峡能源', '电网'),
}

CACHE_DIR = 'data/store/sixnet_fina'


def position_metrics(code):
    """本地kline: 250日分位/vs MA20/距20日高"""
    path = f'data/store/kline/{code}/D.parquet'
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values('date')
    if len(df) < 60:
        return None
    c = df['close'].astype(float)
    close = c.iloc[-1]
    w = c.tail(250)
    pct = (close - w.min()) / (w.max() - w.min()) if w.max() > w.min() else 0.5
    ma20 = c.tail(20).mean()
    ma60 = c.tail(60).mean()
    high20 = df['high'].astype(float).tail(20).max()
    last_date = str(df['date'].iloc[-1])[:10]
    return dict(close=round(close, 2), pct250=round(pct * 100, 1),
                vs_ma20=round((close / ma20 - 1) * 100, 1),
                vs_ma60=round((close / ma60 - 1) * 100, 1),
                d_high20=round((close / high20 - 1) * 100, 1),
                date=last_date)


def fetch_valuation(trade_date='20260828'):
    """全市场估值快照一次拉取"""
    from scripts.core.tushare_client import TushareClient
    pro = TushareClient().pro
    df = pro.daily_basic(trade_date=trade_date,
                         fields='ts_code,close,pe_ttm,pb,dv_ttm,total_mv')
    return df.set_index('ts_code')


def fetch_financials(code):
    """fina_indicator: 扣非/毛利率/ROE(本地缓存sixnet_fina)"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = f'{CACHE_DIR}/{code}.parquet'
    if os.path.exists(cpath):
        df = pd.read_parquet(cpath)
    else:
        from scripts.core.tushare_client import TushareClient
        pro = TushareClient().pro
        dfs = []
        for _ in range(3):
            try:
                df = pro.fina_indicator(ts_code=f'{code}.' + ('SH' if code[0] == '6' else 'SZ'),
                                        fields='end_date,ann_date,update_flag,profit_dedt,'
                                               'grossprofit_margin,netprofit_margin,roe,'
                                               'debt_to_assets,ocf_to_profit,or_yoy')
                dfs.append(df)
                break
            except Exception:
                import time
                time.sleep(1)
        if not dfs:
            return None
        df = pd.concat(dfs)
        df.to_parquet(cpath)
    return df


def dedt_growth(df):
    """最新中报扣非同比(累计口径,update_flag=1最新)"""
    if df is None or df.empty:
        return None, None
    if 'update_flag' in df.columns and df['update_flag'].notna().any():
        # 注意: parquet缓存里update_flag是字符串'1'/'0'
        df = df[df['update_flag'].astype(str) == '1']
    if df.empty:
        return None, None
    df = df.dropna(subset=['end_date']).sort_values('end_date')
    if df.empty:
        return None, None
    # 找最新年报/中报期与去年同期
    ed = df['end_date'].astype(str)
    latest = df.iloc[-1]
    end = str(latest['end_date'])
    yoy_period = str(int(end[:4]) - 1) + end[4:]
    prev = df[ed == yoy_period]
    g = np.nan
    if not prev.empty and pd.notna(latest.get('profit_dedt')) and pd.notna(prev.iloc[0].get('profit_dedt')):
        p0, p1 = float(prev.iloc[0]['profit_dedt']), float(latest['profit_dedt'])
        if p0 != 0:
            g = (p1 / abs(p0) - 1) * 100 if p0 > 0 else (np.nan if p1 <= 0 else np.nan)
    return end, round(g, 1) if pd.notna(g) else None


def main():
    val = fetch_valuation()
    rows = []
    for code, (name, net) in CANDIDATES.items():
        pos = position_metrics(code)
        ts = code + ('.SH' if code[0] == '6' else '.SZ')
        v = val.loc[ts] if ts in val.index else None
        end, g = dedt_growth(fetch_financials(code))
        rows.append({
            '网络': net, '代码': code, '名称': name,
            '价': pos['close'] if pos else None,
            '250分位': pos['pct250'] if pos else None,
            'vsMA20%': pos['vs_ma20'] if pos else None,
            'vsMA60%': pos['vs_ma60'] if pos else None,
            '距20日高': pos['d_high20'] if pos else None,
            'PE': round(float(v['pe_ttm']), 1) if v is not None and pd.notna(v['pe_ttm']) else None,
            '股息%': round(float(v['dv_ttm']), 1) if v is not None and pd.notna(v['dv_ttm']) else None,
            '扣非增%': g, '报告期': end,
            'k线日': pos['date'] if pos else None,
        })
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print("=" * 110)
    print("六张网空白方向候选池 | 2026-08-28收盘 | 20只")
    print("=" * 110)
    print(out.to_string(index=False))
    out.to_parquet('data/store/sixnet_pool.parquet')

    print("\n顶部禁买令检查(距20日高>-5% = 高位冻结):")
    frozen = out[(out['距20日高'] > -5)]
    for _, r in frozen.iterrows():
        print(f"  🔴 {r['名称']}({r['代码']}) 距高{r['距20日高']}% — 在±5%禁区,禁买")
    print("\n低位可研究(250分位<40%):")
    low = out[(out['250分位'] < 40)].sort_values('250分位')
    for _, r in low.iterrows():
        print(f"  🟢 {r['名称']}({r['代码']}) 分位{r['250分位']}% vsMA20 {r['vsMA20%']}% PE{r['PE']}")


if __name__ == '__main__':
    main()
