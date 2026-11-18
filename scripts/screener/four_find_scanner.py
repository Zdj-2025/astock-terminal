#!/usr/bin/env python3
"""四找scanner(独立模块) — 一杆清"4找"选股法,与"交集"解耦。
设计(2026-08-10): 交集随板块/概念不同而变(软件∩AI应用 / 元器件∩汽车电子 / ...),
所以4找独立——输入任意股票代码列表(来自交集/行业/任意筛选),输出4找结果。

4找(一杆清):
  ①成交额前N   主力资金关照
  ②总市值前N   龙头/带板块指数(强者恒强)
  ③涨幅前N     ret5+当日,涨好的会涨更好
  ④技术        均线多头(MA5>MA20>MA60)+量比+突破120高平台
输出: ①②③各TOP10 + 4找交集(①∩②∩③) + ④技术补充

用法:
  python four_find_scanner.py --codes 688111,300033,002230        # 4找指定列表(交集结果)
  python four_find_scanner.py --industry 软件服务,软件开发        # 4找整个行业
  python four_find_scanner.py --industry 软件服务,软件开发 --top 30
"""
import pandas as pd, numpy as np, os, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')


def load_universe():
    return pd.read_parquet('data/store/stock_list.parquet')


def get_codes(args, sl):
    """交集解耦: codes来自外部(交集结果/行业/任意)."""
    if args.codes:
        return [c.strip().zfill(6) for c in args.codes.split(',') if c.strip()]
    if args.industry:
        inds = [i.strip() for i in args.industry.split(',')]
        return sl[sl['industry'].isin(inds)]['symbol'].tolist()
    return []


def four_find(codes, top_n=30):
    """4找核心: 输入code列表, 返回含4找标注的DataFrame."""
    sl = load_universe()
    nm = dict(zip(sl['symbol'], sl['name']))
    ind = dict(zip(sl['symbol'], sl['industry']))
    st = set(sl.loc[sl['name'].astype(str).str.contains('ST'), 'symbol'])
    db = pd.read_parquet('data/store/daily_basic')
    db_l = db.sort_values('trade_date').drop_duplicates('ts_code', keep='last')
    mv = dict(zip(db_l['ts_code'].astype(str).str[:6], db_l['total_mv']))

    recs = []
    for code in codes:
        if code in st or code[0] not in '036':
            continue
        try:
            d = pd.read_parquet(f'data/store/kline/{code}/D.parquet').sort_values('date').reset_index(drop=True)
        except Exception:
            continue
        if len(d) < 60:
            continue
        c = d['close'].values.astype(float)
        last = d.iloc[-1]
        ma5, ma20 = c[-5:].mean(), c[-20:].mean()
        ma60 = c[-60:].mean()
        high120 = c[-120:].max() if len(c) >= 120 else c.max()
        vol_col = 'vol' if 'vol' in d.columns else 'volume'
        vol_ratio = float(d[vol_col].iloc[-1] / d[vol_col].tail(20).mean())
        # ④技术: 均线多头 + 量比 + 突破平台
        多头 = ma5 > ma20 > ma60
        突破 = c[-1] >= high120 * 0.95
        tech4 = 多头 and vol_ratio > 1.0
        recs.append({'code': code, 'name': nm.get(code, ''), 'industry': ind.get(code, ''),
                     'amount': float(last['amount']), 'chg': float(last['change_pct']),
                     'ret5': (c[-1] / c[-6] - 1) * 100, 'mv': mv.get(code, 0),
                     'vol_ratio': round(vol_ratio, 2), '多头': 多头, '突破120': 突破, 'tech4': tech4})
    df = pd.DataFrame(recs)
    if len(df) == 0:
        return df
    n = min(top_n, len(df))
    ta, tm, tc = set(df.nlargest(n, 'amount')['code']), set(df.nlargest(n, 'mv')['code']), set(df.nlargest(n, 'ret5')['code'])
    df['①成交'] = df['code'].isin(ta)
    df['②市值'] = df['code'].isin(tm)
    df['③涨幅'] = df['code'].isin(tc)
    df['hit4'] = df['①成交'].astype(int) + df['②市值'].astype(int) + df['③涨幅'].astype(int)
    df['in4交集'] = df['hit4'] == 3
    return df.sort_values(['hit4', 'ret5'], ascending=[False, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="四找scanner(独立于交集)")
    ap.add_argument('--codes', help='逗号分隔代码(来自交集/任意)')
    ap.add_argument('--industry', help='逗号分隔行业(软件服务,软件开发)')
    ap.add_argument('--top', type=int, default=30, help='每维度取前N(默认30)')
    a = ap.parse_args()
    sl = load_universe()
    codes = get_codes(a, sl)
    if not codes:
        print('用法: --codes 688111,300033 或 --industry 软件服务,软件开发'); return
    df = four_find(codes, a.top)
    if len(df) == 0:
        print('无有效数据'); return
    print(f"=== 四找scanner | 输入{len(codes)}只 → 有效{len(df)}只 | 每维度top{a.top} ===")
    base = ['code', 'name', 'chg', 'ret5', 'mv']
    print('\n①成交额TOP10:'); print(df.nlargest(10, 'amount')[base].to_string(index=False))
    print('\n②市值TOP10:'); print(df.nlargest(10, 'mv')[base].to_string(index=False))
    print('\n③ret5 TOP10:'); print(df.nlargest(10, 'ret5')[base].to_string(index=False))
    hit = df[df['in4交集']]
    print(f'\n=== ★4找交集(①∩②∩③) {len(hit)}只 ===')
    if len(hit):
        print(hit[base + ['vol_ratio', '多头', '突破120', 'tech4']].to_string(index=False))
    print(f'\n=== ④技术4(均线多头+量比>1) 共{df["tech4"].sum()}只 ===')
    t4 = df[df['tech4']].nlargest(10, 'ret5')
    if len(t4): print(t4[base + ['vol_ratio', '突破120']].to_string(index=False))


if __name__ == '__main__':
    main()
