# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

#!/usr/bin/env python3
"""绿珠上梯扫描器 — 数据港形态v2(宽松版,2026-08-15)。

形态特征(窗口W日,数据港8/7-8/13实证):
  绿柱多:  ≥70%日子 close<open(高开低走收阴,吓散户)
  高开多:  ≥50%日子 open>prev_close(隔夜抢筹)
  重心上移: 后段收盘均值>前段(绿珠子整体往上爬,假阴洗盘)
  均线抬升: 末值MA5>首值MA5
  量能放大: 末2日均量>前段均量×1.1(洗盘换手加大,资金介入)
→ 典型主升前蓄势: 数据港8/7-13五连绿但25.5→26.9上移+放量,8/14平开涨停兑现。

CLI:
  python scripts/screener/green_staircase_scanner.py                  # 截至8/13(数据港基准日)
  python scripts/screener/green_staircase_scanner.py --asof 20260814  # 截至8/14
  python scripts/screener/green_staircase_scanner.py --window 7 --green 0.6
"""
import argparse
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')


def scan(asof: str, window: int = 5, green_ratio: float = 0.7, gap_ratio: float = 0.5,
         vol_mult: float = 1.1):
    sl = pd.read_parquet('data/store/stock_list.parquet').set_index('symbol')
    out = []
    kdir = 'data/store/kline'
    for code in os.listdir(kdir):
        p = os.path.join(kdir, code, 'D.parquet')
        if not os.path.isfile(p) or not code.isdigit() or len(code) != 6:
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        dcol = [c for c in df.columns if 'date' in c.lower()]
        if not dcol:
            continue
        df = df.sort_values(dcol[0]).reset_index(drop=True)
        # 日期标准化为YYYYMMDD(datetime→'2026-08-14 00:00'截取会错,须去'-')
        ds = df[dcol[0]].astype(str).str[:10].str.replace('-', '')
        full = df.copy()
        df = df[ds <= asof].reset_index(drop=True)
        if len(df) < window + 6:
            continue
        # MA5在完整历史上算(窗口内rolling首值必nan)
        df['ma5'] = df['close'].astype(float).rolling(5).mean()
        t = df.tail(window).reset_index(drop=True)
        prev = df.iloc[-window - 1]
        o = t['open'].astype(float).values
        c = t['close'].astype(float).values
        v = t['volume'].astype(float).values
        greens = sum(c < o)                      # 绿柱数(高开低走收阴)
        gaps = sum(o[1:] > c[:-1]) + (1 if o[0] > float(prev['close']) else 0)  # 高开数
        half = max(2, window // 2)
        lift = c[-half:].mean() > c[:half].mean()          # 重心上移
        ma_up = pd.notna(df['ma5'].iloc[-window]) and df['ma5'].iloc[-1] > df['ma5'].iloc[-window]
        vol_amp = v[-2:].mean() > v[:-2].mean() * vol_mult  # 近2日放量
        if greens >= int(window * green_ratio) and gaps >= int(window * gap_ratio) \
                and lift and ma_up and vol_amp:
            w0 = float(prev['close'])
            ret_w = (c[-1] / w0 - 1) * 100
            # 次日(兑现日)表现: full中asof之后第一个交易日
            day_after = None
            dsf = full[dcol[0]].astype(str).str[:10].str.replace('-', '')
            after = full[dsf > asof]
            if len(after):
                day_after = (float(after['close'].iloc[0]) / c[-1] - 1) * 100
            name = sl.loc[code, 'name'] if code in sl.index else '?'
            ind = sl.loc[code, 'industry'] if code in sl.index else '?'
            out.append({'code': code, 'name': name, 'industry': ind,
                        'close': c[-1], 'greens': greens, 'gaps': gaps,
                        'ret_w%': round(ret_w, 1), 'vol_x': round(v[-2:].mean() / max(v[:-2].mean(), 1), 2),
                        'next%': day_after, 'date_end': str(df[dcol[0]].iloc[-1])[:10]})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description='绿珠上梯(数据港宽松形态)扫描')
    ap.add_argument('--asof', default='20260813', help='形态窗口截止日(默认20260813=数据港基准)')
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--green', type=float, default=0.7, help='绿柱占比下限')
    ap.add_argument('--gap', type=float, default=0.5, help='高开占比下限')
    a = ap.parse_args()
    res = scan(a.asof, a.window, a.green, a.gap)
    if len(res) == 0:
        print(f'无命中(asof={a.asof} window={a.window})')
        return
    res = res.sort_values('ret_w%', ascending=False)
    print(f"=== 绿珠上梯扫描 | 截至{a.asof} 近{a.window}日 | 命中{len(res)}只 ===")
    print(f"绿柱>= {int(a.window*a.green)}天 高开>= {int(a.window*a.gap)}天 重心上移+MA5抬升+近2日放量")
    print(f"\n{'代码':<8}{'名称':<10}{'行业':<8}{'收盘':>8}{'绿柱':>4}{'高开':>4}{'窗口涨幅':>8}{'量能倍数':>8}{'次日涨跌':>9} 状态")
    for _, r in res.iterrows():
        tag = '⚡已兑现' if (r['next%'] is not None and r['next%'] >= 9.5) else \
              ('↑有反应' if (r['next%'] is not None and r['next%'] > 2) else '待兑现')
        nx = f"{r['next%']:+.1f}%" if r['next%'] is not None else '-'
        print(f"{r['code']:<8}{r['name']:<10}{r['industry']:<8}{r['close']:>8.2f}{r['greens']:>4}{r['gaps']:>4}{r['ret_w%']:>+7.1f}%{r['vol_x']:>7.1f}x{nx:>9} {tag}")
    print('\n💡 数据港基准: 8/7-13五连绿+重心上移+放量 → 8/14涨停(次日+10%)')


if __name__ == '__main__':
    main()
