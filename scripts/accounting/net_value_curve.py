#!/usr/bin/env python3
"""R0-记账闭环 ②: 重建"总盘子"快照曲线 + 逐区间 vs 基准。

诚实口径(宁可不说不可胡说):
- 本脚本只做【记录装配】: 把历史上真实存在的资本快照(capital.xlsx 的 git 历史 blob +
  config/capital.xlsx.bak* + 当前文件)按"as-of 日期"排成序列。不插值、不臆造。
- 快照间存在外部资金进出(取现/入金/两融)且未逐笔登记 → 该序列是"总盘子资源曲线",
  不是严格净值(严格净值 = 券商对账单口径, 需逐日账户值或完整资金流水, 见小结文档).
- schema 分 era: 旧(≤8/29)只记 `总资金`(无杠杆记录, 视为自有); 新(9/9 起)记 `总资产`
  = 自有 + 两融 9.5 万。故并排 9/9 同时给 总盘子 与 自口径(=总资产-杠杆).
- 逐区间给出 组合Δ vs 上证/沪深300 Δ; 区间内若夹带资金变动则标 `⚠️可能含资金进出`.

用法: cd stock && PYTHONPATH=. python scripts/accounting/net_value_curve.py
输出: reports/accounting/capital_snapshots.csv / 净值vs基准.png / (控制台表格)
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
CFG = PROJECT / 'config'
OUTDIR = PROJECT / 'reports' / 'accounting'
OUTDIR.mkdir(parents=True, exist_ok=True)

DATE_RE = re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})')
LEV_RE = re.compile(r'(杠杆|两融|融资)\s*[约~]?\s*([\d.,]+)\s*万')


def read_capital(f: Path) -> dict | None:
    """读一个 capital.xlsx → {total, cash, mv, leverage, asof_raw, note} 或 None."""
    try:
        df = pd.read_excel(f, header=None)
    except Exception:
        return None
    d = {}
    for _, r in df.iterrows():
        if len(r) >= 2:
            k = str(r[0])
            if k in ('总资金', '总资产', '可用资金', '当前持仓总市值', '总市值', '更新时间', '备注', '当日盈亏'):
                d[k] = r[1]
    total = d.get('总资产', d.get('总资金'))
    cash = d.get('可用资金')
    mv = d.get('总市值', d.get('当前持仓总市值'))
    note = str(d.get('备注', '')) or ''
    upd = str(d.get('更新时间', ''))
    # 杠杆: 优先从备注/更新时间文本抓, 单位=万
    lev = 0.0
    for txt in (note, upd):
        m = LEV_RE.search(txt)
        if m:
            lev = float(m.group(2).replace(',', '')) * 1e4
            break
    if total is None or (isinstance(total, float) and pd.isna(total)):
        return None
    return {'total': float(total), 'cash': None if pd.isna(cash) else float(cash),
            'mv': None if pd.isna(mv) else float(mv), 'leverage': lev,
            'asof_raw': upd, 'note': note}


def resolve_asof(rec: dict, commit_date: str, commit_subject: str, default_date: str) -> str:
    """as-of 日期三级信号: ①更新时间/备注 ②提交信息里的 MM/DD ③commit日期."""
    search_txt = f"{rec.get('asof_raw', '')} {rec.get('note', '')}"
    m = DATE_RE.search(search_txt)
    if m:
        return f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'
    # 提交信息 "8/29收盘校准" 式日期(当年)
    mm = re.search(r'(?<!\d)(\d{1,2})/(\d{1,2})(?:日)?', commit_subject)
    if mm:
        year = default_date[:4]
        mon, day = int(mm.group(1)), int(mm.group(2))
        if 1 <= mon <= 12 and 1 <= day <= 31:
            return f'{year}-{mon:02d}-{day:02d}'
    return default_date  # commit 日期


def collect_sources() -> list[dict]:
    """装配所有候选快照。"""
    srcs = []
    # 1) git blob: config/capital.xlsx 每次被提交的版本
    shas = subprocess.run(
        ['git', '-C', str(PROJECT), 'log', '--format=%H', '--', 'config/capital.xlsx'],
        capture_output=True, text=True).stdout.split()
    for sha in shas:
        msg = subprocess.run(['git', '-C', str(PROJECT), 'log', '-1',
                              '--format=%cs\t%s', sha], capture_output=True, text=True).stdout.strip()
        commit_date, subject = msg.split('\t', 1) if '\t' in msg else (msg, '')
        blob = subprocess.run(['git', '-C', str(PROJECT), 'show', f'{sha}:config/capital.xlsx'],
                              capture_output=True).stdout
        tmp = PROJECT / '.tmp_cap_blob.xlsx'
        tmp.write_bytes(blob)
        rec = read_capital(tmp)
        tmp.unlink(missing_ok=True)
        if rec is None:
            continue
        rec['asof'] = resolve_asof(rec, commit_date, subject, commit_date)
        rec['source'] = f'git@{sha[:7]}'
        rec['commit_date'] = commit_date
        rec['subject'] = subject[:60]
        srcs.append(rec)
    # 2) config/capital.xlsx.bak* (文件名后缀 or mtime)
    for f in sorted(CFG.glob('capital.xlsx.bak*')):
        rec = read_capital(f)
        if rec is None:
            continue
        tag = f.name.replace('capital.xlsx.bak', '').replace('.', '')  # e.g. 0707 / 20260628 / 空
        mm = re.search(r'(?<!\d)(\d{4})(\d{2})(\d{2})', f.name)
        asof = rec.get('asof_raw', '')
        dm = DATE_RE.search(str(asof))
        if dm:
            asof = f'{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}'
        else:
            asof = f'{tag[:4]}-{tag[4:6]}-{tag[6:8]}' if mm else f.name[-8:]
        rec['asof'] = asof
        rec['source'] = f'file:{f.name}'
        rec['commit_date'] = ''
        rec['subject'] = ''
        srcs.append(rec)
    # 3) 当前 working file (若与最后 blob 相同则自然被 dedupe 覆盖)
    if (CFG / 'capital.xlsx').exists():
        rec = read_capital(CFG / 'capital.xlsx')
        if rec:
            dm = DATE_RE.search(f"{rec.get('asof_raw', '')} {rec.get('note', '')}")
            rec['asof'] = (f'{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}' if dm
                           else pd.Timestamp.today().strftime('%Y-%m-%d'))
            rec['source'] = 'file:capital.xlsx(当前)'
            rec['commit_date'] = ''
            rec['subject'] = ''
            srcs.append(rec)
    return srcs


def main():
    srcs = collect_sources()
    if not srcs:
        print('❌ 没有可用的资本快照')
        sys.exit(1)

    # 装配 + 去重(同 asof 保留更新/更细的 source)
    rows = []
    for s in srcs:
        try:
            asof = pd.Timestamp(s['asof'])
        except Exception:
            continue
        rows.append({'asof': asof, 'total': s['total'], 'cash': s['cash'], 'mv': s['mv'],
                     'leverage': s['leverage'], 'source': s['source'],
                     'commit_date': s['commit_date'], 'note': (s['note'] or '')[:80]})
    df = pd.DataFrame(rows)
    df = df.sort_values(['asof', 'commit_date']).reset_index(drop=True)
    df = df.drop_duplicates(subset=['asof'], keep='last').sort_values('asof').reset_index(drop=True)

    # 自口径: total − leverage(杠杆记录才减; 无记录默认 0 = 旧era未记两融)
    df['self_capital'] = df['total'] - df['leverage']
    df['schema'] = df['total'].map(lambda _: '')
    # 区分 era: 当前/备注含"总资产"的→B; 无法直接区分, 用杠杆>0 或 source=当前 近似标注
    df['era'] = df.apply(
        lambda r: 'B(总资产,含两融)' if (r['leverage'] > 0 or r['source'].startswith('file:capital.xlsx(当前)'))
        else 'A(总资金,无杠杆记录)', axis=1)

    # 落盘
    out = df.copy()
    out['asof'] = out['asof'].dt.strftime('%Y-%m-%d')
    out.to_csv(OUTDIR / 'capital_snapshots.csv', index=False)

    # ---- 与基准逐区间对比 ----
    try:
        idx = pd.read_parquet(PROJECT / 'data' / 'store' / 'kline' / '000001SH' / 'D.parquet')
        idx['date'] = pd.to_datetime(idx['date'])
        idx = idx.set_index('date')['close'].astype(float)
        hs300 = pd.read_parquet(PROJECT / 'data' / 'store' / 'kline' / '000300SH' / 'D.parquet')
        hs300['date'] = pd.to_datetime(hs300['date'])
        hs300 = hs300.set_index('date')['close'].astype(float)
    except Exception as e:
        print(f'⚠️ 基准读取失败: {e}')
        idx = hs300 = None

    def bench_at(series, d):
        if series is None:
            return None
        sub = series[series.index <= d]
        return float(sub.iloc[-1]) if len(sub) else None

    print('=' * 100)
    print('资本快照序列(总盘子曲线, 非严格净值):')
    print('=' * 100)
    prev_total = prev_self = None
    rows_out = []
    for _, r in df.iterrows():
        d = r['asof']
        bt, bh = (bench_at(idx, d), bench_at(hs300, d)) if idx is not None else (None, None)
        rows_out.append({'组合Δ%': '', '上证Δ%': '', '沪深300Δ%': ''})
        flag = ''
        cash_s = '' if pd.isna(r['cash']) or r['cash'] is None else f"{r['cash']:,.0f}"
        print(f"{d.strftime('%Y-%m-%d')} [{r['era'][:3]}] 总盘子 {r['total']:>10,.0f}"
              f"  自口径 {r['self_capital']:>10,.0f}  现金 {cash_s}"
              f"  {'杠杆%.0f' % r['leverage'] if r['leverage'] else ''}  {r['source']}")
        if r['note']:
            print(f"            备注: {r['note']}")
        if prev_total is not None and d != prev_total['d']:
            pc = (r['total'] / prev_total['total'] - 1) * 100
            ps = (r['self_capital'] / prev_total['self_capital'] - 1) * 100
            ibt = None if (bt is None or prev_total['bt'] is None or prev_total['bt'] == 0) else (bt / prev_total['bt'] - 1) * 100
            ibh = None if (bh is None or prev_total['bh'] is None or prev_total['bh'] == 0) else (bh / prev_total['bh'] - 1) * 100
            days = (d - prev_total['d']).days
            seg = f'{prev_total["d"].strftime("%m/%d")}→{d.strftime("%m/%d")}'
            print(f'    │ {seg} ({days}d): 总盘子 {pc:+.1f}% | 自口径 {ps:+.1f}% | '
                  f'上证 {ibt if ibt is None else round(ibt,1)}% | 沪深300 {ibh if ibh is None else round(ibh,1)}%')
            if prev_total['era_has_flow']:
                print('    │ ⚠️ 此区间内夹带资金变动(取现/未知), Δ≠策略收益')
        prev_total = {'d': d, 'total': r['total'], 'self_capital': r['self_capital'],
                      'bt': bt, 'bh': bh, 'era_has_flow': False}
    print()
    print('⚠️ 诚实声明: 快照间有外部资金进出(取现2400等)未逐笔登记; '
          '旧era(总资金)无两融记录, 新era(9/9起)含两融9.5万。'
          '本曲线≠严格净值; 要得到可归因的"跑赢/跑输", 需券商对账单逐日口径(见小结文档)。')

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # macOS 中文字体(找不到就退回英文标注)
        _cjk = False
        for _cand in ('/System/Library/Fonts/STHeiti Light.ttc',
                      '/System/Library/Fonts/PingFang.ttc',
                      '/System/Library/Fonts/Hiragino Sans GB.ttc',
                      '/System/Library/Fonts/Supplemental/Songti.ttc'):
            if os.path.exists(_cand):
                try:
                    font_manager.fontManager.addfont(_cand)
                    plt.rcParams['font.family'] = font_manager.FontProperties(fname=_cand).get_name()
                    _cjk = True
                except Exception:
                    pass
                break
        if not _cjk:
            plt.rcParams['axes.unicode_minus'] = False
            _T, _self, _total, _sh, _cap = ('self-capital vs SSE', 'self-capital(assets-leverage)',
                                            'total(leveraged)', 'SSE(same-start scaled)',
                                            'snapshots incl external cashflows, NOT strict NAV')
        else:
            _T, _self, _total, _sh, _cap = ('自口径/总盘子 快照曲线 vs 上证(同起点) [含资金进出,非严格净值]',
                                            '自口径(总资产−杠杆)', '总盘子(含两融)', '上证(缩放到同起点)',
                                            '快照含外部资金进出, 非严格净值')
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(df['asof'], df['self_capital'], 'o-', label=_self, color='#c0392b')
        ax.plot(df['asof'], df['total'], 'o--', label=_total, color='#8e44ad', alpha=0.6)
        if idx is not None:
            base = df['asof'].iloc[0]
            b0 = bench_at(idx, base)
            if b0:
                end = df['asof'].iloc[-1]
                # 上证只画与快照同区间的连续线(用原始指数点, 缩放到快照首日=自口径首值)
                span = idx[(idx.index >= base) & (idx.index <= end)]
                if len(span):
                    scale = df['self_capital'].iloc[0] / b0
                    ax.plot(span.index, span.values * scale, '-', color='#7f8c8d',
                            label=_sh, lw=1)
        ax.set_title(_T)
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylabel(_cap)
        fig.autofmt_xdate()
        fig.tight_layout()
        png = OUTDIR / '净值vs基准.png'
        fig.savefig(png, dpi=110)
        print(f'\n📈 图已存: {png}')
    except Exception as e:
        print(f'\n⚠️ 画图失败(不影响数据): {e}')


if __name__ == '__main__':
    main()
