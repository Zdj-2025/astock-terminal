#!/usr/bin/env python3
"""金叉信号scanner(2026-08-28建)—— MA5上穿MA10 + 量价过滤。

回测依据(2026-08-28全信号统一回测,784只×165天):
  金叉(无过滤):   前瞻10日超额+1.16%, t=5.1, 胜率65% ✅有效
  金叉+低位(组合): 前瞻10日超额+1.78%, t=3.0, 胜率63% ✅有效但样本少
→ 低位为标注非门槛(与divergence放宽同理),核心条件=金叉当日阳线。

信号定义: MA5从下方上穿MA10(昨日MA5≤MA10, 今日MA5>MA10) + 当日收盘>开盘(阳线)
输出: 按距120日高排序(低位优先),标注放量/缩量/低位

2026-09-12升级: +smart60d背书列(望远镜用法,backtest_smart_telescope验证):
  金叉×smart背书(60日内score≥4): 20d超额+0.66% vs 无背书-0.48%(差+1.14pct,p<0.0001,N=7567/3551)
  分期稳定: 2025H2差+0.90(p=0.003)/2026H1差+1.40(p=0.001,弱市亏损少5倍)
  → ⚠️2026-09-12方法论审计修订: smart✓=排序过滤器不是买点(vs中证1000基准背书组绝对-0.49%,
    上证口径+1.49%含大小盘beta); 稳健的是组间差+0.71~+1.11(七场景全p<0.0001)——
    用法=候选排序/多空对冲, 独立买入须叠加其他正alpha来源
2026-09-12四臂终审(预注册): 背书金叉内 无低位无背离+1.21% > +低位+0.50% > +背离-0.09% > 双标-0.39%(p=0.045差于原始)
  → ⚠️低位/新鲜背离在此框架为负贡献(接刀污染); 最优=简配"金叉+背书"; 低位★降为中性信息不排序优先

用法: daily_scan_all.py自动跑,或 python scripts/screener/golden_cross_scanner.py
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

KDIR = 'data/store/kline'


def scan_golden_cross(verbose=True):
    sl = pd.read_parquet('data/store/stock_list.parquet')
    sl['code'] = sl['symbol'].astype(str).str.zfill(6)
    name_map = dict(zip(sl['code'], sl['name']))
    ind_map = dict(zip(sl['code'], sl['industry']))

    hits = []
    for d in os.listdir(KDIR):
        if not d.isdigit() or len(d) != 6 or d[:2] in ('51', '15', '56', '58'):
            continue
        try:
            df = pd.read_parquet(f'{KDIR}/{d}/D.parquet').sort_values('date')
            if len(df) < 130:
                continue
            name = str(name_map.get(d, ''))
            if 'ST' in name or '退' in name:
                continue
            c = df['close'].astype(float)
            o = df['open'].astype(float) if 'open' in df.columns else c
            vcol = 'volume' if 'volume' in df.columns else 'vol'
            v = df[vcol].astype(float)

            i = len(df) - 1
            ma5_now = c.iloc[i-4:i+1].mean()
            ma10_now = c.iloc[i-9:i+1].mean()
            ma5_prev = c.iloc[i-5:i].mean()
            ma10_prev = c.iloc[i-10:i-1].mean()
            if any(np.isnan(x) for x in [ma5_now, ma10_now, ma5_prev, ma10_prev]):
                continue

            # 金叉条件: 昨日MA5≤MA10, 今日MA5>MA10
            golden = ma5_now > ma10_now and ma5_prev <= ma10_prev
            if not golden:
                continue
            # 阳线确认
            yang = c.iloc[i] > o.iloc[i]
            if not yang:
                continue

            # 辅助信息
            hi120 = c.iloc[i-120:i].max()
            dist120 = (c.iloc[i] / hi120 - 1) * 100 if hi120 > 0 else np.nan
            low_pos = dist120 < -15
            v20 = v.iloc[i-20:i].mean()
            vr = v.iloc[i] / v20 if v20 > 0 else np.nan
            vol_tag = '✅缩量' if (not np.isnan(vr) and vr < 0.8) else ('⚠️放量' if (not np.isnan(vr) and vr > 2) else '')
            ma20 = c.iloc[i-19:i+1].mean()
            vs_ma20 = (c.iloc[i] / ma20 - 1) * 100

            # smart60d背书(2026-09-12望远镜用法: 近60日内score5≥4)
            try:
                from scripts.research.reverify_smart_money_recency import score5_series
                sc5 = score5_series(c.values, v.values)
                smart_backed = bool((sc5[max(0, i-60):i] >= 4).any())
            except Exception:
                smart_backed = False

            hits.append(dict(code=d, name=name, industry=ind_map.get(d, ''),
                             price=float(c.iloc[i]), dist120=dist120,
                             low_pos=low_pos, vr=vr, vol_tag=vol_tag,
                             vs_ma20=vs_ma20, yang=True, smart=smart_backed))
        except Exception:
            continue

    hits.sort(key=lambda h: h['dist120'])  # 2026-09-12统一审计: smart排序复现失败(+0.12 p=0.43)回退; smart✓仅信息列
    if verbose:
        print(f"📊 金叉信号scanner | MA5上穿MA10+阳线 | t=5.1/胜率65% | smart✓=60日资金背书(背书版+0.66%vs无背-0.48%,p<0.0001)")
        print(f"   金叉+阳线: {len(hits)}只")
        low = [h for h in hits if h['low_pos']]
        print(f"   └低位(距120高<-15%): {len(low)}只\n")
        if hits:
            print(f"{'代码':<8}{'名称':<10}{'行业':<10}{'价':>7}{'距120高':>8}{'vsMA20':>7}{'量比':>5}  标注")
            print('-' * 70)
            for h in hits[:25]:
                tag = 'smart✓' if h.get('smart') else ''
                if h['low_pos']:
                    tag += ' 低位★'
                if h['vol_tag']:
                    tag += ' ' + h['vol_tag']
                vrs = f"{h['vr']:.1f}" if not np.isnan(h['vr']) else '-'
                print(f"{h['code']:<8}{h['name'][:9]:<10}{str(h['industry'])[:8]:<10}"
                      f"{h['price']:>7.2f}{h['dist120']:>+7.0f}%{h['vs_ma20']:>+6.1f}%{vrs:>5}  {tag}")
            print(f"\n💡 金叉=趋势拐点信号;低位★=空间更大;建议叠加五查排雷+主线确认后操作")
    return hits


if __name__ == '__main__':
    hits = scan_golden_cross()
    if hits:
        from scripts.screener.scan_history import record_batch
        record_batch('golden_cross', hits)
        print(f"已写入信号累积器: golden_cross {len(hits)}只")
