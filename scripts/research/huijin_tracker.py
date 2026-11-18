#!/usr/bin/env python3
"""国家队持仓追踪 — 2026-10-20上线(布鲁斯视频触发·汇金清仓传闻核查)
方法: 抽样大盘股/权重股的前十大股东, 统计汇金/中证金融在不同季度的进出变化
定期: 每季度财报披露后跑一次(1/4/7/10月的最后一周)
用法: PYTHONPATH=. python scripts/research/huijin_tracker.py [period]
"""
import sys, time; sys.path.insert(0, '.')

# 抽样池: 覆盖银行/券商/石油/电力/通信/消费的大盘权重股
SAMPLE = [
    ('601398.SH','工商银行'), ('600036.SH','招商银行'), ('600030.SH','中信证券'),
    ('601857.SH','中国石油'), ('600028.SH','中国石化'),
    ('601857.SH','中国石油'), ('601088.SH','中国神华'),
    ('600900.SH','长江电力'), ('601766.SH','中国中车'),
    ('600519.SH','贵州茅台'), ('000858.SZ','五粮液'),
    ('002415.SZ','海康威视'), ('600584.SH','长电科技'),
    ('601138.SH','工业富联'), ('300750.SZ','宁德时代'),
    ('000333.SZ','美的集团'), ('600690.SH','海尔智家'),
    ('601668.SH','中国建筑'), ('601390.SH','中国中铁'),
]
KEYWORDS = ['汇金', '证券金融']  # 汇金投资/汇金资管/中证金融

def track(period=None, tc=None):
    """统计抽样池中汇金/中证金融的持仓标的数和总持股变化"""
    if tc is None:
        from scripts.core.tushare_client import TushareClient
        tc = TushareClient()
    if period is None:
        import pandas as pd
        # 默认用最近一个季末
        q = pd.Period(pd.Timestamp('today'), freq='Q')
        period = (q - 1).end_time.strftime('%Y%m%d')
    
    results = {}
    for code, name in SAMPLE:
        try:
            d = tc.pro.top10_holders(ts_code=code, period=period)
            if not len(d): continue
            gj = d[d['holder_name'].str.contains('|'.join(KEYWORDS), na=False)]
            if len(gj):
                for _, r in gj.iterrows():
                    hname = r['holder_name'][:6]
                    amt = float(r.get('hold_amount') or 0) / 1e8
                    chg = r.get('hold_change')
                    results.setdefault(name, []).append(f'{hname} {amt:.1f}亿({"增" if chg and float(chg)>0 else "减" if chg and float(chg)<0 else "平"})')
            time.sleep(0.3)
        except Exception:
            continue
    print(f'=== {period} 汇金/中证金融持仓({len(results)}只抽样有) ===')
    for name, info in results.items():
        print(f'  {name}: {", ".join(info)}')
    return results

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else None
    track(p)

# ===== 主动路径(2026-10-20升级): 从源头追踪 =====
AUTHORITATIVE_SOURCES = {
    '汇金投资(直接持股)': 'https://data.eastmoney.com/gdfx/shareholder/10061230.html (季度·全部持股不只前十大)',
    '汇金资管(ETF)': '各宽基ETF年报"单一投资者≥20%"段落 (半年·2025Q4=15只ETF·平均38%)',
    '中证金融': 'http://www.csf.com.cn (月度·转融通+直接持股)',
    '汇金官网年报': 'https://www.huijin-inv.cn (年度/半年·完整资产负债)',
    '中投年报': 'https://www.china-inv.cn (年度·汇金受托管理资产全貌)',
}
