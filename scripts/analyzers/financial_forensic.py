#!/usr/bin/env python3
"""财务排雷五查(financial forensic) — 勾稽关系交叉验证(2026-08-23建)。

来源: 陈杰森资本棋局《百亿上市公司的财务疑问》(2026-08-20,点名怡亚通/深圳华强/科大讯飞)
方法论: 利润表好看但勾稽关系打架 = 财务疑问。我们逐条对账后自动化。

五查(基于tushare实算,阈值有教训支撑):
  ①存贷双高: 货币资金>30亿 且 利息费用>营业利润30%
     → 账上巨额现金却付高息(怡亚通模式;借款字段本数据层缺失,用利息费用反推)
  ②净现背离: 净利润>0 但 经营现金流<0 (深圳华强模式: 利润是应收堆的)
  ③应收挤压: 应收账款/货币资金>2 (科大讯飞模式: 应收172亿/现金39亿=4.4倍)
  ④研发资本化: 开发支出字段缺失→na(需年报附注拆研发投入,人工补)
  ⑤应收虚增: 应收增速-营收增速>20pct (收入注水嫌疑:货卖出去钱收不回)

判定: ①②③任一命中=硬雷(接入mainline_leader_scanner前置排除);
      ⑤命中=软雷(标注降级); ④na不算。

用法:
  python scripts/analyzers/financial_forensic.py --code 002230     # 单查
  python scripts/analyzers/financial_forensic.py --codes 002230,002183,000062
  from scripts.analyzers.financial_forensic import FinancialForensic
  ff = FinancialForensic(); ff.check_code('002230')  # → dict
缓存: data/store/forensic_cache.parquet (code+end_date键,新报告期自动刷新)
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_PATH = PROJECT_ROOT / 'data/store/forensic_cache.parquet'

# 阈值(教训支撑,勿随意调)
TH_MONEY_CAP = 30e8        # ①货币资金门槛(30亿以上才谈"巨额现金")
TH_INT_RATIO = 0.30        # ①利息费用/营业利润>30%
TH_AR_SQUEEZE = 2.0        # ③应收/现金>2倍
TH_AR_GROWTH_GAP = 20.0    # ⑤应收增速-营收增速>20pct
TH_INV_MONTHS = 6          # ⑥存货/年化营收>6个月=警戒(2026-09-11寒武纪案)
TH_STOCKPILE_YOY = 100.0   # ⑥存货或预付同比增速>100%
TH_NCR = 0.5               # ⑥净现比OCF/净利<0.5=利润堆在货里


class FinancialForensic:
    def __init__(self, use_cache=True):
        from scripts.core.tushare_client import TushareClient
        self.pro = TushareClient().pro
        self.use_cache = use_cache
        self._cache = None
        if use_cache and CACHE_PATH.exists():
            try:
                self._cache = pd.read_parquet(CACHE_PATH)
            except Exception:
                self._cache = None

    def _ts_code(self, code):
        code = str(code).zfill(6)
        return code + ('.SH' if code.startswith(('6', '5')) else
                       '.BJ' if code.startswith(('4', '8')) else '.SZ')

    def _save_cache(self, code, end_date, res):
        row = pd.DataFrame([dict(code=str(code).zfill(6), end_date=end_date,
                                 checked_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
                                 **{k: str(v) for k, v in res.items()})])
        if self._cache is None:
            self._cache = row
        else:
            self._cache = pd.concat([self._cache[~((self._cache['code'] == str(code).zfill(6)) &
                                                   (self._cache['end_date'] == end_date))], row],
                                    ignore_index=True)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._cache.to_parquet(CACHE_PATH, index=False)

    @staticmethod
    def _same_period_last_year(end_date):
        y, m = int(end_date[:4]), int(end_date[4:6])
        return f"{y-1}{m:02d}{end_date[6:]}"

    def check_code(self, code, verbose=False):
        """单票五查。返回 dict(end_date, ①~⑤判定+数值, hard_hit, reason)"""
        code = str(code).zfill(6)
        tc = self._ts_code(code)
        # 拉三张表近5期(覆盖去年同期)
        bs = self.pro.query('balancesheet', ts_code=tc, order='end_date desc', limit=5)
        time.sleep(0.3)
        inc = self.pro.query('income', ts_code=tc, order='end_date desc', limit=5)
        time.sleep(0.3)
        cf = self.pro.query('cashflow', ts_code=tc, order='end_date desc', limit=5)
        if bs is None or not len(bs):
            return {'error': f'{code} 无资产负债表'}

        cur = bs.iloc[0]
        end_date = cur['end_date']
        prev_key = self._same_period_last_year(end_date)

        # 绿查数据源(2026-08-24 DeepSeek借鉴): 增速+造血趋势(fina_indicator)
        # 毛利率/净利率字段本数据层缺失→从利润表自算(不另耗API)
        try:
            time.sleep(0.3)
            fiz = self.pro.query('fina_indicator', ts_code=tc, order='end_date desc', limit=1,
                                 fields='end_date,or_yoy,ocf_yoy')
            rev_yoy = float(fiz.iloc[0]['or_yoy']) if pd.notna(fiz.iloc[0]['or_yoy']) else None
            ocf_yoy = float(fiz.iloc[0]['ocf_yoy']) if pd.notna(fiz.iloc[0]['ocf_yoy']) else None
        except Exception:
            rev_yoy = ocf_yoy = None

        def _v(df, date_key, col):
            """取指定报告期的字段值(亿元)"""
            if df is None or not len(df) or col not in df.columns:
                return None
            sub = df[df['end_date'] == date_key]
            if not len(sub):
                return None
            v = sub.iloc[0][col]
            if v is None or pd.isna(v):
                return None
            return float(v) / 1e8

        money = _v(bs, end_date, 'money_cap')
        ar = _v(bs, end_date, 'accounts_receiv')
        ar_prev = _v(bs, prev_key, 'accounts_receiv')
        int_exp = _v(inc, end_date, 'int_exp')
        op_profit = _v(inc, end_date, 'operate_profit')
        n_income = _v(inc, end_date, 'n_income')
        rev = _v(inc, end_date, 'total_revenue')
        rev_prev = _v(inc, prev_key, 'total_revenue')
        ocf = _v(cf, end_date, 'n_cashflow_act')

        flags, soft, reasons = [], [], []

        # ① 存贷双高(巨额现金却付高息) — 利息字段季报常缺,年报回退
        annual_key = next((str(d) for d in bs['end_date'].values if str(d).endswith('1231')), None)
        f1 = False
        if money and money > TH_MONEY_CAP / 1e8:
            a_int = _v(inc, annual_key, 'int_exp') if annual_key else None
            a_op = _v(inc, annual_key, 'operate_profit') if annual_key else None
            for ie, op, tag in [(int_exp, op_profit, end_date[:6]), (a_int, a_op, (annual_key or '')[:4] + '年报')]:
                if ie and op and op > 0 and ie / op > TH_INT_RATIO:
                    f1 = True
                    flags.append('①存贷双高')
                    reasons.append(f'现金{money:.0f}亿却付息{ie:.1f}亿(占{tag}营业利润{ie/op:.0%})')
                    break

        # ② 净现背离(利润是应收堆的) — 季节性防误报(2026-08-23立讯/春立教训):
        #   Q1备货/医院回款周期导致消费电子/医药商业Q1现金流常年为负=季节性≠雷
        #   硬雷=最新年报(净利>0且OCF<0); 季报命中但年报正常→软提示(需人工辨季节)
        f2, f2_soft = False, False
        a_inc = _v(inc, annual_key, 'n_income') if annual_key else None
        a_ocf = _v(cf, annual_key, 'n_cashflow_act') if annual_key else None
        if a_inc and a_inc > 0 and a_ocf is not None and a_ocf < 0:
            f2 = True
            flags.append('②净现背离(年报)')
            reasons.append(f'{annual_key[:4]}年报: 净利+{a_inc:.1f}亿但经营现金流{a_ocf:.1f}亿')
        elif n_income and n_income > 0 and ocf is not None and ocf < 0:
            f2_soft = True
            soft.append('②净现背离(季报·或季节性)')
            a_ocf_s = f'{a_ocf:.1f}亿' if a_ocf is not None else 'na'
            reasons.append(f'{end_date[:6]}季报: 净利+{n_income:.1f}亿但OCF{ocf:.1f}亿(年报{a_ocf_s})')

        # ③ 应收挤压 — 结构性+当期恶化才硬雷(2026-08-23联化vs讯飞辨析):
        #   联化3.8倍但Q1盈利+OCF-0.6亿轻微+年现金转化2.7倍 → 结构性软提示
        #   讯飞4.5倍且H1亏损+OCF-9.5亿 → 恶化确认 → 硬雷
        f3 = False
        if money and money > 0 and ar and ar / money > TH_AR_SQUEEZE:
            cur_bad = (n_income is not None and n_income < 0) or (ocf is not None and ocf < -3)
            if cur_bad:
                f3 = True
                flags.append('③应收挤压+当期恶化')
                reasons.append(f'应收{ar:.0f}亿/现金{money:.0f}亿={ar/money:.1f}倍'
                               + (f'+当期净利{n_income:.1f}亿' if n_income is not None and n_income < 0 else '')
                               + (f'+当期OCF{ocf:.1f}亿' if ocf is not None and ocf < -3 else ''))
            else:
                soft.append('③应收挤压(结构性·当期未恶化)')
                reasons.append(f'应收{ar:.0f}亿/现金{money:.0f}亿={ar/money:.1f}倍(当期盈利OCF正常,盯半年报)')

        # ④ 研发资本化(开发支出字段缺失→na)
        f4 = None

        # ⑤ 应收虚增(应收增速远超营收增速)
        f5 = False
        if ar and ar_prev and ar_prev > 0 and rev and rev_prev and rev_prev > 0:
            gap = (ar / ar_prev - 1) * 100 - (rev / rev_prev - 1) * 100
            if gap > TH_AR_GROWTH_GAP:
                f5 = True
                soft.append('⑤应收虚增')
                reasons.append(f'应收增速-营收增速={gap:+.0f}pct')

        # ⑥ 存货-预付-减值三角(2026-09-11立,陈杰森寒武纪案教学: 左手抢料右手减值)
        #   三条件: a)存货/年化营收>6个月 b)存货或预付同比>100% c)净现比<0.5
        #   ≥2条命中=软提示(高成长Fabless备货属正常,需人工辨订单驱动vs过度乐观)
        f6, f6_hits = False, []
        inv = _v(bs, end_date, 'inventories')
        inv_prev = _v(bs, prev_key, 'inventories')
        prep = _v(bs, end_date, 'prepayment')
        prep_prev = _v(bs, prev_key, 'prepayment')
        m = int(end_date[4:6])
        annual_mult = {3: 4, 6: 2, 9: 4 / 3, 12: 1}.get(m)
        if inv and rev and annual_mult:
            inv_months = inv / (rev * annual_mult) * 12
            if inv_months > TH_INV_MONTHS:
                f6_hits.append(f'存货{inv:.0f}亿={inv_months:.0f}个月营收')
            if inv_prev and inv_prev > 0 and (inv / inv_prev - 1) * 100 > TH_STOCKPILE_YOY:
                f6_hits.append(f'存货同比+{(inv / inv_prev - 1) * 100:.0f}%')
            if prep and prep_prev and prep_prev > 0 and (prep / prep_prev - 1) * 100 > TH_STOCKPILE_YOY:
                f6_hits.append(f'预付同比+{(prep / prep_prev - 1) * 100:.0f}%')
            if n_income and n_income > 0 and ocf is not None and ocf / n_income < TH_NCR:
                f6_hits.append(f'净现比{ocf / n_income:.2f}(利润在货里)')
            if len(f6_hits) >= 2:
                f6 = True
                soft.append('⑥存货-预付三角')
                reasons.append('⑥' + '/'.join(f6_hits) + '(辨:订单备货vs计提不足)')

        res = dict(end_date=end_date, money_cap=money, ar=ar, ocf=ocf,
                   n_income=n_income, int_exp=int_exp,
                   rev_yoy=rev_yoy, ocf_yoy=ocf_yoy,
                   f1_dual=f1, f2_ocf=f2, f2_soft=f2_soft, f3_ar=f3, f4_rdcap=f4, f5_argrowth=f5,
                   f6_tri=f6,
                   hard_hit=bool(flags), flags=flags + soft,
                   reason='; '.join(reasons) if reasons else '')
        # 绿查列(只显示不当因子,2026-08-24: 造血改善/营收/毛利/净利)
        op_cost = _v(inc, end_date, 'oper_cost')
        gm = (rev - op_cost) / rev * 100 if (rev and op_cost and rev > 0) else None
        nm = n_income / rev * 100 if (n_income is not None and rev and rev > 0) else None
        res['gross_margin'] = gm
        res['net_margin'] = nm
        greens = []
        if ocf_yoy is not None and ocf_yoy > 100:
            greens.append(f'🌱造血改善(OCF同比{ocf_yoy:+.0f}%)')
        if rev_yoy is not None and rev_yoy < 0:
            greens.append(f'⚠️营收{rev_yoy:+.1f}%(连续收缩需辨结构换挡)')
        res['green'] = ' | '.join(greens)
        if self.use_cache:
            try:
                self._save_cache(code, end_date, res)
            except Exception:
                pass
        return res

    def segment_table(self, code, verbose=True):
        """分部毛利率矩阵(2026-08-24 DeepSeek借鉴: 谁在赚钱比赚多少重要)。
        fina_mainbz按产品分部: 收入×毛利→毛利率,一眼识别利润引擎。"""
        code = str(code).zfill(6)
        try:
            seg = self.pro.query('fina_mainbz', ts_code=self._ts_code(code),
                                 period=None, type='P',
                                 fields='end_date,bz_item,bz_sales,bz_profit',
                                 order='end_date desc', limit=30)
        except Exception:
            # 不带period取最新
            seg = self.pro.query('fina_mainbz', ts_code=self._ts_code(code), type='P',
                                 fields='end_date,bz_item,bz_sales,bz_profit',
                                 order='end_date desc', limit=30)
        if seg is None or not len(seg):
            if verbose:
                print(f"  {code}: 无分部数据")
            return None
        latest = seg['end_date'].iloc[0]
        seg = seg[seg['end_date'] == latest].copy()
        rows = []
        for _, s in seg.iterrows():
            sales = float(s['bz_sales']) if pd.notna(s['bz_sales']) else 0
            profit = float(s['bz_profit']) if pd.notna(s['bz_profit']) else 0
            gm = profit / sales * 100 if sales > 0 else None
            rows.append(dict(item=s['bz_item'], sales=sales / 1e8, gross=profit / 1e8, gm=gm))
        rows.sort(key=lambda x: -x['sales'])
        # 排除合计行(产品/合计等=tushare聚合口径,非真分部)
        segs = [r for r in rows if r['item'] not in ('产品', '合计', '总计', '小计')]
        if verbose:
            print(f"\n  {code} 分部毛利率矩阵 [{latest}] (亿元)")
            print(f"  {'分部':<14}{'收入':>7}{'毛利':>7}{'毛利率':>8}")
            for r in rows:
                gm = f"{r['gm']:.1f}%" if r['gm'] is not None else 'na'
                tag = ' ←合计' if r['item'] == '产品' else ''
                print(f"  {r['item']:<14}{r['sales']:>7.1f}{r['gross']:>7.1f}{gm:>8}{tag}")
            engine = max(segs, key=lambda x: x['gross'])
            print(f"  → 利润引擎: {engine['item']}(毛利{engine['gross']:.1f}亿,毛利率{engine['gm']:.0f}%)")
        return rows

    def scan(self, codes, verbose=True):
        rows = []
        for c in codes:
            try:
                r = self.check_code(c)
                if 'error' in r:
                    if verbose:
                        print(f"  {c}: {r['error']}")
                    continue
                mark = '🚨硬雷' if r['hard_hit'] else ('⚠️软雷' if r['flags'] else '✅过')
                gm = f"{r['gross_margin']:.0f}%" if r.get('gross_margin') is not None else 'na'
                nm = f"{r['net_margin']:.0f}%" if r.get('net_margin') is not None else 'na'
                rev = f"{r['rev_yoy']:+.0f}%" if r.get('rev_yoy') is not None else 'na'
                green = f" | {r['green']}" if r.get('green') else ''
                if verbose:
                    print(f"  {str(c).zfill(6)} [{r['end_date']}] {mark} {r['flags'] or ''} {r['reason']}")
                    print(f"      绿查: 营收{rev} 毛利{gm} 净利{nm}{green}")
                rows.append(dict(code=str(c).zfill(6), **{k: r[k] for k in
                             ('end_date', 'hard_hit', 'flags', 'reason')}))
            except Exception as e:
                if verbose:
                    print(f"  {c}: 查询失败 {str(e)[:40]}")
            time.sleep(0.2)
        return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='财务排雷五查(勾稽关系)+绿查+分部矩阵')
    parser.add_argument('--code', help='单只代码')
    parser.add_argument('--codes', help='逗号分隔多只')
    parser.add_argument('--segment', help='分部毛利率矩阵(单只代码)')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    ff = FinancialForensic(use_cache=not args.no_cache)
    if args.segment:
        ff.segment_table(args.segment)
        raise SystemExit
    print("=== 财务排雷五查 | 勾稽关系交叉验证 ===")
    print("  ①存贷双高(现金>30亿+利息>营业利润30%) ②净现背离(净利>0现金流<0)")
    print("  ③应收挤压(应收/现金>2) ④研发资本化(字段缺na) ⑤应收虚增(增速差>20pct)")
    print("  绿查: 营收增速/毛利率/净利率/OCF造血趋势(只显示,不当买入因子)\n")
    if args.code:
        ff.scan([args.code])
    elif args.codes:
        ff.scan([c.strip() for c in args.codes.split(',') if c.strip()])
    else:
        # 默认: 视频点名三家 + 持仓对照
        ff.scan(['002230', '002183', '000062', '000963', '002140'])


def qingyue_red_flags(ts_code, tc=None):
    """清越三红灯 (2026-10-16 清越科技造假案落地) — 超跌池/推荐前必查
    ①前五客户集中度>60%(年报 TopN 客户) ②应收高企且坏账计提<5% ③存货占流动资产比骤升+减值滞后史
    返回 (flags, detail): flags 0-3, 任一命中即该票需人工深读年报附注
    """
    if tc is None:
        from scripts.core.tushare_client import TushareClient
        tc = TushareClient()
    flags, detail = [], []
    import pandas as pd
    try:
        fin = tc.pro.balancesheet(ts_code=ts_code, fields='end_date,accounts_receiv,inventories,total_cur_assets')
        fin = fin.dropna(subset=['end_date']).sort_values('end_date')
        last = fin.iloc[-1]
        # ② 应收/流动资产 >30% 为高企(间接代理: 应收占比)
        ar_ratio = float(last['accounts_receiv'] or 0) / float(last['total_cur_assets'] or 1)
        if ar_ratio > 0.30:
            flags.append('应收占流动资产{:.0%}'.format(ar_ratio))
        # ③ 存货/流动资产 趋势: 最新 vs 4期前升幅>50%
        if len(fin) >= 5:
            inv_now = float(last['inventories'] or 0); inv_old = float(fin.iloc[-5]['inventories'] or 0)
            if inv_old > 0 and inv_now / inv_old > 1.5:
                flags.append('存货4期膨胀{:.0f}%'.format((inv_now/inv_old-1)*100))
    except Exception as e:
        detail.append(f'fina_indicator: {e}')
    # ① 客户集中度需年报附注, tushare无直连字段 → top5客户占比走 top10_holders 无关, 留人工项
    detail.append('客户集中度: 需人工查年报"前五名客户销售额占比"(>60%红灯)')
    return flags, detail


if __name__ == '__main__' and 'qingyue' in sys.argv if False else None:
    pass
