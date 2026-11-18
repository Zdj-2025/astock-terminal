#!/usr/bin/env python3
"""质量背离扫描器(原名主线龙头,2026-08-29正名) — 技术买点(时机) × 基本面质量(标的)。

教训(2026-06-29): divergence_scanner选的航天智装(技术✓但亏损+48%质押)当天-1.3%不涨;
基本面好的博腾+3.9%/泰格+5.4%全涨。技术买点只解决"何时买","买什么"靠基本面质量。
本扫描器: 复用divergence_scanner技术层 + 新建基本面质量打分卡(盈利/增长/估值/无雷/龙头)。

质量卡(满分100): 盈利pe>0(+30) + 业绩增长(+20) + 估值pe<60/pb<5(+20) + 无高质押<50%/无ST(+20) + 龙头mv>50亿(+10)
档位: 🥇质量背离(≥70) / 🥈中等(50-70) / ⚠️基本面弱(<50,亏损/高质押,即使技术买点也避开)

硬排除(mine_filter, 前置踢雷不靠分数, 2026-07-07用户要求"100多支先过滤"):
  ST/退市 + 亏损pe<=0 + 高质押>=50% + 面值<1.5元 + 暴跌dist120<-75%。
  教训: 航天(亏损+48%质押) / 山鹰(1.33元面值退市) / 洲明(93%质押)。
  阈值均有教训支撑, 非拍脑袋, 不漏边缘好票(这些本就是雷)。硬排除后再质量卡打分排序。

用法: cd stock && PYTHONPATH=. python scripts/screener/mainline_leader_scanner.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math
import pandas as pd
import tushare as ts
from scripts.core.tushare_client import TushareClient
from scripts.analyzers.news_analyzer import NewsAnalyzer
from scripts.screener.divergence_scanner import scan_divergence

pro = ts.pro_api()


def quality_scorecard(pe, pb, mv, profit_yoy, rev_yoy, pledge_rate, is_st, pe_deducted=None, pe_dedt_negative=False, dupont_em=None, is_fin=False):
    """基本面质量打分卡(满分100)。盈利30+增长20+估值20+无雷20+龙头10。
    pe_deducted: 按扣非利润算的PE(如有),用于检测一次性利润PE陷阱(2026-08-06东方精工教训)。"""
    score = 0
    # 盈利(硬门槛): 亏损(pe<=0)直接0分,淘汰
    if pe is not None and pe > 0:
        score += 30
        # 扣非负值检查(2026-08-29星源卓镁教训:PE266为正但扣非-0.169亿亏损)
        # PE含非经常性损益可为正,扣非才是真实经营
        if pe_dedt_negative:
            score -= 30  # 扣非亏损=盈利分清零
            score = max(score, 0)
    # 业绩增长
    if profit_yoy and profit_yoy > 0 and rev_yoy and rev_yoy > 0:
        score += 20
    elif (profit_yoy and profit_yoy > 0) or (rev_yoy and rev_yoy > 0):
        score += 10
    # 估值PE(回测验证250天5413信号: 30-60全环境夏普1.15最优,加权不排除)
    if pe is not None:
        if 30 <= pe <= 60:
            score += 10   # 最优区间
        elif pe < 30 or 60 < pe <= 100:
            score += 5    # 次优(不排除)
        # pe>100: +0
    # 估值PB(回测: 2-5全环境最稳)
    if pb is not None:
        if 2 <= pb <= 5:
            score += 10   # 最优区间
        elif pb < 2 or 5 < pb <= 10:
            score += 5    # 次优
        elif pb > 10:
            score += 3    # 高PB也有alpha(样本小但夏普高)
    # 无雷: 质押+ST
    if pledge_rate is not None and pledge_rate < 50:
        score += 15
    if not is_st:
        score += 5

    # 扣非PE陷阱检测(2026-08-06东方精工教训: 净利38亿扣非0.8亿,PE5→真实PE250)
    # 如果扣非PE远高于PE(一次性利润占比大),扣分
    if pe_deducted is not None and pe_deducted > 0 and pe is not None and pe > 0:
        if pe_deducted > pe * 3:  # 扣非PE是PE的3倍以上=一次性利润占比大
            score -= 15  # 严重扣分(利润质量存疑)
        elif pe_deducted > pe * 2:
            score -= 5   # 轻度扣分
    # 杜邦杠杆闸(2026-10-07): 识别"借出来的ROE"——非金融 EM≥3.5 杠杆警示-15 / ≥3.0 -5
    if dupont_em is not None and not is_fin:
        if dupont_em >= 3.5:
            score -= 15
        elif dupont_em >= 3.0:
            score -= 5
    # 市值(回测: <30亿全环境夏普1.36最强,加权不排除大盘)
    if mv is not None:
        if mv < 30:
            score += 10   # 小盘底背离最强
        elif mv < 100:
            score += 7    # 中盘
        else:
            score += 5    # 大盘不排除
    return score


def mine_filter(pe, pledge, price, name, dist120):
    """硬排除(前置踢雷, 不靠分数)。返回 (passed, reason)。
    阈值均有教训支撑, 非拍脑袋:
      ST/退市     → 退市风险雷
      亏损pe<=0或NaN → 航天教训(tushare对亏损/微利返回NaN, 2026-07-07发现航天pe=NaN漏网)
      高质押>=50% → 洲明93%教训(平仓雷)
      面值<1.5元  → 山鹰1.33教训(面值退市雷, 2026-06-20撤回)
      暴跌<-75%   → 基本面崩坏(divergence层双保险)
    这些是明确雷不是边缘票, 硬排除不漏好票。"""
    if name and ('ST' in name or '退' in name):
        return False, 'ST/退市'
    if pe is not None and (pe <= 0 or (isinstance(pe, float) and math.isnan(pe))):
        return False, '亏损/盈利存疑'
    if pledge is not None and pledge >= 50:
        return False, f'高质押{pledge:.0f}%'
    if price is not None and price < 1.5:
        return False, f'面值{price:.2f}'
    if dist120 is not None and dist120 < -75:
        return False, f'暴跌{dist120:.0f}%'
    return True, ''


def check_profit_quality(code):
    """扣非利润质量检查(2026-08-06东方精工教训: 净利38亿扣非0.8亿=卖子公司一次性).
    对比净利润vs营业利润(经营性), 如果差额巨大(一次性占比>50%)→利润质量存疑.
    返回 (deducted_ratio, flag), deducted_ratio=营业利润/净利润(>0.5正常, <0.2一次性占比大)."""
    # forecast reason优先检查(income表滞后,中报预告reason能提前发现一次性利润)
    try:
        fc = pro.forecast(ts_code=_ts_code(code), count=1)
        if fc is not None and len(fc):
            reason = str(fc.iloc[0].get('change_reason', ''))
            kws = ['出售', '处置', '股权转让', '合并报表', '终止确认', '剥离', '重组', '非经常']
            hits = [k for k in kws if k in reason]
            if hits:
                return -1, f'🚨一次性利润({",".join(hits)}): {reason[:60]}'
    except Exception:
        pass
    # income对比(年报经营性vs净利润)
    try:
        d = pro.income(ts_code=_ts_code(code), fields='period,total_revenue,n_income,operate_profit',
                       count=1)
        if d is None or d.empty:
            return None, ''
        r = d.iloc[0]
        n_income = float(r.get('n_income', 0) or 0)
        op_profit = float(r.get('operate_profit', 0) or 0)
        if n_income <= 0:
            return None, ''
        ratio = op_profit / n_income  # 营业利润/净利润
        if ratio < 0.2:
            return ratio, f'⚠️一次性利润占比>{(1-ratio)*100:.0f}%(营业利润{op_profit/1e8:.1f}亿 vs 净利润{n_income/1e8:.1f}亿)'
        elif ratio < 0.5:
            return ratio, f'利润质量偏弱(营业利润/净利润={ratio:.1%})'
        return ratio, ''
    except Exception:
        pass
    # forecast reason关键词检查(2026-08-06东方精工教训: 中报没发但预告reason说卖子公司)
    # income表滞后, 预告reason能提前发现一次性利润
    try:
        fc = pro.forecast(ts_code=_ts_code(code), count=1)
        if fc is not None and len(fc):
            reason = str(fc.iloc[0].get('change_reason', ''))
            kws = ['出售', '处置', '股权转让', '合并报表', '终止确认', '剥离', '重组', '非经常']
            hits = [k for k in kws if k in reason]
            if hits:
                return -1, f'🚨一次性利润({",".join(hits)}): {reason[:60]}'
    except Exception:
        pass
    return None, ''


def _ts_code(code):
    return f"{code}.{'SH' if code[:1] in ('6', '5', '9') else 'SZ'}"


def get_basics(codes):
    """批量取 pe_ttm/pb/circ_mv(全市场单日daily_basic,一次)。返回 {code:(pe,pb,mv亿)}, latest。"""
    latest = None
    for code in codes[:5]:
        try:
            d = pro.daily_basic(ts_code=_ts_code(code), fields='trade_date').head(1)
            if not d.empty and (latest is None or d.iloc[0]['trade_date'] > latest):
                latest = d.iloc[0]['trade_date']
        except Exception:
            pass
    db = {}
    if latest:
        try:
            dd = pro.daily_basic(trade_date=latest, fields='ts_code,pe_ttm,pb,circ_mv')
            for _, r in dd.iterrows():
                mv = r['circ_mv'] / 1e4 if pd.notna(r['circ_mv']) else None
                db[r['ts_code'][:6]] = (r['pe_ttm'], r['pb'], mv)
        except Exception:
            pass
    return db, latest


def get_growth(codes):
    """取业绩增长 profit_yoy/q_gr_yoy(每只fina_indicator,只对盈利票调以减量)。返回 {code:(profit_yoy,rev_yoy)}。"""
    fin = {}
    for i, code in enumerate(codes):
        try:
            d = pro.fina_indicator(ts_code=_ts_code(code), fields='profit_yoy,q_gr_yoy').head(1)
            if not d.empty:
                fin[code] = (d.iloc[0].get('profit_yoy'), d.iloc[0].get('q_gr_yoy'))
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  [增长 {i+1}/{len(codes)}]")
    return fin


def get_pledges(na, codes):
    """取大股东未解押质押比例(每只akshare)。返回 {code: 比例or None}。"""
    pz = {}
    for i, code in enumerate(codes):
        try:
            p = na.get_pledge_info(code)
            if p is not None and not p.empty and '状态' in p.columns and '占所持股份比例' in p.columns:
                wk = p[p['状态'].astype(str).str.contains('未解押')]
                if not wk.empty:
                    mz = pd.to_numeric(wk['占所持股份比例'], errors='coerce').max()
                    pz[code] = mz if pd.notna(mz) else 0
                else:
                    pz[code] = 0
            else:
                pz[code] = 0
        except Exception:
            pz[code] = None
        if (i + 1) % 20 == 0:
            print(f"  [质押 {i+1}/{len(codes)}]")
    return pz


def scan_mainline_leader():
    """主线龙头扫描: 技术买点(divergence核心) → 硬排雷 → 基本面质量打分。
    返回 (survivors, latest, filtered_out)。filtered_out=被硬排除的(含reason)。"""
    all_hits = scan_divergence()
    # 只对核心买点(confirmed+low_pos)打基本面分,减量
    hits = [h for h in all_hits if h.get('confirmed') and h.get('low_pos')]
    if not hits:
        return [], None, []
    codes = [h['code'] for h in hits]
    na = NewsAnalyzer()
    print(f"核心买点{len(hits)}只, 取基本面...")
    db, latest = get_basics(codes)
    pz = get_pledges(na, codes)
    # 硬排除(前置踢雷, 不靠分数): 先挂基本面字段, 再过mine_filter
    filtered_out = []
    for h in hits:
        pe, pb, mv = db.get(h['code'], (None, None, None))
        pledge = pz.get(h['code'])
        h['pe'] = pe; h['pb'] = pb; h['mv'] = mv; h['pledge'] = pledge
        passed, reason = mine_filter(pe, pledge, h.get('price'), h.get('name'), h.get('dist120'))
        if not passed:
            h['mine_reason'] = reason
            filtered_out.append(h)
    survivors = [h for h in hits if 'mine_reason' not in h]
    # 财务排雷五查(2026-08-23陈杰森视频方法论→自动化: 勾稽关系硬雷一票否决)
    # 只对mine_filter幸存者查(减API); 硬雷=存贷双高/年报净现背离/应收挤压>2倍
    try:
        from scripts.analyzers.financial_forensic import FinancialForensic
        ff = FinancialForensic()
        for h in list(survivors):
            fr = ff.check_code(h['code'])
            if isinstance(fr, dict) and fr.get('hard_hit'):
                h['mine_reason'] = f"财务五查:{'+'.join(fr['flags'])}"
                filtered_out.append(h)
                survivors.remove(h)
            elif isinstance(fr, dict) and fr.get('flags'):
                h['forensic_soft'] = fr['flags']  # 软雷只标注不否决
    except Exception as _e:
        print(f"  ⚠️财务五查跳过: {str(_e)[:50]}")
    if filtered_out:
        from collections import Counter
        cnt = Counter(h['mine_reason'] for h in filtered_out)
        print(f"  ⚠️硬排除{len(filtered_out)}只雷: {dict(cnt)} | 剩{len(survivors)}只打分")
    # 只对盈利survivor取增长,减量
    profit_codes = [h['code'] for h in survivors if h['pe'] and h['pe'] > 0]
    fin = get_growth(profit_codes) if profit_codes else {}
    # 扣非利润质量检查(2026-08-06东方精工教训: PE5但扣非0.8亿=一次性卖资产)
    # 只对低PE(<15)的查(高PE不太可能一次性),减少API调用
    low_pe_codes = [h['code'] for h in survivors if h['pe'] and h['pe'] < 15]
    profit_quality = {}
    for code in low_pe_codes:
        ratio, flag = check_profit_quality(code)
        if ratio is not None:
            profit_quality[code] = (ratio, flag)
    for h in list(survivors):   # 2026-08-31修: 迭代中remove会跳过下一元素→漏设quality→sort KeyError
        profit_yoy, rev_yoy = fin.get(h['code'], (None, None))
        # 毛利率/净利率(决赛圈才查income减量;2026-08-24 DeepSeek借鉴)
        try:
            inc1 = pro.income(ts_code=_ts_code(h['code']), fields='total_revenue,oper_cost,n_income', count=1)
            if inc1 is not None and not inc1.empty:
                rv, oc_, ni_ = (float(inc1.iloc[0][c] or 0) for c in ['total_revenue', 'oper_cost', 'n_income'])
                if rv > 0:
                    h['gross_margin'] = (rv - oc_) / rv * 100
                    h['net_margin'] = ni_ / rv * 100
        except Exception:
            pass
        # 杜邦三因子(2026-10-07: 杠杆型ROE一票降档; 金融豁免)
        try:
            from scripts.screener.dupont import dupont_factors, FIN_KEYWORDS
            d = dupont_factors(_ts_code(h['code']), tc=pro)
            if d and d.get('em') is not None:
                h['dupont'] = d
        except Exception:
            pass
        is_st = 'ST' in h.get('name', '') or '退' in h.get('name', '')
        h['profit_yoy'] = profit_yoy
        # 扣非PE(如果低PE且有利润质量数据)
        pe_ded = None
        pq = profit_quality.get(h['code'])
        if pq and pq[0] and pq[0] > 0 and h['pe']:
            pe_ded = h['pe'] / pq[0]  # 扣非PE = PE / (营业利润/净利润)
        h['profit_quality'] = pq[1] if pq else ''
        # 扣非负值检测(2026-08-29星源卓镁教训)
        pe_dedt_neg = False
        try:
            fi_dedt = pro.fina_indicator(ts_code=_ts_code(h['code']), fields='end_date,profit_dedt', count=1)
            if fi_dedt is not None and len(fi_dedt) and pd.notna(fi_dedt.iloc[0]['profit_dedt']):
                pe_dedt_neg = float(fi_dedt.iloc[0]['profit_dedt']) < 0
                if pe_dedt_neg:
                    h['mine_reason'] = f'扣非亏损({float(fi_dedt.iloc[0]["profit_dedt"])/1e8:.2f}亿)'
                    filtered_out.append(h)
                    survivors.remove(h)
                    continue
        except Exception:
            pass
        d = h.get('dupont') or {}
        is_fin = any(k in h.get('name', '') for k in ('银行', '证券', '保险', '金融'))
        h['quality'] = quality_scorecard(h['pe'], h['pb'], h['mv'], profit_yoy, rev_yoy, h['pledge'], is_st, pe_ded, pe_dedt_neg, dupont_em=d.get('em'), is_fin=is_fin)
        if h['quality'] >= 70:
            h['grade'] = '🥇质量背离'
        elif h['quality'] >= 50:
            h['grade'] = '🥈中等'
        else:
            h['grade'] = '⚠️基本面弱'
    survivors.sort(key=lambda h: (-h['quality'], not h['confirmed'], -h['strength']))
    return survivors, latest, filtered_out


def market_breakdown_status():
    """7大指数破20日低同步率 — 超跌程度标注(非预测信号)。

    回测(scripts/research/multi_index_breakdown_backtest.py): 破前低后收益为正,
    全破7/7组反弹最猛(68%正收益比,+1.82%/20天) → 破位=超跌买点,不是卖点。
    注: sync同步率本身无预测力(p=0.95),仅描述超跌程度;必叠底背离+缩量确认。
    """
    indices = [('000001SH', '上证'), ('399001SZ', '深成'), ('399006SZ', '创业板'),
               ('000016SH', '上证50'), ('000300SH', '沪深300'), ('000905SH', '中证500'),
               ('000852SH', '中证1000')]
    n_break = total = 0
    brk_names = []
    for code, name in indices:
        p = PROJECT_ROOT / f'data/store/kline/{code}/D.parquet'
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            dc = 'date' if 'date' in df.columns else 'trade_date'
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.sort_values(dc).reset_index(drop=True)
            if len(df) < 22:
                continue
            total += 1
            prev_min = df['close'].iloc[-21:-1].min()  # t-1及之前20日最低,不含当日→无未来
            if df['close'].iloc[-1] < prev_min:
                n_break += 1
                brk_names.append(name)
        except Exception:
            continue
    return n_break, total, brk_names


def main():
    hits, latest, filtered_out = scan_mainline_leader()
    if not hits:
        print("无核心买点"); return
    # 大盘环境(回测验证: 大盘坏时底背离夏普1.15 vs 好0.57, p=0.0000)
    market_tag = "未知"
    try:
        idx = pd.read_parquet(PROJECT_ROOT / 'data/store/kline/000001SH/D.parquet')
        for c in ['date', 'day', 'trade_date', 'datetime']:
            if c in idx.columns:
                idx = idx.rename(columns={c: 'date'})
        idx['date'] = pd.to_datetime(idx['date'])
        idx = idx.sort_values('date').reset_index(drop=True)
        idx['close'] = pd.to_numeric(idx['close'], errors='coerce')
        idx_ret20 = (idx['close'].iloc[-1] / idx['close'].iloc[-21] - 1) * 100
        if idx_ret20 > 0:
            market_tag = f"好(上证20日{idx_ret20:+.1f}%) 信号偏弱(夏普0.57) 谨慎"
        else:
            market_tag = f"坏(上证20日{idx_ret20:+.1f}%) ✅超跌反弹环境 信号更可靠(夏普1.15);⚠️防御板块背离例外:坏regime防御股背离20d-3.22% vs 非防御+2.94%(p=0.014,2026-09-11交互回测)"
    except Exception:
        pass
    # 破位超跌(回测:破前低=超跌买点非卖点,全破历史68%正收益;sync无预测力仅描述超跌程度)
    nb, tot, brk_names = market_breakdown_status()
    if tot > 0:
        joined = '/'.join(brk_names) if brk_names else ''
        if nb >= 5:
            brk_tag = f"破{nb}/{tot}({joined}) ✅超跌反弹区(全破历史68%正收益) 左侧买点,别被'破位危险'吓割"
        elif nb >= 3:
            brk_tag = f"破{nb}/{tot}({joined}) 中度超跌,左侧可分批(必叠底背离+缩量)"
        elif nb >= 1:
            brk_tag = f"破{nb}/{tot}({joined}) 轻微超跌"
        else:
            brk_tag = f"破0/{tot} 常规环境,无超跌"
    else:
        brk_tag = "指数数据缺失"
    top = [h for h in hits if h['grade'].startswith('🥇')]
    weak = [h for h in hits if h['grade'].startswith('⚠️')]
    print(f"\n📊 质量背离扫描器 | 技术买点 × 基本面质量(行业仅标注非筛选) | 数据{latest}")
    print(f"   核心买点: {len(hits)}只 | 🥇质量背离(≥70): {len(top)}只 | ⚠️基本面弱(<50): {len(weak)}只")
    print(f"   大盘环境: {market_tag}")
    print(f"   破位超跌: {brk_tag}")
    print(f"   铁律: 技术只选时机, 基本面质量(盈利+无高质押)决定买不买\n")
    print(f"{'代码':<7}{'名称':<10}{'行业':<10}{'价':>7}{'质量':>5}{'PE':>6}{'质押%':>6}  档位")
    print('-' * 70)
    for h in hits[:30]:
        pe = f"{h['pe']:.0f}" if h['pe'] and h['pe'] > 0 else ('亏' if h['pe'] else '-')
        pz = f"{h['pledge']:.0f}" if h['pledge'] is not None else '-'
        print(f"{h['code']:<7}{h['name'][:9]:<10}{h.get('industry', '')[:8]:<10}{h['price']:>7.2f}"
              f"{h['quality']:>5}{pe:>6}{pz:>6}  {h['grade']}")
    # 增速明细行(2026-08-24: 营收增速+毛利净利显性化,DeepSeek教训—扣非增但营收降须亮出来)
    print(f"\n{'':<7}增长明细: 扣非同比/营收同比/毛利率/净利率(营收降=结构换挡,需人工辨)")
    if h.get('dupont'):
        d = h['dupont']
        print(f"{'':<7}杜邦: ROE{d['roe']:.1f}% = 净利率{d['npm']:.1f}% × 周转{d.get('at') or '?'} × 杠杆{d.get('em') or '?'} ({d['dtype']})")
    for h in hits[:30]:
        py = h.get('profit_yoy'); ry = h.get('rev_yoy')
        if py is None and ry is None:
            continue
        pys = f"{py:+.0f}%" if py is not None else 'na'
        rys = f"{ry:+.0f}%" if ry is not None else 'na'
        swap = ' ⚠️营收收缩' if (py is not None and py > 0 and ry is not None and ry < 0) else ''
        gms = h.get('gross_margin'); nms = h.get('net_margin')
        margins = f"/毛利{gms:.0f}%/净利{nms:.0f}%" if gms is not None else ''
        print(f"{'':<7}{h['name'][:9]:<10} 扣非{pys} 营收{rys}{margins}{swap}")
    print(f"\n💡 只跟🥇质量背离(技术✓+基本面≥70);行业冷热仅作参考不加分; ⚠️基本面弱(亏损/高质押)即使技术买点也避开(航天教训)")
    if filtered_out:
        print(f"\n🚫 硬排除{len(filtered_out)}只雷(技术买点✓但基本面雷, 前置踢出不进打分):")
        for h in filtered_out[:12]:
            print(f"  {h['code']} {h.get('name','')[:9]:<10} {h.get('industry','')[:8]:<10} 价{h.get('price',0):>7.2f} → {h['mine_reason']}")


if __name__ == '__main__':
    main()
