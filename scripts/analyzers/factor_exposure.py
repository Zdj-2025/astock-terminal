#!/usr/bin/env python3
"""组合因子暴露监控 — 回答"你的12只持仓实际押了几个赌注"（2026-08-20建）。

背景: 用户灵魂拷问"视角太狭隘"→审计发现12只"分散"持仓实际44%押AI/科技单因子(含1000ETF隐性敞口)。
大跌日-10,000中70%是因子集中问题不是选股问题。本模块每日输出真实因子分布。

核心设计:
1. 个股直接映射到因子(通过行业+人工标注)
2. 指数ETF分解为底层因子(1000ETF≠"宽基"而是35%科技+15%医药+...)
3. 单一因子>35%告警(意味着一次板块崩=组合重创)
4. 有效因子数<5=集中度不足(信号:需要分散)

用法:
  from scripts.analyzers.factor_exposure import analyze_factor_exposure
  result = analyze_factor_exposure()  # 返回dict + 打印报告
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

# ═══════════════ 因子映射表（核心资产，持续维护） ═══════════════

# 个股→因子 直接映射（按行业+实际业务）
# 2026-08-29修正: ①光学光电子(面板:TCL/京东方)移出AI——驱动是面板价格周期非AI capex
#                ②半导体单列——国产替代+周期是主逻辑,与AI相关但不恒等
STOCK_FACTOR_MAP = {
    # AI算力/应用（IDC/算法软件/算力服务）
    'AI科技': {
        '软件服务', 'IT设备', '元器件', '通信设备', '电器仪表',
        '电脑设备',
    },
    # 半导体（芯片设计/制造/设备，国产替代+周期）
    '半导体': {
        '半导体',
    },
    # 面板显示（LCD/OLED，TV/IT需求+价格周期，AI关联弱）
    '面板显示': {
        '光学光电子',
    },
    # 医药（制药/器械/CXO/中药）
    '医药': {
        '化学制药', '生物制药', '医疗保健', '医药商业', '中成药',
    },
    # 金融（券商/银行/保险）
    '金融': {
        '证券', '银行', '保险', '多元金融',
    },
    # 周期/化工（化工/钢铁/有色/煤炭）
    '周期化工': {
        '化工原料', '染料涂料', '农药化肥', '化纤', '塑料', '橡胶',
        '普钢', '钢加工', '小金属', '煤炭开采', '铝', '黄金',
    },
    # 消费（食品/饮料/家电/家居/零售/旅游服务）
    '消费': {
        '食品', '乳制品', '白酒', '红黄酒', '啤酒', '家用电器',
        '超市连锁', '百货', '纺织', '服饰', '日用化工', '家居用品',
        '旅游服务', '酒店餐饮',
    },
    # 制造（机械/汽车/军工/建筑/电气设备）
    # 2026-09-30补: 电气设备原缺失落"其他"——该行业横跨AI链(服务器电源/液冷/结构件)
    # 与六张网(电网设备),先入制造桶,AI算力链主题经制造0.15权重间接受益
    '制造': {
        '专用机械', '机械基件', '汽车配件', '汽车整车', '航空',
        '航天', '船舶', '运输设备', '建筑工程', '装修装饰', '电气设备',
    },
    # 公用/防御（电力/水务/交运/地产）
    '公用防御': {
        '火力发电', '水力发电', '新型电力', '供气供热', '水务',
        '水运', '港口', '路桥', '铁路', '仓储物流', '全国地产', '区域地产',
    },
}

# 行业→因子 查找表（反向索引）
_INDUSTRY_TO_FACTOR = {}
for factor, industries in STOCK_FACTOR_MAP.items():
    for ind in industries:
        _INDUSTRY_TO_FACTOR[ind] = factor

# 个股级覆盖表（优先于行业映射）——行业归类对多元业务公司太粗时用
# 2026-08-29用户纠偏: TCL科技行业=元器件,但主营华星面板+光伏,驱动=面板价格周期,
# 与AI capex无实质关联;按行业划入AI科技会高估AI暴露(42%假数的来源之一)
STOCK_OVERRIDE = {
    '000100': '面板显示',  # TCL科技: 半导体显示+光伏
    '688122': '制造',      # 西部超导: 行业标"小金属"但定价权在军工钛合金+超导磁体(军品/科研定价),
                           # 非商品金属周期——归周期化工会假增高周期暴露(2026-09-30修)
}

# ETF/指数 分解表（底层因子权重，近似值）
INDEX_DECOMP = {
    '512100': {  # 中证1000: 中小盘，科技偏高（电子~20%半导体+软件通信~15%）
        'AI科技': 0.15, '半导体': 0.20, '医药': 0.15, '金融': 0.10,
        '周期化工': 0.15, '制造': 0.15, '消费': 0.05, '公用防御': 0.05,
    },
    '510500': {  # 中证500: 中盘，相对均衡
        'AI科技': 0.12, '半导体': 0.13, '医药': 0.15, '金融': 0.15,
        '周期化工': 0.20, '制造': 0.15, '消费': 0.05, '公用防御': 0.05,
    },
    '159841': {  # 证券ETF: 纯金融
        '金融': 1.00,
    },
    '515120': {  # 创新药ETF: 纯医药
        '医药': 1.00,
    },
    '512480': {  # 半导体ETF: 芯片指数（国产替代+半导体周期，非纯AI）
        '半导体': 1.00,
    },
}

# 告警阈值
ALERT_THRESHOLD = 0.35  # 单因子>35%告警
MIN_EFFECTIVE_FACTORS = 5  # 有效因子数<5=集中度不足

# ═══ 主题层（2026-09-30立）═══
# 动机: 因子层按行业桶回答"买了什么行业"，但风险叙事跨行业——
# 9/28暴跌归因"AI算力链获利兑现"，需要一个直接对应该叙事的数字，
# 此前"AI链33%"是手算口径，每次口径漂移就是一次TCL式误归因。
# 权重标定: AI算力链=AI科技(IDC/光模块/服务器/软件)满额,
#   半导体0.6(国产替代+周期与AI capex部分重合), 制造0.15(设备/结构件/电源间接受益)。
# 人工标定,季度复核; 身份主题(医药1.0等)保留用于统一输出格式。
THEME_MAP = {
    'AI算力链': {'AI科技': 1.00, '半导体': 0.60, '制造': 0.15},
    '医药链': {'医药': 1.00},
    '金融': {'金融': 1.00},
    '内需消费': {'消费': 1.00},
    '周期资源': {'周期化工': 1.00},
    '红利防御': {'公用防御': 1.00},
}
THEME_ALERT_THRESHOLD = 0.25  # 单主题>25%告警(严于因子35%——主题层才是真实相关性上限)


def analyze_factor_exposure(portfolio_path='config/portfolio.xlsx',
                            stock_list_path='data/store/stock_list.parquet',
                            verbose=True):
    """分析组合的真实因子暴露。

    Returns:
        dict: {
            'factors': {factor_name: {'mv': float, 'pct': float, 'stocks': [codes]}},
            'total_mv': float,
            'max_factor': (name, pct),
            'alerts': [str],
            'effective_factors': float,  # 赫芬达尔指数倒数
        }
    """
    pf = pd.read_excel(portfolio_path)
    sl = pd.read_parquet(stock_list_path).set_index('symbol')
    industry_map = sl['industry'].to_dict() if 'industry' in sl.columns else {}

    # 计算各持仓市值
    holdings = []
    for _, r in pf.iterrows():
        code = str(r['code']).zfill(6)
        if code == 'nan' or '奥普' in code or 'nan' in str(r.get('name','')):
            continue
        try:
            shares = int(r['shares'])
            if shares <= 0 or pd.isna(shares):
                continue
            # 价格优先级: K线最新收盘(每日自动更新) > portfolio last_price(手填易过期)
            # 2026-08-29修: 数据港last_price停在8/20的30.20,实际8/28收25.13
            px = 0
            kpath = f'data/store/kline/{code}/D.parquet'
            if os.path.exists(kpath):
                kdf = pd.read_parquet(kpath)
                dcol = [c for c in kdf.columns if 'date' in c.lower()]
                if dcol:
                    px = float(kdf.sort_values(dcol[0])['close'].iloc[-1])
            if px <= 0 or pd.isna(px):
                px_raw = r.get('last_price', 0)
                px = float(px_raw) if pd.notna(px_raw) and px_raw != '' else 0
            if px <= 0 or pd.isna(px):
                continue
            mv = shares * px
            if pd.isna(mv):
                continue
            industry = industry_map.get(code, '未知')
            holdings.append({'code': code, 'name': str(r['name']), 'shares': shares,
                           'price': px, 'mv': mv, 'industry': industry})
        except (ValueError, TypeError):
            continue

    if not holdings:
        if verbose:
            print("⚠️ 无有效持仓")
        return {'factors': {}, 'total_mv': 0, 'alerts': ['无持仓'], 'effective_factors': 0}

    total_mv = sum(h['mv'] for h in holdings)

    # 因子分解
    factor_mv = {}  # {factor: total_mv}
    factor_detail = {}  # {factor: [(code, name, mv, source)]}

    for h in holdings:
        code = h['code']
        if code in INDEX_DECOMP:
            # ETF: 按分解表拆分
            for factor, weight in INDEX_DECOMP[code].items():
                fmv = h['mv'] * weight
                factor_mv[factor] = factor_mv.get(factor, 0) + fmv
                factor_detail.setdefault(factor, []).append(
                    (code, h['name'], fmv, f'ETF分解{weight:.0%}'))
        else:
            # 个股: 覆盖表优先于行业映射
            factor = STOCK_OVERRIDE.get(code) or _INDUSTRY_TO_FACTOR.get(h['industry'], '其他')
            factor_mv[factor] = factor_mv.get(factor, 0) + h['mv']
            factor_detail.setdefault(factor, []).append(
                (code, h['name'], h['mv'], f'个股({h["industry"]})'))

    # 计算百分比
    factors = {}
    for factor, mv in sorted(factor_mv.items(), key=lambda x: -x[1]):
        pct = mv / total_mv
        factors[factor] = {'mv': mv, 'pct': pct, 'detail': factor_detail.get(factor, [])}

    # 告警
    alerts = []
    max_factor = ('', 0)
    for factor, info in factors.items():
        if info['pct'] > ALERT_THRESHOLD:
            alerts.append(f"⚠️ {factor}暴露{info['pct']:.1%}>{ALERT_THRESHOLD:.0%}——单因子崩=组合重创")
        if info['pct'] > max_factor[1]:
            max_factor = (factor, info['pct'])

    # 有效因子数（赫芬达尔指数的倒数）
    hhi = sum(info['pct'] ** 2 for info in factors.values())
    effective_n = 1 / hhi if hhi > 0 else 0
    if effective_n < MIN_EFFECTIVE_FACTORS:
        alerts.append(f"⚠️ 有效因子数{effective_n:.1f}<{MIN_EFFECTIVE_FACTORS}——集中度不足")

    if verbose:
        print(f"\n{'='*70}")
        print(f"🧮 组合因子暴露监控 | 总市值{total_mv:,.0f} | {len(holdings)}只持仓")
        print(f"{'='*70}\n")
        for factor, info in factors.items():
            bar = '█' * int(info['pct'] * 50)
            alert_tag = ' ⚠️' if info['pct'] > ALERT_THRESHOLD else ''
            print(f"  {factor:<8} {info['mv']:>9,.0f} ({info['pct']:>5.1%}) {bar}{alert_tag}")
            if info['pct'] > ALERT_THRESHOLD:
                for code, name, mv, src in info['detail'][:5]:
                    print(f"           └ {name}({code}) {mv:,.0f} [{src}]")

        print(f"\n  有效因子数: {effective_n:.1f}个 (目标≥{MIN_EFFECTIVE_FACTORS})")
        if alerts:
            print(f"\n  {'🚨' if len(alerts)>1 else '⚠️'} 告警:")
            for a in alerts:
                print(f"    {a}")
        else:
            print(f"\n  ✅ 因子分布均衡，无告警")

        # 建议
        if max_factor[1] > ALERT_THRESHOLD:
            print(f"\n  💡 建议: {max_factor[0]}因子超配{max_factor[1]:.0%}，")
            print(f"     下次减仓优先从该因子中选最弱标的")

    # 主题层（2026-09-30: 因子分布→跨因子叙事，AI链集中度的官方口径）
    themes, theme_alerts = analyze_theme_exposure(factors, total_mv, verbose=verbose)
    alerts.extend(theme_alerts)

    return {
        'factors': factors,
        'total_mv': total_mv,
        'max_factor': max_factor,
        'alerts': alerts,
        'effective_factors': effective_n,
        'themes': themes,
    }


def analyze_theme_exposure(factors, total_mv, verbose=True):
    """主题层暴露: 把因子分布再映射到跨因子叙事(AI算力链等)。

    Args:
        factors: analyze_factor_exposure 返回的 factors dict (含'mv')
        total_mv: 组合总市值
    Returns:
        dict: {theme: {'mv': float, 'pct': float}}, alerts: [str]
    """
    theme_mv = {}
    for theme, fw in THEME_MAP.items():
        tmv = sum(factors[f]['mv'] * w for f, w in fw.items() if f in factors)
        theme_mv[theme] = tmv

    themes = {t: {'mv': mv, 'pct': mv / total_mv if total_mv else 0}
              for t, mv in sorted(theme_mv.items(), key=lambda x: -x[1])}

    alerts = []
    for t, info in themes.items():
        if info['pct'] > THEME_ALERT_THRESHOLD:
            w_desc = '+'.join(f'{f}{w:.2f}' for f, w in THEME_MAP[t].items())
            alerts.append(f"⚠️ {t}主题{info['pct']:.1%}>{THEME_ALERT_THRESHOLD:.0%}"
                          f"（{w_desc}）——叙事级集中,单事件冲击无对冲")

    if verbose:
        print(f"\n{'='*70}")
        print(f"🎯 主题层暴露（跨因子叙事） | 告警线{THEME_ALERT_THRESHOLD:.0%}")
        print(f"{'='*70}\n")
        for t, info in themes.items():
            if info['pct'] < 0.005:
                continue
            bar = '█' * int(info['pct'] * 50)
            tag = ' ⚠️' if info['pct'] > THEME_ALERT_THRESHOLD else ''
            print(f"  {t:<6} {info['mv']:>9,.0f} ({info['pct']:>5.1%}) {bar}{tag}")
        if alerts:
            for a in alerts:
                print(f"  {a}")

    return themes, alerts


def compute_portfolio_correlation(codes, kline_dir='data/store/kline', days=20):
    """计算持仓股两两相关系数矩阵，标注>0.6的"同仓位"对。"""
    rets = {}
    for code in codes:
        path = f'{kline_dir}/{code}/D.parquet'
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            dcol = [c for c in df.columns if 'date' in c.lower()]
            if not dcol:
                continue
            t = df.sort_values(dcol[0]).tail(days + 1)
            r = t['close'].astype(float).pct_change().dropna().values
            if len(r) == days:
                rets[code] = r
        except Exception:
            continue

    if len(rets) < 2:
        return None, []

    R = pd.DataFrame(rets)
    corr = R.corr()

    # 找高相关对
    high_corr = []
    cols = list(rets.keys())
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if pd.notna(c) and c > 0.6:
                high_corr.append((cols[i], cols[j], c))

    return corr, high_corr


if __name__ == '__main__':
    result = analyze_factor_exposure()

    def _has_shares(pf, code):
        """该代码在xlsx中是否有正股数(防死行混入相关性)"""
        rows = pf[pf['code'].astype(str).str.zfill(6) == code]['shares']
        try:
            return any(int(s) > 0 for s in rows if pd.notna(s))
        except (ValueError, TypeError):
            return False

    # 相关性分析（2026-09-30修: 过滤shares<=0死行,虹软/松霖已清仓仍残留xlsx导致假高相关对）
    pf = pd.read_excel('config/portfolio.xlsx')
    codes = [str(r['code']).zfill(6) for _, r in pf.iterrows()
             if str(r['code']) != 'nan' and '奥普' not in str(r['code'])]
    codes = [c for c in codes if c != 'nan' and _has_shares(pf, c)]
    corr, high = compute_portfolio_correlation(codes)
    if high:
        print(f"\n{'='*70}")
        print(f"📉 高相关持仓对(>0.6 = 本质同一仓位)\n{'='*70}")
        for c1, c2, r in sorted(high, key=lambda x: -x[2]):
            print(f"  {c1} × {c2} = {r:.2f}")
