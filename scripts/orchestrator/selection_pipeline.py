#!/usr/bin/env python3
"""选股四层 Pipeline Orchestrator(2026-06-17, 选股总纲落地)。

四层漏斗(详见 memory/selection_framework.md):
① 板块方向(sector_rotation 综合分前3)
② 板块内选股(龙头 + V3趋势)
③ 公告催化确认(news_cli 关键词)
④ 择时风控(回踩买点 + 止损仓位)

用法: PYTHONPATH=. python scripts/orchestrator/selection_pipeline.py
      PYTHONPATH=. python scripts/orchestrator/selection_pipeline.py --no-catalyst  # 跳过慢的公告层
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.screener.tcl_pattern import compute_indicators, tcl_mask, STORE

ROTATION_DIR = PROJECT_ROOT / 'reports' / 'sector_rotation'
CATALYST_KEYS = ['融资', '定增', '收购', '重组', '合同', '中标', '业绩预增', '回购',
                 '增持', '政策', '补贴', '突破', '量产', '订单', '合作']
# 治理/法律/退市雷关键词(2026-06-20加, 避雷硬过滤, 学自博瑞医药取保候审案+山鹰质押踩雷)
MINE_KEYS = ['立案', '取保候审', '拘留', '刑事', '违规', '处罚', '问询函',
             '警示', '退市', '非标', '无法表示意见', '被查', '冻结']
LOW_PRICE_THRESHOLD = 2.0  # 低价面值退市风险阈值(<2元谨慎, A股连续20日<1元退市)


def filter_mines(candidates):
    """避雷硬过滤(2026-06-20): 剔除低价(<2元面值退市风险)+ST。返回 (净化列表, 被剔列表)。

    治理雷(涉案/高质押)在layer3查公告后检测, 此处只做价格/ST硬过滤。
    """
    clean, dropped = [], []
    for c in candidates:
        reason = None
        price = c.get('price', 0) or 0
        name = str(c.get('name', ''))
        if price < LOW_PRICE_THRESHOLD:
            reason = f"低价{price:.2f}(<{LOW_PRICE_THRESHOLD}元,面值退市风险)"
        elif 'ST' in name:
            reason = "ST/*ST"
        if reason:
            c['mine'] = reason
            dropped.append(c)
        else:
            c.setdefault('mine', '')
            clean.append(c)
    return clean, dropped


def layer0_scenario():
    """第〇层(2026-06-19新增): 当前预案 + 触发切换条件。

    学自一杆清《节后行情会怎么走?》的"预案思维": 不赌单一走势,
    设当前预案+备预案, 触发条件可验证, 边走边修正("直到拐点迹象越来越集中, 才一声令下")。
    数据基础: MarketRegimeDetector(4因子score, ±0.50为界) + 板块轮动状态。
    定位: 报告开头的"作战预案", 下方四层选股在此框架下执行。
    """
    from scripts.market_regime.regime_detector import MarketRegimeDetector
    from scripts.market_regime.regime_advisor import RegimeAdvisor

    lines = ["【🎯 当前预案 · 边走边修正】(预案思维: 不赌单一, 触发条件可验证)"]
    # ── 大盘预案 ──
    try:
        res = MarketRegimeDetector().detect()
        regime, score, conf = res['regime'], res['score'], res['confidence']
        cfg = RegimeAdvisor.get_config(regime)
        regime_cn = {'risk_on': '进攻🟢', 'neutral': '震荡🟡', 'risk_off': '防御🔴'}[regime]
        lines.append(f"大盘: {regime_cn} (环境分{score:+.2f} 置信度{conf:.0%})")
        lines.append(f"  主预案: {cfg['description']} | 单仓≤{int(cfg['max_position_per_stock']*100)}% "
                     f"总仓≤{int(cfg['max_total_exposure']*100)}% 止损{int(cfg['stop_loss_pct']*100)}%")
        lines.append("  切换触发(环境分±0.50为界, 边走边修正):")
        if regime != 'risk_on':
            lines.append(f"    🟢→进攻: 环境分升破+0.50 (现{score:+.2f}, 还差{0.5-score:+.2f})")
        if regime != 'risk_off':
            lines.append(f"    🔴→防御: 环境分跌破-0.50 (现{score:+.2f}, 还差{-0.5-score:+.2f})")
        # 各因子方向(告诉你离切换差在哪, 可验证)
        factors = res.get('factors', {})
        fc = {'domestic_indices': '指数', 'breadth': '宽度', 'external': '外部', 'capital_flow': '资金'}
        parts = []
        for k in ['domestic_indices', 'breadth', 'external', 'capital_flow']:
            s = factors.get(k, {}).get('score', 0)
            d = factors.get(k, {}).get('detail', '')
            parts.append(f"{fc[k]}{'↑' if s > 0 else '↓' if s < 0 else '→'}({d})")
        lines.append(f"  当前因子: {' | '.join(parts)}")
    except Exception as e:
        lines.append(f"大盘: ⚠️环境检测失败({type(e).__name__}), 默认按震荡精选处理")
    # ── 板块主线预案 ──
    top3, _ = layer1_top_sectors(3)
    if top3:
        names = ' / '.join(f"{n}({s}分)" for n, s in top3)
        lines.append(f"板块主线: {names}")
        lines.append("  退出触发: 综合分跌破50(转⚪观望) 或 资金连出≥2天 或 MACD死叉")
    return '\n'.join(lines)


def layer1_top_sectors(n=3):
    """第一层:读最新板块轮动报告, 解析综合分前N。"""
    files = sorted(glob.glob(str(ROTATION_DIR / 'rotation_*.txt')), reverse=True)
    if not files:
        return [], '?'
    txt = open(files[0], encoding='utf-8').read()
    sectors = [(m.group(1), int(m.group(2)))
               for m in re.finditer(r'\d+\.\s*([一-龥A-Za-z]+)\s*(\d+)分', txt)]
    return sectors[:n], os.path.basename(files[0])


def layer2_leaders_with_v3(top3_names):
    """第二层:前3板块的龙头成分股 + V3趋势加速。"""
    from scripts.screener.leader_tech_scanner import get_leaders
    from scripts.core.tushare_client import TushareClient
    leaders = get_leaders()
    name_map = {}
    try:
        sl = TushareClient().get_stock_list()
        name_map = dict(zip(sl['symbol'].astype(str).str.zfill(6), sl['name']))
    except Exception:
        pass
    out = []
    for code, sector in leaders:
        if sector not in top3_names:
            continue
        p = STORE / code / 'D.parquet'
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            df['date'] = pd.to_datetime(df['date'])
            df = compute_indicators(df)
            last = df.iloc[-1]
            if last['f1'] > 0.3 and last['close'] > last['MA20']:
                out.append({
                    'code': code, 'name': name_map.get(code, ''), 'sector': sector,
                    'price': last['close'], 'f1': round(last['f1'], 2),
                    'pullback': bool(tcl_mask(df).iloc[-1]),
                    'dh60': f"{last['dh60']*100:+.1f}%",
                    'date': last['date'].date(),
                })
        except Exception:
            pass
    return out


def layer3_catalyst(candidates, skip=False):
    """第三层:公告催化(正向) + 治理雷检测(立案/质押/涉案→剔除)。

    2026-06-20加治理雷: 学自博瑞医药实控人取保候审案 + 山鹰控股股东质押踩雷。
    - 正向催化(融资/收购/合同/业绩预增等): 🔥标记
    - 治理雷(立案/取保候审/违规/退市等): 💀标记→排序末尾剔除
    - 质押: ⚠️标记(警示, 高质押有平仓风险)
    - 无催化无雷: 无催化(降级)
    skip=True则标待查(不查公告)。
    """
    if skip:
        for c in candidates:
            c['catalyst'] = '⏳待查'
            c.setdefault('mine', '')
        return candidates
    import subprocess
    for c in candidates:
        try:
            r = subprocess.run(
                [sys.executable, 'scripts/cli/news_cli.py', c['code'], '--announcements', '--days', '30'],
                capture_output=True, text=True, timeout=30,
                cwd=str(PROJECT_ROOT), env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)})
            out = r.stdout
            pos = [k for k in CATALYST_KEYS if k in out]
            mines = [k for k in MINE_KEYS if k in out]
            has_pledge = '质押' in out
            if mines:
                c['catalyst'] = f"💀治理雷:{'/'.join(mines[:2])}"
                c['mine'] = f"治理雷({mines[0]})"
            elif pos:
                c['catalyst'] = '🔥' + '/'.join(pos[:3])
                c.setdefault('mine', '')
            elif has_pledge:
                c['catalyst'] = '⚠️质押(警惕平仓)'
                c['mine'] = '质押'
            else:
                c['catalyst'] = '无催化'
                c.setdefault('mine', '')
        except Exception:
            c['catalyst'] = '?(news失败)'
            c.setdefault('mine', '')
    return candidates


def layer4_timing(candidates):
    """第四层:择时 + 止损仓位建议。"""
    for c in candidates:
        if c['pullback']:
            c['action'] = '🎯回踩买点(可入)'
            c['stop'] = f"止损-{c['price']*0.92:.2f}(-8%) 首仓3-4万"
        else:
            c['action'] = '⏸趋势在(等回踩)'
            c['stop'] = '观望'
    return candidates


def main():
    ap = argparse.ArgumentParser(description='选股四层Pipeline')
    ap.add_argument('--no-catalyst', action='store_true', help='跳过公告层(慢)')
    ap.add_argument('--top', type=int, default=5)
    args = ap.parse_args()

    print('=' * 64 + '\n📊 选股四层 Pipeline(板块→选股→催化→择时)\n' + '=' * 64)
    # ⓪ 当前预案(预案思维: 先摆预案+触发条件, 下方四层选股在此框架下执行)
    print('\n' + layer0_scenario())
    # ① 板块
    top3, src = layer1_top_sectors(args.top)
    print(f"\n【① 板块方向 · 综合分前{args.top}】(来源:{src})")
    for name, score in top3:
        print(f"  {name}  {score}分")
    top3_names = [s[0] for s in top3]
    # ② 选股
    cands = layer2_leaders_with_v3(top3_names)
    # 避雷硬过滤(低价<2元面值退市风险 + ST)
    cands, dropped_mines = filter_mines(cands)
    if dropped_mines:
        print(f"\n[🚫避雷] 剔除低价/ST {len(dropped_mines)}只: " +
              ', '.join(f"{c['code']}{c['name']}({c['mine']})" for c in dropped_mines))
    print(f"\n【② 板块内龙头+V3选股】{len(cands)}只(避雷后)")
    # ③ 择时+风控(程序化)
    cands = layer4_timing(cands)
    # ④ 公告催化 + 治理雷检测(最后人工确认)
    cands = layer3_catalyst(cands, skip=args.no_catalyst)
    # 排序: 治理雷/质押→末尾, 有催化优先, 回踩优先, f1高优先
    cands = sorted(cands, key=lambda x: (
        1 if x.get('mine') else 0,                   # 雷/质押排后(净化)
        0 if x['catalyst'].startswith('🔥') else 1,  # 有催化优先
        not x['pullback'],                            # 回踩买点优先
        -x['f1'],                                     # f1高优先
    ))
    # 分离: 净化(无雷) vs 雷区(治理雷/质押)
    clean = [c for c in cands if not c.get('mine')]
    mined = [c for c in cands if c.get('mine')]
    if not cands:
        print("  (前N板块内无龙头满足V3趋势)")
    print(f"\n【✅ 净化名单】{len(clean)}只(技术+择时, 无治理雷):")
    for c in clean:
        print(f"  {c['code']} {c['name']:<8}[{c['sector']}] 价{c['price']:.2f} "
              f"f1={c['f1']:+.2f} dh60={c['dh60']} | {c['action']} | 催化:{c['catalyst']}")
    if mined:
        print(f"\n【🚫 雷区(已剔除)】{len(mined)}只:")
        for c in mined:
            print(f"  {c['code']} {c['name']:<8} 价{c['price']:.2f} | ⚠️{c['mine']} | {c['catalyst']}")
    # 主力信号(仅净化名单)
    buy = [c for c in clean if c['pullback']]
    print(f"\n💡 主力信号(净化+回踩买点): {len(buy)}只" +
          (f" → {'/'.join(c['code']+c['name'] for c in buy)}" if buy else "(暂无,等回踩)"))


if __name__ == '__main__':
    main()
