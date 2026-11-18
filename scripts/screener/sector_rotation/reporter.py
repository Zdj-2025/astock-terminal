"""
科技板块轮动监控系统 - 报告生成器 v2
"""
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_daily_report(results: List[Dict], report_date: str = None) -> str:
    if report_date is None:
        report_date = datetime.now().strftime('%Y-%m-%d')

    lines = []
    lines.append(f"📊 科技板块轮动日报 {report_date}")
    lines.append("")

    # 轮动排名
    lines.append("🧭 轮动排名：")
    for i, r in enumerate(results):
        s = r['total_score']
        emoji = r['emoji']
        signal = r['signal']
        change = r['change']
        pv = r.get('price_volume', {})
        ff = r.get('fund_flow', {})
        tech = r.get('technicals', {})
        br = r.get('breadth', {})

        line = f"  {i+1}. {r['sector']:<8} {s:>3}分 {emoji}{signal} {change}"
        # 补充关键信息
        extras = []
        if pv.get('surge_days', 0) > 0:
            extras.append(f"⚡{pv['surge_days']}天异动")
        if ff.get('consecutive_days', 0) > 0:
            extras.append(f"💰连{ff['consecutive_days']}天流入")
        if ff.get('consecutive_out_days', 0) > 0:
            extras.append(f"📉连{ff['consecutive_out_days']}天流出")
        if tech.get('macd_state') in ('红柱扩大', '即将金叉'):
            extras.append(f"📊{tech['macd_state']}")
        if extras:
            line += "  " + " ".join(extras)
        lines.append(line)

    # 信号变化
    lines.append("")
    lines.append("⚡ 信号变化：")
    changed = [r for r in results if r['change'] in ('▲', '▼', '↕')]
    if changed:
        for r in changed:
            sym = "📈" if r['change'] == '▲' else "📉" if r['change'] == '▼' else "🔄"
            pv = r.get('price_volume', {})
            ff = r.get('fund_flow', {})
            lines.append(f"  {sym} {r['sector']}：→{r['emoji']}{r['signal']}  {pv.get('details','')} {ff.get('details','')}")
    else:
        lines.append("  无变化")

    # 四维评分
    lines.append("")
    lines.append("📐 四维评分：")
    lines.append(f"  {'板块':<8} {'价格量能(40)':<14} {'资金流(25)':<14} {'技术(20)':<12} {'广度(15)':<12}")
    for r in results:
        pv_s = r.get('price_volume', {}).get('score', 0)
        ff_s = r.get('fund_flow', {}).get('score', 0)
        tech_s = r.get('technicals', {}).get('score', 0)
        br_s = r.get('breadth', {}).get('score', 0)
        lines.append(f"  {r['sector']:<8} {pv_s:<14} {ff_s:<14} {tech_s:<12} {br_s:<12}")

    # 操作建议 + 个股推荐
    lines.append("")
    lines.append("💡 操作建议：")
    actionable = [r for r in results if r['signal'] in ('关注', '潜伏', '启动', '强势')]
    if actionable:
        for r in actionable:
            lines.append(f"  {r['emoji']} {r['sector']}({r['signal']})：{r['action']}")
            if 'stock_picks' in r and r['stock_picks']:
                for p in r['stock_picks']:
                    lines.append(f"    → {p['name'] if p.get('name') else p['code']}({p['code']}) {p['score']}分 {p['signal']} 现价{p['close']:.2f} 5日{p['ret5']:+.1f}%")
    else:
        lines.append("  当前无明确操作信号")

    return "\n".join(lines)


def generate_sector_detail(result: Dict) -> str:
    lines = []
    lines.append(f"📋 {result['sector']} 详细分析")
    lines.append(f"综合评分: {result['total_score']}/100 {result['emoji']}{result['signal']}")
    lines.append(f"操作建议: {result['action']}")
    lines.append("")

    pv = result.get('price_volume', {})
    lines.append("① 价格+量能异动(40分):")
    lines.append(f"  得分: {pv.get('score', 0)}")
    lines.append(f"  20日涨幅: {pv.get('ret20', 0):+.1f}%")
    lines.append(f"  量比: {pv.get('vol_ratio', 0):.2f}x")
    lines.append(f"  近5日异动: {pv.get('surge_days', 0)}天 (最大单日{pv.get('max_surge_pct', 0):.1f}%)")
    lines.append(f"  位置分: {pv.get('pos_score', 0)} 量能分: {pv.get('vol_score', 0)}")
    lines.append("")

    ff = result.get('fund_flow', {})
    lines.append("② 资金流(25分):")
    lines.append(f"  得分: {ff.get('score', 0)}")
    lines.append(f"  5日净流入: {ff.get('net_5d', 0):+.1f}亿")
    lines.append(f"  连续净流入: {ff.get('consecutive_days', 0)}天")
    lines.append(f"  连续净流出: {ff.get('consecutive_out_days', 0)}天")
    lines.append("")

    tech = result.get('technicals', {})
    lines.append("③ 技术形态(20分):")
    lines.append(f"  得分: {tech.get('score', 0)}")
    lines.append(f"  f'={tech.get('f_prime', 0):.5f}  f''={tech.get('f_double_prime', 0):.6f}")
    lines.append(f"  MACD: {tech.get('macd_state', '-')}")
    lines.append("")

    br = result.get('breadth', {})
    lines.append("④ 板块广度(15分):")
    lines.append(f"  得分: {br.get('score', 0)}")
    lines.append(f"  上涨: {br.get('up_ratio', 0):.0%}  强势个股(涨>3%): {br.get('strong_count', 0)}只")

    return "\n".join(lines)
