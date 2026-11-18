#!/usr/bin/env python3
"""行业分析 CLI — 产业链挖掘、生命周期判断、行业对比、完整分析报告。

基于 config/industry_chains.json 的行业定义，泛化 ai_stock_miner 的产业链挖掘逻辑。
数据源: Tushare Pro（主力）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient, to_ts_code


# ── 行业配置加载 ──────────────────────────────────────────────────────────
def load_industry_config() -> Dict[str, Any]:
    """加载 config/industry_chains.json"""
    path = os.path.join(PROJECT_ROOT, "config", "industry_chains.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 过滤掉 _comment 键
    return {k: v for k, v in data.items() if not k.startswith("_")}


def find_matching_industry(keyword: str, config: Dict) -> Optional[str]:
    """模糊匹配行业名"""
    keyword_lower = keyword.lower()
    if keyword in config:
        return keyword
    for name, info in config.items():
        if keyword_lower in name.lower() or any(
            keyword_lower in c.lower() for c in info.get("concepts", [])
        ):
            return name
    return None


def build_dynamic_config(keyword: str, ts: TushareClient) -> Optional[Dict]:
    """未预配置的行业，从Tushare概念板块动态构建产业链配置"""
    all_concepts = ts.get_concepts()
    if all_concepts is None or all_concepts.empty:
        return None

    kw = keyword.lower()
    matched = []
    for _, row in all_concepts.iterrows():
        cname = str(row.get("name", ""))
        if kw in cname.lower() or cname.lower() in kw:
            matched.append({"id": row.get("code", ""), "name": cname})

    if not matched:
        return None

    # 自动分类到上中下游
    upstream_segs, midstream_segs, downstream_segs = [], [], []
    upstream_hints = ["芯片", "材料", "原料", "零部件", "光模块", "硅", "设备",
                     "减速器", "电机", "传感器", "EDA", "光刻", "试剂"]
    midstream_hints = ["制造", "组装", "电池", "数据中心", "云计算", "组件",
                      "晶圆", "封装", "服务器", "加工", "生产", "模块"]
    downstream_hints = ["应用", "汽车", "终端", "零售", "驾驶", "机器人",
                       "智能", "服务", "医疗", "教育", "金融", "电站", "消费"]

    for m in matched:
        cn = m["name"].lower()
        placed = False
        for h in upstream_hints:
            if h in cn:
                upstream_segs.append(m["name"])
                placed = True
                break
        if placed:
            continue
        for h in midstream_hints:
            if h in cn:
                midstream_segs.append(m["name"])
                placed = True
                break
        if placed:
            continue
        for h in downstream_hints:
            if h in cn:
                downstream_segs.append(m["name"])
                placed = True
                break
        if not placed:
            midstream_segs.append(m["name"])  # 默认归中游

    return {
        "upstream": upstream_segs[:5],
        "midstream": midstream_segs[:5],
        "downstream": downstream_segs[:5],
        "concepts": [keyword] + [m["name"] for m in matched[:5]],
        "lifecycle": "unknown",
        "penetration_estimate": "待判断",
        "key_metrics": [],
        "_dynamic": True,
    }


def resolve_industry(keyword: str, config: Dict, ts: TushareClient) -> tuple:
    """解析行业：先匹配预配置，没有则动态构建。返回 (industry_name, chain_config)"""
    name = find_matching_industry(keyword, config)
    if name:
        return name, config[name], False

    # 动态构建
    dynamic = build_dynamic_config(keyword, ts)
    if dynamic:
        return keyword, dynamic, True

    return None, None, False


# ── 产业链挖掘（泛化自 ai_stock_miner）──────────────────────────────────
def discover_chain_stocks(ts: TushareClient, industry_name: str,
                          chain_config: Dict) -> Dict[str, List[Dict]]:
    """按产业链环节挖掘相关股票"""
    concepts = chain_config.get("concepts", [])
    all_concepts_data = ts.get_concepts()
    if all_concepts_data is None or all_concepts_data.empty:
        return {}

    # Step 1: 匹配概念板块（用行业concepts + 产业链环节名一起搜索）
    search_keywords = list(concepts)  # 主关键词
    for tier_key in ["upstream", "midstream", "downstream"]:
        search_keywords.extend(chain_config.get(tier_key, []))
    # 去重
    search_keywords = list(set(k.lower() for k in search_keywords if k))

    matched_concepts = []
    for _, row in all_concepts_data.iterrows():
        cname = str(row.get("name", ""))
        cname_lower = cname.lower()
        for kw in search_keywords:
            if kw in cname_lower or cname_lower in kw:
                matched_concepts.append({
                    "id": row.get("code", ""),
                    "name": cname,
                })
                break

    if not matched_concepts:
        return {"上游": [], "中游": [], "下游": []}

    # Step 2: 为每个概念分配到上/中/下游
    # 策略：用概念名和产业链环节名双向匹配
    tier_keywords = {
        "upstream": chain_config.get("upstream", []),
        "midstream": chain_config.get("midstream", []),
        "downstream": chain_config.get("downstream", []),
    }

    def classify_concept(concept_name: str) -> str:
        """将概念分类到上/中/下游"""
        cn = concept_name.lower()
        for tier_key, keywords in tier_keywords.items():
            for kw in keywords:
                if kw.lower() in cn or cn in kw.lower():
                    return tier_key
        # 未匹配时：根据概念名关键词猜测
        upstream_hints = ["芯片", "光模块", "材料", "硅", "原料", "零部件", "减速器", "电机", "GPU", "半导体", "封装测试"]
        midstream_hints = ["制造", "服务器", "电池", "数据中心", "云计算", "组件", "晶圆", "封装", "存储"]
        downstream_hints = ["应用", "汽车", "终端", "服务", "零售", "驾驶", "机器人", "电站",
                           "智能物流", "智能医疗", "智能金融", "智能交通", "智能穿戴", "AI应用",
                           "虚拟", "工业机器人", "服务机器人"]
        for h in upstream_hints:
            if h.lower() in cn or cn in h.lower():
                return "upstream"
        for h in midstream_hints:
            if h.lower() in cn or cn in h.lower():
                return "midstream"
        for h in downstream_hints:
            if h.lower() in cn or cn in h.lower():
                return "downstream"
        return "midstream"  # 默认归中游

    # Step 3: 获取每个概念下的股票
    tier_stocks = {"upstream": [], "midstream": [], "downstream": []}
    for concept in matched_concepts:
        tier = classify_concept(concept["name"])
        try:
            concept_stocks = ts.get_concept_stocks(concept["id"])
            if concept_stocks is not None and not concept_stocks.empty:
                for _, sr in concept_stocks.head(10).iterrows():
                    tier_stocks[tier].append({
                        "code": sr.get("ts_code", ""),
                        "name": sr.get("name", ""),
                        "concept": concept["name"],
                    })
        except Exception:
            pass

    # 去重
    result = {}
    for tier_key in ["upstream", "midstream", "downstream"]:
        tier_cn = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}[tier_key]
        seen = set()
        unique = []
        for s in tier_stocks[tier_key]:
            if s["code"] not in seen:
                seen.add(s["code"])
                unique.append(s)
        result[tier_cn] = unique[:20]

    return result


# ── 行业龙头财务对比 ─────────────────────────────────────────────────────
def compare_leaders(ts: TushareClient, stock_codes: List[str]) -> Optional[pd.DataFrame]:
    """获取龙头公司最新财务指标对比"""
    rows = []
    for code in stock_codes[:10]:
        ts_code = to_ts_code(code) if len(code) == 6 else code
        try:
            df = ts.get_stock_basic(ts_code=ts_code)
            if df is not None and not df.empty:
                info = df.iloc[0]
                rows.append({
                    "code": ts_code,
                    "name": info.get("name", ""),
                    "industry": info.get("industry", ""),
                })
        except Exception:
            pass

    if not rows:
        return None

    # 补充最新财务数据
    for row in rows:
        try:
            fina = ts.get_financial_indicator(ts_code=row["code"], period="")
            if fina is not None and not fina.empty:
                latest = fina.iloc[0]
                row["roe"] = round(float(latest.get("roe", 0) or 0), 2)
                row["gross_margin"] = round(float(latest.get("grossprofit_margin", 0) or 0), 1)
                row["net_margin"] = round(float(latest.get("netprofit_margin", 0) or 0), 1)
                row["revenue_growth"] = round(float(latest.get("or_yoy", 0) or 0), 1)
                row["debt_ratio"] = round(float(latest.get("debt_to_assets", 0) or 0), 1)
        except Exception:
            pass

    return pd.DataFrame(rows)


# ── 生命周期辅助判断 ─────────────────────────────────────────────────────
def assess_lifecycle(chain_config: Dict, leaders_df: Optional[pd.DataFrame]) -> Dict:
    """辅助判断产业生命周期阶段"""
    result = {
        "configured_stage": chain_config.get("lifecycle", "unknown"),
        "penetration": chain_config.get("penetration_estimate", "N/A"),
        "evidence": [],
    }

    if leaders_df is not None and not leaders_df.empty:
        # 用龙头营收增速辅助判断
        growths = leaders_df["revenue_growth"].dropna()
        if len(growths) > 0:
            avg_growth = growths.mean()
            if avg_growth > 30:
                result["evidence"].append(f"龙头平均营收增速{avg_growth:.1f}%，偏成长期")
            elif avg_growth > 10:
                result["evidence"].append(f"龙头平均营收增速{avg_growth:.1f}%，偏成熟期")
            else:
                result["evidence"].append(f"龙头平均营收增速{avg_growth:.1f}%，增速放缓")

    if not result["evidence"]:
        result["evidence"].append(f"配置标注: {result['configured_stage']}，渗透率约{result['penetration']}")

    return result


# ── 报告生成 ─────────────────────────────────────────────────────────────
def run_tech_analysis_for_stocks(stock_codes: List[str], top_n: int = 5) -> str:
    """对龙头股票批量运行技术分析，返回汇总文本"""
    results = []
    for code in stock_codes[:top_n]:
        code6 = code[:6]
        try:
            import subprocess
            cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "cli", "tech_analysis_cli.py"), code6]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                  cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": PROJECT_ROOT})
            if proc.returncode == 0 and proc.stdout.strip():
                results.append(proc.stdout.strip())
        except Exception as e:
            results.append(f"  技术分析失败({code6}): {e}")
    return "\n\n".join(results)


def generate_report(industry_name: str, chain_config: Dict,
                    chain_stocks: Dict, lifecycle: Dict,
                    leaders_df: Optional[pd.DataFrame],
                    output_dir: Optional[str] = None,
                    include_tech: bool = False) -> str:
    """生成完整行业分析报告"""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# {industry_name}行业深度分析 | {today}",
        "",
        f"> 分析框架: 肖璟《如何快速了解一个行业》",
        f"> 数据截止: {today} | 来源: Tushare Pro",
        "",
        "---",
        "",
        "## 一、行业定义与产业链",
        "",
    ]

    # 产业链
    for tier_key in ["上游", "中游", "下游"]:
        tier_en = {"上游": "upstream", "中游": "midstream", "下游": "downstream"}[tier_key]
        segments = chain_config.get(tier_en, [])
        lines.append(f"**{tier_key}**：{' → '.join(segments) if segments else '待细分'}")
        stocks = chain_stocks.get(tier_key, [])
        if stocks:
            names = [f"{s['name']}({s['code'][:6]})" for s in stocks[:8]]
            lines.append(f"  代表公司：{'、'.join(names)}")
        lines.append("")

    # 生命周期
    lines.append("## 二、产业生命周期判断")
    lines.append("")
    stage_map = {
        "introductory": "导入期（渗透率<15%）",
        "growth": "成长期（渗透率15%-40%）",
        "mature": "成熟期（渗透率40%-70%）",
        "decline": "衰退期（渗透率>70%）",
    }
    stage_label = stage_map.get(lifecycle["configured_stage"], lifecycle["configured_stage"])
    lines.append(f"**当前判断**：{stage_label}")
    lines.append(f"**渗透率**：约 {lifecycle['penetration']}")
    for ev in lifecycle["evidence"]:
        lines.append(f"- {ev}")
    lines.append("")

    # 龙头对比
    if leaders_df is not None and not leaders_df.empty:
        lines.append("## 三、龙头公司财务对比")
        lines.append("")
        cols = ["name", "roe", "gross_margin", "net_margin", "revenue_growth", "debt_ratio"]
        col_labels = {"name": "公司", "roe": "ROE%", "gross_margin": "毛利率%",
                      "net_margin": "净利率%", "revenue_growth": "营收增速%", "debt_ratio": "负债率%"}
        available_cols = [c for c in cols if c in leaders_df.columns]
        if available_cols:
            lines.append("| " + " | ".join(col_labels.get(c, c) for c in available_cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(available_cols)) + " |")
            for _, row in leaders_df.iterrows():
                vals = []
                for c in available_cols:
                    v = row.get(c, "N/A")
                    vals.append(str(v) if v != "N/A" else "-")
                lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # PEST
    lines.append("## 四、PEST 外部因素")
    lines.append("")
    lines.append("| 维度 | 关键问题 | 待分析 |")
    lines.append("|------|---------|--------|")
    lines.append("| Political | 政策走向？扶持还是限制？ | 待补充 |")
    lines.append("| Economic | 宏观周期位置？利率影响？ | 待补充 |")
    lines.append("| Social | 人口/消费习惯变化？ | 待补充 |")
    lines.append("| Technological | 颠覆性技术？替代风险？ | 待补充 |")
    lines.append("")

    # 估值
    lines.append("## 五、估值参考")
    lines.append("")
    valuation_map = {
        "introductory": "VC 估值法（基于愿景/赛道潜力）",
        "growth": "PEG / PS（高增速支撑高倍数）",
        "mature": "PB / 周期股估值（景气底买、景气顶卖）",
        "decline": "警惕低估值陷阱（戴维斯双杀）",
    }
    val_method = valuation_map.get(lifecycle["configured_stage"], "综合评估")
    lines.append(f"**适用估值方式**：{val_method}")
    lines.append("")

    # 景气度指标
    lines.append("## 六、景气度跟踪指标")
    lines.append("")
    metrics = chain_config.get("key_metrics", [])
    if metrics:
        for m in metrics:
            lines.append(f"- {m}")
    else:
        lines.append("- 待设计（量/价/利/库存/预期）")
    lines.append("")

    # 风险
    lines.append("## 七、风险提示")
    lines.append("")
    lines.append("1. 行业分析基于公开数据，可能存在滞后性")
    lines.append("2. 产业链股票匹配为概念关键词关联，可能有遗漏或误配")
    lines.append("3. 渗透率为估算值，实际数据请参考权威来源")
    lines.append("4. 投资决策需结合个股技术面和仓位管理纪律")
    lines.append("")

    # 技术分析（可选）
    if include_tech:
        all_codes = []
        for slist in chain_stocks.values():
            all_codes.extend([s["code"] for s in slist])
        # 去重取前5
        seen = set()
        unique_codes = []
        for c in all_codes:
            if c not in seen:
                seen.add(c)
                unique_codes.append(c)
        if unique_codes:
            lines.append("## 八、龙头股技术分析（4维评分）")
            lines.append("")
            lines.append(f"> 以下对产业链中{min(len(unique_codes), 5)}只代表股运行技术分析")
            lines.append("")
            print(f"  正在运行技术分析（{min(len(unique_codes), 5)}只）...")
            tech_report = run_tech_analysis_for_stocks(unique_codes, top_n=5)
            lines.append(tech_report)
            lines.append("")
            lines.append("## 九、综合投资建议")
            lines.append("")
            lines.append("结合行业分析和技术分析，建议关注：")
            lines.append("- 生命周期阶段 + 技术面共振的标的")
            lines.append("- 行业景气向上 + 个股MACD多头的首选")
            lines.append("- 严格按4维评分操作：综合分≥+0.5才做多")
            lines.append("")

    # 保存

    # 保存
    report = "\n".join(lines)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{industry_name}行业分析_{datetime.now().strftime('%Y%m%d')}.md"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存: {fpath}")

    return report


# ── CLI 入口 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="行业分析工具")
    parser.add_argument("--chain", metavar="行业", help="产业链挖掘")
    parser.add_argument("--industry", metavar="行业", help="完整行业分析")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="行业对比")
    parser.add_argument("--lifecycle", metavar="行业", help="生命周期判断")
    parser.add_argument("--list", action="store_true", help="列出已配置行业")
    parser.add_argument("--output", metavar="DIR", help="报告输出目录")
    parser.add_argument("--tech", action="store_true", help="加入龙头股技术分析（4维评分）")

    args = parser.parse_args()
    config = load_industry_config()
    ts = TushareClient()

    if args.list or not any([args.chain, args.industry, args.compare, args.lifecycle]):
        print("已配置行业:")
        for name, info in config.items():
            stage = info.get("lifecycle", "?")
            concepts = ", ".join(info.get("concepts", [])[:3])
            print(f"  {name}: {stage} | 关键词: {concepts}")
        if not config:
            print("  (无) — 请在 config/industry_chains.json 中添加行业定义")
        return

    if args.chain:
        name, chain_config, is_dynamic = resolve_industry(args.chain, config, ts)
        if not name:
            print(f"未找到匹配行业 '{args.chain}'")
            print(f"可用预配置行业: {', '.join(config.keys())}")
            print("提示: 可以输入任意行业名，系统会从Tushare概念板块自动挖掘")
            return
        tag = "（动态发现）" if is_dynamic else ""
        print(f"\n=== {name} 产业链{tag} ===\n")
        stocks = discover_chain_stocks(ts, name, chain_config)
        for tier, slist in stocks.items():
            tier_en = {"上游": "upstream", "中游": "midstream", "下游": "downstream"}.get(tier, "")
            segments = chain_config.get(tier_en, [])
            print(f"{tier}（{' → '.join(segments) if segments else '自动挖掘'}）:")
            if slist:
                for s in slist[:10]:
                    print(f"  {s['name']}({s['code'][:6]}) [{s['concept']}]")
            else:
                print("  (未匹配到股票)")
            print()

    if args.industry:
        name, chain_config, is_dynamic = resolve_industry(args.industry, config, ts)
        if not name:
            print(f"未找到匹配行业 '{args.industry}'")
            print(f"可用预配置行业: {', '.join(config.keys())}")
            print("提示: 可以输入任意行业名，系统会自动挖掘")
            return
        tag = "（动态发现）" if is_dynamic else ""
        print(f"正在分析 {name}{tag} ...")

        chain_stocks = discover_chain_stocks(ts, name, chain_config)
        all_codes = []
        for slist in chain_stocks.values():
            all_codes.extend([s["code"] for s in slist])
        leaders_df = compare_leaders(ts, all_codes)
        lifecycle = assess_lifecycle(chain_config, leaders_df)

        output = args.output or os.path.join(
            PROJECT_ROOT, "reports", datetime.now().strftime("%Y-%m-%d")
        )
        report = generate_report(name, chain_config, chain_stocks, lifecycle, leaders_df, output,
                                include_tech=args.tech)
        if not output:
            print(report)

    if args.compare:
        results = []
        for kw in args.compare:
            n, c, dyn = resolve_industry(kw, config, ts)
            if n:
                results.append((n, c, dyn))
            else:
                print(f"未找到 '{kw}'")
                return
        n0, c0, d0 = results[0]
        n1, c1, d1 = results[1]
        dyn_tag = lambda d: "（动态）" if d else ""
        print(f"\n=== {n0}{dyn_tag(d0)} vs {n1}{dyn_tag(d1)} ===\n")
        for n, c, d in results:
            print(f"【{n}】")
            print(f"  生命周期: {c.get('lifecycle', '?')} | 渗透率: {c.get('penetration_estimate', '?')}")
            print(f"  上游: {' → '.join(c.get('upstream', []))}")
            print(f"  中游: {' → '.join(c.get('midstream', []))}")
            print(f"  下游: {' → '.join(c.get('downstream', []))}")
            print(f"  景气指标: {', '.join(c.get('key_metrics', []))}")
            print()

    if args.lifecycle:
        name, chain_config, is_dynamic = resolve_industry(args.lifecycle, config, ts)
        if not name:
            print(f"未找到匹配行业 '{args.lifecycle}'")
            return
        chain_stocks = discover_chain_stocks(ts, name, chain_config)
        all_codes = []
        for slist in chain_stocks.values():
            all_codes.extend([s["code"] for s in slist])
        leaders_df = compare_leaders(ts, all_codes[:5])
        lc = assess_lifecycle(chain_config, leaders_df)
        print(f"\n=== {name} 生命周期判断 ===")
        print(f"配置标注: {lc['configured_stage']}")
        print(f"渗透率: {lc['penetration']}")
        for ev in lc["evidence"]:
            print(f"  - {ev}")


if __name__ == "__main__":
    main()
