#!/usr/bin/env python3
"""中美会晤+马斯克产业链筛选器。

筛选逻辑（2026-06-13）：
  1. 马斯克产业链（SpaceX/特斯拉/机器人/星链）相关A股
  2. 中美企业家晚宴出席企业及关联企业
  3. 优先：从政府/中央财政/超大型企业拿订单的
  4. 排除：资源化企业（原材料为主，没有议价权）
  5. 排除：纯市场化企业（to C 消费品，竞争激烈）

筛选维度：
  - 订单来源: 政府/央企/军工/SpaceX/特斯拉 = 加分
  - 客户集中度: 大客户依赖度（前5客户占比）
  - 技术壁垒: 是否有不可替代性
  - 政策方向: 是否符合国家战略

用法:
  PYTHONPATH=. python scripts/screener/musk_chain_screener.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 核心股票池 ──────────────────────────────────────────

@dataclass
class ChainStock:
    """产业链个股。"""
    code: str
    name: str
    chain: str           # SpaceX/特斯拉/机器人/晚宴/星链
    role: str            # 核心供应商/二级供应商/间接关联
    product: str         # 供应产品
    order_source: str    # 订单来源: 政府/央企/SpaceX/特斯拉/市场
    is_excluded: bool = False
    exclude_reason: str = ""
    priority: int = 0    # 1=最高, 5=最低
    notes: str = ""


# ── 候选股票定义 ──────────────────────────────────────────
# 基于公开信息整理，按三个筛选维度

CANDIDATES = [
    # ═══ SpaceX产业链 ═══
    ChainStock("002149", "西部材料", "SpaceX", "核心供应商", "钛合金材料(SpaceX中国唯一供应商传闻)", "市场",
               notes="被追问是否供货，管理层回应'对接海外商业航天'但未确认。市值偏小。"),
    ChainStock("601002", "宝钛股份", "SpaceX", "核心供应商", "航天级钛合金(NASA认证)", "央企",
               priority=1, notes="国企龙头，NASA认证+SpaceX供货，订单来源含军工/央企。"),
    ChainStock("688800", "派克新材", "SpaceX", "核心供应商", "火箭箭体/发动机锻件(NASA认证)", "市场",
               notes="NASA认证通过，精密锻造壁垒高。但非国企。"),
    ChainStock("603985", "超捷股份", "SpaceX", "核心供应商", "钛合金紧固件(墨西哥工厂交付)", "市场",
               notes="通过国际认证，有墨西哥工厂，但规模小。"),
    ChainStock("603612", "再升科技", "SpaceX", "二级供应商", "高硅氧纤维(超高温绝热)", "市场",
               notes="公告称占营收极低，暂无在手订单。"),
    ChainStock("002792", "通宇通讯", "星链", "核心供应商", "卫星互联网终端(SpaceX认证)", "市场",
               notes="星链地面终端通过认证，但市值小，竞争激烈。"),
    ChainStock("300136", "信维通信", "星链", "核心供应商", "星链地面接收终端核心部件", "市场+大客户",
               notes="苹果+星链双供应链，大客户依赖。"),

    # ═══ 特斯拉汽车产业链 ═══
    ChainStock("300124", "汇川技术", "特斯拉", "核心供应商", "伺服电机/驱动器", "大客户+市场",
               notes="工控龙头，特斯拉+国内新能源车双线，但偏市场化。"),
    ChainStock("002594", "比亚迪", "晚宴+特斯拉竞争", "间接关联", "新能源车(竞争+合作)", "政府+市场",
               notes="民企座谈会出席，非晚宴名单，但国家级支持明确。体量太大。"),
    ChainStock("300750", "宁德时代", "特斯拉", "核心供应商", "动力电池(Megapack唯一供应商)", "大客户+政府",
               priority=1, notes="特斯拉储能唯一电池供应商，2026年全球配套超50GWh。政府订单+大客户。"),
    ChainStock("002475", "立讯精密", "特斯拉+苹果", "核心供应商", "连接器/线束", "大客户",
               notes="苹果+特斯拉双线，精密制造能力强。但to C端弱。"),

    # ═══ 特斯拉机器人(Optimus) ═══
    ChainStock("002050", "三花智控", "机器人", "核心供应商", "旋转关节总成", "大客户+政府",
               priority=2, notes="特斯拉机器人核心执行器，同时是新能源车热管理龙头。大客户+政策方向。"),
    ChainStock("601689", "拓普集团", "机器人", "核心供应商", "线性关节总成", "大客户",
               priority=2, notes="特斯拉汽车+机器人双供应商，一体化能力。"),
    ChainStock("601100", "恒立液压", "机器人", "核心供应商", "丝杠(核心传动)", "大客户+政府",
               notes="液压龙头，丝杠是机器人核心部件，壁垒高。"),
    ChainStock("603667", "五洲新春", "机器人", "核心供应商", "丝杠产品", "市场",
               notes="机器人丝杠供应商，但市值小，议价权弱。"),
    ChainStock("300433", "蓝思科技", "晚宴+SpaceX+机器人", "核心供应商", "机器人壳体+特斯拉玻璃+苹果玻璃", "大客户",
               priority=1, notes="⚠️晚宴出席(周群飞)! 苹果+特斯拉+机器人三线核心供应商。大客户模式。"),

    # ═══ 脑机接口(Neuralink) ═══
    ChainStock("002324", "普利特", "Neuralink", "间接关联", "LCP薄膜(脑机接口材料)", "市场",
               notes="与海外客户开发LCP薄膜用于脑机接口，概念偏早。"),

    # ═══ xAI数据中心 ═══
    ChainStock("002112", "三变科技", "xAI", "核心供应商", "变压器(直接用于xAI数据中心)", "大客户",
               notes="xAI变压器直接供货，概念明确但市值小。"),

    # ═══ 晚宴出席企业(A股关联) ═══
    ChainStock("600519", "贵州茅台", "晚宴", "间接关联", "晚宴用酒(推测)", "政府+市场",
               is_excluded=True, exclude_reason="纯消费品，与产业链无关"),
    ChainStock("000651", "格力电器", "晚宴关联(海尔竞对)", "间接关联", "家电", "市场",
               is_excluded=True, exclude_reason="纯市场化to C家电"),
    ChainStock("600690", "海尔智家", "晚宴", "直接出席(周云杰)", "家电(GE Appliances)", "市场",
               is_excluded=True, exclude_reason="纯市场化to C家电，资源化不严重但非政府订单型"),
    ChainStock("600029", "南方航空", "晚宴关联", "间接关联", "航空(国航关联)", "央企",
               notes="国航出席晚宴，南航是同行央企，但与马斯克产业链无关。"),
    ChainStock("600887", "伊利股份", "民企座谈会", "间接关联", "乳制品", "市场",
               is_excluded=True, exclude_reason="纯消费品，与产业链无关"),

    # ═══ 万向集团关联 ═══
    ChainStock("000559", "万向钱潮", "晚宴", "直接出席(鲁伟鼎/万向集团)", "汽车零部件", "大客户+市场",
               notes="⚠️晚宴出席! 万向在美国20+工厂，深度嵌入美国汽车供应链。但A股万向钱潮是汽车零部件，资源化风险。"),

    # ═══ 福耀玻璃关联 ═══
    ChainStock("600660", "福耀玻璃", "晚宴", "直接出席(曹晖)", "汽车玻璃", "大客户+政府",
               priority=2, notes="⚠️晚宴出席! 在美2000+工人，嵌入式出海典范。大客户(车企)订单模式。"),

    # ═══ 小米/联想关联(A股供应链) ═══
    ChainStock("002456", "欧菲光", "晚宴关联(小米/联想供应链)", "二级供应商", "摄像头模组", "大客户",
               notes="小米/联想的摄像头供应商，但财务状况不稳定。"),
    ChainStock("603501", "韦尔股份", "晚宴关联(小米/联想供应链)", "核心供应商", "CIS芯片", "大客户+市场",
               notes="小米/联想核心芯片供应商，技术壁垒高，但半导体周期性强。"),

    # ═══ 中国商飞产业链 ═══
    ChainStock("600893", "航发动力", "晚宴(商飞关联)", "核心供应商", "航空发动机(C919)", "央企+军工",
               priority=1, notes="⚠️商飞出席晚宴! C919发动机核心供应商，央企军工，纯政府订单型。"),
    ChainStock("600760", "中航沈飞", "晚宴(商飞关联)", "核心供应商", "军用飞机/航空制造", "央企+军工",
               priority=1, notes="军工龙头，央企，纯政府订单。但估值偏高。"),
    ChainStock("600862", "中航高科", "晚宴(商飞关联)", "核心供应商", "航空复合材料(C919)", "央企+军工",
               priority=1, notes="C919复合材料供应商，央企，技术壁垒高。"),
    ChainStock("600118", "中国卫星", "晚宴(商飞关联)", "核心供应商", "卫星制造", "央企+军工",
               notes="央企，卫星制造龙头。但利润率低。"),
    ChainStock("600879", "航天电子", "晚宴(商飞关联)", "核心供应商", "航天电子设备", "央企+军工",
               notes="央企，卫星产业链核心。"),

    # ═══ 国家级战略(非晚宴但与政策高度相关) ═══
    ChainStock("688598", "金博股份", "SpaceX关联+国产替代", "核心供应商", "碳纤维复合材料(热场)", "大客户",
               notes="光伏热场龙头，技术可延伸至航天。但当前主营光伏，受行业周期影响大。"),
]


def run_screen(data_root: Path | None = None):
    """执行筛选。"""
    print("=" * 70)
    print("🔍 中美会晤 + 马斯克产业链筛选")
    print("=" * 70)

    # ── 第一步：排除资源化和纯市场化企业 ──
    print("\n📌 第一步：排除资源化/纯市场化企业")
    excluded = [s for s in CANDIDATES if s.is_excluded]
    remaining = [s for s in CANDIDATES if not s.is_excluded]

    for s in excluded:
        print(f"  ❌ {s.code} {s.name} — {s.exclude_reason}")

    print(f"\n  排除 {len(excluded)} 只，剩余 {len(remaining)} 只")

    # ── 第二步：按订单来源分级 ──
    print(f"\n📌 第二步：按订单来源分级")
    print(f"  优先级: 央企/军工/政府订单 > 大客户(SpaceX/特斯拉) > 纯市场化\n")

    tier1 = [s for s in remaining if "央企" in s.order_source or "军工" in s.order_source or "政府" in s.order_source]
    tier2 = [s for s in remaining if s not in tier1 and ("大客户" in s.order_source or "SpaceX" in s.order_source or "特斯拉" in s.order_source)]
    tier3 = [s for s in remaining if s not in tier1 and s not in tier2]

    print(f"  🥇 第一梯队(政府/央企/军工订单): {len(tier1)}只")
    for s in tier1:
        print(f"     {s.code} {s.name} | {s.chain} | {s.product} | 订单:{s.order_source}")

    print(f"\n  🥈 第二梯队(大客户/SpaceX/特斯拉订单): {len(tier2)}只")
    for s in tier2:
        print(f"     {s.code} {s.name} | {s.chain} | {s.product} | 订单:{s.order_source}")

    print(f"\n  🥉 第三梯队(市场化，需精选): {len(tier3)}只")
    for s in tier3:
        print(f"     {s.code} {s.name} | {s.chain} | {s.product} | 订单:{s.order_source}")

    # ── 第三步：技术面筛选（如果有数据）──
    if data_root and data_root.exists():
        print(f"\n📌 第三步：技术面筛选(V3评分)")
        from scripts.screener.v3_derivative_scorer import V3DerivativeScorer

        scorer = V3DerivativeScorer(data_root=data_root)
        results = []

        for s in tier1 + tier2:
            parquet = data_root / "kline" / s.code / "D.parquet"
            if not parquet.exists():
                results.append((s, None, "无数据"))
                continue
            try:
                import pandas as pd
                df = pd.read_parquet(parquet)
                dt_col = "trade_date" if "trade_date" in df.columns else "date"
                df[dt_col] = pd.to_datetime(df[dt_col])
                df = df.sort_values(dt_col).reset_index(drop=True)

                v3 = scorer.score_stock(s.code, df)
                from scripts.screener.market_regime import _sma
                closes = df["close"].values.astype(float)
                ma20 = _sma(closes, 20)[-1]
                close = closes[-1]
                above_ma20 = close > ma20
                ret20 = (closes[-1] / closes[-20] - 1) * 100 if len(closes) >= 20 else 0

                signal = "🚀趋势加速" if v3.f1 > 0 and v3.f2 > 0 and above_ma20 else \
                         "🔄底部反转" if v3.f1 >= 0 and v3.f2 > 0 and not above_ma20 else \
                         "⚠️减速" if v3.f1 > 0 and v3.f2 < 0 else \
                         "🔴下跌" if v3.f1 < 0 and v3.f2 < 0 else "⚪震荡"

                results.append((s, v3, f"{signal} v3={v3.total:.1f} f'={v3.f1:+.3f} f''={v3.f2:+.3f} ret20={ret20:+.1f}%"))
            except Exception as e:
                results.append((s, None, f"计算失败: {e}"))

        print(f"\n  {'代码':<8s} {'名称':<8s} {'梯队':<4s} {'产业链':<14s} {'信号':<6s} {'V3':>4s} {'f1':>7s} {'f2':>7s} {'ret20':>7s}")
        print("  " + "-" * 75)
        for s, v3, info in results:
            tier = "🥇" if s in tier1 else "🥈"
            if v3:
                from scripts.screener.market_regime import _sma
                closes2 = pd.read_parquet(data_root / "kline" / s.code / "D.parquet")["close"].values.astype(float)
                ma20_2 = _sma(closes2, 20)[-1]
                above = closes2[-1] > ma20_2
                ret20_2 = (closes2[-1] / closes2[-20] - 1) * 100 if len(closes2) >= 20 else 0
                sig = "🚀" if v3.f1 > 0 and v3.f2 > 0 and above else \
                      "🔄" if v3.f2 > 0 else \
                      "⚠️" if v3.f2 < 0 and v3.f1 > 0 else "⬇️"
                print(f"  {s.code:<8s} {s.name:<8s} {tier:<4s} {s.chain:<14s} {sig:<6s} {v3.total:>4.1f} {v3.f1:>+7.3f} {v3.f2:>+7.3f} {ret20_2:>+6.1f}%")
            else:
                print(f"  {s.code:<8s} {s.name:<8s} {tier:<4s} {s.chain:<14s} {'❓':<6s} {'N/A':>4s}")
    else:
        print(f"\n⚠️ 无数据目录，跳过技术面筛选")

    # ── 汇总推荐 ──
    print(f"\n{'='*70}")
    print(f"📋 筛选总结")
    print(f"{'='*70}")

    # 晚宴直接出席 + 政府订单 = 最优
    dinner_gov = [s for s in remaining if "晚宴" in s.chain and ("央企" in s.order_source or "政府" in s.order_source)]
    dinner_big = [s for s in remaining if "晚宴" in s.chain and s not in dinner_gov]
    musk_gov = [s for s in remaining if s not in dinner_gov and s not in dinner_big and ("央企" in s.order_source or "政府" in s.order_source)]
    musk_core = [s for s in remaining if s not in dinner_gov and s not in dinner_big and s not in musk_gov]

    print(f"\n  ⭐ 第一优先: 晚宴出席 + 政府订单 ({len(dinner_gov)}只)")
    for s in dinner_gov:
        print(f"     {s.code} {s.name} — {s.notes}")

    print(f"\n  ⭐ 第二优先: 晚宴出席 + 大客户 ({len(dinner_big)}只)")
    for s in dinner_big:
        print(f"     {s.code} {s.name} — {s.notes}")

    print(f"\n  ✅ 第三优先: SpaceX/特斯拉 + 政府/央企 ({len(musk_gov)}只)")
    for s in musk_gov:
        print(f"     {s.code} {s.name} — {s.notes}")

    print(f"\n  ✅ 第四优先: SpaceX/特斯拉核心供应商 ({len(musk_core)}只)")
    for s in musk_core:
        print(f"     {s.code} {s.name} — {s.notes}")

    print(f"""
  💡 筛选逻辑回顾:
     ① 马斯克产业链(SpaceX/特斯拉/机器人) ✓
     ② 中美晚宴出席企业及关联 ✓
     ③ 优先政府/央企/军工订单 ✓
     ④ 排除资源化/纯市场化企业 ✓

  ⚠️ 注意:
     - SpaceX供应链存在信息不对称(公司不确认具体客户)
     - 部分公司对SpaceX供货占营收极低(如再升科技)
     - 机器人Optimus量产进度不确定
     - 需结合技术面(V3评分)和估值综合判断
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="中美会晤+马斯克产业链筛选")
    parser.add_argument("--data-root", default="data/store", help="数据目录")
    args = parser.parse_args()
    data_root = Path(args.data_root) if args.data_root else None
    run_screen(data_root=data_root)


if __name__ == "__main__":
    main()
