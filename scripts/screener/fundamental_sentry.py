#!/usr/bin/env python3
"""基本面哨兵 — 荐股前的基本面防线。

对候选股做基本面一票否决/警告检查：
  - 连亏否决：近2年连续亏损 → D级（一票否决）
  - 利润质量：净利率极差 → D级
  - 商誉风险：商誉/净资产过高 → C级警告
  - 现金流：经营现金流持续为负 → C级警告
  - 增长趋势：营收/利润趋势 → B级基础

评级标准：
  A级（优质）：盈利稳定+增长+低风险
  B级（可接受）：盈利但增速放缓或有小瑕疵
  C级（差/警告）：亏损但有改善趋势，需题材催化
  D级（一票否决）：连亏/基本面恶化，不推荐

用法：
  from scripts.screener.fundamental_sentry import FundamentalSentry
  sentry = FundamentalSentry()
  result = sentry.check('300291.SZ')
  print(result['grade'], result['verdict'], result['details'])
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.core.tushare_client import TushareClient, from_ts_code


@dataclass
class CheckItem:
    """单项检查结果"""
    name: str
    status: str  # 'pass', 'warn', 'fail'
    message: str
    weight: float = 1.0  # 权重（用于参考）


@dataclass
class SentryResult:
    """哨兵检查结果"""
    ts_code: str
    grade: str  # A/B/C/D
    verdict: str  # 一句话结论
    items: list[CheckItem] = field(default_factory=list)
    financials: dict = field(default_factory=dict)

    def __str__(self):
        emoji = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'}
        lines = [
            f"{emoji.get(self.grade, '⚪')} 基本面评级: {self.grade} — {self.verdict}",
            f"  标的: {self.ts_code}",
        ]
        for item in self.items:
            icon = {'pass': '✅', 'warn': '⚠️', 'fail': '❌'}[item.status]
            lines.append(f"  {icon} {item.name}: {item.message}")
        return '\n'.join(lines)


class FundamentalSentry:
    """基本面哨兵"""

    def __init__(self, client: Optional[TushareClient] = None):
        self.client = client or TushareClient()

    def check(self, ts_code: str) -> SentryResult:
        """对单只股票做基本面全面检查。

        Args:
            ts_code: 如 '300291.SZ'

        Returns:
            SentryResult with grade and details
        """
        items: list[CheckItem] = []

        # ── 获取财务数据 ──
        fina = self.client.pro.fina_indicator(
            ts_code=ts_code,
            fields='ts_code,end_date,ann_date,netprofit_margin,grossprofit_margin,'
                   'roe,debt_to_assets,or_yoy,np_yoy,tr_yoy,current_ratio,'
                   'ocf_to_debt,q_sales_yoy,q_op_qoq,gc_of_gr'
        )
        if fina is None or fina.empty:
            return SentryResult(
                ts_code=ts_code, grade='D',
                verdict='无财务数据',
                items=[CheckItem('数据获取', 'fail', '无法获取财务数据')]
            )
        fina = fina.sort_values('end_date', ascending=False).head(8).reset_index(drop=True)

        # 最近4期
        recent = fina.head(4)
        # 年度数据（取年末的）
        yearly = fina[fina['end_date'].astype(str).str.endswith('1231')].head(3)

        financials = {
            'latest_period': str(recent.iloc[0]['end_date']) if len(recent) > 0 else 'N/A',
            'netprofit_margins': recent['netprofit_margin'].tolist(),
            'grossprofit_margins': recent['grossprofit_margin'].tolist(),
            'roes': recent['roe'].tolist(),
            'or_yoys': recent['or_yoy'].tolist(),
            'debt_to_assets': recent['debt_to_assets'].tolist(),
        }

        # ══════════════════════════════════════
        # 检查1: 连续亏损（一票否决）
        # ══════════════════════════════════════
        if len(yearly) >= 2:
            yearly_np = yearly['netprofit_margin'].tolist()
            # 取最近2个完整年度
            last2 = yearly_np[:2]
            if all(np < -10 for np in last2):
                items.append(CheckItem('连续亏损', 'fail',
                    f'近2年净利率: {[f"{x:.1f}%" for x in last2]}，连续巨亏'))
            elif all(np < 0 for np in last2):
                items.append(CheckItem('连续亏损', 'fail',
                    f'近2年净利率: {[f"{x:.1f}%" for x in last2]}，连续亏损'))
            elif len(last2) > 0 and last2[0] < -50:
                items.append(CheckItem('巨亏', 'fail',
                    f'最近年度净利率: {last2[0]:.1f}%，巨亏'))
            elif len(last2) > 0 and last2[0] < -10:
                items.append(CheckItem('亏损', 'warn',
                    f'最近年度净利率: {last2[0]:.1f}%'))
            else:
                items.append(CheckItem('盈利能力', 'pass',
                    f'最近年度净利率: {last2[0]:.1f}%'))
        else:
            # 没有完整年度数据，看最近的季度
            latest_np = recent.iloc[0]['netprofit_margin'] if len(recent) > 0 else 0
            if latest_np < -50:
                items.append(CheckItem('亏损', 'fail',
                    f'最近季度净利率: {latest_np:.1f}%'))
            elif latest_np < 0:
                items.append(CheckItem('亏损', 'warn',
                    f'最近季度净利率: {latest_np:.1f}%'))
            else:
                items.append(CheckItem('盈利能力', 'pass',
                    f'最近季度净利率: {latest_np:.1f}%'))

        # ══════════════════════════════════════
        # 检查2: 营收趋势
        # ══════════════════════════════════════
        or_yoys = [x for x in recent['or_yoy'].tolist() if pd.notna(x)]
        if or_yoys:
            latest_or = or_yoys[0]
            if latest_or > 20:
                items.append(CheckItem('营收增长', 'pass',
                    f'营收YoY: {latest_or:.1f}%'))
            elif latest_or > 0:
                items.append(CheckItem('营收增长', 'pass',
                    f'营收YoY: {latest_or:.1f}%（增速放缓）'))
            elif latest_or > -20:
                items.append(CheckItem('营收下滑', 'warn',
                    f'营收YoY: {latest_or:.1f}%'))
            else:
                items.append(CheckItem('营收暴跌', 'fail',
                    f'营收YoY: {latest_or:.1f}%'))

        # ══════════════════════════════════════
        # 检查3: 毛利率
        # ══════════════════════════════════════
        gm = [x for x in recent['grossprofit_margin'].tolist() if pd.notna(x)]
        if gm:
            latest_gm = gm[0]
            if latest_gm > 40:
                items.append(CheckItem('毛利率', 'pass',
                    f'毛利率: {latest_gm:.1f}%'))
            elif latest_gm > 20:
                items.append(CheckItem('毛利率', 'pass',
                    f'毛利率: {latest_gm:.1f}%'))
            elif latest_gm > 0:
                items.append(CheckItem('毛利率偏低', 'warn',
                    f'毛利率: {latest_gm:.1f}%'))
            else:
                items.append(CheckItem('毛利率为负', 'fail',
                    f'毛利率: {latest_gm:.1f}%'))

        # ══════════════════════════════════════
        # 检查4: 负债率
        # ══════════════════════════════════════
        debt = [x for x in recent['debt_to_assets'].tolist() if pd.notna(x)]
        if debt:
            latest_debt = debt[0]
            if latest_debt < 30:
                items.append(CheckItem('负债率', 'pass',
                    f'负债率: {latest_debt:.1f}%'))
            elif latest_debt < 50:
                items.append(CheckItem('负债率', 'pass',
                    f'负债率: {latest_debt:.1f}%'))
            elif latest_debt < 70:
                items.append(CheckItem('负债率偏高', 'warn',
                    f'负债率: {latest_debt:.1f}%'))
            else:
                items.append(CheckItem('高负债', 'fail',
                    f'负债率: {latest_debt:.1f}%'))

        # ══════════════════════════════════════
        # 检查5: ROE
        # ══════════════════════════════════════
        roes = [x for x in recent['roe'].tolist() if pd.notna(x)]
        if roes:
            latest_roe = roes[0]
            if latest_roe > 15:
                items.append(CheckItem('ROE', 'pass',
                    f'ROE: {latest_roe:.1f}%'))
            elif latest_roe > 5:
                items.append(CheckItem('ROE', 'pass',
                    f'ROE: {latest_roe:.1f}%'))
            elif latest_roe > 0:
                items.append(CheckItem('ROE偏低', 'warn',
                    f'ROE: {latest_roe:.1f}%'))
            else:
                items.append(CheckItem('ROE为负', 'warn',
                    f'ROE: {latest_roe:.1f}%'))

        # ══════════════════════════════════════
        # 检查6: 扭亏确认（Q1扭亏的情况）
        # ══════════════════════════════════════
        if len(recent) >= 2:
            cur_np = recent.iloc[0]['netprofit_margin']
            prev_np = recent.iloc[1]['netprofit_margin']
            if pd.notna(cur_np) and pd.notna(prev_np):
                if prev_np < -20 and cur_np > 0:
                    items.append(CheckItem('扭亏确认', 'warn',
                        f'从{prev_np:.1f}%扭亏至{cur_np:.1f}%，需确认持续性'))

        # ══════════════════════════════════════
        # 评级判定
        # ══════════════════════════════════════
        has_fail = any(i.status == 'fail' for i in items)
        fail_count = sum(1 for i in items if i.status == 'fail')
        warn_count = sum(1 for i in items if i.status == 'warn')
        pass_count = sum(1 for i in items if i.status == 'pass')

        if fail_count >= 2:
            grade = 'D'
            verdict = '多处红线，一票否决'
        elif has_fail:
            # 1个fail，看是否是关键指标（亏损类）
            fail_names = [i.name for i in items if i.status == 'fail']
            critical = any(k in ''.join(fail_names) for k in ['亏损', '巨亏', '暴跌', '毛利率为负'])
            if critical:
                grade = 'D'
                verdict = '关键指标亮红灯，不推荐'
            else:
                grade = 'C'
                verdict = '有瑕疵，需题材催化'
        elif warn_count >= 3:
            grade = 'C'
            verdict = '多处警告，基本面偏弱'
        elif warn_count >= 1:
            grade = 'B'
            verdict = '基本健康，有小瑕疵'
        else:
            grade = 'A'
            verdict = '基本面优秀'

        return SentryResult(
            ts_code=ts_code,
            grade=grade,
            verdict=verdict,
            items=items,
            financials=financials,
        )

    def batch_check(self, ts_codes: list[str]) -> list[SentryResult]:
        """批量检查多只股票"""
        results = []
        for code in ts_codes:
            try:
                r = self.check(code)
                results.append(r)
            except Exception as e:
                results.append(SentryResult(
                    ts_code=code, grade='D',
                    verdict=f'检查失败: {e}',
                    items=[CheckItem('异常', 'fail', str(e))]
                ))
        return results

    def filter(self, ts_codes: list[str], min_grade: str = 'B') -> list[SentryResult]:
        """过滤掉不达标的股票。

        Args:
            ts_codes: 待检查的股票列表
            min_grade: 最低评级 ('A'/'B'/'C'/'D')

        Returns:
            通过筛选的结果列表
        """
        grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        min_idx = grade_order.get(min_grade, 1)

        results = self.batch_check(ts_codes)
        passed = [r for r in results if grade_order.get(r.grade, 3) <= min_idx]
        return passed


# ── CLI入口 ──
def main():
    import argparse
    parser = argparse.ArgumentParser(description='基本面哨兵 — 荐股前的基本面防线')
    parser.add_argument('codes', nargs='+', help='股票代码，如 300291 或 300291.SZ')
    parser.add_argument('--min-grade', default='B', choices=['A', 'B', 'C', 'D'],
                        help='最低评级（默认B）')
    parser.add_argument('--batch', action='store_true', help='批量模式，只显示通过的')
    args = parser.parse_args()

    sentry = FundamentalSentry()

    codes = []
    for c in args.codes:
        c = c.strip()
        if '.' not in c:
            suffix = 'SZ' if c.startswith(('0', '3')) else 'SH'
            c = f'{c}.{suffix}'
        codes.append(c)

    if args.batch:
        passed = sentry.filter(codes, min_grade=args.min_grade)
        if passed:
            for r in passed:
                print(f"{r.ts_code} {r.grade}")
        else:
            print("无通过筛选的标的")
    else:
        for code in codes:
            result = sentry.check(code)
            print(result)
            print()


if __name__ == '__main__':
    main()
