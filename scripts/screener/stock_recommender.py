#!/usr/bin/env python3
"""荐股整合器 — 三道防线 + 标准化推荐卡片。

流程：
  第一道：daily_scan 策略引擎评分 ≥ min_score（量化初筛）
  第二道：fundamental_sentry 基本面 ≥ min_grade（基本面过滤）
  第三道：技术面确认（趋势/背离/波浪） — 可选，由 --deep 开启
  输出：标准化荐股卡片（含仓位/止损/目标/风险提示）

用法：
  PYTHONPATH=. python scripts/screener/stock_recommender.py
  PYTHONPATH=. python scripts/screener/stock_recommender.py --min-score 7.0
  PYTHONPATH=. python scripts/screener/stock_recommender.py --min-grade B
  PYTHONPATH=. python scripts/screener/stock_recommender.py --top 10
  PYTHONPATH=. python scripts/screener/stock_recommender.py --deep  # 含技术面深度分析
  PYTHONPATH=. python scripts/screener/stock_recommender.py --codes 300291,603881  # 指定股票检查
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
from scripts.screener.strategy_engine import StrategyEngine
from scripts.screener.fundamental_sentry import FundamentalSentry
from scripts.core.tushare_client import TushareClient, from_ts_code


def to_ts_code(code: str) -> str:
    """纯数字 → Tushare格式: '603881' → '603881.SH'"""
    code = code.split('.')[0]  # 已经带后缀也没事
    suffix = 'SZ' if code.startswith(('0', '3')) else 'SH'
    return f'{code}.{suffix}'


# ── 名称映射 ──
def _load_name_map() -> dict:
    """从tushare获取股票名称映射"""
    try:
        client = TushareClient()
        basic = client.pro.stock_basic(
            exchange='', list_status='L',
            fields='ts_code,name'
        )
        return dict(zip(basic['ts_code'], basic['name']))
    except Exception:
        return {}


def _get_name(ts_code: str, name_map: dict) -> str:
    return name_map.get(ts_code, ts_code.split('.')[0])


# ── 荐股卡片 ──
@dataclass
class StockCard:
    """标准化荐股卡片"""
    ts_code: str
    name: str
    strategy_score: float
    strategy_type: str  # '趋势加速' / '底部反转'
    fund_grade: str  # A/B/C/D
    fund_verdict: str
    signal_summary: str  # 量化信号摘要
    risks: list[str]
    position_pct: float  # 建议仓位(%)
    stop_loss: float  # 止损价
    target: Optional[float] = None  # 目标价
    holding_days: str = '5-10个交易日'
    details: dict = None

    def __str__(self):
        emoji = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'}
        lines = [
            '━' * 40,
            f'📋 荐股卡片',
            f'  标的: {self.name} {self.ts_code.split(".")[0]}',
            f'  策略分: {self.strategy_score:.1f} | 基本面: {emoji.get(self.fund_grade, "?")}{self.fund_grade} | {self.strategy_type}',
            '',
            f'📊 推荐理由:',
            f'  {self.signal_summary}',
            '',
            f'⚠️ 风险提示:',
        ]
        for r in self.risks:
            lines.append(f'  • {r}')
        lines += [
            '',
            f'💰 仓位建议:',
            f'  • 首仓: 总资金的{self.position_pct:.0f}%',
            f'  • 止损: {self.stop_loss:.2f}以下',
        ]
        if self.target:
            lines.append(f'  • 目标: {self.target:.2f}')
        lines.append(f'  • 持有期限: {self.holding_days}')
        lines.append('━' * 40)
        return '\n'.join(lines)


class StockRecommender:
    """荐股整合器"""

    def __init__(self, data_root: str = 'data/store'):
        self.data_root = Path(data_root)
        self.scorer = V3DerivativeScorer(data_root=self.data_root)
        self.sentry = FundamentalSentry()
        self.name_map = {}

    def _ensure_names(self):
        if not self.name_map:
            self.name_map = _load_name_map()

    def scan(
        self,
        min_score: float = 7.0,
        min_grade: str = 'B',
        top_n: int = 10,
        strategy_filter: Optional[str] = None,
    ) -> list[StockCard]:
        """全市场扫描 → 三道防线 → 输出推荐卡片。

        Args:
            min_score: 最低策略分
            min_grade: 最低基本面评级
            top_n: 每类最多输出N只
            strategy_filter: 'reversal' / 'accel' / None(全部)

        Returns:
            通过所有防线的StockCard列表
        """
        self._ensure_names()
        grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        min_grade_idx = grade_order.get(min_grade, 1)

        # ══ 第一道：量化初筛 ══
        store = self.data_root
        codes = sorted(
            [d.name for d in (store / 'kline').iterdir()
             if d.is_dir() and (d / 'D.parquet').exists()]
        )

        accel, reversal, top_sell = self.scorer.score_dual_branch(codes, min_score=min_score)

        # 收集候选
        candidates = []
        engine = StrategyEngine(scorer=self.scorer)

        # 底部反转候选 — 复用daily_scan的策略评分逻辑
        if strategy_filter in (None, 'reversal') and len(reversal) > 0:
            from scripts.screener.v3_derivative_scorer import V3Score
            for _, row in reversal.iterrows():
                code = row['code']
                ts_code = to_ts_code(code)
                # 重建V3Score用于策略评分
                try:
                    v3s = V3Score(
                        f1=row.get('f1', 0), f2=row.get('f2', 0),
                        details={
                            'dh180': row.get('dh180', 0),
                            'vol_ratio': row.get('vol_ratio', 1.0),
                            'minute_details': row.get('minute_details', {}),
                        }
                    )
                    strat_score = engine.score_bottom_reversal(v3s).total
                except Exception:
                    strat_score = row.get('v3_raw', 0)

                if strat_score >= min_score:
                    row_dict = row.to_dict()
                    row_dict['strategy_score'] = strat_score
                    row_dict['close'] = row.get('price', 0)
                    candidates.append({
                        'ts_code': ts_code,
                        'strategy_score': strat_score,
                        'strategy_type': '底部反转',
                        'raw': row.get('v3_raw', 0),
                        'row': row_dict,
                    })

        # 趋势加速候选
        if strategy_filter in (None, 'accel'):
            for _, row in accel.head(top_n).iterrows():
                code = row['code']
                ts_code = to_ts_code(code)
                raw = row.get('raw', row.get('v3_raw', 0))
                candidates.append({
                        'ts_code': ts_code,
                        'strategy_score': raw,
                        'strategy_type': '趋势加速',
                        'raw': raw,
                        'row': row.to_dict() if hasattr(row, 'to_dict') else {},
                    })

        if not candidates:
            print('⚠️ 量化初筛无候选')
            return []

        print(f'🎯 量化初筛通过: {len(candidates)}只')

        # ══ 第二道：基本面过滤 ══
        ts_codes = [c['ts_code'] for c in candidates]
        fund_results = self.sentry.batch_check(ts_codes)
        fund_map = {r.ts_code: r for r in fund_results}

        passed = []
        filtered_count = 0
        for cand in candidates:
            ts = cand['ts_code']
            fr = fund_map.get(ts)
            if fr and grade_order.get(fr.grade, 3) <= min_grade_idx:
                cand['fund_result'] = fr
                passed.append(cand)
            else:
                filtered_count += 1
                # 显示被过滤的
                name = _get_name(ts, self.name_map)
                grade = fr.grade if fr else '?'
                verdict = fr.verdict if fr else '无数据'
                print(f'  ❌ {name} {ts.split(".")[0]} 基本面{grade} — {verdict}')

        if filtered_count:
            print(f'🛡️ 基本面过滤: 淘汰{filtered_count}只')
        print(f'✅ 通过基本面: {len(passed)}只')

        # ══ 生成推荐卡片 ══
        cards = []
        for cand in sorted(passed, key=lambda x: x['strategy_score'], reverse=True)[:top_n]:
            ts = cand['ts_code']
            fr = cand['fund_result']
            row = cand['row']
            name = _get_name(ts, self.name_map)

            # 信号摘要
            signals = []
            if cand['strategy_type'] == '底部反转':
                dh180 = row.get('dh180_0', row.get('dh180', 0))
                vol = row.get('vol_ratio', row.get('vol', 0))
                signals.append(f'策略分{cand["strategy_score"]:.1f}')
                if pd.notna(dh180):
                    signals.append(f'20日跌幅{dh180:.1f}%')
                if pd.notna(vol):
                    signals.append(f'放量{vol:.1f}倍')
            else:
                f1 = row.get('f1', row.get('f_prime', 0))
                ret20 = row.get('ret20_0', row.get('ret20', 0))
                signals.append(f'策略分{cand["strategy_score"]:.1f}')
                if pd.notna(f1):
                    signals.append(f"f'={f1:.3f}")
                if pd.notna(ret20):
                    signals.append(f'20日涨跌{ret20:+.1f}%')

            # 风险提示
            risks = []
            for item in fr.items:
                if item.status in ('warn', 'fail'):
                    risks.append(f'{item.name}: {item.message}')
            # 加策略级风险
            if cand['strategy_type'] == '底部反转':
                risks.append('底部反转可能需要多次确认，注意控制仓位')
            if fr.grade in ('B', 'C'):
                risks.append('基本面有瑕疵，需关注题材催化')

            # 仓位建议
            if fr.grade == 'A':
                pos = 10
            elif fr.grade == 'B':
                pos = 7
            else:
                pos = 5

            # 止损价（用策略引擎的ATR止损）
            stop = row.get('stop_loss', row.get('stop', 0))
            if not stop or stop <= 0:
                # 简单估算：当前价-5%
                close = row.get('close', row.get('latest_close', 0))
                stop = close * 0.95 if close > 0 else 0

            card = StockCard(
                ts_code=ts,
                name=name,
                strategy_score=cand['strategy_score'],
                strategy_type=cand['strategy_type'],
                fund_grade=fr.grade,
                fund_verdict=fr.verdict,
                signal_summary=' | '.join(signals),
                risks=risks if risks else ['无重大风险'],
                position_pct=pos,
                stop_loss=stop,
                details={'fund_items': fr.items, 'row': row.to_dict() if hasattr(row, 'to_dict') else {}},
            )
            cards.append(card)

        return cards

    def check_specific(
        self,
        codes: list[str],
        min_grade: str = 'B',
    ) -> list[StockCard]:
        """对指定股票做三道防线检查。

        Args:
            codes: 股票代码列表（6位数字）
            min_grade: 最低基本面评级
        """
        self._ensure_names()

        ts_codes = []
        for c in codes:
            c = c.strip()
            if '.' not in c:
                suffix = 'SZ' if c.startswith(('0', '3')) else 'SH'
                ts_codes.append(f'{c}.{suffix}')
            else:
                ts_codes.append(c)

        # 量化评分
        results = []
        for ts in ts_codes:
            code = ts.split('.')[0]
            try:
                scorer = V3DerivativeScorer(data_root=Path('data/store'))
                engine = StrategyEngine(scorer=scorer)
                # 尝试底部反转评分
                try:
                    strat_r = engine.score(code, 'reversal')
                    if strat_r.total_score >= 5:
                        results.append({
                            'ts_code': ts, 'code': code,
                            'strategy_score': strat_r.total_score,
                            'strategy_type': '底部反转',
                        })
                except Exception:
                    pass
                # 尝试趋势加速评分
                try:
                    strat_a = engine.score(code, 'accel')
                    if strat_a.total_score >= 5:
                        results.append({
                            'ts_code': ts, 'code': code,
                            'strategy_score': strat_a.total_score,
                            'strategy_type': '趋势加速',
                        })
                except Exception:
                    pass
                # 如果都没通过，给个基本信息
                if not any(r['ts_code'] == ts for r in results):
                    results.append({
                        'ts_code': ts, 'code': code,
                        'strategy_score': 0,
                        'strategy_type': '未达标',
                    })
            except Exception as e:
                results.append({
                    'ts_code': ts, 'code': code,
                    'strategy_score': 0,
                    'strategy_type': f'评分失败: {e}',
                })

        # 基本面检查
        fund_results = self.sentry.batch_check(ts_codes)
        fund_map = {r.ts_code: r for r in fund_results}

        cards = []
        for r in results:
            ts = r['ts_code']
            fr = fund_map.get(ts)
            name = _get_name(ts, self.name_map)

            card = StockCard(
                ts_code=ts,
                name=name,
                strategy_score=r['strategy_score'],
                strategy_type=r['strategy_type'],
                fund_grade=fr.grade if fr else 'D',
                fund_verdict=fr.verdict if fr else '无数据',
                signal_summary=f"量化分: {r['strategy_score']:.1f}",
                risks=[f'{i.name}: {i.message}' for i in fr.items if i.status in ('warn', 'fail')] if fr else ['无数据'],
                position_pct=0 if r['strategy_score'] < 5 else 5,
                stop_loss=0,
            )
            cards.append(card)

        return cards


def main():
    parser = argparse.ArgumentParser(
        description='荐股整合器 — 三道防线 + 标准化推荐卡片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--min-score', type=float, default=7.0, help='最低策略分（默认7.0）')
    parser.add_argument('--min-grade', default='B', choices=['A', 'B', 'C', 'D'], help='最低基本面评级（默认B）')
    parser.add_argument('--top', type=int, default=10, help='每类最多输出N只（默认10）')
    parser.add_argument('--strategy', choices=['reversal', 'accel', 'all'], default='all', help='策略类型')
    parser.add_argument('--codes', type=str, help='指定股票检查，逗号分隔 如 300291,603881')
    parser.add_argument('--data-root', default='data/store', help='数据目录')
    args = parser.parse_args()

    recommender = StockRecommender(data_root=args.data_root)

    if args.codes:
        codes = [c.strip() for c in args.codes.split(',')]
        # 确保是ts_code格式
        ts_codes_input = []
        for c in codes:
            if '.' not in c:
                suffix = 'SZ' if c.startswith(('0', '3')) else 'SH'
                ts_codes_input.append(f'{c}.{suffix}')
            else:
                ts_codes_input.append(c)
        cards = recommender.check_specific(ts_codes_input, min_grade=args.min_grade)
    else:
        strategy = None if args.strategy == 'all' else args.strategy
        cards = recommender.scan(
            min_score=args.min_score,
            min_grade=args.min_grade,
            top_n=args.top,
            strategy_filter=strategy,
        )

    if not cards:
        print('\n📭 今日无符合标准的推荐标的')
        return

    print(f'\n📬 今日推荐: {len(cards)}只\n')
    for card in cards:
        print(card)


if __name__ == '__main__':
    main()
