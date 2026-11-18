#!/usr/bin/env python3
"""研报一致预期模块(2026-08-24建,DeepSeek借鉴:补我们研报层空白)。

数据源: 东方财富免费研报API reportapi.eastmoney.com(与公告监控同源)
  GET /report/list?code=xxx&qType=0&beginTime&endTime&pageSize=50
字段: sRatingName评级 / predictThisYearEps今年EPS预测 / predictNextYearEps / indvAimPrice目标价

⚠️使用纪律(固定打七折):
  - 卖方几乎不发"卖出"评级,全买入=噪音不是信号
  - 目标价锚定旧预期,仅当参考坐标
  - 盈利预测均值可作预期锚,与我们扣非实算交叉
  - 覆盖缺口: 部分票东财无研报索引(如北方国际)≠市场无覆盖,新浪可能有→0篇时提示换源人工核

用法:
  python scripts/analyzers/report_consensus.py --code 600519
  python scripts/analyzers/report_consensus.py --codes 000065,601998
  from scripts.analyzers.report_consensus import get_consensus
"""
import json
import urllib.request
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def get_consensus(code, months=6, verbose=True):
    """拉取近N月研报,返回汇总dict。"""
    code = str(code).zfill(6)
    end = datetime.now().strftime('%Y-%m-%d')
    begin = (datetime.now() - timedelta(days=months * 31)).strftime('%Y-%m-%d')
    url = (f"https://reportapi.eastmoney.com/report/list?industryCode=*&pageSize=50&industry=*"
           f"&rating=*&ratingChange=*&beginTime={begin}&endTime={end}&pageNo=1&fields="
           f"&qType=0&orgCode=&code={code}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode())
    except Exception as e:
        return {'code': code, 'error': str(e)[:50]}
    reports = d.get('data', []) or []
    if not reports:
        if verbose:
            print(f"\n📌 {code} 东财研报库近{months}月0篇")
            print("   ⚠️覆盖缺口≠无机构关注(北方国际案例:东财0篇但新浪有7家)→需人工换源核")
        return {'code': code, 'n': 0,
                'note': '东财研报库0篇——覆盖缺口≠无机构关注(新浪/华西等或有,人工核)'}

    # 评级分布
    from collections import Counter
    ratings = Counter(str(x.get('sRatingName') or '').strip() for x in reports if x.get('sRatingName'))
    # 今年EPS预测均值(有值的)
    eps = [float(x['predictThisYearEps']) for x in reports
           if x.get('predictThisYearEps') not in (None, '', '-')]
    eps_next = [float(x['predictNextYearEps']) for x in reports
                if x.get('predictNextYearEps') not in (None, '', '-')]
    # 目标价(如有)
    aims = [float(x['indvAimPriceT']) for x in reports
            if x.get('indvAimPriceT') not in (None, '', '-')]
    res = {'code': code, 'n': len(reports),
           'ratings': dict(ratings),
           'eps_this_avg': sum(eps) / len(eps) if eps else None,
           'eps_next_avg': sum(eps_next) / len(eps_next) if eps_next else None,
           'target_price': sum(aims) / len(aims) if aims else None,
           'latest': {'org': reports[0].get('orgSName'), 'date': str(reports[0].get('publishDate'))[:10],
                      'title': str(reports[0].get('title'))[:40], 'rating': reports[0].get('sRatingName')}}
    if verbose:
        print(f"\n📌 {code} 研报一致预期(近{months}月 {len(reports)}篇) ⚠️卖方偏乐观,打七折用")
        rt = ' / '.join(f"{k}{v}篇" for k, v in ratings.most_common())
        print(f"   评级: {rt}")
        if res['eps_this_avg'] is not None:
            print(f"   EPS预测均值: 今年{res['eps_this_avg']:.2f}" +
                  (f" | 明年{res['eps_next_avg']:.2f}" if res['eps_next_avg'] is not None else ''))
        if res['target_price']:
            print(f"   目标价均值: {res['target_price']:.2f}(锚定旧预期,仅作坐标)")
        print(f"   最新: [{res['latest']['date']}] {res['latest']['org']} {res['latest']['rating']} 《{res['latest']['title']}》")
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='研报一致预期(东财API,打七折用)')
    parser.add_argument('--code')
    parser.add_argument('--codes')
    parser.add_argument('--months', type=int, default=6)
    args = parser.parse_args()
    print("=== 研报一致预期 | 东方财富研报库 ===")
    print("   纪律: 卖方不发卖出评级,全买入=噪音; 目标价只当坐标; EPS预测与我们扣非实算交叉\n")
    if args.code:
        get_consensus(args.code, args.months)
    elif args.codes:
        for c in args.codes.split(','):
            get_consensus(c.strip(), args.months)
    else:
        # 默认演示: 持仓股几只
        for c in ['603881', '000100', '601998']:
            get_consensus(c, args.months)
