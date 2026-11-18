#!/usr/bin/env python3
"""财务数据工具 — 归母口径唯一化(2026-09-10,国机精工-58%/-72%口径事故的工程修复).

铁律⑦的代码形态: 对外引用的净利一律用归母(n_income_attr/n_income),列名缺错误必须显式失败,
禁止静默换口径。所有深研/分析脚本从此import本模块,不再裸调pro.income。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def attr_net_profit(tushare_pro, ts_code, start='20240630'):
    """返回DataFrame[end_date, rev亿, 归母净利亿, 归母同比%]. 缺归母列→显式报错不降级."""
    inc = tushare_pro.income(ts_code=ts_code, start_date=start,
                             fields='end_date,revenue,n_income,n_income_attr_p')
    if 'n_income_attr_p' not in inc.columns:
        raise ValueError(f'{ts_code}: 接口未返回n_income_attr,禁止降级用总净利——正确列名是n_income_attr_p(注意_p后缀)')
    inc = inc.drop_duplicates('end_date').sort_values('end_date')
    out = inc[['end_date', 'revenue', 'n_income_attr_p']].copy()
    out['rev亿'] = (out['revenue'] / 1e8).round(2)
    out['归母亿'] = (out['n_income_attr_p'] / 1e8).round(3)
    out['归母同比%'] = (out['归母亿'].pct_change(4) * 100).round(1)
    return out.drop(columns=['revenue', 'n_income_attr_p']).reset_index(drop=True)


if __name__ == '__main__':
    from scripts.core.tushare_client import TushareClient
    code = sys.argv[1] if len(sys.argv) > 1 else '002046.SZ'
    c = TushareClient()
    print(attr_net_profit(c.pro, code).to_string(index=False))
