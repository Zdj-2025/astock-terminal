#!/usr/bin/env python3
"""R0-记账闭环 ①: 清洗 trade_log.csv → 一本可被程序正确读回、逐行可审计的干净账。

背景(2026-09-10, 见 docs/stock项目全面诊断_2026-09-09.md): 原 config/trade_log.csv
78 行里 ~24 行(31%)结构错位。根因是"写入侧不引号 note + realized_pnl 为空时整列被省略",
与 csv 解析叠加后产生多种畸形(用前 7 列 date..amount 恒干净这一事实统一恢复):
  1. nf=9 : realized 空被省略 → note 左移。             [tail = note, source]
  2. nf=10: 空 realized 省略 + note 含逗号未引号 →      [tail = note片段, 剩余note, source]
            note 恰好被 csv 切成两段, 首段落进 realized 槽(如 8/6 中航高科, 最隐蔽)。
  3. nf=11+: 同 2 但 note 被切成 >2 段, realized 空则首 token 为 ''; 有值则首 token 为数字。
  4. 特例: 取现事件(空 code/name=取现)、BUY 行 shares/price 颠倒(9/9 手动录入)。
后果: realized_pnl 列"时而数字、时而备注、时而时间戳", 下游 PnL 汇总/对账不可信。

本脚本只做字段重排, 绝不臆造数值: realized 槽遇数字才认作已实现盈亏, 遇文本一律判为 note
并置空。还原的 note 保持原样。修复逐行落 repair 列注明依据; 原文件不动。
输出: config/trade_log_clean.csv (原 10 列 + repair 列)。
用法: cd stock && PYTHONPATH=. python scripts/accounting/clean_trade_log.py
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
RAW = PROJECT / 'config/trade_log.csv'
OUT = PROJECT / 'config/trade_log_clean.csv'

HDR = ['date', 'code', 'name', 'action', 'price', 'shares', 'amount',
       'realized_pnl', 'note', 'source', 'repair']
REALIZED_RE = re.compile(r'^[-+]?\d+(\.\d+)?$')


def looks_numeric(s: str) -> bool:
    return bool(REALIZED_RE.match(s.strip()))


def fix_row(n: int, r: list[str]) -> dict:
    """重排一条行成标准 schema。返回 repair 说明(空=原样干净)。"""
    nf = len(r)

    # 特例 A: 取现/非交易事件(空 code, name=取现, 现金额在数值槽)
    if not r[1] and r[2] == '取现':
        amt = next((t for t in (r[5], r[6]) if looks_numeric(t)), '')
        amt = str(int(float(amt))) if amt else ''
        return {'date': r[0], 'code': '', 'name': '取现', 'action': '取现',
                'price': '', 'shares': '', 'amount': amt, 'realized_pnl': '',
                'note': r[7] if nf >= 8 else '', 'source': r[-1],
                'repair': f'现金事件: code空/name=取现, 现金额 {amt} 从数值槽归位到 amount'}

    # 特例 B: BUY 行 shares/price 颠倒(9/9 1000ETF, 含 ~ 与均价推算)
    if r[3].strip().upper() == 'BUY':
        price, shares = r[5], r[4].replace('~', '')
        try:
            amount = str(round(float(price) * float(shares)))
        except ValueError:
            amount = r[6] if len(r) > 6 else ''
        return {'date': r[0], 'code': r[1], 'name': r[2], 'action': '买',
                'price': price, 'shares': shares, 'amount': amount,
                'realized_pnl': '', 'note': r[6] if nf > 6 else '',
                'source': r[-1],
                'repair': f'BUY行: shares/price列颠倒, amount=price×shares={amount} '
                          f'(均价{price}为推算口径, 低置信见source)'}

    # 统一恢复: 前 7 列(date..amount)恒干净, 尾段 = [realized?, note碎片..., source]
    lead, tail = r[:7], r[7:]
    realized = ''
    if tail and looks_numeric(tail[0]):
        realized = tail[0].strip()          # 首 token 是数字 → 真正的已实现盈亏
        note_parts = tail[1:-1]
        what = 'realized有值'
    else:
        start = 1 if (tail and tail[0] == '') else 0   # 空realized被省略时: 首token空则跳
        note_parts = tail[start:-1]
        what = 'realized空(被省略)'
    note = ','.join(note_parts)
    source = tail[-1] if tail else ''

    repair = ''
    if nf != 10:
        repair = f'字段重排: 原{nf}列→10列; {what}, note重拼 {len(note_parts)} 段'
    elif tail and tail[0] and not looks_numeric(tail[0]):
        # nf==10 但 realized 槽被 note 碎片污染(如 8/6 中航高科): 空realized被省略 + note含未引号逗号
        repair = f'note含未引号逗号被拆断且首段落入realized槽, 已还原为note({len(note_parts)}段)'
    else:
        repair = ''  # 正常行: 物理上带空realized列 或 realized为真数字 → 原样干净

    return {'date': lead[0], 'code': lead[1], 'name': lead[2], 'action': lead[3],
            'price': lead[4], 'shares': lead[5], 'amount': lead[6],
            'realized_pnl': realized, 'note': note, 'source': source,
            'repair': repair}


def main():
    if not RAW.exists():
        print(f'❌ {RAW} 不存在')
        sys.exit(1)
    with open(RAW, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    assert hdr[:10] == HDR[:10], f'表头异常: {hdr}'

    out = [fix_row(i + 2, r) for i, r in enumerate(body)]

    # 自检: 恰好 10 个业务字段; realized 只允许 数字/空
    bad = []
    for row in out:
        data_keys = [k for k in row if k != 'repair']
        if len(data_keys) != 10:
            bad.append(f"{row['date']}: 业务列数{len(data_keys)}≠10")
        rp = row['realized_pnl']
        if rp and not looks_numeric(rp):
            bad.append(f"{row['date']} {row['name']}: realized_pnl非数字 '{rp}'")
    if bad:
        print('❌ 修复后仍存在异常:')
        [print('   ', b) for b in bad]
        sys.exit(2)

    n_repair = sum(1 for r in out if r['repair'])
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=HDR)
        w.writeheader()
        w.writerows(out)

    # 汇总
    print(f'✅ 清洗完成: {len(out)} 行(修复 {n_repair} 行, 其余原样干净) → {OUT.name}')
    sells = [r for r in out if r['action'] == '卖']
    realized_ok = [r for r in sells if r['realized_pnl']]
    total = sum(float(r['realized_pnl']) for r in realized_ok)
    print(f'   卖单 {len(sells)} 条, realized_pnl 有数值 {len(realized_ok)} 条, 直接相加 {total:+.0f} 元')
    print('   (多条卖单 realized 原为空 = 未记录, 此和数偏低, 只用于对账不做净值)')

    print('\n   按代码 已记录 realized_pnl 明细(供对账):')
    for code in sorted({r['code'] for r in realized_ok if r['code']}):
        sub = [r for r in realized_ok if r['code'] == code]
        s = sum(float(r['realized_pnl']) for r in sub)
        print(f'     {code}: {s:+.0f} 元  ({" + ".join(r["realized_pnl"] for r in sub)})')


if __name__ == '__main__':
    main()
