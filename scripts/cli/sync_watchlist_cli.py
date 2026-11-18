"""持仓数据同步工具：watchlist.json ↔ portfolio.xlsx/capital.xlsx

用法:
    python scripts/cli/sync_watchlist_cli.py --export   # xlsx → watchlist.json
    python scripts/cli/sync_watchlist_cli.py --import   # watchlist.json → xlsx
    python scripts/cli/sync_watchlist_cli.py --verify    # 对比两边，报差异
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
WATCHLIST_PATH = CONFIG_DIR / "watchlist.json"
PORTFOLIO_PATH = CONFIG_DIR / "portfolio.xlsx"
CAPITAL_PATH = CONFIG_DIR / "capital.xlsx"


def _load_watchlist() -> dict:
    if not WATCHLIST_PATH.exists():
        raise FileNotFoundError(f"watchlist.json not found: {WATCHLIST_PATH}")
    with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_watchlist(data: dict):
    data["updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_portfolio() -> pd.DataFrame:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(f"portfolio.xlsx not found: {PORTFOLIO_PATH}")
    df = pd.read_excel(PORTFOLIO_PATH)
    df["股票代码"] = df["股票代码"].astype(str)
    return df


def _save_portfolio(df: pd.DataFrame):
    df["股票代码"] = df["股票代码"].astype(str)
    df.to_excel(PORTFOLIO_PATH, index=False)


def do_export(updated_by: str = "sync-export"):
    """xlsx → watchlist.json：从 portfolio.xlsx 读取持仓，更新 watchlist.json"""
    wl = _load_watchlist()
    pf = _load_portfolio()

    # 建立 xlsx 索引
    xlsx_map = {}
    for _, row in pf.iterrows():
        code = row["股票代码"]
        xlsx_map[code] = {
            "name": row["股票名称"],
            "shares": int(row["持仓数量"]),
            "cost": float(row["买入价格"]),
            "stop_loss": float(row["止损价"]) if pd.notna(row["止损价"]) else None,
            "take_profit": float(row["止盈价"]) if pd.notna(row["止盈价"]) else None,
            "type": "etf" if "ETF" in str(row["股票名称"]) else "stock",
            "note": str(row.get("备注", "")),
        }

    # 更新 holdings
    new_holdings = []
    for code, info in xlsx_map.items():
        new_holdings.append({
            "code": code,
            "name": info["name"],
            "type": info["type"],
            "shares": info["shares"],
            "cost": info["cost"],
            "stop_loss": info["stop_loss"],
            "take_profit": info["take_profit"],
            "note": info.get("note", ""),
        })

    wl["version"] = 2
    wl["updated_by"] = updated_by
    wl["holdings"] = new_holdings
    _save_watchlist(wl)

    print(f"[sync] export 成功: {len(new_holdings)} 只持仓 → watchlist.json")
    return wl


def do_import(updated_by: str = "sync-import"):
    """watchlist.json → xlsx：从 watchlist.json 读取持仓，更新 portfolio.xlsx"""
    wl = _load_watchlist()
    holdings = wl.get("holdings", [])

    if not holdings:
        print("[sync] watchlist.json 无持仓数据，跳过")
        return

    # 读取现有 xlsx
    pf = _load_portfolio()
    pf_codes = set(pf["股票代码"].tolist())

    wl_map = {h["code"]: h for h in holdings}

    # 更新已有行
    for idx, row in pf.iterrows():
        code = row["股票代码"]
        if code in wl_map:
            h = wl_map[code]
            pf.at[idx, "持仓数量"] = h.get("shares", row["持仓数量"])
            pf.at[idx, "买入价格"] = h.get("cost", row["买入价格"])
            if h.get("stop_loss") is not None:
                pf.at[idx, "止损价"] = h["stop_loss"]
            if h.get("take_profit") is not None:
                pf.at[idx, "止盈价"] = h["take_profit"]

    # 新增行（watchlist 有但 xlsx 没有）
    for code, h in wl_map.items():
        if code not in pf_codes:
            new_row = {
                "股票代码": code,
                "股票名称": h.get("name", code),
                "买入日期": datetime.now().strftime("%Y-%m-%d"),
                "买入价格": h.get("cost", 0),
                "持仓数量": h.get("shares", 0),
                "止损价": h.get("stop_loss"),
                "止盈价": h.get("take_profit"),
                "仓位%": 0,
                "持仓类型": "卫星持仓",
                "备注": h.get("note", ""),
            }
            pf = pd.concat([pf, pd.DataFrame([new_row])], ignore_index=True)

    _save_portfolio(pf)

    # 标记 watchlist 更新
    wl["updated_by"] = updated_by
    _save_watchlist(wl)

    print(f"[sync] import 成功: {len(holdings)} 只持仓 → portfolio.xlsx")


def do_verify():
    """对比 watchlist.json 和 portfolio.xlsx，报差异"""
    wl = _load_watchlist()
    pf = _load_portfolio()

    wl_map = {h["code"]: h for h in wl.get("holdings", [])}
    pf_map = {}
    for _, row in pf.iterrows():
        pf_map[row["股票代码"]] = {
            "name": row["股票名称"],
            "shares": int(row["持仓数量"]),
            "cost": float(row["买入价格"]),
            "stop_loss": float(row["止损价"]) if pd.notna(row["止损价"]) else None,
            "take_profit": float(row["止盈价"]) if pd.notna(row["止盈价"]) else None,
        }

    wl_codes = set(wl_map.keys())
    pf_codes = set(pf_map.keys())

    only_wl = wl_codes - pf_codes
    only_pf = pf_codes - wl_codes
    common = wl_codes & pf_codes

    diffs = []
    for code in common:
        w, p = wl_map[code], pf_map[code]
        if w["shares"] != p["shares"]:
            diffs.append(f"  {code} {w['name']}: shares watchlist={w['shares']} xlsx={p['shares']}")
        if abs(w["cost"] - p["cost"]) > 0.001:
            diffs.append(f"  {code} {w['name']}: cost watchlist={w['cost']:.3f} xlsx={p['cost']:.3f}")
        w_sl = w.get("stop_loss")
        p_sl = p.get("stop_loss")
        if w_sl != p_sl and not (w_sl is None and p_sl is None):
            diffs.append(f"  {code} {w['name']}: stop_loss watchlist={w_sl} xlsx={p_sl}")

    if only_wl:
        print(f"[sync] 仅在 watchlist.json: {only_wl}")
    if only_pf:
        print(f"[sync] 仅在 portfolio.xlsx: {only_pf}")
    if diffs:
        print(f"[sync] 数据差异 ({len(diffs)} 项):")
        for d in diffs:
            print(d)
    if not only_wl and not only_pf and not diffs:
        print("[sync] ✓ watchlist.json 和 portfolio.xlsx 完全一致")


def main():
    parser = argparse.ArgumentParser(description="持仓数据同步: watchlist.json ↔ portfolio.xlsx")
    parser.add_argument("--export", action="store_true", help="xlsx → watchlist.json")
    parser.add_argument("--import", action="store_true", help="watchlist.json → xlsx")
    parser.add_argument("--verify", action="store_true", help="对比两边，报差异")
    parser.add_argument("--updated-by", default=None, help="标记更新者")
    args = parser.parse_args()

    count = sum([args.export, args.__dict__["import"], args.verify])
    if count == 0:
        parser.print_help()
        return
    if count > 1:
        print("错误: 只能指定一个操作 (--export / --import / --verify)")
        return

    if args.export:
        do_export(updated_by=args.updated_by or "sync-export")
    elif args.__dict__["import"]:
        do_import(updated_by=args.updated_by or "sync-import")
    elif args.verify:
        do_verify()


if __name__ == "__main__":
    main()
