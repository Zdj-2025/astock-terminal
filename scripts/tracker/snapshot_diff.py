"""Snapshot comparison: detect position changes between two snapshots.

Usage:
    from .snapshot_diff import diff_snapshots
    changes = diff_snapshots(prev_holdings, curr_holdings)
"""
from typing import Optional


def diff_snapshots(prev: list[dict], curr: list[dict]) -> dict:
    """Compare two holdings snapshots and detect changes.

    Args:
        prev: Previous snapshot holdings (list of dicts with stock_code, shares, etc.).
        curr: Current snapshot holdings.

    Returns:
        Dict with keys:
        - new_positions: Holdings in curr but not in prev.
        - closed: Holdings in prev but not in curr.
        - increased: Holdings with more shares in curr.
        - decreased: Holdings with fewer shares in curr.
        - unchanged: Holdings with same shares.
    """
    prev_map = {h["stock_code"]: h for h in prev}
    curr_map = {h["stock_code"]: h for h in curr}

    prev_codes = set(prev_map.keys())
    curr_codes = set(curr_map.keys())

    new_codes = curr_codes - prev_codes
    closed_codes = prev_codes - curr_codes
    common_codes = curr_codes & prev_codes

    new_positions = []
    for code in new_codes:
        h = dict(curr_map[code])
        h["change_type"] = "new"
        new_positions.append(h)

    closed = []
    for code in closed_codes:
        h = dict(prev_map[code])
        h["change_type"] = "closed"
        closed.append(h)

    increased = []
    decreased = []
    unchanged = []

    for code in common_codes:
        p = prev_map[code]
        c = curr_map[code]
        diff_shares = c["shares"] - p["shares"]

        entry = dict(c)
        entry["prev_shares"] = p["shares"]
        entry["diff_shares"] = diff_shares

        if diff_shares > 0:
            entry["change_type"] = "increased"
            increased.append(entry)
        elif diff_shares < 0:
            entry["change_type"] = "decreased"
            decreased.append(entry)
        else:
            entry["change_type"] = "unchanged"
            # Check if cost changed
            if p.get("cost_price") != c.get("cost_price"):
                entry["prev_cost_price"] = p.get("cost_price")
                entry["cost_changed"] = True
            unchanged.append(entry)

    return {
        "new_positions": new_positions,
        "closed": closed,
        "increased": increased,
        "decreased": decreased,
        "unchanged": unchanged,
        "summary": {
            "new": len(new_positions),
            "closed": len(closed),
            "increased": len(increased),
            "decreased": len(decreased),
            "unchanged": len(unchanged),
            "total_changes": len(new_positions) + len(closed) + len(increased) + len(decreased),
        },
    }


def format_diff_report(changes: dict) -> str:
    """Format diff results into a human-readable report."""
    lines = []
    summary = changes.get("summary", {})

    if summary.get("total_changes", 0) == 0:
        return "持仓无变化。"

    lines.append(f"📊 持仓变化检测（共{summary['total_changes']}项变化）\n")

    for h in changes.get("new_positions", []):
        lines.append(f"  🆕 新建仓: {h['stock_name']}({h['stock_code']}) "
                      f"{h['shares']}股 成本¥{h.get('cost_price', '?')}")

    for h in changes.get("closed", []):
        lines.append(f"  ❌ 清仓: {h['stock_name']}({h['stock_code']}) "
                      f"原{h['shares']}股")

    for h in changes.get("increased", []):
        ds = h["diff_shares"]
        lines.append(f"  ⬆️ 加仓: {h['stock_name']}({h['stock_code']}) "
                      f"+{ds}股 ({h['prev_shares']}→{h['shares']})")

    for h in changes.get("decreased", []):
        ds = abs(h["diff_shares"])
        lines.append(f"  ⬇️ 减仓: {h['stock_name']}({h['stock_code']}) "
                      f"-{ds}股 ({h['prev_shares']}→{h['shares']})")

    for h in changes.get("unchanged", []):
        if h.get("cost_changed"):
            lines.append(f"  💰 成本变化: {h['stock_name']}({h['stock_code']}) "
                          f"¥{h.get('prev_cost_price', '?')}→¥{h.get('cost_price', '?')}")

    return "\n".join(lines)
