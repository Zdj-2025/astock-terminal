#!/usr/bin/env python3
"""Import historical snapshot data from screenshots into tracker.db.

This is a one-time import script for the 0421 and 0422 screenshots.
"""
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.tracker.portfolio_tracker import PortfolioTracker


def import_all():
    db_path = str(Path(project_root) / "data" / "tracker.db")
    tracker = PortfolioTracker(db_path)

    # 0421 snapshot data (extracted from pics/0421.jpg)
    holdings_0421 = [
        {"stock_code": "603881", "stock_name": "数据港", "shares": 1760, "cost_price": 38.100, "current_price": 36.650, "market_value": 64584.0, "pnl_amount": -2554.0, "pnl_pct": -3.81},
        {"stock_code": "515070", "stock_name": "AI智能", "shares": 30000, "cost_price": 1.223, "current_price": 1.446, "market_value": 43380.0, "pnl_amount": 6690.0, "pnl_pct": 18.24},
        {"stock_code": "300059", "stock_name": "东方财富", "shares": 2000, "cost_price": 26.181, "current_price": 22.600, "market_value": 45200.0, "pnl_amount": -7162.0, "pnl_pct": -13.67},
        {"stock_code": "603617", "stock_name": "君禾股份", "shares": 5000, "cost_price": 7.695, "current_price": 7.280, "market_value": 36400.0, "pnl_amount": -2075.0, "pnl_pct": -5.39},
        {"stock_code": "512760", "stock_name": "芯片ETF", "shares": 40000, "cost_price": 0.549, "current_price": 0.767, "market_value": 30680.0, "pnl_amount": 8720.0, "pnl_pct": 39.71},
        {"stock_code": "512480", "stock_name": "半导体", "shares": 26000, "cost_price": 1.596, "current_price": 1.630, "market_value": 42380.0, "pnl_amount": 884.0, "pnl_pct": 2.13},
        {"stock_code": "515700", "stock_name": "大数据", "shares": 40000, "cost_price": 0.971, "current_price": 0.960, "market_value": 38400.0, "pnl_amount": -440.0, "pnl_pct": -1.13},
        {"stock_code": "510050", "stock_name": "50ETF", "shares": 12000, "cost_price": 2.906, "current_price": 2.740, "market_value": 32880.0, "pnl_amount": -1992.0, "pnl_pct": -5.71},
        {"stock_code": "000159", "stock_name": "国际实业", "shares": 5000, "cost_price": 5.498, "current_price": 5.830, "market_value": 29150.0, "pnl_amount": 1660.0, "pnl_pct": 6.04},
        {"stock_code": "300459", "stock_name": "汤姆猫", "shares": 5800, "cost_price": 3.930, "current_price": 4.040, "market_value": 23432.0, "pnl_amount": 638.0, "pnl_pct": 2.80},
        {"stock_code": "515120", "stock_name": "创新药", "shares": 30000, "cost_price": 0.623, "current_price": 0.625, "market_value": 18750.0, "pnl_amount": 60.0, "pnl_pct": 0.32},
    ]

    capital_0421 = {
        "total_assets": 453358.0,
        "available_cash": 30913.94,
        "market_value": 422444.06,
        "total_pnl": 71911.66,
        "position_ratio": 0.936,
    }

    print("📥 导入 2026-04-21 快照...")
    result = tracker.save_snapshot("2026-04-21", "screenshot", holdings_0421, capital_0421)
    print(f"  保存: {result['saved_count']}只持仓")
    if result["changes"].get("message"):
        print(f"  {result['changes']['message']}")

    # 0422 snapshot data (corrected from pics/截图0422.jpg — 收盘后截图)
    holdings_0422 = [
        {"stock_code": "603881", "stock_name": "数据港", "shares": 2260, "cost_price": 23.857, "current_price": 39.570, "market_value": 89428.20, "pnl_amount": 35512.45, "pnl_pct": 65.86},
        {"stock_code": "159995", "stock_name": "芯片ETF", "shares": 60000, "cost_price": 0.549, "current_price": 0.908, "market_value": 54480.00, "pnl_amount": 21516.04, "pnl_pct": 65.39},
        {"stock_code": "300059", "stock_name": "东方财富", "shares": 2000, "cost_price": 26.081, "current_price": 20.180, "market_value": 40360.00, "pnl_amount": -11801.63, "pnl_pct": -22.63},
        {"stock_code": "603617", "stock_name": "君禾股份", "shares": 5000, "cost_price": 7.760, "current_price": 7.300, "market_value": 36500.00, "pnl_amount": -2299.79, "pnl_pct": -5.93},
        {"stock_code": "159819", "stock_name": "AI智能", "shares": 20000, "cost_price": 0.984, "current_price": 1.777, "market_value": 35540.00, "pnl_amount": 15869.07, "pnl_pct": 80.59},
        {"stock_code": "512480", "stock_name": "半导体", "shares": 20000, "cost_price": 1.589, "current_price": 1.677, "market_value": 33540.00, "pnl_amount": 1768.00, "pnl_pct": 5.54},
        {"stock_code": "300459", "stock_name": "汤姆猫", "shares": 5800, "cost_price": 4.060, "current_price": 4.060, "market_value": 23548.00, "pnl_amount": 0.0, "pnl_pct": 0.0},
        {"stock_code": "515120", "stock_name": "创新药", "shares": 20000, "cost_price": 0.625, "current_price": 0.629, "market_value": 12580.00, "pnl_amount": 75.00, "pnl_pct": 0.64},
    ]

    capital_0422 = {
        "total_assets": 458605.37,
        "available_cash": 132628.67,
        "market_value": 325976.20,
        "total_pnl": 112289.71,
        "position_ratio": 0.7108,
    }

    print("\n📥 导入 2026-04-22 快照...")
    result = tracker.save_snapshot("2026-04-22", "screenshot", holdings_0422, capital_0422)
    print(f"  保存: {result['saved_count']}只持仓")

    from scripts.tracker.snapshot_diff import format_diff_report
    if result["changes"]:
        print()
        print(format_diff_report(result["changes"]))

    if result["trades_detected"]:
        print(f"\n📝 自动检测到 {len(result['trades_detected'])} 笔交易:")
        for t in result["trades_detected"]:
            print(f"  [{t['id']}] {t['action']}: {t['stock']}"
                  + (f" {t.get('shares', '')}股" if t.get('shares') else ""))

    print("\n✅ 历史数据导入完成！")


if __name__ == "__main__":
    import_all()
