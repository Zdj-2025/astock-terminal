"""
选股推荐跟踪器

收盘后自动更新推荐记录的状态和收益率。
用Tushare日线数据获取最新价格，更新每条pending/triggered记录。

用法:
    PYTHONPATH=... python -m scripts.backtest.pick_tracker --date 2026-05-20
    PYTHONPATH=... python -m scripts.backtest.pick_tracker --all
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.tushare_client import TushareClient
from scripts.backtest.pick_recorder import (
    load_picks, load_all_picks, update_status, _filepath, COLUMNS
)


def track_daily(date: str = None, client: TushareClient = None):
    """
    收盘后跟踪：更新指定日期推荐记录的状态。
    
    逻辑:
    1. pending + 现价进入buy_zone → triggered
    2. triggered + 现价 >= target_price → hit_target
    3. triggered + 现价 <= stop_loss → stopped_out
    4. pending/triggered超过15天 → expired
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    if client is None:
        client = TushareClient()
    
    df = load_picks(date)
    if len(df) == 0:
        print(f"{date} 无推荐记录")
        return
    
    print(f"跟踪 {date} 的 {len(df)} 条推荐...")
    
    for idx, row in df.iterrows():
        symbol = row["symbol"]
        name = row["name"]
        status = row["status"]
        stop_loss = row["stop_loss"]
        target_price = row["target_price"]
        recommend_price = row["recommend_price"]
        buy_zone = str(row.get("buy_zone", ""))
        
        # 跳过已结束的
        if status in ("hit_target", "stopped_out", "expired"):
            continue
        
        # 获取最新价格（用Tushare日线，T-1）
        try:
            ts_code = symbol + (".SH" if symbol.startswith("6") else ".SZ")
            kline = client.get_kline(symbol, freq="D", start=date.replace("-", ""), adj="qfq")
            if kline is None or len(kline) == 0:
                print(f"  {name}({symbol}): 无价格数据")
                continue
            
            current = float(kline["close"].values[-1])
            
            # 计算相对推荐价的收益
            result_pct = round((current / recommend_price - 1) * 100, 2)
            
            # 计算持有天数
            rec_date = pd.Timestamp(date)
            latest_date = pd.Timestamp(kline["trade_date"].values[-1])
            days = (latest_date - rec_date).days
            
        except Exception as e:
            print(f"  {name}({symbol}): 获取价格失败 {e}")
            continue
        
        # 状态更新逻辑
        new_status = status
        notes = ""
        
        if status == "pending":
            # 检查是否进入买入区间
            buy_low, buy_high = None, None
            if "-" in buy_zone:
                parts = buy_zone.split("-")
                try:
                    buy_low = float(parts[0])
                    buy_high = float(parts[1])
                except:
                    pass
            
            if buy_low and buy_high and buy_low <= current <= buy_high:
                new_status = "triggered"
                notes = f"进入买入区间{buy_zone}，现价{current:.2f}"
            elif buy_low and current <= buy_high:
                # 价格已低于买入区间下限
                if current <= stop_loss:
                    new_status = "stopped_out"
                    notes = f"跌破止损{stop_loss}，现价{current:.2f}"
            
            # 超过15天未触发
            if days > 15 and new_status == "pending":
                new_status = "expired"
                notes = f"超过15天未触发"
        
        elif status == "triggered":
            if current >= target_price:
                new_status = "hit_target"
                notes = f"到达目标价{target_price}，现价{current:.2f}"
            elif current <= stop_loss:
                new_status = "stopped_out"
                notes = f"触发止损{stop_loss}，现价{current:.2f}"
            elif days > 20:
                new_status = "expired"
                notes = f"超过20天未了结"
        
        if new_status != status:
            update_status(
                symbol=symbol,
                date=date,
                status=new_status,
                result_pct=result_pct if new_status in ("hit_target", "stopped_out") else None,
                result_days=days if new_status in ("hit_target", "stopped_out") else None,
                notes=notes,
            )
            print(f"  {name}({symbol}): {status} → {new_status} | 现价{current:.2f} | {notes}")
        else:
            # 更新收益快照（不改变状态）
            fp = _filepath(date)
            df_day = pd.read_csv(fp)
            mask = df_day["symbol"] == symbol
            df_day.loc[mask, "notes"] = f"最新{current:.2f}({result_pct:+.1f}%)"
            df_day.to_csv(fp, index=False)


def track_all():
    """跟踪所有历史推荐"""
    client = TushareClient()
    df = load_all_picks()
    
    dates = df["date"].unique()
    print(f"跟踪 {len(dates)} 天的推荐记录...")
    
    for date in sorted(dates):
        track_daily(date=date, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="选股推荐跟踪器")
    parser.add_argument("--date", help="跟踪指定日期的推荐 (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="跟踪所有历史推荐")
    args = parser.parse_args()
    
    if args.all:
        track_all()
    elif args.date:
        track_daily(args.date)
    else:
        track_daily()
