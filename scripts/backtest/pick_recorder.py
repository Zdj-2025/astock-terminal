"""
选股推荐记录器

每次推荐/筛选股票时，生成结构化CSV记录。
支持：写入、更新状态、回测统计。

存储路径: data/picks/YYYY-MM-DD.csv
字段说明见 memory/stock-picking/03-选股记录系统设计.md
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PICKS_DIR = PROJECT_ROOT / "data" / "picks"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest"

# CSV列定义（固定顺序）
COLUMNS = [
    # 基本信息
    "date", "symbol", "name", "industry",
    # 算法来源
    "source", "strategy", "score", "grade",
    # 推荐价位
    "recommend_price", "buy_zone", "target_price", "stop_loss",
    # 推荐理由
    "reason", "algorithm_tags", "risk_tags",
    # 技术指标快照
    "position_score", "rsi", "ma_bull",
    # 操作建议
    "action",
    # 状态跟踪
    "status", "triggered_date", "closed_date",
    # 回测结果
    "result_pct", "result_days", "max_gain_pct", "max_drawdown_pct",
    # 备注
    "notes",
]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _filepath(date: str = None) -> Path:
    """获取指定日期的CSV路径"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    _ensure_dir(PICKS_DIR)
    return PICKS_DIR / f"{date}.csv"


def record_pick(
    symbol: str,
    name: str,
    source: str,           # confirmed_scan / emerging_scan / manual_pick / holding_review
    strategy: str,         # trend_v2 / gain_momentum / signal_fusion / w_bottom
    score: float,
    grade: str,
    recommend_price: float,
    buy_zone: str,         # "11.5-11.7"
    target_price: float,
    stop_loss: float,
    reason: str,
    algorithm_tags: str,   # "trend_v2_AA,gm_A,momentum_resonance"
    risk_tags: str = "",   # "rsi_overbought,high_position"
    position_score: float = None,
    rsi: float = None,
    ma_bull: bool = None,
    action: str = "pending",  # buy_now / wait_pullback / watch
    industry: str = "",
    notes: str = "",
    date: str = None,
) -> pd.DataFrame:
    """
    记录一条推荐。如果当日文件已存在，追加；否则新建。
    
    返回更新后的DataFrame。
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    row = {
        "date": date,
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "source": source,
        "strategy": strategy,
        "score": round(score, 4),
        "grade": grade,
        "recommend_price": round(recommend_price, 2),
        "buy_zone": buy_zone,
        "target_price": round(target_price, 2),
        "stop_loss": round(stop_loss, 2),
        "reason": reason,
        "algorithm_tags": algorithm_tags,
        "risk_tags": risk_tags,
        "position_score": position_score,
        "rsi": rsi,
        "ma_bull": ma_bull,
        "action": action,
        "status": "pending",
        "triggered_date": None,
        "closed_date": None,
        "result_pct": None,
        "result_days": None,
        "max_gain_pct": None,
        "max_drawdown_pct": None,
        "notes": notes,
    }
    
    fp = _filepath(date)
    
    if fp.exists():
        df = pd.read_csv(fp)
        # 检查是否已存在同一条记录（同symbol+source+strategy）
        mask = (df["symbol"] == symbol) & (df["source"] == source) & (df["strategy"] == strategy)
        if mask.any():
            # 更新已有记录
            for k, v in row.items():
                if v is not None:
                    df.loc[mask, k] = v
            df.to_csv(fp, index=False)
            return df
    else:
        df = pd.DataFrame(columns=COLUMNS)
    
    # 追加新记录
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df[COLUMNS]  # 保证列顺序
    df.to_csv(fp, index=False)
    return df


def update_status(
    symbol: str,
    date: str,
    status: str,           # pending / triggered / hit_target / stopped_out / expired
    result_pct: float = None,
    result_days: int = None,
    max_gain_pct: float = None,
    max_drawdown_pct: float = None,
    notes: str = "",
):
    """更新某条推荐的状态"""
    fp = _filepath(date)
    if not fp.exists():
        raise FileNotFoundError(f"找不到 {fp}")
    
    df = pd.read_csv(fp)
    mask = df["symbol"] == symbol
    if not mask.any():
        raise ValueError(f"{date} 没有 {symbol} 的记录")
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    df.loc[mask, "status"] = status
    if result_pct is not None:
        df.loc[mask, "result_pct"] = round(result_pct, 2)
    if result_days is not None:
        df.loc[mask, "result_days"] = result_days
    if max_gain_pct is not None:
        df.loc[mask, "max_gain_pct"] = round(max_gain_pct, 2)
    if max_drawdown_pct is not None:
        df.loc[mask, "max_drawdown_pct"] = round(max_drawdown_pct, 2)
    if notes:
        existing = df.loc[mask, "notes"].values[0]
        df.loc[mask, "notes"] = f"{existing} | {notes}" if pd.notna(existing) and existing else notes
    
    if status == "triggered":
        df.loc[mask, "triggered_date"] = today
    elif status in ("hit_target", "stopped_out", "expired"):
        df.loc[mask, "closed_date"] = today
    
    df.to_csv(fp, index=False)
    return df


def load_picks(date: str = None) -> pd.DataFrame:
    """加载指定日期的推荐记录"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    fp = _filepath(date)
    if not fp.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(fp)


def load_all_picks() -> pd.DataFrame:
    """加载所有推荐记录"""
    dfs = []
    if PICKS_DIR.exists():
        for f in sorted(PICKS_DIR.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except:
                pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=COLUMNS)


def backtest_summary(picks_df: pd.DataFrame = None) -> dict:
    """
    生成回测摘要统计。
    
    返回:
    {
        "total": 总推荐数,
        "closed": 已结束数,
        "win_rate": 胜率%,
        "avg_return": 平均收益率%,
        "by_strategy": {strategy: {win_rate, avg_return, count}},
        "by_grade": {grade: {win_rate, avg_return, count}},
        "by_source": {source: {win_rate, avg_return, count}},
    }
    """
    if picks_df is None:
        picks_df = load_all_picks()
    
    if len(picks_df) == 0:
        return {"total": 0, "closed": 0, "win_rate": 0, "avg_return": 0}
    
    closed = picks_df[picks_df["status"].isin(["hit_target", "stopped_out", "expired"])]
    
    result = {
        "total": len(picks_df),
        "pending": len(picks_df[picks_df["status"] == "pending"]),
        "triggered": len(picks_df[picks_df["status"] == "triggered"]),
        "closed": len(closed),
    }
    
    if len(closed) > 0:
        results = closed["result_pct"].dropna()
        if len(results) > 0:
            result["win_rate"] = round((results > 0).sum() / len(results) * 100, 1)
            result["avg_return"] = round(results.mean(), 2)
            result["max_return"] = round(results.max(), 2)
            result["min_return"] = round(results.min(), 2)
    
    # 按strategy分组
    by_strategy = {}
    for s, group in closed.groupby("strategy"):
        results = group["result_pct"].dropna()
        if len(results) > 0:
            by_strategy[s] = {
                "count": len(results),
                "win_rate": round((results > 0).sum() / len(results) * 100, 1),
                "avg_return": round(results.mean(), 2),
            }
    result["by_strategy"] = by_strategy
    
    # 按grade分组
    by_grade = {}
    for g, group in closed.groupby("grade"):
        results = group["result_pct"].dropna()
        if len(results) > 0:
            by_grade[g] = {
                "count": len(results),
                "win_rate": round((results > 0).sum() / len(results) * 100, 1),
                "avg_return": round(results.mean(), 2),
            }
    result["by_grade"] = by_grade
    
    return result


def print_summary(summary: dict = None, picks_df: pd.DataFrame = None):
    """打印回测摘要"""
    if summary is None:
        summary = backtest_summary(picks_df)
    
    if summary["total"] == 0:
        print("暂无推荐记录")
        return
    
    print(f"📊 选股回测摘要")
    print(f"  总推荐: {summary['total']} | 待触发: {summary.get('pending',0)} | 已触发: {summary.get('triggered',0)} | 已结束: {summary['closed']}")
    
    if "win_rate" in summary:
        print(f"  胜率: {summary['win_rate']}% | 平均收益: {summary['avg_return']}% | 最佳: {summary['max_return']}% | 最差: {summary['min_return']}%")
    
    if "by_strategy" in summary and summary["by_strategy"]:
        print(f"\n  按策略:")
        for s, r in summary["by_strategy"].items():
            print(f"    {s}: {r['count']}笔 胜率{r['win_rate']}% 平均{r['avg_return']}%")
    
    if "by_grade" in summary and summary["by_grade"]:
        print(f"\n  按等级:")
        for g, r in summary["by_grade"].items():
            print(f"    {g}: {r['count']}笔 胜率{r['win_rate']}% 平均{r['avg_return']}%")
