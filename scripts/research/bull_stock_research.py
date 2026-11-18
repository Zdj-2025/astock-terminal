#!/usr/bin/env python3
"""牛股特征研究 — 寻找启动前的共性指标。

思路:
1. 用 daily_basic 的 close 数据，计算全市场股票在2023-2025的6个月滚动收益
2. 找出涨幅>100%的牛股，记录启动点（区间最低价日期）
3. 分析启动点前后的技术/基本面特征：
   - MA排列、成交量变化、MACD状态
   - PE/PB、市值、换手率
   - 波浪位置（W2底部？W3刚启动？）
4. 汇总共性
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scripts.core.tushare_client import TushareClient, from_ts_code


def find_bull_stocks(
    min_gain_pct: float = 100.0,
    window_months: int = 6,
    year_range: tuple = (2023, 2025),
    min_mcap_yi: float = 20.0,
    top_n: int = 50,
) -> pd.DataFrame:
    """找出指定年份区间内的牛股。

    Returns:
        DataFrame with columns:
        ts_code, name, launch_date, peak_date, gain_pct,
        launch_close, peak_close, launch_mcap, launch_pe
    """
    client = TushareClient()

    # 按月取样，每月取一个交易日
    sample_dates = []
    for year in range(year_range[0], year_range[1] + 1):
        for month in range(1, 13):
            date_str = f"{year}{month:02d}01"
            td = client.get_recent_trade_day(base_date=date_str)
            if td:
                sample_dates.append(td)

    sample_dates = sorted(set(sample_dates))
    print(f"采样日期: {len(sample_dates)}个 ({sample_dates[0]}~{sample_dates[-1]})")

    # 收集每月全市场 daily_basic
    all_data = {}
    for i, dt in enumerate(sample_dates):
        if i % 6 == 0:
            print(f"  获取 daily_basic: {i+1}/{len(sample_dates)} ({dt})")
        try:
            df = client.get_daily_basic(trade_date=dt)
            if df is not None and not df.empty:
                all_data[dt] = df[["ts_code", "close", "pe_ttm", "pb",
                                   "total_mv", "turnover_rate", "volume_ratio"]]
        except Exception:
            continue

    if not all_data:
        print("  无数据返回")
        return pd.DataFrame()

    # 构建面板: ts_code × date → close
    print("  构建价格面板...")
    panels = []
    for dt, df in all_data.items():
        sub = df[["ts_code", "close"]].copy()
        sub["date"] = dt
        panels.append(sub)

    price_panel = pd.concat(panels, ignore_index=True)
    price_pivot = price_panel.pivot(index="ts_code", columns="date", values="close")

    # 计算6个月滚动收益
    window = window_months  # 近似6个月=6个采样点
    results = []

    dates = price_pivot.columns.tolist()
    for start_idx in range(len(dates) - window):
        end_idx = start_idx + window
        start_date = dates[start_idx]
        end_date = dates[end_idx]

        start_prices = price_pivot[start_date].dropna()
        end_prices = price_pivot[end_date].dropna()

        common = start_prices.index.intersection(end_prices.index)
        for ts_code in common:
            sp = float(start_prices[ts_code])
            ep = float(end_prices[ts_code])
            if sp <= 0:
                continue
            gain = (ep / sp - 1) * 100
            if gain >= min_gain_pct:
                results.append({
                    "ts_code": ts_code,
                    "start_date": start_date,
                    "end_date": end_date,
                    "gain_pct": round(gain, 1),
                    "start_close": sp,
                    "end_close": ep,
                })

    if not results:
        print("  未找到符合条件的牛股")
        return pd.DataFrame()

    df_bulls = pd.DataFrame(results)
    print(f"  原始牛股记录: {len(df_bulls)}条")

    # 去重：同一股票取最大涨幅区间
    df_bulls = df_bulls.sort_values("gain_pct", ascending=False)
    df_bulls = df_bulls.drop_duplicates(subset=["ts_code"], keep="first")

    # 过滤最小市值
    # 查 start_date 的 total_mv
    stock_list = client.get_stock_list()
    name_map = dict(zip(stock_list["ts_code"], stock_list["name"]))
    df_bulls["name"] = df_bulls["ts_code"].map(name_map)

    # 排除北交所 + ST股
    df_bulls = df_bulls[~df_bulls["ts_code"].str.endswith(".BJ")]
    if "name" in df_bulls.columns:
        df_bulls = df_bulls[~df_bulls["name"].str.contains("ST", case=False, na=False)]

    df_bulls = df_bulls.head(top_n).reset_index(drop=True)
    print(f"  最终牛股: {len(df_bulls)}只")

    return df_bulls


def analyze_launch_features(bull_stocks: pd.DataFrame) -> pd.DataFrame:
    """分析牛股在启动点的特征。

    改进: 用日K线精确定位启动低点（区间内最低价），
    而非依赖月度采样的粗糙日期。
    """
    client = TushareClient()

    features = []
    total = len(bull_stocks)

    for idx, row in bull_stocks.iterrows():
        ts_code = row["ts_code"]
        start_date = str(row["start_date"])
        end_date = str(row["end_date"])
        name = row.get("name", from_ts_code(ts_code))

        if pd.isna(name) or name == "nan":
            name = from_ts_code(ts_code)

        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  分析 {idx+1}/{total}: {name}({from_ts_code(ts_code)})")

        try:
            symbol = from_ts_code(ts_code)

            # 获取整个区间的K线 + 前180天
            sd = pd.Timestamp(str(start_date))
            ed = pd.Timestamp(str(end_date))
            fetch_start = (sd - pd.Timedelta(days=180)).strftime("%Y%m%d")
            fetch_end = (ed + pd.Timedelta(days=10)).strftime("%Y%m%d")

            df = client.get_kline(symbol, freq="D", start=fetch_start, end=fetch_end, adj="qfq")
            if df is None or len(df) < 60:
                continue

            if "date" in df.columns and "trade_date" not in df.columns:
                df = df.rename(columns={"date": "trade_date"})

            df = df.sort_values("trade_date").reset_index(drop=True)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)

            # 精确定位：在[start_date, end_date]区间内找最低价日
            mask = (df["trade_date"] >= sd) & (df["trade_date"] <= ed)
            period = df[mask]
            if period.empty:
                continue

            low_idx = period["close"].idxmin()
            launch_idx = low_idx
            launch_date_val = df.iloc[launch_idx]["trade_date"]
            # 确保launch_date是字符串格式YYYYMMDD
            if hasattr(launch_date_val, "strftime"):
                launch_date = launch_date_val.strftime("%Y%m%d")
            else:
                launch_date = str(launch_date_val).replace("-", "")

            if launch_idx < 30:
                continue

            close = df["close"]
            vol = df["volume"]
            f = {}

            # ── 技术指标 ──
            # MA
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()

            f["ma5_above_ma20"] = int(close.iloc[launch_idx] > ma20.iloc[launch_idx])
            f["ma20_above_ma60"] = int(ma20.iloc[launch_idx] > ma60.iloc[launch_idx]) if not pd.isna(ma60.iloc[launch_idx]) else None
            f["ma_alignment"] = (
                "bullish" if (f["ma5_above_ma20"] and f.get("ma20_above_ma60"))
                else "mixed"
            )

            # MA20方向（5日变化率）
            ma20_slope = (ma20.iloc[launch_idx] - ma20.iloc[max(0, launch_idx - 5)]) / ma20.iloc[max(0, launch_idx - 5)] * 100 if launch_idx >= 5 else 0
            f["ma20_slope_pct"] = round(ma20_slope, 2)

            # 成交量变化
            if launch_idx >= 20:
                vol_recent_5 = vol.iloc[launch_idx - 4:launch_idx + 1].mean()
                vol_prior_20 = vol.iloc[launch_idx - 24:launch_idx - 4].mean()
                f["vol_ratio_launch"] = round(vol_recent_5 / vol_prior_20, 2) if vol_prior_20 > 0 else None
            else:
                f["vol_ratio_launch"] = None

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            hist = (dif - dea) * 2

            f["macd_dif"] = round(float(dif.iloc[launch_idx]), 4)
            f["macd_golden_cross"] = int(
                dif.iloc[launch_idx] > dea.iloc[launch_idx] and
                dif.iloc[max(0, launch_idx - 1)] <= dea.iloc[max(0, launch_idx - 1)]
            )
            f["macd_dif_above_dea"] = int(dif.iloc[launch_idx] > dea.iloc[launch_idx])

            # RSI(14)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            f["rsi_14"] = round(float(rsi.iloc[launch_idx]), 1) if not pd.isna(rsi.iloc[launch_idx]) else None

            # 距60日低点涨幅
            low_60 = close.iloc[max(0, launch_idx - 60):launch_idx + 1].min()
            f["pct_from_60d_low"] = round((close.iloc[launch_idx] / low_60 - 1) * 100, 1)

            # 距20日低点涨幅
            low_20 = close.iloc[max(0, launch_idx - 20):launch_idx + 1].min()
            f["pct_from_20d_low"] = round((close.iloc[launch_idx] / low_20 - 1) * 100, 1)

            # ── 基本面 ──
            db = client.get_daily_basic(trade_date=launch_date)
            if db is not None and not db.empty:
                stock_db = db[db["ts_code"] == ts_code]
                if not stock_db.empty:
                    r = stock_db.iloc[0]
                    f["pe_ttm"] = float(r.get("pe_ttm", 0)) if pd.notna(r.get("pe_ttm")) else None
                    f["pb"] = float(r.get("pb", 0)) if pd.notna(r.get("pb")) else None
                    f["total_mv_yi"] = round(float(r.get("total_mv", 0)) / 1e4, 1)
                    f["turnover_rate"] = float(r.get("turnover_rate", 0)) if pd.notna(r.get("turnover_rate")) else None
                    f["volume_ratio"] = float(r.get("volume_ratio", 0)) if pd.notna(r.get("volume_ratio")) else None

            f["ts_code"] = ts_code
            f["name"] = name
            f["launch_date"] = str(launch_date)
            f["gain_pct"] = row["gain_pct"]
            f["launch_close"] = float(close.iloc[launch_idx])

            features.append(f)

            features.append(f)

        except Exception as e:
            import traceback
            if idx < 3:
                print(f"    ⚠ {name}({from_ts_code(ts_code)}) 分析失败: {e}")
                traceback.print_exc()
            continue

    return pd.DataFrame(features)


def summarize_features(features_df: pd.DataFrame) -> str:
    """汇总牛股特征"""
    if features_df.empty:
        return "无数据"

    lines = []
    lines.append("=" * 65)
    lines.append("  牛股启动前特征研究报告")
    lines.append(f"  样本: {len(features_df)}只牛股")
    lines.append("=" * 65)

    # ── 技术指标共性 ──
    lines.append("\n  ── 一、技术指标共性 ──\n")

    # MA排列
    bullish_pct = features_df["ma_alignment"].value_counts(normalize=True).get("bullish", 0) * 100
    ma5_above_20 = features_df["ma5_above_ma20"].mean() * 100
    ma20_above_60 = features_df["ma20_above_ma60"].dropna().mean() * 100
    lines.append(f"  MA5>MA20:       {ma5_above_20:.0f}% 的牛股在启动时站上MA20")
    lines.append(f"  MA20>MA60:      {ma20_above_60:.0f}% 的牛股MA20在MA60上方")
    lines.append(f"  多头排列:        {bullish_pct:.0f}% 的牛股呈现MA多头排列")

    # MA20斜率
    slope = features_df["ma20_slope_pct"].dropna()
    lines.append(f"  MA20斜率(5日):   均值{slope.mean():+.2f}%, 中位{slope.median():+.2f}%")
    slope_pos = (slope > 0).sum() / len(slope) * 100
    lines.append(f"  MA20上行比例:    {slope_pos:.0f}%")

    # MACD
    dif_above_dea = features_df["macd_dif_above_dea"].mean() * 100
    golden = features_df["macd_golden_cross"].mean() * 100
    lines.append(f"  DIF>DEA:        {dif_above_dea:.0f}% 的牛股MACD金叉状态")
    lines.append(f"  当日金叉:        {golden:.0f}% 在启动日当天或附近发生金叉")

    # 成交量
    vol_ratio = features_df["vol_ratio_launch"].dropna()
    lines.append(f"  量比(5日/20日):  均值{vol_ratio.mean():.2f}, 中位{vol_ratio.median():.2f}")
    vol_above_1 = (vol_ratio > 1.0).sum() / len(vol_ratio) * 100
    lines.append(f"  放量比例:        {vol_above_1:.0f}% 在启动时放量")

    # RSI
    rsi = features_df["rsi_14"].dropna()
    lines.append(f"  RSI(14):         均值{rsi.mean():.1f}, 中位{rsi.median():.1f}")
    rsi_40_60 = ((rsi >= 40) & (rsi <= 60)).sum() / len(rsi) * 100
    lines.append(f"  RSI在40-60:      {rsi_40_60:.0f}% (中性偏强区间)")

    # 距低点涨幅
    from_60d = features_df["pct_from_60d_low"].dropna()
    from_20d = features_df["pct_from_20d_low"].dropna()
    lines.append(f"  距60日低点:      均值+{from_60d.mean():.1f}%, 中位+{from_60d.median():.1f}%")
    lines.append(f"  距20日低点:      均值+{from_20d.mean():.1f}%, 中位+{from_20d.median():.1f}%")

    # ── 基本面共性 ──
    lines.append("\n  ── 二、基本面共性 ──\n")

    pe = features_df["pe_ttm"].dropna()
    pe_pos = pe[pe > 0]
    if len(pe_pos) > 0:
        lines.append(f"  PE(TTM):         均值{pe_pos.mean():.1f}, 中位{pe_pos.median():.1f}")
        pe_20_60 = ((pe_pos >= 20) & (pe_pos <= 60)).sum() / len(pe_pos) * 100
        lines.append(f"  PE在20-60:       {pe_20_60:.0f}%")

    pb = features_df["pb"].dropna()
    pb_pos = pb[pb > 0]
    if len(pb_pos) > 0:
        lines.append(f"  PB:              均值{pb_pos.mean():.1f}, 中位{pb_pos.median():.1f}")

    mv = features_df["total_mv_yi"].dropna()
    if len(mv) > 0:
        lines.append(f"  市值(亿):        均值{mv.mean():.0f}, 中位{mv.median():.0f}")
        mv_50_500 = ((mv >= 50) & (mv <= 500)).sum() / len(mv) * 100
        lines.append(f"  市值50-500亿:    {mv_50_500:.0f}%")

    tr = features_df["turnover_rate"].dropna()
    if len(tr) > 0:
        lines.append(f"  换手率:          均值{tr.mean():.1f}%, 中位{tr.median():.1f}%")

    vr = features_df["volume_ratio"].dropna()
    if len(vr) > 0:
        lines.append(f"  量比:            均值{vr.mean():.2f}, 中位{vr.median():.2f}")
        vr_above_1 = (vr > 1.0).sum() / len(vr) * 100
        lines.append(f"  量比>1:          {vr_above_1:.0f}%")

    # ── 关键发现 ──
    lines.append("\n  ── 三、关键发现 ──\n")

    findings = []
    if ma5_above_20 >= 60:
        findings.append(f"✓ {ma5_above_20:.0f}%牛股启动时站上MA20")
    if dif_above_dea >= 60:
        findings.append(f"✓ {dif_above_dea:.0f}%牛股启动时MACD金叉")
    if vol_above_1 >= 60:
        findings.append(f"✓ {vol_above_1:.0f}%牛股启动时放量")
    if slope_pos >= 60:
        findings.append(f"✓ {slope_pos:.0f}%牛股MA20方向向上")

    for finding in findings:
        lines.append(f"  {finding}")

    lines.append(f"\n{'=' * 65}")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="牛股特征研究")
    parser.add_argument("--min-gain", type=float, default=100.0,
                        help="最小涨幅%% (默认100%%)")
    parser.add_argument("--window", type=int, default=6,
                        help="滚动窗口月数 (默认6)")
    parser.add_argument("--top", type=int, default=50,
                        help="分析TOP N只牛股 (默认50)")
    parser.add_argument("--from-year", type=int, default=2023)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--skip-find", action="store_true",
                        help="跳过牛股发现，从缓存读取")
    args = parser.parse_args()

    cache_path = Path("data/cache/bull_research")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Step 1: 找牛股
    bull_cache = cache_path / "bull_stocks.csv"
    if args.skip_find and bull_cache.exists():
        print("从缓存读取牛股列表...")
        bull_stocks = pd.read_csv(bull_cache)
    else:
        print("Step 1: 寻找牛股...")
        bull_stocks = find_bull_stocks(
            min_gain_pct=args.min_gain,
            window_months=args.window,
            year_range=(args.from_year, args.to_year),
            top_n=args.top,
        )

        if bull_stocks.empty:
            print("未找到牛股")
            return

        bull_stocks.to_csv(bull_cache, index=False)
        print(f"\n  牛股列表已缓存: {bull_cache}")

    print(f"\n  牛股样本: {len(bull_stocks)}只, "
          f"涨幅范围: {bull_stocks['gain_pct'].min():.0f}%~{bull_stocks['gain_pct'].max():.0f}%")
    print(f"  TOP 10:")
    for _, r in bull_stocks.head(10).iterrows():
        name = r.get("name", from_ts_code(r["ts_code"]))
        print(f"    {name:10} {from_ts_code(r['ts_code'])}  "
              f"{r['start_date']}→{r['end_date']}  +{r['gain_pct']:.0f}%")

    # Step 2: 分析特征
    feat_cache = cache_path / "bull_features.csv"
    if args.skip_find and feat_cache.exists():
        print("\n从缓存读取特征数据...")
        features = pd.read_csv(feat_cache)
    else:
        print("\nStep 2: 分析启动前特征...")
        features = analyze_launch_features(bull_stocks)
        if features.empty:
            print("分析失败")
            return
        features.to_csv(feat_cache, index=False)
        print(f"  特征数据已缓存: {feat_cache}")

    # Step 3: 汇总报告
    print("\nStep 3: 汇总...")
    report = summarize_features(features)
    print(report)

    # 保存报告
    report_path = cache_path / "bull_research_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  报告已保存: {report_path}")


if __name__ == "__main__":
    main()
