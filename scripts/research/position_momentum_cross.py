#!/usr/bin/env python3
"""位置 × 二阶导 交叉回测。

测试：不同位置（估值/技术）下的f''信号效果是否有差异？

核心问题：
  动量(f'/f'') 和 回归(位置) 是否可以联合提升预测力？
  - 低位 + f''拐点 = 回归确认？胜率更高？
  - 高位 + f''拐点 = 动量延续？还是回归风险？
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

STORE_ROOT = Path("./data/store")
KLINE_ROOT = STORE_ROOT / "kline"
BASIC_ROOT = STORE_ROOT / "daily_basic"
SG_WIN, SG_POLY = 21, 3


def load_daily_basic():
    """加载daily_basic数据，构建截面百分位。"""
    print("加载 daily_basic...")
    all_dfs = []
    basic_files = sorted(BASIC_ROOT.glob("*.parquet"))
    # 只加载最近2年的数据（算百分位够用）
    target_dates = set()
    for f in basic_files[-500:]:  # 最近500个交易日
        target_dates.add(f.stem)

    for f in basic_files:
        if f.stem in target_dates:
            df = pd.read_parquet(f)
            all_dfs.append(df)

    if not all_dfs:
        print("  无daily_basic数据")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 按日期分组计算截面百分位
    print(f"  原始: {len(df):,}条, {df['trade_date'].nunique()}天")

    # 只保留最近2年
    max_date = df["trade_date"].max()
    min_date = max_date - pd.Timedelta(days=730)
    df = df[df["trade_date"] >= min_date]

    # 截面rank（每天所有股票的PE/PB排名）
    df["pe_rank"] = df.groupby("trade_date")["pe"].rank(pct=True)
    df["pb_rank"] = df.groupby("trade_date")["pb"].rank(pct=True)
    df["mv_rank"] = df.groupby("trade_date")["total_mv"].rank(pct=True)
    df["turnover_rank"] = df.groupby("trade_date")["turnover_rate"].rank(pct=True)

    # 个股历史PE百分位（需要pivot）
    print("  计算个股历史PE百分位...")
    pe_pivot = df.pivot_table(index="trade_date", columns="ts_code", values="pe")
    pe_pctl = pe_pivot.expanding(min_periods=60).rank(pct=True)
    # melt回long format
    pe_hist = pe_pctl.stack().reset_index()
    pe_hist.columns = ["trade_date", "ts_code", "pe_hist_pctl"]
    df = df.merge(pe_hist, on=["trade_date", "ts_code"], how="left")

    print(f"  处理后: {len(df):,}条")
    return df


def compute_derivatives(close):
    ln_p = np.log(close.astype(float))
    f1 = savgol_filter(ln_p, SG_WIN, SG_POLY, deriv=1) * 100
    f2 = savgol_filter(ln_p, SG_WIN, SG_POLY, deriv=2) * 10000
    return f1, f2


def scan(codes, basic_df):
    """扫描日线数据，联合基本面+二阶导。"""
    # 建立basic索引
    if basic_df is not None:
        basic_lookup = {}
        for _, row in basic_df.iterrows():
            key = (row["ts_code"], row["trade_date"])
            basic_lookup[key] = row

    all_rows = []
    t0 = time.time()

    for idx, code in enumerate(codes):
        try:
            path = KLINE_ROOT / code / "D.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            dt_col = "trade_date" if "trade_date" in df.columns else "date"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)
            close = df["close"].values.astype(float)
            n = len(close)
            if n < 180:
                continue

            f1, f2 = compute_derivatives(close)

            # ts_code格式
            if code.startswith("6"):
                ts_code = code + ".SH"
            else:
                ts_code = code + ".SZ"

            margin = max(SG_WIN, 60)
            for i in range(margin, n - 20, 5):
                current = close[i]
                f1v = f1[i]
                f2v = f2[i]

                # f''拐点检测
                f2_cross = False
                if i >= 3:
                    f2_past = np.mean(f2[i-3:i])
                    if f2_past < 0 and f2v > 0:
                        f2_cross = True

                # 技术位置：dh60/dh180/价格百分位
                dh60 = (current / np.max(close[max(0,i-60):i+1]) - 1) * 100 if i >= 60 else 0
                dh180 = (current / np.max(close[max(0,i-180):i+1]) - 1) * 100 if i >= 180 else 0

                # 价格在60日内的百分位
                if i >= 60:
                    price_pctl60 = (close[i-60:i+1] < current).sum() / 61
                else:
                    price_pctl60 = 0.5

                if i >= 180:
                    price_pctl180 = (close[i-180:i+1] < current).sum() / 181
                else:
                    price_pctl180 = 0.5

                # 基本面数据
                date = df.iloc[i][dt_col]
                basic_key = (ts_code, date)
                basic = basic_lookup.get(basic_key, {}) if basic_df is not None else {}

                pe = basic.get("pe", np.nan)
                pe_rank = basic.get("pe_rank", np.nan)
                pb_rank = basic.get("pb_rank", np.nan)
                mv_rank = basic.get("mv_rank", np.nan)
                turnover_rank = basic.get("turnover_rank", np.nan)
                pe_hist_pctl = basic.get("pe_hist_pctl", np.nan)

                # 标签
                ret20 = (close[min(i+20, n-1)] / current - 1) * 100

                all_rows.append({
                    "code": code,
                    "date": date,
                    "f1": round(f1v, 4),
                    "f2": round(f2v, 4),
                    "f2_cross": int(f2_cross),
                    "f1_pos": int(f1v > 0),
                    "dh60": round(dh60, 1),
                    "dh180": round(dh180, 1),
                    "price_pctl60": round(price_pctl60, 3),
                    "price_pctl180": round(price_pctl180, 3),
                    "pe": pe,
                    "pe_rank": pe_rank if not np.isnan(pe_rank) else np.nan,
                    "pb_rank": pb_rank if not np.isnan(pb_rank) else np.nan,
                    "mv_rank": mv_rank if not np.isnan(mv_rank) else np.nan,
                    "turnover_rank": turnover_rank if not np.isnan(turnover_rank) else np.nan,
                    "pe_hist_pctl": pe_hist_pctl if not np.isnan(pe_hist_pctl) else np.nan,
                    "ret20": round(ret20, 2),
                })
        except Exception:
            continue

        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(codes)} ({time.time()-t0:.0f}s) {len(all_rows):,}条")

    df = pd.DataFrame(all_rows)
    print(f"\n扫描完成: {len(df):,}条, {df['code'].nunique()}只")
    return df


def analyze(df):
    sep = "=" * 70

    # ============================================================
    print(f"\n{sep}")
    print("【一、位置维度 × f''拐点+f'>0 交叉胜率】")
    print(sep)

    mask_signal = (df["f2_cross"] == 1) & (df["f1"] > 0)
    mask_no = (df["f2_cross"] == 0)

    # 按dh180分位
    print(f"\n--- 按 dh180（距180日高点距离）---")
    print(f"{'dh180段':>15s}  {'有信号wr':>8s}  {'无信号wr':>8s}  {'提升':>6s}  {'有信号avg':>9s}  {'无信号avg':>9s}  {'n_sig':>6s}")

    for lo, hi in [(-999,-40),(-40,-20),(-20,-10),(-10,0),(0,999)]:
        mask_pos = (df["dh180"] >= lo) & (df["dh180"] < hi)
        sub_sig = df[mask_signal & mask_pos]
        sub_no = df[mask_no & mask_pos]
        if len(sub_sig) < 20 or len(sub_no) < 20:
            continue
        sig_wr = (sub_sig["ret20"]>0).mean()*100
        no_wr = (sub_no["ret20"]>0).mean()*100
        sig_avg = sub_sig["ret20"].mean()
        no_avg = sub_no["ret20"].mean()
        print(f"  [{lo:+4d},{hi:+4d})  {sig_wr:>6.1f}%  {no_wr:>6.1f}%  {sig_wr-no_wr:>+5.1f}%  {sig_avg:>+7.2f}%  {no_avg:>+7.2f}%  {len(sub_sig):>5,}")

    # 按价格百分位180
    print(f"\n--- 按 180日价格百分位 ---")
    print(f"{'pctl段':>15s}  {'有信号wr':>8s}  {'无信号wr':>8s}  {'提升':>6s}  {'有信号avg':>9s}  {'n_sig':>6s}")

    for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
        mask_pos = (df["price_pctl180"] >= lo) & (df["price_pctl180"] < hi)
        sub_sig = df[mask_signal & mask_pos]
        sub_no = df[mask_no & mask_pos]
        if len(sub_sig) < 20 or len(sub_no) < 20:
            continue
        sig_wr = (sub_sig["ret20"]>0).mean()*100
        no_wr = (sub_no["ret20"]>0).mean()*100
        sig_avg = sub_sig["ret20"].mean()
        no_avg = sub_no["ret20"].mean()
        print(f"  [{lo:.1f},{hi:.1f})   {sig_wr:>6.1f}%  {no_wr:>6.1f}%  {sig_wr-no_wr:>+5.1f}%  {sig_avg:>+7.2f}%  {len(sub_sig):>5,}")

    # 按PE截面百分位
    pe_valid = df.dropna(subset=["pe_rank"])
    if len(pe_valid) > 100:
        print(f"\n--- 按 PE截面百分位（当天所有股票排名）---")
        print(f"{'pe_rank段':>15s}  {'有信号wr':>8s}  {'无信号wr':>8s}  {'提升':>6s}  {'有信号avg':>9s}  {'n_sig':>6s}")

        mask_signal_pe = (pe_valid["f2_cross"] == 1) & (pe_valid["f1"] > 0)
        mask_no_pe = (pe_valid["f2_cross"] == 0)

        for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
            mask_pos = (pe_valid["pe_rank"] >= lo) & (pe_valid["pe_rank"] < hi)
            sub_sig = pe_valid[mask_signal_pe & mask_pos]
            sub_no = pe_valid[mask_no_pe & mask_pos]
            if len(sub_sig) < 20 or len(sub_no) < 20:
                continue
            sig_wr = (sub_sig["ret20"]>0).mean()*100
            no_wr = (sub_no["ret20"]>0).mean()*100
            sig_avg = sub_sig["ret20"].mean()
            print(f"  [{lo:.1f},{hi:.1f})   {sig_wr:>6.1f}%  {no_wr:>6.1f}%  {sig_wr-no_wr:>+5.1f}%  {sig_avg:>+7.2f}%  {len(sub_sig):>5,}")

    # 按PE历史百分位
    peh_valid = df.dropna(subset=["pe_hist_pctl"])
    if len(peh_valid) > 100:
        print(f"\n--- 按 PE历史百分位（个股自身5年百分位）---")
        print(f"{'pe_hist段':>15s}  {'有信号wr':>8s}  {'无信号wr':>8s}  {'提升':>6s}  {'有信号avg':>9s}  {'n_sig':>6s}")

        mask_signal_peh = (peh_valid["f2_cross"] == 1) & (peh_valid["f1"] > 0)
        mask_no_peh = (peh_valid["f2_cross"] == 0)

        for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
            mask_pos = (peh_valid["pe_hist_pctl"] >= lo) & (peh_valid["pe_hist_pctl"] < hi)
            sub_sig = peh_valid[mask_signal_peh & mask_pos]
            sub_no = peh_valid[mask_no_peh & mask_pos]
            if len(sub_sig) < 20 or len(sub_no) < 20:
                continue
            sig_wr = (sub_sig["ret20"]>0).mean()*100
            no_wr = (sub_no["ret20"]>0).mean()*100
            sig_avg = sub_sig["ret20"].mean()
            print(f"  [{lo:.1f},{hi:.1f})   {sig_wr:>6.1f}%  {no_wr:>6.1f}%  {sig_wr-no_wr:>+5.1f}%  {sig_avg:>+7.2f}%  {len(sub_sig):>5,}")

    # ============================================================
    print(f"\n{sep}")
    print("【二、PB_rank × f''拐点 — 估值+动量联合】")
    print(sep)

    pb_valid = df.dropna(subset=["pb_rank"])
    if len(pb_valid) > 100:
        mask_signal_pb = (pb_valid["f2_cross"] == 1) & (pb_valid["f1"] > 0)

        print(f"\n{'pb_rank段':>15s}  {'n':>6s}  {'ret20_wr':>8s}  {'ret20_avg':>9s}  {'vs无信号':>8s}")
        for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
            mask_pos = (pb_valid["pb_rank"] >= lo) & (pb_valid["pb_rank"] < hi)
            sub_sig = pb_valid[mask_signal_pb & mask_pos]
            sub_no = pb_valid[(pb_valid["f2_cross"]==0) & mask_pos]
            if len(sub_sig) < 20:
                continue
            sig_wr = (sub_sig["ret20"]>0).mean()*100
            sig_avg = sub_sig["ret20"].mean()
            no_avg = sub_no["ret20"].mean() if len(sub_no) > 10 else 0
            diff = sig_avg - no_avg
            print(f"  [{lo:.1f},{hi:.1f})   {len(sub_sig):>5,}  {sig_wr:>6.1f}%  {sig_avg:>+7.2f}%  {diff:>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【三、市值 × f''拐点 — 大票小票差异】")
    print(sep)

    mv_valid = df.dropna(subset=["mv_rank"])
    if len(mv_valid) > 100:
        mask_signal_mv = (mv_valid["f2_cross"] == 1) & (mv_valid["f1"] > 0)

        print(f"\n{'mv_rank段':>15s}  {'n':>6s}  {'ret20_wr':>8s}  {'ret20_avg':>9s}")
        for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
            mask_pos = (mv_valid["mv_rank"] >= lo) & (mv_valid["mv_rank"] < hi)
            sub_sig = mv_valid[mask_signal_mv & mask_pos]
            if len(sub_sig) < 20:
                continue
            sig_wr = (sub_sig["ret20"]>0).mean()*100
            sig_avg = sub_sig["ret20"].mean()
            print(f"  [{lo:.1f},{hi:.1f})   {len(sub_sig):>5,}  {sig_wr:>6.1f}%  {sig_avg:>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【四、换手率 × f''拐点 — 活跃度维度】")
    print(sep)

    tr_valid = df.dropna(subset=["turnover_rank"])
    if len(tr_valid) > 100:
        mask_signal_tr = (tr_valid["f2_cross"] == 1) & (tr_valid["f1"] > 0)

        print(f"\n{'turnover_rank段':>15s}  {'n':>6s}  {'ret20_wr':>8s}  {'ret20_avg':>9s}")
        for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]:
            mask_pos = (tr_valid["turnover_rank"] >= lo) & (tr_valid["turnover_rank"] < hi)
            sub_sig = tr_valid[mask_signal_tr & mask_pos]
            if len(sub_sig) < 20:
                continue
            sig_wr = (sub_sig["ret20"]>0).mean()*100
            sig_avg = sub_sig["ret20"].mean()
            print(f"  [{lo:.1f},{hi:.1f})   {len(sub_sig):>5,}  {sig_wr:>6.1f}%  {sig_avg:>+7.2f}%")

    # ============================================================
    print(f"\n{sep}")
    print("【五、IC矩阵 — 各指标与ret20的相关性】")
    print(sep)

    cols_to_test = ["f1", "f2", "dh60", "dh180", "price_pctl60", "price_pctl180",
                    "pe_rank", "pb_rank", "mv_rank", "turnover_rank", "pe_hist_pctl"]
    print(f"\n{'指标':>15s}  {'IC':>7s}")
    for col in cols_to_test:
        valid = df.dropna(subset=[col, "ret20"])
        if len(valid) < 100:
            continue
        ic = np.corrcoef(valid[col].values, valid["ret20"].values)[0, 1]
        print(f"  {col:>13s}  {ic:>+7.4f}")

    # f''拐点+f'>0 的IC（按pe_rank分段）
    print(f"\n--- f''拐+f'>0 的IC按PE位置分段 ---")
    mask_sig = (df["f2_cross"]==1) & (df["f1"]>0)
    peh_v = df.dropna(subset=["pe_hist_pctl"])
    for lo, hi in [(0,0.3),(0.3,0.7),(0.7,1.01)]:
        sub = peh_v[(peh_v["pe_hist_pctl"]>=lo) & (peh_v["pe_hist_pctl"]<hi)]
        sig_sub = sub[mask_sig.reindex(sub.index).fillna(False).astype(bool)]
        if len(sig_sub) < 20:
            continue
        wr = (sig_sub["ret20"]>0).mean()*100
        avg = sig_sub["ret20"].mean()
        print(f"  pe_hist [{lo:.1f},{hi:.1f}): n={len(sig_sub):,} wr={wr:.1f}% avg={avg:+.2f}%")


def main():
    print("="*70)
    print("位置 × 二阶导 交叉回测")
    print("="*70)

    # 加载基本面
    basic_df = load_daily_basic()

    # 获取代码列表
    codes = sorted([d.name for d in KLINE_ROOT.iterdir() if d.is_dir() and (d/"D.parquet").exists()])
    print(f"\n有日线数据的标的: {len(codes)}只")

    # 扫描
    print("\n开始扫描...")
    df = scan(codes, basic_df)
    if len(df) == 0:
        return

    # 分析
    analyze(df)

    out = "/tmp/position_momentum_cross.parquet"
    df.to_parquet(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
