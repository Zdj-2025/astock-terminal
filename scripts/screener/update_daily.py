#!/usr/bin/env python3
"""每日数据更新脚本 — 多数据源安全版。

日线: Tushare（主力）→ BaoStock（备用）
分钟线(60/30/15min): BaoStock（主力，不限频）→ Tushare（备用）
实时+日线备用: 腾讯接口

用法:
    PYTHONPATH=. python scripts/screener/update_daily.py --periods D,W
    PYTHONPATH=. python scripts/screener/update_daily.py --periods 60min,30min,15min
    PYTHONPATH=. python scripts/screener/update_daily.py --periods D,W,60min,30min,15min
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 断点续传进度文件
PROGRESS_FILE = PROJECT_ROOT / "data" / "store" / "update_progress.json"


def load_progress() -> dict:
    """加载上次更新进度"""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except:
            return {}
    return {}


def save_progress(period: str, codes: list, last_index: int, updated: int, skipped: int, failed: int):
    """保存当前进度"""
    progress = load_progress()
    progress[period] = {
        "last_code": codes[last_index] if last_index < len(codes) else None,
        "last_index": last_index,
        "total": len(codes),
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "timestamp": datetime.now().isoformat(),
    }
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def clear_progress(period: str = None):
    """清除进度（完成后或强制重新开始）"""
    if period:
        progress = load_progress()
        progress.pop(period, None)
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False))
    else:
        PROGRESS_FILE.write_text("{}")


def show_progress():
    """显示上次更新进度"""
    progress = load_progress()
    if not progress:
        print("没有找到上次更新进度记录。")
        return
    print(f"{'='*60}")
    print(f"📊 上次更新进度")
    print(f"{'='*60}")
    for period, info in progress.items():
        last_code = info.get('last_code', '?')
        idx = info.get('last_index', 0)
        total = info.get('total', 0)
        pct = idx / total * 100 if total > 0 else 0
        ts = info.get('timestamp', '?')
        print(f"  {period}: {idx}/{total} ({pct:.1f}%) | 最后处理: {last_code} | 更新{info.get('updated',0)} 跳过{info.get('skipped',0)} 失败{info.get('failed',0)}")
        print(f"         时间: {ts}")
        if idx < total:
            print(f"         ⚠️ 未完成！下次将从第{idx+1}只({codes_at(idx, period)})继续")
        else:
            print(f"         ✅ 已完成")
    print(f"{'='*60}")


def codes_at(idx, period):
    """获取指定位置的代码（用于显示）"""
    progress = load_progress()
    return progress.get(period, {}).get('last_code', '?')


# ── BaoStock 分钟线下载 ──
_BAOSTOCK_FREQ = {'60min': '60', '30min': '30', '15min': '15', 'D': 'd', 'W': 'w'}


def _code_to_baostock(code: str) -> str:
    """6位代码 → baostock格式 sh.600001 / sz.000001"""
    if code.startswith('6') or code.startswith('9'):
        return f'sh.{code}'
    return f'sz.{code}'


def baostock_kline(code: str, period: str, start_date: str = '') -> pd.DataFrame | None:
    """从BaoStock获取K线数据。"""
    import baostock as bs
    bs_code = _code_to_baostock(code)
    freq = _BAOSTOCK_FREQ.get(period, 'd')
    adjust = '2'  # 前复权

    fields = 'date,time,open,high,low,close,volume'
    if period in ('D', 'W'):
        fields = 'date,open,high,low,close,volume'

    try:
        rs = bs.query_history_k_data_plus(bs_code, fields,
            start_date=start_date, frequency=freq, adjustflag=adjust)
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None

        if period in ('D', 'W'):
            df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        else:
            df = pd.DataFrame(rows, columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            # 合并 date+time → datetime
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].str[8:10] + ':' + df['time'].str[10:12])
            df = df.drop(columns=['date', 'time'])
            df = df.rename(columns={'datetime': 'date'})

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['amount'] = (df['close'] * df['volume']).round(2)
        df = df.dropna(subset=['close'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close'])
        return df
    except Exception as e:
        return None


def update_period(store, codes: list[str], period: str, resume: bool = True):
    """更新一个周期的数据。"""
    is_minute = period in ('60min', '30min', '15min')
    print(f"\n{'─'*50}")
    print(f"  更新周期: {period} ({'BaoStock' if is_minute else 'Tushare'})")
    print(f"{'─'*50}")

    # 断点续传：检查上次进度
    start_idx = 0
    progress = load_progress()
    if resume and period in progress:
        last = progress[period]
        last_idx = last.get('last_index', 0)
        last_code = last.get('last_code', '')
        if last_idx < last.get('total', 0) and last_idx > 0:
            # 验证codes列表是否一致（防止股票池变了）
            if last_idx < len(codes) and codes[last_idx] == last_code:
                # 从last_idx的下一只开始（last_idx那只可能没跑完）
                start_idx = last_idx
                print(f"  🔄 断点续传: 从第{start_idx+1}只 {codes[start_idx]} 继续 (共{len(codes)}只)")
                # 继承上次的统计
                updated = last.get('updated', 0)
                skipped = last.get('skipped', 0)
                failed = last.get('failed', 0)
            else:
                print(f"  ⚠️ 上次进度中的代码({last_code})与当前不一致，从头开始")
                updated, skipped, failed = 0, 0, 0
        else:
            # 上次已完成
            if last_idx >= last.get('total', 0):
                print(f"  ✅ 上次已完成 {period}，如需重新运行请用 --restart")
                return last.get('updated', 0), last.get('skipped', 0), last.get('failed', 0)
            updated, skipped, failed = 0, 0, 0
    else:
        updated, skipped, failed = 0, 0, 0

    consecutive_fails = 0

    for i in range(start_idx, len(codes)):
        code = codes[i]
        try:
            # 读现有数据
            old_df = store.get_kline(code, period=period, auto_update=False)
            if old_df is not None and not old_df.empty:
                date_col = "date" if "date" in old_df.columns else old_df.columns[0]
                old_latest = pd.to_datetime(old_df[date_col].iloc[-1])
                # 过滤掉NaT，从有效数据中找最新
                if pd.isna(old_latest):
                    valid = pd.to_datetime(old_df[date_col]).dropna()
                    old_latest = valid.iloc[-1] if len(valid) > 0 else None
            else:
                old_latest = None

            # 已是最新？跳过
            if old_latest is not None and old_latest >= pd.Timestamp.now() - pd.Timedelta(hours=12):
                skipped += 1
                consecutive_fails = 0
                continue

            if is_minute:
                # 分钟线走BaoStock（只拉最近7天，增量合并）
                if old_latest is not None and not pd.isna(old_latest):
                    start_str = (old_latest - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    start_str = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                new_data = baostock_kline(code, period, start_str)
                # 单只超时保护
                if new_data is not None and len(new_data) > 500:
                    # 只保留最后500行（太多了说明拉了全量）
                    new_data = new_data.tail(500)
                if new_data is not None and not new_data.empty:
                    if old_df is not None and not old_df.empty:
                        merged = pd.concat([old_df, new_data], ignore_index=True)
                        merged['date'] = pd.to_datetime(merged['date'])
                        merged = merged.drop_duplicates(subset=['date'], keep='last')
                        merged = merged.sort_values('date').reset_index(drop=True)
                        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                            if col in merged.columns:
                                merged[col] = pd.to_numeric(merged[col], errors='coerce')
                    else:
                        merged = new_data

                    p = Path(f'data/store/kline/{code}/{period}.parquet')
                    p.parent.mkdir(parents=True, exist_ok=True)
                    merged.to_parquet(p, index=False)
                    updated += 1
                    consecutive_fails = 0
                else:
                    failed += 1
                    consecutive_fails += 1
            else:
                # 日线/周线走Tushare增量
                new_df = store.get_kline(code, period=period, auto_update=True)
                if new_df is not None and not new_df.empty:
                    date_col = "date" if "date" in new_df.columns else new_df.columns[0]
                    new_latest = pd.to_datetime(new_df[date_col].iloc[-1])
                    if old_latest is not None and new_latest > old_latest:
                        updated += 1
                    else:
                        skipped += 1
                    consecutive_fails = 0
                else:
                    failed += 1
                    consecutive_fails += 1

        except Exception as e:
            failed += 1
            consecutive_fails += 1
            err = str(e)
            if "limit" in err.lower() or "频" in err:
                print(f"  [{i+1}/{len(codes)}] {code} 限流，等待3秒...")
                time.sleep(3)
            elif consecutive_fails <= 3:
                print(f"  [{i+1}/{len(codes)}] {code} 失败: {err[:60]}")

        # 连续失败保护
        if consecutive_fails >= 50:
            print(f"\n  ⚠️ 连续{consecutive_fails}次失败，暂停30秒...")
            time.sleep(30)
            consecutive_fails = 0

        # 每100只保存进度 + 显示
        if (i + 1) % 100 == 0:
            save_progress(period, codes, i, updated, skipped, failed)
            elapsed = time.time() - _start
            rate = (i + 1) / elapsed
            eta = (len(codes) - i - 1) / rate
            print(f"  [{i+1}/{len(codes)}] 更新{updated} 跳过{skipped} 失败{failed} | ETA {eta:.0f}s")

    # 完成后保存最终进度并标记完成
    save_progress(period, codes, len(codes), updated, skipped, failed)
    print(f"\n  ✅ {period}: 更新{updated} 跳过{skipped} 失败{failed}")
    return updated, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="每日数据增量更新（多数据源，支持断点续传）")
    parser.add_argument("--periods", type=str, default="D,W",
                        help="要更新的周期，逗号分隔")
    parser.add_argument("--pool", type=str, default="",
                        help="股票池CSV路径，为空则全量")
    parser.add_argument("--restart", action="store_true",
                        help="忽略断点记录，从头开始")
    parser.add_argument("--status", action="store_true",
                        help="只查看上次更新进度，不执行")
    args = parser.parse_args()

    if args.status:
        show_progress()
        return

    periods = [p.strip() for p in args.periods.split(",")]

    if args.pool:
        pool_df = pd.read_csv(args.pool)
        codes = pool_df["code"].astype(str).str.zfill(6).tolist()
    else:
        kline_dir = PROJECT_ROOT / "data" / "store" / "kline"
        codes = sorted([
            d.name for d in kline_dir.iterdir()
            if d.is_dir() and (d / "D.parquet").exists() and d.name[0].isdigit()
        ])

    print(f"{'='*60}")
    print(f"📊 每日数据更新 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   股票: {len(codes)}只 | 周期: {periods}")
    print(f"   日线: Tushare | 分钟线: BaoStock")
    print(f"{'='*60}")

    # BaoStock login（分钟线用）
    has_minute = any(p in ('60min', '30min', '15min') for p in periods)
    if has_minute:
        import baostock as bs
        bs.login()
        print("  BaoStock已登录")

    from scripts.core.data_store import DataStore
    store = DataStore()

    resume = not args.restart
    totals = {"updated": 0, "skipped": 0, "failed": 0}
    for period in periods:
        if args.restart:
            clear_progress(period)
        u, s, f = update_period(store, codes, period, resume=resume)
        totals["updated"] += u
        totals["skipped"] += s
        totals["failed"] += f

    if has_minute:
        import baostock as bs
        bs.logout()

    elapsed = time.time() - _start
    print(f"\n{'='*60}")
    print(f"🏁 全部完成 | 更新{totals['updated']} 跳过{totals['skipped']} 失败{totals['failed']}")
    print(f"   耗时: {elapsed:.0f}秒 ({elapsed/60:.1f}分钟)")
    print(f"{'='*60}")

    # 验证
    for period in periods:
        p = Path(f'data/store/kline/000001/{period}.parquet')
        if p.exists():
            df = pd.read_parquet(p)
            dc = "date" if "date" in df.columns else df.columns[0]
            print(f"验证 000001 {period} 最新: {df[dc].iloc[-1]}")


_start = time.time()

if __name__ == "__main__":
    main()
