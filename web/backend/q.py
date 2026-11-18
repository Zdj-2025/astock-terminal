"""查询层 — dashboard/loader.py 的 FastAPI 版（无 streamlit 依赖）。

红线同 loader.py：只读、直读 parquet（禁 DataStore）、tracker 只读、
不调任何会写文件的既有函数。TTL 缓存以 (key, mtime) 失效。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.backend import settings as S  # noqa: E402

STORE = S.DATA_DIR / "store"
KLINE = STORE / "kline"
REPORTS = S.REPORTS_DIR
CONFIG = S.CONFIG_DIR
ACCOUNTING = REPORTS / "accounting"

_cache: dict = {}


def _ttl(key, fn, ttl_s=300):
    """(key, mtime) 失效的进程内缓存。"""
    hit = _cache.get(key)
    now = time.time()
    if hit and now - hit[0] < ttl_s:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def fmt_date(d) -> str:
    s = str(d).replace("-", "").strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else str(d)


# ═══════════════ K线 ═══════════════

def load_kline(code, period: str = "D") -> pd.DataFrame:
    code = str(code).zfill(6)
    p = KLINE / f"{code}/{period}.parquet"
    if not p.exists():
        return pd.DataFrame()

    def _load():
        df = pd.read_parquet(p)
        dc = "date" if "date" in df.columns else "trade_date"
        df[dc] = df[dc].astype(str).str.replace("-", "", regex=False)
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df[c] = df[c].astype(float)
        return df.sort_values(dc).reset_index(drop=True)

    return _ttl(f"kline:{code}:{period}", _load, 3600)


def last_close(code):
    df = load_kline(code)
    if df.empty:
        return None, None
    return str(df["date"].iloc[-1]), float(df["close"].iloc[-1])


def stock_name(code) -> str:
    def _n():
        try:
            sl = pd.read_parquet(STORE / "stock_list.parquet", columns=["symbol", "name"])
            hit = sl[sl["symbol"].astype(str).str.zfill(6) == str(code).zfill(6)]
            return str(hit.iloc[0]["name"]) if not hit.empty else str(code)
        except Exception:
            return str(code)
    return _ttl(f"name:{code}", _n, 86400)


# ═══════════════ 持仓 / 资金 ═══════════════

def load_portfolio(active_only=False) -> pd.DataFrame:
    def _p():
        if not (CONFIG / "portfolio.xlsx").exists():
            return pd.DataFrame()
        df = pd.read_excel(CONFIG / "portfolio.xlsx")
        df["code"] = df["code"].astype(str).str.zfill(6)
        for c in ("avg_cost", "stop_loss", "take_profit", "position_pct", "last_price"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        return df
    df = _ttl("portfolio", _p, 120)
    if df.empty or "shares" not in df.columns:
        return pd.DataFrame()
    return df[df["shares"] > 0].reset_index(drop=True) if active_only else df


def load_capital() -> dict:
    def _c():
        if not (CONFIG / "capital.xlsx").exists():
            return pd.DataFrame()
        df = pd.read_excel(CONFIG / "capital.xlsx")
        out = {}
        for _, r in df.iterrows():
            k, v = str(r.iloc[0]).strip(), r.iloc[1]
            try:
                out[k] = float(str(v).replace(",", ""))
            except (TypeError, ValueError):
                out[k] = v
        return out
    return _ttl("capital", _c, 120)


def load_capital_snapshots() -> pd.DataFrame:
    def _s():
        if not (ACCOUNTING / "capital_snapshots.csv").exists():
            return pd.DataFrame()
        df = pd.read_csv(ACCOUNTING / "capital_snapshots.csv", dtype=str).fillna("")
        _num = lambda s: pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
        # 2026-11 起行在 asof 后多一个来源列（重构/券商截图）→ 整行右移，字符串阶段修复
        shifted = _num(df["total"]).isna() & _num(df["cash"]).notna()
        df.loc[shifted, ["total", "cash", "mv", "leverage"]] = (
            df.loc[shifted, ["cash", "mv", "leverage", "source"]].values)
        for c in ("total", "cash", "mv"):
            df[c] = _num(df[c])
        df["asof"] = df["asof"].astype(str).str.strip()
        return df.dropna(subset=["total", "cash"]).sort_values("asof").reset_index(drop=True)
    return _ttl("snaps", _s, 600)


def load_cash_ratio():
    snaps = load_capital_snapshots()
    if not snaps.empty:
        last = snaps.iloc[-1]
        if last["total"]:
            return (float(last["cash"] / last["total"] * 100),
                    fmt_date(last["asof"]), "capital_snapshots 锚点")
    cap = load_capital()
    if cap.get("总资产") and cap.get("可用资金"):
        return (cap["可用资金"] / cap["总资产"] * 100, "—", "capital.xlsx")
    return float("nan"), "—", "无数据"


def load_daily_equity() -> pd.DataFrame:
    def _e():
        if not (ACCOUNTING / "daily_equity_reconstructed.csv").exists():
            return pd.DataFrame()
        df = pd.read_csv(ACCOUNTING / "daily_equity_reconstructed.csv")
        df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
        return df.sort_values("date").reset_index(drop=True)
    return _ttl("equity", _e, 600)


def load_trade_log(n=50) -> pd.DataFrame:
    def _t():
        if not (CONFIG / "trade_log.csv").exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(CONFIG / "trade_log.csv")
        except pd.errors.ParserError:
            return pd.read_csv(CONFIG / "trade_log.csv", on_bad_lines="skip")
    df = _ttl("tradelog", _t, 300)
    return df.tail(n).iloc[::-1].reset_index(drop=True)


# ═══════════════ 信号 / 推荐 / 公告 ═══════════════

def load_scan_history(n_days=20) -> pd.DataFrame:
    def _s():
        if not (STORE / "scan_history.parquet").exists():
            return pd.DataFrame()
        df = pd.read_parquet(STORE / "scan_history.parquet")
        df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
        dates = sorted(df["date"].unique(), reverse=True)[:n_days]
        return df[df["date"].isin(dates)].copy()
    return _ttl("scanhist", _s, 300)


def signals_latest(strategy: str):
    sh = load_scan_history()
    sub = sh[sh["strategy"] == strategy]
    if sub.empty:
        return "", pd.DataFrame()
    latest = sub["date"].max()
    return latest, sub[sub["date"] == latest].reset_index(drop=True)


def get_stop_loss_rows() -> list[dict]:
    from scripts.screener.stop_loss_check import check_stop_loss
    return _ttl("stoploss", check_stop_loss, 300)


def get_accumulate(n_days=20) -> pd.DataFrame:
    from scripts.screener.scan_history import accumulate
    return _ttl("accumulate", lambda: accumulate(n_days=n_days, verbose=False), 300)


def load_recommendations(n=0) -> pd.DataFrame:
    def _r():
        if not (REPORTS / "recommendations.csv").exists():
            return pd.DataFrame()
        df = pd.read_csv(REPORTS / "recommendations.csv")
        for c in ("pnl5", "pnl10", "pnl20"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    df = _ttl("recs", _r, 300)
    return df.tail(n).iloc[::-1].reset_index(drop=True) if n else df


def load_announcements():
    def _a():
        files = sorted(glob.glob(str(REPORTS / "announcements_*.csv")))
        if not files:
            return "", pd.DataFrame()
        p = files[-1]
        day = re.search(r"(\d{8})\.csv$", p).group(1)
        return fmt_date(day), pd.read_csv(p)
    return _ttl("ann", _a, 600)


def morning_report(refresh=False) -> str:
    key = f"prem:{datetime.now():%Y%m%d}"
    if refresh:
        _cache.pop(key, None)

    def _run():
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), MPLBACKEND="Agg")
        try:
            r = subprocess.run(
                [sys.executable, "scripts/research/premarket_position_check.py"],
                cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300)
            out = r.stdout
            if r.returncode != 0:
                out += f"\n\n[exit={r.returncode}]\n{r.stderr[-1500:]}"
            return out or f"[无输出 exit={r.returncode}]"
        except subprocess.TimeoutExpired:
            return "[超时 300s — 财报季历走 tushare 逐票限速，网络不佳时会拖长]"
    return _ttl(key, _run, 3600)


def get_profit_guard() -> list[dict]:
    from scripts.analyzers.profit_guard import ProfitGuard
    port = load_portfolio(active_only=True)
    by_code = {r["code"]: r for r in get_stop_loss_rows()}
    pos = []
    for _, r in port.iterrows():
        sl = by_code.get(r["code"])
        if sl is None or r["avg_cost"] <= 0:
            continue
        pos.append({"code": r["code"], "name": r["name"], "cost": r["avg_cost"],
                    "price": sl["cur"], "shares": 0, "pnl_pct": sl["pnl"],
                    "position_pct": (r["position_pct"] or 0) / 100,
                    "stop_loss": r["stop_loss"] or 0})
    if not pos:
        return []
    return _ttl("pguard", lambda: ProfitGuard().check_portfolio(pos).get("sl_updates", []), 300)


# ═══════════════ 个股分析 ═══════════════

def kline_payload(code, days=250) -> dict | None:
    """ECharts K线数据：全量算指标，尾部 days 天输出。"""
    df = load_kline(code)
    if df.empty:
        return None
    c = df["close"]
    for p in (5, 10, 20, 60):
        df[f"ma{p}"] = c.rolling(p).mean()

    def _macd(fast, slow, sig):
        dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
        dea = dif.ewm(span=sig, adjust=False).mean()
        return dif, dea, 2 * (dif - dea)
    df["dif"], df["dea"], df["hist"] = _macd(12, 26, 9)
    df["dif_f"], df["dea_f"], df["hist_f"] = _macd(5, 34, 5)
    d = df.tail(days)
    rnd = lambda s: [None if v != v else round(float(v), 4) for v in s]
    return {
        "code": str(code).zfill(6), "name": stock_name(code),
        "dates": [fmt_date(s)[2:] for s in d["date"]],
        "kline": [[round(float(r.open), 4), round(float(r.close), 4),
                   round(float(r.low), 4), round(float(r.high), 4)]
                  for r in d.itertuples()],
        "volume": [None if v != v else float(v) for v in d["volume"]],
        **{f"ma{p}": rnd(d[f"ma{p}"]) for p in (5, 10, 20, 60)},
        "dif": rnd(d["dif"]), "dea": rnd(d["dea"]), "hist": rnd(d["hist"]),
        "dif_f": rnd(d["dif_f"]), "dea_f": rnd(d["dea_f"]), "hist_f": rnd(d["hist_f"]),
    }


def get_wave(code, days=250):
    from scripts.analyzers.elliott_wave import ElliottWaveAnalyzer
    def _w():
        df = load_kline(code).tail(days).reset_index(drop=True)
        if df.empty:
            return None
        wr = ElliottWaveAnalyzer().analyze(df)
        return {
            "direction": wr.direction, "current_wave": wr.current_wave,
            "total_score": wr.total_score, "confidence": wr.confidence,
            "waves": [{"label": w.label, "start_idx": w.start_idx, "end_idx": w.end_idx,
                       "start_price": w.start_price, "end_price": w.end_price,
                       "pct": round(w.pct_change, 2),
                       "start_date": w.start_date, "end_date": w.end_date}
                      for w in wr.waves],
            "iron_rules": [{"rule": r, "ok": bool(ok), "detail": d}
                           for r, ok, d in wr.iron_rules],
        }
    return _ttl(f"wave:{code}", _w, 1800)


def get_chanlun(code, with_60min=False):
    from scripts.analyzers.chanlun.engine import ChanLunEngine
    def _c():
        levels = ["daily", "60min"] if with_60min else ["daily"]
        res = ChanLunEngine().analyze(str(code).zfill(6), levels=levels)
        out = []
        for lv in res.levels:
            out.append({
                "level": lv.level,
                "strokes": [{"b_t": s.begin_time[:10], "e_t": s.end_time[:10],
                             "b_p": s.begin_price, "e_p": s.end_price}
                            for s in lv.strokes],
                "segments": [{"b_t": s.begin_time[:10], "e_t": s.end_time[:10],
                              "b_p": s.begin_price, "e_p": s.end_price}
                             for s in lv.segments],
                "pivots": [{"b_t": z.begin_time[:10], "e_t": z.end_time[:10],
                            "low": z.low, "high": z.high} for z in lv.pivots],
                "bs_points": [{"buy": b.is_buy, "types": b.bsp_types, "price": b.price,
                               "time": b.time[:10], "sure": b.is_sure}
                              for b in lv.bs_points],
            })
        return out
    return _ttl(f"chan:{code}:{with_60min}", _c, 1800)


def get_comprehensive(code) -> dict:
    from scripts.analyzers.comprehensive_analyzer import ComprehensiveAnalyzer
    res = ComprehensiveAnalyzer(str(code).zfill(6)).analyze()
    if "error" in res:
        return res
    sig = res.get("signal_4d", {})
    dims = sig.get("dims", {})
    return {
        "symbol": res.get("symbol"), "price": res.get("price"),
        "verdict": sig.get("verdict"), "composite": sig.get("composite"),
        "raw_score": sig.get("raw_score"), "conflict": sig.get("conflict"),
        "weights": sig.get("weights"),
        "dims": {k: round(float(v), 3) for k, v in dims.items()
                 if not k.endswith("_details")},
        "key_levels": res.get("key_levels", []),
        "advice": res.get("advice", {}),
    }


def get_realtime(code) -> list[dict]:
    from scripts.core.free_data_client import get_realtime_quote
    df = get_realtime_quote([str(code).zfill(6)])
    return df.to_dict("records") if not df.empty else []


# ═══════════════ 系统健康 ═══════════════

def scan_freshness():
    bench_df = load_kline("000001")
    bench = str(bench_df["date"].iloc[-1]) if not bench_df.empty else ""
    rows = []
    if not KLINE.exists():
        return {"bench": "", "files": 0, "ok_rate": float("nan"), "hist": []}
    with os.scandir(KLINE) as it:
        for e in it:
            if e.is_dir():
                p = Path(e.path) / "D.parquet"
                if p.exists():
                    rows.append((e.name, datetime.fromtimestamp(p.stat().st_mtime)))
    fdf = pd.DataFrame(rows, columns=["code", "mtime"])
    fdf["day"] = fdf["mtime"].dt.strftime("%Y-%m-%d")
    ok = (fdf["day"] >= fmt_date(bench)).mean() * 100 if bench and not fdf.empty else float("nan")
    hist = fdf.groupby("day").size().reset_index(name="n")
    return {"bench": fmt_date(bench), "files": len(fdf), "ok_rate": round(float(ok), 1),
            "hist": [{"day": r.day, "n": int(r.n)} for r in hist.itertuples()]}


def load_cron() -> list[dict]:
    """计划任务展示：读可选的 <CONFIG_DIR>/cron.json（模板见 templates/config/）。"""
    p = CONFIG / "cron.json"
    if not p.exists():
        return []
    try:
        jobs = json.loads(p.read_text())
        return [{"name": x.get("name", ""), "expr": x.get("expr", "—"),
                 "enabled": bool(x.get("enabled", True))}
                for x in jobs if isinstance(x, dict)]
    except (json.JSONDecodeError, AttributeError):
        return []


def state_sections() -> dict:
    """运维笔记（可选）：CONFIG_DIR/notes.md，按 `## 标题` 分节，展示在系统健康页。"""
    p = CONFIG / "notes.md"
    if not p.exists():
        return {}
    parts = re.split(r"(?m)^## ", p.read_text(encoding="utf-8"))
    out = {}
    for part in parts[1:]:
        lines = part.splitlines()
        if lines:
            out[lines[0].strip()] = "## " + part
    return out


def log_tail(name: str, n=10) -> list[str]:
    p = REPORTS / name
    if not p.exists():
        return []
    return [l for l in p.read_text(errors="replace").splitlines() if l.strip()][-n:]


def minute_sentinel() -> str:
    p = KLINE / "_last_minute_update"
    return p.read_text().strip() if p.exists() else ""


def monthly_report_md() -> str:
    p = ACCOUNTING / "净值月报_最新.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""
