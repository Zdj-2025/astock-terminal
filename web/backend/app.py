"""stock Web 看板 v3 — FastAPI 后端。

页面级聚合端点（一次请求整页数据）+ 个股动作端点 + 构建产物静态托管。
启动: cd stock && python -m uvicorn web.backend.app:app \
      --port 8601 --host 0.0.0.0
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q  # noqa: E402
from web.backend import settings as S  # noqa: E402

app = FastAPI(title="astock-terminal", docs_url=None, redoc_url=None)

TXT = lambda v: None if v is None or v != v else v  # NaN→None


@app.get("/api/healthz")
def healthz():
    return {"ok": True, "ts": q.datetime.now().isoformat(timespec="seconds")}


# ═══════════════ 页面1：今日决策驾驶舱 ═══════════════

@app.get("/api/page/dashboard")
def page_dashboard():
    bench, _ = q.last_close("000001")
    sl = q.get_stop_loss_rows()
    cnt = lambda e: len([r for r in sl if e in r["trigger"]])
    cash, cash_asof, _src = q.load_cash_ratio()
    sig = {}
    for s in ("golden_cross", "divergence", "oversold_reversal"):
        d, sub = q.signals_latest(s)
        cols = {
            "golden_cross": ["code", "name", "industry", "price", "dist120", "vol_tag", "smart"],
            "divergence": ["code", "name", "industry", "price", "strength", "confirmed"],
            "oversold_reversal": ["code", "name", "industry", "price", "ret20", "pos250", "upside"],
        }[s]
        sig[s] = {"date": q.fmt_date(d), "rows": sub[[c for c in cols if c in sub.columns]]
                  .fillna("").to_dict("records")}
    acc = q.get_accumulate(20)
    ann_day, ann = q.load_announcements()
    recs = q.load_recommendations(20)
    pri = {"🔴高": 0, "🟡中": 1, "🟢低": 2}
    return {
        "bench": q.fmt_date(bench), "today": q.datetime.now().strftime("%m-%d"),
        "stop": {"red": cnt("🔴"), "orange": cnt("🟠"), "yellow": cnt("🟡")},
        "cash": {"pct": TXT(round(cash, 1)), "asof": cash_asof},
        "signals": sig,
        "stoploss": sorted(
            [{k: TXT(v) for k, v in r.items()} for r in sl],
            key=lambda r: ({"🔴": 0, "🟠": 1, "🟡": 2, "🟢": 3}.get(r["trigger"][:1], 9),
                           -(r["pnl"] or 0))),
        "accumulate": acc.head(20).fillna("").to_dict("records") if acc is not None and not acc.empty else [],
        "announcements": {"day": ann_day, "rows": (
            ann.sort_values("priority", key=lambda s: s.map(lambda x: pri.get(str(x), 3)))
            .fillna("").to_dict("records") if not ann.empty else [])},
        "recommendations": recs.fillna("").to_dict("records") if not recs.empty else [],
    }


@app.get("/api/morning-report")
def morning_report(refresh: bool = False):
    return {"text": q.morning_report(refresh=refresh)}


@app.get("/api/evening-report")
def evening_report():
    """收盘总结（晚报）：纯组装当日已落盘数据，零写入、秒出。

    区别于晨报（subprocess 跑盘前检查）：晚报不跑任何脚本——16:15 的
    daily_scan_all 已把信号写进 scan_history，这里只是把它读成人话。
    """
    L = []
    bar = "▓"
    L.append(f"=== 收盘总结（晚报） | {q.datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    L.append("")

    bench, _ = q.last_close("000001")
    sl = q.get_stop_loss_rows()
    red = [r for r in sl if "🔴" in r["trigger"]]
    org = [r for r in sl if "🟠" in r["trigger"]]
    yel = [r for r in sl if "🟡" in r["trigger"]]
    L.append(f"{bar} ① 持仓止损（数据至 {q.fmt_date(bench)}）")
    L.append(f"   🚨 真止损 {len(red)} 只 | 见底 {len(org)} | 移动 {len(yel)} | 共 {len(sl)} 只持仓")
    for r in red:
        L.append(f"   🔴 {r['name']}({r['code']}) 现价{r['cur']:.2f} 盈亏{r['pnl']:+.1f}% → {r['trigger'][1:]}")
    for r in org[:3]:
        L.append(f"   🟠 {r['name']}({r['code']}) {r['trigger'][1:]}")
    L.append("")

    L.append(f"{bar} ② 今日信号（scan_history）")
    for s, nm in (("golden_cross", "金叉"), ("divergence", "底背离"), ("oversold_reversal", "超跌反转")):
        d, sub = q.signals_latest(s)
        L.append(f"   {nm} {len(sub)} 只 · 数据日 {q.fmt_date(d)}")
    L.append("")

    acc = q.get_accumulate(20)
    if acc is not None and not acc.empty:
        L.append(f"{bar} ③ 信号累积榜 Top5（近20交易日 横向×纵向复合分）")
        for _, r in acc.head(5).iterrows():
            L.append(f"   {r['name']}({r['code']}) 综合{r['composite']:.3f} "
                     f"横向{int(r['horiz'])} 纵向{r['total_vert']:.1f} 策略{int(r['strategies'])}个")
        L.append("")

    ann_day, ann = q.load_announcements()
    if not ann.empty:
        L.append(f"{bar} ④ 公告（{ann_day} · {len(ann)} 条）")
        hi = ann[ann["priority"].astype(str).str.contains("高", na=False)]
        for _, r in hi.head(5).iterrows():
            L.append(f"   🔴 {r['name'] if 'name' in ann.columns else r['code']} {str(r['title'])[:38]}")
        if len(hi) == 0:
            L.append("   无高优先级公告")
        L.append("")

    recs = q.load_recommendations(0)
    if not recs.empty:
        fam = recs.dropna(subset=["pnl20"]).groupby("family")["pnl20"].mean().sort_values(ascending=False)
        L.append(f"{bar} ⑤ 推荐闭环（族均值 pnl20 · 共{len(recs)}条）")
        for f, v in fam.items():
            flag = "⚠️连续为负已列该删清单" if f == "smart" and v < 0 else ""
            L.append(f"   {f:16s} {v:+6.1f}%  {flag}")
        L.append("")

    cash, asof, _ = q.load_cash_ratio()
    L.append(f"{bar} ⑥ 现金比例 {cash:.1f}%（锚点 {asof} · 铁线20%/防守30%）")
    L.append("")
    L.append("💡 🔴真止损→次日开盘执行 · 买点必须数字区间 · 度量优先：先看净值再谈策略")
    L.append("💡 本报纯只读组装（不跑扫描）；扫描结果由 16:15 cron 落盘后自动反映于此")
    return {"text": "\n".join(L)}


# ═══════════════ 页面2：持仓与净值中心 ═══════════════

@app.get("/api/page/portfolio")
def page_portfolio():
    snaps = q.load_capital_snapshots()
    anchor = snaps.iloc[-1] if not snaps.empty else None
    cap = q.load_capital()
    eq = q.load_daily_equity()
    total = float(anchor["total"]) if anchor is not None else cap.get("总资产", float("nan"))
    cash_v = float(anchor["cash"]) if anchor is not None else cap.get("可用资金", float("nan"))
    mv_v = float(anchor["mv"]) if anchor is not None else cap.get("总市值", float("nan"))
    anchor_day = q.fmt_date(anchor["asof"]) if anchor is not None else "capital.xlsx"

    positions, bond, wide, attack = [], 0.0, 0.0, 0.0
    for _, r in q.load_portfolio(active_only=True).iterrows():
        d, close = q.last_close(r["code"])
        mv = close * r["shares"] if close else float("nan")
        pnl = (close / r["avg_cost"] - 1) * 100 if close and r["avg_cost"] > 0 else float("nan")
        d2sl = (close / r["stop_loss"] - 1) * 100 if close and r["stop_loss"] > 0 else None
        positions.append({"code": r["code"], "name": r["name"], "type": r["position_type"],
                          "cost": r["avg_cost"], "last": TXT(close),
                          "pnl": TXT(round(pnl, 1)) if pnl == pnl else None,
                          "shares": int(r["shares"]), "mv_wan": TXT(round(mv / 1e4, 1)) if mv == mv else None,
                          "pos_pct": TXT(r["position_pct"]), "stop": TXT(r["stop_loss"]),
                          "d2stop": TXT(round(d2sl, 1)) if d2sl is not None and d2sl == d2sl else None,
                          "date": q.fmt_date(d)[5:] if d else "—"})
        if mv == mv:
            if "类债" in str(r["position_type"]):
                bond += mv
            elif r["code"][:2] in ("51", "15", "56", "58"):
                wide += mv
            else:
                attack += mv
    tot4 = bond + wide + attack + cash_v
    benchmarks = {}
    for key, code in (("sse", "000001SH"), ("hs300", "000300SH")):
        b = q.load_kline(code)
        if not b.empty and not eq.empty:
            b = b[b["date"] >= eq["date"].iloc[0]]
            benchmarks[key] = {"dates": [q.fmt_date(s) for s in b["date"]],
                               "values": [round(float(v), 4) for v in b["close"]]}
    upd = q.get_profit_guard()
    isetf = lambda c: str(c)[:2] in ("51", "15", "56", "58")
    day_pnl = cap.get("当日盈亏")
    return {
        "anchor_day": anchor_day,
        "metrics": {"total": TXT(round(total / 1e4, 1)), "cash": TXT(round(cash_v / 1e4, 1)),
                    "mv": TXT(round(mv_v / 1e4, 1)),
                    "cash_pct": TXT(round(cash_v / total * 100, 1)) if total else None,
                    "pos_pct": TXT(round((1 - cash_v / total) * 100, 1)) if total else None,
                    "day_pnl": TXT(day_pnl)},
        "positions": positions,
        "profit_guard": {"stocks": [u for u in upd if not isetf(u["code"])],
                         "etfs": [u for u in upd if isetf(u["code"])]},
        "equity": {"dates": [q.fmt_date(s) for s in eq["date"]],
                   "total": [round(float(v), 2) for v in eq["total"]],
                   "dd": [round(float(v), 2) for v in eq["dd"]]},
        "snapshots": [{"asof": q.fmt_date(s), "total": round(float(t), 0)}
                      for s, t in zip(snaps["asof"], snaps["total"])] if not snaps.empty else [],
        "benchmarks": benchmarks,
        "three_axis": {"类债": round(bond / tot4 * 100, 1), "宽基防守": round(wide / tot4 * 100, 1),
                       "权益进攻": round(attack / tot4 * 100, 1), "现金": round(cash_v / tot4 * 100, 1)},
        "trades": q.load_trade_log(20).fillna("").to_dict("records"),
        "monthly_md": q.monthly_report_md(),
        "equity_end": q.fmt_date(eq["date"].iloc[-1]) if not eq.empty else "—",
    }


# ═══════════════ 页面3：个股分析工作台 ═══════════════

@app.get("/api/page/stock/{code}")
def page_stock(code: str, days: int = 120):
    payload = q.kline_payload(code, days=max(days, 250))
    if payload is None:
        raise HTTPException(404, f"本地无 {code} 的 K线数据")
    payload["wave"] = None
    try:
        payload["wave"] = q.get_wave(code)
    except Exception as e:  # 波浪本地失败不阻塞页面
        payload["wave_error"] = str(e)[:200]
    payload["kline_days"] = days
    # 发给前端的 K线裁剪到请求窗口（指标已按 250 天预热计算）
    for k in ("dates", "kline", "volume", "ma5", "ma10", "ma20", "ma60",
              "dif", "dea", "hist", "dif_f", "dea_f", "hist_f"):
        payload[k] = payload[k][-days:]
    return payload


@app.get("/api/chanlun/{code}")
def chanlun(code: str, with60: bool = False):
    try:
        return {"levels": q.get_chanlun(code, with_60min=with60)}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)[:300]})


@app.get("/api/comprehensive/{code}")
def comprehensive(code: str):
    try:
        return q.get_comprehensive(code)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)[:300]})


@app.get("/api/realtime/{code}")
def realtime(code: str):
    try:
        return {"rows": q.get_realtime(code)}
    except Exception as e:
        return JSONResponse(status_code=502, content={"rows": [], "error": str(e)[:200]})


# ═══════════════ 页面4：系统健康 ═══════════════

@app.get("/api/page/health")
def page_health():
    fresh = q.scan_freshness()
    codes = {r["code"]: r["name"] for _, r in q.load_portfolio(active_only=True).iterrows()}
    try:
        import pandas as pd
        wl = pd.read_csv(q.CONFIG / "watchlist.csv", engine="python", on_bad_lines="skip")
        wl["code"] = wl["code"].astype(str).str.zfill(6)
        for _, r in wl.head(8).iterrows():
            codes.setdefault(r["code"], str(r["name"]))
    except Exception:
        pass
    lag = []
    for code, nm in codes.items():
        d, _ = q.last_close(code)
        if d:
            lag.append({"code": code, "name": nm, "date": q.fmt_date(d),
                        "ok": d == fresh["bench"].replace("-", "")})
    checks = []
    import re as _re
    for line in q.log_tail("health_check.log", 10):
        m = _re.match(r"\[(\d{4}-\d{2}-\d{2})[\s\S]*?(PASS|FAIL)", line)
        if m:
            fails = _re.findall(r"✗(T\d+[^;；]*)", line)
            checks.append({"date": m.group(1), "pass": m.group(2) == "PASS",
                           "fails": [f.strip() for f in fails]})
    secs = list(q.state_sections().items())
    progress = secs[0][1] if secs else ""
    delist = secs[1][1] if len(secs) > 1 else ""
    if fresh["ok_rate"] != fresh["ok_rate"]:
        fresh["ok_rate"] = None
    return {
        "freshness": fresh, "lag": lag,
        "health_checks": checks[::-1],
        "cron": q.load_cron(),
        "state": {"progress": progress, "delist": delist},
        "sentinel": q.minute_sentinel()[:19],
        "logs": {"smart_tracker": q.log_tail("smart_tracker.log", 3),
                 "first_board": q.log_tail("first_board.log", 3)},
    }



# ═══════════════ AI 解读（OpenAI 兼容协议，可选） ═══════════════

@app.get("/api/settings/status")
def settings_status():
    return {"llm": bool(S.LLM_API_KEY), "tushare": bool(S.TUSHARE_TOKEN),
            "data_dir": str(S.DATA_DIR), "version": "0.3.0"}


@app.get("/api/ai/digest")
def ai_digest():
    """把晚报数据交给 LLM 生成解盘（需在 .env 配置 LLM_*）。"""
    if not S.LLM_API_KEY:
        return JSONResponse(status_code=400, content={
            "error": "未配置 LLM_API_KEY。复制 .env.example 为 .env，填入任意 OpenAI 兼容服务的 "
                     "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（GLM/OpenAI/DeepSeek/Ollama 均可）。"})
    import requests as _rq
    text = evening_report().get("text", "")[:6000]
    prompt = (
        "你是一名严谨的 A 股投研助理。以下是本地量化系统的收盘总结数据（程序实算，非行情软件）：\n\n"
        f"{text}\n\n"
        "请输出 markdown 格式的解盘，包含：①持仓风险（重点逐条核对🔴真止损）；②信号面解读"
        "（金叉/底背离/超跌的结构含义，不超过5条）；③明日关注清单（具体到价位）。"
        "约束：只依据给出的数据，不编造行情；数字必须引用原文；结尾固定加一行"
        "「> ⚠️ 以上为程序数据 + AI 生成的解读，不构成投资建议」。中文，600字内。")
    try:
        r = _rq.post(
            f"{S.LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {S.LLM_API_KEY}"},
            json={"model": S.LLM_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3},
            timeout=90)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]["content"]
        return {"markdown": msg, "model": S.LLM_MODEL}
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"LLM 调用失败：{str(e)[:200]}"})

# ═══════════════ 静态托管（前端构建产物） ═══════════════

DIST = Path(__file__).resolve().parent.parent / "frontend/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        f = DIST / full_path
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
