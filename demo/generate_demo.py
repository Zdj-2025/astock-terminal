#!/usr/bin/env python3
"""生成演示数据集（demo/）——让访客克隆后 30 秒看到完整界面。

全部数据为**合成**：真实股票代码/名称（公开事实）+ 随机游走价格（不构成任何
真实行情）。重新生成：python demo/generate_demo.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent

STOCKS = [  # code, name, industry, price0
    ("000001", "平安银行", "银行", 12), ("600519", "贵州茅台", "白酒", 1500), ("000858", "五粮液", "白酒", 130),
    ("601318", "中国平安", "保险", 45), ("600036", "招商银行", "银行", 33),
    ("000333", "美的集团", "家电", 65), ("600900", "长江电力", "电力", 27),
    ("601012", "隆基绿能", "光伏", 22), ("300750", "宁德时代", "电池", 180),
    ("002594", "比亚迪", "汽车", 240), ("600276", "恒瑞医药", "医药", 45),
    ("603881", "数据港", "IDC", 25), ("688981", "中芯国际", "半导体", 90),
    ("510300", "沪深300ETF", "宽基", 3.9), ("512100", "中证1000ETF", "宽基", 2.6),
    ("515120", "创新药ETF", "行业", 0.6), ("512480", "半导体ETF", "行业", 0.7),
]
INDICES = [("000001SH", 3200.0), ("000300SH", 3900.0)]
N = 260  # 交易日


def dates(n):
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return [int(d.strftime("%Y%m%d")) for d in days]


def ohlc(n, p0, vol=0.02, drift=0.0002):
    ret = rng.normal(drift, vol, n)
    close = p0 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 4, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 4, n)))
    volume = rng.lognormal(13, 0.5, n)
    return open_, high, low, close, volume


def main():
    D = dates(N)
    kdir = OUT / "data/store/kline"
    kdir.mkdir(parents=True, exist_ok=True)
    for code, _, _, p0 in STOCKS:
        (kdir / code).mkdir(parents=True, exist_ok=True)
        o, h, l, c, v = ohlc(N, p0)
        pd.DataFrame({"date": D, "open": o, "high": h, "low": l, "close": c,
                      "volume": v, "amount": v * c, "change_pct": np.append(0, np.diff(c) / c[:-1] * 100)}
                     ).to_parquet(kdir / f"{code}/D.parquet", index=False)
    for code, p0 in INDICES:
        (kdir / code).mkdir(parents=True, exist_ok=True)
        o, h, l, c, v = ohlc(N, p0, 0.01, 0.0001)
        pd.DataFrame({"date": D, "open": o, "high": h, "low": l, "close": c,
                      "volume": v, "amount": v * c}
                     ).to_parquet(kdir / f"{code}/D.parquet", index=True)
    pd.DataFrame([{"ts_code": f"{c}.SH" if c.startswith(("6", "5")) else f"{c}.SZ",
                   "symbol": c, "name": n, "industry": ind, "market": "", "cnspell": ""}
                  for c, n, ind, _ in STOCKS]
                 ).to_parquet(OUT / "data/store/stock_list.parquet", index=False)

    # scan_history：三策略 × 近 20 日随机信号
    rows = []
    for i, day in enumerate(D[-20:]):
        for strat, pool, nmax in (("golden_cross", STOCKS[::2], 12), ("divergence", STOCKS[1::2], 20), ("oversold_reversal", STOCKS[::3], 8)):
            k = min(nmax, len(pool))
            for j in rng.choice(len(pool), size=k, replace=False):
                code, name, ind, p0 = pool[j]
                px = p0 * (1 + rng.normal(0, 0.1))
                base = {"date": day, "strategy": strat, "code": code, "name": name,
                        "industry": ind, "price": round(px, 3), "metadata": ""}
                if strat == "golden_cross":
                    base.update(dist120=-rng.uniform(5, 60), vol_tag=str(rng.choice(["✅缩量", "⚠️放量", "—"], p=[.4, .3, .3])), smart=bool(rng.random() < .3), vs_ma20=rng.uniform(-8, 8))
                elif strat == "divergence":
                    base.update(strength=round(rng.uniform(.5, 1), 2), confirmed=bool(rng.random() < .6), db=bool(rng.random() < .2), vratio=round(rng.uniform(.3, 2), 2), turnover=round(rng.uniform(.2, 8), 2), vol_tag="✅缩量", dist120=-rng.uniform(5, 40))
                else:
                    base.update(ret20=-rng.uniform(15, 45), pos250=rng.uniform(1, 30), upside=rng.uniform(30, 200))
                rows.append(base)
    pd.DataFrame(rows).to_parquet(OUT / "data/store/scan_history.parquet", index=False)

    # 持仓 / 资金 / 流水（全虚构）
    cfg = OUT / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    pos = [("600519", "贵州茅台", "2025-09-10", 1420.0, 200, 1350.0, 1650.0, 28.0, "核心"),
           ("300750", "宁德时代", "2025-11-03", 165.0, 600, 152.0, 198.0, 20.0, "卫星"),
           ("510300", "沪深300ETF", "2025-08-20", 3.55, 80000, 3.30, 0.0, 30.0, "宽基防守"),
           ("515120", "创新药ETF", "2025-12-01", 0.58, 100000, 0.52, 0.0, 8.0, "行业卫星"),
           ("600900", "长江电力", "2025-06-15", 25.8, 4000, 24.0, 0.0, 12.0, "类债-核心"),
           ("603881", "数据港", "2026-01-20", 27.5, 1500, 24.8, 32.9, 5.0, "卫星")]
    pd.DataFrame(pos, columns=["code", "name", "buy_date", "avg_cost", "shares", "stop_loss", "take_profit", "position_pct", "position_type"]
                 ).to_excel(cfg / "portfolio.xlsx", index=False)
    pd.DataFrame({"项目": ["总资产", "可用资金", "总市值", "仓位pct", "当日盈亏"],
                  "数值": [1000000, 315000, 685000, 68.5, 5230.0]}
                 ).to_excel(cfg / "capital.xlsx", index=False)
    trades = pd.DataFrame({
        "date": [D[-i] for i in range(1, 11)],
        "code": rng.choice([p[0] for p in pos], 10),
        "name": "", "action": rng.choice(["买", "卖"], 10, p=[.4, .6]),
        "price": rng.uniform(3, 200, 10).round(3), "shares": rng.integers(100, 8000, 10),
        "amount": 0.0, "realized_pnl": rng.normal(800, 3000, 10).round(0), "note": "演示数据", "source": "demo"})
    trades["name"] = trades["code"].map({p[0]: p[1] for p in pos})
    trades["amount"] = (trades["price"] * trades["shares"]).round(0)
    trades.to_csv(cfg / "trade_log.csv", index=False)

    # 净值（250 日：先扬后抑再修复）+ 快照锚点
    acc = OUT / "reports/accounting"
    acc.mkdir(parents=True, exist_ok=True)
    nv = 1000000 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, N)))
    nv[120:180] *= np.linspace(1, 0.86, 60)          # 一段回撤
    peak = np.maximum.accumulate(nv)
    pd.DataFrame({"date": D, "total": nv.round(0), "cash": nv.round(0) * .35,
                  "mv": nv.round(0) * .65, "peak": peak.round(0),
                  "dd": ((nv / peak - 1) * 100).round(2)}
                 ).to_csv(acc / "daily_equity_reconstructed.csv", index=False)
    snaps = pd.DataFrame({
        "asof": [f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}" for d in D[::40]],
        "total": nv[::40].round(0), "cash": nv[::40].round(0) * .35,
        "mv": nv[::40].round(0) * .65, "leverage": "68%仓位", "source": "demo",
        "commit_date": "", "note": "", "self_capital": "", "schema": "", "era": ""})
    snaps.to_csv(acc / "capital_snapshots.csv", index=False)

    # 推荐闭环 / 公告 / cron / 运维笔记
    rep = OUT / "reports"
    recs = []
    for i in range(15):
        code, name, ind, p0 = STOCKS[i % len(STOCKS)]
        recs.append({"code": code, "name": name, "rec_date": f"{str(D[-i*7])[:4]}-{str(D[-i*7])[4:6]}-{str(D[-i*7])[6:]}",
                     "reason": "演示推荐：底背离+缩量", "mainline": ind, "regime": "震荡",
                     "buy_point": f"{p0*0.95:.2f}-{p0:.2f}", "stop_loss": round(p0*0.9, 2),
                     "note": "", "rec_close": round(p0, 2),
                     "pnl5": round(rng.normal(0, 4), 2), "pnl10": round(rng.normal(-.5, 7), 2),
                     "pnl20": round(rng.normal(-1, 10), 2), "buy_hit": "",
                     "family": rng.choice(["divergence", "golden_cross", "smart"], p=[.5, .3, .2]),
                     "entry_style": "wait_pullback"})
    pd.DataFrame(recs).to_csv(rep / "recommendations.csv", index=False)
    today = str(D[-1])
    pd.DataFrame({"code": ["600519", "300750", "600900"],
                  "date": [f"{today[:4]}-{today[4:6]}-{today[6:]}"] * 3,
                  "title": ["贵州茅台：关于回购公司股份的进展公告（演示）",
                            "宁德时代：2026年年度业绩预告（演示）",
                            "长江电力：季度发电量完成情况公告（演示）"],
                  "priority": ["🔴高", "🟡中", "🟢低"],
                  "link": "", "art_code": "", "source": "演示数据"}
                 ).to_csv(rep / f"announcements_{today}.csv", index=False)
    (cfg / "cron.json").write_text(
        '[{"name": "日线数据更新", "expr": "10 16 * * 1-5", "enabled": true},\n'
        ' {"name": "收盘扫描", "expr": "15 16 * * 1-5", "enabled": true},\n'
        ' {"name": "周末审计", "expr": "0 20 * * 6", "enabled": true}]\n', encoding="utf-8")
    (cfg / "notes.md").write_text(
        "# 运维笔记（示例）\n\n## 本周关注\n\n- demo 模式数据为合成，仅用于界面演示\n"
        "- 接入真实数据：python -m scripts.cli.bootstrap\n\n## 待办\n\n- [ ] 配置 TUSHARE_TOKEN\n"
        "- [ ] 首次全市场日线下载\n", encoding="utf-8")
    print(f"demo 数据生成完毕 → {OUT}")


def install():
    """把 demo/ 安装到仓库标准目录 ./data ./config ./reports（make demo 用）。

    这些目录都在 .gitignore 里——安装不会污染仓库，也兼容全部脚本的
    默认路径假设。清掉重来：删除三个目录再 --install。
    """
    import shutil
    root = OUT.parent
    for sub in ("data", "config", "reports"):
        src, dst = OUT / sub, root / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"installed {sub}/ → {dst}")


if __name__ == "__main__":
    import sys
    main()
    if "--install" in sys.argv:
        install()
