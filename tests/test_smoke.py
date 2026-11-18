"""冒烟：demo 数据下全部页面端点 200 且含关键字段（CI 与本地 make test 共用）。"""
import subprocess, sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _ensure_demo():
    if not (ROOT / "data/store/scan_history.parquet").exists():
        subprocess.run([sys.executable, "demo/generate_demo.py", "--install"], cwd=ROOT, check=True)


def test_pages():
    _ensure_demo()
    from web.backend.app import app
    c = TestClient(app)
    r = c.get("/api/page/dashboard"); assert r.status_code == 200 and "signals" in r.json()
    r = c.get("/api/page/portfolio"); assert r.status_code == 200 and "positions" in r.json()
    r = c.get("/api/page/health"); assert r.status_code == 200 and "freshness" in r.json()
    r = c.get("/api/page/stock/600519?days=60"); assert r.status_code == 200 and r.json()["kline"]
    r = c.get("/api/evening-report"); assert r.status_code == 200 and "收盘总结" in r.json()["text"]
    r = c.get("/api/settings/status"); assert r.status_code == 200 and "version" in r.json()
