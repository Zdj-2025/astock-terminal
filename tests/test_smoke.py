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


def test_settings_roundtrip():
    """设置页：GET 脱敏返回 → POST 保存 → 立即生效 → 清除回落默认。"""
    from web.backend.app import app
    c = TestClient(app)
    r = c.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "masked" in body["tushare_token"] and "source" in body["llm_base_url"]

    r = c.post("/api/settings", json={"llm_base_url": "http://localhost:11434/v1",
                                      "llm_model": "qwen2.5:7b"})
    assert r.status_code == 200 and r.json()["ok"]
    eff = c.get("/api/settings").json()
    assert eff["llm_base_url"]["value"] == "http://localhost:11434/v1"
    assert eff["llm_model"]["value"] == "qwen2.5:7b"

    c.post("/api/settings", json={"llm_base_url": "", "llm_model": ""})   # 清除
    eff = c.get("/api/settings").json()
    assert eff["llm_base_url"]["source"] == "默认"

    r = c.post("/api/settings", json={"evil_key": "x"})                    # 未知键被忽略
    assert r.status_code == 200 and r.json()["saved"] == []
