#!/usr/bin/env python3
"""运行时配置 —— 环境变量与界面配置的统一读取层。

优先级：环境变量 / .env（部署级，命令行与 Docker 场景）
       > config/settings.local.json（「设置」页面保存的值）
       > 内置默认值。

两个消费方：
  - web/backend/settings.py（看板与 API）
  - scripts/core/tushare_client.py（命令行脚本，PYTHONPATH=. 运行）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_DEFAULTS = {
    "tushare_token": "",
    "llm_api_key": "",
    "llm_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-4o-mini",
}
# 环境变量名 → 配置键
_ENV = {
    "TUSHARE_TOKEN": "tushare_token",
    "LLM_API_KEY": "llm_api_key",
    "LLM_BASE_URL": "llm_base_url",
    "LLM_MODEL": "llm_model",
}


def config_dir() -> Path:
    return Path(os.environ.get("STOCK_CONFIG_DIR", str(ROOT / "config"))).expanduser().resolve()


def local_file() -> Path:
    return config_dir() / "settings.local.json"


def _load_local() -> dict:
    try:
        data = json.loads(local_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def effective() -> dict:
    """返回 {key: {"value": str, "source": "env"|"界面"|"默认"}}。"""
    local = _load_local()
    out = {}
    for key, default in _DEFAULTS.items():
        env_name = next(e for e, k in _ENV.items() if k == key)
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            out[key] = {"value": env_val, "source": "env"}
        elif str(local.get(key, "")).strip():
            out[key] = {"value": str(local[key]).strip(), "source": "界面"}
        else:
            out[key] = {"value": default, "source": "默认"}
    return out


def get(key: str) -> str:
    return effective()[key]["value"]


def save(patch: dict) -> None:
    """合并写入 config/settings.local.json（空字符串 = 清除该项，回落到默认值）。"""
    cfg = _load_local()
    for key, val in patch.items():
        if key not in _DEFAULTS:
            continue
        s = str(val).strip() if val is not None else ""
        if s:
            cfg[key] = s
        else:
            cfg.pop(key, None)
    local_file().parent.mkdir(parents=True, exist_ok=True)
    local_file().write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mask(val: str) -> str:
    """脱敏显示：只留前 4 后 4。"""
    if not val:
        return ""
    return val if len(val) <= 8 else f"{val[:4]}…{val[-4:]}"
