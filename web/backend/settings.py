"""运行配置 —— 目录与密钥，统一走 scripts/core/runtime_settings.py。

优先级：环境变量 / .env > 「设置」页面保存的 config/settings.local.json > 默认值。
目录类变量（STOCK_*_DIR）只认环境变量；demo 模式见 make demo / docs/quickstart.md。
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env 优先（复制 .env.example 为 .env）
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from scripts.core import runtime_settings as _rs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent


def _dir(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser().resolve()


# ── 数据目录（K线/信号等 parquet；默认 <仓库>/data） ──
DATA_DIR = _dir("STOCK_DATA_DIR", str(ROOT / "data"))
# ── 持仓配置目录（portfolio.xlsx / capital.xlsx / settings.local.json） ──
CONFIG_DIR = _rs.config_dir()
# ── 报告目录（净值/公告/推荐闭环） ──
REPORTS_DIR = _dir("STOCK_REPORTS_DIR", str(ROOT / "reports"))


# ── 密钥（env 优先，其次「设置」页保存的值；来源见 source_of） ──

def reload() -> None:
    """「设置」页保存后调用：重新读取环境与本地配置文件。"""
    import importlib
    importlib.reload(_rs)


def _v(key: str) -> str:
    return _rs.get(key)


def llm_ready() -> bool:
    return bool(_v("llm_api_key"))


# 兼容旧引用：模块属性访问时代取值（app.py 里 S.LLM_API_KEY 等）
def __getattr__(name):  # PEP 562
    m = {"TUSHARE_TOKEN": "tushare_token", "LLM_API_KEY": "llm_api_key",
         "LLM_BASE_URL": "llm_base_url", "LLM_MODEL": "llm_model"}
    if name in m:
        return _v(m[name])
    raise AttributeError(name)
