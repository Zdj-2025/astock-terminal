"""运行配置 — 全部来自环境变量 / .env（复制 .env.example 为 .env 开始）。

目录类变量指向"你的数据在哪里"；demo 模式把三个目录都指到 demo/ 下即可
（见 make demo / docs/quickstart.md）。
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env 优先（复制 .env.example 为 .env）
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent


def _dir(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser().resolve()


# ── 数据目录（K线/信号等 parquet；默认 <仓库>/data） ──
DATA_DIR = _dir("STOCK_DATA_DIR", str(ROOT / "data"))
# ── 持仓配置目录（portfolio.xlsx / capital.xlsx / trade_log.csv） ──
CONFIG_DIR = _dir("STOCK_CONFIG_DIR", str(ROOT / "config"))
# ── 报告目录（净值/公告/推荐闭环） ──
REPORTS_DIR = _dir("STOCK_REPORTS_DIR", str(ROOT / "reports"))

# ── 数据源 ──
# Tushare token（必配才有点击"下载数据/缠论/综合评分"能力）：https://tushare.pro
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")

# ── AI 解读（OpenAI 兼容协议：GLM/OpenAI/DeepSeek/Ollama 均可） ──
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ── 服务 ──
HOST = os.environ.get("HOST", "127.0.0.1")   # 对外服务改 0.0.0.0（自行承担暴露风险）
PORT = int(os.environ.get("PORT", "8601"))


def llm_ready() -> bool:
    return bool(LLM_API_KEY)
