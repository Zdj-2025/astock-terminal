#!/usr/bin/env python3
"""初始化向导 —— 从零到有自己的数据，三步。

用法: python -m scripts.cli bootstrap（或 make init）
  ① 生成 .env（可选填 TUSHARE_TOKEN）
  ② 从 templates/config 复制持仓/资金模板到 config/
  ③ 指引首次数据下载（全市场日线）与收盘扫描
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    print("═" * 56)
    print("astock-terminal 初始化向导")
    print("═" * 56)

    # ① .env
    env_path = ROOT / ".env"
    if not env_path.exists():
        tpl = ROOT / ".env.example"
        if tpl.exists():
            shutil.copy(tpl, env_path)
            print(f"✓ 已生成 {env_path.relative_to(ROOT)}（参考 .env.example 填写）")
    else:
        print(f"· {env_path.relative_to(ROOT)} 已存在，跳过")
    token = input("  现在填 Tushare token 吗？（注册 https://tushare.pro，回车跳过）: ").strip()
    if token:
        text = env_path.read_text()
        env_path.write_text(text.replace("TUSHARE_TOKEN=", f"TUSHARE_TOKEN={token}"))
        print("  ✓ token 已写入 .env")
    else:
        print("  · 跳过（缠论/综合评分/数据下载需要它；K线看板与demo不需要）")

    # ② 配置模板
    cfg = ROOT / "config"
    if (cfg / "portfolio.xlsx").exists():
        print(f"· config/portfolio.xlsx 已存在，跳过（不覆盖你的持仓）")
    else:
        cfg.mkdir(exist_ok=True)
        for f in ("portfolio.xlsx", "capital.xlsx", "trade_log.csv", "cron.json", "notes.md"):
            src = ROOT / "templates/config" / f
            if src.exists():
                shutil.copy(src, cfg / f)
        print(f"✓ 配置模板已复制到 {cfg.relative_to(ROOT)}/ —— 打开 portfolio.xlsx 填入你的持仓")

    # ③ 下一步指引
    print("""
③ 接下来（可选，按需执行）：
   # 首次下载全市场日线（需 TUSHARE_TOKEN，约 20-40 分钟）
   PYTHONPATH=. python scripts/data/update_all_kline.py

   # 首次收盘扫描（生成信号历史 scan_history.parquet，约 8-12 分钟）
   PYTHONPATH=. python scripts/screener/daily_scan_all.py

   # 启动看板
   make serve        # 或: uvicorn web.backend.app:app --port 8601

   # 30 秒先看看界面长什么样（合成演示数据）
   make demo
""")


if __name__ == "__main__":
    sys.exit(main())
