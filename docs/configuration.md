# 配置

全部配置走 `.env`（复制 `.env.example`）：

| 变量 | 作用 | 默认 |
|---|---|---|
| TUSHARE_TOKEN | tushare 数据源 token | 空 |
| LLM_API_KEY / LLM_BASE_URL / LLM_MODEL | AI 解盘（OpenAI 兼容） | 空 / openai / gpt-4o-mini |
| STOCK_DATA_DIR / STOCK_CONFIG_DIR / STOCK_REPORTS_DIR | 数据三目录 | 仓库内 data/ config/ reports/ |
| HOST / PORT | 服务绑定 | 127.0.0.1 / 8601 |

持仓文件 `config/portfolio.xlsx` 列说明：
`code(6位) name buy_date avg_cost shares stop_loss take_profit position_pct position_type notes`
（shares=0 表示已清仓保留历史；position_type 含"类债"计入三轴的类债轴）。

`config/cron.json`（可选）：看板"系统健康"页展示的计划任务清单。
`config/notes.md`（可选）：按 `## 标题` 分节的运维笔记，展示在健康页。
