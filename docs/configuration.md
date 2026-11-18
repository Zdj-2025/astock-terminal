# 配置

配置有两条途径，优先级：**.env / 环境变量 > 设置页**。

| 途径 | 怎么改 | 适合场景 | 生效方式 |
|---|---|---|---|
| 设置页 | 看板 →「设置」→ 填值保存 | 本机日常使用，改完即用 | 写入 `config/settings.local.json`，无需重启 |
| .env | 复制 `.env.example` 为 `.env` 编辑 | 服务器、Docker、写死在 crontab 环境 | 下次启动进程时读取 |

同一项两边都配时，`.env` 的值生效。设置页每项旁边的「来源」徽章标明当前生效的是哪边。

## 可配置项

| 项 | .env 变量 | 设置页入口 | 作用 |
|---|---|---|---|
| Tushare token | `TUSHARE_TOKEN` | 数据源 · Tushare | 日线下载、缠论、综合评分（见 [data-sources.md](data-sources.md)） |
| LLM 密钥 | `LLM_API_KEY` | 大模型 · API Key | AI 解盘（OpenAI 兼容协议） |
| LLM 地址 | `LLM_BASE_URL` | 大模型 · Base URL | 默认 `https://api.openai.com/v1`；GLM/DeepSeek/Ollama 见 `.env.example` 注释 |
| LLM 模型 | `LLM_MODEL` | 大模型 · Model | 默认 `gpt-4o-mini` |
| 数据目录 | `STOCK_DATA_DIR` | 无（仅环境变量） | K线/信号 parquet，默认 `<仓库>/data` |
| 配置目录 | `STOCK_CONFIG_DIR` | 无（仅环境变量） | 持仓/资金/设置，默认 `<仓库>/config` |
| 报告目录 | `STOCK_REPORTS_DIR` | 无（仅环境变量） | 净值/公告/推荐闭环，默认 `<仓库>/reports` |
| 服务绑定 | `HOST` / `PORT` | 无（仅环境变量） | 默认 `127.0.0.1:8601`；对外服务改 `0.0.0.0` 自行承担暴露风险 |

设置页的两个「测试连接」按钮分别用当前生效的 token / LLM 配置发一次最小请求，
返回明确的成功或报错信息。

密钥安全：设置页返回的值经过脱敏（只显示前 4 后 4 位）；`settings.local.json`
在 `.gitignore` 里，不会被提交。看板默认只绑 127.0.0.1，不要在无鉴权情况下
直接暴露到公网。

## 持仓与运维文件

持仓文件 `config/portfolio.xlsx` 列说明：
`code(6位) name buy_date avg_cost shares stop_loss take_profit position_pct position_type notes`
（shares=0 表示已清仓保留历史；position_type 含"类债"计入三轴的类债轴）。

`config/cron.json`（可选）：看板"系统健康"页展示的计划任务清单。
`config/notes.md`（可选）：按 `## 标题` 分节的运维笔记，展示在健康页。
