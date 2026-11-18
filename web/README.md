# Web 看板 v3 使用说明（FastAPI + React，只读展示层）

> 2026-11-18 晚 v3 重构：Streamlit → **FastAPI 后端 + React/Vite/ECharts 前端**，单进程
> uvicorn 常驻 :8601（launchd）。设计延续 v2.1 的交易终端系统，色板仍为 dataviz 六项
> 校验版。Streamlit 版（dashboard/）保留在仓库作备用，未启用。
> **纯只读红线**：只消费 data/store、reports/、config/ 的现有产出，不写交易数据、
> 不触发任何管道（项目宪法"减法是alpha"）。

## 架构与访问

```
web/
├── backend/
│   ├── app.py     # FastAPI：4 个页面级聚合端点 + 个股动作端点 + dist 静态托管
│   └── q.py       # 查询层（loader.py 的无 streamlit 移植，TTL+mtime 缓存）
└── frontend/      # React 18 + Vite + ECharts + lucide-react（构建产物 dist/）
    └── src/pages/ # Dashboard / Portfolio / Stock / Health
```

| 方式 | 命令 / 地址 | 说明 |
|---|---|---|
| **生产**（默认） | `http://localhost:8601` / `http://100.109.9.100:8601`（Tailscale） | launchd `com.river.stock.dashboard` 跑 uvicorn，KeepAlive |
| 开发前端 | `cd web/frontend && npm run dev` | :5173，/api 代理到 :8602 |
| 开发后端 | `PYTHONPATH=. python -m uvicorn web.backend.app:app --port 8602 --reload` | |
| 构建部署 | `cd web/frontend && npm run build` → `launchctl kickstart -k gui/$(id -u)/com.river.stock.dashboard` | dist 变更后必须重启服务（静态目录在启动时挂载） |

launchd 管理：

```bash
launchctl list | grep stock.dashboard                          # 状态
launchctl kickstart -k gui/$(id -u)/com.river.stock.dashboard  # 重启
launchctl unload ~/Library/LaunchAgents/com.river.stock.dashboard.plist
tail -50 ~/Logs/stock-dashboard.err.log                        # 排错
```

## API 端点（页面级聚合）

- `GET /api/page/dashboard` — 指标/止损/信号/累积榜/公告/推荐，一次取齐
- `GET /api/morning-report?refresh=0` — 晨报 subprocess（慢，前端按需触发）
- `GET /api/page/portfolio` — 持仓重估/profit_guard/净值三件套/三轴/流水/月报
- `GET /api/page/stock/{code}?days=` — K线+指标+波浪（250 天预热，尾部窗口）
- `GET /api/chanlun/{code}?with60=` — 缠论（tushare 网络不稳时 502 降级）
- `GET /api/comprehensive/{code}` — 四维评分（联网，前端按钮触发）
- `GET /api/realtime/{code}` — 实时报价（TTL 60s）
- `GET /api/page/health` — 水位实测/cron/health_check/STATE 节选

## 已知问题与排查

| 现象 | 处理 |
|---|---|
| 缠论/晨报/评分失败 | tushare/akshare 网络波动，页面有降级提示；K线/波浪/持仓走本地不受影响 |
| 晨报 1-3 分钟 | 正常（财报季历逐票限速），当日进程内缓存 |
| 改了前端代码没变化 | **hash 路由不重载页面**：`npm run build` 后必须 `location.reload()`（或重启服务） |
| 指标卡日期不一致 | 诚实口径：各有数据源，卡片脚注即日期 |

## 扩展

- 加页面：`frontend/src/pages/X.jsx` + `main.jsx` 路由 + `App.jsx` NAV 项；数据加
  `backend/q.py` 函数 + `app.py` 端点。
- 图表：统一走 `lib/chart.jsx` 的 Chart 封装与验证色板（勿自配色）。
- 红线：后端只读（禁 DataStore / 禁写函数），UI 无"运行扫描/更新数据"按钮。

## 环境

- 后端：`python`（fastapi/uvicorn 已装，requirements 已补）
- 前端：Node 22；依赖 `echarts / lucide-react / react-router-dom / marked / @fontsource/ibm-plex-mono`
- 端口 8601（避开 OpenClaw 18789）；Streamlit 版如需临时启用：
  `make web`（:8601 会冲突，先 `launchctl unload`）
