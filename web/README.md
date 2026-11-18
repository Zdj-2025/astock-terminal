# web/ — 看板（FastAPI + React）

- 后端 `backend/app.py`：页面级聚合端点 + AI 解盘；`q.py` 查询层（只读）；`settings.py` 环境配置
- 前端 `frontend/`：React + ECharts；`npm run dev` 开发（:5173 代理 :8602），`npm run build` 产出 dist/
- 运维细节（端点清单/容错/扩展指引）见仓库根 README 与 docs/
