# 快速开始

## 0. 环境
Python ≥ 3.10，Node ≥ 18。克隆仓库后：

```bash
pip install -r requirements.txt
cd web/frontend && npm install && npm run build && cd ../..
```

## 1. demo 模式（30 秒看界面）
```bash
make demo    # 合成数据 → http://localhost:8601
```
demo 数据安装到 `./data ./config ./reports`（均在 .gitignore）。切换真实数据前删除这三个目录即可。

## 2. 真实数据
```bash
make init    # 向导：.env（可选 TUSHARE_TOKEN）+ 配置模板
```
- `config/portfolio.xlsx`：持仓（code/avg_cost/shares/stop_loss/position_type...）
- `config/capital.xlsx`：总资产/可用资金 键值对
- 首次下载与扫描：
```bash
PYTHONPATH=. python scripts/data/update_all_kline.py     # 全市场日线（约20-40分钟）
PYTHONPATH=. python scripts/screener/daily_scan_all.py   # 收盘扫描（约8-12分钟）
```
- 启动：`make serve`

## 3. 日常运转（crontab 示例）
```
10 16 * * 1-5  cd /path/to/astock-terminal && PYTHONPATH=. python scripts/data/update_all_kline.py
15 16 * * 1-5  cd /path/to/astock-terminal && PYTHONPATH=. python scripts/screener/daily_scan_all.py
```
看板自动展示最新信号。晨报在页面内按需生成（首次1-3分钟，当日缓存）；晚报为只读组装，秒出。

## 4. AI 解盘（可选）
复制 `.env.example` → `.env`，填任意 OpenAI 兼容服务三件套（示例含 GLM/OpenAI/Ollama），
驾驶舱页即出现可用的「AI 解读」。
