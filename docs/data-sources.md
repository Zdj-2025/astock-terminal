# 数据源说明

| 档 | 来源 | 成本 | 点亮的功能 | 说明 |
|---|---|---|---|---|
| 0 | demo 合成数据 | 免费 | 全部界面 | `make demo` |
| 1 | tushare Pro | 免费注册送积分 | 日线下载/缠论日线/综合评分/财报季历 | [注册](https://tushare.pro/user/register)，token 填 `.env` 的 `TUSHARE_TOKEN`。积分越高频次/接口越多（基础 120 点够日线） |
| 2 | akshare | 免费 | 缠论60分钟/部分实时行情 | `pip install -r requirements-full.txt`；免费爬取接口，偶发限流，重试即可 |
| 3 | 已有 parquet | — | 全部本地功能 | 若你已有 `data/store/kline/<code>/D.parquet`（列：date,open,high,low,close,volume[,amount,change_pct]，date 为 YYYYMMDD，前复权），直接可用 |

数据落盘位置：`data/store/kline/<code>/D.parquet`、`data/store/scan_history.parquet` 等，
全部本地、随时可备份迁移。
