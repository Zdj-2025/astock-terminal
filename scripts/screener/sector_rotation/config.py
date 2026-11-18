"""
科技板块轮动监控系统 - 板块定义与配置 v2
核心改进：ETF成分股从基金持仓自动获取，权重前10只
"""
import tushare as ts

# 10个科技板块定义
# stocks会在运行时通过 fetch_sector_stocks() 自动补充权重前10的ETF成分股
SECTOR_DEFS = {
    "半导体": {
        "etf_primary": "512480.SH",
        "etf_backup": "512760.SH",
        "seed_stocks": ["688981.SH", "002371.SZ", "603501.SH", "688012.SH", "300782.SZ"],
    },
    "AI算力": {
        "etf_primary": "159819.SZ",
        "etf_backup": "588730.SH",
        "seed_stocks": ["688256.SH", "603019.SH", "000977.SZ", "688041.SH"],
    },
    "消费电子": {
        "etf_primary": "159732.SZ",
        "etf_backup": "159772.SZ",
        "seed_stocks": ["002475.SZ", "002241.SZ", "688036.SH"],
    },
    "新能源车": {
        "etf_primary": "515030.SH",
        "etf_backup": "515250.SH",
        "seed_stocks": ["002594.SZ", "300750.SZ", "601227.SH"],
    },
    "光伏": {
        "etf_primary": "515790.SH",
        "etf_backup": "159546.SZ",
        "seed_stocks": ["601012.SH", "600438.SH", "300274.SZ"],
    },
    "通信": {
        "etf_primary": "515880.SH",
        "etf_backup": None,
        "seed_stocks": ["300308.SZ", "300502.SZ", "300394.SZ"],
    },
    "软件信创": {
        "etf_primary": "159540.SZ",
        "etf_backup": "512720.SH",
        "seed_stocks": ["688111.SH", "600588.SH", "600536.SH"],
    },
    "机器人": {
        "etf_primary": "562500.SH",
        "etf_backup": "159551.SZ",
        "seed_stocks": ["300124.SZ", "688017.SH", "601689.SH"],
    },
    "军工": {
        "etf_primary": "512660.SH",
        "etf_backup": "512810.SH",
        "seed_stocks": ["002179.SZ", "000733.SZ", "002025.SZ"],
    },
    "生物医药": {
        "etf_primary": "515120.SH",
        "etf_backup": "159828.SZ",
        "seed_stocks": ["600276.SH", "688235.SH", "603259.SH"],
    },
}

# 运行时填充的完整板块配置
SECTORS = {}


def fetch_sector_stocks(etf_code: str, top_n: int = 10):
    """从ETF持仓获取权重前N的成分股"""
    try:
        pro = ts.pro_api()
        df = pro.fund_portfolio(ts_code=etf_code)
        if df is None or df.empty:
            return {}
        latest = df['end_date'].max()
        df_latest = df[df['end_date'] == latest].drop_duplicates('symbol').nlargest(top_n, 'stk_mkv_ratio')
        return dict(zip(df_latest['symbol'], df_latest['stk_mkv_ratio']))
    except Exception:
        return {}


def init_sectors():
    """初始化板块配置，合并ETF成分股和seed_stocks"""
    for name, defn in SECTOR_DEFS.items():
        etf_stocks = fetch_sector_stocks(defn['etf_primary'], top_n=10)
        # 合并：ETF成分股 + seed_stocks（去重）
        all_codes = list(etf_stocks.keys())
        for s in defn['seed_stocks']:
            if s not in all_codes:
                all_codes.append(s)
        
        SECTORS[name] = {
            "etf_primary": defn['etf_primary'],
            "etf_backup": defn['etf_backup'],
            "stocks": {code: code for code in all_codes},  # name用code代替，后续按需查
            "etf_weight": etf_stocks,
        }


# 评分权重 v2 — 价格量能异动为主，资金流为辅
SCORE_WEIGHTS = {
    "price_volume": 40,   # 价格+量能异动（领先指标）
    "fund_flow": 25,      # 资金流（确认指标，降权）
    "technicals": 20,     # 技术形态（f'/f''/MACD）
    "breadth": 15,        # 板块广度（上涨占比+个股强度）
}

# 信号阈值 v2 — 更灵敏
SIGNAL_THRESHOLDS = {
    "watch": {  # 🟡 关注（新增，比潜伏更早）
        "min_score": 45,
        "etf_surge": True,  # ETF单日涨幅>2%且放量
    },
    "latent": {  # 🟢 潜伏
        "min_score": 55,
        "min_fund_days": 2,       # 降低：连续2天即可（原来是3）
        "max_ret20": 8.0,         # 放宽
        "etf_vol_surge": True,    # ETF成交额环比+30%
    },
    "launch": {  # 🔵 启动
        "min_score": 65,
        "require_f_positive": True,
    },
    "accelerate": {  # 🟡 加速
        "min_score": 80,
        "min_ret20": 10.0,
        "min_up_ratio": 0.60,     # 降低
    },
    "overheat": {  # 🟠 过热
        "min_ret20": 20.0,
        "fund_net_outflow": True,
    },
    "cooldown": {  # 🔴 冷却
        "fund_outflow_days": 2,   # 降低
        "require_f_negative": True,
    },
}

TUSHARE_API_LIMIT = 180
