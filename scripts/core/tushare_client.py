#!/usr/bin/env python3
"""Tushare Pro统一数据访问层 — token管理、限速、缓存、代码格式转换。

替代AKShare作为主力数据源，解决东方财富限流封IP问题。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# 国内API不走系统代理(2026-08-24教训: macOS系统代理指向已死的7897时,
# tushare/东财全部超时,收盘扫描卡死15分钟)。tushare/waditu为国内直连,
# no_proxy='*'必须在实际requests调用前设置。
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'

try:  # 可选依赖：日线下载/缠论/综合评分等需要；未安装时本模块仍可 import
    import tushare as ts
except ImportError:
    ts = None

# ── 常量 ──────────────────────────────────────────────────────────────

def _require_ts():
    """tushare 未安装时给出安装指引。"""
    if ts is None:
        raise RuntimeError(
            "此功能依赖 tushare（日线下载/财务/板块等）。安装：pip install -r requirements-full.txt")
    return ts


def get_token() -> str:
    """读取 Tushare token：环境变量 TUSHARE_TOKEN 优先，其次「设置」页面保存的
    config/settings.local.json（不内置任何默认值）。

    注册获取：https://tushare.pro（积分决定接口权限，见 docs/data-sources.md）
    """
    from scripts.core.runtime_settings import get as _get
    tok = _get("tushare_token")
    if not tok:
        raise RuntimeError(
            "未配置 Tushare token。两种方式任选：① 看板「设置」页填入并保存；"
            "② 复制 .env.example 为 .env，填 TUSHARE_TOKEN=你的token"
            "（注册：https://tushare.pro/user/register），详见 docs/data-sources.md")
    return tok
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "tushare"
_RATE_INTERVAL = 0.22  # 秒，约280次/分钟（5000积分权限500次/分钟留余量）
_CACHE_TTL = {
    "kline_day": 3600,       # 日线缓存1小时
    "kline_min": 300,        # 分钟线缓存5分钟
    "daily_basic": 1800,     # 行情指标缓存30分钟
    "financial": 86400 * 7,  # 财务数据缓存7天
    "concept": 86400 * 3,    # 概念板块缓存3天
    "other": 86400,          # 其他缓存1天
}


# ── 代码格式转换 ──────────────────────────────────────────────────────

def to_ts_code(symbol: str) -> str:
    """纯数字 → Tushare格式: '603881' → '603881.SH'"""
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    # 去掉sh/sz前缀
    for prefix in ("SH", "SZ", "BJ"):
        if symbol.startswith(prefix):
            symbol = symbol[2:]
            break
    if symbol.startswith(("6", "9")):
        return f"{symbol}.SH"
    elif symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    elif symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    return f"{symbol}.SH"


def from_ts_code(ts_code: str) -> str:
    """Tushare格式 → 纯数字: '603881.SH' → '603881'"""
    return ts_code.split(".")[0]


def _freq_map(freq: str) -> str:
    """统一频率参数映射（Tushare pro_bar格式）"""
    m = {
        "daily": "D", "D": "D",
        "weekly": "W", "W": "W",
        "monthly": "M", "M": "M",
        "1": "1min", "5": "5min", "15": "15min",
        "30": "30min", "60": "60min",
        "1min": "1min", "5min": "5min", "15min": "15min",
        "30min": "30min", "60min": "60min",
    }
    return m.get(freq, freq)


# ── 缓存 ──────────────────────────────────────────────────────────────

class _DiskCache:
    """简单的磁盘JSON缓存"""

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prefix: str, **kw) -> Path:
        parts = "_".join(f"{k}={v}" for k, v in sorted(kw.items()) if v is not None)
        safe = "".join(c if c.isalnum() or c in ("_", "-", "=") else "_" for c in parts)
        return self._dir / f"{prefix}__{safe}.parquet"

    def get(self, prefix: str, ttl: int, **kw) -> Optional[pd.DataFrame]:
        p = self._key(prefix, **kw)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > ttl:
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def set(self, prefix: str, df: pd.DataFrame, **kw):
        p = self._key(prefix, **kw)
        try:
            df.to_parquet(p, index=False)
        except Exception:
            pass


# ── 主类 ──────────────────────────────────────────────────────────────

class TushareClient:
    """Tushare Pro统一数据访问层"""

    def __init__(self, token: Optional[str] = None):
        _require_ts()
        self._token = token or get_token()
        ts.set_token(self._token)
        self._pro = ts.pro_api()
        self._cache = _DiskCache(_CACHE_DIR)
        self._last_call = 0.0

    @property
    def pro(self):
        """暴露 Tushare pro_api 对象，供 ETF 等高级接口直接调用。"""
        return self._pro

    # ── 限速 ──────────────────────────────────────────────────────────

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < _RATE_INTERVAL:
            time.sleep(_RATE_INTERVAL - elapsed)
        self._last_call = time.time()

    def _call(self, func, **kw):
        self._throttle()
        return func(**kw)

    # ── K线数据 ───────────────────────────────────────────────────────

    def get_kline(
        self,
        symbol: str,
        freq: str = "D",
        start: Optional[str] = None,
        end: Optional[str] = None,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        """获取K线数据（统一接口）。

        Args:
            symbol: 股票代码（支持纯数字 '603881' 或 Tushare格式 '603881.SH'）
            freq: 频率 D/W/M/1/5/15/30/60
            start: 开始日期 YYYYMMDD 或 YYYY-MM-DD HH:MM:SS（分钟线）
            end: 结束日期
            adj: 复权类型 qfq/hfq/None

        Returns:
            DataFrame: date, open, high, low, close, volume, amount, change_pct
        """
        ts_code = to_ts_code(symbol)
        freq = _freq_map(freq)
        is_bar = freq in ("D", "W", "M")

        # 缓存检查
        # 日/周/月线：一股一文件的全量历史缓存（键不含start/end），内存切片。
        # 旧按窗口缓存键含start/end，滚动窗口每天变键 → 28万小文件爆炸（2026-09-09清理）。
        # Tushare按调用次数计费，全量与窄窗口同为1次调用，故规范缓存为全量历史。
        # 分钟线用量极小（走baostock本地仓），保持原窗口缓存。
        if is_bar:
            cache_ttl = _CACHE_TTL["kline_day"]
            lo = pd.to_datetime(start) if start else None
            hi = pd.to_datetime(end) if end else None
            cached = self._cache.get("kline", cache_ttl, ts_code=ts_code, freq=freq, adj=adj)
            if cached is not None and not cached.empty:
                # 覆盖检查：缓存为全量历史，仅右端不足（新交易日）时重拉
                if hi is None or cached["date"].max() >= hi:
                    mask = pd.Series(True, index=cached.index)
                    if lo is not None:
                        mask &= cached["date"] >= lo
                    if hi is not None:
                        mask &= cached["date"] <= hi
                    return cached.loc[mask].reset_index(drop=True)
            fetch_start, fetch_end = None, None  # 全量历史
        else:
            cache_ttl = _CACHE_TTL["kline_min"]
            cached = self._cache.get("kline", cache_ttl, ts_code=ts_code, freq=freq,
                                      start=str(start), end=str(end), adj=adj)
            if cached is not None:
                return cached
            # 格式化日期
            if start and freq.isdigit():
                # 分钟线用 datetime 格式
                if len(start) == 8:
                    start = f"{start[:4]}-{start[4:6]}-{start[6:8]} 09:30:00"
            if end and freq.isdigit():
                if len(end) == 8:
                    end = f"{end[:4]}-{end[4:6]}-{end[6:8]} 15:00:00"
            fetch_start, fetch_end = start, end

        df = self._call(
            ts.pro_bar,
            ts_code=ts_code,
            freq=freq,
            start_date=fetch_start,
            end_date=fetch_end,
            adj=adj if adj else None,
            asset="E",
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 统一列名
        col_map = {
            "trade_date": "date",
            "trade_time": "date",
            "vol": "volume",
            "pct_chg": "change_pct",
        }
        df = df.rename(columns=col_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 只保留标准列
        keep = ["date", "open", "high", "low", "close", "volume", "amount", "change_pct"]
        keep = [c for c in keep if c in df.columns]
        result = df[keep].copy()

        if is_bar:
            # 缓存全量，返回时按请求窗口切片
            self._cache.set("kline", result, ts_code=ts_code, freq=freq, adj=adj)
            mask = pd.Series(True, index=result.index)
            if lo is not None:
                mask &= result["date"] >= lo
            if hi is not None:
                mask &= result["date"] <= hi
            result = result.loc[mask].reset_index(drop=True)
        else:
            self._cache.set("kline", result, ts_code=ts_code, freq=freq,
                             start=str(start), end=str(end), adj=adj)
        return result

    def get_kline_batch(
        self,
        symbols: list[str],
        freq: str = "D",
        start: Optional[str] = None,
        end: Optional[str] = None,
        adj: str = "qfq",
    ) -> dict[str, pd.DataFrame]:
        """批量获取多只股票K线。返回 {symbol: DataFrame}"""
        results = {}
        for s in symbols:
            try:
                results[s] = self.get_kline(s, freq=freq, start=start, end=end, adj=adj)
            except Exception:
                results[s] = pd.DataFrame()
        return results

    # ── 行情指标 ───────────────────────────────────────────────────────

    def get_daily_basic(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取每日指标（PE/PB/换手率/市值/量比）。

        一次调用可获取全市场数据。
        """
        if trade_date:
            trade_date = trade_date.replace("-", "")
        if ts_code:
            ts_code = to_ts_code(ts_code)

        cached = self._cache.get("daily_basic", _CACHE_TTL["daily_basic"],
                                  trade_date=trade_date, ts_code=ts_code)
        if cached is not None:
            return cached

        df = self._call(self._pro.daily_basic,
                        trade_date=trade_date, ts_code=ts_code,
                        fields="ts_code,trade_date,close,turnover_rate,volume_ratio,"
                               "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
                               "total_mv,circ_mv")

        if df is not None and not df.empty:
            self._cache.set("daily_basic", df, trade_date=trade_date, ts_code=ts_code)
        return df if df is not None else pd.DataFrame()

    # ── 财务数据 ───────────────────────────────────────────────────────

    def get_financial(
        self, symbol: str, fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取财务指标（fina_indicator）。170+字段。"""
        ts_code = to_ts_code(symbol)

        cached = self._cache.get("financial", _CACHE_TTL["financial"], ts_code=ts_code)
        if cached is not None:
            return cached

        kw = {"ts_code": ts_code}
        if fields:
            kw["fields"] = fields
        else:
            kw["fields"] = (
                "ts_code,ann_date,end_date,"
                "roe,grossprofit_margin,netprofit_margin,"
                "or_yoy,q_gr_yoy,profit_yoy,op_yoy,"
                "revenue,operate_profit,total_profit,n_income,"
                "debt_to_assets,current_ratio,quick_ratio,"
                "eps,bps,cfps,undistr_profit_ps,"
                "op_yofcf,"
                "inv_turn,ar_turn"
            )

        df = self._call(self._pro.fina_indicator, **kw)
        if df is not None and not df.empty:
            self._cache.set("financial", df, ts_code=ts_code)
        return df if df is not None else pd.DataFrame()

    def get_income(self, symbol: str, period: Optional[str] = None) -> pd.DataFrame:
        """获取利润表"""
        ts_code = to_ts_code(symbol)
        kw = {"ts_code": ts_code}
        if period:
            kw["period"] = period
        df = self._call(self._pro.income, **kw)
        return df if df is not None else pd.DataFrame()

    def get_balance_sheet(self, symbol: str, period: Optional[str] = None) -> pd.DataFrame:
        """获取资产负债表"""
        ts_code = to_ts_code(symbol)
        kw = {"ts_code": ts_code}
        if period:
            kw["period"] = period
        df = self._call(self._pro.balancesheet, **kw)
        return df if df is not None else pd.DataFrame()

    def get_cashflow(self, symbol: str, period: Optional[str] = None) -> pd.DataFrame:
        """获取现金流量表"""
        ts_code = to_ts_code(symbol)
        kw = {"ts_code": ts_code}
        if period:
            kw["period"] = period
        df = self._call(self._pro.cashflow, **kw)
        return df if df is not None else pd.DataFrame()

    # ── 概念板块 ───────────────────────────────────────────────────────

    def get_concepts(self) -> pd.DataFrame:
        """获取所有概念板块列表"""
        cached = self._cache.get("concepts", _CACHE_TTL["concept"])
        if cached is not None:
            return cached

        df = self._call(self._pro.concept, fields="code,name")
        if df is not None and not df.empty:
            self._cache.set("concepts", df)
        return df if df is not None else pd.DataFrame()

    def get_concept_stocks(self, concept_id: str) -> pd.DataFrame:
        """获取概念板块成分股"""
        cached = self._cache.get("concept_stocks", _CACHE_TTL["concept"],
                                  concept_id=concept_id)
        if cached is not None:
            return cached

        df = self._call(self._pro.concept_detail, id=concept_id,
                        fields="ts_code,name")
        if df is not None and not df.empty:
            self._cache.set("concept_stocks", df, concept_id=concept_id)
        return df if df is not None else pd.DataFrame()

    def search_concepts(self, keyword: str) -> pd.DataFrame:
        """按关键词搜索概念板块"""
        concepts = self.get_concepts()
        if concepts.empty:
            return concepts
        mask = concepts["name"].str.contains(keyword, case=False, na=False)
        return concepts[mask]

    # ── 特色数据 ───────────────────────────────────────────────────────

    def get_moneyflow(
        self,
        symbol: Optional[str] = None,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取个股/日期资金流向"""
        kw = {}
        if symbol:
            kw["ts_code"] = to_ts_code(symbol)
        if trade_date:
            kw["trade_date"] = trade_date.replace("-", "")
        # 默认取最近一周数据
        if not start_date and not trade_date:
            start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        if start_date:
            kw["start_date"] = start_date.replace("-", "")
        if end_date:
            kw["end_date"] = end_date.replace("-", "")
        df = self._call(self._pro.moneyflow, **kw)
        return df if df is not None else pd.DataFrame()

    def get_shareholders(self, symbol: str) -> pd.DataFrame:
        """获取前十大股东"""
        ts_code = to_ts_code(symbol)
        df = self._call(self._pro.top10_holders, ts_code=ts_code)
        return df if df is not None else pd.DataFrame()

    # 外资/机构关键词
    _FOREIGN_KW = [
        'MORGAN', 'GOLDMAN', 'SACHS', 'J.P.', 'JP ', 'UBS', 'SCHRODER',
        'BARCLAYS', 'CITIGROUP', 'BNP', 'NORGES', 'FIDELITY', 'VANGUARD',
        'BLACKROCK', 'BERKSHIRE', 'FMR', 'STATE STREET', 'ALLIANCE',
        'ABU DHABI', 'KUWAIT', 'TEMASEK', 'GIC', 'NATIONAL COUNCIL',
    ]

    def get_foreign_holders(self, symbol: str) -> dict:
        """获取最近两期前十大股东中的外资变动

        Returns:
            {
                'latest_period': '20260331',
                'latest_ann_date': '20260428',
                'prev_period': '20251231',
                'foreign_count': 4,       # 本期外资家数
                'foreign_total_ratio': 3.03,  # 本期外资合计占比%
                'prev_foreign_count': 2,  # 上期外资家数
                'details': [
                    {'name': '高盛...', 'ratio': 1.15, 'prev_ratio': 0.36, 'change': '+219%'},
                    {'name': '摩根士丹利...', 'ratio': 0.82, 'prev_ratio': None, 'change': '新进'},
                    ...
                ],
                'signal': 'foreign_increase'  # foreign_increase/foreign_decrease/foreign_stable/none
            }
        """
        df = self.get_shareholders(symbol)
        if df is None or df.empty:
            return {'signal': 'none', 'details': [], 'foreign_count': 0,
                    'foreign_total_ratio': 0}

        periods = sorted(df['end_date'].unique(), reverse=True)
        if len(periods) < 1:
            return {'signal': 'none', 'details': [], 'foreign_count': 0,
                    'foreign_total_ratio': 0}

        def _is_foreign(name):
            upper = name.upper()
            # 中文外资名
            if any(kw in name for kw in ['高盛', '摩根士丹利', '摩根大通', '瑞银', '巴克莱']):
                return True
            return any(kw in upper for kw in self._FOREIGN_KW)

        def _extract_foreign(sub_df):
            results = []
            for _, r in sub_df.iterrows():
                if _is_foreign(str(r['holder_name'])):
                    results.append({
                        'name': str(r['holder_name']),
                        'ratio': float(r.get('hold_ratio', 0) or 0),
                        'amount': float(r.get('hold_amount', 0) or 0),
                    })
            return results

        # 本期
        latest_period = periods[0]
        latest_sub = df[df['end_date'] == latest_period].head(10)
        latest_ann = str(latest_sub.iloc[0].get('ann_date', ''))
        latest_foreign = _extract_foreign(latest_sub)
        latest_total = sum(f['ratio'] for f in latest_foreign)

        # 上期
        prev_period = periods[1] if len(periods) > 1 else None
        prev_foreign = []
        prev_total = 0
        if prev_period:
            prev_sub = df[df['end_date'] == prev_period].head(10)
            prev_foreign = _extract_foreign(prev_sub)
            prev_total = sum(f['ratio'] for f in prev_foreign)

        # 生成变动明细
        details = []
        for f in latest_foreign:
            # 在上期找匹配（用名称前8字符模糊匹配）
            prev_match = [p for p in prev_foreign
                         if f['name'][:8] in p['name'] or p['name'][:8] in f['name']]
            if prev_match:
                prev_r = prev_match[0]['ratio']
                if prev_r > 0:
                    chg = f'{(f["ratio"]/prev_r - 1)*100:+.0f}%'
                else:
                    chg = '新进'
            else:
                prev_r = None
                chg = '新进'
            details.append({
                'name': f['name'][:20],
                'ratio': f['ratio'],
                'prev_ratio': prev_r,
                'change': chg,
            })

        # 上期有但本期不在前十的（退出）
        for p in prev_foreign:
            still_in = any(p['name'][:8] in f['name'] or f['name'][:8] in p['name']
                         for f in latest_foreign)
            if not still_in:
                details.append({
                    'name': p['name'][:20],
                    'ratio': 0,
                    'prev_ratio': p['ratio'],
                    'change': '退出',
                })

        # 信号判断
        if latest_total > prev_total * 1.3 and latest_total > 0.5:
            signal = 'foreign_increase'
        elif latest_total < prev_total * 0.7 and prev_total > 0.5:
            signal = 'foreign_decrease'
        elif latest_total > 0.5:
            signal = 'foreign_stable'
        else:
            signal = 'none'

        return {
            'latest_period': latest_period,
            'latest_ann_date': latest_ann,
            'prev_period': prev_period,
            'foreign_count': len(latest_foreign),
            'foreign_total_ratio': round(latest_total, 2),
            'prev_foreign_count': len(prev_foreign),
            'prev_foreign_total_ratio': round(prev_total, 2),
            'details': details,
            'signal': signal,
        }

    def get_forecast(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """获取业绩预告"""
        kw = {}
        if symbol:
            kw["ts_code"] = to_ts_code(symbol)
        df = self._call(self._pro.forecast, **kw)
        return df if df is not None else pd.DataFrame()

    def get_dividend(self, symbol: str) -> pd.DataFrame:
        """获取分红送股"""
        ts_code = to_ts_code(symbol)
        df = self._call(self._pro.dividend, ts_code=ts_code)
        return df if df is not None else pd.DataFrame()

    def get_pledge(self, symbol: str) -> pd.DataFrame:
        """获取股权质押"""
        ts_code = to_ts_code(symbol)
        df = self._call(self._pro.pledge_stat, ts_code=ts_code)
        return df if df is not None else pd.DataFrame()

    # ── 指数/基金 ──────────────────────────────────────────────────────

    def get_index_daily(self, ts_code: str = "000001.SH",
                        start: Optional[str] = None,
                        end: Optional[str] = None) -> pd.DataFrame:
        """获取指数日线"""
        kw = {"ts_code": ts_code}
        if start:
            kw["start_date"] = start.replace("-", "")
        if end:
            kw["end_date"] = end.replace("-", "")
        df = self._call(self._pro.index_daily, **kw)
        return df if df is not None else pd.DataFrame()

    def get_fund_daily(self, ts_code: str,
                       start: Optional[str] = None,
                       end: Optional[str] = None) -> pd.DataFrame:
        """获取基金日线（ETF等）"""
        kw = {"ts_code": ts_code}
        if start:
            kw["start_date"] = start.replace("-", "")
        if end:
            kw["end_date"] = end.replace("-", "")
        df = self._call(self._pro.fund_daily, **kw)
        return df if df is not None else pd.DataFrame()

    # ── 交易日历 ───────────────────────────────────────────────────────

    def get_trade_cal(self, start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
        """获取交易日历"""
        kw = {"exchange": "SSE"}
        if start:
            kw["start_date"] = start.replace("-", "")
        if end:
            kw["end_date"] = end.replace("-", "")
        df = self._call(self._pro.trade_cal, **kw)
        return df if df is not None else pd.DataFrame()

    def is_trade_day(self, date: Optional[str] = None) -> bool:
        """判断是否为交易日"""
        d = date or datetime.now().strftime("%Y%m%d")
        d = d.replace("-", "")
        df = self.get_trade_cal(start=d, end=d)
        if df.empty:
            return False
        return df.iloc[0].get("is_open", 0) == 1

    def get_recent_trade_day(self, n: int = 1, base_date: str = None) -> str:
        """获取最近第n个交易日（YYYYMMDD格式）。

        Args:
            n: 第n个交易日（1=最近，2=前一个）
            base_date: 参考日期 YYYYMMDD，默认今天
        """
        ref = base_date or datetime.now().strftime("%Y%m%d")
        end = ref
        start = (datetime.strptime(ref, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
        df = self.get_trade_cal(start=start, end=end)
        if df.empty:
            return end
        open_days = df[df["is_open"] == 1].sort_values("cal_date", ascending=False)
        if len(open_days) < n:
            return end
        return open_days.iloc[n - 1]["cal_date"]

    # ── 便捷查询 ───────────────────────────────────────────────────────

    def get_stock_list(self, list_status: str = "L") -> pd.DataFrame:
        """获取全A股列表。

        Returns:
            DataFrame: ts_code, symbol, name, area, industry, market, list_date
        """
        cached = self._cache.get("stock_list", _CACHE_TTL["other"],
                                  list_status=list_status)
        if cached is not None:
            return cached

        df = self._call(self._pro.stock_basic,
                        exchange="", list_status=list_status,
                        fields="ts_code,symbol,name,area,industry,market,list_date")
        if df is not None and not df.empty:
            self._cache.set("stock_list", df, list_status=list_status)
        return df if df is not None else pd.DataFrame()

    def get_industry_stocks(self, industry: str) -> pd.DataFrame:
        """按Tushare行业字段过滤股票列表"""
        all_stocks = self.get_stock_list()
        if all_stocks.empty:
            return all_stocks
        return all_stocks[all_stocks["industry"] == industry]

    def get_stock_info(self, symbol: str) -> dict:
        """获取股票基本信息（名称、行业、市场）"""
        ts_code = to_ts_code(symbol)
        df = self._call(self._pro.stock_basic, ts_code=ts_code,
                        fields="ts_code,symbol,name,area,industry,market,list_date")
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    def get_realtime_quote(self, symbol: str) -> dict:
        """获取实时行情 — AKShare实时价格 + Tushare PE/PB/市值补充。

        优先东方财富(stock_zh_a_spot_em)，备用新浪(stock_zh_a_spot)。
        盘中返回实时价格（is_realtime=True），非交易时间返回昨收。
        """
        result = {
            "symbol": symbol,
            "ts_code": to_ts_code(symbol),
            "is_realtime": False,
            "price_time": "",
        }

        # AKShare 实时价格 — 接口1: 东方财富
        price_found = False
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            stock = df[df['代码'] == symbol]
            if not stock.empty:
                row = stock.iloc[0]
                result.update({
                    "name": row.get('名称', ''),
                    "price": float(row['最新价']),
                    "change": float(row['涨跌额']),
                    "change_pct": float(row['涨跌幅']),
                    "volume": int(row['成交量']),
                    "amount": float(row['成交额']),
                    "high": float(row['最高']),
                    "low": float(row['最低']),
                    "open": float(row['今开']),
                    "prev_close": float(row['昨收']),
                })
                price_found = True
        except Exception:
            pass

        # AKShare 实时价格 — 接口2: 新浪（东方财富被限流时备用）
        if not price_found:
            try:
                import akshare as ak
                # 新浪代码格式: sh603881, sz300059, sh512480
                prefix = 'sh' if symbol.startswith(('6', '9', '51', '52', '50', '58')) else \
                         'sz' if symbol.startswith(('0', '3', '15', '16', '18')) else 'bj'
                sina_code = prefix + symbol
                df = ak.stock_zh_a_spot()
                stock = df[df['代码'] == sina_code]
                if stock.empty:
                    # ETF在新接口可能需要不同方式
                    stock = df[df['代码'].str.endswith(symbol)]
                if not stock.empty:
                    row = stock.iloc[0]
                    result.update({
                        "name": row.get('名称', ''),
                        "price": float(row['最新价']),
                        "change": float(row.get('涨跌额', 0)),
                        "change_pct": float(row.get('涨跌幅', 0)),
                        "volume": int(row.get('成交量', 0)),
                        "amount": float(row.get('成交额', 0)),
                        "high": float(row.get('最高', 0)),
                        "low": float(row.get('最低', 0)),
                        "open": float(row.get('今开', 0)),
                        "prev_close": float(row.get('昨收', 0)),
                    })
                    price_found = True
            except Exception:
                pass

        if price_found:
            now = datetime.now()
            if now.hour >= 9 and (now.hour < 15 or (now.hour == 15 and now.minute == 0)):
                result["is_realtime"] = True
            result["price_time"] = now.strftime('%Y-%m-%d %H:%M')

        # Tushare PE/PB/市值补充（T+1，盘中当天无数据则用前一日）
        try:
            trade_date = self.get_recent_trade_day()
            basic = self.get_daily_basic(trade_date=trade_date, ts_code=result["ts_code"])
            if basic.empty:
                trade_date = self.get_recent_trade_day(n=2)
                basic = self.get_daily_basic(trade_date=trade_date, ts_code=result["ts_code"])
            if not basic.empty:
                row = basic.iloc[0]
                result.update({
                    "pe": row.get("pe"),
                    "pe_ttm": row.get("pe_ttm"),
                    "pb": row.get("pb"),
                    "turnover_rate": row.get("turnover_rate"),
                    "volume_ratio": row.get("volume_ratio"),
                    "total_mv": row.get("total_mv"),
                    "circ_mv": row.get("circ_mv"),
                })
        except Exception:
            pass

        return result

    # ── 分钟线多源接口（Baostock/AKShare/Tushare自动降级） ───────────

    def get_kline_minute(
        self,
        symbol: str,
        freq: str = "60",
        start: Optional[str] = None,
        end: Optional[str] = None,
        adj: str = "qfq",
        source: str = "auto",
    ) -> pd.DataFrame:
        """获取分钟K线数据（多源自动降级）。

        数据源优先级（source='auto'时）：
          1. 本地缓存 (data/store/kline/{code}/) — 最快，离线可用
          2. Baostock — 无限流，推荐，速度~0.3s
          3. AKShare (stock_zh_a_minute) — 无限流，备选
          4. Tushare — 严格限流(1次/分)，最后手段

        Args:
            symbol: 股票代码（纯数字或ts格式）
            freq: 频率 5/15/30/60
            start: 开始日期 YYYYMMDD
            end: 结束日期 YYYYMMDD
            adj: 复权 qfq/hfq/None
            source: 'auto'(自动降级) / 'baostock' / 'akshare' / 'tushare' / 'local'

        Returns:
            DataFrame: date, open, high, low, close, volume, amount
        """
        ts_code = to_ts_code(symbol)
        pure_code = ts_code.split(".")[0]

        # 1. 本地缓存
        if source in ("auto", "local"):
            local_df = self._read_local_minute_cache(pure_code, freq)
            if local_df is not None and not local_df.empty:
                if start:
                    local_df = local_df[local_df["date"] >= pd.Timestamp(start)]
                if end:
                    local_df = local_df[local_df["date"] <= pd.Timestamp(end)]
                if source == "local" or not local_df.empty:
                    return local_df

        if source == "local":
            return pd.DataFrame()

        # 2. Baostock（无限流，速度快）
        if source in ("auto", "baostock"):
            try:
                df = self._fetch_minute_baostock(pure_code, freq, start, end, adj)
                if df is not None and not df.empty:
                    return df
            except Exception:
                if source == "baostock":
                    raise

        # 3. AKShare
        if source in ("auto", "akshare"):
            try:
                df = self._fetch_minute_akshare(pure_code, freq, start, end, adj)
                if df is not None and not df.empty:
                    return df
            except Exception:
                if source == "akshare":
                    raise

        # 4. Tushare（限流严格，最后手段）
        if source in ("auto", "tushare"):
            try:
                return self.get_kline(symbol, freq=freq, start=start, end=end, adj=adj)
            except Exception:
                if source == "tushare":
                    raise

        return pd.DataFrame()

    # ── 本地缓存读取 ────────────────────────────────────────────────

    @staticmethod
    def _read_local_minute_cache(pure_code: str, freq: str) -> Optional[pd.DataFrame]:
        """读取 data/store/kline/{code}/{freq}min.parquet。"""
        store_dir = Path(__file__).resolve().parent.parent.parent / "data" / "store" / "kline"
        path = store_dir / pure_code / f"{freq}min.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            # 统一列名
            if "datetime" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"datetime": "date"})
            if "day" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"day": "date"})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return None

    # ── Baostock 分钟线 ─────────────────────────────────────────────

    @staticmethod
    def _fetch_minute_baostock(
        pure_code: str,
        freq: str,
        start: Optional[str],
        end: Optional[str],
        adj: str,
    ) -> Optional[pd.DataFrame]:
        """Baostock分钟线（无限流，推荐首选在线源）。

        优点: 无调用限制，速度快(~0.3s)，全市场覆盖。
        缺点: 需login/logout，复权精度略低于Tushare。
        """
        import baostock as bs

        bs_code = f"sh.{pure_code}" if pure_code.startswith(("6", "9")) else f"sz.{pure_code}"
        bs_freq = {"5": "5", "15": "15", "30": "30", "60": "60"}.get(freq, "60")
        adj_flag = {"qfq": "2", "hfq": "1"}.get(adj, "3")
        # Baostock日期格式: YYYY-MM-DD
        if start:
            start_str = start if "-" in start else f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        else:
            start_str = "1990-01-01"
        if end:
            end_str = end if "-" in end else f"{end[:4]}-{end[4:6]}-{end[6:8]}"
        else:
            end_str = datetime.now().strftime("%Y-%m-%d")

        lg = bs.login()
        if lg.error_code != "0":
            return None
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,time,open,high,low,close,volume,amount",
                start_date=start_str, end_date=end_str,
                frequency=bs_freq, adjustflag=adj_flag,
            )
            if rs is None or rs.error_code != "0":
                return None
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None

            df = pd.DataFrame(rows, columns=["date_str", "time_str", "open", "high", "low", "close", "volume", "amount"])
            df["date"] = pd.to_datetime(df["time_str"].str[:14], format="%Y%m%d%H%M%S")
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
            df = df.sort_values("date").reset_index(drop=True)
            return df
        finally:
            bs.logout()

    # ── AKShare 分钟线 ──────────────────────────────────────────────

    @staticmethod
    def _fetch_minute_akshare(
        pure_code: str,
        freq: str,
        start: Optional[str],
        end: Optional[str],
        adj: str,
    ) -> Optional[pd.DataFrame]:
        """AKShare分钟线 (stock_zh_a_minute 接口，无限流)。

        注意: stock_zh_a_hist_min_em 接口在Mac Mini上连接失败(RemoteDisconnected)，
        但 stock_zh_a_minute 正常。返回近500个交易日全量数据。
        """
        import akshare as ak

        ak_code = f"sh{pure_code}" if pure_code.startswith(("6", "9")) else f"sz{pure_code}"
        df = ak.stock_zh_a_minute(symbol=ak_code, period=freq, adjust=adj)
        if df is None or df.empty:
            return None

        if "day" in df.columns:
            df = df.rename(columns={"day": "date"})
        df["date"] = pd.to_datetime(df["date"])

        if start:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["date"] < pd.Timestamp(end) + pd.Timedelta(days=1)]

        df = df.sort_values("date").reset_index(drop=True)
        return df
