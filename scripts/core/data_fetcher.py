#!/usr/bin/env python3
"""
数据获取模块 - 基于AKShare
封装所有数据获取接口，提供统一的访问层
"""
try:  # 可选依赖：实时/分钟数据源。未安装时本模块可 import，调用相关接口会给出指引
    import akshare as ak
except ImportError:
    ak = None
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import time

def _require_akshare():
    """akshare 未安装时给出指引（可选依赖，见 requirements-full.txt）。"""
    if ak is None:
        raise RuntimeError(
            "此功能依赖 akshare（实时行情/分钟线/财务/资金流）。安装：pip install -r requirements-full.txt"
        )



class DataFetcher:
    """
    数据获取器

    封装AKShare接口，提供统一的A股数据访问

    示例:
        fetcher = DataFetcher()
        quote = fetcher.get_realtime_quote('600519')
        kline = fetcher.get_historical_kline('600519', period='daily')
    """

    def __init__(self, cache_dir: str = 'data/cache'):
        """
        初始化数据获取器

        参数:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存设置
        self.cache_duration = {
            'realtime': 0,        # 实时数据不缓存
            'kline_1min': 300,    # 1分钟K线缓存5分钟
            'kline_30min': 1800,  # 30分钟K线缓存30分钟
            'kline_daily': 3600,  # 日线缓存1小时
            'financial': 86400,   # 财务数据缓存24小时
            'fund_flow': 300      # 资金流向缓存5分钟
        }

        print("✅ DataFetcher初始化成功")

    def _get_cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """生成缓存键"""
        params = '_'.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return f"{symbol}_{data_type}_{params}.json"

    def _load_cache(self, cache_key: str, max_age: int) -> Optional[Dict]:
        """
        加载缓存数据

        参数:
            cache_key: 缓存键
            max_age: 最大缓存时长（秒）

        返回:
            缓存数据或None
        """
        if max_age == 0:
            return None

        cache_file = self.cache_dir / cache_key
        if not cache_file.exists():
            return None

        # 检查缓存是否过期
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > max_age:
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  缓存读取失败: {e}")
            return None

    def _save_cache(self, cache_key: str, data: Dict):
        """保存缓存数据"""
        cache_file = self.cache_dir / cache_key
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

    def get_realtime_quote(self, symbol: str) -> Dict:
        _require_akshare()
        """
        获取实时行情

        参数:
            symbol: 股票代码（6位数字）

        返回:
            {
                'symbol': '600519',
                'name': '贵州茅台',
                'price': 1680.00,
                'change': 25.50,
                'change_percent': 1.54,
                'volume': 1234567,
                'amount': 2073548090,
                'high': 1690.00,
                'low': 1670.00,
                'open': 1675.00,
                'prev_close': 1654.50,
                'time': '2026-04-04 15:00:00'
            }
        """
        try:
            # 使用akshare获取实时行情
            df = ak.stock_zh_a_spot_em()

            # 查找指定股票
            stock = df[df['代码'] == symbol]
            if stock.empty:
                raise ValueError(f"股票代码 {symbol} 不存在")

            row = stock.iloc[0]

            # 提取数据
            data = {
                'symbol': symbol,
                'name': row['名称'],
                'price': float(row['最新价']),
                'change': float(row['涨跌额']),
                'change_percent': float(row['涨跌幅']),
                'volume': int(row['成交量']),
                'amount': float(row['成交额']),
                'high': float(row['最高']),
                'low': float(row['最低']),
                'open': float(row['今开']),
                'prev_close': float(row['昨收']),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return data

        except Exception as e:
            print(f"❌ 获取实时行情失败: {e}")
            return None

    def get_historical_kline(
        self,
        symbol: str,
        period: str = 'daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        _require_akshare()
        """
        获取历史K线数据

        参数:
            symbol: 股票代码
            period: 周期（daily/weekly/monthly/30min/5min/1min）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            adjust: 复权类型（qfq前复权/hfq后复权/空不复权）

        返回:
            DataFrame包含列：date, open, high, low, close, volume
        """
        try:
            # 设置默认日期范围
            if not end_date:
                end_date = datetime.now().strftime('%Y%m%d')
            if not start_date:
                # 默认获取过去1年数据
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

            # 周期映射(含60min修复"不支持的周期"bug, 2026-06-17)
            HIST_PERIODS = {'daily': 'daily', 'weekly': 'weekly', 'monthly': 'monthly'}
            MIN_PERIODS = {'60min': '60', '30min': '30', '5min': '5', '1min': '1'}  # akshare要数字串
            df = None
            if period in HIST_PERIODS:
                for attempt in range(3):  # 重试3次(网络RemoteDisconnected)
                    try:
                        df = ak.stock_zh_a_hist(symbol=symbol, period=HIST_PERIODS[period],
                                                start_date=start_date, end_date=end_date, adjust=adjust)
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1.0 * (attempt + 1)); continue
                        print(f"⚠️ {period}K线获取失败(重试3次): {e}")
            elif period in MIN_PERIODS:
                for attempt in range(3):
                    try:
                        df = ak.stock_zh_a_minute(symbol=symbol, period=MIN_PERIODS[period], adjust=adjust)
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1.0 * (attempt + 1)); continue
                        print(f"⚠️ {period}分钟线获取失败(重试3次): {e}")
                if df is not None and not df.empty:
                    df['datetime'] = pd.to_datetime(df['day'])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date) + timedelta(days=1)
                    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
            else:
                raise ValueError(f"不支持的周期: {period}")

            # akshare失败→tushare降级(2026-06-17, 修akshare持续RemoteDisconnected限流)
            if df is None or df.empty:
                try:
                    from scripts.core.tushare_client import TushareClient
                    tc = TushareClient()
                    _freq = {'daily':'D','weekly':'W','monthly':'M','60min':'60','30min':'30','5min':'5','1min':'1'}.get(period, 'D')
                    df_ts = tc.get_kline(symbol, freq=_freq, start=start_date, end=end_date)
                    if df_ts is not None and not df_ts.empty:
                        df = df_ts
                        print(f"⚠️ {period}降级tushare")
                except Exception as te:
                    print(f"❌ {period} tushare降级失败: {te}")
            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                'day': 'date'
            }
            df = df.rename(columns=column_mapping)

            # 确保必要列存在
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"数据缺少必要列，当前列: {df.columns.tolist()}")

            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            return df[required_cols]

        except Exception as e:
            print(f"❌ 获取历史K线失败: {e}")
            return pd.DataFrame()

    def get_financial_data(self, symbol: str) -> Dict:
        _require_akshare()
        """
        获取财务数据

        参数:
            symbol: 股票代码

        返回:
            {
                'pe': 25.3,
                'pb': 8.5,
                'roe': 30.2,
                'debt_ratio': 23.5,
                'revenue_growth': 12.5,
                'profit_growth': 15.2,
                'gross_margin': 91.5,
                'net_margin': 52.3
            }
        """
        try:
            # 获取财务指标(重试3次, 2026-06-17修网络失败)
            df = None
            for attempt in range(3):
                try:
                    df = ak.stock_financial_analysis_indicator(symbol=symbol)
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1.0 * (attempt + 1)); continue
                    print(f"⚠️ akshare财务失败(重试3次): {e}")

            if df is None or df.empty:
                # 降级: tushare daily_basic(仅PE/PB/换手, 2026-06-17)
                try:
                    from scripts.core.tushare_client import TushareClient
                    tc = TushareClient()
                    _td = tc.get_recent_trade_day()
                    basic = tc.get_daily_basic(trade_date=_td, ts_code=symbol)  # 纯数字, 内部转ts_code
                    if basic is None or basic.empty:  # 今天盘中可能空, 试前一交易日
                        basic = tc.get_daily_basic(trade_date=tc.get_recent_trade_day(n=2), ts_code=symbol)
                    if basic is not None and not basic.empty:
                        row = basic.iloc[0]
                        print("⚠️ 财务降级tushare_basic(仅PE/PB)")
                        return {
                            'symbol': symbol,
                            'pe': float(row.get('pe') or 0), 'pb': float(row.get('pb') or 0),
                            'roe': 0, 'debt_ratio': 0, 'revenue_growth': 0, 'profit_growth': 0,
                            'gross_margin': 0, 'net_margin': 0,
                            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'data_source': 'tushare_basic(降级)', 'data_complete': False,
                        }
                except Exception as e2:
                    print(f"❌ 财务数据全失败: akshare+tushare降级均失败")
                return None

            # 提取最新数据
            latest = df.iloc[0]

            data = {
                'symbol': symbol,
                'pe': float(latest.get('市盈率', 0)),
                'pb': float(latest.get('市净率', 0)),
                'roe': float(latest.get('净资产收益率', 0)),
                'debt_ratio': float(latest.get('资产负债率', 0)),
                'revenue_growth': float(latest.get('营业收入同比增长率', 0)),
                'profit_growth': float(latest.get('净利润同比增长率', 0)),
                'gross_margin': float(latest.get('销售毛利率', 0)),
                'net_margin': float(latest.get('销售净利率', 0)),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_source': 'akshare', 'data_complete': True,
            }

            return data

        except Exception as e:
            print(f"❌ 获取财务数据失败: {e}")
            return None

    def get_fund_flow(self, symbol: str, market: str = 'sh') -> Dict:
        _require_akshare()
        """
        获取资金流向数据

        参数:
            symbol: 股票代码
            market: 市场代码（sh/sz）

        返回:
            {
                'main_inflow': 12345678,
                'main_outflow': 9876543,
                'main_net': 2469135,
                'retail_net': -1234567,
                'date': '2026-04-04'
            }
        """
        try:
            # 判断市场
            if symbol.startswith('6'):
                market = 'sh'
            else:
                market = 'sz'

            # 获取个股资金流
            df = ak.stock_individual_fund_flow(stock=symbol, market=market)

            if df.empty:
                raise ValueError(f"无法获取 {symbol} 的资金流向数据")

            # 提取最新数据
            latest = df.iloc[0]

            data = {
                'symbol': symbol,
                'main_inflow': float(latest.get('主力流入', 0)),
                'main_outflow': float(latest.get('主力流出', 0)),
                'main_net': float(latest.get('主力净流入', 0)),
                'retail_net': float(latest.get('散户净流入', 0)),
                'date': str(latest.get('日期', '')),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return data

        except Exception as e:
            print(f"❌ 获取资金流向失败: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Dict:
        _require_akshare()
        """
        获取股票基本信息

        参数:
            symbol: 股票代码

        返回:
            {
                'symbol': '600519',
                'name': '贵州茅台',
                'industry': '白酒',
                'market': '上海证券交易所',
                'list_date': '2001-08-27'
            }
        """
        try:
            # 获取股票信息
            df = ak.stock_info_a_code_name()

            stock = df[df['code'] == symbol]
            if stock.empty:
                raise ValueError(f"股票代码 {symbol} 不存在")

            row = stock.iloc[0]

            data = {
                'symbol': symbol,
                'name': row['name'],
                'market': '上海证券交易所' if symbol.startswith('6') else '深圳证券交易所'
            }

            return data

        except Exception as e:
            print(f"❌ 获取股票信息失败: {e}")
            return None

    def get_multiple_quotes(self, symbols: List[str]) -> pd.DataFrame:
        _require_akshare()
        """
        批量获取实时行情

        参数:
            symbols: 股票代码列表

        返回:
            DataFrame包含所有股票的实时行情
        """
        try:
            # 获取全市场行情
            df = ak.stock_zh_a_spot_em()

            # 筛选指定股票
            result = df[df['代码'].isin(symbols)].copy()

            # 标准化列名
            column_mapping = {
                '代码': 'symbol',
                '名称': 'name',
                '最新价': 'price',
                '涨跌额': 'change',
                '涨跌幅': 'change_percent',
                '成交量': 'volume',
                '成交额': 'amount',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'prev_close'
            }
            result = result.rename(columns=column_mapping)

            return result

        except Exception as e:
            print(f"❌ 批量获取行情失败: {e}")
            return pd.DataFrame()


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 DataFetcher 模块")
    print("=" * 60)

    fetcher = DataFetcher()

    # 测试1: 获取实时行情
    print("\n【测试1】获取贵州茅台实时行情")
    print("-" * 60)
    quote = fetcher.get_realtime_quote('600519')
    if quote:
        print(f"股票: {quote['name']} ({quote['symbol']})")
        print(f"价格: ¥{quote['price']:.2f}")
        print(f"涨跌: {quote['change']:+.2f} ({quote['change_percent']:+.2f}%)")
        print(f"成交量: {quote['volume']:,}")
        print(f"时间: {quote['time']}")

    # 测试2: 获取历史K线
    print("\n【测试2】获取贵州茅台历史K线（日线）")
    print("-" * 60)
    kline = fetcher.get_historical_kline('600519', period='daily')
    if not kline.empty:
        print(f"获取到 {len(kline)} 条记录")
        print(f"日期范围: {kline['date'].iloc[0]} 至 {kline['date'].iloc[-1]}")
        print("\n最近5天数据:")
        print(kline.tail())

    # 测试3: 获取财务数据
    print("\n【测试3】获取贵州茅台财务数据")
    print("-" * 60)
    financial = fetcher.get_financial_data('600519')
    if financial:
        print(f"PE: {financial['pe']:.2f}")
        print(f"PB: {financial['pb']:.2f}")
        print(f"ROE: {financial['roe']:.2f}%")
        print(f"营收增长: {financial['revenue_growth']:.2f}%")
        print(f"利润增长: {financial['profit_growth']:.2f}%")

    # 测试4: 获取资金流向
    print("\n【测试4】获取贵州茅台资金流向")
    print("-" * 60)
    fund_flow = fetcher.get_fund_flow('600519')
    if fund_flow:
        print(f"主力净流入: ¥{fund_flow['main_net']:,.0f}")
        print(f"散户净流入: ¥{fund_flow['retail_net']:,.0f}")
        print(f"日期: {fund_flow['date']}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
