#!/usr/bin/env python3
"""
基于BaoStock的数据获取器
BaoStock接口稳定，数据全面，适合替代AKShare
"""
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
from pathlib import Path


class BaoStockFetcher:
    """
    基于BaoStock的数据获取器

    优势:
    - 接口稳定，无连接问题
    - 数据全面（历史K线、财务数据）
    - 支持前复权、后复权
    - 无需API密钥
    - 免费开源

    示例:
        fetcher = BaoStockFetcher()
        quote = fetcher.get_realtime_quote('603881')
        kline = fetcher.get_historical_kline('603881')
    """

    def __init__(self, auto_login: bool = True):
        """
        初始化BaoStock数据获取器

        参数:
            auto_login: 是否自动登录（默认True）
        """
        self.logged_in = False

        if auto_login:
            self.login()

        print("✅ BaoStockFetcher初始化成功")

    def login(self) -> bool:
        """登录BaoStock系统"""
        if self.logged_in:
            return True

        lg = bs.login()
        if lg.error_code == '0':
            self.logged_in = True
            print("✅ BaoStock登录成功")
            return True
        else:
            print(f"❌ BaoStock登录失败: {lg.error_msg}")
            return False

    def logout(self):
        """登出BaoStock系统"""
        if self.logged_in:
            bs.logout()
            self.logged_in = False
            print("✅ BaoStock登出成功")

    def _convert_code(self, symbol: str) -> str:
        """
        转换股票代码格式
        603881 -> sh.603881
        000858 -> sz.000858
        """
        symbol = str(symbol).zfill(6)

        # 判断市场
        if symbol.startswith('6'):
            return f'sh.{symbol}'
        else:
            return f'sz.{symbol}'

    def _convert_code_back(self, bs_code: str) -> str:
        """
        转换回标准格式
        sh.603881 -> 603881
        """
        return bs_code.split('.')[1]

    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        获取最新行情（通过最近日线数据模拟）

        参数:
            symbol: 股票代码（6位数字）

        返回:
            行情字典
        """
        if not self.logged_in:
            print("❌ 未登录BaoStock")
            return None

        bs_code = self._convert_code(symbol)

        # 获取最近7天的数据，取最新一条
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 2: 前复权
        )

        if rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                latest = df.iloc[-1]

                data = {
                    'symbol': symbol,
                    'name': '',  # BaoStock不提供股票名称，需要额外查询
                    'price': float(latest['close']),
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'volume': int(float(latest['volume'])),
                    'amount': float(latest['amount']),
                    'change_percent': float(latest['pctChg']) if latest['pctChg'] else 0.0,
                    'date': latest['date'],
                    'source': 'baostock'
                }

                print(f"✅ 获取 {symbol} 最新行情成功: ¥{data['price']:.2f}")
                return data

        print(f"❌ 获取 {symbol} 行情失败: {rs.error_msg}")
        return None

    def get_historical_kline(
        self,
        symbol: str,
        period: str = 'daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取历史K线数据

        参数:
            symbol: 股票代码（6位数字）
            period: 周期（daily/weekly/monthly）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            adjust: 复权类型（qfq前复权/hfq后复权/none不复权）

        返回:
            K线数据DataFrame
        """
        if not self.logged_in:
            print("❌ 未登录BaoStock")
            return pd.DataFrame()

        bs_code = self._convert_code(symbol)

        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # 复权标志
        adjust_flag = {
            'qfq': '2',  # 前复权
            'hfq': '1',  # 后复权
            'none': '3'  # 不复权
        }.get(adjust, '2')

        # 周期映射
        frequency = {
            'daily': 'd',
            'weekly': 'w',
            'monthly': 'm'
        }.get(period, 'd')

        print(f"📈 获取 {symbol} 历史K线 ({start_date} 至 {end_date})...")

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjust_flag
        )

        if rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)

            if not df.empty:
                # 标准化列名
                df = df.rename(columns={
                    'date': 'date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'amount': 'amount',
                    'pctChg': 'pct_change'
                })

                # 转换数据类型
                for col in ['open', 'high', 'low', 'close', 'amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

                # 转换日期
                df['date'] = pd.to_datetime(df['date'])

                print(f"   ✅ 成功获取 {len(df)} 条记录")
                return df[['date', 'open', 'high', 'low', 'close', 'volume']]

        print(f"   ❌ 失败: {rs.error_msg}")
        return pd.DataFrame()

    def get_financial_data(self, symbol: str, year: int = None, quarter: int = None) -> Optional[Dict]:
        """
        获取财务数据（盈利能力）

        参数:
            symbol: 股票代码（6位数字）
            year: 年份（默认当前年）
            quarter: 季度（1-4，默认最新）

        返回:
            财务数据字典
        """
        if not self.logged_in:
            print("❌ 未登录BaoStock")
            return None

        bs_code = self._convert_code(symbol)

        if not year:
            year = datetime.now().year

        print(f"💰 获取 {symbol} 财务数据 ({year}年)...")

        # 获取盈利能力数据
        rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter if quarter else 1)

        if rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                latest = df.iloc[0]

                data = {
                    'symbol': symbol,
                    'roe': float(latest['roeAvg']) if latest['roeAvg'] else 0.0,
                    'net_margin': float(latest['npMargin']) if latest['npMargin'] else 0.0,
                    'gross_margin': float(latest['gpMargin']) if latest['gpMargin'] else 0.0,
                    'net_profit': float(latest['netProfit']) if latest['netProfit'] else 0.0,
                    'eps': float(latest['epsTTM']) if latest['epsTTM'] else 0.0,
                    'update_time': latest['pubDate'] if latest['pubDate'] else '',
                    'source': 'baostock'
                }

                print(f"   ✅ 成功 (ROE: {data['roe']:.1%}, 净利率: {data['net_margin']:.1%})")
                return data

        print(f"   ❌ 失败: {rs.error_msg}")
        return None

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        返回:
            股票列表DataFrame
        """
        if not self.logged_in:
            print("❌ 未登录BaoStock")
            return pd.DataFrame()

        print("📋 获取股票列表...")

        rs = bs.query_stock_basic()

        if rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 只保留A股
            df = df[df['type'] == '1']  # 1: 股票
            df = df[df['status'] == '1']  # 1: 上市

            print(f"   ✅ 成功获取 {len(df)} 只股票")
            return df
        else:
            print(f"   ❌ 失败: {rs.error_msg}")
            return pd.DataFrame()

    def get_multiple_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        批量获取行情

        参数:
            symbols: 股票代码列表

        返回:
            行情数据DataFrame
        """
        results = []

        for symbol in symbols:
            quote = self.get_realtime_quote(symbol)
            if quote:
                results.append(quote)

        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()

    def __del__(self):
        """析构函数：自动登出"""
        self.logout()


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 BaoStockFetcher 模块")
    print("=" * 60)

    fetcher = BaoStockFetcher()

    # 测试1: 获取最新行情
    print("\n【测试1】获取最新行情")
    quote = fetcher.get_realtime_quote('603881')
    if quote:
        print(f"数据港最新价: ¥{quote['price']:.2f}")

    # 测试2: 获取历史K线
    print("\n【测试2】获取历史K线")
    kline = fetcher.get_historical_kline('603881', period='daily')
    if not kline.empty:
        print(f"获取到 {len(kline)} 条K线数据")
        print(kline.tail())

    # 测试3: 获取财务数据
    print("\n【测试3】获取财务数据")
    financial = fetcher.get_financial_data('603881', year=2024)
    if financial:
        print(f"ROE: {financial['roe']:.1%}")
        print(f"净利率: {financial['net_margin']:.1%}")
        print(f"毛利率: {financial['gross_margin']:.1%}")

    # 测试4: 获取股票列表
    print("\n【测试4】获取股票列表")
    stock_list = fetcher.get_stock_list()
    if not stock_list.empty:
        print(f"总计 {len(stock_list)} 只股票")
        print(stock_list.head())

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
