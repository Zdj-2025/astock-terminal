#!/usr/bin/env python3
"""
增强版数据获取模块 - 添加重试和备用接口
解决AKShare接口不稳定的问题
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import time
from functools import wraps


def retry_on_failure(max_retries=3, delay=2, backoff=2):
    """
    重试装饰器

    参数:
        max_retries: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"❌ {func.__name__} 失败，已达最大重试次数 ({max_retries})")
                        raise

                    print(f"⚠️  {func.__name__} 第{retries}次失败: {str(e)[:100]}")
                    print(f"   等待 {current_delay} 秒后重试...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None
        return wrapper
    return decorator


class EnhancedDataFetcher:
    """
    增强版数据获取器

    特点：
    - 自动重试机制
    - 多个备用接口
    - 智能错误处理
    - 数据缓存

    示例:
        fetcher = EnhancedDataFetcher()
        quote = fetcher.get_realtime_quote('600519')
        kline = fetcher.get_historical_kline('600519')
    """

    def __init__(self, cache_dir: str = 'data/cache', enable_cache: bool = True):
        """
        初始化增强版数据获取器

        参数:
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache

        # 请求间隔（避免被限流）
        self.request_delay = 1.0
        self.last_request_time = 0

        print("✅ EnhancedDataFetcher初始化成功")
        print(f"   - 缓存目录: {self.cache_dir}")
        print(f"   - 缓存状态: {'启用' if enable_cache else '禁用'}")
        print(f"   - 请求间隔: {self.request_delay}秒")

    def _rate_limit(self):
        """请求频率限制"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def _get_cache(self, cache_key: str, max_age: int = 3600) -> Optional[Dict]:
        """获取缓存"""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        # 检查缓存是否过期
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > max_age:
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def _save_cache(self, cache_key: str, data: Dict):
        """保存缓存"""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

    @retry_on_failure(max_retries=3, delay=2)
    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        获取实时行情（多接口备用）

        参数:
            symbol: 股票代码

        返回:
            实时行情字典
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)  # 确保是6位

        # 尝试接口1: 东方财富
        try:
            print(f"📊 获取 {symbol} 实时行情（接口1: 东方财富）...")
            df = ak.stock_zh_a_spot_em()

            stock = df[df['代码'] == symbol]
            if not stock.empty:
                row = stock.iloc[0]
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
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'eastmoney'
                }
                print(f"   ✅ 成功（东方财富）")
                return data
        except Exception as e:
            print(f"   ❌ 接口1失败: {str(e)[:50]}")

        # 尝试接口2: 新浪
        try:
            print(f"📊 获取 {symbol} 实时行情（接口2: 新浪）...")
            df = ak.stock_zh_a_spot()

            # 新浪接口的列名可能是 '代码' 而不是 'code'
            code_col = '代码' if '代码' in df.columns else 'code'
            stock = df[df[code_col] == symbol]

            if not stock.empty:
                row = stock.iloc[0]

                # 动态适配列名
                name_col = '名称' if '名称' in df.columns else 'name'
                price_col = '最新价' if '最新价' in df.columns else 'trade'

                data = {
                    'symbol': symbol,
                    'name': str(row[name_col]),
                    'price': float(row[price_col]) if price_col in row else 0.0,
                    'change': 0.0,
                    'change_percent': 0.0,
                    'volume': 0,
                    'amount': 0.0,
                    'high': float(row['最高']) if '最高' in row else 0.0,
                    'low': float(row['最低']) if '最低' in row else 0.0,
                    'open': float(row['今开']) if '今开' in row else 0.0,
                    'prev_close': float(row['昨收']) if '昨收' in row else 0.0,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'sina'
                }
                print(f"   ✅ 成功（新浪）")
                return data
        except Exception as e:
            print(f"   ❌ 接口2失败: {str(e)[:50]}")

        # 所有接口都失败
        print(f"❌ {symbol} 所有接口都无法获取实时行情")
        return None

    @retry_on_failure(max_retries=3, delay=2)
    def get_historical_kline(
        self,
        symbol: str,
        period: str = 'daily',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取历史K线数据（多接口备用）

        参数:
            symbol: 股票代码
            period: 周期（daily/weekly/monthly）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            adjust: 复权类型（qfq/hfq/空）

        返回:
            K线数据DataFrame
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)

        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

        # 尝试接口1: 东方财富（最常用）
        try:
            print(f"📈 获取 {symbol} 历史K线（接口1: 东方财富）...")
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=30
            )

            if not df.empty:
                # 标准化列名
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
                df = df.rename(columns=column_mapping)

                print(f"   ✅ 成功（东方财富）：{len(df)}条记录")
                return df[['date', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"   ❌ 接口1失败: {str(e)[:50]}")

        # 尝试接口2: 腾讯
        try:
            print(f"📈 获取 {symbol} 历史K线（接口2: 腾讯）...")
            # 腾讯接口使用不同的symbol格式
            if symbol.startswith('6'):
                ts_symbol = f'sh{symbol}'
            else:
                ts_symbol = f'sz{symbol}'

            df = ak.stock_zh_a_daily(symbol=ts_symbol, adjust=adjust)

            if not df.empty:
                # 筛选日期范围
                df['date'] = pd.to_datetime(df['date'])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

                print(f"   ✅ 成功（腾讯）：{len(df)}条记录")
                return df[['date', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"   ❌ 接口2失败: {str(e)[:50]}")

        # 尝试接口3: 新浪（仅日线）
        if period == 'daily':
            try:
                print(f"📈 获取 {symbol} 历史K线（接口3: 新浪）...")
                df = ak.stock_zh_a_hist_sina(symbol=symbol, adjust="qfq")

                if not df.empty:
                    # 筛选日期范围
                    df['date'] = pd.to_datetime(df['date'])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

                    print(f"   ✅ 成功（新浪）：{len(df)}条记录")
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

            except Exception as e:
                print(f"   ❌ 接口3失败: {str(e)[:50]}")

        # 所有接口都失败，返回空DataFrame
        print(f"❌ {symbol} 所有接口都无法获取历史K线")
        return pd.DataFrame()

    def get_multiple_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        批量获取实时行情

        参数:
            symbols: 股票代码列表

        返回:
            行情数据DataFrame
        """
        self._rate_limit()

        try:
            print(f"📊 批量获取 {len(symbols)} 只股票行情...")
            df = ak.stock_zh_a_spot_em()

            # 筛选指定股票
            symbols_str = [str(s).zfill(6) for s in symbols]
            result = df[df['代码'].isin(symbols_str)].copy()

            if not result.empty:
                print(f"   ✅ 成功获取 {len(result)} 只股票")
                return result
            else:
                print(f"   ❌ 未找到任何股票")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ 批量获取失败: {e}")
            return pd.DataFrame()

    @retry_on_failure(max_retries=2, delay=1)
    def get_financial_data(self, symbol: str) -> Optional[Dict]:
        """
        获取财务数据（简化版）

        参数:
            symbol: 股票代码

        返回:
            财务数据字典
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)

        try:
            print(f"💰 获取 {symbol} 财务数据...")
            df = ak.stock_financial_analysis_indicator(symbol=symbol)

            if not df.empty:
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
                    'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                print(f"   ✅ 成功")
                return data

        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}")

        return None

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表（稳定接口）

        返回:
            股票列表DataFrame
        """
        self._rate_limit()

        try:
            print("📋 获取股票列表...")
            df = ak.stock_info_a_code_name()
            print(f"   ✅ 成功：{len(df)}只股票")
            return df
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            return pd.DataFrame()

    def test_connection(self) -> Dict:
        """
        测试所有接口连接状态

        返回:
            测试结果字典
        """
        print("\n" + "=" * 60)
        print("🔍 测试AKShare各接口连接状态")
        print("=" * 60)

        results = {}

        # 测试1: 股票列表
        print("\n【测试1】股票列表接口")
        try:
            df = self.get_stock_list()
            results['stock_list'] = {
                'status': '✅ 正常' if not df.empty else '❌ 失败',
                'count': len(df)
            }
        except Exception as e:
            results['stock_list'] = {'status': f'❌ 失败: {str(e)[:30]}', 'count': 0}

        # 测试2: 实时行情
        print("\n【测试2】实时行情接口")
        try:
            quote = self.get_realtime_quote('600519')
            results['realtime_quote'] = {
                'status': '✅ 正常' if quote else '❌ 失败',
                'sample': '贵州茅台' if quote else '无'
            }
        except Exception as e:
            results['realtime_quote'] = {'status': f'❌ 失败: {str(e)[:30]}', 'sample': '无'}

        # 测试3: 历史K线
        print("\n【测试3】历史K线接口")
        try:
            df = self.get_historical_kline('600519', period='daily', start_date='20240101', end_date='20240131')
            results['historical_kline'] = {
                'status': '✅ 正常' if not df.empty else '❌ 失败',
                'count': len(df)
            }
        except Exception as e:
            results['historical_kline'] = {'status': f'❌ 失败: {str(e)[:30]}', 'count': 0}

        # 测试4: 财务数据
        print("\n【测试4】财务数据接口")
        try:
            financial = self.get_financial_data('600519')
            results['financial_data'] = {
                'status': '✅ 正常' if financial else '❌ 失败'
            }
        except Exception as e:
            results['financial_data'] = {'status': f'❌ 失败: {str(e)[:30]}'}

        # 打印总结
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)
        for test_name, result in results.items():
            print(f"{test_name:20s}: {result['status']}")

        return results


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 EnhancedDataFetcher 模块")
    print("=" * 60)

    fetcher = EnhancedDataFetcher(enable_cache=True)

    # 完整的连接测试
    fetcher.test_connection()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
