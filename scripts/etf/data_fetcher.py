#!/usr/bin/env python3
"""
ETF数据获取器 - 专门处理ETF基金数据
使用AKShare的fund_etf_*系列接口
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import time


class ETFDataFetcher:
    """
    ETF数据获取器

    数据源: AKShare fund_etf_* 系列接口
    支持: 历史K线、实时行情、基金信息、折溢价率

    示例:
        fetcher = ETFDataFetcher()
        kline = fetcher.get_etf_kline('512760')
        quote = fetcher.get_etf_quote('512760')
    """

    def __init__(self, cache_dir: str = 'data/etf_cache', enable_cache: bool = True):
        """
        初始化ETF数据获取器

        参数:
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache
        self.request_delay = 0.5  # 请求间隔，避免限流
        self.last_request_time = 0

        print("✅ ETFDataFetcher初始化成功")
        print(f"   - 缓存目录: {self.cache_dir}")
        print(f"   - 数据源: AKShare fund_etf_* 接口")

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
        except Exception as e:
            print(f"⚠️  缓存读取失败: {e}")
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

    def get_etf_kline(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = 'daily',
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取ETF历史K线数据

        参数:
            symbol: ETF代码（6位数字，如512760）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            period: 周期（daily/weekly/monthly）
            adjust: 复权类型（qfq前复权/hfq后复权/空不复权）

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

        # 尝试从缓存读取
        cache_key = f"kline_{symbol}_{start_date}_{end_date}_{period}_{adjust}"
        cached = self._get_cache(cache_key, max_age=3600)
        if cached:
            print(f"📦 从缓存读取 {symbol} K线数据")
            return pd.DataFrame(cached['data'])

        print(f"📈 获取 {symbol} ETF历史K线 ({start_date} 至 {end_date})...")

        # 尝试多个数据源
        df = pd.DataFrame()

        # 尝试接口1: 东方财富
        try:
            print(f"   尝试东方财富接口...")
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
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
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_change',
                    '涨跌额': 'change'
                }
                df = df.rename(columns=column_mapping)
                df['date'] = pd.to_datetime(df['date'])

                # 保存缓存
                self._save_cache(cache_key, {'data': df.to_dict('records')})

                print(f"   ✅ 成功（东方财富）：{len(df)}条记录")
                return df[['date', 'open', 'high', 'low', 'close', 'volume', 'pct_change']]

        except Exception as e:
            print(f"   ❌ 东方财富接口失败: {str(e)[:50]}")

        # 尝试接口2: 新浪
        try:
            print(f"   尝试新浪接口...")
            df = ak.fund_etf_hist_sina(symbol=symbol)

            if not df.empty:
                # 筛选日期范围
                df['date'] = pd.to_datetime(df['date'])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

                if not df.empty:
                    # 保存缓存
                    self._save_cache(cache_key, {'data': df.to_dict('records')})

                    print(f"   ✅ 成功（新浪）：{len(df)}条记录")
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"   ❌ 新浪接口失败: {str(e)[:50]}")

        # 所有接口都失败
        print(f"❌ {symbol} 所有接口都无法获取ETF K线")
        return pd.DataFrame()

    def get_etf_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        获取ETF实时行情

        参数:
            symbol: ETF代码（6位数字）

        返回:
            实时行情字典
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)

        print(f"📊 获取 {symbol} ETF实时行情...")

        try:
            # 获取所有ETF实时行情
            df = ak.fund_etf_spot_em()

            # 筛选指定ETF
            etf_data = df[df['代码'] == symbol]

            if not etf_data.empty:
                row = etf_data.iloc[0]

                data = {
                    'symbol': symbol,
                    'name': row['名称'],
                    'price': float(row['最新价']),
                    'change': float(row['涨跌额']),
                    'change_percent': float(row['涨跌幅']),
                    'volume': int(row['成交量']) if '成交量' in row else 0,
                    'amount': float(row['成交额']) if '成交额' in row else 0,
                    'high': float(row['最高']),
                    'low': float(row['最低']),
                    'open': float(row['今开']),
                    'prev_close': float(row['昨收']),
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'eastmoney'
                }

                print(f"   ✅ 成功: {data['name']} ¥{data['price']:.3f} ({data['change_percent']:+.2f}%)")
                return data

        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}")

        return None

    def get_etf_info(self, symbol: str) -> Optional[Dict]:
        """
        获取ETF基金信息

        参数:
            symbol: ETF代码（6位数字）

        返回:
            ETF基金信息字典
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)

        # 尝试从缓存读取
        cache_key = f"info_{symbol}"
        cached = self._get_cache(cache_key, max_age=86400)  # 1天缓存
        if cached:
            print(f"📦 从缓存读取 {symbol} 基金信息")
            return cached

        print(f"📋 获取 {symbol} ETF基金信息...")

        try:
            # 获取基金信息
            nav_start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            nav_end = datetime.now().strftime('%Y%m%d')
            df = ak.fund_etf_fund_info_em(fund=symbol, start_date=nav_start, end_date=nav_end)

            if not df.empty:
                # 数据按日期升序，最后一行是最新的
                latest = df.iloc[-1]

                data = {
                    'symbol': symbol,
                    'nav': float(latest.get('单位净值', 0)),
                    'nav_date': latest.get('净值日期', ''),
                    'accumulated_nav': float(latest.get('累计净值', 0)),
                    'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # 保存缓存
                self._save_cache(cache_key, data)

                print(f"   ✅ 成功: NAV={data['nav']:.3f}")
                return data

        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}")

        return None

    def calculate_premium_discount_rate(
        self,
        price: float,
        nav: float
    ) -> float:
        """
        计算ETF折溢价率

        参数:
            price: ETF市场价格
            nav: ETF净值

        返回:
            折溢价率（百分比）
        """
        if nav == 0:
            return 0.0

        premium_rate = (price - nav) / nav * 100
        return premium_rate

    def get_etf_premium_discount(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        获取ETF历史折溢价率数据

        参数:
            symbol: ETF代码
            days: 回溯天数

        返回:
            折溢价率DataFrame
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)

        print(f"📊 计算 {symbol} ETF折溢价率（最近{days}天）...")

        try:
            # 获取ETF价格数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days+10)).strftime('%Y%m%d')

            price_df = self.get_etf_kline(symbol, start_date=start_date, end_date=end_date)

            if price_df.empty:
                print(f"   ❌ 无法获取价格数据")
                return None

            # 获取净值数据
            nav_start = (datetime.now() - timedelta(days=days+30)).strftime('%Y%m%d')
            nav_end = datetime.now().strftime('%Y%m%d')
            info_df = ak.fund_etf_fund_info_em(fund=symbol, start_date=nav_start, end_date=nav_end)

            if info_df is None or info_df.empty:
                print(f"   ❌ 无法获取净值数据")
                return None

            # 准备日期列用于合并
            price_df['date'] = pd.to_datetime(price_df['date'])
            info_df['净值日期'] = pd.to_datetime(info_df['净值日期'])

            # 确保单位净值是数值类型
            info_df['单位净值'] = pd.to_numeric(info_df['单位净值'], errors='coerce')

            merged = pd.merge(
                price_df,
                info_df[['净值日期', '单位净值']],
                left_on='date',
                right_on='净值日期',
                how='left'
            )

            merged['nav'] = merged['单位净值']
            merged['premium_rate'] = merged.apply(
                lambda row: self.calculate_premium_discount_rate(row['close'], row['nav'])
                if pd.notna(row['nav']) and row['nav'] > 0 else 0.0,
                axis=1
            )

            # 取最近N天
            result = merged.tail(days)[['date', 'close', 'nav', 'premium_rate']]

            print(f"   ✅ 成功计算 {len(result)} 天折溢价率")
            return result

        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}")
            return None

    def get_etf_iopv_quote(self, symbol: str) -> Optional[Dict]:
        """
        从实时行情获取 IOPV 实时估值和折溢价率

        比调用 fund_etf_fund_info_em 更快更直接，
        直接从 fund_etf_spot_em 提取。

        参数:
            symbol: ETF代码 (6位)

        返回:
            {'symbol', 'name', 'price', 'iopv', 'premium_rate',
             'change_pct', 'volume', 'amount', ...}
        """
        self._rate_limit()
        symbol = str(symbol).zfill(6)
        print(f"📊 获取 {symbol} IOPV实时估值...")

        try:
            df = ak.fund_etf_spot_em()
            row = df[df['代码'] == symbol]

            if row.empty:
                print(f"   ⚠️  未找到 {symbol}")
                return None

            r = row.iloc[0]
            data = {
                'symbol': symbol,
                'name': str(r.get('名称', '')),
                'price': float(r.get('最新价', 0)),
                'iopv': float(r.get('IOPV实时估值', 0)),
                'premium_rate': float(r.get('基金折价率', 0)),
                'change_pct': float(r.get('涨跌幅', 0)),
                'volume': int(r.get('成交量', 0)) if r.get('成交量') else 0,
                'amount': float(r.get('成交额', 0)) if r.get('成交额') else 0,
                'total_market_cap': float(r.get('总市值', 0)) if r.get('总市值') else 0,
                'latest_shares': float(r.get('最新份额', 0)) if r.get('最新份额') else 0,
                'time': str(r.get('更新时间', ''))
            }

            print(f"   ✅ 价格={data['price']:.3f} IOPV={data['iopv']:.3f} "
                  f"折溢价={data['premium_rate']:.2f}%")
            return data

        except Exception as e:
            print(f"   ❌ IOPV获取失败: {str(e)[:50]}")
            return None

    def get_multiple_etf_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        批量获取ETF实时行情

        参数:
            symbols: ETF代码列表

        返回:
            行情数据DataFrame
        """
        self._rate_limit()

        try:
            print(f"📊 批量获取 {len(symbols)} 只ETF行情...")
            df = ak.fund_etf_spot_em()

            # 筛选指定ETF
            symbols_str = [str(s).zfill(6) for s in symbols]
            result = df[df['代码'].isin(symbols_str)].copy()

            if not result.empty:
                print(f"   ✅ 成功获取 {len(result)} 只ETF")
                return result
            else:
                print(f"   ❌ 未找到任何ETF")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ 批量获取失败: {e}")
            return pd.DataFrame()

    def test_connection(self) -> Dict:
        """
        测试ETF数据接口连接状态

        返回:
            测试结果字典
        """
        print("\n" + "=" * 60)
        print("🔍 测试ETF数据接口连接状态")
        print("=" * 60)

        results = {}

        # 测试1: 获取ETF列表
        print("\n【测试1】获取ETF列表")
        try:
            df = ak.fund_etf_spot_em()
            results['etf_list'] = {
                'status': '✅ 正常' if not df.empty else '❌ 失败',
                'count': len(df)
            }
            print(f"   ✅ 成功获取 {len(df)} 只ETF")
        except Exception as e:
            results['etf_list'] = {'status': f'❌ 失败: {str(e)[:30]}', 'count': 0}
            print(f"   ❌ 失败: {str(e)[:50]}")

        # 测试2: 获取芯片ETF历史K线
        print("\n【测试2】获取ETF历史K线")
        test_symbol = '512760'
        try:
            df = self.get_etf_kline(test_symbol)
            results['etf_kline'] = {
                'status': '✅ 正常' if not df.empty else '❌ 失败',
                'count': len(df)
            }
        except Exception as e:
            results['etf_kline'] = {'status': f'❌ 失败: {str(e)[:30]}', 'count': 0}

        # 测试3: 获取ETF实时行情
        print("\n【测试3】获取ETF实时行情")
        try:
            quote = self.get_etf_realtime_quote(test_symbol)
            results['etf_quote'] = {
                'status': '✅ 正常' if quote else '❌ 失败',
                'sample': quote['name'] if quote else '无'
            }
        except Exception as e:
            results['etf_quote'] = {'status': f'❌ 失败: {str(e)[:30]}', 'sample': '无'}

        # 测试4: 获取ETF基金信息
        print("\n【测试4】获取ETF基金信息")
        try:
            info = self.get_etf_info(test_symbol)
            results['etf_info'] = {
                'status': '✅ 正常' if info else '❌ 失败',
                'nav': info['nav'] if info else 0
            }
        except Exception as e:
            results['etf_info'] = {'status': f'❌ 失败: {str(e)[:30]}', 'nav': 0}

        # 打印总结
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)
        for test_name, result in results.items():
            if 'count' in result:
                print(f"{test_name:15s}: {result['status']} (数量: {result['count']})")
            elif 'sample' in result:
                print(f"{test_name:15s}: {result['status']} (样本: {result['sample']})")
            elif 'nav' in result:
                print(f"{test_name:15s}: {result['status']} (NAV: {result['nav']:.3f})")
            else:
                print(f"{test_name:15s}: {result['status']}")

        return results


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试 ETFDataFetcher 模块")
    print("=" * 60)

    fetcher = ETFDataFetcher()

    # 完整的连接测试
    fetcher.test_connection()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
