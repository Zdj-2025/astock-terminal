#!/usr/bin/env python3
"""
综合技术分析器

四维加权信号框架（克罗趋势交易+缠论波浪融合）：
维度1：均线趋势(30%) — MA排列+周线方向
维度2：缠论/波浪(25%) — 结构位置+浪型
维度3：MACD双参数(25%) — 趋势方向+背离
维度4：量价配合(20%) — 量价确认+资金流向

两步分析：长周期判方向(周→日→60分)，短周期找买卖点(日→60分→15分)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


MA_PERIODS = {
    'weekly': [5, 10, 20, 60],
    'daily': [5, 10, 20, 60, 120],
    '60min': [5, 10, 20, 60, 120],
    '15min': [5, 10, 20, 60],
}

# 行业估值分档：Tushare industry → (档次, PE高估阈值, PE正常上限)
# traditional=传统低估值 | manufacturing=制造消费 | growth=科技成长
_INDUSTRY_TIERS = {
    # 传统行业 — PE正常5-15，>20高估
    'traditional': 20,
    # 制造/消费/医药 — PE正常15-30，>35高估
    'manufacturing': 35,
    # 科技/成长 — PE正常30-80，>80高估
    'growth': 80,
}
_INDUSTRY_MAP = {
    # 传统行业
    '银行': 'traditional', '保险': 'traditional', '证券': 'traditional',
    '全国地产': 'traditional', '区域地产': 'traditional', '园区开发': 'traditional', '房产服务': 'traditional',
    '火力发电': 'traditional', '水力发电': 'traditional', '新型电力': 'traditional', '供气供热': 'traditional',
    '煤炭开采': 'traditional', '石油开采': 'traditional', '石油加工': 'traditional', '石油贸易': 'traditional',
    '公路': 'traditional', '港口': 'traditional', '水运': 'traditional', '路桥': 'traditional',
    '普钢': 'traditional', '特种钢': 'traditional', '钢加工': 'traditional', '焦炭加工': 'traditional',
    '水泥': 'traditional', '玻璃': 'traditional', '陶瓷': 'traditional',
    '水务': 'traditional', '环境保护': 'traditional',
    # 制造/消费/医药
    '汽车整车': 'manufacturing', '汽车配件': 'manufacturing', '汽车服务': 'manufacturing',
    '家用电器': 'manufacturing', '家居用品': 'manufacturing', '电气设备': 'manufacturing',
    '化工原料': 'manufacturing', '塑料': 'manufacturing', '橡胶': 'manufacturing', '化纤': 'manufacturing',
    '纺织': 'manufacturing', '服饰': 'manufacturing', '日用化工': 'manufacturing',
    '专用机械': 'manufacturing', '工程机械': 'manufacturing', '机械基件': 'manufacturing',
    '轻工机械': 'manufacturing', '化工机械': 'manufacturing', '纺织机械': 'manufacturing', '机床制造': 'manufacturing',
    '建筑工程': 'manufacturing', '装修装饰': 'manufacturing', '其他建材': 'manufacturing',
    '白酒': 'manufacturing', '啤酒': 'manufacturing', '红黄酒': 'manufacturing', '软饮料': 'manufacturing', '食品': 'manufacturing', '饲料': 'manufacturing',
    '乳制品': 'manufacturing', '农林牧渔': 'manufacturing', '农业综合': 'manufacturing', '渔业': 'manufacturing', '林业': 'manufacturing', '种植业': 'manufacturing',
    '农药化肥': 'manufacturing', '农用机械': 'manufacturing',
    '化学制药': 'manufacturing', '中成药': 'manufacturing', '生物制药': 'manufacturing', '医药商业': 'manufacturing', '医疗保健': 'manufacturing',
    '出版业': 'manufacturing', '文教休闲': 'manufacturing', '广告包装': 'manufacturing', '造纸': 'manufacturing', '染料涂料': 'manufacturing',
    '百货': 'manufacturing', '超市连锁': 'manufacturing', '商品城': 'manufacturing', '商贸代理': 'manufacturing', '批发业': 'manufacturing',
    '酒店餐饮': 'manufacturing', '旅游景点': 'manufacturing', '旅游服务': 'manufacturing', '公共交通': 'manufacturing',
    '运输设备': 'manufacturing', '船舶': 'manufacturing', '航空': 'manufacturing', '空运': 'manufacturing', '机场': 'manufacturing', '铁路': 'manufacturing', '摩托车': 'manufacturing',
    '元器件': 'manufacturing', '电器仪表': 'manufacturing', '电器连锁': 'manufacturing',
    '铝': 'manufacturing', '铜': 'manufacturing', '铅锌': 'manufacturing', '小金属': 'manufacturing', '矿物制品': 'manufacturing', '黄金': 'manufacturing',
    '仓储物流': 'manufacturing', '综合类': 'manufacturing', '其他商业': 'manufacturing',
    # 科技/成长
    '软件服务': 'growth', '互联网': 'growth', 'IT设备': 'growth',
    '半导体': 'growth', '通信设备': 'growth', '电信运营': 'growth',
    '影视音像': 'growth', '多元金融': 'growth',
}


class ComprehensiveAnalyzer:
    """
    综合技术分析器 — 两步六维度框架。

    用法:
        analyzer = ComprehensiveAnalyzer('603881')
        result = analyzer.analyze()
        print(ComprehensiveAnalyzer.format_report(result))
    """

    ETF_PREFIXES = ('51', '15', '56', '58', '52', '50', '11', '12', '13', '16', '18')

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._ak_code = self._convert_code(symbol)
        self.is_etf = symbol.startswith(self.ETF_PREFIXES)

    @staticmethod
    def _convert_code(symbol: str) -> str:
        symbol = symbol.replace('sh.', '').replace('sz.', '')
        if symbol.startswith(('6', '9')):
            return 'sh' + symbol
        return 'sz' + symbol

    def analyze(self, include_15min: bool = False) -> Dict:
        """执行两步六维度综合分析。"""
        data = self._fetch_all_data(include_15min)

        if data.get('daily') is None or data['daily'].empty:
            return {'error': '无法获取日线数据', 'symbol': self.symbol}

        results = {
            'symbol': self.symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'levels': {},
            'long_term': {},
            'short_term': {},
            'advice': {},
            'is_etf': self.is_etf,
        }

        for tf in ['weekly', 'daily', '60min', '15min']:
            df = data.get(tf)
            if df is not None and len(df) > 20:
                results['levels'][tf] = self._analyze_level(df, MA_PERIODS.get(tf, [20, 60]))

        results['price'] = results['levels'].get('daily', {}).get('price', 0)

        # 实时价格替换（仅股票，ETF无实时接口）
        results['realtime'] = self._fetch_realtime_price()

        # 资金流向（Tushare — 仅股票）
        results['moneyflow'] = self._fetch_moneyflow() if not self.is_etf else {}

        # 大盘趋势（Tushare index_daily）
        results['market'] = self._fetch_market_trend()

        # 基本面速览（Tushare fina_indicator — 仅股票）
        results['fundamentals'] = self._fetch_fundamentals(results.get('realtime', {}))

        # 外资持仓信号（Tushare top10_holders — 仅股票）
        results['foreign'] = self._fetch_foreign_signal()

        results['long_term'] = self._long_term_trend(results['levels'])
        results['short_term'] = self._short_term_opportunity(results['levels'])
        results['signal_4d'] = self._score_4d(results['levels'])
        rt_price = results.get('realtime', {}).get('price', 0)
        results['key_levels'] = self._collect_key_levels(results['levels'], rt_price)
        results['advice'] = self._generate_advice(results)

        return results

    def _fetch_realtime_price(self) -> Dict:
        """获取AKShare实时价格，替换日K线最后一行的昨收。"""
        try:
            from scripts.core.tushare_client import TushareClient
            ts = TushareClient()
            rt = ts.get_realtime_quote(self.symbol)
            if rt.get('price'):
                return rt
        except Exception:
            pass
        return {}

    def _fetch_moneyflow(self) -> Dict:
        """获取Tushare资金流向。"""
        try:
            from scripts.core.tushare_client import TushareClient
            ts = TushareClient()
            df = ts.get_moneyflow(symbol=self.symbol)
            if df is not None and not df.empty:
                row = df.iloc[-1]
                return {
                    'net_mf_vol': float(row.get('net_mf_vol', 0)),         # 净流入量（手）
                    'net_mf_amount': float(row.get('net_mf_amount', 0)),   # 净流入额（元）
                    'buy_elg_vol': float(row.get('buy_elg_vol', 0)),       # 大单买入（手）
                    'sell_elg_vol': float(row.get('sell_elg_vol', 0)),     # 大单卖出（手）
                    'buy_elg_amount': float(row.get('buy_elg_amount', 0)), # 大单买入额（元）
                    'sell_elg_amount': float(row.get('sell_elg_amount', 0)),
                    'trade_date': str(row.get('trade_date', '')),
                }
        except Exception:
            pass
        return {}

    def _fetch_market_trend(self) -> Dict:
        """获取上证指数近期趋势作为大盘背景。"""
        try:
            from scripts.core.tushare_client import TushareClient
            ts = TushareClient()
            df = ts.get_index_daily(ts_code='000001.SH', start='20260101')
            if df is not None and not df.empty and len(df) >= 5:
                df = df.sort_values('trade_date')
                latest = float(df.iloc[-1]['close'])
                ma5 = float(df['close'].tail(5).mean())
                ma20 = float(df['close'].tail(20).mean()) if len(df) >= 20 else ma5
                pct_5d = (latest / float(df.iloc[-5]['close']) - 1) * 100
                trend = '上升' if latest > ma5 > ma20 else '下降' if latest < ma5 < ma20 else '震荡'
                return {'close': latest, 'ma5': ma5, 'ma20': ma20, 'pct_5d': round(pct_5d, 2), 'trend': trend}
        except Exception:
            pass
        return {}

    def _fetch_fundamentals(self, realtime: Dict) -> Dict:
        """获取基本面数据：PE/PB(来自daily_basic) + ROE/营收增长/负债率(来自fina_indicator)。"""
        if self.is_etf:
            return {}

        fundamentals = {}

        # 行业分类 — 用于分档估值阈值
        fundamentals['industry'] = ''
        fundamentals['pe_tier'] = 'manufacturing'  # 默认制造档
        try:
            from scripts.core.tushare_client import TushareClient
            ts = TushareClient()
            ts_code = f"{self.symbol}.{'SH' if self.symbol.startswith(('6','9')) else 'SZ'}"
            info = ts._call(ts._pro.stock_basic, ts_code=ts_code, fields='ts_code,industry')
            if info is not None and not info.empty:
                industry = str(info.iloc[0].get('industry', ''))
                fundamentals['industry'] = industry
                fundamentals['pe_tier'] = _INDUSTRY_MAP.get(industry, 'manufacturing')
        except Exception:
            pass

        # PE/PB/市值 — 从 realtime quote 中提取（已通过 daily_basic 获取）
        for key in ('pe', 'pe_ttm', 'pb', 'total_mv', 'turnover_rate'):
            val = realtime.get(key)
            if val is not None:
                try:
                    fundamentals[key] = float(val)
                except (TypeError, ValueError):
                    pass

        # ROE/营收增长/负债率 — 从 fina_indicator 获取
        try:
            from scripts.core.tushare_client import TushareClient
            ts = TushareClient()
            df = ts.get_financial(self.symbol)
            if df is not None and not df.empty:
                row = df.iloc[0]  # 最新一期
                fundamentals['roe'] = self._safe_float(row.get('roe'))
                fundamentals['revenue_yoy'] = self._safe_float(row.get('or_yoy'))
                fundamentals['profit_yoy'] = self._safe_float(row.get('profit_yoy'))
                fundamentals['gross_margin'] = self._safe_float(row.get('grossprofit_margin'))
                fundamentals['debt_ratio'] = self._safe_float(row.get('debt_to_assets'))
                fundamentals['eps'] = self._safe_float(row.get('eps'))
                end_date = str(row.get('end_date', ''))
                if end_date:
                    fundamentals['report_date'] = end_date
        except Exception:
            pass

        return fundamentals if fundamentals else {}

    def _fetch_foreign_signal(self) -> Dict:
        """获取外资持仓变动信号（辅助指标）。"""
        if self.is_etf:
            return {}
        try:
            from scripts.core.tushare_client import TushareClient
            ts = TushareClient()
            return ts.get_foreign_holders(self.symbol)
        except Exception:
            return {}

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _fetch_all_data(self, include_15min: bool) -> Dict[str, Optional[pd.DataFrame]]:
        from scripts.core.tushare_client import TushareClient

        data = {}

        # ── 数据获取 ──

    # 日线/周线 — ETF用 fund_daily，股票用 get_kline
        try:
            ts = TushareClient()
            if self.is_etf:
                ts_code = f"{self.symbol}.{'SH' if self.symbol.startswith(('5','11')) else 'SZ'}"
                raw = ts.get_fund_daily(ts_code=ts_code, start='20250101')
                if raw is not None and not raw.empty:
                    raw = raw.rename(columns={'trade_date': 'date', 'vol': 'volume',
                                              'pct_chg': 'change_pct'})
                    if 'date' in raw.columns:
                        raw['date'] = pd.to_datetime(raw['date'])
                    raw = raw.sort_values('date').reset_index(drop=True)
                    keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change_pct'] if c in raw.columns]
                    df = raw[keep].copy()
                    df = self._normalize(df)
                    data['daily'] = df
                    data['weekly'] = self._resample_weekly(df)
            else:
                df = ts.get_kline(self.symbol, freq='D', start='20250101', adj='qfq')
                if df is not None and not df.empty:
                    df = self._normalize(df)
                    data['daily'] = df
                    data['weekly'] = self._resample_weekly(df)
        except Exception as e:
            print(f"⚠️ 日线数据获取失败: {e}")

        # 分钟线 — AKShare备用（Tushare分钟线限2次/天）
        import akshare as ak

        for freq, period in [('60min', '60'), ('15min', '15')]:
            if freq == '15min' and not include_15min:
                continue
            try:
                df = ak.stock_zh_a_minute(symbol=self._ak_code, period=period, adjust='qfq')
                if df is not None and not df.empty:
                    data[freq] = self._normalize_minute(df)
            except Exception as e:
                print(f"⚠️ {freq}数据获取失败: {e}")

        return data

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def _normalize_minute(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {'day': 'date'}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df.tail(500).reset_index(drop=True)

    def _resample_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.isocalendar().week
        groups = df.groupby(['year', 'week'])
        rows = []
        for (_, _), g in groups:
            rows.append({
                'date': g['date'].iloc[-1],
                'open': g['open'].iloc[0],
                'high': g['high'].max(),
                'low': g['low'].min(),
                'close': g['close'].iloc[-1],
                'volume': g['volume'].sum(),
            })
        return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    # ── 单级别六维度分析 ──

    def _analyze_level(self, df: pd.DataFrame, ma_periods: List[int]) -> Dict:
        """对单个时间级别计算全部六维度指标。"""
        price = float(df['close'].iloc[-1])
        result = {'price': price, 'bars': len(df)}

        # MA
        ma_vals = {}
        for p in ma_periods:
            v = df['close'].rolling(window=p).mean().iloc[-1]
            if not np.isnan(v):
                ma_vals[p] = float(v)
        result['ma'] = ma_vals

        sorted_p = sorted(ma_vals.keys())
        if len(sorted_p) >= 3:
            bull = all(ma_vals[sorted_p[i]] > ma_vals[sorted_p[i+1]] for i in range(len(sorted_p)-1))
            bear = all(ma_vals[sorted_p[i]] < ma_vals[sorted_p[i+1]] for i in range(len(sorted_p)-1))
            result['ma_alignment'] = '多头排列' if bull else '空头排列' if bear else '混乱'
        else:
            result['ma_alignment'] = '数据不足'

        # MACD (标准 12,26,9 — 趋势方向)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = 2 * (dif - dea)

        result['dif'] = float(dif.iloc[-1])
        result['dea'] = float(dea.iloc[-1])
        result['macd_hist'] = float(hist.iloc[-1])
        result['macd_dir'] = '多' if dif.iloc[-1] > dea.iloc[-1] else '空'
        result['hist_trend'] = '红柱扩大' if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2] else \
                               '红柱缩短' if hist.iloc[-1] > 0 and hist.iloc[-1] <= hist.iloc[-2] else \
                               '绿柱扩大' if hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2] else '绿柱缩短'

        # MACD (快速 5,34,5 — 背离检测更灵敏, 短线做T更适用)
        ema5 = df['close'].ewm(span=5, adjust=False).mean()
        ema34 = df['close'].ewm(span=34, adjust=False).mean()
        dif_f = ema5 - ema34
        dea_f = dif_f.ewm(span=5, adjust=False).mean()
        hist_f = 2 * (dif_f - dea_f)

        result['dif_f'] = float(dif_f.iloc[-1])
        result['dea_f'] = float(dea_f.iloc[-1])
        result['macd_hist_f'] = float(hist_f.iloc[-1])
        result['macd_dir_f'] = '多' if dif_f.iloc[-1] > dea_f.iloc[-1] else '空'
        result['hist_trend_f'] = '红柱扩大' if hist_f.iloc[-1] > 0 and hist_f.iloc[-1] > hist_f.iloc[-2] else \
                                 '红柱缩短' if hist_f.iloc[-1] > 0 and hist_f.iloc[-1] <= hist_f.iloc[-2] else \
                                 '绿柱扩大' if hist_f.iloc[-1] < 0 and hist_f.iloc[-1] < hist_f.iloc[-2] else '绿柱缩短'

        # 背离 — 峰谷对比法（使用 5,34,5 的DIF_f检测）
        # 改进: ①金叉/死叉窗口搜索(替代固定60根) ②就近配对峰谷
        def _find_peaks(arr, distance=5):
            """纯pandas实现: 找局部极大值的索引"""
            peaks = []
            for i in range(1, len(arr) - 1):
                is_peak = True
                lo = max(0, i - distance)
                hi = min(len(arr), i + distance + 1)
                for j in range(lo, hi):
                    if j != i and arr[j] >= arr[i]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
            return peaks

        def _find_nearest_peak(peak_indices, target_idx, max_dist=10):
            """在 peak_indices 中找距 target_idx 最近的峰, 返回峰索引或 None"""
            best = None
            best_dist = max_dist + 1
            for p in peak_indices:
                d = abs(p - target_idx)
                if d < best_dist:
                    best_dist = d
                    best = p
            return best

        result['top_div'] = False
        result['bot_div'] = False
        lookback = min(120, len(df))  # 扩大初始窗口, 后续用金叉/死叉裁剪
        if lookback > 15:
            h_vals = df['high'].tail(lookback).values
            l_vals = df['low'].tail(lookback).values
            d_vals = dif_f.tail(lookback).values
            dea_vals = dea_f.tail(lookback).values

            # 改进①: 用金叉/死叉确定搜索窗口
            cross_above = []  # DIF上穿DEA (金叉)
            cross_below = []  # DIF下穿DEA (死叉)
            for i in range(1, len(d_vals)):
                if d_vals[i-1] <= dea_vals[i-1] and d_vals[i] > dea_vals[i]:
                    cross_above.append(i)
                elif d_vals[i-1] >= dea_vals[i-1] and d_vals[i] < dea_vals[i]:
                    cross_below.append(i)

            price_peaks = _find_peaks(h_vals, distance=5)
            price_troughs = _find_peaks(-l_vals, distance=5)
            macd_peaks = _find_peaks(d_vals, distance=5)
            macd_troughs = _find_peaks(-d_vals, distance=5)

            # 顶背离: 在最近一个死叉段内搜索
            if len(cross_below) >= 1:
                # 搜索范围: 从倒数第二个金叉到最新死叉(或从开头到最新死叉)
                start = cross_above[-2] if len(cross_above) >= 2 else (
                    cross_above[-1] if len(cross_above) >= 1 else 0)
                end = cross_below[-1]
                if end > start + 10:  # 窗口太窄无意义
                    pp = [p for p in price_peaks if start <= p <= end]
                    mp = [p for p in macd_peaks if start <= p <= end]
                    if len(pp) >= 2:
                        p1, p2 = pp[-2], pp[-1]
                        if h_vals[p2] > h_vals[p1]:
                            m1 = _find_nearest_peak(mp, p1)
                            m2 = _find_nearest_peak(mp, p2)
                            if m1 is not None and m2 is not None:
                                if d_vals[m2] < d_vals[m1]:
                                    result['top_div'] = True

            # 底背离: 在最近一个金叉段内搜索
            if len(cross_above) >= 1:
                start = cross_below[-2] if len(cross_below) >= 2 else (
                    cross_below[-1] if len(cross_below) >= 1 else 0)
                end = cross_above[-1]
                if end > start + 10:
                    pt = [t for t in price_troughs if start <= t <= end]
                    mt = [t for t in macd_troughs if start <= t <= end]
                    if len(pt) >= 2:
                        t1, t2 = pt[-2], pt[-1]
                        if l_vals[t2] < l_vals[t1]:
                            m1 = _find_nearest_peak(mt, t1)
                            m2 = _find_nearest_peak(mt, t2)
                            if m1 is not None and m2 is not None:
                                if d_vals[m2] > d_vals[m1]:
                                    result['bot_div'] = True

        # RSI (Wilder EMA)
        delta = df['close'].diff()
        alpha = 1.0 / 14
        gain = delta.where(delta > 0, 0).ewm(alpha=alpha, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=alpha, adjust=False).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        result['rsi'] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50

        # Bollinger
        mid = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std(ddof=0)
        upper = mid + 2 * std
        lower = mid - 2 * std
        rng = upper.iloc[-1] - lower.iloc[-1]
        result['boll_upper'] = float(upper.iloc[-1]) if not np.isnan(upper.iloc[-1]) else 0
        result['boll_mid'] = float(mid.iloc[-1]) if not np.isnan(mid.iloc[-1]) else 0
        result['boll_lower'] = float(lower.iloc[-1]) if not np.isnan(lower.iloc[-1]) else 0
        result['boll_pos'] = float((price - lower.iloc[-1]) / rng * 100) if rng > 0 and not np.isnan(rng) else 50
        prev_rng = upper.iloc[-5] - lower.iloc[-5] if len(df) > 5 else rng
        result['boll_squeeze'] = rng < prev_rng if not np.isnan(prev_rng) else False

        # KDJ
        low9 = df['low'].rolling(9).min()
        high9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low9) / (high9 - low9).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        result['k'] = float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50
        result['d'] = float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50
        result['j'] = float(j.iloc[-1]) if not np.isnan(j.iloc[-1]) else 50

        # ── 新增指标：MA动能+新鲜度+支撑位+动量 ──
        # MA价差动能
        if len(df) >= 10:
            ma5_s = df['close'].rolling(5).mean()
            ma20_s = df['close'].rolling(20).mean()
            spread = ma5_s - ma20_s
            spread_now = float(spread.iloc[-1]) if not np.isnan(spread.iloc[-1]) else 0
            spread_5d = float(spread.iloc[-6]) if len(df) > 5 and not np.isnan(spread.iloc[-6]) else spread_now
            result['ma_spread'] = spread_now
            result['ma_spread_rate'] = (spread_now - spread_5d) / max(abs(spread_5d), 0.01) if abs(spread_5d) > 0.01 else 0
        else:
            result['ma_spread'] = 0
            result['ma_spread_rate'] = 0

        # MA排列新鲜度
        if len(df) >= 60:
            ma5_s = df['close'].rolling(5).mean()
            ma10_s = df['close'].rolling(10).mean()
            ma20_s = df['close'].rolling(20).mean()
            ma60_s = df['close'].rolling(60).mean()
            bull_now = (float(ma5_s.iloc[-1]) > float(ma10_s.iloc[-1]) >
                        float(ma20_s.iloc[-1]) > float(ma60_s.iloc[-1]))
            freshness = 0
            for j in range(2, min(20, len(df) - 60)):
                was_bull = (float(ma5_s.iloc[-j]) > float(ma10_s.iloc[-j]) >
                            float(ma20_s.iloc[-j]) > float(ma60_s.iloc[-j]))
                if was_bull == bull_now:
                    freshness += 1
                else:
                    break
            result['ma_freshness'] = freshness
        else:
            result['ma_freshness'] = 99

        # 价格距MA距离
        ma20_val = result['ma'].get(20, price) if isinstance(result.get('ma'), dict) else price
        ma60_val = result['ma'].get(60, price) if isinstance(result.get('ma'), dict) else price
        result['price_dist_ma20'] = (price - ma20_val) / ma20_val * 100 if ma20_val else 0
        result['price_dist_ma60'] = (price - ma60_val) / ma60_val * 100 if ma60_val else 0

        # RSI趋势
        if len(rsi) >= 6:
            result['rsi_trend'] = float(rsi.iloc[-1]) - float(rsi.iloc[-6])
        else:
            result['rsi_trend'] = 0

        # 量价（使用VolumeAnalyzer）
        from scripts.analyzers.volume_analyzer import VolumeAnalyzer
        is_minute = 'date' in df.columns and len(df) > 0 and \
                    hasattr(df['date'].iloc[-1], 'hour') and df['date'].iloc[-1].hour > 0
        vol_result = VolumeAnalyzer().analyze_level(df, is_minute=is_minute)
        result.update(vol_result)

        return result

    # ── 第一步：长期趋势 ──

    def _long_term_trend(self, levels: Dict) -> Dict:
        """汇总周线/日线/60分钟的多空信号。"""
        signals = []
        for tf in ['weekly', 'daily', '60min']:
            r = levels.get(tf)
            if not r:
                continue
            if '多头' in r.get('ma_alignment', ''):
                signals.append((tf, 'MA多头', '多'))
            elif '空头' in r.get('ma_alignment', ''):
                signals.append((tf, 'MA空头', '空'))
            signals.append((tf, f"DIF{'>' if r['macd_dir'] == '多' else '<'}DEA", r['macd_dir']))
            if r.get('bot_div'):
                signals.append((tf, '底背离', '多'))
            if r.get('top_div'):
                signals.append((tf, '顶背离', '空'))

        bull = [s for s in signals if s[2] == '多']
        bear = [s for s in signals if s[2] == '空']

        if len(bull) > len(bear) * 1.5:
            verdict = '多头趋势'
        elif len(bear) > len(bull) * 1.5:
            verdict = '空头趋势'
        else:
            verdict = '多空分歧'

        return {'bull_signals': bull, 'bear_signals': bear, 'verdict': verdict}

    # ── 第二步：短期机会 ──

    def _short_term_opportunity(self, levels: Dict) -> Dict:
        buy = 0.0
        sell = 0.0
        b_reasons = []
        s_reasons = []

        r_d = levels.get('daily', {})
        r_60 = levels.get('60min', {})

        # 日线信号
        if r_d.get('j', 50) < 20:
            buy += 0.2; b_reasons.append(f"日线J值偏低({r_d['j']:.0f})")
        if r_d.get('bot_div'):
            buy += 0.3; b_reasons.append("日线MACD底背离")
        if r_d.get('top_div'):
            sell += 0.3; s_reasons.append("日线MACD顶背离")
        boll_pos = r_d.get('boll_pos', 50)
        if boll_pos < 30:
            buy += 0.15; b_reasons.append(f"日线布林偏低({boll_pos:.0f}%)")
        elif boll_pos > 70:
            sell += 0.15; s_reasons.append(f"日线布林偏高({boll_pos:.0f}%)")

        # 日线量价确认
        vp = r_d.get('vp_pattern', '')
        if vp == '量增价升':
            buy += 0.1; b_reasons.append("量增价升确认")
        elif vp == '量缩价升':
            sell += 0.05; s_reasons.append("缩量上涨动力不足")
        elif vp == '量缩价跌':
            buy += 0.05; b_reasons.append("缩量下跌卖盘枯竭")
        elif vp == '天量天价':
            sell += 0.15; s_reasons.append("天量天价见顶风险")
        elif vp == '地量地价':
            buy += 0.15; b_reasons.append("地量地价底部信号")

        # 60分钟信号
        if r_60.get('bot_div'):
            buy += 0.3; b_reasons.append("60min底背离")
        if r_60.get('top_div'):
            sell += 0.3; s_reasons.append("60min顶背离")
        if r_60.get('macd_dir') == '多':
            buy += 0.15; b_reasons.append("60min DIF>DEA")
        else:
            sell += 0.15; s_reasons.append("60min DIF<DEA")
        if r_60.get('j', 50) < 0:
            buy += 0.2; b_reasons.append(f"60min J超卖({r_60['j']:.0f})")
        elif r_60.get('j', 50) > 100:
            sell += 0.2; s_reasons.append(f"60min J超买({r_60['j']:.0f})")

        return {'buy_score': buy, 'sell_score': sell, 'buy_reasons': b_reasons, 'sell_reasons': s_reasons}

    # ── 关键价位汇总 ──

    def _collect_key_levels(self, levels: Dict, realtime_price: float = 0) -> List[Dict]:
        """从所有级别提取支撑/压力位，按距离排序。"""
        price = realtime_price or levels.get('daily', {}).get('price', 0)
        if price == 0:
            return []

        entries = []
        seen = {}  # 去重：价位2%范围内合并

        for tf in ['daily', '60min', '15min']:
            r = levels.get(tf)
            if not r:
                continue
            tf_label = {'daily': '日线', '60min': '60m', '15min': '15m', 'weekly': '周线'}.get(tf, tf)

            for p, v in r.get('ma', {}).items():
                self._add_level(entries, seen, price, v, f'{tf_label}MA{p}',
                                '强' if p >= 60 else '中' if p == 20 else '弱')

            for name, val in [('布林上轨', r.get('boll_upper')), ('布林中轨', r.get('boll_mid')),
                              ('布林下轨', r.get('boll_lower'))]:
                if val and val > 0 and abs(val - price) / price < 0.15:
                    self._add_level(entries, seen, price, val, f'{tf_label}{name}', '中')

        entries.sort(key=lambda x: x['dist'])
        return entries

    def _add_level(self, entries, seen, price, val, source, strength):
        dist = abs(price - val) / val * 100
        if dist > 15:
            return
        role = '压力' if val > price else '支撑'

        # 聚集检测
        key = round(val, 1)
        merged = False
        for existing_key in list(seen.keys()):
            if abs(existing_key - key) / max(existing_key, key) < 0.02:
                # 合并
                entry = seen[existing_key]
                entry['sources'].append(source)
                entry['strength'] = '强'  # 多源验证→强
                merged = True
                break

        if not merged:
            entry = {
                'role': role, 'value': round(val, 2), 'source': source,
                'sources': [source], 'dist': round(dist, 1), 'strength': strength,
            }
            entries.append(entry)
            seen[key] = entry

    # ── 四维加权信号评分 ──

    # 维度权重 (基于回测IC优化)
    _W_MA = 0.25       # 均线趋势 (原0.30, 负IC降权)
    _W_STRUCT = 0.25   # 动量 (原"缠论/波浪"MA代理, 重设计)
    _W_MACD = 0.35     # MACD双参数 (原0.25, 唯一正IC加权重)
    _W_VOL = 0.15      # 量价配合 (原0.20, IC近零降权)

    def _score_4d(self, levels: Dict) -> Dict:
        """四维加权信号评分 v2。

        基于回测优化：
        - 维度1(MA): 交叉动能+对齐新鲜度+支撑接近度 (替代简单排列判断)
        - 维度2(动量): RSI趋势+布林位置 (替代MA20/MA60代理)
        - 维度3(MACD): 不变，唯一正IC维度
        - 维度4(量价): 不含OBV (回测确认OBV有害)
        - 冲突降级: 阈值0.4, 分级惩罚 (替代一刀切0.2减半)
        """
        dims = {}
        r_d = levels.get('daily', {})

        # ── 维度1: 均线趋势 (25%) ──
        # 三子组件: 交叉动能(40%) + 对齐新鲜度(30%) + 支撑接近度(30%)
        ma_details = []

        # 子A: MA交叉动能
        spread_rate = r_d.get('ma_spread_rate', 0)
        ma_momentum = np.tanh(spread_rate * 3.0)
        ma_details.append(f"交叉动能={ma_momentum:+.2f}")

        # 子B: MA对齐新鲜度
        freshness = r_d.get('ma_freshness', 99)
        align = r_d.get('ma_alignment', '')
        if '多头' in align:
            base_dir = 1.0
        elif '空头' in align:
            base_dir = -1.0
        else:
            base_dir = 0.0
        if freshness <= 5:
            ff = 1.0
        elif freshness <= 10:
            ff = 1.0 - (freshness - 5) / 10.0
        else:
            ff = 0.3
        freshness_score = base_dir * ff
        ma_details.append(f"对齐新鲜度={freshness_score:+.2f}(形成{freshness}天)")

        # 子C: MA支撑接近度
        dist20 = r_d.get('price_dist_ma20', 0)
        dist60 = r_d.get('price_dist_ma60', 0)
        if dist20 > 8:
            proximity = -0.5
            ma_details.append("远离MA20(超买)")
        elif dist20 > 2:
            proximity = 0.2
            ma_details.append("适度高于MA20")
        elif dist20 > -2:
            proximity = 0.5
            ma_details.append("接近MA20支撑")
        else:
            proximity = 0.3 if dist60 > 0 else -0.3
            ma_details.append("MA20回踩" if dist60 > 0 else "跌破均线")

        dims['ma'] = max(-1.0, min(1.0,
            ma_momentum * 0.40 + freshness_score * 0.30 + proximity * 0.30))
        dims['ma_details'] = ma_details

        # ── 维度2: 动量 (25%) ──
        # RSI趋势 + 布林位置 (替代原"缠论/波浪"MA代理，IC=-0.015)
        mom_details = []

        if r_d:
            rsi = r_d.get('rsi', 50)
            rsi_trend = r_d.get('rsi_trend', 0)

            # RSI组件 (50%)
            rsi_level = (rsi - 50) / 50.0
            rsi_trend_score = np.tanh(rsi_trend / 20.0)
            rsi_combined = rsi_level * 0.4 + rsi_trend_score * 0.6
            mom_details.append(f"RSI={rsi:.0f}(趋势{rsi_trend:+.0f})")

            # 布林组件 (50%)
            boll_pos = r_d.get('boll_pos', 50)
            boll_squeeze = r_d.get('boll_squeeze', False)

            if boll_squeeze:
                if boll_pos < 30:
                    boll_score = 0.6
                    mom_details.append("布林收口低位(突破前兆)")
                elif boll_pos > 70:
                    boll_score = -0.6
                    mom_details.append("布林收口高位(回落风险)")
                else:
                    boll_score = 0.0
                    mom_details.append("布林收口中位")
            else:
                if boll_pos > 80:
                    boll_score = 0.3
                    mom_details.append("布林上轨(趋势确认)")
                elif boll_pos < 20:
                    boll_score = -0.3
                    mom_details.append("布林下轨(下行中)")
                else:
                    boll_score = (boll_pos - 50) / 100.0
                    mom_details.append(f"布林位置{boll_pos:.0f}%")

            mom_score = rsi_combined * 0.5 + boll_score * 0.5

            # MACD背离修正 (保留)
            if r_d.get('bot_div'):
                mom_score = max(mom_score, 0.5)
                mom_details.append("MACD底背离")
            if r_d.get('top_div'):
                mom_score = min(mom_score, -0.5)
                mom_details.append("MACD顶背离")
        else:
            mom_score = 0

        dims['struct'] = max(-1.0, min(1.0, mom_score))
        dims['struct_details'] = mom_details

        # ── 维度3: MACD双参数 (35%) ──
        # 不变：标准12,26,9定方向 + 快速5,34,5定背离
        macd_score = 0
        macd_details = []

        if r_d:
            # 标准MACD方向
            if r_d.get('macd_dir') == '多':
                macd_score += 0.6
                macd_details.append("日线DIF>DEA")
            else:
                macd_score -= 0.6
                macd_details.append("日线DIF<DEA")

            # 柱状线趋势
            ht = r_d.get('hist_trend', '')
            if '红柱扩大' in ht:
                macd_score += 0.2
                macd_details.append("红柱扩大")
            elif '绿柱扩大' in ht:
                macd_score -= 0.2
                macd_details.append("绿柱扩大")

            # 快速MACD背离
            if r_d.get('bot_div'):
                macd_score += 0.2
                macd_details.append("快速MACD底背离")
            if r_d.get('top_div'):
                macd_score -= 0.2
                macd_details.append("快速MACD顶背离")

        dims['macd'] = max(-1.0, min(1.0, macd_score))
        dims['macd_details'] = macd_details

        # ── 维度4: 量价配合 (15%) ──
        # OBV已剔除 (回测确认OBV有害，4/7股票含OBV更差)
        vol_score = 0
        vol_details = []

        if r_d:
            vp = r_d.get('vp_pattern', '')
            if vp == '量增价升':
                vol_score = 1.0
                vol_details.append("量增价升（多头确认）")
            elif vp == '量缩价升':
                vol_score = -0.3
                vol_details.append("缩量上涨（动力不足）")
            elif vp == '天量天价':
                vol_score = -0.8
                vol_details.append("天量天价（见顶风险）")
            elif vp == '地量地价':
                vol_score = 0.8
                vol_details.append("地量地价（底部信号）")
            elif vp == '量增价跌':
                vol_score = -0.6
                vol_details.append("放量下跌（空头压力）")
            elif vp == '量缩价跌':
                vol_score = 0.3
                vol_details.append("缩量下跌（卖盘枯竭）")

            # OBV仅信息展示，不参与评分
            obv = r_d.get('obv_divergence')
            if obv:
                vol_details.append(f"OBV{obv}(仅供参考)")

        dims['vol'] = max(-1.0, min(1.0, vol_score))
        dims['vol_details'] = vol_details

        # ── 综合分计算 ──
        raw = (dims['ma'] * self._W_MA +
               dims['struct'] * self._W_STRUCT +
               dims['macd'] * self._W_MACD +
               dims['vol'] * self._W_VOL)

        # 冲突降级v2: 阈值0.4, 分级惩罚 (原0.2一刀切, 60-69%触发)
        scores = [dims['ma'], dims['struct'], dims['macd'], dims['vol']]
        pos_count = sum(1 for s in scores if s > 0.4)
        neg_count = sum(1 for s in scores if s < -0.4)
        conflict_level = min(pos_count, neg_count)

        if conflict_level > 0:
            penalty = 1.0 - 0.2 * conflict_level
            composite = raw * penalty
            conflict = True
        else:
            composite = raw
            conflict = False

        # 判定
        if composite >= 0.5:
            verdict = '做多'
        elif composite >= 0.2:
            verdict = '偏多观望'
        elif composite > -0.2:
            verdict = '中性观望'
        elif composite > -0.5:
            verdict = '偏空观望'
        else:
            verdict = '做空'

        return {
            'dims': dims,
            'raw_score': round(raw, 3),
            'composite': round(composite, 3),
            'conflict': conflict,
            'verdict': verdict,
            'weights': {'ma': self._W_MA, 'struct': self._W_STRUCT,
                        'macd': self._W_MACD, 'vol': self._W_VOL},
        }

    # ── 操作建议 ──

    def _generate_advice(self, results: Dict) -> Dict:
        lt = results.get('long_term', {})
        st = results.get('short_term', {})
        levels = results.get('key_levels', [])
        rt = results.get('realtime', {})
        price = rt.get('price') or results.get('price', 0)

        supports = [l for l in levels if l['role'] == '支撑' and l['dist'] < 8]
        resistances = [l for l in levels if l['role'] == '压力' and l['dist'] < 8]

        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None

        # 基本面确认
        fund = results.get('fundamentals', {})
        fundamental_signal, fund_reasons = self._evaluate_fundamentals(fund)

        # 外资持仓信号（辅助）
        foreign = results.get('foreign', {})

        return {
            'trend': lt.get('verdict', ''),
            'short_bias': '偏多' if st.get('buy_score', 0) > st.get('sell_score', 0) else '偏空' if st.get('sell_score', 0) > st.get('buy_score', 0) else '中性',
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'fundamental_signal': fundamental_signal,
            'fund_reasons': fund_reasons,
            'foreign_signal': foreign.get('signal', 'none'),
            'foreign_details': foreign,
        }

    @staticmethod
    def _evaluate_fundamentals(fund: Dict) -> Tuple[str, List[str]]:
        """评估基本面是否支撑技术信号。返回 (signal, reasons)。"""
        if not fund:
            return '中性', []

        bullish = []
        bearish = []

        rev_yoy = fund.get('revenue_yoy')
        roe = fund.get('roe')
        debt = fund.get('debt_ratio')
        pe_ttm = fund.get('pe_ttm')

        # 按行业分档PE阈值
        pe_tier = fund.get('pe_tier', 'manufacturing')
        pe_threshold = _INDUSTRY_TIERS.get(pe_tier, 35)

        if rev_yoy is not None and rev_yoy > 10:
            bullish.append(f'营收+{rev_yoy:.0f}%')
        if roe is not None and roe > 15:
            bullish.append(f'ROE {roe:.1f}%')
        if debt is not None and debt < 50:
            bullish.append('负债率健康')

        if rev_yoy is not None and rev_yoy < -5:
            bearish.append(f'营收{rev_yoy:+.0f}%')
        if debt is not None and debt > 70:
            bearish.append(f'高负债{debt:.0f}%')
        if pe_ttm is not None and pe_ttm > pe_threshold:
            bearish.append(f'PE {pe_ttm:.0f}偏高')

        if len(bullish) >= 2:
            return '支撑', bullish
        elif len(bearish) >= 2:
            return '不支撑', bearish
        elif len(bearish) >= 1:
            return '偏弱', bearish
        elif len(bullish) >= 1:
            return '偏强', bullish
        return '中性', []

    # ── 格式化报告 ──

    @staticmethod
    def format_report(result: Dict, holding: Dict = None) -> str:
        if 'error' in result:
            return f"分析失败: {result['error']}"

        lines = []
        lines.append(f"{'='*65}")

        # 实时价格展示
        rt = result.get('realtime', {})
        rt_price = rt.get('price')
        price = rt_price if rt_price else result.get('price', 0)
        price_label = "(盘中实时)" if rt.get('is_realtime') else "(昨日收盘)"
        name = rt.get('name', '')
        if name:
            name = f" {name}"
        etf_label = " [ETF]" if result.get('is_etf') else ""
        lines.append(f"📊 综合技术分析: {result['symbol']}{name}{etf_label}")
        lines.append(f"   价格: {price:.2f} {price_label}  {rt.get('price_time', '')}")
        if rt.get('change_pct') is not None:
            lines.append(f"   涨跌: {rt['change']:+.2f} ({rt['change_pct']:+.2f}%)")
        lines.append(f"{'='*65}")

        # 大盘趋势
        mkt = result.get('market', {})
        if mkt:
            lines.append(f"\n🌐 大盘环境: 上证 {mkt.get('close', 0):.0f}  "
                         f"近5日 {mkt.get('pct_5d', 0):+.2f}%  趋势: {mkt.get('trend', '?')}")
            lines.append(f"{'─'*65}")

        # 资金流向
        mf = result.get('moneyflow', {})
        if mf and mf.get('trade_date'):
            net = mf.get('net_mf_amount', 0)
            buy_big = mf.get('buy_elg_amount', 0)
            sell_big = mf.get('sell_elg_amount', 0)
            # Tushare moneyflow单位：量(手), 额(千元)
            direction = "净流入" if net > 0 else "净流出"
            lines.append(f"\n💰 资金流向({mf['trade_date']}): "
                         f"{direction}{abs(net)/1e4:.2f}亿  "
                         f"大单买{buy_big/1e4:.2f}亿/卖{sell_big/1e4:.2f}亿")
            lines.append(f"{'─'*65}")

        # 基本面速览（仅股票）
        fund = result.get('fundamentals', {})
        tier_labels = {'traditional': '传统', 'manufacturing': '制造消费', 'growth': '科技成长'}
        if fund:
            report_date = fund.get('report_date', '')
            date_label = f" (财报期: {report_date})" if report_date else ""
            industry = fund.get('industry', '')
            ind_label = f" [{industry}·{tier_labels.get(fund.get('pe_tier', ''), '')}]" if industry else ""
            lines.append(f"\n📊 基本面速览{date_label}{ind_label}")

            # 估值
            val_parts = []
            pe = fund.get('pe')
            pe_ttm = fund.get('pe_ttm')
            pb = fund.get('pb')
            if pe is not None:
                val_parts.append(f"PE: {pe:.1f}")
            if pe_ttm is not None:
                val_parts.append(f"PE(TTM): {pe_ttm:.1f}")
            if pb is not None:
                val_parts.append(f"PB: {pb:.2f}")
            if val_parts:
                lines.append(f"  {' | '.join(val_parts)}")

            # 盈利能力
            profit_parts = []
            roe = fund.get('roe')
            gm = fund.get('gross_margin')
            if roe is not None:
                profit_parts.append(f"ROE: {roe:.1f}%")
            if gm is not None:
                profit_parts.append(f"毛利率: {gm:.1f}%")
            if profit_parts:
                lines.append(f"  {' | '.join(profit_parts)}")

            # 增长
            growth_parts = []
            rev_yoy = fund.get('revenue_yoy')
            prof_yoy = fund.get('profit_yoy')
            if rev_yoy is not None:
                growth_parts.append(f"营收增长: {rev_yoy:+.1f}% YoY")
            if prof_yoy is not None:
                growth_parts.append(f"利润增长: {prof_yoy:+.1f}% YoY")
            if growth_parts:
                lines.append(f"  {' | '.join(growth_parts)}")

            # 负债+市值
            risk_parts = []
            debt = fund.get('debt_ratio')
            mv = fund.get('total_mv')
            if debt is not None:
                risk_parts.append(f"负债率: {debt:.1f}%")
            if mv is not None:
                risk_parts.append(f"总市值: {mv/1e4:.0f}亿")
            if risk_parts:
                lines.append(f"  {' | '.join(risk_parts)}")

            # 自动风险提示（按行业分档估值）
            warnings = []
            pe_tier = fund.get('pe_tier', 'manufacturing')
            pe_threshold = _INDUSTRY_TIERS.get(pe_tier, 35)
            tier_name = tier_labels.get(pe_tier, '制造消费')

            if pe_ttm is not None and pe_ttm > pe_threshold:
                warnings.append(f"PE(TTM){pe_ttm:.0f}>{pe_threshold}({tier_name}高估线)")
            elif pe is not None and pe < 0:
                warnings.append("PE为负（亏损）")
            if rev_yoy is not None and rev_yoy < -5:
                warnings.append(f"营收负增长({rev_yoy:+.1f}%)")
            if debt is not None and debt > 70:
                warnings.append(f"负债率偏高({debt:.0f}%)")
            if roe is not None and roe < 5:
                warnings.append(f"ROE偏低({roe:.1f}%)")
            if warnings:
                lines.append(f"  ⚠️ {' | '.join(warnings)}")

            lines.append(f"{'─'*65}")

        # 外资持仓信号（辅助）
        foreign = result.get('foreign', {})
        if foreign and foreign.get('signal', 'none') != 'none':
            period = foreign.get('latest_period', '')
            fc = foreign.get('foreign_count', 0)
            fr = foreign.get('foreign_total_ratio', 0)
            pfc = foreign.get('prev_foreign_count', 0)
            pfr = foreign.get('prev_foreign_total_ratio', 0)
            sig_label = {'foreign_increase': '增持', 'foreign_decrease': '减持',
                         'foreign_stable': '稳定'}.get(foreign['signal'], foreign['signal'])
            lines.append(f"\n🌍 外资持仓({period}): {sig_label}")
            lines.append(f"  本期{fc}家 合计{fr:.2f}% | 上期{pfc}家 合计{pfr:.2f}%")
            for d in foreign.get('details', []):
                if d['change'] not in ('退出',):
                    change_str = d['change'] if d['change'] != '退出' else ''
                    lines.append(f"  · {d['name']}: {d['ratio']:.2f}% ({change_str})")
            lines.append(f"{'─'*65}")

        tf_labels = {'weekly': '周线', 'daily': '日线', '60min': '60分钟', '15min': '15分钟'}

        # 每个级别
        for tf in ['weekly', 'daily', '60min', '15min']:
            r = result.get('levels', {}).get(tf)
            if not r:
                continue
            lines.append(f"\n── {tf_labels.get(tf, tf)}级别 ──")

            # MA
            lines.append(f"  均线: {r.get('ma_alignment', '?')}")
            for p, v in sorted(r.get('ma', {}).items()):
                dist = (r['price'] - v) / v * 100
                role = "压力" if dist < 0 else "支撑"
                lines.append(f"    MA{p:>3d} = {v:>8.2f}  {dist:>+6.2f}%  {role}")

            # MACD
            lines.append(f"  MACD: DIF{'>' if r['macd_dir']=='多' else '<'}DEA  柱状={r['macd_hist']:.4f}  {r['hist_trend']}")
            if r.get('top_div'):
                lines.append(f"  ⚠️ 顶背离")
            if r.get('bot_div'):
                lines.append(f"  ✅ 底背离")

            # RSI
            rsi = r.get('rsi', 50)
            lines.append(f"  RSI: {rsi:.1f}  {'超买' if rsi>70 else '超卖' if rsi<30 else '偏强' if rsi>50 else '偏弱'}")

            # Boll
            lines.append(f"  布林: {r.get('boll_pos', 50):.0f}%  上{r.get('boll_upper',0):.2f} 中{r.get('boll_mid',0):.2f} 下{r.get('boll_lower',0):.2f}  {'收口' if r.get('boll_squeeze') else '开口'}")

            # KDJ
            lines.append(f"  KDJ: K={r.get('k',0):.1f} D={r.get('d',0):.1f} J={r.get('j',0):.1f}  {'K>D' if r.get('k',0)>r.get('d',0) else 'K<D'}")

            # 量价
            vp = r.get('vp_pattern', '')
            vr = r.get('vol_ratio', 1)
            vc = r.get('vol_class', '')
            mfi = r.get('mfi', 50)
            cmf = r.get('cmf', 0)
            obv_div = r.get('obv_divergence')
            vol_line = f"  量价: {vp}  量比{vr:.2f}x({vc})  MFI{mfi:.0f}  CMF{cmf:+.3f}"
            if obv_div:
                vol_line += f"  OBV{obv_div}"
            vwap = r.get('vwap')
            if vwap:
                vol_line += f"  VWAP{vwap:.2f}"
            lines.append(vol_line)

        # 长期趋势
        lt = result.get('long_term', {})
        lines.append(f"\n{'='*65}")
        lines.append(f"第一步·长期趋势: {lt.get('verdict', '?')}")
        lines.append(f"{'='*65}")
        for tf, sig, d in lt.get('bull_signals', []):
            lines.append(f"  📈 {tf_labels.get(tf, tf)}: {sig}")
        for tf, sig, d in lt.get('bear_signals', []):
            lines.append(f"  📉 {tf_labels.get(tf, tf)}: {sig}")

        # 四维信号评分
        sig4 = result.get('signal_4d', {})
        if sig4:
            lines.append(f"\n{'─'*65}")
            lines.append(f"📐 四维信号评分: {sig4.get('verdict', '?')} "
                         f"综合分={sig4.get('composite', 0):+.3f}"
                         f"{' [冲突降级]' if sig4.get('conflict') else ''}")
            dims = sig4.get('dims', {})
            w = sig4.get('weights', {})
            dim_labels = [('ma', '均线趋势'), ('struct', '动量'), ('macd', 'MACD双参数'), ('vol', '量价配合')]
            for key, label in dim_labels:
                d = dims.get(key, 0)
                arrow = '📈' if d > 0.2 else '📉' if d < -0.2 else '➡️'
                detail = ', '.join(dims.get(f'{key}_details', [])) or '中性'
                lines.append(f"  {arrow} {label}({w.get(key,0)*100:.0f}%): {d:+.2f} — {detail}")
            lines.append(f"{'─'*65}")

        # 短期信号
        st = result.get('short_term', {})
        lines.append(f"\n第二步·短期信号: 买入{st.get('buy_score',0):.2f} 卖出{st.get('sell_score',0):.2f}")
        for r in st.get('buy_reasons', []):
            lines.append(f"  📈 {r}")
        for r in st.get('sell_reasons', []):
            lines.append(f"  📉 {r}")

        # 关键价位
        lines.append(f"\n📍 关键价位:")
        for l in result.get('key_levels', [])[:10]:
            sources = '+'.join(l['sources'][:3])
            lines.append(f"  {l['role']}: {sources} = {l['value']:.2f}  距{l['dist']:.1f}%  {l['strength']}")

        # 操作建议
        adv = result.get('advice', {})
        lines.append(f"\n🎯 综合结论:")
        lines.append(f"  趋势: {adv.get('trend', '?')}  短期: {adv.get('short_bias', '?')}")
        ns = adv.get('nearest_support')
        nr = adv.get('nearest_resistance')
        if ns:
            lines.append(f"  最近支撑: {'+'.join(ns['sources'][:2])} = {ns['value']:.2f}")
        if nr:
            lines.append(f"  最近压力: {'+'.join(nr['sources'][:2])} = {nr['value']:.2f}")

        # 基本面确认
        fund_signal = adv.get('fundamental_signal', '')
        fund_reasons = adv.get('fund_reasons', [])
        if fund_signal and fund_signal != '中性':
            icon = '📊' if fund_signal in ('支撑', '偏强') else '⚠️'
            reasons_str = ' | '.join(fund_reasons)
            lines.append(f"  {icon} 基本面{fund_signal} ({reasons_str})")

        # 资金效率（如有持仓数据）
        if holding:
            from scripts.analyzers.capital_efficiency import CapitalEfficiencyAnalyzer
            ce = CapitalEfficiencyAnalyzer()
            price = result.get('realtime', {}).get('price', 0)
            if price == 0:
                price = result.get('price', 0)
            holding_data = {
                'code': result.get('symbol', ''),
                'name': holding.get('name', ''),
                'cost': holding.get('cost', 0),
                'price': price,
                'shares': holding.get('shares', 0),
                'market_value': price * holding.get('shares', 0),
                'buy_date': holding.get('buy_date', ''),
                'pnl_pct': (price / holding.get('cost', 1) - 1) * 100 if holding.get('cost', 0) > 0 else 0,
            }
            eff = ce.score_position(holding_data)
            gi = eff['grade_info']
            lines.append(f"  💰 资金效率: {gi['color']}{eff['grade']}({eff['score']}) | "
                        f"年化回报{eff['annualized_return_pct']:+.1f}% | {gi['action']}")

        return '\n'.join(lines)
