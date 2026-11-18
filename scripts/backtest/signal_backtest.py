#!/usr/bin/env python3
"""
4D信号回测引擎

在历史数据上逐日计算4D信号，验证各维度的预测能力。
核心问题：MA/MACD/量价/OBV是否真的有预测力？
"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.core.tushare_client import TushareClient
from scripts.backtest.metrics import (
    information_coefficient, signal_accuracy_by_level,
    sharpe_ratio, max_drawdown, total_return, annualized_return,
    win_rate, profit_factor, alpha_vs_buyhold, ic_decay,
    format_backtest_report,
)
from scripts.backtest.trade_simulator import TradeSimulator

warnings.filterwarnings('ignore')


class SignalBacktester:
    """4D信号历史回测引擎"""

    # 量价模式阈值
    VOL_THRESHOLDS = {'天量': 3.0, '放量': 1.5, '缩量': 0.7, '地量': 0.33}

    def __init__(self, symbol: str, start: str, end: str = None):
        self.symbol = symbol
        self.start = start
        self.end = end or pd.Timestamp.now().strftime('%Y%m%d')
        self.ts = TushareClient()
        self.daily = None
        self.weekly = None
        self.results = {}

    def _is_etf(self, symbol: str) -> bool:
        """判断是否为ETF/基金代码"""
        s = symbol.split('.')[0]  # 去掉.SH/.SZ
        return s[:2] in ('51', '52', '15', '16', '50', '56', '59')

    def fetch_data(self) -> pd.DataFrame:
        """获取历史日K线（含周K重采样）"""
        ext_start = pd.Timestamp(self.start) - pd.Timedelta(days=365)
        ext_start_str = ext_start.strftime('%Y%m%d')

        if self._is_etf(self.symbol):
            # ETF用fund_daily接口
            from scripts.core.tushare_client import to_ts_code
            ts_code = to_ts_code(self.symbol)
            raw = self.ts.get_fund_daily(ts_code, start=ext_start_str, end=self.end)
            if raw is None or raw.empty:
                raise ValueError(f"ETF数据不足: {self.symbol}")
            col_map = {'trade_date': 'date', 'vol': 'volume', 'pct_chg': 'change_pct', 'pre_close': 'pre_close'}
            self.daily = raw.rename(columns=col_map)
            if 'date' in self.daily.columns:
                self.daily['date'] = pd.to_datetime(self.daily['date'])
            self.daily = self.daily.sort_values('date').reset_index(drop=True)
        else:
            self.daily = self.ts.get_kline(
                self.symbol, freq='D',
                start=ext_start_str, end=self.end, adj='hfq'
            )
        if self.daily is None or len(self.daily) < 200:
            raise ValueError(f"数据不足: {self.symbol} 从{ext_start_str}到{self.end}仅{len(self.daily) if self.daily is not None else 0}条")

        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.daily[col] = pd.to_numeric(self.daily[col], errors='coerce')
        self.daily['date'] = pd.to_datetime(self.daily['date'])
        self.daily = self.daily.sort_values('date').reset_index(drop=True)

        # 计算所有指标（向量化）
        self._compute_all_indicators()

        # 周K线
        self._resample_weekly()

        return self.daily

    def _compute_all_indicators(self):
        """向量化计算所有技术指标"""
        df = self.daily
        c = df['close']
        v = df['volume']

        # MA
        for p in [5, 10, 20, 60, 120]:
            df[f'ma{p}'] = c.rolling(p).mean()

        # MACD标准(12,26,9)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['dif'] = ema12 - ema26
        df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = 2 * (df['dif'] - df['dea'])
        df['macd_dir'] = (df['dif'] > df['dea']).astype(int)  # 1=多, 0=空

        # MACD柱状线趋势
        df['hist_trend'] = np.where(
            (df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1)), 1,  # 红柱扩大
            np.where(
                (df['macd_hist'] > 0) & (df['macd_hist'] <= df['macd_hist'].shift(1)), 2,  # 红柱缩短
                np.where(
                    (df['macd_hist'] < 0) & (df['macd_hist'] < df['macd_hist'].shift(1)), -1,  # 绿柱扩大
                    -2  # 绿柱缩短
                )
            )
        )

        # MACD快速(5,34,5)
        ema5 = c.ewm(span=5, adjust=False).mean()
        ema34 = c.ewm(span=34, adjust=False).mean()
        df['dif_f'] = ema5 - ema34
        df['dea_f'] = df['dif_f'].ewm(span=5, adjust=False).mean()
        df['macd_hist_f'] = 2 * (df['dif_f'] - df['dea_f'])

        # MACD背离检测(用快速5,34,5的DIF)
        df['top_div'] = False
        df['bot_div'] = False
        lb = 30
        for i in range(lb + 5, len(df)):
            window = df.iloc[i-lb:i+1]
            # 顶背离：价格新高但DIF未新高
            if window['high'].iloc[-1] >= window['high'].iloc[-5]:
                high_idx = window['high'].idxmax()
                if df['dif_f'].iloc[i] < df['dif_f'].loc[high_idx]:
                    df.loc[df.index[i], 'top_div'] = True
            # 底背离：价格新低但DIF未新低
            if window['low'].iloc[-1] <= window['low'].iloc[-5]:
                low_idx = window['low'].idxmin()
                if df['dif_f'].iloc[i] > df['dif_f'].loc[low_idx]:
                    df.loc[df.index[i], 'bot_div'] = True

        # RSI
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # Bollinger
        df['boll_mid'] = c.rolling(20).mean()
        df['boll_std'] = c.rolling(20).std(ddof=0)
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

        # OBV
        direction = np.sign(c.diff())
        direction.iloc[0] = 0
        df['obv'] = (direction * v).cumsum()

        # OBV背离
        df['obv_top_div'] = False
        df['obv_bot_div'] = False
        for i in range(lb + 5, len(df)):
            window = df.iloc[i-lb:i+1]
            obv_window = df['obv'].iloc[i-lb:i+1]
            if window['high'].iloc[-1] >= window['high'].iloc[-5]:
                high_idx = window['high'].idxmax()
                if obv_window.iloc[-1] < obv_window.loc[high_idx]:
                    df.loc[df.index[i], 'obv_top_div'] = True
            if window['low'].iloc[-1] <= window['low'].iloc[-5]:
                low_idx = window['low'].idxmin()
                if obv_window.iloc[-1] > obv_window.loc[low_idx]:
                    df.loc[df.index[i], 'obv_bot_div'] = True

        # 量价模式
        df['vol_ma5'] = v.rolling(5).mean()
        df['vol_ratio'] = v / df['vol_ma5'].replace(0, np.nan)
        df['vol_ratio'] = df['vol_ratio'].fillna(1.0)
        df['price_chg_pct'] = c.pct_change() * 100
        df['is_boll_high'] = c > (df['boll_mid'] + 1.5 * df['boll_std'])
        df['is_boll_low'] = c < (df['boll_mid'] - 1.5 * df['boll_std'])
        df['vp_pattern'] = self._calc_vp_vectorized(df)

        # ── 新增：MA动能+新鲜度+支撑位指标 ──
        # MA价差动能 (MA5-MA20)
        df['ma_spread'] = df['ma5'] - df['ma20']
        prev_spread = df['ma_spread'].shift(5)
        df['ma_spread_rate'] = np.where(
            prev_spread.abs() > 0.01,
            (df['ma_spread'] - prev_spread) / prev_spread.abs(), 0)
        df['ma_spread_rate'] = df['ma_spread_rate'].fillna(0)

        # MA排列新鲜度 (当前排列持续天数)
        bull_align = (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma60'])
        bear_align = (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma60'])
        current_sign = bull_align.astype(int) - bear_align.astype(int)
        freshness = pd.Series(0, index=df.index)
        for j in range(1, 20):
            prev_sign = current_sign.shift(j).fillna(0).astype(int)
            freshness += (current_sign == prev_sign) & (current_sign != 0)
        df['ma_freshness'] = freshness

        # 价格距MA距离(%)
        df['price_dist_ma20'] = np.where(
            df['ma20'] > 0, (df['close'] - df['ma20']) / df['ma20'] * 100, 0)
        df['price_dist_ma60'] = np.where(
            df['ma60'] > 0, (df['close'] - df['ma60']) / df['ma60'] * 100, 0)

        # RSI趋势 (5日变化)
        df['rsi_trend'] = df['rsi'].diff(5).fillna(0)

        # 布林带宽度
        df['boll_width'] = np.where(
            df['boll_mid'] > 0,
            (df['boll_upper'] - df['boll_lower']) / df['boll_mid'], np.nan)
        # boll_width前N根NaN是正常的（数据不足），不应填入未来值

        # 前向收益
        for n in [1, 3, 5, 10, 20]:
            df[f'fwd_{n}d'] = c.shift(-n) / c - 1

        self.daily = df

    def _calc_vp_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """向量化量价模式计算"""
        vr = df['vol_ratio']
        pc = df['price_chg_pct']
        is_h = df['is_boll_high']
        is_l = df['is_boll_low']

        result = pd.Series('量平价平', index=df.index)

        vol_up = vr >= self.VOL_THRESHOLDS['放量']
        vol_down = vr < self.VOL_THRESHOLDS['缩量']
        price_up = pc > 0.3
        price_down = pc < -0.3

        result.loc[vol_up & price_up] = '量增价升'
        result.loc[vol_down & price_up] = '量缩价升'
        result.loc[vol_up & price_down] = '量增价跌'
        result.loc[vol_down & price_down] = '量缩价跌'
        result.loc[(vr >= self.VOL_THRESHOLDS['天量']) & is_h] = '天量天价'
        result.loc[(vr <= self.VOL_THRESHOLDS['地量']) & is_l] = '地量地价'

        # 未被特殊模式覆盖的量平情况
        normal = ~(vol_up | vol_down)
        result.loc[normal & price_up] = '量平价升'
        result.loc[normal & price_down] = '量平价跌'

        return result

    def _resample_weekly(self):
        """日K → 周K"""
        df = self.daily.copy()
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        rows = []
        for (_, _), g in df.groupby(['year', 'week']):
            rows.append({
                'date': g['date'].iloc[-1],
                'open': g['open'].iloc[0],
                'high': g['high'].max(),
                'low': g['low'].min(),
                'close': g['close'].iloc[-1],
                'volume': g['volume'].sum(),
            })
        wdf = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        c = wdf['close']
        for p in [5, 10, 20, 60]:
            wdf[f'ma{p}'] = c.rolling(p).mean()
        self.weekly = wdf

    def _score_dim_ma(self, idx: int) -> Tuple[float, str]:
        """维度1: 均线趋势 — 交叉动能+对齐新鲜度+支撑接近度"""
        row = self.daily.iloc[idx]
        details = []

        # 子组件A: MA交叉动能 (40%)
        spread_rate = row.get('ma_spread_rate', 0)
        if pd.isna(spread_rate):
            spread_rate = 0
        ma_momentum = np.tanh(spread_rate * 3.0)
        details.append(f"动能={ma_momentum:+.2f}")

        # 子组件B: MA对齐新鲜度 (30%)
        freshness = row.get('ma_freshness', 99)
        if pd.isna(freshness):
            freshness = 99
        ma5 = row.get('ma5')
        ma10 = row.get('ma10')
        ma20 = row.get('ma20')
        ma60 = row.get('ma60')
        if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and pd.notna(ma60):
            if ma5 > ma10 > ma20 > ma60:
                base_dir = 1.0
            elif ma5 < ma10 < ma20 < ma60:
                base_dir = -1.0
            else:
                base_dir = 0.0
        else:
            base_dir = 0.0
        if freshness <= 5:
            ff = 1.0
        elif freshness <= 10:
            ff = 1.0 - (freshness - 5) / 10.0
        else:
            ff = 0.3
        freshness_score = base_dir * ff
        details.append(f"新鲜度={int(freshness)}天")

        # 子组件C: MA支撑接近度 (30%)
        dist20 = row.get('price_dist_ma20', 0)
        if pd.isna(dist20):
            dist20 = 0
        dist60 = row.get('price_dist_ma60', 0)
        if pd.isna(dist60):
            dist60 = 0
        if dist20 > 8:
            proximity = -0.5
            details.append("远离MA20(超买)")
        elif dist20 > 2:
            proximity = 0.2
            details.append("适度高于MA20")
        elif dist20 > -2:
            proximity = 0.5
            details.append("接近MA20支撑")
        else:
            proximity = 0.3 if dist60 > 0 else -0.3
            details.append("MA20回踩" if dist60 > 0 else "跌破均线")

        total = ma_momentum * 0.40 + freshness_score * 0.30 + proximity * 0.30
        return max(-1.0, min(1.0, total)), " ".join(details)

    def _score_dim_struct(self, idx: int) -> Tuple[float, str]:
        """维度2: 动量(25%) — RSI趋势+布林位置"""
        row = self.daily.iloc[idx]
        details = []
        mom_score = 0.0

        rsi = row.get('rsi', 50)
        rsi_trend = row.get('rsi_trend', 0)
        if pd.isna(rsi):
            rsi = 50
        if pd.isna(rsi_trend):
            rsi_trend = 0

        # RSI组件 (50%)
        rsi_level = (rsi - 50) / 50.0
        rsi_trend_score = np.tanh(rsi_trend / 20.0)
        rsi_combined = rsi_level * 0.4 + rsi_trend_score * 0.6
        details.append(f"RSI={rsi:.0f}")

        # 布林组件 (50%)
        boll_upper = row.get('boll_upper', 0)
        boll_lower = row.get('boll_lower', 0)
        boll_mid = row.get('boll_mid', 0)
        price = row['close']
        rng = boll_upper - boll_lower
        boll_pos = (price - boll_lower) / rng * 100 if rng > 0 and not pd.isna(rng) else 50

        boll_width = row.get('boll_width', 0)
        prev_width = self.daily.iloc[max(0, idx-5)].get('boll_width', boll_width) if idx >= 5 else boll_width
        squeeze = (not pd.isna(boll_width) and not pd.isna(prev_width) and boll_width < prev_width)

        if squeeze:
            if boll_pos < 30:
                boll_score = 0.6
            elif boll_pos > 70:
                boll_score = -0.6
            else:
                boll_score = 0.0
        else:
            if boll_pos > 80:
                boll_score = 0.3
            elif boll_pos < 20:
                boll_score = -0.3
            else:
                boll_score = (boll_pos - 50) / 100.0

        mom_score = rsi_combined * 0.5 + boll_score * 0.5

        # MACD背离修正
        if row.get('bot_div', False):
            mom_score = max(mom_score, 0.5)
        if row.get('top_div', False):
            mom_score = min(mom_score, -0.5)

        return max(-1.0, min(1.0, mom_score)), " ".join(details)

    def _score_dim_macd(self, idx: int) -> Tuple[float, str]:
        """维度3: MACD双参数(25%)"""
        row = self.daily.iloc[idx]
        s = 0.0
        details = []

        # DIF vs DEA
        if row['macd_dir'] == 1:
            s += 0.6
            details.append("DIF>DEA")
        else:
            s -= 0.6
            details.append("DIF<DEA")

        # 柱状线
        ht = row['hist_trend']
        if ht == 1:
            s += 0.2
            details.append("红柱扩大")
        elif ht == -1:
            s -= 0.2
            details.append("绿柱扩大")

        # 背离
        if row.get('bot_div', False):
            s += 0.2
            details.append("底背离")
        if row.get('top_div', False):
            s -= 0.2
            details.append("顶背离")

        return max(-1.0, min(1.0, s)), " ".join(details)

    def _score_dim_vol(self, idx: int) -> Tuple[float, str]:
        """维度4: 量价(15%) — 纯量价模式，不含OBV"""
        row = self.daily.iloc[idx]
        vp = row.get('vp_pattern', '量平价平')

        vp_scores = {
            '量增价升': 1.0, '量缩价升': -0.3, '天量天价': -0.8,
            '地量地价': 0.8, '量增价跌': -0.6, '量缩价跌': 0.3,
            '量增价平': 0.2, '量缩价平': 0.0, '量平价升': 0.3,
            '量平价跌': -0.3, '量平价平': 0.0,
        }
        s = vp_scores.get(vp, 0.0)
        return max(-1.0, min(1.0, s)), vp

    def _combine_4d(self, dims: dict) -> dict:
        """4D加权综合 v2 (基于回测优化权重+冲突降级)"""
        raw = (dims['ma'] * 0.25 + dims['struct'] * 0.25 +
               dims['macd'] * 0.35 + dims['vol'] * 0.15)

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

        if composite >= 0.5:
            verdict = '做多'
        elif composite >= 0.2:
            verdict = '偏多'
        elif composite > -0.2:
            verdict = '中性'
        elif composite > -0.5:
            verdict = '偏空'
        else:
            verdict = '做空'

        return {
            'raw': round(raw, 3),
            'composite': round(composite, 3),
            'conflict': conflict,
            'verdict': verdict,
        }

    def run(self) -> dict:
        """运行完整回测"""
        if self.daily is None:
            self.fetch_data()

        df = self.daily
        start_ts = pd.Timestamp(self.start)
        mask = df['date'] >= start_ts
        start_idx = mask.idxmax() if mask.any() else 150
        start_idx = max(start_idx, 150)  # 至少150天预热

        # 逐日评分
        records = []
        for i in range(start_idx, len(df) - 20):  # 最后20天无前向收益
            row = df.iloc[i]
            if pd.isna(row.get('fwd_5d')):
                continue

            dims = {}
            dims['ma'], ma_d = self._score_dim_ma(i)
            dims['struct'], struct_d = self._score_dim_struct(i)
            dims['macd'], macd_d = self._score_dim_macd(i)
            dims['vol'], vol_d = self._score_dim_vol(i)

            full_4d = self._combine_4d(dims)

            rec = {
                'date': row['date'],
                'open': row['open'],
                'close': row['close'],
                # 各维度得分
                'ma': dims['ma'], 'struct': dims['struct'],
                'macd': dims['macd'], 'vol': dims['vol'],
                # 4D综合
                'composite': full_4d['composite'],
                'verdict': full_4d['verdict'],
                'conflict': full_4d['conflict'],
                # 前向收益
                'fwd_1d': row.get('fwd_1d', np.nan),
                'fwd_3d': row.get('fwd_3d', np.nan),
                'fwd_5d': row.get('fwd_5d', np.nan),
                'fwd_10d': row.get('fwd_10d', np.nan),
                'fwd_20d': row.get('fwd_20d', np.nan),
            }
            records.append(rec)

        signals = pd.DataFrame(records)
        if len(signals) == 0:
            return {'error': '无有效信号数据'}

        # ── 分析结果 ──
        results = {
            'symbol': self.symbol,
            'start': self.start,
            'end': self.end,
            'trading_days': len(signals),
            'signals': signals,
        }

        # 1. 买入持有基准
        bh_total = signals['close'].iloc[-1] / signals['close'].iloc[0] - 1
        bh_ann = annualized_return(bh_total, len(signals))
        bh_returns = signals.set_index('date')['close'].pct_change().dropna()
        results['buy_hold'] = {
            'total_return': bh_total,
            'annualized_return': bh_ann,
            'sharpe': sharpe_ratio(bh_returns),
            'max_drawdown': max_drawdown(signals.set_index('date')['close']),
            'win_rate': win_rate(bh_returns),
            'profit_factor': profit_factor(bh_returns),
        }

        # 2. 4D信号策略（做多/偏多买入，偏空/做空空仓）
        self._run_strategy_backtest(signals, results)

        # 3. 各信号级别胜率
        fwd_col = 'fwd_5d'
        valid = signals[fwd_col].notna()
        results['signal_accuracy'] = signal_accuracy_by_level(
            signals.loc[valid, 'verdict'],
            signals.loc[valid, fwd_col]
        ).to_dict('records')

        # 4. IC（信息系数）
        for col in ['composite', 'ma', 'struct', 'macd', 'vol']:
            valid = signals[col].notna() & signals[fwd_col].notna()
            if valid.sum() > 10:
                results[f'ic_{col}'] = information_coefficient(
                    signals.loc[valid, col], signals.loc[valid, fwd_col])

        # 5. 单维度分析
        results['dimension_analysis'] = {}
        for dim in ['ma', 'struct', 'macd', 'vol']:
            valid = signals[dim].notna() & signals[fwd_col].notna()
            if valid.sum() > 10:
                bull = (signals[dim] > 0.2) & valid
                bear = (signals[dim] < -0.2) & valid
                wr_bull = (signals.loc[bull, fwd_col] > 0).mean() if bull.sum() > 0 else 0
                wr_bear = (signals.loc[bear, fwd_col] < 0).mean() if bear.sum() > 0 else 0
                ic = information_coefficient(signals.loc[valid, dim], signals.loc[valid, fwd_col])
                results['dimension_analysis'][dim] = {
                    'ic': ic,
                    'win_rate': (wr_bull + wr_bear) / 2,
                    'bull_count': int(bull.sum()),
                    'bear_count': int(bear.sum()),
                    'bull_fwd_avg': signals.loc[bull, fwd_col].mean() if bull.sum() > 0 else 0,
                    'bear_fwd_avg': signals.loc[bear, fwd_col].mean() if bear.sum() > 0 else 0,
                }

        # 6. IC衰减
        ic_cols = {}
        for p in [1, 3, 5, 10, 20]:
            col = f'fwd_{p}d'
            if col in signals.columns:
                valid = signals['composite'].notna() & signals[col].notna()
                if valid.sum() > 10:
                    ic_cols[f'{p}d'] = information_coefficient(
                        signals.loc[valid, 'composite'], signals.loc[valid, col])
        results['ic_decay'] = ic_cols

        # 汇总
        results['ic'] = results.get('ic_composite', 0)
        results['alpha'] = results['strategy'].get('annualized_return', 0) - bh_ann

        self.results = results
        return results

    def _run_strategy_backtest(self, signals: pd.DataFrame, results: dict):
        """运行基于4D信号的实际交易策略"""
        sim = TradeSimulator(initial_capital=100000)
        equity_records = []
        signals_list = signals['verdict'].tolist()

        position = 0
        for i in range(len(signals)):
            row = signals.iloc[i]
            verdict = row['verdict']

            # 交易逻辑: 信号在bar[i]收盘后产生, 以bar[i+1]开盘价执行
            next_open = signals.iloc[i+1]['open'] if i+1 < len(signals) else None
            next_date = str(signals.iloc[i+1]['date']) if i+1 < len(signals) else None

            if verdict in ('做多', '偏多') and position == 0 and next_open:
                res = sim.buy(next_open, next_date, pct=0.90)
                if res.get('success'):
                    position = 1
            elif verdict in ('做空', '偏空') and position > 0 and next_open:
                res = sim.sell(next_open, next_date)
                if res.get('success'):
                    position = 0
            # 中性不动

            sim.record_equity(str(row['date']), row['close'])

        eq = sim.get_equity_series()
        rets = sim.get_returns_series()
        trade_rets = sim.get_trade_returns()

        total_ret = total_return(eq) if len(eq) > 1 else 0
        results['strategy'] = {
            'total_return': total_ret,
            'annualized_return': annualized_return(total_ret, len(signals)),
            'sharpe': sharpe_ratio(rets),
            'max_drawdown': max_drawdown(eq),
            'win_rate': win_rate(trade_rets) if len(trade_rets) > 0 else 0,
            'profit_factor': profit_factor(trade_rets) if len(trade_rets) > 0 else 0,
            'trade_count': len([t for t in sim.trades if t['action'] == 'buy']),
        }

    def run_no_obv(self) -> dict:
        """无OBV版本的4D回测"""
        if self.results:
            return self.results
        return self.run()


def run_batch(symbols: list, start: str, end: str = None) -> dict:
    """批量回测"""
    all_results = {}
    for sym in symbols:
        try:
            bt = SignalBacktester(sym, start, end)
            r = bt.run()
            all_results[sym] = r
        except Exception as e:
            all_results[sym] = {'error': str(e)}
    return all_results


def format_batch_report(all_results: dict) -> str:
    """格式化批量回测报告"""
    lines = []
    lines.append(f"\n{'='*90}")
    lines.append(f"📊 批量回测汇总 ({len(all_results)}只)")
    lines.append(f"{'='*90}")

    # 表头
    lines.append(f"{'代码':<10} {'交易天数':>6} {'4D胜率':>8} {'4D Sharpe':>10} "
                 f"{'买入持有':>10} {'Alpha':>8} {'IC':>8} {'OBV影响':>8}")
    lines.append(f"{'─'*90}")

    strat_wrs = []
    bh_rets = []
    ics = []

    for sym, r in all_results.items():
        if 'error' in r:
            lines.append(f"{sym:<10} 错误: {r['error']}")
            continue

        days = r.get('trading_days', 0)
        strat = r.get('strategy', {})
        bh = r.get('buy_hold', {})
        wr = strat.get('win_rate', 0)
        sharpe = strat.get('sharpe', 0)
        bh_ret = bh.get('total_return', 0)
        alpha = r.get('alpha', 0)
        ic = r.get('ic', 0)

        strat_wrs.append(wr)
        bh_rets.append(bh_ret)
        ics.append(ic)

        obv = r.get('obv_comparison', {})
        obv_diff = obv.get('without_obv_ic', 0) - obv.get('with_obv_ic', 0)
        obv_label = '有害' if obv_diff > 0.01 else '有益' if obv_diff < -0.01 else '中性'

        lines.append(f"{sym:<10} {days:>6} {wr:>7.1%} {sharpe:>10.3f} "
                      f"{bh_ret:>9.2%} {alpha:>+7.2%} {ic:>+7.4f} {obv_label:>8}")

    # 汇总
    if strat_wrs:
        lines.append(f"\n{'─'*90}")
        lines.append(f"平均胜率: {np.mean(strat_wrs):.1%} | "
                      f"平均Alpha: {np.mean([r.get('alpha',0) for r in all_results.values() if 'error' not in r]):+.2%} | "
                      f"平均IC: {np.mean(ics):+.4f}")
        lines.append(f"买入持有平均: {np.mean(bh_rets):.2%}")

    # 单维度汇总
    lines.append(f"\n{'─'*90}")
    lines.append(f"单维度IC排名:")
    dim_ics = {}
    for sym, r in all_results.items():
        if 'dimension_analysis' not in r:
            continue
        for dim, data in r['dimension_analysis'].items():
            dim_ics.setdefault(dim, []).append(data.get('ic', 0))
    for dim, vals in sorted(dim_ics.items(), key=lambda x: -np.mean(x[1])):
        lines.append(f"  {dim:<10} 平均IC: {np.mean(vals):+.4f} "
                      f"(最好:{max(vals):+.4f} 最差:{min(vals):+.4f})")

    return '\n'.join(lines)
