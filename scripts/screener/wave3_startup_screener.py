# ⚠️ FROZEN(2026-09-11 R2减法): 已并入背离一族,家族hub=divergence_scanner.py
# 本文件保留供回测复现,禁止扩展/接入日常管道. 生产路径(daily_scan_all/premarket)无引用.
# 依据: 8/28全信号审计(买跌族收敛到背离/超跌/金叉三个已验证信号)

"""W3刚启动选股器 — 三系统共振：波浪+缠论二买+MACD

找W3刚启动的股票，相当于缠论二买位置。
两阶段架构：Phase 1轻量扫描(全市场) → Phase 2缠论深度验证(TOP候选)

算法：
  1. 波浪：锚点低→W1/W2检测→W3启动阶段分类(pre_breakout/just_broke/early_w3)
  2. 缠论简化：不创新低+MACD底背离+Fibonacci理想回撤
  3. MACD：近期金叉+柱状图转正+快速MACD底背离
  4. 三系统组合评分，各≥30且组合≥50才通过

用法：
  python scripts/screener/screener_cli.py --template wave3_startup
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)


class Wave3StartupScreener:
    """W3刚启动扫描器 — 三系统共振。"""

    def __init__(self, client):
        self.client = client

    def scan(
        self,
        candidates: pd.DataFrame,
        lookback_days: int = 500,
        min_confidence: float = 50.0,
        top: int = 50,
        as_of_date: str = None,
    ) -> pd.DataFrame:
        """批量扫描候选股，找出W3刚启动的股票。

        Args:
            candidates: Layer 1 UniverseFilter输出
            lookback_days: K线回溯天数
            min_confidence: 最低组合分(0-100)
            top: 最多返回N只
        """
        trade_date = as_of_date or datetime.now().strftime("%Y%m%d")

        cached = self._load_cache(trade_date)
        if cached is not None and len(cached) > 0:
            print(f"  [W3Startup] 使用缓存({trade_date}): {len(cached)}只")
            filtered = cached[cached['w3s_composite'] >= min_confidence]
            return filtered.head(top)

        prefiltered = self._prefilter(candidates)
        print(f"  [W3Startup] 预过滤: {len(candidates)} → {len(prefiltered)}只")

        start_date = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)).strftime("%Y%m%d")
        results = []
        total = len(prefiltered)

        for i, row in prefiltered.iterrows():
            symbol = str(row['symbol'])
            ts_code = row['ts_code']
            name = row.get('name', symbol)

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [W3Startup] 扫描进度: {i+1}/{total} "
                      f"({len(results)}只W3刚启动候选)")

            try:
                df = self.client.get_kline(
                    symbol, freq="D", start=start_date, end=as_of_date, adj="qfq")
                if df is None or len(df) < 60:
                    continue

                info = self._detect_single(df)
                if info is not None:
                    record = row.to_dict()
                    record.update(info)
                    results.append(record)
            except Exception:
                continue

        if not results:
            print(f"  [W3Startup] 未找到W3刚启动候选股")
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('w3s_composite', ascending=False)

        self._save_cache(result_df, trade_date)

        filtered = result_df[result_df['w3s_composite'] >= min_confidence]
        print(f"  [W3Startup] 三系统检测: {total}只 → {len(result_df)}只W3启动 → "
              f"{len(filtered)}只达标(组合分≥{min_confidence})")

        return filtered.head(top)

    # ── 预过滤 ──

    def _prefilter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        df = candidates.copy()
        if 'pe_ttm' in df.columns:
            df = df[(df['pe_ttm'] > 0) | df['pe_ttm'].isna()]
        if 'turnover_rate' in df.columns:
            df = df[df['turnover_rate'] >= 1.0]
        if 'close' in df.columns:
            df = df[df['close'] >= 5.0]
        return df.reset_index(drop=True)

    # ── 核心检测 ──

    def _detect_single(self, df: pd.DataFrame) -> dict or None:
        """对单只股票做三系统检测。"""
        if len(df) < 60:
            return None

        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        dates = df['date'].values

        # Step 1: 找锚点低
        anchor_idx = self._find_anchor_low(low)
        if anchor_idx is None:
            return None
        anchor_price = low[anchor_idx]

        # Step 1.5: 前高压制检测 — 排除"回调反弹"误判为"W3刚启动"
        # 找锚点低之前的主升结构：之前最低→之前最高→锚点低
        # 如果之前已有一波主升，且当前价接近前高(距前高>-8%)，
        # 说明这不是底部W3刚启动，而是大级别回调后的反弹
        if anchor_idx > 30:
            prior_high = float(np.max(high[:anchor_idx]))
            prior_high_idx = int(np.argmax(high[:anchor_idx]))
            if prior_high_idx > 5:
                prior_low = float(np.min(low[:prior_high_idx]))
                prior_trend_gain = (prior_high / prior_low - 1) * 100
                current_close = float(close[-1])
                dist_to_prior_high = (current_close / prior_high - 1) * 100
                if prior_trend_gain > 20 and dist_to_prior_high > -8:
                    return None

        # Step 2: 找W1/W2
        w1_result = self._find_w1_w2(high, low, close, anchor_idx, anchor_price)
        if w1_result is None:
            return None
        w1_idx, w1_price, w2_idx, w2_price = w1_result

        # Step 3: W3启动阶段分类
        stage_info = self._classify_w3_startup(
            high, low, close, dates, volume,
            w1_idx, w1_price, w2_idx, w2_price,
            anchor_idx, anchor_price)
        if stage_info is None:
            return None

        # Step 4: 三系统评分
        wave_score = self._compute_wave_score(
            anchor_idx, anchor_price, w1_idx, w1_price,
            w2_idx, w2_price, stage_info, volume)

        chanlun_score = self._detect_simplified_ermai(
            close, high, low, volume, anchor_idx, w1_idx, w2_idx,
            anchor_price, w1_price, w2_price)

        macd_score = self._check_macd_bullish(close, w2_idx)

        composite = self._compute_composite(wave_score, chanlun_score, macd_score)

        # 通过条件：三系统各≥30，组合≥50
        if wave_score < 30 or chanlun_score < 30 or macd_score < 30:
            return None
        if composite < 50:
            return None

        current_close = float(close[-1])
        current_date = pd.Timestamp(dates[-1])
        dist_to_w1 = (current_close - w1_price) / w1_price * 100

        # MACD金叉标记
        dif, dea, hist = self._calc_macd(close, 12, 26, 9)
        macd_cross = False
        n = len(close)
        for j in range(max(0, n - 5), n):
            if j > 0 and dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                macd_cross = True
                break

        result = {
            'w3s_wave_score': round(wave_score, 1),
            'w3s_chanlun_score': round(chanlun_score, 1),
            'w3s_macd_score': round(macd_score, 1),
            'w3s_composite': round(composite, 1),
            'w3s_stage': stage_info['stage'],
            'w3s_dist_to_w1_pct': round(dist_to_w1, 2),
            'w3s_anchor_date': pd.Timestamp(dates[anchor_idx]).strftime('%Y-%m-%d'),
            'w3s_anchor_price': float(anchor_price),
            'w3s_w1_end_date': pd.Timestamp(dates[w1_idx]).strftime('%Y-%m-%d'),
            'w3s_w1_gain_pct': round((w1_price / anchor_price - 1) * 100, 1),
            'w3s_w2_end_date': pd.Timestamp(dates[w2_idx]).strftime('%Y-%m-%d'),
            'w3s_w2_retrace_pct': round(
                (w1_price - w2_price) / (w1_price - anchor_price) * 100, 1),
            'w3s_current_price': current_close,
            'w3s_current_date': current_date.strftime('%Y-%m-%d'),
            'w3s_macd_cross': macd_cross,
            'w3s_confirmed_ermai': None,
            # ── 新增：复盘改进 #001 ──
            'w3s_suggested_position_pct': self._suggest_position(composite),
            'w3s_limit_up_confirmed': self._check_limit_up_confirmation(close, volume, n),
            'w3s_signal_freshness': self._calc_signal_freshness(dates, w2_idx),
            'w3s_target_price': round(w1_price * 1.15, 2),  # 目标价=W1顶+15%
            'w3s_stop_loss_price': round(current_close * 0.92, 2),  # 止损-8%
            'w3s_expected_hold_days': max(5, 30 - stage_info.get('w3_days', 10)),  # 预期持有天数
        }
        return result

    # ── 波浪子系统 ──

    def _find_anchor_low(self, low: np.ndarray) -> int or None:
        """找起始低点（复用WaveScreener逻辑）。"""
        n = len(low)
        local_lows = argrelextrema(low, np.less, order=10)[0]
        if len(local_lows) == 0:
            return int(np.argmin(low))

        cutoff = n - 30
        valid = local_lows[local_lows <= cutoff]
        if len(valid) == 0:
            valid = local_lows

        prices_at_lows = low[valid]
        return int(valid[np.argmin(prices_at_lows)])

    def _find_w1_w2(self, high, low, close, start_idx, start_price):
        """检测W1顶和W2底（复用WaveScreener逻辑）。"""
        n = len(high)

        # W1顶
        search_end = min(start_idx + 120, n)
        seg_high = high[start_idx:search_end]
        local_peaks = argrelextrema(seg_high, np.greater, order=8)[0]
        if len(local_peaks) == 0:
            return None

        for peak_offset in local_peaks:
            peak_idx = start_idx + peak_offset
            peak_price = high[peak_idx]
            if (peak_price / start_price - 1) > 0.05:
                w1_idx = peak_idx
                w1_price = peak_price
                break
        else:
            return None

        # W2底
        search_end2 = min(w1_idx + 80, n)
        seg_low = low[w1_idx:search_end2]
        local_troughs = argrelextrema(seg_low, np.less, order=6)[0]
        if len(local_troughs) == 0:
            return None

        for trough_offset in local_troughs:
            trough_idx = w1_idx + trough_offset
            trough_price = low[trough_idx]
            if trough_price > start_price:
                w2_idx = trough_idx
                w2_price = trough_price
                break
        else:
            if len(local_troughs) > 0:
                trough_idx = w1_idx + local_troughs[0]
                trough_price = low[trough_idx]
                if trough_price > start_price * 0.98:
                    w2_idx = trough_idx
                    w2_price = trough_price
                else:
                    return None
            else:
                return None

        # W2回撤检查
        w1_range = w1_price - start_price
        if w1_range > 0:
            retrace = (w1_price - w2_price) / w1_range
            if retrace < 0.15 or retrace > 0.95:
                return None

        return (int(w1_idx), float(w1_price), int(w2_idx), float(w2_price))

    def _classify_w3_startup(self, high, low, close, dates, volume,
                              w1_idx, w1_price, w2_idx, w2_price,
                              anchor_idx, anchor_price):
        """W3启动阶段分类。"""
        n = len(close)
        current_close = float(close[-1])

        dist_to_w1 = (current_close - w1_price) / w1_price * 100
        w3_gain = (current_close - w2_price) / w2_price * 100

        # 阶段分类
        if dist_to_w1 >= 0 and w3_gain < 10:
            stage = 'just_broke'
        elif -5 <= dist_to_w1 < 0:
            stage = 'pre_breakout'
        elif 10 <= w3_gain <= 20:
            stage = 'early_w3'
        else:
            return None

        # 拒绝：W3涨幅>20%
        if w3_gain > 20:
            return None

        # 拒绝：W2之后价格仍在下跌趋势（所有近K线低于W2底）
        recent = high[-10:] if n >= 10 else high[w2_idx:]
        if len(recent) > 0 and np.max(recent) < w2_price:
            return None

        w3_days = n - 1 - w2_idx

        return {
            'stage': stage,
            'dist_to_w1_pct': dist_to_w1,
            'w3_gain_pct': w3_gain,
            'w3_days': w3_days,
        }

    def _compute_wave_score(self, anchor_idx, anchor_price,
                            w1_idx, w1_price, w2_idx, w2_price,
                            stage_info, volume):
        """波浪子系统评分(0-100)。"""
        score = 0.0

        # W2回撤质量 (0-30)
        w1_range = w1_price - anchor_price
        if w1_range > 0:
            retrace = (w1_price - w2_price) / w1_range
            if 0.382 <= retrace <= 0.618:
                score += 30
            elif 0.236 <= retrace <= 0.786:
                score += 20
            else:
                score += 5

        # 阶段接近度 (0-30)
        stage = stage_info['stage']
        if stage == 'just_broke':
            score += 30
        elif stage == 'pre_breakout':
            score += 25
        elif stage == 'early_w3':
            score += 20

        # W1强度 (0-20) — 涨幅适中最佳
        w1_gain = (w1_price / anchor_price - 1) * 100
        if 15 <= w1_gain <= 40:
            score += 20
        elif 10 <= w1_gain < 15 or 40 < w1_gain <= 60:
            score += 12
        else:
            score += 5

        # 量能确认 (0-20)
        n = len(volume)
        recent_vol = volume[-5:].mean() if n >= 5 else volume.mean()
        w2_end = min(w2_idx + 5, n)
        w2_vol = volume[w2_idx:w2_end].mean() if w2_end > w2_idx else 1
        if w2_vol > 0 and recent_vol > w2_vol * 1.2:
            score += 20
        elif w2_vol > 0 and recent_vol > w2_vol:
            score += 10

        return round(min(score, 100), 1)

    # ── 缠论简化子系统 ──

    def _detect_simplified_ermai(self, close, high, low, volume, anchor_idx, w1_idx, w2_idx,
                                  anchor_price, w1_price, w2_price):
        """轻量二买检测(0-100)。

        基于全ChanLunEngine验证校准的五条件（按判别力排序）：
        1. W2底部MACD状态 (0-30) — DIF正值/转正是最佳T2判别因子
        2. W2后量能恢复 (0-25) — 近期均量 vs W2区均量
        3. W2底结构强度 (0-20) — W2底距锚点的高度
        4. W2后反弹速度 (0-15) — V型反转确认底部
        5. MACD双参数背离 (0-10) — 次要确认
        """
        score = 0.0
        n = len(close)

        # 条件1：W2底部MACD状态 (0-30)
        dif, dea, hist = self._calc_macd(close, 12, 26, 9)
        w2_dif = float(dif[w2_idx])
        w2_hist = float(hist[w2_idx])
        if w2_dif > 0 and w2_hist > 0:
            score += 30
        elif w2_dif > 0:
            score += 22
        elif w2_dif > -0.5 and w2_idx > 0:
            prev_dif = float(dif[max(0, w2_idx - 3)])
            if w2_dif > prev_dif:
                score += 15
            else:
                score += 5

        # 条件2：W2后量能恢复 (0-25)
        w2_vol_start = max(0, w2_idx - 3)
        w2_vol_end = min(n, w2_idx + 5)
        w2_avg_vol = float(np.mean(volume[w2_vol_start:w2_vol_end]))
        recent_avg_vol = float(np.mean(volume[-5:])) if n >= 5 else float(volume[-1])
        if w2_avg_vol > 0:
            vol_ratio = recent_avg_vol / w2_avg_vol
            if vol_ratio >= 2.0:
                score += 25
            elif vol_ratio >= 1.5:
                score += 18
            elif vol_ratio >= 1.0:
                score += 10

        # 条件3：W2底结构强度 (0-20)
        w1_range = w1_price - anchor_price
        if w1_range > 0:
            w2_lift = (w2_price - anchor_price) / w1_range
            if w2_lift > 0.6:
                score += 20
            elif w2_lift > 0.382:
                score += 12
            elif w2_lift > 0.2:
                score += 5

        # 条件4：W2后反弹速度 (0-15)
        recovery_window = min(10, n - w2_idx - 1)
        if recovery_window >= 3:
            recovery_high = float(np.max(high[w2_idx:w2_idx + recovery_window]))
            recovery_pct = (recovery_high - w2_price) / w2_price * 100
            if recovery_pct >= 15:
                score += 15
            elif recovery_pct >= 8:
                score += 10
            elif recovery_pct >= 3:
                score += 5

        # 条件5：MACD双参数背离 (0-10)
        dif_fast, dea_fast, _ = self._calc_macd(close, 5, 34, 5)
        a_start = max(0, anchor_idx - 5)
        a_end = min(n, anchor_idx + 5)
        w_start = max(0, w2_idx - 5)
        w_end = min(n, w2_idx + 5)
        macd_div_count = 0
        if a_end > a_start and w_end > w_start:
            anchor_min_dif = float(np.min(dif[a_start:a_end]))
            w2_min_dif = float(np.min(dif[w_start:w_end]))
            if anchor_min_dif < 0 and w2_min_dif > anchor_min_dif:
                macd_div_count += 1
            anchor_min_dif_f = float(np.min(dif_fast[a_start:a_end]))
            w2_min_dif_f = float(np.min(dif_fast[w_start:w_end]))
            if anchor_min_dif_f < 0 and w2_min_dif_f > anchor_min_dif_f:
                macd_div_count += 1
        score += 10 if macd_div_count == 2 else (5 if macd_div_count == 1 else 0)

        return round(min(score, 100), 1)

    # ── MACD子系统 ──

    def _check_macd_bullish(self, close, w2_idx):
        """MACD看涨确认(0-100)。"""
        score = 0.0
        n = len(close)

        # 标准MACD(12,26,9)
        dif, dea, hist = self._calc_macd(close, 12, 26, 9)

        # 1. 近期金叉 (0-40)
        golden_cross = False
        for j in range(max(0, n - 5), n):
            if j > 0 and dif[j] > dea[j] and dif[j - 1] <= dea[j - 1]:
                golden_cross = True
                break
        if golden_cross:
            score += 40

        # 2. 柱状图 (0-25)
        if hist[-1] > 0:
            score += 25
        elif n >= 2 and hist[-1] > hist[-2]:
            score += 15

        # 3. 快速MACD(5,34,5)底背离 (0-35)
        dif_fast, dea_fast, hist_fast = self._calc_macd(close, 5, 34, 5)
        search_start = max(0, w2_idx - 40)
        search_end = max(0, w2_idx - 3)
        if search_end > search_start + 3:
            seg = dif_fast[search_start:search_end]
            prior_macd_low_idx = search_start + int(np.argmin(seg))
            if (close[w2_idx] <= close[prior_macd_low_idx] and
                    dif_fast[w2_idx] > dif_fast[prior_macd_low_idx]):
                score += 35
            elif dif_fast[w2_idx] > dif_fast[prior_macd_low_idx]:
                score += 15

        return round(min(score, 100), 1)

    def _calc_macd(self, close, fast=12, slow=26, signal=9):
        """计算MACD，返回(DIF, DEA, histogram)。"""
        close_s = pd.Series(close, dtype=float)
        ema_fast = close_s.ewm(span=fast, adjust=False).mean()
        ema_slow = close_s.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = (dif - dea) * 2
        return dif.values, dea.values, hist.values

    # ── 组合评分 ──

    def _compute_composite(self, wave_score, chanlun_score, macd_score):
        """三系统组合评分。"""
        return round(
            wave_score * 0.40 + chanlun_score * 0.30 + macd_score * 0.30, 1)

    # ── Phase 2：缠论深度验证 ──

    def verify_with_chanlun(self, results: pd.DataFrame) -> pd.DataFrame:
        """对TOP候选运行完整缠论分析，确认二买。"""
        if results.empty:
            return results

        try:
            from scripts.analyzers.chanlun.engine import ChanLunEngine
        except ImportError:
            print("  [W3Startup] 缠论引擎不可用，跳过深度验证")
            return results

        engine = ChanLunEngine()
        results = results.copy()
        results['w3s_confirmed_ermai'] = False

        verified = 0
        for idx, row in results.iterrows():
            symbol = str(row.get('symbol', ''))
            if not symbol:
                continue
            try:
                chan_result = engine.analyze(symbol, levels=["daily"])
                daily = chan_result.daily
                if daily and daily.bs_points:
                    for bsp in daily.bs_points[-5:]:
                        if bsp.is_buy and any(
                                'T2' in t for t in bsp.bsp_types):
                            results.at[idx, 'w3s_confirmed_ermai'] = True
                            verified += 1
                            break
            except Exception:
                continue

        print(f"  [W3Startup] 缠论验证: {verified}/{len(results)}只确认二买")
        return results

    # ── 缓存 ──

    def _cache_dir(self):
        return Path(project_root) / "data" / "cache" / "wave3_startup_screener"

    def _load_cache(self, trade_date: str):
        cache_path = self._cache_dir() / f"{trade_date}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return None

    def _save_cache(self, results: pd.DataFrame, trade_date: str):
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(cache_dir / f"{trade_date}.csv", index=False)

    # ── 报告格式化 ──

    @staticmethod
    def format_report(results: pd.DataFrame, meta: dict) -> str:
        """格式化控制台输出。"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"  W3刚启动选股报告  {meta.get('trade_date', '')}")
        lines.append(f"  策略: 三系统共振(波浪+缠论二买+MACD) | "
                     f"候选池: {meta.get('candidates_count', 0)}只 → "
                     f"W3启动: {len(results)}只")
        lines.append("=" * 80)

        if results.empty:
            lines.append("  未找到符合W3刚启动条件的股票。")
            return '\n'.join(lines)

        lines.append("")
        lines.append(f"  {'排名':>4}  {'代码':8}  {'名称':8}  "
                     f"{'组合':>5}  {'波浪':>5}  {'缠论':>5}  {'MACD':>5}  "
                     f"{'阶段':14}  {'距W1顶':>7}  {'当前价':>8}")
        lines.append("  " + "-" * 74)

        stage_cn = {
            'pre_breakout': '蓄势待发',
            'just_broke': '刚突破W1',
            'early_w3': 'W3早期',
        }

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            symbol = str(row.get('symbol', ''))
            name = str(row.get('name', ''))[:6]
            comp = row.get('w3s_composite', 0)
            wave = row.get('w3s_wave_score', 0)
            cl = row.get('w3s_chanlun_score', 0)
            macd = row.get('w3s_macd_score', 0)
            stage = stage_cn.get(row.get('w3s_stage', ''), row.get('w3s_stage', ''))
            dist = row.get('w3s_dist_to_w1_pct', 0)
            price = row.get('w3s_current_price', 0)

            lines.append(f"  {rank:4d}  {symbol:8s}  {name:8s}  "
                         f"{comp:5.1f}  {wave:5.1f}  {cl:5.1f}  {macd:5.1f}  "
                         f"{stage:14s}  {dist:+6.1f}%  {price:8.2f}")

        lines.append("")
        lines.append("  注意: 三系统共振量化结果，不构成投资建议。")
        return '\n'.join(lines)

    # ── 复盘改进 #001 新增方法 ──

    @staticmethod
    def _suggest_position(composite: float) -> float:
        """根据组合分建议仓位（占总资金池比例）。

        复盘 #001 教训：弱信号小仓位或不动。
        """
        if composite >= 80:
            return 0.10  # 10%
        elif composite >= 65:
            return 0.05  # 5%
        elif composite >= 50:
            return 0.03  # 3%
        else:
            return 0.0   # 不建议建仓

    @staticmethod
    def _check_limit_up_confirmation(close, volume, n) -> str:
        """检查近期是否有涨停板确认。

        复盘 #001 教训：涨停板≠趋势启动，需要次日确认。
        检测最近5天是否有涨幅>9%的K线，以及次日走势。
        """
        if n < 3:
            return "数据不足"

        for i in range(max(1, n - 5), n):
            chg = (close[i] - close[i-1]) / close[i-1]
            if chg > 0.095:  # 涨停（创业板/科创板20%，这里用保守值）
                # 检查次日是否确认
                if i + 1 < n:
                    next_chg = (close[i+1] - close[i]) / close[i]
                    if next_chg > 0:
                        return f"涨停后次日确认(涨幅{next_chg*100:.1f}%)"
                    elif next_chg > -0.02:
                        return f"涨停后次日企稳({next_chg*100:.1f}%)"
                    else:
                        return f"涨停后次日回落({next_chg*100:.1f}%)⚠️"
                else:
                    return "涨停当日（待确认）"
        return "近期无涨停"

    @staticmethod
    def _calc_signal_freshness(dates, w2_idx) -> str:
        """计算信号新鲜度（W2底距今多久）。

        复盘 #001 教训：信号延迟会导致买入点偏晚。
        """
        n = len(dates)
        if w2_idx >= n:
            return ""
        days = n - 1 - w2_idx
        if days <= 2:
            return f"新鲜({days}天前)"
        elif days <= 5:
            return f"较新({days}天前)"
        elif days <= 10:
            return f"偏旧({days}天前)⚠️"
        else:
            return f"过期({days}天前)❌"
