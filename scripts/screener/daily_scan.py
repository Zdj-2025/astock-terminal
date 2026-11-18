#!/usr/bin/env python3
"""每日全市场扫描 — v3.5 三分支筛选 + 多周期确认。

输出：趋势加速 + 底部反转(L2/L3) + 顶部卖出信号。
增强：多周期(D/60min/15min)导数一致性 + 统计置信度。
用法：
  PYTHONPATH=. python scripts/screener/daily_scan.py [--min-score 4.0] [--top 20]
  PYTHONPATH=. python scripts/screener/daily_scan.py --no-mtf  # 不使用多周期确认
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.screener.v3_derivative_scorer import V3DerivativeScorer
from scripts.screener.strategy_engine import StrategyEngine
from scripts.screener.multi_timeframe_scorer import MultiTimeframeScorer


def main():
    parser = argparse.ArgumentParser(description="每日全市场v3.5扫描(+多周期确认)")
    parser.add_argument("--min-score", type=float, default=4.0, help="最低raw分")
    parser.add_argument("--top", type=int, default=20, help="每个分支输出Top N")
    parser.add_argument("--data-root", type=str, default="data/store")
    parser.add_argument("--no-mtf", action="store_true", help="禁用多周期确认")
    parser.add_argument("--wave-position", action="store_true",
                        help="启用波浪位置子模型（5浪下跌→W2/W3买点）")
    parser.add_argument("--macd-momentum", action="store_true",
                        help="对accel+reversal候选股启用MACD动量二次评分")
    args = parser.parse_args()

    store = Path(args.data_root)
    scorer = V3DerivativeScorer(data_root=store)
    all_codes = sorted(
        [d.name for d in (store / "kline").iterdir() if d.is_dir() and (d / "D.parquet").exists()]
    )
    # 拿名称映射 + 排除ST(2026-06-17: 000903 ST云动混入Top5的教训)
    name_map = {}
    try:
        from scripts.core.tushare_client import TushareClient
        _sl = TushareClient().get_stock_list()
        for _, _r in _sl.iterrows():
            _code = str(_r.get('symbol', _r.get('ts_code', ''))).split('.')[0]
            name_map[_code] = _r.get('name', '')
        st_codes = {c for c, n in name_map.items() if 'ST' in str(n)}
        codes = [c for c in all_codes if c not in st_codes]
        print(f"(排除ST {len(all_codes)-len(codes)}只(全市场ST池{len(st_codes)}), 扫描{len(codes)}只)")
    except Exception:
        codes = all_codes

    accel, reversal, top_sell = scorer.score_dual_branch(codes, min_score=args.min_score)

    # ── 多周期导数确认 ──
    mtf = MultiTimeframeScorer(data_root=store) if not args.no_mtf else None
    if mtf:
        if len(accel) > 0:
            accel = mtf.augment_v3_batch(accel)
        if len(reversal) > 0:
            reversal = mtf.augment_v3_batch(reversal)
        if len(top_sell) > 0:
            top_sell = mtf.augment_v3_batch(top_sell)

    # ── 新策略引擎：底部反转策略评分 ──
    engine = StrategyEngine(scorer)
    reversal_enhanced = reversal.copy()
    if len(reversal_enhanced) > 0:
        strategy_scores = []
        for _, row in reversal_enhanced.iterrows():
            # 从V3结果重建V3Score
            from scripts.screener.v3_derivative_scorer import V3Score
            v3s = V3Score(
                f1=row.get('f1', 0), f2=row.get('f2', 0),
                details={
                    'dh180': row.get('dh180', 0),
                    'vol_ratio': row.get('vol_ratio', 1.0),
                    'minute_details': row.get('minute_details', {}),
                }
            )
            ss = engine.score_bottom_reversal(v3s)
            strategy_scores.append(ss.total)
        reversal_enhanced['strategy_score'] = strategy_scores

    # ── 波浪位置子模型 ──
    wp_results = None
    if args.wave_position:
        from scripts.screener.wave_position_screener import WavePositionScreener
        wp_scanner = WavePositionScreener(data_root=store)
        wp_codes = codes
        wp_results = wp_scanner.scan(wp_codes, min_score=40, top=args.top)

    # ── MACD动量二次评分 ──
    macd_screener = None
    if args.macd_momentum:
        from scripts.screener.macd_momentum_screener import MACDMomentumScreener
        macd_screener = MACDMomentumScreener()

    def _apply_macd_score(df_in: pd.DataFrame) -> pd.DataFrame:
        """对候选股DataFrame追加macd_total和macd_detail列"""
        if macd_screener is None or len(df_in) == 0:
            return df_in
        scores = []
        details = []
        for _, row in df_in.iterrows():
            code = row["code"]
            parquet_path = store / "kline" / code / "D.parquet"
            if not parquet_path.exists():
                scores.append(0)
                details.append("")
                continue
            try:
                daily_df = pd.read_parquet(parquet_path)
                r = macd_screener.score(code, daily_df)
                scores.append(r.total_score)
                details.append(r.detail_desc)
            except Exception:
                scores.append(0)
                details.append("")
        df_out = df_in.copy()
        df_out["macd_total"] = scores
        df_out["macd_detail"] = details
        return df_out

    if macd_screener:
        accel = _apply_macd_score(accel)
        reversal_enhanced = _apply_macd_score(reversal_enhanced)

    # ── ATR止损价计算 ──────────────────────────────────────────────
    def _get_atr_stop(code: str, store: Path, n_atr: float = 2.0) -> tuple[float, float, float]:
        """快速读取最新收盘价和ATR，返回 (price, atr, stop_price)。"""
        try:
            path = store / "kline" / code / "D.parquet"
            if not path.exists():
                return 0, 0, 0
            df = pd.read_parquet(path)
            dt_col = "trade_date" if "trade_date" in df.columns else "date"
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.sort_values(dt_col).reset_index(drop=True)
            if len(df) < 22:
                return 0, 0, 0
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            # True Range
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr = np.mean(tr[-20:])  # 20日ATR
            price = close[-1]
            stop = price - n_atr * atr
            return price, atr, stop
        except Exception:
            return 0, 0, 0

    # ── 大盘阶段判断 ──────────────────────────────────────────────
    from scripts.screener.market_regime import MarketRegimeDetector
    regime_detector = MarketRegimeDetector(data_root=store)
    regime = regime_detector.detect()

    # ── 格式化输出 ──────────────────────────────────────────────
    lines = []
    mtf_tag = "+MTF" if mtf else ""
    macd_hdr = "+MACD动量" if macd_screener else ""
    lines.append(f"📊 **v3.5 全市场扫描{mtf_tag}{macd_hdr}** | {len(codes)}只")
    lines.append(f"🏛️ 大盘: {regime.label} 置信{regime.confidence:.0%} | "
                 f"f'={regime.f1:+.4f} f''={regime.f2:+.4f} | "
                 f"MA20{'✅' if regime.above_ma20 else '❌'} MA60{'✅' if regime.above_ma60 else '❌'} | "
                 f"均线:{regime.ma_alignment} 量:{regime.vol_trend}")
    # 策略权重提示
    w = regime.strategy_weights
    top_strat = max(w, key=w.get)
    lines.append(f"📌 当前推荐: **{top_strat}**({w[top_strat]:.0%}) "
                 f"| 弹簧{w.get('弹簧起涨',0):.0%} 底反{w.get('底部反转',0):.0%} "
                 f"趋势{w.get('趋势加速',0):.0%} 空仓{w.get('空仓等待',0):.0%}")
    lines.append("─" * 40)
    lines.append(f"🚀趋势加速 {len(accel)}只 | 🔄底部反转 {len(reversal)}只 | ⚠️顶部卖出 {len(top_sell)}只")
    lines.append("─" * 40)

    # 趋势加速 Top N
    lines.append(f"\n🚀 **趋势加速 Top{min(args.top, len(accel))}**")
    sort_col = "v3_total" if "v3_total" in accel.columns else "v3_raw"
    # 加载突破检测器
    from scripts.core.breakout_detector import BreakoutDetector
    from scripts.core.turtle_position import TurtlePosition
    bd = BreakoutDetector(min_volume_ratio=1.0)  # daily_scan不强求量能
    tp_calc = TurtlePosition(total_capital=425510)
    # accel 已由 score_dual_branch 按 _accel_rank 排序(含趋势阶段/数据存疑加权), 用 head() 取 Top
    try:
        from scripts.core.tushare_client import TushareClient
        recent_td = TushareClient().get_recent_trade_day()
    except Exception:
        recent_td = None  # 离线环境降级, 跳过新鲜度检查
    for i, (_, r) in enumerate(accel.head(args.top).iterrows()):
        md = r.get("minute_details", {})
        min_tag = ""
        if isinstance(md, dict):
            if md.get("accel_bull"):
                min_tag = " [60min加速+多头]"
            elif md.get("accel_confirm"):
                min_tag = " [60min加速]"
        # 趋势阶段标签(回测修正0617: 不代表收益排序, 仅择时提示)
        phase_tag = ""
        _phase = r.get("trend_phase", "")
        if _phase == "延伸期": phase_tag = " ⚠️延伸期·追高择时"  # 动量延续强但买点要等回踩
        elif _phase == "启动期": phase_tag = " ·启动期"
        elif _phase == "中继期": phase_tag = " ·中继期"
        # 数据存疑标签(P0-1)
        suspect_tag = " 🚫数据存疑" if r.get("data_suspect") else ""
        # 数据新鲜度(P0-1): parquet最新日 vs 最近交易日
        stale_tag = ""
        try:
            _pp = store / "kline" / r['code'] / "D.parquet"
            if _pp.exists():
                _dfx = pd.read_parquet(_pp)
                _dc = "trade_date" if "trade_date" in _dfx.columns else "date"
                _latest = pd.to_datetime(_dfx[_dc]).max().strftime("%Y-%m-%d")
                if recent_td and _latest.replace("-", "") < recent_td:
                    stale_tag = f" 🕒旧数据({_latest})"
        except Exception:
            pass
        # MTF 确认标签
        mtf_tag = ""
        if "mtf_level" in r.index:
            mtf_level = r.get("mtf_level", "")
            mtf_conf = r.get("mtf_confidence", 0)
            if mtf_level and mtf_level != "数据不足":
                mtf_tag = f" [{mtf_level} {mtf_conf:.0%}]"
            elif mtf_level == "数据不足":
                mtf_tag = " [无分钟数据]"
        # CUSUM 确认标签
        cusum_tag = ""
        cusum_info = r.get("cusum", {})
        if isinstance(cusum_info, dict) and cusum_info.get("reason"):
            cusum_tag = f" [CUSUM:{cusum_info.get('confidence','')}]"
        # MACD动量标签
        macd_tag = ""
        if "macd_total" in r.index and r.get("macd_total", 0) > 0:
            macd_tag = f" MACD={r['macd_total']:.0f}/100"
        # 分钟顶背离检测(P1, 仅Top5把关避免全量慢; buy_timing纪律自动化)
        div_tag = ""
        if i < 5:
            try:
                from scripts.core.divergence_detector import check_minute_divergence
                div_r = check_minute_divergence(r['code'])
                if div_r["has_top_div"]:
                    div_tag = " 🚫分钟背离" + ("(数据旧)" if div_r["stale"] else "")
                elif div_r["hist_decay"]:
                    div_tag = " ⚠️动能衰竭"
            except Exception:
                pass
        # TCL回踩买点检测(2026-06-17, TCL模式定位为V3趋势加速的回踩择时)
        # V3选股+TCL择时+主线 = TCL 6/1成功路径。tcl_pattern用V3 savgol f1, 无未来
        tcl_tag = ""
        try:
            from scripts.screener.tcl_pattern import compute_indicators, tcl_mask
            _tdf = pd.read_parquet(store / "kline" / r['code'] / "D.parquet")
            _tdf['date'] = pd.to_datetime(_tdf['date'])
            _tdf = compute_indicators(_tdf)
            if bool(tcl_mask(_tdf).iloc[-1]):
                tcl_tag = " 🎯TCL回踩买点"
        except Exception:
            pass
        lines.append(
            f"  {i+1}. {r['code']} {name_map.get(r['code'],'')} raw={r['v3_raw']:.1f} "
            f"f1={r['f1']:+.3f} ret20={r.get('ret20_0',0):+.1f}%"
            f"{phase_tag}{suspect_tag}{stale_tag}{tcl_tag}{div_tag}{min_tag}{mtf_tag}{cusum_tag}{macd_tag}"
        )
        # ATR止损价 + 突破信号 + 仓位建议
        px, atr_val, sp = _get_atr_stop(r['code'], store)
        # 突破检测
        bk_tag = ""
        pos_tag = ""
        if px > 0 and atr_val > 0:
            try:
                parquet_path = store / "kline" / r['code'] / "D.parquet"
                bk_df = pd.read_parquet(parquet_path)
                dt_col = "trade_date" if "trade_date" in bk_df.columns else "date"
                bk_df[dt_col] = pd.to_datetime(bk_df[dt_col])
                bk_df = bk_df.sort_values(dt_col).reset_index(drop=True)
                bk_r = bd.detect(bk_df, v3_score=r.get('v3_raw', 7.0), atr=atr_val)
                from scripts.core.breakout_detector import SignalStrength
                if bk_r.signal_strength == SignalStrength.STRONG_BUY:
                    bk_tag = " 📈突破20日高"
                elif bk_r.signal_strength == SignalStrength.BUY:
                    bk_tag = " 📈突破"
                # 仓位建议（1单位）
                pos_r = tp_calc.calculate(
                    code=r['code'], price=px, atr=atr_val,
                    v3_score=r.get('v3_raw', 7.0),
                    signal_type='uptrend',
                )
                if pos_r.unit_shares > 0:
                    pos_tag = f" | 买{pos_r.unit_shares}股"
            except Exception:
                pass
        if px > 0:
            lines.append(f"      💀止损={sp:.2f} (ATR={atr_val:.3f}, 2ATR止损){bk_tag}{pos_tag}")

    # 底部反转
    if len(reversal_enhanced) > 0:
        lines.append(f"\n🔄 **底部反转** ({len(reversal_enhanced)}只)")
        if "reversal_level" in reversal_enhanced.columns:
            L3 = reversal_enhanced[reversal_enhanced["reversal_level"] == "L3确认买入"]
            L2 = reversal_enhanced[reversal_enhanced["reversal_level"] == "L2关注"]
            # 新策略高分(>=7.5)标记
            if "strategy_score" in reversal_enhanced.columns:
                high_score = reversal_enhanced[reversal_enhanced["strategy_score"] >= 7.5]
                if len(high_score) > 0:
                    lines.append(f"  ⭐ **新策略高分(≥7.5)** ({len(high_score)}只, 回测胜率77.2%)")
                    for i, (_, r) in enumerate(high_score.head(10).iterrows()):
                        mtf_tag = ""
                        if "mtf_level" in r.index:
                            ml = r.get("mtf_level", "")
                            mc = r.get("mtf_confidence", 0)
                            if ml and ml != "数据不足":
                                mtf_tag = f" MTF[{ml} {mc:.0%}]"
                            elif ml == "数据不足":
                                mtf_tag = " MTF[无分钟数据]"
                        macd_tag = ""
                        if "macd_total" in r.index and r.get("macd_total", 0) > 0:
                            macd_tag = f" MACD={r['macd_total']:.0f}/100"
                        lines.append(
                            f"  {i+1}. {r['code']} {name_map.get(r['code'],'')} 策略分={r['strategy_score']:.1f} "
                            f"raw={r['v3_raw']:.1f} "
                            f"f1={r['f1']:+.3f} dh180={r.get('dh180',0):+.1f}% "
                            f"vol={r.get('vol_ratio',0):.2f}{mtf_tag}{macd_tag}"
                        )
                        px, atr_val, sp = _get_atr_stop(r['code'], store)
                        if px > 0:
                            lines.append(f"      💀止损={sp:.2f} (现价={px:.2f}, ATR={atr_val:.3f})")
            if len(L3) > 0:
                lines.append(f"  **L3确认买入** ({len(L3)}只)")
                for i, (_, r) in enumerate(L3.head(10).iterrows()):
                    ss = r.get('strategy_score', 0)
                    ss_tag = f" 策略分={ss:.1f}" if ss else ""
                    macd_tag = ""
                    if "macd_total" in r.index and r.get("macd_total", 0) > 0:
                        macd_tag = f" MACD={r['macd_total']:.0f}/100"
                    lines.append(
                        f"  {i+1}. {r['code']} {name_map.get(r['code'],'')} raw={r['v3_raw']:.1f} "
                        f"f1={r['f1']:+.3f} dh180={r.get('dh180',0):+.1f}% "
                        f"macd={r.get('macd_hist',0):+.2f} vol={r.get('vol_ratio',0):.2f}"
                        f"{ss_tag}{macd_tag}"
                    )
            if len(L2) > 0:
                lines.append(f"  **L2关注** ({len(L2)}只)")
                for i, (_, r) in enumerate(L2.head(10).iterrows()):
                    ss = r.get('strategy_score', 0)
                    ss_tag = f" 策略分={ss:.1f}" if ss else ""
                    macd_tag = ""
                    if "macd_total" in r.index and r.get("macd_total", 0) > 0:
                        macd_tag = f" MACD={r['macd_total']:.0f}/100"
                    lines.append(
                        f"  {i+1}. {r['code']} {name_map.get(r['code'],'')} raw={r['v3_raw']:.1f} "
                        f"f1={r['f1']:+.3f} dh180={r.get('dh180',0):+.1f}%"
                        f"{ss_tag}{macd_tag}"
                    )
        else:
            for i, (_, r) in enumerate(reversal_enhanced.head(10).iterrows()):
                lines.append(
                    f"  {i+1}. {r['code']} raw={r['v3_raw']:.1f} "
                    f"f1={r['f1']:+.3f} dh180={r.get('dh180',0):+.1f}%"
                )

    # 顶部卖出
    if len(top_sell) > 0:
        lines.append(f"\n⚠️ **顶部卖出信号** ({len(top_sell)}只)")
        for i, (_, r) in enumerate(top_sell.head(10).iterrows()):
            lines.append(
                f"  {i+1}. {r['code']} {name_map.get(r['code'],'')} raw={r['v3_raw']:.1f} "
                f"f2={r['f2']:+.3f} ret20={r.get('ret20_0',0):+.1f}% "
                f"macd={r.get('macd_hist',0):+.2f}"
            )

    # 波浪位置子模型
    if wp_results is not None and len(wp_results) > 0:
        lines.append(f"\n🌊 **波浪位置子模型** ({len(wp_results)}只)")
        for i, (_, r) in enumerate(wp_results.iterrows()):
            if r["position"] == "w2_bottom":
                pos_label = "W2底部"
            else:
                pos_label = "W3突破"
            grade_icon = "⭐" if r["grade"] == "A" else ("✅" if r["grade"] == "B" else "📊")
            fib_tag = ""
            if r.get("w2_at_fib") and str(r.get("w2_at_fib", "")) != "nan":
                fib_tag = f" Fib={r.get('w2_at_fib', '')}"
            lines.append(
                f"  {i+1}. {grade_icon} {r['code']} [{r['grade']}级] "
                f"{pos_label} 综合={r['composite_score']:.0f}"
                f" W1涨幅={r.get('w1_gain_pct', 0):.0f}%"
                f"{fib_tag} "
                f"突破={r.get('breakout_pct', 0):+.1f}% "
                f"前期跌={r.get('prior_decline', 0):.0f}%"
            )

    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    main()
