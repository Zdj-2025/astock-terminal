#!/usr/bin/env python3
"""每日扫描统一入口 — 9 scanner 串联 + 推荐闭环跟踪。

流程:
  1. stop_loss_check          持仓止损检查(破位即走)
  2. sector_heat_scanner      行业热度(个股聚合等权)
  3. mainline_leader_scanner  主线龙头(技术买点 × 基本面质量)
  4. breakout_scanner         突破(平台+放量+股性验证)
  5. divergence_scanner       底背离量能确认(超跌反弹; 2026H2复测逆势增强+2.18/5日,下跌期最强买入信号)
  # 6. smart_money_scanner 已出队(2026-11-14审计#5:族卡-2.2%连续8周负超额;独立追踪器cron 17:45保留)
  7. catalyst_scanner         催化驱动(预增>30%; ⚠️2026H2追买已证伪, 名单仅信息参考)
  8. defensive_setup_scanner  防御抗跌(回踩+缩量+盈利+主线)
  9. track_recommendations    推荐闭环跟踪

每日收盘后跑(需先update_all_kline到当日)。约8-12分钟。
用法: cd stock && PYTHONPATH=. python scripts/screener/daily_scan_all.py
"""
import subprocess
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCANNERS = [
    ('post_collection_check.py --fix', '🔬采集后核查(数据新鲜度+失效线+豁免待办·2026-09-30用户立规)'),
    ('stop_loss_check.py', '⚠️持仓止损检查(破位即走🔴纪律第一)'),
    ('sector_heat_scanner.py', '🔥行业热度(业界标准·个股聚合等权,真实主线轮动)'),
    ('sentiment_thermometer.py', '🌡️情绪温度计(涨停池四指标+绝对量维度,涨停<40只降级)'),
    ('mainline_leader_scanner.py', '主线龙头(技术×基本面)'),
    # 2026-08-28降级: breakout信号衰减(t=0.9不显著+滚动<10分位🔴),移出核心队列
    # ('breakout_scanner.py', '突破(股性验证)⚠️已降级'),
    ('golden_cross_scanner.py', '✨金叉信号(MA5上穿MA10,2026-08-28新增,t=5.1有效)'),
    ('divergence_scanner.py', '📉底背离量能确认(超跌反弹·下跌期逆势增强2026H2超额+2.18)'),
    ('oversold_reversal_scanner.py', '⚡超跌反转(20日跌>15%+低位+阳线,t=6.2)'),
    # ('relative_strength_scanner.py', '💪相对强度'),  # 2026-09-11 R1撤出每日队列: 480只命中纯噪声无验证,文件保留冻结
    # smart_money_scanner 2026-11-14出队: 12个月回测超额5d-0.86%全月皆负; 前向验证由smart_tracker.py(cron 17:45)独立承担
    ('catalyst_scanner.py', '催化驱动(预增>30%×主线×聪明钱·⚠️追买已证伪仅信息)'),
    ('defensive_setup_scanner.py', '防御抗跌(回踩+缩量+盈利+主线,暴跌日15/15跑赢创业板)'),
    ('track_recommendations.py', '推荐闭环跟踪'),
    ('scan_history.py', '📊信号累积器(横向×纵向·权重引擎·策略簇修正)'),  # 内部运行accumulate()
    ('top_structure_scanner.py --portfolio', '🔺持仓顶部结构警示(提醒非指令·2026-09-22用户设计)'),
]
NOISE = ['pkg_resources is deprecated', 'UserWarning', 'import pkg_resources',
         'login success', 'logout success']


def is_progress_bar(ln):
    """识别tqdm/进度条行(含|...it/s或|%|)。"""
    if 'it/s' in ln or 'it]' in ln:
        return True
    if '|' in ln and '%' in ln and '[' in ln:
        return True
    return False


def main():
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    print(f"📊 每日扫描统一入口 | {len(SCANNERS)} scanner 串联\n")
    t0 = time.time()
    for i, (script, desc) in enumerate(SCANNERS, 1):
        print(f"\n{'=' * 72}\n[{i}/{len(SCANNERS)}] ▶ {desc} ({script})\n{'=' * 72}", flush=True)
        parts = script.split()
        path = ROOT / 'scripts/screener' / parts[0]
        try:
            r = subprocess.run([sys.executable, str(path)] + parts[1:], capture_output=True, text=True,
                               env=env, cwd=str(ROOT))
            for ln in (r.stdout or '').split('\n'):
                if any(n in ln for n in NOISE):
                    continue
                if is_progress_bar(ln):
                    continue
                print(ln)
            if r.returncode != 0:
                print(f"  ⚠️ {script} 返回码{r.returncode}: {(r.stderr or '')[-300:]}")
        except Exception as e:
            print(f"  ❌ {script} 错误: {e}")
    # 总结器(2026-08-27用户需求:结论直给+焦点清单收敛多角度)
    try:
        # [10] 信号健康度(推荐政策v2026-08-31: 60日胜率/超额<阈值自动降级推荐)
        try:
            print()
            from scripts.analyzers.signal_decay import signal_health_dashboard, decay_slope_alert
            signal_health_dashboard()
            decay_slope_alert()  # 2026-10-01 P1-4: 衰减斜率警报(事前预警)
        except Exception as e:
            print(f'⚠️ signal_decay失败: {e}')

        from scripts.screener.scan_digest import main as digest_main
        digest_main()
    except Exception as e:
        print(f"⚠️ 总结器跳过: {e}")
    print(f"\n{'=' * 72}\n✓ 全部完成, 耗时{time.time() - t0:.0f}秒")


if __name__ == '__main__':
    main()
