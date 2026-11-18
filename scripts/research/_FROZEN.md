# research/ 冻结清单（R2减法, 2026-09-11立）

> 诊断书原则: 不删只封。以下一次性回测脚本已完成使命,保留用于复现,不再维护/扩展。
> 新回测一律新建文件,禁止改动冻结清单内的脚本。

## 活跃（日常管道引用,继续维护）
- backtest_div_double_bottom.py（双底算法本体,db_regime复用其引擎）
- backtest_db_regime.py / backtest_breakout_entry.py / backtest_divergence_sector_regime.py（9/11三连,新）
- backtest_smallcap_mine.py（Phase1.1,截面缓存可复用）
- backtest_reco_policy.py（推荐政策闸门）
- backtest_signal_accumulator.py（累积器验证）

## 冻结（一次性验证,历史结论已进memory）
- 其余约100个: 二阶导族(derivative_*)/四因子(four_factor)/波浪(wave_position)/挤压抑(squeeze)/绿珠(green_staircase)/sector_strength/tcl系列/spring系列/turtle_fusion 等
- 冻结原因: 结论已落memory或已证伪,代码无日常引用
- 例外条款: 若需复用某冻结脚本的信号引擎(如db_regime复用double_bottom),import只读不改
