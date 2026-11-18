#!/usr/bin/env python3
"""策略权重引擎测试 v2 — 简洁写法,全面覆盖"""
import sys, os, json
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.core.strategy_weight_engine import (
    WeightEngine, STRATEGY_REGISTRY, MAX_WEIGHT, BASELINE_WEIGHT
)

passed = failed = 0
def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} {detail}")

print("═══ 策略权重引擎测试 v2 ═══\n")
we = WeightEngine()
base = we.compute_base_weights()
weights = we.get_weights()

# 1. 基础权重
check("基础权重总和=1", 0.95 <= sum(base.values()) <= 1.05, f"总和={sum(base.values()):.3f}")
check("底背离(t=7.4)权重最高", base.get('divergence',0) >= max(base.values()) - 0.001)
check("行业热度不在活跃列表", 'sector_heat' not in base)

# 2. 最终权重
non_zero = sum(v for v in weights.values() if v > 0)
check("非零权重总和≈1", 0.90 <= non_zero <= 1.10, f"总和={non_zero:.3f}")
check("单策略不超上限", all(v <= MAX_WEIGHT + 0.01 for v in weights.values()))
check("停用策略权重=0", all(weights.get(k,1) == 0 for k,v in STRATEGY_REGISTRY.items() if v['status']=='deprecated'))
check("新策略有权重", all(weights.get(k,0) > 0 for k,v in STRATEGY_REGISTRY.items() if v['status']=='new'))
check("底背离>聪明钱", weights.get('divergence',0) > weights.get('smart_money',0))

# 3. 调整系数
adj = we.compute_adjustment('divergence')
check("中性(50%)→1.0", 0.95 <= adj <= 1.05, f"adj={adj}")

# 4. 持久化
we.update()
check("weights.json存在", os.path.exists('./data/store/strategy_weights.json'))
try:
    data = json.loads(open('./data/store/strategy_weights.json').read())
    check("JSON格式正确", 'weights' in data and 'last_update' in data)
except:
    check("JSON格式正确", False, "解析失败")

# 5. 自检
errors = we.validate()
check("自检无错误", len(errors) == 0, f"错误: {errors}")

# 6. 排序合理性
sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
check("第一名是divergence或oversold", sorted_w[0][0] in ('divergence','oversold_reversal'), f"第一名={sorted_w[0][0]}")

print(f"\n{'═'*40}")
print(f"结果: {passed}通过 {failed}失败")
if failed == 0:
    print("✅ 全部通过")
else:
    print("❌ 需修复")

print()
we.report()
sys.exit(1 if failed else 0)
