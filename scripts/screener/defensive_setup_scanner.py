#!/usr/bin/env python3
"""防御抗跌选股scanner v2: 回踩到位+缩量+盈利+主线+无ST/高质押+二阶导(f'/f'', 因果右端)。
⚠️文案更正(2026-09-10): 原宣称"二阶导主升浪+7~8%/10日77~86%胜率、减速见顶-1~4%该减"是建在
shift1泄漏(IC虚高0.459)上的假象; 修正后 IC≈0.029≈噪声, 减速 p=0.90 不显著。二阶导仅作弱标签,
不作独立买卖/减仓依据(见 memory scoring_v3_derivative 与 R0 度量)。本scanner价值=回踩+缩量+质量过滤。
每天17:30跑,输出当日匹配+二阶导信号标签+追踪log。全绝对路径,无cwd依赖。"""
import os, glob, time
from datetime import datetime
import pandas as pd, numpy as np
import tushare as ts
from scipy.signal import savgol_filter, savgol_coeffs

KDIR = './data/store/kline'
LOG = './reports/defensive_setup_log.csv'
SG_W, SG_P = 21, 3  # S-G窗口+多项式阶数
from scripts.core.tushare_client import get_token
ts.set_token(get_token())
pro = ts.pro_api()
MAIN = ['食', '消费', '饮', '医', '药', '军', '航空', '航天', '船舶', '电', '有色', '煤', '钢',
        '石油', '石', '化工', '化纤', '机械', '汽车', '通信', '算', '半导体', '元件', '电气',
        '仪器', '塑料', '橡胶', '纺织', '家电', '轻工', '玻璃', '建材', '运输', '物流']


def qfq(s):
    c = s.astype(float).values.copy()
    r = pd.Series(c).pct_change().values
    for i in range(len(c) - 2, -1, -1):
        if not np.isnan(r[i + 1]) and r[i + 1] < -0.25:
            c[:i + 1] = c[:i + 1] * (c[i + 1] / c[i])
    return c


def compute_derivatives(close):
    """因果左导数(2026-08-04修正,无look-ahead). pos=W-1窗口末端无未来.
    真实IC=0.029弱因子(原shift1版0.459泄漏假象). W=21最优(扫描9-33)."""
    if len(close) < SG_W + 5:
        return np.nan, np.nan
    ln_p = np.log(close.astype(float))
    c1 = savgol_coeffs(SG_W, SG_P, deriv=1, pos=SG_W-1)
    c2 = savgol_coeffs(SG_W, SG_P, deriv=2, pos=SG_W-1)
    f1 = np.convolve(ln_p, c1[::-1], 'valid') * 100
    f2 = np.convolve(ln_p, c2[::-1], 'valid') * 10000
    return float(f1[-1]), float(f2[-1])
    # ★备份-原shift1版(可回滚,但有look-ahead泄漏,IC虚高)★
    # f1 = np.roll(savgol_filter(ln_p, SG_W, SG_P, deriv=1) * 100, 1); f1[0] = np.nan
    # f2 = np.roll(savgol_filter(ln_p, SG_W, SG_P, deriv=2) * 10000, 1); f2[0] = np.nan


def deriv_tag(f1, f2):
    """二阶导信号标签(回测验证)。"""
    if np.isnan(f1) or np.isnan(f2):
        return '·', 0
    if f1 > 0 and f2 > 0:
        return '🚀主升浪', 3   # ⚠️旧回测数字已作废(2026-09-10泄漏更正),仅弱标签
    if f1 < 0 and f2 > 0:
        return '🔄拐点', 2     # +0.5~3% 50~65% → 关注等f'转正
    if f1 > 0 and f2 < 0:
        return '⚠️减速', 1     # 因果回测无效(p=0.90),不作减仓依据,仅参考
    return '🔴加速跌', 0       # -5~8% 11~17% → 剔除


# stage1: 全市场扫 回踩到位+缩量 + 二阶导
cands = []
files = glob.glob(KDIR + '/*/D.parquet')
for p in files:
    code = os.path.basename(os.path.dirname(p))
    try:
        df = pd.read_parquet(p).sort_values('date').reset_index(drop=True)
    except Exception:
        continue
    if len(df) < 30:
        continue
    c = qfq(df['close'])
    v = (df['volume'].astype(float) if 'volume' in df else df['vol'].astype(float)).values
    price = float(c[-1]); ma20 = float(pd.Series(c).tail(20).mean())
    vs20 = (price / ma20 - 1) * 100
    vr = float(v[-1] / pd.Series(v).tail(20).mean())
    if -3 <= vs20 <= 5 and vr < 0.9:
        f1, f2 = compute_derivatives(c)
        cands.append((code, price, vs20, vr, f1, f2))
cands.sort(key=lambda x: (abs(x[2]), -x[3]))

# stage2: 基本面+主线+排ST
bb = pro.stock_basic(fields='ts_code,name,industry')
bb['code'] = bb['ts_code'].str.split('.').str[0]
nm = dict(zip(bb['code'], bb['name'])); ind = dict(zip(bb['code'], bb['industry']))
today = datetime.now().strftime('%Y-%m-%d')
final = []
for code, price, vs20, vr, f1, f2 in cands[:50]:
    name = nm.get(code); industry = ind.get(code, '')
    if not name or 'ST' in name or '*ST' in name:
        continue
    if not any(k in str(industry) for k in MAIN):
        continue
    tc = [x for x in bb['ts_code'] if x.split('.')[0] == code]
    if not tc:
        continue
    try:
        db = pro.daily_basic(ts_code=tc[0], fields='pe_ttm')
        pe = float(db.iloc[0]['pe_ttm']) if db is not None and len(db) else float('nan')
    except Exception:
        pe = float('nan')
    try:
        pg = pro.pledge_stat(ts_code=tc[0])
        pledge = float(pg.iloc[0]['pledge_ratio']) if len(pg) and pg.iloc[0]['pledge_ratio'] is not None else 0
    except Exception:
        pledge = 0
    time.sleep(0.06)
    if np.isnan(pe) or pe <= 0 or pledge > 50:
        continue
    tag, priority = deriv_tag(f1, f2)
    final.append((code, name, str(industry)[:7], price, vs20, vr, round(pe), pledge,
                  f1, f2, tag, priority))

# 按二阶导优先级排序(🚀>🔄>⚠️>🔴), 同级按贴近MA20
final.sort(key=lambda x: (-x[11], abs(x[4])))

print(f"🛡️ 防御抗跌scanner v2 | 回踩+缩量+盈利+主线+二阶导(f'/f'') | {today}")
print(f"   回踩+缩量候选 {len(cands)} → 模板命中 {len(final)} 只")
n_main = sum(1 for f in final if f[11] >= 2)
print(f"   二阶导信号: 🚀主升浪+🔄拐点 = {n_main} 只(优先) | ⚠️减速+🔴加速 = {len(final)-n_main} 只(警惕/剔除)\n")
print(f"{'代码':<7}{'名称':<10}{'行业':<8}{'现价':>8}{'vsMA20':>8}{'量比':>6}{'PE':>5}{'f′':>7}{'f″':>7}  信号")
print('-' * 80)
for c in final:
    f1s = f"{c[8]:+.2f}" if not np.isnan(c[8]) else '  NaN'
    f2s = f"{c[9]:+.1f}" if not np.isnan(c[9]) else '  NaN'
    print(f"{c[0]:<7}{c[1]:<10}{c[2]:<8}{c[3]:>8.2f}{c[4]:>+7.1f}%{c[5]:>6.2f}{c[6]:>5}{f1s:>7}{f2s:>7}  {c[10]}")

# 追踪log(加f'/f'')
log_exists = os.path.exists(LOG)
with open(LOG, 'a') as f:
    if not log_exists:
        f.write('date,code,name,industry,price,vsMA20,vol_ratio,pe,pledge,f1,f2,signal\n')
    for c in final:
        f1v = f"{c[8]:.3f}" if not np.isnan(c[8]) else ''
        f2v = f"{c[9]:.2f}" if not np.isnan(c[9]) else ''
        f.write(f"{today},{c[0]},{c[1]},{c[2]},{c[3]},{c[4]:.1f},{c[5]:.2f},{c[6]},{c[7]},{f1v},{f2v},{c[10]}\n")
print(f"\n📝 已追加 {len(final)} 只到 {LOG}")
print(f"💡 [2026-09-10更正] 二阶导为弱因子(修正IC≈0.029≈噪声), 旧'主升+7~8%/减速该减'回测数字作废")
print(f"💡 本scanner=防御质量清单(回踩+缩量+质量过滤), 二阶导标签仅供弱参考, 不作独立买卖依据")
