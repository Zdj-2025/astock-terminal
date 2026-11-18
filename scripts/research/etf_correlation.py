import sys; sys.path.insert(0,'.')
import os; os.chdir('.')
import pandas as pd, numpy as np, requests
from scipy import stats

NAMES = {
    '512480':'半导体','515120':'创新药','512100':'中证1000','159841':'证券',
    '562500':'机器人','159611':'电力','518880':'黄金','510300':'沪深300','512690':'酒',
    '512890':'红利低波','159928':'消费','512800':'银行','510880':'红利',
}
LOCAL = ['512480','515120','512100','159841']
FETCH = {'562500':'sh562500','159611':'sz159611','518880':'sh518880',
         '510300':'sh510300','512690':'sh512690','512890':'sh512890',
         '159928':'sz159928','512800':'sh512800','510880':'sh510880'}

closes = {}
for code in LOCAL:
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    s = df.sort_values(dcol).set_index(pd.to_datetime(df[dcol], format='%Y%m%d', errors='coerce'))['close'].astype(float)
    closes[code] = s[~s.index.isna()]
for code, sym in FETCH.items():
    d = requests.get(f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,320,qfq',
                     headers={'User-Agent':'Mozilla/5.0'}, timeout=15).json()['data'][sym]
    key = 'qfqday' if 'qfqday' in d else 'day'
    rows = d[key]
    idx = pd.to_datetime([r[0] for r in rows], format='%Y-%m-%d', errors='coerce')
    closes[code] = pd.Series([float(r[2]) for r in rows], index=idx)

px = pd.DataFrame(closes).sort_index().tail(320)
ret = px.pct_change().dropna(how='all')
print(f"样本: {ret.index[0].date()} → {ret.index[-1].date()} | {len(ret)}个交易日 | 13只ETF")

for w, lab in [(60,'近60日'), (250,'近250日')]:
    r = ret.tail(w).dropna()
    c = r.corr(method='pearson')
    print(f"\n═══ {lab} (n={len(r)}) ═══")
    cols = list(c.columns)
    print('        ' + ''.join(f'{NAMES[k]:>7s}' for k in cols))
    for i, ri in enumerate(cols):
        cells = ''
        for j in range(len(cols)):
            if j > i: cells += '       '
            elif j == i: cells += '   --- '
            else: cells += f'{c.iloc[i,j]:+7.2f}'
        print(f'{NAMES[ri]:>6s}  {cells}')

print('\n═══ 防御四兄弟×核心轴 显著性(250日) ═══')
r250 = ret.tail(250).dropna()
pairs = [('512890','512480'),('512890','515120'),('512890','510300'),('512890','159611'),
         ('510880','512480'),('512800','512480'),('512800','512890'),
         ('159928','512480'),('159928','512890'),('159928','515120'),
         ('512690','159928'),('512890','159611'),('512800','159611'),('510880','159611')]
for a,b in pairs:
    x, y = r250[a], r250[b]
    m = x.notna() & y.notna()
    n = m.sum(); pr, pv = stats.pearsonr(x[m], y[m])
    star = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else '  '
    print(f'{NAMES[a]:6s}×{NAMES[b]:6s} n={n}  r={pr:+.3f}{star}(p={pv:.3f})')

print('\n═══ 滚动60日: 红利低波×半导体 / 红利低波×创新药 (月采样) ═══')
roll = pd.DataFrame({
    '红利低波×半导体': ret['512890'].rolling(60).corr(ret['512480']),
    '红利低波×创新药': ret['512890'].rolling(60).corr(ret['515120']),
    '红利低波×沪深300': ret['512890'].rolling(60).corr(ret['510300']),
}).dropna()
print(roll.iloc[::20].round(2).to_string())

# ═══════════ 方法论注释(2026-09-30与用户对齐) ═══════════
# 数据口径: 必须用收益率(涨跌幅=简单收益率), 禁止价格水平(非平稳→伪相关:
#   实测半导体×创新药价格相关-0.35 vs 收益+0.31符号翻转; 独立随机游走2000次模拟
#   |r|>0.5概率38%)。日频下对数收益率与简单等价(差<0.5%)。
# 算法匹配: Pearson主算(组合协方差语言)+Spearman复核(抗离群)——已全对同号;
#   对冲有效性须加测尾部条件相关(崩盘日r): 红利低波全样本-0.39/半导体最差10%日-0.30(对冲成立)。

# ═══════════ 残差分析模块(2026-10-01并入,原/tmp/corr_decomp.py) ═══════════
# 用法: python etf_correlation.py --residual
# 剥离大盘=回归残差(ε=Ri-α-β·Rm),验收标准=残差×大盘相关为0(OLS正规方程保证)
# 直接相减Ri-Rm隐含β=1,残差仍带大盘相关,禁用
def residual_analysis(ret, market='510300', pairs=None):
    from scipy import stats as st
    import numpy as np
    mkt = ret[market]
    def strip(s):
        b, a = np.polyfit(mkt, s, 1)
        return s - (b*mkt + a), b, a, st.pearsonr(s, mkt)[0]
    if pairs is None:
        pairs = [('512480','515120'),('512480','512890'),('512480','512100'),
                 ('515120','512890'),('159841','512890'),('512480','159841')]
    print(f"\n═══ 残差分析(剥离{market},n={len(ret)}) ═══")
    cache = {}
    for a, b in pairs:
        for c in (a, b):
            if c not in cache:
                r, beta, alpha, r_m = strip(ret[c])
                # 三关验证②: 残差与大盘相关应=0
                orth = st.pearsonr(r, mkt)[0]
                cache[c] = (r, beta, r_m, orth)
        ra, ba, rma, oa = cache[a]; rb, bb, rmb, ob = cache[b]
        raw = st.pearsonr(ret[a], ret[b])[0]
        res, p = st.pearsonr(ra, rb)
        print(f'{NAMES.get(a,a):6s}×{NAMES.get(b,b):6s}: 原始{raw:+.3f} → 残差{res:+.3f}(p={p:.3f}) '
              f'| β {ba:.2f}/{bb:.2f} | 方差大盘占比{rma**2*100:.0f}%/{rmb**2*100:.0f}% '
              f'| 正交验收{oa:+.4f}/{ob:+.4f}')
    return cache

if __name__ == '__main__' and '--residual' in sys.argv:
    import numpy as np
    ret_r = ret.tail(250).dropna()   # 复用主脚本装载的数据
    residual_analysis(ret_r)
    sys.exit(0)
