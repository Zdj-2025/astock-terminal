import sys; sys.path.insert(0,'.')
import os; os.chdir('.')
import pandas as pd, numpy as np, requests, json
from scipy import stats
from scipy.optimize import minimize

NAMES = {'512480':'半导体','515120':'创新药','512100':'中证1000','159841':'证券',
         '562500':'机器人','159611':'电力','518880':'黄金','510300':'沪深300','512690':'酒',
         '512890':'红利低波','159928':'消费','512800':'银行','510880':'红利'}
LOCAL = ['512480','515120','512100','159841']
closes = {}
for code in LOCAL:
    df = pd.read_parquet(f'data/store/kline/{code}/D.parquet')
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    s = df.sort_values(dcol).set_index(pd.to_datetime(df[dcol],format='%Y%m%d',errors='coerce'))['close'].astype(float)
    closes[code] = s[~s.index.isna()]
for code in NAMES:
    if code in LOCAL: continue
    sym = ('sh' if code[0]=='5' else 'sz')+code
    d = requests.get(f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,320,qfq',
                     headers={'User-Agent':'Mozilla/5.0'}, timeout=15).json()['data'][sym]
    rows = d.get('qfqday') or d['day']
    closes[code] = pd.Series([float(r[2]) for r in rows], index=pd.to_datetime([r[0] for r in rows]))
px = pd.DataFrame(closes).sort_index()
ret = px.pct_change().dropna().tail(250)
cols = list(ret.columns); n = len(cols)
from sklearn.covariance import LedoitWolf
S = LedoitWolf().fit(ret.values).covariance_
shrink = LedoitWolf().fit(ret.values).shrinkage_

def stats_of(w, S):
    sig = np.sqrt(w@S@w)
    return sig*np.sqrt(252), (w@np.sqrt(np.diag(S)))/sig, 1/np.sum(w**2)
def rc(w,S):
    r = w*(S@w); return r/r.sum()

def erc_fixedpoint(S, iters=20000, tol=1e-12):
    n = S.shape[0]; w = np.ones(n)/n
    for _ in range(iters):
        mrc = S@w
        w_new = np.sqrt(w/mrc); w_new/= w_new.sum()
        if np.max(np.abs(w_new-w))<tol: w=w_new; break
        w = w_new
    return w

# ── 全域四组合 ──
bounds=[(0.0001,0.6)]*n; cons=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}]
w_mvp = minimize(lambda w:w@S@w, np.ones(n)/n, jac=lambda w:2*S@w, bounds=bounds, constraints=cons,
                 method='SLSQP', options={'maxiter':500,'ftol':1e-14}).x
w_erc = erc_fixedpoint(S)
sig_vec = np.sqrt(np.diag(S))
w_md = minimize(lambda w:-(w@sig_vec)/np.sqrt(w@S@w), np.ones(n)/n, bounds=bounds, constraints=cons,
                method='SLSQP', options={'maxiter':500,'ftol':1e-14}).x
w_ew = np.ones(n)/n

print('═══ 13ETF全域 · 年化σ/DR/有效N · 权重>1% ═══')
for nm,w in [('等权',w_ew),('最小方差',w_mvp),('风险平价ERC',w_erc),('最大分散',w_md)]:
    s,dr,en = stats_of(w,S)
    top = sorted([(cols[i],wi) for i,wi in enumerate(w) if wi>0.01],key=lambda x:-x[1])
    print(f'{nm:8s} σ={s*100:4.1f}% DR={dr:.2f} Ne={en:4.1f} | '+' '.join(f'{NAMES[c]}{wi*100:.0f}' for c,wi in top))
r_erc = rc(w_erc,S)
print(f'ERC收敛验证: 风险贡献 min={r_erc.min()*100:.1f}% max={r_erc.max()*100:.1f}% (应≈{100/n:.1f}%×资产数=每只{100/n:.1f}%)')

# ── 四资产轴 ──
AX = ['512480','515120','512890','159611']
idx=[cols.index(c) for c in AX]; Sa=S[np.ix_(idx,idx)]
wa_mvp = minimize(lambda w:w@Sa@w, np.ones(4)/4, jac=lambda w:2*Sa@w, bounds=[(0.0001,1)]*4,
                  constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(4)}],
                  method='SLSQP', options={'maxiter':500,'ftol':1e-14}).x
wa_erc = erc_fixedpoint(Sa)
print('\n═══ 四资产轴(半导体/创新药/红利低波/电力) ═══')
for nm,w in [('MVP',wa_mvp),('ERC',wa_erc)]:
    s,dr,en = stats_of(w,Sa); r = rc(w,Sa)
    print(f'{nm}: σ={s*100:.1f}% DR={dr:.2f} | '+' | '.join(f'{NAMES[AX[i]]}{w[i]*100:.0f}%(RC{r[i]*100:.0f}%)' for i in range(4)))

# ── 当前ETF袖珍 ──
pf = pd.read_excel('config/portfolio.xlsx')
cur_w = np.zeros(n); mkt = ret['510300']; betas={}
for _,r0 in pf.iterrows():
    code=str(r0['code']).zfill(6)
    try: sh=int(r0.get('shares',0) or 0)
    except: continue
    if code in NAMES and sh>0:
        mv = sh*closes[code].iloc[-1]
        cur_w[cols.index(code)] += mv
tot = cur_w.sum(); cur_w/=tot
for i,c in enumerate(cols):
    if cur_w[i]>0: betas[c]=np.polyfit(mkt,ret[c],1)[0]
beta_cur = sum(cur_w[i]*betas[cols[i]] for i in range(n) if cur_w[i]>0)
s_cur,dr_cur,en_cur = stats_of(cur_w,S)
hold = [(NAMES[c],cur_w[cols.index(c)],betas[c]) for c in betas]
print(f'\n═══ 当前ETF袖珍({tot:,.0f}元): σ={s_cur*100:.1f}% β={beta_cur:.2f} DR={dr_cur:.2f} Ne={en_cur:.1f} ═══')
for nm,w0,b in hold: print(f'  {nm} {w0*100:.0f}% β={b:.2f}')

# 稀释路径
i_d = cols.index('512890'); sig_d=np.sqrt(S[i_d,i_d]); sig_c=np.sqrt(cur_w@S@cur_w)
sleeve = ret[list(NAMES)].values@cur_w[[cols.index(c) for c in NAMES]]
rho_cd = np.corrcoef(sleeve, ret['512890'])[0,1]
beta_d = betas.setdefault('512890', np.polyfit(mkt,ret['512890'],1)[0])
print(f'\n红利低波: β={beta_d:.2f} | 袖珍相关={rho_cd:+.2f}')
dilu=[]
for w in [0,0.1,0.2,0.3,0.4]:
    nb = beta_cur*(1-w)+beta_d*w
    v = (1-w)**2*sig_c**2+w**2*sig_d**2+2*w*(1-w)*rho_cd*sig_c*sig_d
    dilu.append((w,nb,np.sqrt(v)*np.sqrt(252)))
    print(f'  加入{w*100:.0f}% → β={nb:.2f} σ={np.sqrt(v)*np.sqrt(252)*100:.1f}%')

# ── 固定权重回测(线性近似) ──
print('\n═══ 固定权重持有250日(线性组合近似,样本内) ═══')
R = ret[list(NAMES)].values
def full_w(sub_w, codes):
    wv=np.zeros(n)
    for i,c in enumerate(codes): wv[cols.index(c)]=sub_w[i] if len(sub_w)==len(codes) else None
    return wv
wv_amvp = np.zeros(n)
for i,c in enumerate(AX): wv_amvp[cols.index(c)]=wa_mvp[i]
wv_aerc = np.zeros(n)
for i,c in enumerate(AX): wv_aerc[cols.index(c)]=wa_erc[i]
for nm,wv in [('等权13',w_ew),('MVP13',w_mvp),('ERC13',w_erc),('四轴MVP',wv_amvp),('四轴ERC',wv_aerc),('当前袖珍',cur_w)]:
    pr = R@wv; cum=(1+pr).prod()-1; vol=pr.std()*np.sqrt(252)
    eq=np.maximum.accumulate(np.cumprod(1+pr)); dd=(np.cumprod(1+pr)/eq-1).min()
    print(f'{nm:8s} 累计{cum:+7.1%} σ{vol*100:4.1f}% 回撤{dd*100:5.1f}%')
bm=(1+mkt).prod()-1
print(f'沪深300同期 {bm:+.1%}')

json.dump({'mvp':w_mvp.tolist(),'erc':w_erc.tolist(),'md':w_md.tolist(),'ew':w_ew.tolist(),
           'amvp':wa_mvp.tolist(),'aerc':wa_erc.tolist(),'ax':AX,'cols':cols,
           'cur':cur_w.tolist(),'beta_cur':beta_cur,'dilu':dilu,'shrink':shrink,
           'names':NAMES}, open('/tmp/alloc_result.json','w'))
print('\n已存 /tmp/alloc_result.json')
