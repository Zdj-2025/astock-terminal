#!/usr/bin/env python3
"""1800只综合分析报告: 全因子融合评分 + 持仓分析"""
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, argrelextrema

PROJECT = Path(__file__).parent.parent.parent
data_dir = PROJECT / 'data' / 'store' / 'kline'
pool = pd.read_csv(PROJECT / 'data' / 'stock_pool.csv')
pool_codes = set(pool['code'].astype(str).tolist())
sl = pd.read_parquet(PROJECT / 'data' / 'store' / 'stock_list.parquet')
name_map = dict(zip(sl['symbol'], sl['name']))
ind_map = dict(zip(sl['symbol'], sl['industry']))

SG_W=21; SG_P=3

def _sma(s, n, m):
    a = m/n; r = np.empty_like(s); r[0]=s[0]
    for i in range(1,len(s)): r[i]=a*s[i]+(1-a)*r[i-1]
    return r

def safe_savgol(ln_p):
    w = min(SG_W, len(ln_p)//4*2+1)
    if w<5: return np.zeros(len(ln_p)), np.zeros(len(ln_p))
    from scripts.core.derivatives import causal_savgol  # 2026-09-10: 权威因果(pos=0右端), 弃中心化+shift滞后
    return causal_savgol(ln_p, w, SG_P, deriv=1), causal_savgol(ln_p, w, SG_P, deriv=2)

results = []
for code_dir in sorted(data_dir.iterdir()):
    if not code_dir.is_dir(): continue
    code = code_dir.name
    if code not in pool_codes: continue
    f = code_dir / 'D.parquet'
    if not f.exists(): continue
    try:
        df = pd.read_parquet(f)
        if len(df) < 140: continue
        date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        opens = df['open'].values.astype(float)
        vol = df['vol'].values.astype(float) if 'vol' in df.columns else df['volume'].values.astype(float) if 'volume' in df.columns else None
        if str(df[date_col].values[0])>str(df[date_col].values[-1]):
            close=close[::-1]; high=high[::-1]; low=low[::-1]; opens=opens[::-1]
            if vol is not None: vol=vol[::-1]
        n = len(close)
        name = name_map.get(code, code)
        industry = ind_map.get(code, '')
        if 'ST' in name: continue
        price = close[-1]

        # f'/f''
        ln_p = np.log(close)
        f1, f2 = safe_savgol(ln_p)
        f1v, f2v = f1[-1], f2[-1]
        f2_rising = f2[-1] > f2[-3] if n>=3 else False

        # trend
        F1T=0.002
        if f1v>0.05 and f2v>0 and f2_rising: signal='up_accel'; trend=4.0
        elif f1v>0.05 and f2v>0: signal='up_accel'; trend=3.5
        elif f1v>F1T and f2v>0: signal='up_accel'; trend=3.0
        elif f1v>F1T and f2v<=0: signal='up_decel'; trend=2.0
        elif f1v>0 and f2v>0: signal='turn_up'; trend=2.5
        elif f1v>0: signal='up_decel'; trend=1.5
        elif f2v>0 and f2_rising: signal='turn_up'; trend=1.0
        elif f2v>0: signal='dn_decel'; trend=0.5
        else: signal='dn_accel'; trend=0.0

        # position
        lb180=min(180,n); lb360=min(360,n); lb1000=min(1000,n)
        dh180=(price/np.max(high[-lb180:])-1)*100
        dh360=(price/np.max(high[-lb360:])-1)*100
        dh1000=(price/np.max(high[-lb1000:])-1)*100
        pos=0
        if dh180<-30: pos+=1.5
        elif dh180<-20: pos+=1.0
        elif dh180<-10: pos+=0.5
        if dh360<-30: pos+=1.0
        elif dh360<-20: pos+=0.5
        if dh1000<-30: pos+=1.0
        elif dh1000<-20: pos+=0.5
        pos=min(pos,4.0)

        # volume
        vol_score=0
        if vol is not None and n>=20:
            avg_v=np.mean(vol[-20:])
            if avg_v>0:
                vr=vol[-1]/avg_v
                if vr>1.5: vol_score=2.0
                elif vr>1.2: vol_score=1.5
                elif vr>1.0: vol_score=1.0
                else: vol_score=0.5

        v3 = 0.40*trend + 0.40*pos + 0.20*vol_score

        # boll
        ma20=np.mean(close[-20:]); std20=np.std(close[-20:])
        ub=ma20+2*std20; lb_b=ma20-2*std20
        boll_pct=(price-lb_b)/(ub-lb_b)*100 if ub>lb_b else 50

        # MACD
        s=pd.Series(close)
        e12=s.ewm(span=12).mean().values; e26=s.ewm(span=26).mean().values
        dif=e12-e26; dea=pd.Series(dif).ewm(span=9).mean().values; hist=2*(dif-dea)
        macd_cross = dif[-1]>dea[-1]
        dif_pos = dif[-1]>0
        macd_state = ('gold' if macd_cross else 'dead') + ('+DIF>0' if dif_pos else '+DIF<0')

        # B signal
        f2_turn = any(f2[-k]>0 and f2[-k-1]<=0 for k in range(1, min(4, n-1)))
        is_yang = close[-1]>opens[-1]
        dif_rising = dif[-1]>dif[-2] and dif[-2]>dif[-3] if n>=3 else False
        ma5_now=np.mean(close[-5:]); ma5_prev=np.mean(close[-6:-1]) if n>=6 else ma5_now
        ma5_turning = ma5_now > ma5_prev
        b_signal = boll_pct<35 and f2_turn and is_yang and dif_rising and ma5_turning

        # divergence
        divergence=False
        if n>40:
            lows_idx=argrelextrema(low, np.less, order=10)[0]
            if len(lows_idx)>=2:
                l1,l2=lows_idx[-2],lows_idx[-1]
                if low[l2]<low[l1] and dif[l2]>dif[l1]: divergence=True

        # Alpha109
        a109=1.0
        if n>=30:
            hl=high-low
            a109_s=_sma(hl,10,2); a109_l=_sma(a109_s,10,2)
            a109=a109_s[-1]/a109_l[-1] if a109_l[-1]>0 else 1.0

        # Alpha40
        a40=100.0
        if n>=27 and vol is not None:
            up_v=sum(vol[j] for j in range(n-26,n) if close[j]>close[j-1])
            dn_v=sum(vol[j] for j in range(n-26,n) if close[j]<=close[j-1])
            a40=up_v/dn_v*100 if dn_v>0 else 200

        squeeze = a109 < 0.9
        expanding = a109 > 1.05

        # squeeze ratio (boll width)
        width_pct = (ub-lb_b)/ma20*100 if ma20>0 else 0
        widths=[]
        for j in range(max(0,n-60), n):
            m2=np.mean(close[max(0,j-19):j+1]); s2=np.std(close[max(0,j-19):j+1])
            widths.append(4*s2/m2*100 if m2>0 else 0)
        avg_width=np.mean(widths) if widths else width_pct
        sq_ratio = width_pct/avg_width if avg_width>0 else 1

        # catalysis
        text=name+industry
        HOT_KW=['semiconductor','chip','AI','robot','aerospace','military','new energy',
                 'auto','communication','software','component','IT','electrical']
        catalysis=0.5
        for kw in ['半导体','芯片','AI','人工智能','算力','大模型','机器人','航天','军工','国防',
                     '新能源','储能','光伏','自动驾驶','低空','量子','数据','信创','国产替代',
                     '软件','元器件','通信','IT设备','电气自动化','汽车','有色','锂电','充电桩']:
            if kw in text: catalysis=3.0; break
        if catalysis<3:
            for kw in ['医药','创新药','银行','券商','保险','食品','白酒','家电','建材','化工','环保','物流']:
                if kw in text: catalysis=1.5; break

        # MA
        ma5=np.mean(close[-5:])
        ma60=np.mean(close[-60:]) if n>=60 else ma20
        ma120=np.mean(close[-120:]) if n>=120 else ma60

        # RSI
        delta=np.diff(close[-15:])
        gains=delta[delta>0]; losses=-delta[delta<0]
        avg_g=np.mean(gains) if len(gains)>0 else 0
        avg_l=np.mean(losses) if len(losses)>0 else 1
        rsi=100-100/(1+avg_g/avg_l) if avg_l>0 else 100

        vol_ratio = vol[-1]/np.mean(vol[-20:]) if vol is not None and n>=20 else 1.0

        # ===== FUSION SCORE =====
        bonus = 0
        if b_signal: bonus += 1.5
        if boll_pct < 20: bonus += 0.5
        if boll_pct < 35: bonus += 0.3
        if divergence: bonus += 0.5
        if catalysis >= 3: bonus += 0.5
        if boll_pct > 85: bonus -= 1.0
        if boll_pct > 70: bonus -= 0.5
        if expanding: bonus += 1.0
        if squeeze: bonus -= 0.3
        if not squeeze and a109>1.0: bonus += 0.5
        if a40 < 80: bonus += 0.5
        # squeeze breakout (60% win rate)
        squeeze_breakout = sq_ratio < 0.5 and f2v > 0 and macd_cross
        if squeeze_breakout: bonus += 1.5

        total_score = v3 + bonus

        # TAG
        if b_signal and boll_pct < 20:
            tag = 'SQUEEZE_BUY'
        elif squeeze_breakout:
            tag = 'SQ_BREAKOUT'
        elif boll_pct < 30 and f2_turn and ma5_turning:
            tag = 'BOTTOM_WATCH'
        elif boll_pct > 85 and f2v < 0:
            tag = 'TOP_SELL'
        elif boll_pct > 85:
            tag = 'HIGH_WARN'
        elif boll_pct > 70 and f2v < 0:
            tag = 'REDUCE'
        elif signal == 'up_accel' and boll_pct > 40:
            tag = 'TREND_UP'
        elif signal == 'up_accel':
            tag = 'BOTTOM_START'
        elif signal == 'up_decel':
            tag = 'UP_DECEL'
        else:
            tag = 'WATCH'

        signal_cn = {'up_accel':'up_accel','up_decel':'up_decel','turn_up':'turn_up',
                     'dn_decel':'dn_decel','dn_accel':'dn_accel'}.get(signal, signal)

        results.append({
            'code':code, 'name':name, 'industry':industry, 'price':round(price,2),
            'total':round(total_score,2), 'v3':round(v3,2), 'bonus':round(bonus,1),
            'trend':round(trend,1), 'pos':round(pos,1),
            'signal':signal, 'tag':tag,
            'boll_pct':round(boll_pct,0), 'macd':macd_state,
            'rsi':round(rsi,1), 'f1':round(f1v,5), 'f2':round(f2v,5),
            'a109':round(a109,3), 'a40':round(a40,0),
            'sq_ratio':round(sq_ratio,3), 'squeeze':squeeze,
            'dh180':round(dh180,1), 'dh360':round(dh360,1),
            'catalysis':catalysis,
            'ma5_up': price>ma5, 'ma60_up': price>ma60,
            'vol_ratio':round(vol_ratio,2),
            'divergence':divergence, 'b_signal':b_signal,
            'squeeze_breakout': squeeze_breakout,
        })
    except: continue

df = pd.DataFrame(results).sort_values('total', ascending=False)
df.to_csv(PROJECT / 'data' / 'full_analysis_1800.csv', index=False)

# ========== OUTPUT ==========
signal_labels = {'up_accel':'accel','up_decel':'decel','turn_up':'turn','dn_decel':'dn_slow','dn_accel':'dn_fast'}
tag_labels = {'SQUEEZE_BUY':'SQ_BUY', 'SQ_BREAKOUT':'SQ_BRK', 'BOTTOM_WATCH':'BTM_W',
              'BOTTOM_START':'BTM_S', 'TREND_UP':'TREND', 'UP_DECEL':'DECEL',
              'TOP_SELL':'SELL', 'HIGH_WARN':'HIGH', 'REDUCE':'REDUCE', 'WATCH':'WAIT'}

print(f'DONE: {len(df)} stocks, saved to data/full_analysis_1800.csv')
print()

# Tag stats
print('TAG DISTRIBUTION:')
for tag in ['SQ_BUY','SQ_BRK','BTM_S','BTM_W','TREND','DECEL','REDUCE','HIGH','SELL','WAIT']:
    t = tag
    sub = df[df['tag']==tag.replace('SQ_BUY','SQUEEZE_BUY').replace('SQ_BRK','SQ_BREAKOUT').replace('BTM_S','BOTTOM_START').replace('BTM_W','BOTTOM_WATCH').replace('TREND','TREND_UP').replace('DECEL','UP_DECEL').replace('SELL','TOP_SELL').replace('HIGH','HIGH_WARN').replace('REDUCE','REDUCE').replace('WAIT','WATCH')]
    if len(sub)==0: continue
    print(f'  {tag}: {len(sub)} avg={sub["total"].mean():.1f}')

print()

# Holdings
holdings = {'603881':'data_port','512480':'semi_ETF','512760':'chip_ETF',
            '002250':'lianhua','603617':'junhe','301060':'lanwei'}
print('HOLDINGS:')
for code, label in holdings.items():
    r = df[df['code']==code]
    if len(r)==0: print(f'  {label}({code}): no data'); continue
    r = r.iloc[0]
    print(f'  {label}({code}) | {r["price"]} | {r["tag"]} | total={r["total"]} v3={r["v3"]} | a109={r["a109"]} a40={r["a40"]} | boll={r["boll_pct"]}% | {r["macd"]}')
