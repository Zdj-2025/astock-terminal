#!/usr/bin/env python3
"""
商业航天板块每日跟踪脚本
- 板块整体行情
- 个股技术面分析（MA20/MA30/支撑位）
- 买入机会判断
"""
import urllib.request
import json
import sys
from datetime import datetime

# ===================== 配置 =====================
# 板块成分股（关键个股）
TRACK_STOCKS = {
    # 第一梯队：龙头
    'sh600118': '中国卫星',
    'sh600879': '航天电子',
    'sh601698': '中国卫通',
    # 第二梯队：材料与零部件（重点关注）
    'sz300699': '光威复材',
    'sh688122': '西部超导',
    'sh688102': '斯瑞新材',
    'sh688333': '铂力特',
    'sz300136': '信维通信',
    # SpaceX链
    'sz002792': '通宇通讯',
    # 其他关注
    'sz002149': '西部材料',
    'sz300034': '钢研高纳',
}

# 重点关注（回调买入候选）
FOCUS_STOCKS = ['sz300699', 'sh688122', 'sh688102', 'sh688333', 'sz300136']


def fetch_realtime(codes_dict):
    """通过腾讯接口获取实时行情"""
    codes_str = ','.join(codes_dict.keys())
    url = f'https://qt.gtimg.cn/q={codes_str}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    resp = opener.open(req, timeout=15)
    data = resp.read().decode('gbk')
    
    results = {}
    for line in data.strip().split(';'):
        line = line.strip()
        if not line or '=' not in line:
            continue
        parts = line.split('=\"')
        if len(parts) < 2:
            continue
        code_key = parts[0].replace('v_', '')
        val = parts[1].rstrip('"')
        fields = val.split('~')
        if len(fields) > 5:
            code = code_key
            results[code] = {
                'name': fields[1],
                'price': float(fields[3]) if fields[3] else 0,
                'yesterday_close': float(fields[4]) if fields[4] else 0,
                'open': float(fields[5]) if fields[5] else 0,
                'volume': fields[6] if len(fields) > 6 else '',
                'high': float(fields[33]) if len(fields) > 33 and fields[33] else 0,
                'low': float(fields[34]) if len(fields) > 34 and fields[34] else 0,
                'change_pct': float(fields[32]) if len(fields) > 32 and fields[32] else 0,
                'pe': float(fields[39]) if len(fields) > 39 and fields[39] else 0,
                'total_mv': float(fields[45]) if len(fields) > 45 and fields[45] else 0,  # 万亿
            }
    return results


def fetch_kline_tencent(code, days=60):
    """通过腾讯接口获取日K线（避免代理问题）"""
    # 腾讯日K接口
    # code格式: sh600118 或 sz300699
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
    full_url = f'{url}?_var=kline_dayqfq{code}&param={code},day,,,{days},qfq'
    
    req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    
    try:
        resp = opener.open(req, timeout=15)
        text = resp.read().decode('utf-8')
        # 解析 JSONP
        json_str = text.split('=', 1)[1].strip()
        data = json.loads(json_str)
        
        # 提取K线数据
        stock_data = data.get('data', {}).get(code, {})
        day_data = stock_data.get('qfqday', stock_data.get('day', []))
        
        result = []
        for item in day_data:
            # 腾讯格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
            result.append({
                'date': item[0],
                'open': float(item[1]),
                'close': float(item[2]),
                'high': float(item[3]),
                'low': float(item[4]),
                'volume': float(item[5]) if len(item) > 5 else 0,
            })
        return result
    except Exception as e:
        # 备用：用新浪接口
        try:
            return fetch_kline_sina(code, days)
        except:
            print(f"获取K线失败 {code}: {e}", file=sys.stderr)
            return []


def fetch_kline_sina(code, days=60):
    """通过新浪接口获取日K线（备用）"""
    sina_code = code  # sh600118 格式
    # 新浪K线接口
    d = 60  # 日K
    url = f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
    params = {
        'symbol': sina_code,
        'scale': '240',  # 日K
        'ma': 'no',
        'datalen': str(days),
    }
    query = '&'.join(f'{k}={v}' for k, v in params.items())
    full_url = f'{url}?{query}'
    
    req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    
    resp = opener.open(req, timeout=15)
    data = json.loads(resp.read().decode('utf-8'))
    
    result = []
    for item in data:
        result.append({
            'date': item.get('day', ''),
            'open': float(item.get('open', 0)),
            'close': float(item.get('close', 0)),
            'high': float(item.get('high', 0)),
            'low': float(item.get('low', 0)),
            'volume': float(item.get('volume', 0)),
        })
    return result


def calc_ma(klines, period):
    """计算移动平均线"""
    if len(klines) < period:
        return None
    recent = klines[-period:]
    return sum(k['close'] for k in recent) / period


def analyze_stock(code, name, realtime_data, klines):
    """分析个股技术面"""
    if not klines or len(klines) < 30:
        return None
    
    rt = realtime_data.get(code, {})
    current_price = rt.get('price', klines[-1]['close'])
    
    # 计算均线
    ma5 = calc_ma(klines, 5)
    ma10 = calc_ma(klines, 10)
    ma20 = calc_ma(klines, 20)
    ma30 = calc_ma(klines, 30)
    ma60 = calc_ma(klines, 60) if len(klines) >= 60 else None
    
    # 近期高低点
    recent_20 = klines[-20:]
    high_20 = max(k['high'] for k in recent_20)
    low_20 = min(k['low'] for k in recent_20)
    
    recent_60 = klines[-min(60, len(klines)):]
    high_60 = max(k['high'] for k in recent_60)
    low_60 = min(k['low'] for k in recent_60)
    
    # 从高点回撤幅度
    drawdown_from_high = (current_price - high_60) / high_60 * 100
    
    # 距MA20/MA30的距离
    dist_ma20 = (current_price - ma20) / ma20 * 100 if ma20 else None
    dist_ma30 = (current_price - ma30) / ma30 * 100 if ma30 else None
    
    # 判断信号
    signals = []
    
    # 1. 接近MA20支撑
    if dist_ma20 is not None and -3 < dist_ma20 < 1:
        signals.append(f"📍 接近MA20({ma20:.2f})，距MA20 {dist_ma20:+.1f}%")
    
    # 2. 接近MA30支撑
    if dist_ma30 is not None and -3 < dist_ma30 < 1:
        signals.append(f"📍 接近MA30({ma30:.2f})，距MA30 {dist_ma30:+.1f}%")
    
    # 3. 从高点回撤超15%
    if drawdown_from_high < -15:
        signals.append(f"📉 从60日高点({high_60:.2f})回撤 {drawdown_from_high:.1f}%")
    
    # 4. 缩量回调（近5日成交量 < 近20日平均的70%）
    if len(klines) >= 20:
        vol_5 = sum(k['volume'] for k in klines[-5:]) / 5
        vol_20 = sum(k['volume'] for k in klines[-20:]) / 20
        if vol_20 > 0 and vol_5 / vol_20 < 0.7:
            signals.append(f"🔇 缩量回调，近5日均量仅为20日均量的 {vol_5/vol_20*100:.0f}%")
    
    # 5. MA20上穿MA30金叉（近3日内）
    if len(klines) >= 32:
        # 3天前
        for i in range(-3, 0):
            sub = klines[:len(klines)+i]
            m20_prev = calc_ma(sub, 20)
            m30_prev = calc_ma(sub, 30)
            if m20_prev and m30_prev:
                # 前一天MA20<MA30，当天MA20>MA30
                sub_prev = klines[:len(klines)+i-1]
                m20_pp = calc_ma(sub_prev, 20)
                m30_pp = calc_ma(sub_prev, 30)
                if m20_pp and m30_pp and m20_pp < m30_pp and m20_prev > m30_prev:
                    signals.append(f"✨ MA20上穿MA30金叉！")
                    break
    
    pe = rt.get('pe', 0)
    
    return {
        'code': code,
        'name': name,
        'price': current_price,
        'change_pct': rt.get('change_pct', 0),
        'pe': pe,
        'ma5': ma5,
        'ma10': ma10,
        'ma20': ma20,
        'ma30': ma30,
        'ma60': ma60,
        'high_20': high_20,
        'low_20': low_20,
        'high_60': high_60,
        'low_60': low_60,
        'drawdown': drawdown_from_high,
        'dist_ma20': dist_ma20,
        'dist_ma30': dist_ma30,
        'signals': signals,
        'has_buy_signal': len(signals) > 0,
    }


def main():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"{'='*60}")
    print(f"商业航天板块每日跟踪 | {today}")
    print(f"{'='*60}")
    
    # 1. 获取实时行情
    print("\n📊 一、实时行情概览")
    print("-" * 50)
    rt_data = fetch_realtime(TRACK_STOCKS)
    
    all_stocks = []
    for code in TRACK_STOCKS:
        name = TRACK_STOCKS[code]
        klines = fetch_kline_tencent(code, 60)
        analysis = analyze_stock(code, name, rt_data, klines)
        if analysis:
            all_stocks.append(analysis)
            signal_mark = " ⚡" if analysis['has_buy_signal'] else ""
            print(f"{name}({code[2:]}): {analysis['price']:.2f} ({analysis['change_pct']:+.2f}%) PE={analysis['pe']:.0f}{signal_mark}")
    
    # 2. 板块整体评估
    print(f"\n📈 二、板块整体评估")
    print("-" * 50)
    
    if all_stocks:
        avg_change = sum(s['change_pct'] for s in all_stocks) / len(all_stocks)
        positive_count = sum(1 for s in all_stocks if s['change_pct'] > 0)
        avg_drawdown = sum(s['drawdown'] for s in all_stocks) / len(all_stocks)
        
        print(f"板块均涨跌幅: {avg_change:+.2f}%")
        print(f"上涨/下跌: {positive_count}/{len(all_stocks)-positive_count}")
        print(f"平均距60日高点回撤: {avg_drawdown:.1f}%")
        
        if avg_change > 1:
            print("📊 板块状态: 偏强，注意追高风险")
        elif avg_change > -1:
            print("📊 板块状态: 震荡，观望为主")
        else:
            print("📊 板块状态: 偏弱，关注支撑位")
    
    # 3. 重点关注个股详细分析
    print(f"\n🎯 三、重点关注个股技术面")
    print("-" * 50)
    
    focus_results = [s for s in all_stocks if s['code'] in FOCUS_STOCKS]
    
    any_buy_signal = False
    for s in focus_results:
        print(f"\n【{s['name']}({s['code'][2:]})】 现价: {s['price']:.2f}")
        print(f"  MA5={s['ma5']:.2f} MA10={s['ma10']:.2f} MA20={s['ma20']:.2f} MA30={s['ma30']:.2f}", end="")
        if s['ma60']:
            print(f" MA60={s['ma60']:.2f}")
        else:
            print()
        print(f"  20日区间: {s['low_20']:.2f} - {s['high_20']:.2f}")
        print(f"  60日高点: {s['high_60']:.2f}，距高点: {s['drawdown']:.1f}%")
        if s['dist_ma20'] is not None:
            print(f"  距MA20: {s['dist_ma20']:+.1f}%", end="")
        if s['dist_ma30'] is not None:
            print(f"  距MA30: {s['dist_ma30']:+.1f}%")
        
        if s['signals']:
            print(f"  信号:")
            for sig in s['signals']:
                print(f"    {sig}")
            any_buy_signal = True
        else:
            print(f"  信号: 无明显买入信号")
    
    # 4. 综合判断
    print(f"\n{'='*60}")
    print("📋 综合判断")
    print("=" * 60)
    
    buy_candidates = [s for s in focus_results if s['has_buy_signal']]
    if buy_candidates:
        print("🟢 有买入机会的个股:")
        for s in buy_candidates:
            print(f"  ★ {s['name']}({s['code'][2:]}) @ {s['price']:.2f}")
            for sig in s['signals']:
                print(f"    {sig}")
    else:
        print("⚪ 暂无明显买入信号，继续观望")
    
    # 5. 建议关注的价位
    print(f"\n💡 建议关注价位:")
    for s in focus_results:
        supports = []
        if s['ma20']:
            supports.append(f"MA20={s['ma20']:.2f}")
        if s['ma30']:
            supports.append(f"MA30={s['ma30']:.2f}")
        if s['ma60']:
            supports.append(f"MA60={s['ma60']:.2f}")
        print(f"  {s['name']}: 支撑位 [{', '.join(supports)}]")


if __name__ == '__main__':
    main()
