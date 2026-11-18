#!/usr/bin/env python3
"""因果Savitzky-Golay导数 — 唯一权威实现(全项目共享)。

2026-08-17 P0-1C建,同日实测修正scipy语义陷阱:
- scipy 1.17.1的 savgol_coeffs(pos=0) 才是窗口【右端=最新点】(直觉相反!),
  pos=W-1是左端(21天前)。原premarket实现(pos=W-1 + c[::-1]卷积)双重错误,
  输出错位值——x²验证(f''≡2常数)掩盖了错位,x³的"474≈480边界效应"实为错误。
- pos=0系数的时间轴向左 → 奇数阶导数需反号(deriv=1取负),偶数阶不用。
- 全序列滑动用 np.convolve(x, c, 'valid')(convolve内含kernel翻转,正好抵消
  系数-窗口直积c@win,勿再手动[::-1])。

铁律: 禁止 savgol_filter(x) 中心化窗口(未来泄漏,v3_derivative_scorer曾用,假IC0.459)。
验证基准: x²→f'(100)=200,f''=2.0; x³→f''(80)=480(全部精确,非近似)。

用法:
    from scripts.core.derivatives import causal_savgol, compute_derivatives
    f1_series = causal_savgol(np.log(close), 21, 3, deriv=1) * 100   # 全序列
    f1_last, f2_last = compute_derivatives(close)                     # 末值便捷版
"""
import numpy as np
from scipy.signal import savgol_coeffs

SG_W, SG_P = 21, 3   # W=21最优(9-33扫描,短<15长>25失效)——注:该结论基于旧错误实现,Phase2重测


def causal_savgol(x, window: int = SG_W, poly: int = SG_P, deriv: int = 1) -> np.ndarray:
    """因果SG导数·全序列。x为已变换序列(如np.log(close)); 前window-1位为NaN。
    每个位置只用该位置及之前的数据; 值为该位置(窗口右端)处的导数。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window:
        return np.full(n, np.nan)
    coeffs = savgol_coeffs(window, poly, deriv=deriv, pos=0)   # pos=0=右端(最新)
    if deriv % 2 == 1:
        coeffs = -coeffs                                       # 时间轴向左,奇数阶反号
    y = np.correlate(x, coeffs, 'valid')         # 滑动直积Σx[i+j]c[j];勿用convolve(会翻转系数)
    out = np.full(n, np.nan)
    out[window - 1:] = y
    return out


def compute_derivatives(close, window: int = SG_W, poly: int = SG_P):
    """末值因果f1/f2(对ln(close))。f1=一阶导%×100, f2=二阶导×10000。
    len<window+5返回(nan,nan)。与causal_savgol全序列末位严格一致。"""
    close = np.asarray(close, dtype=float)
    if len(close) < window + 5:
        return np.nan, np.nan
    ln_p = np.log(close)
    f1 = causal_savgol(ln_p, window, poly, 1)[-1] * 100
    f2 = causal_savgol(ln_p, window, poly, 2)[-1] * 10000
    return float(f1), float(f2)


def _self_test():
    """已知函数解析验证(全部精确)+因果性断言。"""
    ok = []
    x = np.arange(1.0, 101.0) ** 2      # x[i]=(i+1)², f'(v)=2v, f''=2
    d1 = causal_savgol(x, deriv=1)
    d2 = causal_savgol(x, deriv=2)
    ok.append(('x² f\'(100)=200(精确)', abs(d1[-1] - 200.0) < 1e-6))
    ok.append(('x² f\'\'=2.0(精确)', abs(d2[-1] - 2.0) < 1e-6))
    x3 = np.arange(1.0, 101.0) ** 3    # f''(v)=6v
    d3_2 = causal_savgol(x3, deriv=2)
    ok.append(('x³ f\'\'(80)=480(精确)', abs(d3_2[79] - 480.0) < 1e-6))
    ok.append(('x³ f\'\'(60)=360(内部点)', abs(d3_2[59] - 360.0) < 1e-6))
    # 符号: 上升序列一阶导>0(未反号会<0)
    ok.append(('上升序列 f1>0(符号正确)', d1[-1] > 0))
    # 因果性: 截断尾部不改变历史值(savgol_filter中心化版会变)
    trunc = causal_savgol(x[:80], deriv=2)
    ok.append(('因果: 截断后历史不变', abs(d2[79] - trunc[79]) < 1e-9))
    # 前导NaN
    ok.append(('前window-1位NaN', bool(np.isnan(d1[:SG_W - 1]).all()) and not np.isnan(d1[SG_W - 1])))
    # 末值便捷版==全序列末位
    price = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.02, 200)))
    f1a, f2a = compute_derivatives(price)
    f1s = causal_savgol(np.log(price), deriv=1) * 100
    f2s = causal_savgol(np.log(price), deriv=2) * 10000
    ok.append(('末值==全序列末位', abs(f1a - f1s[-1]) < 1e-12 and abs(f2a - f2s[-1]) < 1e-12))
    # 对照savgol_filter中心化(未来版): 信息项(相关度随噪声/窗口变化,不作判定)
    from scipy.signal import savgol_filter
    sf = savgol_filter(np.log(price), SG_W, SG_P, deriv=1)
    corr = np.corrcoef(f1s[SG_W:-SG_W], sf[SG_W:-SG_W])[0, 1]
    print(f'  ℹ️ 与中心化(泄漏)版相关系数={corr:.3f}(信息项,两者本差半个窗口时点)')
    all_pass = all(v for _, v in ok)
    for name, v in ok:
        print(f"  {'✅' if v else '❌'} {name}")
    return all_pass


if __name__ == '__main__':
    print("=== derivatives.py 自检(真·因果,scipy pos语义已修正) ===")
    import sys
    sys.exit(0 if _self_test() else 1)


# ===== VWAP-MACD 确认层 (2026-10-23上线, 布鲁斯VW-MACD的正确实现) =====
def vwap_macd_hist(df, fast=12, slow=26, sig=9, vwap_win=12):
    """返回VWAP-MACD柱值. df需high/low/close/vol列. 收敛需量支撑."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap = (tp * df['vol']).rolling(vwap_win).sum() / df['vol'].rolling(vwap_win).sum()
    vwap = vwap.fillna(df['close'])
    ef = vwap.ewm(span=fast, adjust=False).mean()
    es = vwap.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    return (dif - dif.ewm(span=sig, adjust=False).mean()) * 2
