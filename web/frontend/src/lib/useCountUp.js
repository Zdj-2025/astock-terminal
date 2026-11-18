import { useEffect, useRef, useState } from "react";

/** 数字滚动动画：800ms ease-out，减少动态偏好时直接跳终值。 */
export default function useCountUp(target, { decimals = 1, ms = 800 } = {}) {
  const [v, setV] = useState(target);
  const from = useRef(null);
  useEffect(() => {
    if (target == null || Number.isNaN(target)) return setV(target);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return setV(target);
    const start = performance.now();
    const a = from.current ?? target;
    let raf;
    const tick = (t) => {
      const p = Math.min(1, (t - start) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      setV(a + (target - a) * e);
      if (p < 1) raf = requestAnimationFrame(tick);
      else from.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v == null || Number.isNaN(v) ? "—" : v.toLocaleString("zh-CN", {
    minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
