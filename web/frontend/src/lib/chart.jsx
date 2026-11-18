import { useEffect, useRef } from "react";
import * as echarts from "echarts";

/* 与后端/设计系统一致的验证色板（dataviz 六项校验，surface #111722） */
export const C = {
  ink1: "#e8edf5", ink2: "#9aa7bd", ink3: "#64748b", grid: "#1f2836",
  up: "#ff6a52", down: "#1fc79f",
  blue: "#3987e5", orange: "#d95926", aqua: "#199e70", yellow: "#c98500",
  magenta: "#d55181", warn: "#fab219", neutral: "#4a5568",
};
export const MONO = '"IBM Plex Mono", Menlo, monospace';

export const axisX = (extra = {}) => ({
  axisLine: { show: false }, axisTick: { show: false },
  axisLabel: { color: C.ink3, fontSize: 10, fontFamily: MONO },
  splitLine: { show: false }, ...extra,
});
export const axisY = (extra = {}) => ({
  axisLine: { show: false }, axisTick: { show: false },
  axisLabel: { color: C.ink3, fontSize: 10, fontFamily: MONO },
  splitLine: { lineStyle: { color: C.grid, width: 1 } },
  z: 0, ...extra,
});
export const tooltip = (extra = {}) => ({
  trigger: "axis",
  backgroundColor: "#161d2b", borderColor: "#2a3547", borderWidth: 1,
  textStyle: { color: C.ink1, fontSize: 11.5, fontFamily: MONO },
  axisPointer: { type: "cross", lineStyle: { color: "#3a4658" }, crossStyle: { color: "#3a4658" } },
  ...extra,
});
export const legend = (extra = {}) => ({
  textStyle: { color: C.ink2, fontSize: 10.5, fontFamily: MONO },
  itemWidth: 14, itemHeight: 2, icon: "rect",
  top: 0, ...extra,
});

/** ECharts React 封装：resize 观察 + setOption(notMerge) */
export default function Chart({ option, height = 320, style }) {
  const ref = useRef(null);
  const inst = useRef(null);
  useEffect(() => {
    inst.current = echarts.init(ref.current, null, { renderer: "canvas" });
    const ro = new ResizeObserver(() => inst.current?.resize());
    ro.observe(ref.current);
    return () => { ro.disconnect(); inst.current?.dispose(); inst.current = null; };
  }, []);
  useEffect(() => { inst.current?.setOption(option); }, [option]);
  return <div ref={ref} className="chart" style={{ height, ...style }} />;
}
