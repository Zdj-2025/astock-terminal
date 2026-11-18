import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import Chart, { C, axisX, axisY, tooltip, legend } from "../lib/chart";
import { PageHead, Panel, Tabs, DT, Loading, Md } from "../components/ui";
import { Search, Zap, AlertTriangle } from "lucide-react";

const MONO = '"IBM Plex Mono", Menlo, monospace';
const MA_COLORS = { ma5: C.blue, ma10: C.orange, ma20: C.aqua, ma60: C.yellow };  // 固定槽位顺序

export default function Stock() {
  const urlCode = (() => {                      // 支持 #/stock?code=603881 深链
    const m = window.location.hash.match(/[?&]code=(\d{4,6})/);
    return m ? m[1].padStart(6, "0") : "603881";
  })();
  const [input, setInput] = useState(urlCode);
  const [code, setCode] = useState(urlCode);
  const [days, setDays] = useState(120);
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  const [tab, setTab] = useState("k");

  useEffect(() => {
    setD(null); setErr("");
    api(`/api/page/stock/${code}?days=${days}`).then(setD).catch((e) => setErr(String(e)));
  }, [code, days]);

  const submit = (e) => {
    e.preventDefault();
    const v = input.trim();
    if (/^\d{5,6}$/.test(v)) setCode(v.padStart(6, "0"));
  };

  return (
    <>
      <form onSubmit={submit} style={{ display: "flex", gap: 8, marginBottom: 16, maxWidth: 420 }}>
        <div style={{ position: "relative", flex: 1 }}>
          <Search size={14} style={{ position: "absolute", left: 11, top: 11, color: "var(--ink-3)" }} />
          <input className="input" style={{ paddingLeft: 32 }} value={input} onChange={(e) => setInput(e.target.value)}
                 placeholder="股票代码，如 603881 / 512100" />
        </div>
        <button className="btn primary" type="submit">分析</button>
        <select className="input" style={{ width: 92 }} value={days} onChange={(e) => setDays(+e.target.value)}>
          {[60, 120, 180, 250].map((n) => <option key={n} value={n}>{n}日</option>)}
        </select>
      </form>

      {err && (
        <Panel title="加载失败">
          {err}
          {err.includes("本地无") && (
            <div className="note" style={{ marginTop: 8 }}>
              本地还没下载过这只票的 K线。两种解决方式：<br />
              ① 先看演示：<code>make demo</code>（合成数据，含 17 只常见标的）<br />
              ② 下载真实日线：「设置」页配好 Tushare token 后运行
              <code> PYTHONPATH=. python scripts/data/update_all_kline.py</code>
            </div>
          )}
        </Panel>
      )}
      {!d && !err && <Loading />}

      {d && (
        <>
          <PageHead eyebrow="Single Stock Workbench" title={`${d.name}（${d.code}）`} chips={[
            ["日线 ", `${d.dates.length} 根`], ["窗口 ", `${days}日`], ["数据 ", "本地 parquet 只读"],
          ]} />

          <Tabs tabs={[["k", "K线技术"], ["chan", "缠论结构"], ["wave", "波浪"], ["score", "四维评分"], ["rt", "实时资金"]]}
                active={tab} onTab={setTab} />

          {tab === "k" && <KlinePanel d={d} />}
          {tab === "chan" && <ChanPanel code={d.code} />}
          {tab === "wave" && <WavePanel d={d} />}
          {tab === "score" && <ScorePanel code={d.code} />}
          {tab === "rt" && <RtPanel code={d.code} />}
        </>
      )}
    </>
  );
}

/* ── K线四联图 ── */
function KlinePanel({ d }) {
  const option = useMemo(() => {
    const mk = (xIdx, yIdx) => ({ xAxisIndex: xIdx, yAxisIndex: yIdx });
    return {
      grid: [
        { left: 48, right: 14, top: 28, height: "48%" },
        { left: 48, right: 14, top: "61%", height: "10%" },
        { left: 48, right: 14, top: "75%", height: "10%" },
        { left: 48, right: 14, top: "89%", height: "8%" },
      ],
      xAxis: [
        ...[0, 1, 2].map((i) => axisX({ type: "category", data: d.dates, gridIndex: i, boundaryGap: true, axisLabel: { show: false }, boundaryGap: true })),
        axisX({ type: "category", data: d.dates, gridIndex: 3 }),
      ],
      yAxis: [
        axisY({ gridIndex: 0, scale: true, splitNumber: 5 }),
        axisY({ gridIndex: 1, splitLine: { show: false }, axisLabel: { show: false } }),
        axisY({ gridIndex: 2, scale: true, name: "12,26,9", nameTextStyle: { color: C.ink3, fontSize: 9, align: "left" } }),
        axisY({ gridIndex: 3, scale: true, name: "5,34,5", nameTextStyle: { color: C.ink3, fontSize: 9, align: "left" } }),
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: tooltip({ trigger: "axis" }),
      legend: legend(),
      dataZoom: [
        { type: "inside", xAxisIndex: [0, 1, 2, 3], start: 0, end: 100 },
        { type: "slider", xAxisIndex: [0, 1, 2, 3], bottom: 2, height: 14, borderColor: "var(--border)",
          backgroundColor: "rgba(0,0,0,0.25)", fillerColor: "rgba(232,112,42,0.10)",
          handleStyle: { color: "#161d2b", borderColor: "#e8702a" }, textStyle: { color: C.ink3, fontSize: 9 } },
      ],
      series: [
        { name: "K线", type: "candlestick", data: d.kline, ...mk(0, 0),
          itemStyle: { color: C.up, color0: C.down, borderColor: C.up, borderColor0: C.down } },
        ...Object.entries(MA_COLORS).map(([k, cl]) => ({
          name: k.toUpperCase(), type: "line", data: d[k], ...mk(0, 0), showSymbol: false,
          lineStyle: { color: cl, width: 1.3 }, z: 4,
        })),
        { name: "成交量", type: "bar", data: d.volume?.map((v, i) => ({
            value: v, itemStyle: { color: d.kline[i]?.[1] >= d.kline[i]?.[0] ? "rgba(255,106,82,.5)" : "rgba(31,199,159,.5)" } })),
          ...mk(1, 1), barMaxWidth: 6 },
        ...macdSeries("MACD", d.dif, d.dea, d.hist, 2),
        ...macdSeries("MACD·F", d.dif_f, d.dea_f, d.hist_f, 3),
      ],
    };
  }, [d]);
  return (
    <Panel title="K线 · MA · 成交量 · MACD 双参数" extra="滚轮/拖动缩放"
           note="标准 12,26,9 看趋势 · 快速 5,34,5 灵敏做T/背离检测（口径对齐 comprehensive_analyzer）">
      <Chart height={640} option={option} />
    </Panel>
  );
}

function macdSeries(name, dif, dea, hist, grid) {
  return [
    { name: `${name}柱`, type: "bar", xAxisIndex: grid, yAxisIndex: grid, barMaxWidth: 5,
      data: hist.map((v) => ({ value: v, itemStyle: { color: v >= 0 ? "rgba(255,106,82,.55)" : "rgba(31,199,159,.55)" } })) },
    { name: "DIF", type: "line", xAxisIndex: grid, yAxisIndex: grid, data: dif, showSymbol: false,
      lineStyle: { color: C.yellow, width: 1.2 } },
    { name: "DEA", type: "line", xAxisIndex: grid, yAxisIndex: grid, data: dea, showSymbol: false,
      lineStyle: { color: C.blue, width: 1.2 } },
  ];
}

/* ── 缠论 ── */
function ChanPanel({ code }) {
  const [with60, setWith60] = useState(false);
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setD(null); setErr("");
    api(`/api/chanlun/${code}?with60=${with60}`).then((r) => setD(r.levels)).catch((e) => setErr(String(e)));
  }, [code, with60]);

  return (
    <Panel title="缠论结构" extra="ChanLunEngine"
           note="日线走 Tushare 缓存；60min 走 akshare 实时网络（显式开启）">
      <label style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--ink-2)", marginBottom: 10, cursor: "pointer" }}>
        <input type="checkbox" checked={with60} onChange={(e) => setWith60(e.target.checked)} />
        加 60min 级别（akshare 网络）
      </label>
      {err && <div className="empty" style={{ color: "var(--warn)" }}>
        <AlertTriangle size={14} style={{ verticalAlign: -2, marginRight: 6 }} />
        缠论分析失败（常见：tushare 网络不通/积分限流）。稍后重试；K线/波浪走本地不受影响。<br />
        <span style={{ fontSize: 11 }}>{err}</span></div>}
      {!d && !err && <Loading text="缠论计算中" />}
      {d?.map((lv) => <ChanLevel key={lv.level} lv={lv} code={code} />)}
    </Panel>
  );
}

function ChanLevel({ lv, code }) {
  const [k, setK] = useState(null);
  useEffect(() => { api(`/api/page/stock/${code}?days=180`).then(setK); }, [code]);
  if (!k) return <Loading />;
  const dates = k.dates;
  const keys = dates.map((s) => s.slice(2));                     // 'MM-DD'
  const pos = (t) => {                                          // 'YYYY-MM-DD' → 窗口下标（二分最近邻）
    const key = t?.slice(5) ?? "";
    let lo = 0, hi = keys.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; keys[mid] < key ? lo = mid + 1 : hi = mid; }
    return lo;
  };
  const zig = (arr) => {
    const pts = [];
    arr.forEach((s) => { const x1 = pos(s.e_t); if (x1 <= 0) return;
      pts.push([pos(s.b_t), +s.b_p.toFixed(3)], [x1, +s.e_p.toFixed(3)]); });
    return pts;
  };
  const toXY = (pts) => pts.map(([x, y]) => [x, y]);
  const stP = zig(lv.strokes), sgP = zig(lv.segments);
  const pivots = lv.pivots.filter((z) => pos(z.e_t) > 0);
  const option = {
    grid: { left: 48, right: 14, top: 30, bottom: 24 },
    xAxis: axisX({ type: "category", data: dates, boundaryGap: true }),
    yAxis: axisY({ scale: true }),
    tooltip: tooltip(),
    legend: legend(),
    series: [
      { name: "K线", type: "candlestick", data: k.kline, silent: true,
        itemStyle: { color: C.up, color0: C.down, borderColor: C.up, borderColor0: C.down } },
      { name: `笔(${lv.strokes.length})`, type: "line", data: toXY(stP),
        showSymbol: false, connectNulls: false, lineStyle: { color: C.yellow, width: 1.5 }, z: 4 },
      { name: `线段(${lv.segments.length})`, type: "line", data: toXY(sgP),
        showSymbol: false, connectNulls: false, lineStyle: { color: C.blue, width: 2.2 }, z: 5 },
      { name: "买点", type: "scatter", data: lv.bs_points.filter((b) => b.buy).map((b) => ({
          value: [pos(b.time), b.price],
          itemStyle: { color: C.up, borderColor: "#111722", borderWidth: 2 } })),
        symbol: "triangle", symbolSize: 11, z: 6,
        tooltip: { formatter: (p) => { const b = lv.bs_points.filter((x) => x.buy)[p.dataIndex];
          return `${b.time} B${b.types.join("/")}${b.sure ? "✓" : "?"} @${b.price}`; } } },
      { name: "卖点", type: "scatter", data: lv.bs_points.filter((b) => !b.buy).map((b) => ({
          value: [pos(b.time), b.price],
          itemStyle: { color: C.down, borderColor: "#111722", borderWidth: 2 } })),
        symbol: "triangle", symbolSize: 11, symbolRotate: 180, z: 6,
        tooltip: { formatter: (p) => { const b = lv.bs_points.filter((x) => !x.buy)[p.dataIndex];
          return `${b.time} S${b.types.join("/")}${b.sure ? "✓" : "?"} @${b.price}`; } } },
    ],
    // 中枢：markArea（比 graphic 像素矩形可靠，随 dataZoom 联动）
  };
  option.series[1].markArea = {
    silent: true, z: 2,
    itemStyle: { color: "rgba(57,135,229,0.07)", borderColor: C.blue, borderWidth: 1, borderDash: "dotted" },
    label: { show: false },
    data: pivots.map((z) => [
      { xAxis: pos(z.b_t), yAxis: +z.low.toFixed(3) },
      { xAxis: pos(z.e_t), yAxis: +z.high.toFixed(3) },
    ]),
  };
  return (
    <>
      <div className="note" style={{ marginTop: 0, marginBottom: 6 }}>
        {lv.level} · 笔 {lv.strokes.length} · 线段 {lv.segments.length} · 中枢 {lv.pivots.length} · 买卖点 {lv.bs_points.length}
      </div>
      <Chart height={480} option={option} />
      {lv.bs_points?.length > 0 && (
        <details className="fold" style={{ marginTop: 10 }}><summary>最近买卖点</summary>
          <div className="body"><DT cols={[
            { k: "buy", t: "类型", render: (b) => <span className={b.buy ? "pos" : "neg"}>{b.buy ? "买" : "卖"}</span> },
            { k: "types", t: "级别", render: (b) => b.types.join("/") },
            { k: "price", t: "价格", num: true, render: (b) => Number(b.price).toFixed(2) },
            { k: "time", t: "时间", num: true }, { k: "sure", t: "确认", render: (b) => b.sure ? "✓" : "?" },
          ]} rows={[...lv.bs_points].reverse().slice(0, 10)} maxH={240} /></div></details>
      )}
    </>
  );
}

/* ── 波浪 ── */
function WavePanel({ d }) {
  const w = d.wave;
  if (!w) return <Panel title="波浪分析"><div className="empty">{d.wave_error || "无结果"}</div></Panel>;
  const hi = Math.max(...d.kline.map((k) => k[3]));
  const lo = Math.min(...d.kline.map((k) => k[2]));
  const option = {
    grid: { left: 48, right: 60, top: 34, bottom: 24 },
    xAxis: axisX({ type: "category", data: d.dates, boundaryGap: true }),
    yAxis: axisY({ scale: true }),
    tooltip: tooltip(),
    legend: legend(),
    series: [
      { name: "K线", type: "candlestick", data: d.kline,
        itemStyle: { color: C.up, color0: C.down, borderColor: C.up, borderColor0: C.down } },
      { name: "浪型", type: "line", data: w.waves.map((x) => [x.end_idx, +x.end_price.toFixed(3)]),
        showSymbol: true, symbolSize: 6, symbol: "circle",
        itemStyle: { color: C.yellow, borderColor: "#111722", borderWidth: 2 },
        lineStyle: { color: C.yellow, width: 2, type: "dotted" }, z: 5 },
    ],
  };
  // 浪型标签（markPoint 随坐标系定位）
  option.series[1].markPoint = {
    symbol: "circle", symbolSize: 1, label: { show: true, position: "top", color: C.ink1, fontSize: 11, fontWeight: 600, fontFamily: MONO },
    data: w.waves.map((x) => ({ coord: [x.end_idx, +x.end_price.toFixed(3)], value: x.label })),
  };
  option.series[0].markLine = {
    symbol: "none", silent: true, label: { color: C.ink3, fontSize: 10, fontFamily: MONO, position: "insideEndTop" },
    lineStyle: { color: C.neutral, type: "dotted", width: 1, opacity: 0.65 },
    data: [0.382, 0.5, 0.618].map((r) => {
      const v = w.direction === "up" ? hi - (hi - lo) * r : lo + (hi - lo) * r;
      return { yAxis: +v.toFixed(2), label: { formatter: `${r} · ${v.toFixed(2)}` } };
    }),
  };
  return (
    <Panel title={`波浪 · 方向 ${w.direction} · 当前 ${w.current_wave} · 综合 ${w.total_score?.toFixed?.(2)} · 置信 ${w.confidence}`}
           extra="ElliottWaveAnalyzer（本地）">
      <Chart height={460} option={option} />
      <div className="grid-2" style={{ marginTop: 10 }}>
        <div>
          <h3 style={{ fontSize: 12, color: "var(--ink-3)", margin: "6px 0" }}>浪型结构</h3>
          <DT cols={[
            { k: "label", t: "浪", num: true }, { k: "start_date", t: "起", num: true },
            { k: "end_date", t: "止", num: true },
            { k: "start_price", t: "起价", num: true, render: (r) => Number(r.start_price).toFixed(2) },
            { k: "end_price", t: "止价", num: true, render: (r) => Number(r.end_price).toFixed(2) },
            { k: "pct", t: "幅度", num: true, render: (r) => <span className={r.pct >= 0 ? "pos" : "neg"}>{r.pct > 0 ? "+" : ""}{r.pct}%</span> },
          ]} rows={w.waves} maxH={260} />
        </div>
        <div>
          <h3 style={{ fontSize: 12, color: "var(--ink-3)", margin: "6px 0" }}>三大铁律</h3>
          <DT cols={[
            { k: "rule", t: "铁律" },
            { k: "ok", t: "判定", render: (r) => <span className={r.ok ? "pos" : "neg"}>{r.ok ? "✓ 通过" : "✗ 违反"}</span> },
          ]} rows={w.iron_rules} maxH={260} />
        </div>
      </div>
    </Panel>
  );
}

/* ── 四维评分 ── */
function ScorePanel({ code }) {
  const [d, setD] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const run = async () => {
    setBusy(true); setErr("");
    try { setD(await api(`/api/comprehensive/${code}`)); } catch (e) { setErr(String(e)); }
    setBusy(false);
  };
  const NAMES = { ma: "均线", struct: "动量", macd: "MACD双参", vol: "量价" };
  return (
    <Panel title="四维综合评分" extra="ComprehensiveAnalyzer（联网）"
           note="实时价/资金流/基本面走网络，失败自动降级为本地指标；结果缓存 15 分钟">
      <button className="btn primary" onClick={run} disabled={busy}>
        <Zap size={13} />{busy ? "分析中… 10-30s" : d ? "重新分析" : "运行综合分析（约 10-30 秒）"}
      </button>
      {err && <div className="empty" style={{ color: "var(--warn)" }}>{err}</div>}
      {d?.error && <div className="empty" style={{ color: "var(--crit)" }}>{d.error}</div>}
      {d && !d.error && (
        <>
          <div className="cards" style={{ marginTop: 14 }}>
            {[["综合判定", d.verdict], ["复合分", d.composite?.toFixed?.(2)], ["原始分", d.raw_score?.toFixed?.(2)],
              ["维度冲突", d.conflict ? "⚠️ 有降级" : "无"]].map(([l, v], i) => (
              <div className="stat" key={i}><div className="label">{l}</div>
                <div className="value" style={{ fontSize: i === 0 ? 18 : 25 }}>{v ?? "—"}</div></div>
            ))}
          </div>
          <Chart height={210} option={{
            grid: { left: 108, right: 48, top: 8, bottom: 26 },
            xAxis: { ...axisX({ type: "value", min: -1.3, max: 1.3 }) },
            yAxis: { ...axisY({ type: "category", splitLine: { show: false } }),
                     data: Object.keys(NAMES).map((k) => `${NAMES[k]}（${Math.round((d.weights?.[k] || 0) * 100)}%）`).reverse(),
                     axisLabel: { color: C.ink2, fontSize: 11 } },
            tooltip: tooltip({ trigger: "item", axisPointer: { type: "none" } }),
            series: [{ type: "bar", barMaxWidth: 16,
              data: Object.keys(NAMES).map((k) => d.dims?.[k]).reverse().map((v) => ({
                value: v, itemStyle: { color: v >= 0 ? C.blue : C.magenta, borderRadius: v >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3] } })),
              label: { show: true, position: "right", color: C.ink2, fontSize: 10, fontFamily: MONO,
                       formatter: (p) => (p.value >= 0 ? "+" : "") + p.value.toFixed(2) } }],
          }} />
          {d.key_levels?.length > 0 && (
            <details className="fold" style={{ marginTop: 10 }}><summary>关键位（支撑/压力）</summary>
              <div className="body"><DT cols={[
                { k: "role", t: "角色" }, { k: "value", t: "价位", num: true },
                { k: "sources", t: "来源", render: (r) => (Array.isArray(r.sources) ? r.sources : [r.source]).join("+") },
                { k: "dist", t: "距离%", num: true, render: (r) => r.dist?.toFixed?.(1) ?? r.dist },
                { k: "strength", t: "强度" },
              ]} rows={d.key_levels} maxH={220} /></div></details>
          )}
          {d.advice && Object.keys(d.advice).length > 0 && (
            <details className="fold" style={{ marginTop: 8 }}><summary>操作建议</summary>
              <div className="body"><Md text={"```json\n" + JSON.stringify(d.advice, null, 2) + "\n```"} /></div></details>
          )}
        </>
      )}
      {!d && !busy && !err && <div className="empty">点击上方按钮运行</div>}
    </Panel>
  );
}

/* ── 实时资金 ── */
function RtPanel({ code }) {
  const [d, setD] = useState(null);
  useEffect(() => { api(`/api/realtime/${code}`).then(setD).catch(() => setD({ rows: [] })); }, [code]);
  const q = d?.rows?.[0];
  if (!d) return <Loading />;
  if (!q) return <Panel title="实时资金"><div className="empty">实时报价无数据（非交易时间或接口限流）</div></Panel>;
  const F = (v) => v == null || Number.isNaN(v) ? "—" : v;
  return (
    <Panel title="实时资金" extra="free_data_client · TTL 60s" note="非交易时间为最近收盘快照（免费接口）">
      <div className="cards">
        <div className="stat"><div className="label">现价</div>
          <div className="value" style={{ color: q["涨跌幅%"] >= 0 ? C.up : C.down }}>{F(q["现价"])}</div>
          <div className="sub">{F(q["涨跌幅%"])}%</div></div>
        <div className="stat"><div className="label">PE(TTM)</div><div className="value">{F(q["PE_TTM"])}</div></div>
        <div className="stat"><div className="label">PB</div><div className="value">{F(q["PB"])}</div></div>
        <div className="stat"><div className="label">总市值</div>
          <div className="value">{q["总市值亿"] > 10000 ? (q["总市值亿"] / 10000).toFixed(1) + "万亿" : F(q["总市值亿"]) + "亿"}</div></div>
        <div className="stat"><div className="label">换手率</div><div className="value">{F(q["换手率%"])}%</div></div>
      </div>
    </Panel>
  );
}
