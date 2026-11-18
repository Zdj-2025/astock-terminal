import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import Chart, { C, axisX, axisY, tooltip, legend } from "../lib/chart";
import { PageHead, Stat, Panel, DT, Pct, Loading, Md } from "../components/ui";

const MONO = '"IBM Plex Mono", Menlo, monospace';

export default function Portfolio() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api("/api/page/portfolio").then(setD).catch((e) => setErr(String(e))); }, []);
  if (err) return <Panel title="加载失败">{err}</Panel>;
  if (!d) return <Loading />;

  const m = d.metrics;
  return (
    <>
      <PageHead eyebrow="Portfolio & Net Value" title="持仓与净值中心" chips={[
        ["锚点 ", d.anchor_day], ["现金 ", `${m.cash_pct}%`], ["日频重构至 ", d.equity_end],
      ]} />

      <div className="cards">
        <Stat label="总资产" text={`${m.total}万`} sub={`锚点 ${d.anchor_day}`} />
        <Stat label="可用资金" text={`${m.cash}万`} sub={m.cash_pct < 20 ? "⚠️ 破20%铁线" : "≥20% 铁线"}
              alert={m.cash_pct < 20} />
        <Stat label="持仓市值" text={`${m.mv}万`} sub={`仓位 ${m.pos_pct}%`} />
        <Stat label="现金比例" value={m.cash_pct} sub="铁线20% · 防守30%" decimals={1}
              dot={m.cash_pct < 20 ? C.up : C.aqua} alert={m.cash_pct < 20} />
        <Stat label="当日盈亏" text={m.day_pnl == null ? "—" : m.day_pnl} sub="无当日数据" />
      </div>

      <Panel title="持仓明细" extra="现价 = 各股 D.parquet 最新收盘（UI 内重估，不写回）">
        <DT cols={[
          { k: "code", t: "代码", num: true },
          { k: "name", t: "名称", render: (r) => <span className="strong">{r.name}</span> },
          { k: "type", t: "类型" },
          { k: "cost", t: "成本", num: true, render: (r) => Number(r.cost).toFixed(3) },
          { k: "last", t: "现价", num: true, render: (r) => Number(r.last).toFixed(3) },
          { k: "pnl", t: "盈亏%", num: true, render: (r) => <Pct v={r.pnl} bg /> },
          { k: "shares", t: "股数", num: true, render: (r) => Number(r.shares).toLocaleString() },
          { k: "mv_wan", t: "市值万", num: true },
          { k: "pos_pct", t: "仓位%", num: true, render: (r) => r.pos_pct?.toFixed?.(1) ?? r.pos_pct },
          { k: "stop", t: "止损线", num: true, render: (r) => r.stop?.toFixed?.(3) ?? r.stop },
          { k: "d2stop", t: "距止损%", num: true, render: (r) => r.d2stop == null ? "—" :
            <span style={{ color: r.d2stop < 0 ? "var(--crit)" : undefined, fontWeight: r.d2stop < 0 ? 700 : 400 }}>+{r.d2stop}%</span> },
          { k: "date", t: "数据日", num: true },
        ]} rows={d.positions} maxH={380} />
      </Panel>

      <div className="grid-2">
        <div>
          <NetPanel d={d} />
          <Panel title="近 20 笔交易" extra="trade_log.csv">
            <DT cols={[
              { k: "date", t: "日期", num: true }, { k: "name", t: "名称", render: (r) => <span className="strong">{r.name}</span> },
              { k: "action", t: "操作", render: (r) => <span className={r.action === "卖" ? "neg" : r.action === "买" ? "pos" : undefined}>{r.action}</span> },
              { k: "price", t: "价格", num: true, render: (r) => r.price?.toFixed?.(3) ?? r.price },
              { k: "shares", t: "数量", num: true, render: (r) => r.shares?.toLocaleString?.() ?? r.shares },
              { k: "realized_pnl", t: "已实现", render: (r) => r.realized_pnl || "" },
            ]} rows={d.trades} maxH={300} empty="无交易流水" />
          </Panel>
        </div>
        <div>
          <GuardPanel pg={d.profit_guard} />
          <AxisPanel ta={d.three_axis} />
          <Panel title="净值月报" extra="旧债口径，重述跳票中">
            {d.monthly_md ? (
              <details className="fold"><summary>展开月报</summary>
                <div className="body"><Md text={d.monthly_md} /></div></details>
            ) : <div className="empty">无月报文件</div>}
          </Panel>
        </div>
      </div>
    </>
  );
}

function NetPanel({ d }) {
  const option = useMemo(() => {
    const eq = d.equity, base = eq.total[0];
    const nv = eq.total.map((v) => +(v / base).toFixed(4));
    const dd = eq.dd.map((v) => +v.toFixed(2));
    const series = [
      { name: "总盘子", type: "line", data: nv, showSymbol: false, smooth: 0.25,
        lineStyle: { color: C.orange, width: 2.4 }, z: 5,
        areaStyle: { color: "rgba(217,89,38,0.04)" } },
    ];
    [["上证", "sse", C.blue], ["沪深300", "hs300", C.aqua]].forEach(([nm, key, cl]) => {
      const b = d.benchmarks[key];
      if (!b) return;
      const b0 = b.values[0];
      series.push({ name: nm, type: "line", showSymbol: false,
        data: b.dates.map((dt, i) => [dt, +(b.values[i] / b0).toFixed(4)]),
        lineStyle: { color: cl, width: 1.4, type: "dotted" }, z: 4 });
    });
    series.push({ name: "回撤", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: dd,
      showSymbol: false, lineStyle: { color: C.down, width: 1.1 },
      areaStyle: { color: "rgba(31,199,159,0.10)" }, z: 3 });
    series.push({ name: "对账锚点", type: "scatter", data: d.snapshots.map((s) => [s.asof, +(s.total / base).toFixed(4)]),
      symbol: "diamond", symbolSize: 9, itemStyle: { color: C.warn, borderColor: "#111722", borderWidth: 2 }, z: 6 });
    return {
      grid: [{ left: 44, right: 12, top: 30, height: "55%" }, { left: 44, right: 12, top: "76%", height: "16%" }],
      xAxis: [axisX({ type: "category", data: eq.dates, boundaryGap: false }),
              { ...axisX({ type: "category", data: eq.dates, boundaryGap: false }), gridIndex: 1 }],
      yAxis: [axisY({ scale: true, name: "归一", nameTextStyle: { color: C.ink3, fontSize: 10 } }),
              axisY({ gridIndex: 1, max: 0, name: "回撤%", nameTextStyle: { color: C.ink3, fontSize: 10 } })],
      tooltip: tooltip({ trigger: "axis" }),
      legend: legend(),
      series,
    };
  }, [d]);
  return (
    <Panel title="净值 vs 基准" note={<>诚实口径：曲线为「总盘子」（含出入金干扰，非严格净值）；日频重构停在 <b style={{ color: "var(--ink-1)" }}>{d.equity_end}</b>，之后仅剩锚点——R0 断档在此可见</>}>
      <Chart height={430} option={option} />
    </Panel>
  );
}

function GuardPanel({ pg }) {
  return (
    <Panel title="profit_guard 止损建议" extra="算法口径">
      {pg.stocks?.length ? (
        <DT cols={[
          { k: "name", t: "名称", render: (r) => <span className="strong">{r.name}</span> },
          { k: "old_sl", t: "旧止损", num: true, render: (r) => Number(r.old_sl).toFixed(2) },
          { k: "new_sl", t: "新止损", num: true, render: (r) => Number(r.new_sl).toFixed(2) },
          { k: "reason", t: "理由" },
        ]} rows={pg.stocks} maxH={200} />
      ) : <div className="empty">个股止损已按算法到位 ✓</div>}
      {pg.etfs?.length > 0 && (
        <div className="note">宽基ETF算法-10%不采纳（用再评估线/不砍地板）：
          {pg.etfs.map((u) => ` ${u.name}${Number(u.new_sl).toFixed(2)}`).join("、")}</div>
      )}
    </Panel>
  );
}

function AxisPanel({ ta }) {
  const colors = { 类债: C.aqua, 宽基防守: C.blue, 权益进攻: C.orange, 现金: C.neutral };
  const keys = Object.keys(colors);
  return (
    <Panel title="三轴配置" extra="总盘子口径" note="⚠️ 目标带矛盾未统一（STATE.md 六）：#13 防守 20/50/30 vs 九·shake 50/20/30，只展示现状不做判定">
      <Chart height={150} option={{
        grid: { left: 40, right: 20, top: 30, bottom: 42 },
        xAxis: { ...axisX({ type: "value", max: 100 }), name: "占总盘子 %", nameTextStyle: { color: C.ink3, fontSize: 10 } },
        yAxis: { ...axisY({ type: "category", data: ["配置"], splitLine: { show: false } }) },
        tooltip: tooltip({ trigger: "item", axisPointer: { type: "none" }, formatter: (p) => `${p.seriesName} ${p.value}%` }),
        legend: legend({ top: 0, left: 60 }),
        series: keys.map((k) => ({
          name: k, type: "bar", stack: "all", barWidth: 34, data: [ta[k]],
          itemStyle: { color: colors[k], borderColor: "#111722", borderWidth: 2 },
          label: { show: ta[k] >= 5, position: "inside", color: "#fff", fontSize: 11, fontFamily: MONO,
                   formatter: () => `${k} ${Math.round(ta[k])}%` },
        })),
      }} />
    </Panel>
  );
}
