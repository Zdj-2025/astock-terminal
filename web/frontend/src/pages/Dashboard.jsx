import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import Chart, { C, axisX, axisY, tooltip, legend } from "../lib/chart";
import { PageHead, Stat, Panel, Tabs, DT, Level, Pct, Loading, Md } from "../components/ui";
import { FileText, Play, RotateCw } from "lucide-react";

const fmtWan = (v) => v == null || v === "" ? "—" : (Number(v) / 1e4).toFixed(1);

/** 名称 → 可点击跳个股工作台 */
const NameLink = ({ r }) => (
  <Link to={`/stock?code=${r.code}`} className="strong" style={{ color: "var(--ink-1)", textDecoration: "none" }}
        title="进入个股分析">{r.name}</Link>
);
const Num2 = ({ v }) => (v == null || v === "" ? "—" : Number(v).toFixed(2));
const Bool = ({ v, t }) => (v == null || v === "" ? "—" : v === true || v === "True" || v === "true" ? `✓${t || ""}` : "—");
const Tag = ({ v }) => (v ? <span style={{ color: String(v).includes("放") ? "var(--warn)" : "var(--down)" }}>{v}</span> : "—");

export default function Dashboard() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api("/api/page/dashboard").then(setD).catch((e) => setErr(String(e))); }, []);

  const [tab, setTab] = useState("sl");
  const [prem, setPrem] = useState(null);        // 晨报/晚报共用视图
  const [premTitle, setPremTitle] = useState("");
  const [premBusy, setPremBusy] = useState(false);
  const [premShow, setPremShow] = useState(false);

  if (err) return <Panel title="加载失败">{err}</Panel>;
  if (!d) return <Loading />;

  const sg = d.signals || {};
  const tabDefs = [
    ["sl", "止损警示", d.stoploss?.length],
    ["golden_cross", "金叉", sg.golden_cross?.rows?.length],
    ["divergence", "底背离", sg.divergence?.rows?.length],
    ["oversold_reversal", "超跌反转", sg.oversold_reversal?.rows?.length],
  ];

  const genReport = async (kind, refresh = false) => {
    setPremBusy(true); setPremShow(true);
    setPremTitle(kind === "am" ? "盘前晨报（等宽）" : "收盘总结（晚报 · 只读组装）");
    try {
      const url = kind === "am"
        ? `/api/morning-report${refresh ? "?refresh=1" : ""}`
        : "/api/evening-report";
      const r = await api(url);
      setPrem(r.text);
    } catch (e) { setPrem(String(e)); }
    setPremBusy(false);
  };

  return (
    <>
      <PageHead eyebrow="Daily Cockpit" title="今日决策驾驶舱" chips={[
        ["数据 ", `${d.bench?.slice(5)} · 延迟 T-1`],
        ["现金 ", `${d.cash.pct}% · ${d.cash.asof?.slice(5)}`],
      ]} />

      <div className="cards">
        <Stat label="数据水位" text={d.bench?.slice(5) === d.today ? "T-0" : "T-1"}
              sub={`${d.bench?.slice(5)} · 今日 ${d.today}`} dot={d.bench?.slice(5) === d.today ? "#0ca30c" : C.warn} />
        <Stat label="真止损" value={d.stop.red} sub={`见底${d.stop.orange} · 移动${d.stop.yellow}`} decimals={0}
              dot={d.stop.red > 0 ? C.up : "#0ca30c"} alert={d.stop.red > 0} />
        <Stat label="金叉" value={sg.golden_cross?.rows?.length ?? 0} sub={sg.golden_cross?.date?.slice(5)} decimals={0} />
        <Stat label="底背离" value={sg.divergence?.rows?.length ?? 0} sub={sg.divergence?.date?.slice(5)} decimals={0} />
        <Stat label="超跌反转" value={sg.oversold_reversal?.rows?.length ?? 0} sub={sg.oversold_reversal?.date?.slice(5)} decimals={0} />
        <Stat label="现金比例" value={d.cash.pct} sub={`锚点 ${d.cash.asof?.slice(5)}`} decimals={1}
              dot={d.cash.pct < 20 ? C.up : C.aqua} alert={d.cash.pct < 20} />
      </div>

      <div className="grid-2">
        <div>
          <Panel title="扫描信号" extra="数据 = scan_history.parquet">
            <Tabs tabs={tabDefs} active={tab} onTab={setTab} />
            {tab === "sl" ? (
              <>
                <DT cols={[
                  { k: "trigger", t: "级", render: (r) => <Level trigger={r.trigger} /> },
                  { k: "code", t: "代码", num: true },
                  { k: "name", t: "名称", render: (r) => <NameLink r={r} /> },
                  { k: "cost", t: "成本", num: true, render: (r) => Number(r.cost).toFixed(3) },
                  { k: "cur", t: "现价", num: true, render: (r) => Number(r.cur).toFixed(3) },
                  { k: "pnl", t: "盈亏%", num: true, render: (r) => <Pct v={r.pnl} bg /> },
                  { k: "ma20", t: "MA20", num: true, render: (r) => Number(r.ma20).toFixed(2) },
                  { k: "v_ratio", t: "量比", num: true, render: (r) => Number(r.v_ratio).toFixed(2) },
                  { k: "val", t: "市值万", num: true, render: (r) => fmtWan(r.val) },
                ]} rows={d.stoploss} maxH={400} />
                <div className="note">🔴真止损→次日执行 · 🟠见底→持有等反弹（别砍地板） · 🟡移动止损→减半保护利润 · 复用 <code>stop_loss_check()</code></div>
              </>
            ) : (
              <StrategyTab s={sg[tab]} strat={tab} />
            )}
          </Panel>

          <AccumulatePanel acc={d.accumulate} />
        </div>

        <div>
          <Panel title="晨报 / 晚报" extra="盘前检查 · 收盘总结">
            <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
              <button className="btn primary" onClick={() => genReport("am")} disabled={premBusy}>
                <Play size={13} />{premBusy ? "生成中…" : "生成晨报"}
              </button>
              <button className="btn" onClick={() => genReport("pm")} disabled={premBusy}>
                <FileText size={13} />生成晚报
              </button>
              <button className="btn" onClick={() => genReport("am", true)} disabled={premBusy}
                      title="重跑晨报 subprocess（忽略当日缓存）">
                <RotateCw size={13} />重生成
              </button>
            </div>
            {premShow && (
              <details className="fold" open>
                <summary><FileText size={13} /> {premTitle}</summary>
                <div className="body">
                  {premBusy && !prem ? <Loading text="晨报较慢：财报季历走 tushare 逐票限速；晚报即时" /> :
                    <pre className="term">{prem}</pre>}
                </div>
              </details>
            )}
            {!premShow && (
              <div className="empty">
                晨报 = 盘前持仓检查（subprocess，首次 1-3 分钟，当日缓存）<br />
                晚报 = 收盘总结（只读组装当日信号/止损/累积榜/公告/闭环，秒出）
              </div>
            )}
          </Panel>

          <AnnPanel ann={d.announcements} />
          <RecPanel recs={d.recommendations} />
        </div>
      </div>
    </>
  );
}

/* 各策略列定义（对象格式，含格式化与语义色）——数据列见 scan_history.parquet */
const STRAT_COLS = {
  golden_cross: [
    { k: "industry", t: "行业" },
    { k: "price", t: "价格", num: true, render: (r) => <Num2 v={r.price} /> },
    { k: "dist120", t: "距120日高", num: true, render: (r) => <Pct v={r.dist120} /> },
    { k: "vs_ma20", t: "距MA20", num: true, render: (r) => <Pct v={r.vs_ma20} /> },
    { k: "vol_tag", t: "量能", render: (r) => <Tag v={r.vol_tag} /> },
    { k: "vr", t: "量能位", num: true, render: (r) => <Num2 v={r.vr} /> },
    { k: "low_pos", t: "低位标记" },
    { k: "smart", t: "smart", render: (r) => <Bool v={r.smart} /> },
  ],
  divergence: [
    { k: "industry", t: "行业" },
    { k: "price", t: "价格", num: true, render: (r) => <Num2 v={r.price} /> },
    { k: "dist120", t: "距120日高", num: true, render: (r) => <Pct v={r.dist120} /> },
    { k: "strength", t: "背离强度", num: true, render: (r) => <Num2 v={r.strength} /> },
    { k: "confirmed", t: "确认", render: (r) => <Bool v={r.confirmed} /> },
    { k: "db", t: "双底", render: (r) => <Bool v={r.db} /> },
    { k: "vol_tag", t: "量能", render: (r) => <Tag v={r.vol_tag} /> },
    { k: "vratio", t: "量比", num: true, render: (r) => <Num2 v={r.vratio} /> },
    { k: "turnover", t: "换手%", num: true, render: (r) => <Num2 v={r.turnover} /> },
  ],
  oversold_reversal: [
    { k: "industry", t: "行业" },
    { k: "price", t: "价格", num: true, render: (r) => <Num2 v={r.price} /> },
    { k: "ret20", t: "近20日", num: true, render: (r) => <Pct v={r.ret20} /> },
    { k: "pos250", t: "250日位%", num: true, render: (r) => <Num2 v={r.pos250} /> },
    { k: "upside", t: "反弹空间", num: true, render: (r) => <Pct v={r.upside} /> },
  ],
};

function StrategyTab({ s, strat }) {
  if (!s?.rows?.length) return <div className="empty">该策略近 20 个交易日无记录</div>;
  const cols = [
    { k: "code", t: "代码", num: true },
    { k: "name", t: "名称", render: (r) => <NameLink r={r} /> },
    ...STRAT_COLS[strat] || [],
  ];
  return (
    <>
      <div className="note" style={{ marginTop: 0, marginBottom: 8 }}>
        数据日期 <b style={{ color: "var(--ink-1)" }}>{s.date}</b>（该策略最近有记录日）· {s.rows.length} 只 ·
        点击名称进入个股分析
      </div>
      <DT cols={cols} rows={s.rows} maxH={400} />
    </>
  );
}

function AccumulatePanel({ acc }) {
  if (!acc?.length) return null;
  const top = acc.slice(0, 20).reverse();
  return (
    <Panel title="信号累积榜" extra="近20日 · 横向²×log纵向">
      <Chart height={Math.max(340, 24 * top.length + 70)} option={{
        grid: { left: 118, right: 56, top: 8, bottom: 24 },
        xAxis: { ...axisX({ type: "value", max: (v) => v.max * 1.25 }), name: "综合分", nameTextStyle: { color: C.ink3, fontSize: 10 } },
        yAxis: { ...axisY({ type: "category", splitLine: { show: false } }),
                 data: top.map((r) => `${r.name}(${r.code})`), axisLabel: { color: C.ink2, fontSize: 10.5 } },
        tooltip: tooltip({ trigger: "item", axisPointer: { type: "none" },
                           formatter: (p) => `${p.name}<br/>综合分 ${Number(p.value).toFixed(4)}` }),
        series: [{ type: "bar", data: top.map((r) => r.composite), itemStyle: { color: C.orange, borderRadius: [0, 3, 3, 0] }, barMaxWidth: 16,
                   label: { show: true, position: "right", color: C.ink2, fontSize: 10, fontFamily: MONO_,
                           formatter: (p) => Number(p.value).toFixed(3) } }],
      }} />
    </Panel>
  );
}

const MONO_ = '"IBM Plex Mono", Menlo, monospace';

function AnnPanel({ ann }) {
  if (!ann?.rows?.length) return null;
  return (
    <Panel title="公告监控" extra={ann.day}>
      {ann.rows.map((r, i) => (
        <div className="ann" key={i}>
          <span className="code">{r.code}</span>
          <span className={`pri ${r.priority?.includes("高") ? "pri-high" : r.priority?.includes("中") ? "pri-mid" : "pri-low"}`}>
            {r.priority}
          </span>
          <a href={r.link} target="_blank" rel="noreferrer">{r.title}</a>
        </div>
      ))}
    </Panel>
  );
}

function RecPanel({ recs }) {
  if (!recs?.length) return null;
  const d = recs.filter((r) => r.pnl20 != null && r.pnl20 !== "").slice(0, 15).reverse();
  return (
    <Panel title="推荐闭环" extra="近20条 · pnl20" note={<>族卡口径见 STATE.md「四、度量滚动」：<Md text="divergence 持续为正；smart 连续为负已列该删清单" /></>}>
      {d.length > 0 && (
        <Chart height={Math.max(300, 22 * d.length + 60)} option={{
          grid: { left: 108, right: 52, top: 8, bottom: 24 },
          xAxis: axisX({ type: "value" }),
          yAxis: { ...axisY({ type: "category", splitLine: { show: false } }),
                   data: d.map((r) => `${r.name}(${String(r.code).padStart(6, "0")})`), axisLabel: { color: C.ink2, fontSize: 10.5 } },
          tooltip: tooltip({ trigger: "item", axisPointer: { type: "none" },
                             formatter: (p) => `${p.name}<br/>pnl20 ${p.value >= 0 ? "+" : ""}${p.value.toFixed(2)}%` }),
          series: [{ type: "bar", data: d.map((r) => ({ value: r.pnl20, itemStyle: { color: r.pnl20 >= 0 ? C.up : C.down, borderRadius: r.pnl20 >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3] } })),
                     barMaxWidth: 15,
                     label: { show: true, position: "right", color: C.ink2, fontSize: 10, fontFamily: MONO_,
                             formatter: (p) => `${p.value.toFixed(1)}%` } }],
        }} />
      )}
      <details className="fold" style={{ marginTop: 10 }}>
        <summary>推荐明细</summary>
        <div className="body">
          <DT cols={[
            { k: "rec_date", t: "日期", num: true }, { k: "name", t: "名称", render: (r) => <span className="strong">{r.name}</span> },
            { k: "family", t: "族" }, { k: "buy_point", t: "买点" },
            { k: "pnl5", t: "pnl5", num: true, render: (r) => <Pct v={r.pnl5} /> },
            { k: "pnl10", t: "pnl10", num: true, render: (r) => <Pct v={r.pnl10} /> },
            { k: "pnl20", t: "pnl20", num: true, render: (r) => <Pct v={r.pnl20} /> },
          ]} rows={recs} maxH={300} />
        </div>
      </details>
    </Panel>
  );
}
