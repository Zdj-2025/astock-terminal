import { marked } from "marked";
import useCountUp from "../lib/useCountUp";

export function PageHead({ eyebrow, title, chips }) {
  return (
    <div className="pagehead">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {chips?.length > 0 && (
        <div className="chips">
          {chips.map(([k, v]) => v == null || v === "" ? null : (
            <span key={k} className="chip">{k}<b>{v}</b></span>
          ))}
        </div>
      )}
    </div>
  );
}

export function Stat({ label, value, text, sub, decimals = 0, dot, alert }) {
  const shown = useCountUp(value, { decimals });
  return (
    <div className={`stat${alert ? " alert" : ""}`}>
      <div className="label">
        {dot && <span className="dot" style={{ background: dot }} />} {label}
      </div>
      <div className="value">{text ?? shown}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Panel({ title, extra, note, children, style }) {
  return (
    <div className="panel" style={style}>
      {title && <h3><span className="t">{title}</span>{extra && (
        <span style={{ marginLeft: "auto", fontWeight: 400, fontSize: 11, color: "var(--ink-3)" }}>{extra}</span>)}</h3>}
      {children}
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function Tabs({ tabs, onTab, active }) {
  return (
    <div className="tabs">
      {tabs.map(([key, label, n]) => (
        <button key={key} className={`tab${active === key ? " on" : ""}`} onClick={() => onTab(key)}>
          {label}{n != null && <span className="n">{n}</span>}
        </button>
      ))}
    </div>
  );
}

/** 通用数据表：cols = [{k, t(标题), num, render}] */
export function DT({ cols, rows, maxH = 460, empty = "暂无数据" }) {
  if (!rows?.length) return <div className="empty">{empty}</div>;
  return (
    <div className="dt-scroll" style={{ maxHeight: maxH }}>
      <table className="dt">
        <thead>
          <tr>{cols.map((c) => <th key={c.k} className={c.num ? "num" : ""}>{c.t}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{cols.map((c) => (
              <td key={c.k} className={c.num ? "num" : ""}>{c.render ? c.render(r) : r[c.k]}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const LV = { "🔴": ["lv-red", "真止损"], "🟠": ["lv-orange", "见底"], "🟡": ["lv-yellow", "移动止损"], "🟢": ["lv-green", "持有"] };
export function Level({ trigger }) {
  const emoji = [...String(trigger ?? "")][0];           // emoji 是代理对，必须按码点取
  const [cls, name] = LV[emoji] || ["lv-gray", ""];
  const text = String(trigger ?? "").slice(emoji?.length ?? 0)
    .replace(/（[^）]*）|\([^)]*\)/g, "").trim() || name;   // 去掉与 MA20 列重复的括号值
  return <span className={`lv ${cls}`} title={trigger}><i />{text}</span>;
}

export function Pct({ v, bg }) {
  if (v == null || v === "" || Number.isNaN(v)) return <span>—</span>;
  return <span className={v >= 0 ? (bg ? "pos-bg" : "pos") : (bg ? "neg-bg" : "neg")}>
    {v > 0 ? "+" : ""}{Number(v).toFixed(1)}%
  </span>;
}

export function Num({ v, d = 2 }) {
  if (v == null || v === "" || Number.isNaN(v)) return <span>—</span>;
  return <>{Number(v).toFixed(d)}</>;
}

export function Md({ text }) {
  return <div className="md" dangerouslySetInnerHTML={{ __html: marked.parse(text || "") }} />;
}

export function Loading({ text = "加载中" }) {
  return <div className="loading"><span className="spinner" />{text}</div>;
}
