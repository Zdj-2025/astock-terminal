import { useEffect, useState } from "react";
import { api } from "../lib/api";
import Chart, { C, axisX, axisY, tooltip } from "../lib/chart";
import { PageHead, Stat, Panel, DT, Loading, Md } from "../components/ui";

export default function Health() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api("/api/page/health").then(setD).catch((e) => setErr(String(e))); }, []);
  if (err) return <Panel title="加载失败">{err}</Panel>;
  if (!d) return <Loading />;

  const f = d.freshness;
  return (
    <>
      <PageHead eyebrow="System Health" title="系统健康" chips={[
        ["基准 ", f.bench], ["达标率 ", `${f.ok_rate}%`], ["K线文件 ", f.files.toLocaleString()],
      ]} />

      <div className="cards">
        <Stat label="基准交易日" text={f.bench?.slice(5)} sub="kline/000001 max" />
        <Stat label="全市场K线文件" value={f.files} sub="D.parquet 实测扫描" decimals={0} />
        <Stat label="水位达标率" value={f.ok_rate} sub="mtime ≥ 基准日" decimals={1}
              dot={f.ok_rate >= 99 ? "#0ca30c" : C.warn} />
        <Stat label="分钟哨兵" text={d.sentinel?.slice(5) || "缺文件"}
              sub={d.sentinel ? "" : "缺失=分钟cron可能静默失败"} alert={!d.sentinel} />
      </div>

      <div className="grid-2">
        <div>
          <Panel title="K线新鲜度分布" extra="文件 mtime 实测"
                 note="manifest.json 停更 2026-09-15、inventory.json 停更 2026-05-23（元数据仅参考，本图为实测文件水位）">
            <Chart height={280} option={{
              grid: { left: 52, right: 16, top: 14, bottom: 26 },
              xAxis: axisX({ type: "category", data: f.hist.map((h) => h.day) }),
              yAxis: axisY({ name: "文件数", nameTextStyle: { color: C.ink3, fontSize: 10 } }),
              tooltip: tooltip({ trigger: "axis" }),
              series: [{ type: "bar", data: f.hist.map((h) => h.n), barMaxWidth: 22,
                         itemStyle: { color: C.blue, borderRadius: [3, 3, 0, 0] } }],
            }} />
          </Panel>

          <Panel title="持仓 & 自选股数据滞后" extra="parquet max(date) vs 基准">
            <DT cols={[
              { k: "code", t: "代码", num: true },
              { k: "name", t: "名称", render: (r) => <span className="strong">{r.name}</span> },
              { k: "date", t: "数据至", num: true },
              { k: "ok", t: "状态", render: (r) => r.ok
                ? <span className="lv lv-green"><i />最新</span>
                : <span className="lv lv-orange"><i />落后</span> },
            ]} rows={d.lag} maxH={300} />
          </Panel>

          <Panel title="health_check 近 10 日">
            <DT cols={[
              { k: "date", t: "日期", num: true },
              { k: "pass", t: "结果", render: (r) => <span className={r.pass ? "pos" : "neg"}>{r.pass ? "PASS" : "FAIL"}</span> },
              { k: "fails", t: "失败项", render: (r) => r.fails?.length ? r.fails.join("；") : "—" },
            ]} rows={d.health_checks} maxH={280} empty="无 health_check.log" />
          </Panel>
        </div>

        <div>
          <Panel title="OpenClaw cron 任务" extra="jobs.json.migrated">
            <DT cols={[
              { k: "name", t: "任务", render: (r) => <span className="strong">{r.name}</span> },
              { k: "expr", t: "schedule", num: true },
              { k: "enabled", t: "状态", render: (r) => r.enabled
                ? <span className="lv lv-green"><i />启用</span>
                : <span className="lv lv-gray"><i />停用</span> },
            ]} rows={d.cron} maxH={300} empty="OpenClaw cron 配置不可读" />
          </Panel>

          <Panel title="STATE.md 蓝图进度 R0-R4">
            {d.state.progress ? <Md text={d.state.progress} /> : <div className="empty">无</div>}
            {d.state.delist && (
              <details className="fold" style={{ marginTop: 10 }}><summary>该删清单（全文）</summary>
                <div className="body"><Md text={d.state.delist} /></div></details>
            )}
          </Panel>

          <Panel title="日志尾行" extra="smart_tracker / first_board">
            {Object.entries(d.logs).map(([k, v]) => (
              <details className="fold" key={k} style={{ marginBottom: 8 }}><summary>{k}.log</summary>
                <div className="body"><pre className="term" style={{ maxHeight: 160 }}>{v.join("\n") || "（空）"}</pre></div>
              </details>
            ))}
          </Panel>
        </div>
      </div>
    </>
  );
}
