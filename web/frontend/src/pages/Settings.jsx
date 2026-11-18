import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { PageHead, Panel, Loading } from "../components/ui";
import { Save, PlugZap, ExternalLink } from "lucide-react";

const SRC_LABEL = { env: ".env / 环境变量", 界面: "设置页保存", 默认: "内置默认" };

/** 来源徽章 */
const Src = ({ s }) => (
  <span className="tag" title={`当前值来源：${SRC_LABEL[s] || s}`}>{SRC_LABEL[s] || s}</span>
);

/** 一行可编辑配置：label + 当前值(脱敏) + 输入框 */
function Field({ label, cur, children, hint }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, flexWrap: "wrap" }}>
        <span className="strong">{label}</span>
        {cur?.source && <Src s={cur.source} />}
        {cur?.masked && <code style={{ color: "var(--ink-3)" }}>{cur.masked}</code>}
        {cur?.value && !cur.masked && cur.source !== "默认" && <code style={{ color: "var(--ink-3)" }}>{cur.value}</code>}
      </div>
      {children}
      {hint && <div className="note" style={{ marginTop: 4 }}>{hint}</div>}
    </div>
  );
}

const input = { width: "100%", boxSizing: "border-box" };

export default function Settings() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");           // 保存/测试结果提示
  const [busy, setBusy] = useState("");

  const [tsToken, setTsToken] = useState("");
  const [llmKey, setLlmKey] = useState("");
  const [llmUrl, setLlmUrl] = useState("");
  const [llmModel, setLlmModel] = useState("");

  const load = () => api("/api/settings").then(setD).catch((e) => setErr(String(e)));
  useEffect(() => { load(); }, []);

  if (err) return <Panel title="加载失败">{err}</Panel>;
  if (!d) return <Loading />;

  const save = async (patch, what) => {
    setBusy(what); setMsg("");
    try {
      const r = await api("/api/settings", { method: "POST", body: patch });
      setD(r); setMsg("已保存 → config/settings.local.json（立即生效）");
      setTsToken(""); setLlmKey(""); setLlmUrl(""); setLlmModel("");
    } catch (e) { setMsg(`保存失败：${e}`); }
    setBusy("");
  };

  const test = async (ep, what) => {
    setBusy(what); setMsg("");
    try {
      const r = await api(`/api/settings/${ep}`, { method: "POST" });
      setMsg(r.ok ? `${what} 连通正常${r.model ? `（${r.model}）` : ""}${r.reply ? `：${r.reply}` : ""}`
                  : `${what} 失败：${r.error}`);
    } catch (e) { setMsg(`${what} 失败：${e}`); }
    setBusy("");
  };

  return (
    <>
      <PageHead eyebrow="Settings" title="设置" chips={[["数据目录 ", d.dirs.data]]} />

      <div className="grid-2">
        <Panel title="数据源 · Tushare" extra="日线下载 / 缠论 / 综合评分">
          <Field label="API Token" cur={d.tushare_token}
                 hint={<>免费注册即送 120 积分（够日线/基础接口）：<a href="https://tushare.pro/user/register" target="_blank" rel="noreferrer">tushare.pro <ExternalLink size={11} /></a>。付费充值提高积分档位后同一 token 自动解锁更多接口与频次，无需改配置。</>}>
            <input style={input} value={tsToken} onChange={(e) => setTsToken(e.target.value)}
                   placeholder={d.tushare_token.configured ? "留空 = 保持现状" : "粘贴你的 token"} />
          </Field>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn primary" disabled={!tsToken.trim() || busy === "保存"}
                    onClick={() => save({ tushare_token: tsToken.trim() }, "保存")}>
              <Save size={13} />{busy === "保存" ? "保存中…" : "保存"}
            </button>
            <button className="btn" disabled={!d.tushare_token.configured || busy === "测试数据源"}
                    onClick={() => test("test-tushare", "测试数据源")}>
              <PlugZap size={13} />测试连接
            </button>
          </div>
          <div className="note" style={{ marginTop: 10 }}>
            命令行脚本（update_all_kline / daily_scan_all）读取同一份配置——界面上保存后，crontab 里的脚本直接可用。
          </div>
        </Panel>

        <Panel title="大模型 · OpenAI 兼容" extra="AI 解盘">
          <Field label="Base URL" cur={d.llm_base_url}>
            <input style={input} value={llmUrl} onChange={(e) => setLlmUrl(e.target.value)}
                   placeholder={d.llm_base_url.value} />
          </Field>
          <Field label="Model" cur={d.llm_model}>
            <input style={input} value={llmModel} onChange={(e) => setLlmModel(e.target.value)}
                   placeholder={d.llm_model.value} />
          </Field>
          <Field label="API Key" cur={d.llm_api_key}>
            <input style={{ ...input, fontFamily: "var(--mono, monospace)" }} type="password"
                   value={llmKey} onChange={(e) => setLlmKey(e.target.value)}
                   placeholder={d.llm_api_key.configured ? "留空 = 保持现状" : "sk-… / 任意兼容服务密钥"} />
          </Field>
          <div className="note" style={{ marginBottom: 10 }}>
            常用服务（点击填入 Base URL，Key 各自官网获取）：
            {[
              ["GLM", "https://open.bigmodel.cn/api/paas/v4"],
              ["DeepSeek", "https://api.deepseek.com/v1"],
              ["OpenAI", "https://api.openai.com/v1"],
              ["本地 Ollama", "http://localhost:11434/v1"],
            ].map(([n, u]) => (
              <a key={n} href="#" onClick={(e) => { e.preventDefault(); setLlmUrl(u); setLlmModel(n === "本地 Ollama" ? "qwen2.5:7b" : n === "GLM" ? "glm-4-flash" : n === "DeepSeek" ? "deepseek-chat" : "gpt-4o-mini"); }}
                 style={{ marginLeft: 8 }}>{n}</a>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn primary"
                    disabled={busy === "保存模型" || !(llmKey.trim() || llmUrl.trim() || llmModel.trim())}
                    onClick={() => save({
                      ...(llmKey.trim() && { llm_api_key: llmKey.trim() }),
                      ...(llmUrl.trim() && { llm_base_url: llmUrl.trim() }),
                      ...(llmModel.trim() && { llm_model: llmModel.trim() }),
                    }, "保存模型")}>
              <Save size={13} />{busy === "保存模型" ? "保存中…" : "保存"}
            </button>
            <button className="btn" disabled={!d.llm_api_key.configured || busy === "测试模型"}
                    onClick={() => test("test-llm", "测试模型")}>
              <PlugZap size={13} />测试连接
            </button>
          </div>
        </Panel>
      </div>

      {msg && <div className="panel" style={{ marginTop: 12 }}><div className="note" style={{ margin: 0 }}>{msg}</div></div>}

      <Panel title="两种配置途径" extra="优先级说明">
        <div className="note" style={{ margin: 0 }}>
          <b>.env / 环境变量</b>：部署级配置，优先级最高。复制 <code>.env.example</code> 为 <code>.env</code> 编辑，
          适合 Docker、服务器、写进 crontab 的场景。<br />
          <b>设置页</b>：保存到 <code>{d.dirs.config}/settings.local.json</code>，改完即生效（无需重启）。
          同一项若两边都配了，以 .env 为准（上方「来源」徽章标明当前生效的是哪边）。<br />
          两个目录位置也可用 <code>STOCK_DATA_DIR / STOCK_CONFIG_DIR / STOCK_REPORTS_DIR</code> 环境变量重定向，详见 docs/configuration.md。
        </div>
      </Panel>
    </>
  );
}
