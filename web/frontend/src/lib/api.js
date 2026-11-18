const BASE = import.meta.env.DEV ? "http://localhost:8602" : "";

export async function api(path, { method = "GET", body } = {}) {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`.slice(0, 200));
  return r.json();
}
