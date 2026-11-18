const BASE = import.meta.env.DEV ? "http://localhost:8602" : "";

export async function api(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`.slice(0, 200));
  return r.json();
}
