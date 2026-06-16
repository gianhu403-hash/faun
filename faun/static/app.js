/* Faun shared frontend helpers — vanilla JS, no framework, no bundler.
   Loaded by index.html, dashboard.html and review.html. */

const $ = (id) => document.getElementById(id);

/* Build the top nav. `active` is one of "upload" | "dashboard" | "review". */
function renderNav(active) {
  const items = [
    { key: "upload", href: "/", label: "Загрузка" },
    { key: "dashboard", href: "/dashboard", label: "Карта" },
    { key: "review", href: "/review", label: "Переразметка" },
  ];
  const nav = document.createElement("nav");
  nav.className = "topnav";
  items.forEach((it) => {
    const a = document.createElement("a");
    a.href = it.href;
    a.textContent = it.label;
    if (it.key === active) a.className = "active";
    nav.appendChild(a);
  });
  document.body.insertBefore(nav, document.body.firstChild);
}

/* GET JSON with explicit error surfacing (never silent). */
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status + " — " + url);
  return res.json();
}

/* POST JSON; throws on non-2xx so callers can show the error inline. */
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = "HTTP " + res.status;
    try {
      const j = await res.json();
      if (j && j.detail) detail += " — " + j.detail;
    } catch (_e) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

/* Minimal CSV row parser: respects double-quoted fields and "" escaping. */
function parseCsvRow(line) {
  const out = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; } else { q = false; }
      } else { cur += c; }
    } else if (c === '"') { q = true; }
    else if (c === ",") { out.push(cur); cur = ""; }
    else { cur += c; }
  }
  out.push(cur);
  return out;
}

/* Short job id for compact display. */
function shortId(id) {
  return id && id.length > 8 ? id.slice(0, 8) : (id || "");
}

/* Seconds -> "mm:ss" (kept readable for ranger UI). */
function fmtTime(sec) {
  if (typeof sec !== "number" || !isFinite(sec)) return String(sec);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

/* Map a label.source to a colour-coded badge {cls, text}. */
function sourceBadge(source) {
  if (typeof source !== "string") return { cls: "src-model", text: "модель" };
  if (source.startsWith("model:")) return { cls: "src-model", text: "модель" };
  if (source.startsWith("expert:")) return { cls: "src-expert", text: "эксперт" };
  if (source.startsWith("operator:")) return { cls: "src-ranger", text: "лесник" };
  return { cls: "src-model", text: "модель" };
}

/* The current (most recent) label of a detection, or null. */
function currentLabel(det) {
  const labels = det && det.labels;
  if (!Array.isArray(labels) || labels.length === 0) return null;
  return labels[labels.length - 1];
}
