/* Faun shared frontend helpers — vanilla JS, no framework, no bundler.
   Loaded by index.html, dashboard.html and review.html. */

const $ = (id) => document.getElementById(id);

/* Tiny inline-SVG helper — returns an <svg> node from a 24x24 path string.
   Keeps the UI dependency-free (no icon font, no external SVG file). */
function icon(paths, opts) {
  const o = opts || {};
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", o.stroke || "currentColor");
  svg.setAttribute("stroke-width", o.strokeWidth || "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  (Array.isArray(paths) ? paths : [paths]).forEach((d) => {
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", d);
    svg.appendChild(p);
  });
  return svg;
}

/* Icon path data (Feather-style, 24x24). */
const ICONS = {
  leaf: "M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z M2 21c0-3 1.85-5.36 5.08-6",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
  map: "M9 18l-6 3V6l6-3 6 3 6-3v15l-6 3-6-3Z M9 3v15 M15 6v15",
  tag: "M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82Z M7 7h.01",
};

/* Build the shared app header (brand + top nav). `active` is one of
   "upload" | "dashboard" | "review". Markup-light, dependency-free. */
function renderNav(active) {
  const items = [
    { key: "upload", href: "/", label: "Загрузка", ic: ICONS.upload },
    { key: "dashboard", href: "/dashboard", label: "Карта", ic: ICONS.map },
    { key: "review", href: "/review", label: "Переразметка", ic: ICONS.tag },
  ];

  const header = document.createElement("header");
  header.className = "app-header";

  const bar = document.createElement("div");
  bar.className = "app-header__bar";

  const brand = document.createElement("a");
  brand.className = "brand";
  brand.href = "/";
  const mark = document.createElement("span");
  mark.className = "brand__mark";
  mark.appendChild(icon(ICONS.leaf, { stroke: "#bfe8cb" }));
  brand.appendChild(mark);
  const bt = document.createElement("span");
  bt.textContent = "Faun";
  const bs = document.createElement("small");
  bs.textContent = "Биоакустический мониторинг";
  brand.appendChild(bt);
  brand.appendChild(bs);
  bar.appendChild(brand);

  const nav = document.createElement("nav");
  nav.className = "topnav";
  nav.setAttribute("aria-label", "Основная навигация");
  items.forEach((it) => {
    const a = document.createElement("a");
    a.href = it.href;
    a.appendChild(icon(it.ic));
    const span = document.createElement("span");
    span.textContent = it.label;
    a.appendChild(span);
    if (it.key === active) {
      a.className = "active";
      a.setAttribute("aria-current", "page");
    }
    nav.appendChild(a);
  });
  bar.appendChild(nav);

  header.appendChild(bar);
  document.body.insertBefore(header, document.body.firstChild);
}

/* Append a quiet shared footer. Offline-first, no links out. */
function renderFooter() {
  const f = document.createElement("footer");
  f.className = "app-footer";
  f.textContent =
    "Faun — офлайн batch-pipeline распознавания видов птиц · " +
    "записи акустических ловушек обрабатываются локально.";
  document.body.appendChild(f);
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
