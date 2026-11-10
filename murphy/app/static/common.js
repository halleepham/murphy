// Small helpers shared by every page.
const $ = id => document.getElementById(id);
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
const pct = x => (100 * x).toFixed(0) + "%";
const time = t => t ? t.slice(0, 2) + ":" + t.slice(2) : "?";

async function get(url) {
  const r = await fetch(url);
  const body = await r.json();
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Request failed");
  return body;
}

// A white section card in a tab (styling pass): an optional heading, then the content.
function card(title, ...kids) {
  const s = el("section", null, "card sec");
  if (title) s.append(typeof title === "string" ? el("h3", title) : title);
  s.append(...kids.filter(x => x != null));
  return s;
}

// A folded section: "How to read this" for plain explanations, "Technical details" for sources and
// model detail. Items are strings (one paragraph each) or elements.
function fold(summary, ...items) {
  const d = el("details", null, "fold");
  d.append(el("summary", summary, "small"), ...items.filter(x => x != null && x !== "").map(x => typeof x === "string" ? el("p", x, "muted small") : x));
  return d;
}
