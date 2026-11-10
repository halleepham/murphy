// The map tab. Every number comes from /api/map and /api/trip; this file only draws.
// Layers: route (always), regions, linked airports, weather; a slider over
// the 24 hours up to the forecast hour. Forecasts only, never what happened.

const US = fetch("https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json").then(r => r.json());
const mapCache = {};  // by lead time: the layers do not depend on the deadline
const mapView = { regions: true, links: true, weather: true, hour: null, playing: null, zoom: null };
const VERDICT_COLOR = { keep: "--good", too_uncertain: "--warn" };  // anything else is --bad

async function map(box) {
  box.replaceChildren(card(null, el("p", "Drawing the map...", "loading")));
  const lead = $("lead").value;
  let d, us;
  try { [d, us] = await Promise.all([mapCache[lead] ||= get("/api/map?" + new URLSearchParams({ flight_id: flightId, lead_h: lead })), US]); }
  catch (e) { delete mapCache[lead]; box.replaceChildren(card(null, el("p", e.message || "The map could not load.", "error"))); return; }
  if (tab !== "map") return;
  const last = d.hours_utc.length - 1;
  if (mapView.hour == null || mapView.hour > last) mapView.hour = last;

  const toggles = el("div", null, "controls map-toggles");
  [["regions", "Delay risk by area"], ["links", "Linked airports"], ["weather", "Weather"]].forEach(([k, label]) => {
    const l = el("label", null, "chip"), c = el("input"); c.type = "checkbox"; c.checked = mapView[k];
    c.onchange = () => { mapView[k] = c.checked; draw(); renderWeather(); };
    l.append(c, " " + label); toggles.append(l);
  });
  const slider = el("input"); Object.assign(slider, { type: "range", min: 0, max: last, step: 1, value: mapView.hour });
  const hourLabel = el("span", null, "small"), play = el("button", "Play", "btn small");
  slider.oninput = () => { mapView.hour = +slider.value; draw(); };
  play.type = "button";
  play.onclick = () => {
    if (mapView.playing) { clearInterval(mapView.playing); mapView.playing = null; play.textContent = "Play"; return; }
    if (mapView.hour === last) mapView.hour = 0;
    play.textContent = "Pause";
    mapView.playing = setInterval(() => {
      if (tab !== "map" || mapView.hour >= last) { clearInterval(mapView.playing); mapView.playing = null; play.textContent = "Play"; return; }
      mapView.hour++; slider.value = mapView.hour; draw();
    }, 500);
  };
  const timeRow = el("div", null, "map-slider"); timeRow.append(play, slider, hourLabel);

  const svgBox = el("div", null, "map-box"), side = el("div", null, "map-side");
  const grid = el("div", null, "map-grid"); grid.append(svgBox, side);
  // Two sections: the map with its controls and notes, then the weather at each end.
  const wxBox = el("section", null, "card sec"), notes = el("div");
  box.replaceChildren(card("Around your route", toggles, timeRow, grid, notes), wxBox);

  // --- base map --------------------------------------------------------------------------------
  const W = 900, H = 540, proj = d3.geoAlbersUsa().scale(1150).translate([W / 2, H / 2]), path = d3.geoPath(proj);
  const svg = d3.select(svgBox).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img")
    .attr("aria-label", `Map of ${trip.flight.origin} to ${trip.flight.dest} with regions and linked airports`);
  const world = svg.append("g");  // everything that zooms
  world.append("path").datum(topojson.feature(us, us.objects.nation)).attr("class", "nation").attr("d", path);
  world.append("path").datum(topojson.mesh(us, us.objects.states, (a, b) => a !== b)).attr("class", "state-lines").attr("d", path);
  const xy = a => a && proj([a.lon, a.lat]);
  const placed = d.airports.filter(a => xy(a)), offMap = d.airports.length - placed.length;
  const byCode = Object.fromEntries(placed.map(a => [a.airport, a]));
  const gLinks = world.append("g"), gDots = world.append("g"), gRoute = world.append("g"), gLabels = world.append("g");

  // Zoom: buttons, pinch, or Ctrl/Cmd + scroll (plain scrolling still scrolls the page). Drag to pan.
  let k = mapView.zoom ? mapView.zoom.k : 1;
  const zoom = d3.zoom().scaleExtent([1, 12]).translateExtent([[0, 0], [W, H]])
    .filter(e => (e.type !== "wheel" || e.ctrlKey || e.metaKey) && !e.button)
    .on("zoom", e => { world.attr("transform", e.transform); k = e.transform.k; mapView.zoom = e.transform; draw(); });
  svg.call(zoom);
  const zoomBar = el("div", null, "zoom-bar");
  [["+", "Zoom in", () => svg.call(zoom.scaleBy, 2)],
   ["−", "Zoom out", () => svg.call(zoom.scaleBy, 0.5)],
   ["Reset", "Show the whole map", () => svg.call(zoom.transform, d3.zoomIdentity)]].forEach(([t, label, f]) => {
    const b = el("button", t, "btn small"); b.type = "button"; b.title = label; b.setAttribute("aria-label", label); b.onclick = f; zoomBar.append(b);
  });
  svgBox.append(zoomBar, el("p", "Zoom with the buttons, a pinch, or Ctrl/⌘ + scroll. Drag to move around.", "muted small"));

  const risks = d.regions.flatMap(r => r.risk).filter(v => v != null);
  const color = d3.scaleSequential(d3.interpolateOrRd).domain([0, d3.max(risks) || 1]);
  const region = Object.fromEntries(d.regions.map(r => [r.community, r]));
  const at = (arr, h) => arr ? arr[h] : null;
  const o = d.route.origin, dst = d.route.dest, vcol = css(VERDICT_COLOR[trip.answer.verdict] || "--bad");
  const ends = [o, dst].filter(a => xy(a)), wxAt = new Set([d.weather.origin, d.weather.dest].filter(w => !w.limited).map(w => w.airport));

  function draw() {
    const h = mapView.hour;
    hourLabel.textContent = `${d.hours_local[h]} at ${trip.flight.origin}` + (h === last ? ", the hour your forecast was made" : `, ${last - h} h earlier`);

    gDots.selectAll("circle").data(placed, a => a.airport).join("circle")
      .attr("cx", a => xy(a)[0]).attr("cy", a => xy(a)[1]).attr("r", a => (a.big ? 4 : 2.5) / Math.sqrt(k))
      .attr("fill", a => { const v = at(region[a.community]?.risk, h); return mapView.regions && v != null ? color(v) : css("--line"); })
      .attr("stroke", css("--surface")).attr("stroke-width", .6 / k)
      .selectAll("title").data(a => [a]).join("title")
      .text(a => { const g = at(d.gcn[a.airport], h), r = at(region[a.community]?.risk, h);
        return `${a.airport} ${a.name}\n${region[a.community]?.name}: ${r == null ? "no estimate" : pct(r)} of departures expected late, next 3 h` +
               (g == null ? "" : `\nThis airport: ${pct(g)} expected late, next 3 h`); });

    const top = new Set(d.links.top_by_gcn[h] || []), op = xy(o);
    const nb = mapView.links && op ? d.links.neighbors.filter(n => byCode[n.airport]) : [];
    const pos = n => xy(byCode[n.airport]);
    gLinks.selectAll("line").data(nb, n => n.airport).join("line")
      .attr("x1", op?.[0]).attr("y1", op?.[1]).attr("x2", n => pos(n)[0]).attr("y2", n => pos(n)[1])
      .attr("class", n => top.has(n.airport) ? "link top" : "link").attr("stroke-width", n => (0.5 + 2 * n.weight) / k);
    gLabels.selectAll("text").data(nb.filter(n => top.has(n.airport)), n => n.airport).join("text")
      .attr("x", n => pos(n)[0] + 6 / k).attr("y", n => pos(n)[1] + 4 / k).attr("class", "link-label")
      .style("font-size", `${12 / k}px`).style("stroke-width", `${3 / k}px`)
      .text(n => `${n.airport} ${pct(at(n.gcn, h))}`);

    if (op && xy(dst)) gRoute.selectAll("path").data([0]).join("path")
      .attr("d", path({ type: "LineString", coordinates: [[o.lon, o.lat], [dst.lon, dst.lat]] }))
      .attr("class", "route").attr("stroke", vcol).attr("stroke-width", 3.5 / k);
    gRoute.selectAll("circle.wx").data(mapView.weather ? ends.filter(a => wxAt.has(a.airport)) : [], a => a.airport).join("circle")
      .attr("class", "wx").attr("cx", a => xy(a)[0]).attr("cy", a => xy(a)[1]).attr("r", 12 / k).attr("stroke-width", 2 / k);
    gRoute.selectAll("circle.end").data(ends, a => a.airport).join("circle")
      .attr("class", "end").attr("cx", a => xy(a)[0]).attr("cy", a => xy(a)[1]).attr("r", 7 / k).attr("fill", vcol).attr("stroke-width", 2 / k);
    gRoute.selectAll("text").data(ends, a => a.airport).join("text")
      .attr("class", "end-label").attr("x", a => xy(a)[0] + 10 / k).attr("y", a => xy(a)[1] - 8 / k)
      .style("font-size", `${14 / k}px`).style("stroke-width", `${3 / k}px`).text(a => a.airport);

    renderSide(h);
  }

  function renderSide(h) {
    const out = [];
    if (mapView.regions) {
      out.push(el("h3", "Delay risk by area"),
        el("p", "Share of departures expected to leave late in the next 3 hours. Darker means more delays.", "muted small"));
      const t = el("table"), hr = el("tr"); ["Area", "Expected late"].forEach(x => hr.append(el("th", x))); t.append(hr);
      [...d.regions].sort((a, b) => (at(b.risk, h) ?? -1) - (at(a.risk, h) ?? -1)).forEach(r => {
        const tr = el("tr"), sw = el("span", null, "swatch"), v = at(r.risk, h);
        sw.style.background = v == null ? css("--line") : color(v);
        const c0 = el("td"); c0.append(sw, ` ${r.name}`);
        tr.append(c0, el("td", v == null ? "no estimate" : pct(v))); t.append(tr);
      });
      out.push(t, fold("How to read this",
        "An area is a group of airports whose delays have tended to happen at the same time (2023–24), so it is not always a neat patch of the map. Each area is named after its busiest airports.",
        "Every dot is an airport, colored by its area. The slider shows the day leading up to your forecast, never later."),
        fold("Technical details", `Areas are co-disruption communities. Each value averages a GCN's predicted late share over the area's big airports: ${d.regions.map(r => `${r.name} ${r.airports_scored} of ${r.airports}`).join("; ")}.`));
    }
    if (mapView.links) {
      out.push(el("h3", `Airports linked to ${trip.flight.origin}`));
      if (d.links.limited) out.push(el("p", d.links.reason, "muted small"));
      else {
        const top = d.links.top_by_gcn[h] || [], gcn = Object.fromEntries(d.links.neighbors.map(n => [n.airport, at(n.gcn, h)]));
        out.push(el("p", `Airports whose delays tend to spread to and from ${trip.flight.origin}. A thicker line means they have had delays at the same time more often.`, "muted small"));
        out.push(el("p", top.length ? `Most likely to have delays next: ${top.map(a => `${a} (${gcn[a] == null ? "no estimate" : pct(gcn[a]) + " expected late"})`).join(", ")}.` : "No estimate for this hour.", "small"));
        out.push(fold("Why these three?", d.uncertainty.why_these),
          fold("Technical details", `Top 3 by GCN at this hour: ${top.join(", ")}. Top 3 by link weight, the simpler baseline: ${d.links.top_by_weight.join(", ")}. ${d.uncertainty.gcn_order}`,
            `Link weights (2023–24): ${d.links.neighbors.slice(0, 8).map(n => `${n.airport} ${n.weight.toFixed(2)}`).join(", ")}${d.links.neighbors.length > 8 ? "…" : ""}.`));
      }
    }
    side.replaceChildren(...out);
  }

  // Weather in plain words (murphy/wxtext.py): one line per airport, the rest folded.
  function renderWeather() {
    if (!mapView.weather) { wxBox.replaceChildren(); wxBox.hidden = true; return; }
    wxBox.hidden = false;
    const cols = el("div", null, "wx-grid");
    [[d.weather.origin, "departure"], [d.weather.dest, "arrival"]].forEach(([w, role]) => {
      const c = el("div", null, "wx-card");
      c.append(el("h3", `${w.airport}, at your ${role}${w.flight_time ? ` (${w.flight_time})` : ""}`));
      if (w.limited) { c.append(el("p", w.reason, "muted small")); cols.append(c); return; }
      const f = w.notable_at_flight;
      c.append(el("p", f == null ? "The latest forecast doesn't reach that time yet."
        : f.length ? `Expected around then: ${f.join(", ")}.` : "Nothing notable expected around then.",
        f && f.length ? "wx-flag" : "small"));
      if (w.metar) { const p = el("p", null, "small"); p.append(el("span", `Right now (${w.metar.plain.when.replace("Observed ", "")}): `, "strong"), w.metar.plain.short); c.append(p); }
      else c.append(el("p", "No recent weather report.", "muted small"));
      if (w.taf) {
        const ul = el("ul", null, "wx-list");
        w.taf.plain.forEach(x => { const li = el("li", null, x.near ? "near" : ""); li.append(el("span", x.when + ": ", "strong"), x.text); ul.append(li); });
        c.append(fold("Full forecast, hour by hour",
          `Each row is a time window from the airport's official forecast, issued ${w.taf.issued_local}. The highlighted row covers your flight.`, ul));
      }
      c.append(fold("Technical details", ...(w.metar ? [`Observation (METAR), known ${w.metar.known_at} UTC: ${w.metar.text}`, `In full: ${w.metar.plain.text}`] : []),
        ...(w.taf ? [`Forecast (TAF), issued ${w.taf.issued_utc} UTC: ${w.taf.lines.join(" · ")}`] : []),
        `"Expected around then" covers forecast windows within an hour of your flight and uses Murphy's simple thresholds, not an aviation rule: ${w.thresholds}.`, w.source));
      cols.append(c);
    });
    wxBox.replaceChildren(el("h3", "Weather"), cols);
  }

  renderWeather();
  const lim = [...d.uncertainty.limits]; if (offMap) lim.push(`Airports outside the map's projection, not drawn: ${offMap}.`);
  const p = d.provenance;
  notes.replaceChildren(fold("How to read this map", "The line is your route, colored by Murphy's advice. Dots are airports. Use the switches to show or hide each layer, and the slider to see the day leading up to your forecast.", ...lim),
    fold("Technical details", `GCN: ${p.gcn}. Areas: ${p.regions}. Links: ${p.links}.`,
      `Forecast as of ${p.forecast_as_of_utc} UTC; the slider's last hour is ${p.forecast_hour_utc} UTC.`));
  if (mapView.zoom) svg.call(zoom.transform, mapView.zoom); else draw();
}

// --- Alternatives as a timeline: same route, so a bar per flight in time. Shown in Options.
function renderTimeline(box, tz) {
  const f = trip.flight, rows = [{ ...f, original: true }, ...trip.alternatives.filter(a => a.schedule_ok)];
  const bad = trip.alternatives.length - rows.length + 1;
  const out = [el("h3", "Your flight and the alternatives, by scheduled time")];
  if (rows.length === 1) out.push(el("p", trip.answer.verdict === "keep" ? "No change suggested, so no alternatives are listed." : "No alternative within your limits.", "muted small"));
  const t = s => new Date(s.replace(" ", "T") + "Z");
  const deadline = new Date(+t(f.arrives_utc) + 60000 * +$("deadline").value);
  const W = 900, rowH = 22, top = 24, H = top + rowH * rows.length + 8, left = 110, right = 40;
  const x = d3.scaleUtc().domain([d3.min(rows, r => t(r.departs_utc)), d3.max([deadline, ...rows.map(r => t(r.arrives_utc))])]).nice()
    .range([left, W - right]);
  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("class", "timeline");
  svg.append("g").attr("transform", `translate(0,${top - 4})`).call(d3.axisTop(x).ticks(8).tickFormat(tz ? (v => new Intl.DateTimeFormat("en-US", { timeZone: tz, hour: "numeric", minute: "2-digit" }).format(v).toLowerCase()) : d3.utcFormat("%H:%M"))).attr("class", "axis");
  svg.append("line").attr("x1", x(deadline)).attr("x2", x(deadline)).attr("y1", top - 4).attr("y2", H).attr("class", "deadline");
  const nearEnd = x(deadline) > W - right - 90;  // keep the label inside the box
  svg.append("text").attr("x", x(deadline) + (nearEnd ? -4 : 4)).attr("y", H - 4).attr("text-anchor", nearEnd ? "end" : "start")
    .attr("class", "small-label").text("your deadline");
  const detail = el("p", "Click a bar for its details.", "muted small");
  rows.forEach((r, i) => {
    const y = top + i * rowH, g = svg.append("g").attr("class", "bar-row").style("cursor", "pointer");
    g.append("text").attr("x", left - 8).attr("y", y + 14).attr("text-anchor", "end").attr("class", "small-label").text(r.original ? `${r.flight} (yours)` : r.flight);
    g.append("rect").attr("x", x(t(r.departs_utc))).attr("y", y + 3).attr("height", rowH - 6).attr("rx", 3)
      .attr("width", Math.max(2, x(t(r.arrives_utc)) - x(t(r.departs_utc))))
      .attr("class", r.original ? "bar mine" : r.best ? "bar best" : r.gain > 0 ? "bar better" : "bar worse");
    g.on("click", () => detail.textContent = r.original
      ? `${r.flight}, yours: departs ${time(r.departs_local)}, arrives ${time(r.arrives_local)} local. Chance of making your deadline ${pct(trip.answer.chance)}.`
      : `${r.flight} on ${r.date}: departs ${time(r.departs_local)}, arrives ${time(r.arrives_local)} local. Chance ${pct(r.chance)} (${r.gain >= 0 ? "+" : ""}${(100 * r.gain).toFixed(0)} pts vs yours)${r.best ? ", the best option" : ""}.`);
  });
  out.push(svg.node(), detail, el("p", (tz ? `Times are local at ${f.origin}.` : "Times are UTC.") + " Green: a better chance than yours (darkest: the best). Grey: not better." +
    (bad ? ` ${bad} alternative${bad > 1 ? "s are" : " is"} not drawn: the source schedule arrives before it leaves.` : ""), "muted small"));
  box.replaceChildren(...out);
}
