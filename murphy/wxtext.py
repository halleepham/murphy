"""Weather reports in plain words, by fixed rules. No LLM: every word comes from a field.

A METAR row (data/weather/asos.parquet) and a TAF's periods (taf.parquet) become short sentences a
traveler can read, with times in the airport's local time. The raw text stays beside them as the
evidence. `notable` flags conditions by simple fixed thresholds, which are shown with the flags.
"""
import math
import re

import pandas as pd

from murphy.pastdays import DESCRIPTOR, INTENSITY, PHENOMENON, _list

SKY = {"FEW": "a few clouds", "SCT": "scattered clouds", "BKN": "broken clouds", "OVC": "overcast",
       "VV": "sky hidden, vertical visibility"}
CLEAR = {"CLR", "SKC", "NSC", "NCD"}
COMPASS = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast", "southeast",
           "south-southeast", "south", "south-southwest", "southwest", "west-southwest", "west",
           "west-northwest", "northwest", "north-northwest"]
OVER_6 = 6.01  # IEM stores "P6SM" (more than 6 statute miles) as 6.01
VRB = re.compile(r"\bVRB(\d{2,3})(?:G(\d{2,3}))?KT\b")  # the TAF parser leaves speed blank for variable wind
# murphy's own simple thresholds for "notable", not an aviation standard (AI choice, untested)
WIND_KT, GUST_KT, VIS_MI, CEILING_FT = 20, 30, 3, 1000
MPH = 1.15078  # knots to miles per hour: reports use knots, travelers read mph
SERIOUS = {"TS": "thunderstorms", "SN": "snow", "FZ": "freezing rain or drizzle", "PL": "ice pellets",
           "FG": "fog", "GR": "hail"}
THRESHOLDS = (f"wind {WIND_KT * MPH:.0f} mph ({WIND_KT} kt) or more, gusts {GUST_KT * MPH:.0f} mph ({GUST_KT} kt) or more, "
              f"visibility under {VIS_MI} miles, "
              f"broken or overcast clouds under {CEILING_FT:,} ft, or " + ", ".join(SERIOUS.values()))
NEAR_H = 1         # a forecast period counts for the flight if it overlaps the flight time +/- this
# Periods have no stored end, so the last runs to issue + span. Measured on 2025 TAFs: periods start up to
# ~30.7 h after issue at 29 of the 30 airports (30-hour TAFs), and at most 24.7 h at DAL (24-hour TAFs).
TAF_SPAN_H, TAF_SPAN_24H = 30, {"DAL"}


def _num(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def local(ts, tz: str | None) -> str:
    """'Thu 3:51 pm' in the airport's time zone (UTC, said so, when the zone is unknown)."""
    t = pd.Timestamp(ts)
    t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
    if tz:
        t = t.tz_convert(tz)
    out = t.strftime("%a %-I:%M ") + t.strftime("%p").lower()
    return out if tz else out + " UTC"


def phrase(code: str) -> str:
    """'-FZRAPL' -> 'light freezing rain and ice pellets'. Unknown letters are kept, never dropped."""
    words, c = [], code
    if c[:1] in INTENSITY:
        words.append(INTENSITY[c[0]]); c = c[1:]
    while c[:2] in DESCRIPTOR:
        words.append(DESCRIPTOR[c[:2]].removesuffix(" with").removesuffix(" of")); c = c[2:]
    things = []
    while c[:2] in PHENOMENON:
        things.append(PHENOMENON[c[:2]]); c = c[2:]
    if things:
        words.append(" and ".join(things))
    if c:
        words.append(c)
    return " ".join(words)


def wind(drct, sknt, gust, raw: str = "") -> str:
    s, g, d = _num(sknt), _num(gust), _num(drct)
    if s is None and (m := VRB.search(raw or "")):
        s, g, d = float(m.group(1)), _num(m.group(2)), None
    if s is None:
        return "wind not reported"
    if s == 0:
        return "calm wind"
    frm = "variable direction" if d is None else "from the " + COMPASS[round(d / 22.5) % 16]
    return f"wind {frm} at {s * MPH:.0f} mph" + (f", gusts to {g * MPH:.0f} mph" if g else "")


def short_sky(codes, heights) -> str:
    """'clear', 'some clouds', 'cloudy' or 'low clouds': the visible line; heights stay in the full text."""
    layers = [(c, _num(h)) for c, h in zip(codes, list(heights) + [None] * len(codes)) if c]
    if any(c in ("BKN", "OVC", "VV") and h is not None and h < CEILING_FT for c, h in layers):
        return "low clouds"
    if any(c in ("BKN", "OVC", "VV") for c, _ in layers):
        return "cloudy"
    if any(c in ("FEW", "SCT") for c, _ in layers):
        return "some clouds"
    return "clear" if layers else ""


def visibility(v) -> str:
    v = _num(v)
    if v is None:
        return "visibility not reported"
    return "visibility over 6 miles" if v >= OVER_6 else f"visibility {v:g} mile{'' if v == 1 else 's'}"


def sky(codes, heights) -> str:
    layers = [(c, _num(h)) for c, h in zip(codes, list(heights) + [None] * len(codes)) if c]
    if not layers:
        return "sky not reported"
    if all(c in CLEAR for c, _ in layers):
        return "clear sky"
    return ", ".join(SKY.get(c, c) + (f" at {h:,.0f} ft" if h is not None else "") for c, h in layers if c not in CLEAR)


def sentence(parts: list[str]) -> str:
    text = "; ".join(p for p in parts if p)
    return text[:1].upper() + text[1:] + "."


def obs_text(r, tz) -> dict:
    """One METAR row -> {when, text} plus the fields `notable` reads."""
    codes = r.wxcodes.split() if isinstance(r.wxcodes, str) else []
    skyc = [c for c in (getattr(r, f"skyc{i}") for i in range(1, 5)) if isinstance(c, str)]
    skyl = [getattr(r, f"skyl{i}") for i in range(1, 5)]
    t, s, g, v = _num(r.tmpf), _num(r.sknt), _num(r.gust), _num(r.vsby)
    brief = [", ".join(phrase(c) for c in codes), f"{t:.0f}°F" if t is not None else "",
             "" if s is None else "calm" if s == 0 else f"wind {s * MPH:.0f} mph" + (f", gusts {g * MPH:.0f} mph" if g else ""),
             short_sky(skyc, skyl), f"visibility {v:g} miles" if v is not None and v < VIS_MI else ""]
    return {"when": f"Observed {local(r.obs_utc, tz)}", "short": ", ".join(x for x in brief if x),
            "text": sentence([", ".join(phrase(c) for c in codes), f"{t:.0f}°F" if t is not None else "",
                              wind(r.drct, r.sknt, r.gust), visibility(r.vsby), sky(skyc, skyl)]),
            "codes": codes, "skyc": skyc, "skyl": skyl, "sknt": _num(r.sknt), "gust": _num(r.gust), "vis": _num(r.vsby)}


def taf_lines(periods: pd.DataFrame, tz, around=None) -> list[dict]:
    """A TAF's periods -> [{when, text, near}] in order. `near`: the period overlaps `around` (UTC)
    +/- NEAR_H. A prevailing period (initial, FM) runs until the next one starts."""
    periods = periods.reset_index(drop=True)
    prev = periods.index[periods.period_type.isin(["initial", "FM"])].tolist()
    span = 24 if periods.airport.iloc[0] in TAF_SPAN_24H else TAF_SPAN_H
    stop = pd.Timestamp(periods.issued_utc.iloc[0]) + pd.Timedelta(hours=span)
    out = []
    for i, p in periods.iterrows():
        start = pd.Timestamp(p.period_start_utc)
        if i in prev:
            nxt = [j for j in prev if j > i]
            end = pd.Timestamp(periods.period_start_utc[nxt[0]]) if nxt else max(stop, start)
        else:
            end = pd.Timestamp(p.period_end_utc)
        s, e = local(start, tz), local(end, tz)
        when = {"TEMPO": f"At times, {s} to {e}", "PROB30": f"30% chance, {s} to {e}"}.get(p.period_type, f"From {s}")
        codes, skyc, skyl = _list(p.presentwx), _list(p.skyc), _list(p.skyl)
        full = i in prev  # TEMPO and PROB30 list only what changes
        parts = [", ".join(phrase(c) for c in codes),
                 wind(p.drct, p.sknt, p.gust, p.raw) if full or _num(p.sknt) is not None else "",
                 visibility(p.visibility) if full or _num(p.visibility) is not None else "",
                 sky(skyc, skyl) if full or skyc else ""]
        near = around is not None and start <= pd.Timestamp(around) + pd.Timedelta(hours=NEAR_H) \
            and end >= pd.Timestamp(around) - pd.Timedelta(hours=NEAR_H)
        out.append({"when": when, "text": sentence(parts), "near": bool(near), "codes": codes, "skyc": skyc,
                    "skyl": skyl, "sknt": _num(p.sknt), "gust": _num(p.gust), "vis": _num(p.visibility)})
    return out


def notable(items: list[dict]) -> list[str]:
    """Conditions past the thresholds above, in any of the items, each named once, with the worst value."""
    found = []
    def add(s):
        if s not in found:
            found.append(s)
    wind_ = [x["sknt"] for x in items if (x["sknt"] or 0) >= WIND_KT]
    gust = [x["gust"] for x in items if (x["gust"] or 0) >= GUST_KT]
    vis = [x["vis"] for x in items if x["vis"] is not None and x["vis"] < VIS_MI]
    ceil = [_num(h) for x in items for c, h in zip(x["skyc"], x["skyl"])
            if c in ("BKN", "OVC", "VV") and _num(h) is not None and _num(h) < CEILING_FT]
    if wind_:
        add(f"strong wind (up to {max(wind_) * MPH:.0f} mph)")
    if gust:
        add(f"strong gusts (up to {max(gust) * MPH:.0f} mph)")
    if vis:
        add(f"poor visibility (down to {min(vis):g} mile{'' if min(vis) == 1 else 's'})")
    if ceil:
        add(f"low clouds (down to {min(ceil):,.0f} ft)")
    for x in items:
        for c in x["codes"]:
            body = c.lstrip("+-")
            for k in (body[i:i + 2] for i in range(0, len(body), 2)):  # two-letter groups, e.g. FZ RA
                if k in SERIOUS:
                    add(SERIOUS[k])
    return found
