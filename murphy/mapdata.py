"""The map's data. Code computes every number here; the page only draws.

Everything is as of the trip's forecast time: GCN scores for the hours up to that hour (never after),
weather known before it. No outcomes. Regions are shaded by the GCN, not by the day's flight
forecasts, because about half of a day's forecasts are made after the trip's.
"""
import functools
from statistics import fmean

import pandas as pd

from murphy.forecast import rank
from murphy.forecast.sample import ROOT

HOURS_BACK = 24  # the slider runs from 24 h before the forecast hour up to it
LINKS = ROOT / "data" / "forecast" / "network" / "co_disrupts.parquet"  # scripts/export_links.py
COMMUNITIES = ROOT / "data" / "forecast" / "network" / "communities.csv"
AIRPORTS = ROOT / "data" / "airports.parquet"
NAMES = ROOT / "data" / "forecast" / "network" / "community_busiest.parquet"  # scripts/export_links.py


@functools.cache
def inputs() -> dict | None:
    """The gitignored files the map reads, or None when any is missing on this machine."""
    s = rank.gcn_scores()
    if s is None or not (LINKS.exists() and AIRPORTS.exists()):
        return None
    cm = pd.read_csv(COMMUNITIES, keep_default_na=False)
    ap = pd.read_parquet(AIRPORTS, columns=["airport", "name", "lat", "lon", "tz"]).drop_duplicates("airport")
    names = pd.read_parquet(NAMES).set_index("community").busiest.map(list).to_dict() if NAMES.exists() else None
    return {"scores": s[0], "big": s[1], "places": cm.merge(ap, on="airport", how="left"),
            "links": pd.read_parquet(LINKS), "weather": weather_at, "names": names}


def weather_at(airport: str, as_of, around=None) -> dict:
    """The latest observation and TAF known before as_of: as written, and in plain words by fixed
    rules (murphy/wxtext.py). `around`: the flight's time at this airport (UTC), so the forecast
    periods near it are marked and flagged. Limited outside the 30 airports."""
    from murphy import wxtext as wt
    from murphy.datalayer import asof
    from murphy.datalayer.weather import AIRPORTS as WX
    if airport not in WX:
        return {"airport": airport, "limited": True,
                "reason": f"Murphy has weather reports for 30 large airports, and {airport} is not one of them."}
    t = pd.Timestamp(as_of).tz_localize("UTC").to_pydatetime()
    obs, taf = asof.asos_as_of(t, airport, lookback_hours=3), asof.taf_as_of(t, airport)
    tz = pd.read_parquet(AIRPORTS, columns=["airport", "tz"]).set_index("airport").tz.get(airport)
    o = wt.obs_text(obs.iloc[-1], tz) if len(obs) else None
    lines = wt.taf_lines(taf, tz, around) if len(taf) else []
    near = [x for x in lines if x["near"]]
    return {"airport": airport, "limited": False, "thresholds": wt.THRESHOLDS,
            "flight_time": None if around is None else wt.local(around, tz),
            "notable_now": wt.notable([o] if o else []),
            "notable_at_flight": wt.notable(near) if near else None,  # None: the TAF does not reach the flight
            "metar": None if o is None else {"plain": {"when": o["when"], "text": o["text"], "short": o["short"]}, "text": obs.metar.iloc[-1],
                                             "obs_utc": str(obs.obs_utc.iloc[-1]), "known_at": str(obs.known_at.iloc[-1])},
            "taf": None if taf.empty else {"issued_utc": str(taf.issued_utc.iloc[0]), "issued_local": wt.local(taf.issued_utc.iloc[0], tz),
                                           "plain": [{k: x[k] for k in ("when", "text", "near")} for x in lines],
                                           "lines": taf.raw.tolist()},
            "source": "IEM ASOS and TAF archive, known before the forecast time; "
                      "plain words by fixed rules in murphy/wxtext.py"}


def build(origin: str, dest: str, as_of, scores, big: dict, places: pd.DataFrame, links: pd.DataFrame, weather,
          dep_utc=None, arr_utc=None, names: dict | None = None) -> dict:
    h = rank.hour_index(as_of)
    rows = list(range(h - HOURS_BACK, h + 1))  # oldest first; the last is the forecast hour
    ok = [0 <= r < len(scores) for r in rows]
    gcn = {a: [float(scores[r, i]) if k else None for r, k in zip(rows, ok)] for a, i in big.items()}

    have = places.dropna(subset=["lat", "lon"])
    regions = []
    for c, g in places.groupby("community"):
        scored = [a for a in g.airport if a in gcn]
        busy = (names or {}).get(int(c))  # a region is named after its busiest airports, not a number
        regions.append({"community": int(c), "name": f"{', '.join(busy[:3])} area" if busy else f"Region {c}",
                        "airports": len(g), "airports_scored": len(scored),
                        "risk": [fmean(v) if (v := [gcn[a][j] for a in scored if gcn[a][j] is not None]) else None
                                 for j in range(len(rows))]})

    if origin in big:
        nb = pd.concat([links[links.a == origin].rename(columns={"b": "airport"}),
                        links[links.b == origin].rename(columns={"a": "airport"})])
        nb = [{"airport": r.airport, "weight": float(r.weight), "gcn": gcn.get(r.airport)} for r in nb.itertuples()]
        nb.sort(key=lambda x: -x["weight"])  # the simpler baseline: link weight, as the `why` query retrieves today
        by_weight = [x["airport"] for x in nb[:rank.K]]
        # GCN order per hour; sorted is stable, so ties and blanks keep link-weight order
        by_gcn = [[x["airport"] for x in sorted(nb, key=lambda x: -x["gcn"][j] if x["gcn"] and x["gcn"][j] is not None
                                                 else float("inf"))[:rank.K]] for j in range(len(rows))]
        lk = {"limited": False, "neighbors": nb, "top_by_weight": by_weight, "top_by_gcn": by_gcn}
    else:
        lk = {"limited": True, "neighbors": [], "top_by_weight": [], "top_by_gcn": [],
              "reason": f"{origin} is a smaller airport. Linked airports are worked out only for the 100 busiest US airports."}

    def place(a):
        p = places[places.airport == a]
        tz = p.tz.iloc[0] if "tz" in p and not p.empty and isinstance(p.tz.iloc[0], str) else None
        return None if p.empty or p.lat.isna().iloc[0] else {"airport": a, "name": p.name.iloc[0], "tz": tz,
                                                           "lat": float(p.lat.iloc[0]), "lon": float(p.lon.iloc[0])}
    from murphy.wxtext import local
    first = pd.Timestamp(rank.START)
    hours = [first + pd.Timedelta(hours=r) for r in rows]
    otz = (place(origin) or {}).get("tz")
    return {"route": {"origin": place(origin), "dest": place(dest)},
            "hours_utc": [str(t) for t in hours],
            "hours_local": [local(t, otz) for t in hours],  # at the origin; says UTC when its zone is unknown
            "airports": have[["airport", "name", "lat", "lon", "community", "big"]].assign(
                big=have.big.astype(str) == "True", community=have.community.astype(int)).to_dict("records"),
            "gcn": gcn, "regions": regions, "links": lk,
            "weather": {"origin": weather(origin, as_of, dep_utc), "dest": weather(dest, as_of, arr_utc)},
            "provenance": {"gcn": f"{rank.SCORES}, predicted late share of departures in the next 3 hours",
                           "regions": "co-disruption communities, 2023-24",
                           "links": "CO_DISRUPTS, 2023-24, exported from Neo4j",
                           "forecast_as_of_utc": str(as_of), "forecast_hour_utc": str(first + pd.Timedelta(hours=h))},
            "uncertainty": {"limits": [
                "The map shows forecasts only, never what happened. The slider stops when your forecast was made.",
                "Area colors average the 100 busiest airports only.",
                f"Airports left off the map for lack of coordinates: {len(places) - len(have)}.",
                "Weather reports cover 30 large airports."],
                # The GCN ranking result, in plain words: hit share 0.450 (GCN) vs 0.299 (link weight) vs 0.294 (random)
                "why_these": "Murphy picks them with a model of how delays spread between airports. In tests on "
                             "2025, 45% of its picks turned out to be among the 3 worst-hit linked airports over the "
                             "next 3 hours. Picking the airports with the thickest lines got 30%, about the same as "
                             "picking at random (29%).",
                "gcn_order": "Hit share of the top 3 by GCN 0.450 vs by link weight 0.299 vs random 0.294, "
                             "on 653,549 airport-hours of 2025."}}
