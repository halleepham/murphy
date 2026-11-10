"""The first screen's API: one precomputed 2025 trip and its advice.

Numbers come from code here. No outcome is ever returned for a trip, so the screen cannot show
hindsight; the track record is the only figure built from outcomes.
"""
from pathlib import Path

import duckdb
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from murphy.cost import estimated_fare
from murphy import policies as pol
from murphy.graph import DEADLINE_MIN as GRAPH_DEADLINE_MIN  # the graph's chances use this deadline
from murphy.graph import WINDOW as GRAPH_WINDOW  # the dates the questions cover
from murphy.forecast.sample import ROOT
from murphy.forecast.score import PCOLS
from murphy.recommend import FIRM_FROM, MAX_EARLIER_H, UNSURE_FROM, p_make, verdict

DB = ROOT / "data" / "forecast" / "app" / "replay.duckdb"
MODEL = "final_network"  # scripts/build_app_data.py reads its replay
LEADS = {24, 6, 2}
CHECKED_TO = 120  # alternatives were saved up to 120 min after the original's arrival; band checked to here
DEADLINES = set(range(0, CHECKED_TO + 1, 15))
CURVE = list(range(0, 181, 15))  # the chart draws the forecast's full range; past CHECKED_TO it is marked unchecked
STATIC = Path(__file__).parent / "static"
LIMITS = [
    "Replay of 2025 only. Forecasts exist for sampled flights at 24, 6 and 2 hours before departure.",
    "Murphy can rarely warn about flights later swapped or cancelled.",
    "Past schedules show that a flight existed, not that it had a seat.",
    "Estimated costs are what people typically paid, not a quote. A last-minute seat may cost more.",
]


def advice(con, flight_id: str, lead_h: int, deadline_min: int, max_earlier_h: float, sort: str = "chance") -> dict | None:
    cols = ", ".join(PCOLS)
    t = con.execute(f"SELECT flight_id, lead_h, flight_date, as_of_utc, carrier, flight_number, origin, dest, crs_dep_local, "
                    f"crs_arr_local, sched_dep_utc, sched_arr_utc, known_cancelled, {cols} FROM trips WHERE flight_id = ? AND lead_h = ?",
                    [flight_id, lead_h]).df()
    if t.empty:
        return None
    t = t.iloc[0]
    probs = t[PCOLS].fillna(0).to_numpy(dtype=float)
    current = float(p_make(probs[None], [deadline_min])[0])
    a = con.execute(f"SELECT alt, alt_lead, arr_ahead_min, carrier, flight_number, flight_date, crs_dep_local, crs_arr_local, sched_dep_utc, sched_arr_utc, {cols} "
                    "FROM alts WHERE flight_id = ? AND lead_h = ? AND arr_ahead_min + ? >= 0 AND alt_lead - lead_h >= ?",
                    [flight_id, lead_h, deadline_min, -max_earlier_h]).df()
    fares = con.execute("SELECT * FROM fares WHERE origin = ? AND dest = ?", [t.origin, t.dest]).df()
    alts = []
    if len(a):
        a["chance"] = p_make(a[PCOLS].fillna(0).to_numpy(dtype=float), (a.arr_ahead_min + deadline_min).to_numpy())
        a = a.sort_values("chance", ascending=False, kind="stable")
        for i, (_, r) in enumerate(a.iterrows()):
            alts.append({"flight": f"{r.carrier} {r.flight_number}", "flight_id": r.alt, "date": str(r.flight_date)[:10],
                         "departs_local": r.crs_dep_local, "arrives_local": r.crs_arr_local,
                         "departs_utc": str(r.sched_dep_utc), "arrives_utc": str(r.sched_arr_utc),  # scheduled, known in advance
                         "schedule_ok": bool(r.sched_arr_utc > r.sched_dep_utc),  # 2 BTS rows arrive before they leave
                         "chance": float(r.chance), "gain": float(r.chance - current), "best": i == 0 and r.chance > current,  # never mark a worse flight as best
                         "same_airline": r.carrier == t.carrier,
                         "cost": estimated_fare(fares, t.origin, t.dest, r.carrier)})
    best = alts[0]["chance"] if alts else None
    # The what-if curve: the same p_make at every deadline, for this flight, its best
    # alternative at the chosen deadline, and the average 2025 flight at this lead time.
    curve = {"deadline_min": CURVE, "checked_to": CHECKED_TO,
             "chance": [float(x) for x in p_make(np.repeat(probs[None], len(CURVE), 0), CURVE)],
             "best_alt": None,
             "typical": [float(x) for x in con.execute("SELECT chance FROM typical WHERE lead_h = ? ORDER BY deadline_min",
                                                       [lead_h]).df().chance]}
    if alts and alts[0]["best"]:
        r = a.iloc[0]
        curve["best_alt"] = {"flight": alts[0]["flight"], "chance": [float(x) for x in p_make(
            np.repeat(r[PCOLS].fillna(0).to_numpy(dtype=float)[None], len(CURVE), 0), np.array(CURVE) + r.arr_ahead_min)]}
    if sort == "price":  # no estimate goes last; ties keep the order by chance
        alts.sort(key=lambda x: (x["cost"]["fare"] is None, x["cost"]["fare"] or 0))
    v = verdict(current, best, bool(t.known_cancelled))
    band = "keep" if 1 - current < UNSURE_FROM else "too_uncertain" if 1 - current < FIRM_FROM else "high"
    track = con.execute("SELECT forecasts, predicted_miss, true_miss FROM track WHERE deadline_min = ? AND band = ?",
                        [deadline_min, band]).df().iloc[0]
    return {"answer": {"verdict": v, "chance": current, "miss": 1 - current},
            "flight": {"flight": f"{t.carrier} {t.flight_number}", "flight_id": t.flight_id, "date": str(t.flight_date)[:10],
                       "origin": t.origin, "dest": t.dest, "departs_local": t.crs_dep_local, "arrives_local": t.crs_arr_local,
                       "departs_utc": str(t.sched_dep_utc), "arrives_utc": str(t.sched_arr_utc)},
            "alternatives": alts,
            "curve": curve,
            "askable": GRAPH_WINDOW[0] <= str(t.flight_date)[:10] <= GRAPH_WINDOW[1],
            "policy_situation": pol.DEFAULT_SITUATION[v],
            "provenance": {"model": MODEL, "forecast_as_of_utc": str(t.as_of_utc), "lead_h": int(t.lead_h),
                           "data": "BTS On-Time Performance 2025; IEM weather"},
            "uncertainty": {"track_record": {"band": band, "forecasts": int(track.forecasts),
                                             "predicted_miss": float(track.predicted_miss), "true_miss": float(track.true_miss)},
                            "limits": LIMITS}}


def neo4j_sessions():
    """Opens the Neo4j driver on first use, so the rest of the app runs without it."""
    state = {}
    def session():
        if "driver" not in state:
            from murphy import graph as kg
            state["driver"], state["db"] = kg.connect()
        return state["driver"].session(database=state["db"])
    return session


def pastdays_index():
    """Loads the similar-past-days index on first use; None if it is not built."""
    state = {}
    def index():
        if "ix" not in state:
            import pandas as pd
            from murphy import pastdays as pdays
            if not (pdays.OUT / "corpus.parquet").exists():
                return None
            state["ix"] = pdays.Index(pd.read_parquet(pdays.OUT / "corpus.parquet"))  # plain numbers: no embeddings needed
        return state["ix"]
    return index


def policy_index():
    """Loads the policy passages on first use; None if they are not built here."""
    state = {}
    def index():
        if "ix" not in state:
            import pandas as pd
            if not (pol.OUT / "passages.parquet").exists():
                return None
            state["ix"] = pol.Index(pd.read_parquet(pol.OUT / "passages.parquet"), np.load(pol.OUT / "embed.npy"))
        return state["ix"]
    return index


def create_app(db: Path = DB, graph_session=None, past_index=None, taf_as_of=None, policy_ix=None, map_inputs=None) -> FastAPI:
    """graph_session: a function returning a Neo4j session (a fake one in tests).
    past_index, taf_as_of: the similar-past-days index and TAF lookup (fakes in tests).
    policy_ix: the policy-passage index (a fake in tests).
    map_inputs: a function returning the map's files, or None if not built (a fake in tests)."""
    app = FastAPI(title="Murphy")
    con = duckdb.connect(str(db), read_only=True)
    graph_session = graph_session or neo4j_sessions()
    past_index = past_index or pastdays_index()
    policy_ix = policy_ix or policy_index()
    if map_inputs is None:
        from murphy import mapdata
        map_inputs = mapdata.inputs

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def landing():
        return FileResponse(STATIC / "index.html")

    @app.get("/trip")
    def trip_page():
        return FileResponse(STATIC / "trip.html")

    @app.get("/about")
    def about_page():
        return FileResponse(STATIC / "about.html")

    @app.get("/api/flights")
    def flights(date: str, carrier: str, number: int):
        rows = con.cursor().execute(
            "SELECT DISTINCT flight_id, origin, dest, crs_dep_local FROM trips WHERE flight_date = TRY_CAST(? AS DATE) "
            "AND carrier = ? AND flight_number = ? ORDER BY crs_dep_local", [date, carrier.strip().upper(), number]).df()
        return rows.to_dict("records")

    @app.get("/api/examples")
    def examples():
        """Picked by forecast alone: flights with a high risk at 2 h and a 60-minute deadline.
        Three from the dates the questions cover, three from the rest of 2025."""
        miss = " + ".join(f"coalesce({k}, 0)" for k in PCOLS[:5])
        rows = con.cursor().execute(
            f"SELECT flight_id, carrier || ' ' || flight_number AS flight, origin, dest, flight_date::VARCHAR AS date, "
            f"flight_date::DATE BETWEEN ?::DATE AND ?::DATE AS askable FROM trips "
            f"WHERE lead_h = 2 AND flight_id IN (SELECT DISTINCT flight_id FROM alts WHERE lead_h = 2) AND 1 - ({miss}) >= ? "
            "QUALIFY row_number() OVER (PARTITION BY askable ORDER BY md5(flight_id)) <= 3 ORDER BY NOT askable, md5(flight_id)",
            [*GRAPH_WINDOW, FIRM_FROM]).df()
        return rows.to_dict("records")

    @app.get("/api/trip")
    def trip(flight_id: str, lead_h: int = 2, deadline_min: int = 60,
             max_earlier_h: float = Query(MAX_EARLIER_H, ge=0, le=24), sort: str = Query("chance", pattern="^(chance|price)$")):
        if lead_h not in LEADS or deadline_min not in DEADLINES:
            raise HTTPException(422, f"lead_h must be one of {sorted(LEADS)}, deadline_min one of {sorted(DEADLINES)}")
        got = advice(con.cursor(), flight_id, lead_h, deadline_min, max_earlier_h, sort)
        if got is None:
            raise HTTPException(404, "No forecast for this flight at this time. Forecasts cover sampled 2025 flights only.")
        return got

    @app.get("/api/pastdays")
    def pastdays(flight_id: str, lead_h: int = 2):
        """Similar earlier days at the origin, with what happened on them. Never the trip's own day."""
        from datetime import datetime
        import pandas as pd
        from murphy import pastdays as pdays
        if lead_h not in LEADS:
            raise HTTPException(422, f"lead_h must be one of {sorted(LEADS)}")
        t = con.cursor().execute("SELECT origin, flight_date, as_of_utc FROM trips WHERE flight_id = ? AND lead_h = ?",
                                 [flight_id, lead_h]).df()
        if t.empty:
            raise HTTPException(404, "No forecast for this flight at this time.")
        ix = past_index()
        if ix is None:
            raise HTTPException(503, "Similar past days are not built on this machine.")
        t = t.iloc[0]
        tz = pd.read_parquet(ROOT / "data" / "airports.parquet").set_index("airport").tz.get(t.origin)
        return pdays.similar(ix, t.origin, pd.Timestamp(t.flight_date).date(), pd.Timestamp(t.as_of_utc).to_pydatetime(),
                             tz, taf_as_of)

    @app.get("/api/policies")
    def policies(flight_id: str, situation: str):
        """Policy passages for the trip's airline and a situation, quoted as written. No LLM."""
        if situation not in pol.SITUATIONS:
            raise HTTPException(422, f"situation must be one of {sorted(pol.SITUATIONS)}")
        t = con.cursor().execute("SELECT carrier FROM trips WHERE flight_id = ? LIMIT 1", [flight_id]).df()
        if t.empty:
            raise HTTPException(404, "No forecast for this flight.")
        ix = policy_ix()
        if ix is None:
            raise HTTPException(503, "Policy passages are not built on this machine.")
        label, question = pol.SITUATIONS[situation]
        r = ix.search(question, t.carrier.iloc[0], "embed")
        keep = ("id", "carrier", "section", "part", "text", "url", "collected", "score")
        return {"answer": {"situation": situation, "label": label, "question": question, "carrier": r["carrier"]},
                "passages": [{k: x[k] for k in keep} | {"sha256": x["sha256"][:12]} for x in r["passages"]],
                "note": r["note"],
                "provenance": {"method": f"{pol.MODEL} cosine, top {pol.K}, over the airline's contract of carriage "
                                         "and 14 CFR parts 250, 259 and 260"},
                "uncertainty": {"retrieval": "Tested on 16 situation and airline pairs: a correct passage was in the "
                                             "top 3 for 13.",
                                "terms": "Quoted from the source as collected; the airline's current terms may differ."}}

    @app.get("/api/map")
    def map_layers(flight_id: str, lead_h: int = 2):
        """The map's layers as of the trip's forecast time: route, regions, linked airports, weather."""
        from murphy import mapdata
        if lead_h not in LEADS:
            raise HTTPException(422, f"lead_h must be one of {sorted(LEADS)}")
        t = con.cursor().execute("SELECT origin, dest, as_of_utc, sched_dep_utc, sched_arr_utc FROM trips WHERE flight_id = ? AND lead_h = ?",
                                 [flight_id, lead_h]).df()
        if t.empty:
            raise HTTPException(404, "No forecast for this flight at this time.")
        m = map_inputs()
        if m is None:
            raise HTTPException(503, "The map's data files are not built on this machine.")
        t = t.iloc[0]
        return mapdata.build(t.origin, t.dest, t.as_of_utc, **m, dep_utc=t.sched_dep_utc, arr_utc=t.sched_arr_utc)

    class Feedback(BaseModel):
        flight_id: str
        question: str
        verdict: str = Field(pattern="^(right|wrong)$")
        note: str = Field("", max_length=300)
        lead_h: int = 2
        deadline_min: int = 60
        max_earlier_h: float = Field(MAX_EARLIER_H, ge=0, le=24)

    @app.post("/api/feedback")
    def feedback(f: Feedback):
        """Human validation: "right" is saved; "wrong" with a note returns one refined answer."""
        from murphy import graphrag as gr
        if f.question not in gr.QUESTIONS or f.lead_h not in LEADS or f.deadline_min != GRAPH_DEADLINE_MIN:
            raise HTTPException(422, "Unknown question or settings.")
        base = {"flight_id": f.flight_id, "question": f.question, "lead_h": f.lead_h, "max_earlier_h": f.max_earlier_h,
                "verdict": f.verdict, "note": f.note.strip()}
        if f.verdict == "right":
            gr.log_feedback(base)
            return {"status": "saved"}
        if not f.note.strip():
            raise HTTPException(422, "Say what looks wrong, so Murphy can check it.")
        try:
            with graph_session() as s:
                out = gr.refine(s, f.question, f.flight_id, f.lead_h, f.max_earlier_h, f.note.strip())
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(503, "The knowledge graph is not reachable right now.")
        gr.log_feedback(base | {"refined": out["status"] == "refined", "status": out["status"],
                                "first": out["first"], "answer": out["explanation"], "checks": out["checks"]})
        return out

    @app.get("/api/ask")
    def ask(flight_id: str, question: str, lead_h: int = 2, deadline_min: int = 60,
            max_earlier_h: float = Query(MAX_EARLIER_H, ge=0, le=24)):
        """A suggested question about the trip. The graph's chances use a 60-minute deadline,
        so other deadlines are refused rather than shown next to a different number."""
        from murphy import graphrag as gr
        if question not in gr.QUESTIONS:
            raise HTTPException(422, f"question must be one of {sorted(gr.QUESTIONS)}")
        if lead_h not in LEADS:
            raise HTTPException(422, f"lead_h must be one of {sorted(LEADS)}")
        if deadline_min != GRAPH_DEADLINE_MIN:
            raise HTTPException(422, f"Questions use a {GRAPH_DEADLINE_MIN}-minute deadline. Set the deadline to {GRAPH_DEADLINE_MIN} to ask.")
        try:
            with graph_session() as s:
                return gr.answer(s, question, flight_id, lead_h, max_earlier_h)
        except Exception as e:  # Neo4j unreachable (e.g. Aura paused): say so, do not guess
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(503, "The knowledge graph is not reachable right now.")

    return app
