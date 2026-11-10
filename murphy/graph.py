"""The knowledge graph. Code builds the rows; Neo4j stores them.

Node types: Airport, Community, Carrier, Flight, Metar, Taf.
Relationships:
  (Flight)-[:DEPARTS_FROM|ARRIVES_AT]->(Airport)
  (Flight)-[:MARKETED_BY|OPERATED_BY]->(Carrier)
  (Flight)-[:ALTERNATIVE_FOR {lead_h, ...}]->(Flight)   the alternative points at the trip's flight
  (Airport)-[:IN_COMMUNITY]->(Community)
  (Airport)-[:ROUTE_TO {flights}]->(Airport)            2023-24 departures
  (Airport)-[:CO_DISRUPTS {weight}]-(Airport)           the GCN's co-disruption links
  (Metar|Taf)-[:REPORTED_AT]->(Airport)

Anything learned after a flight leaves is prefixed `outcome_` and has `outcome_known_at`.
A Metar or Taf has `known_at`. Aircraft are left out: tail numbers are hindsight.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from murphy.forecast.score import PCOLS
from murphy.recommend import p_make

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ("2025-01-04", "2025-01-12")  # inclusive; two storms and calm days
DEADLINE_MIN = 60  # the deadline the replay reports at: 60 minutes after scheduled arrival
LEADS = (24, 6, 2)
BATCH = 5000

CONSTRAINTS = {"Airport": "iata", "Community": "id", "Carrier": "code", "Flight": "id",
               "Metar": "id", "Taf": "id"}


def connect():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / ".env")
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    return driver, os.environ.get("NEO4J_DATABASE") or None


def records(df: pd.DataFrame) -> list[dict]:
    """Rows for Neo4j: blanks become None, times become UTC datetimes, arrays become lists."""
    out = []
    for r in df.to_dict("records"):
        row = {}
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                v = None if pd.isna(v) else (v.tz_localize("UTC") if v.tzinfo is None else v).to_pydatetime()
            elif isinstance(v, np.ndarray):
                v = [float(x) for x in v]
            elif isinstance(v, (np.floating, float)):
                v = None if math.isnan(v) else float(v)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            elif v is pd.NaT or v is None or (not isinstance(v, (str, list)) and pd.isna(v)):
                v = None
            row[k] = v
        out.append(row)
    return out


def forecast_columns(trips: pd.DataFrame) -> pd.DataFrame:
    """One row per flight: each lead's 16 class probabilities, as_of, and chance of missing a
    deadline DEADLINE_MIN after scheduled arrival (murphy.recommend.p_make, the app's rule)."""
    out = trips[["flight_id"]].drop_duplicates().set_index("flight_id")
    for lead in LEADS:
        t = trips[trips.lead_h == lead].set_index("flight_id")
        p = t[PCOLS].fillna(0).to_numpy(dtype=float)
        out[f"as_of_{lead}h"] = t.as_of_utc
        out[f"p_{lead}h"] = pd.Series(list(p), index=t.index)
        out[f"p_miss_{lead}h"] = pd.Series(1 - p_make(p, np.full(len(t), DEADLINE_MIN)), index=t.index)
    return out.reset_index().rename(columns={"flight_id": "id"})


def alternative_rows(alts: pd.DataFrame) -> pd.DataFrame:
    """One row per (trip flight, alternative, lead). The alternative's chance of missing the same
    deadline: its slack is the trip's plus how much earlier it is scheduled to arrive."""
    p = alts[PCOLS].fillna(0).to_numpy(dtype=float)
    slack = (alts.arr_ahead_min + DEADLINE_MIN).to_numpy(dtype=float)
    return pd.DataFrame({"trip": alts.flight_id, "alt": alts.alt, "lead_h": alts.lead_h,
                         "alt_lead_h": alts.alt_lead, "arr_ahead_min": alts.arr_ahead_min,
                         "p_miss": 1 - p_make(p, slack)})


def flight_facts(flights: pd.DataFrame) -> pd.DataFrame:
    """Schedule from BTS, plus outcomes that are labeled as known only after the flight."""
    f = flights
    return pd.DataFrame({
        "id": f.flight_id, "date": f.flight_date.astype(str), "origin": f.origin, "dest": f.dest,
        "marketing": f.marketing_carrier, "operating": f.operating_carrier,
        "flight_number": f.flight_number, "crs_dep_local": f.crs_dep_local,
        "crs_arr_local": f.crs_arr_local, "sched_dep_utc": f.sched_dep_utc, "sched_arr_utc": f.sched_arr_utc,
        "distance_mi": f.distance_mi, "source_file": f.source_file,
        "outcome_arr_delay_min": f.arr_delay_min, "outcome_cancelled": f.cancelled,
        "outcome_cancellation_code": f.cancellation_code, "outcome_diverted": f.diverted,
        "outcome_weather_delay_min": f.weather_delay_min, "outcome_nas_delay_min": f.nas_delay_min,
        "outcome_carrier_delay_min": f.carrier_delay_min,
        "outcome_late_aircraft_delay_min": f.late_aircraft_delay_min,
        # Arrival is when the outcome is known; a cancellation, at departure.
        "outcome_known_at": f.arr_known_at.fillna(f.dep_known_at).fillna(f.sched_dep_utc),
    })


def run(session, cypher: str, rows: list[dict] | None = None, batch=BATCH) -> int:
    if rows is None:
        session.run(cypher).consume()
        return 0
    for i in range(0, len(rows), batch):
        session.run(cypher, rows=rows[i:i + batch]).consume()
    return len(rows)


def clear(session) -> None:
    """Delete every node, constraint and index."""
    session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
    for r in session.run("SHOW CONSTRAINTS YIELD name").data():
        session.run(f"DROP CONSTRAINT `{r['name']}`").consume()
    for r in session.run("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP'").data():
        session.run(f"DROP INDEX `{r['name']}`").consume()


def schema(session) -> None:
    for label, key in CONSTRAINTS.items():
        session.run(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE").consume()
    session.run("CREATE INDEX metar_time IF NOT EXISTS FOR (n:Metar) ON (n.known_at)").consume()
    session.run("CREATE INDEX taf_time IF NOT EXISTS FOR (n:Taf) ON (n.known_at)").consume()
    session.run("CREATE INDEX flight_date IF NOT EXISTS FOR (n:Flight) ON (n.date)").consume()


# The fixed queries. Each reads only what was known at the forecast's as_of,
# except `outcome`, which is labeled hindsight. Parameters: flight id and lead_h (24, 6 or 2).
AS_OF = "f['as_of_' + toString($lead_h) + 'h']"
QUERIES = {
    "flight": ("Basic: one flight's forecast at each lead",
        """MATCH (f:Flight {id: $flight_id})-[:MARKETED_BY]->(m:Carrier), (f)-[:OPERATED_BY]->(op:Carrier),
                 (f)-[:DEPARTS_FROM]->(o:Airport), (f)-[:ARRIVES_AT]->(d:Airport)
           RETURN f.id AS flight, m.code AS marketed_by, op.code AS operated_by, o.iata AS origin, d.iata AS dest,
                  f.crs_dep_local AS dep_local, f.crs_arr_local AS arr_local,
                  [l IN [24, 6, 2] | {lead_h: l, as_of: f['as_of_' + toString(l) + 'h'],
                                      p_miss: f['p_miss_' + toString(l) + 'h']}] AS forecasts"""),
    "alternatives": ("Traversal: alternatives offered at a lead, least likely to miss first",
        """MATCH (a:Flight)-[r:ALTERNATIVE_FOR {lead_h: $lead_h}]->(f:Flight {id: $flight_id}), (a)-[:MARKETED_BY]->(c:Carrier)
           WHERE r.alt_lead_h - $lead_h >= -$max_earlier_h
           RETURN a.id AS alternative, c.code AS carrier, a.crs_dep_local AS dep_local, r.arr_ahead_min AS arr_ahead_min,
                  r.p_miss AS alt_p_miss, f['p_miss_' + toString($lead_h) + 'h'] AS trip_p_miss
           ORDER BY r.p_miss LIMIT 5"""),
    "why": ("Multi-hop: weather known at as_of at the origin and the airports that usually go bad with it",
        f"""MATCH (f:Flight {{id: $flight_id}})-[:DEPARTS_FROM]->(o:Airport)-[:IN_COMMUNITY]->(c:Community)
           WITH f, o, c, {AS_OF} AS as_of
           OPTIONAL MATCH (o)-[k:CO_DISRUPTS]-(n:Airport)
           WITH f, o, c, as_of, collect({{airport: n.iata, weight: k.weight}}) AS neighbors
           UNWIND [{{airport: o.iata, weight: null}}] + neighbors AS x
           OPTIONAL MATCH (m:Metar)-[:REPORTED_AT]->(:Airport {{iata: x.airport}}) WHERE m.known_at <= as_of
           WITH f, o, c, as_of, x, m ORDER BY m.known_at DESC
           WITH f, o, c, as_of, x, head(collect(m)) AS m
           RETURN f.id AS flight, as_of, o.iata AS origin, c.id AS community, c.busiest[0..4] AS community_busiest,
                  x.airport AS airport, x.weight AS co_disruption_weight,
                  m.id AS metar_id, m.known_at AS metar_known_at, m.metar AS metar, m.tmpf AS temp_f, m.sknt AS wind_kt,
                  m.gust AS gust_kt, m.vsby AS visibility_mi, m.wxcodes AS weather_codes
           ORDER BY x.weight IS NULL DESC, x.weight DESC"""),
    "taf": ("Multi-hop: the latest TAF for the origin known at as_of",
        f"""MATCH (f:Flight {{id: $flight_id}})-[:DEPARTS_FROM]->(o:Airport)<-[:REPORTED_AT]-(t:Taf)
           WHERE t.known_at <= {AS_OF}
           RETURN f.id AS flight, {AS_OF} AS as_of, t.id AS taf_id, t.known_at AS known_at, t.raw AS raw
           ORDER BY t.known_at DESC LIMIT 1"""),
    "community_day": ("Analytics: forecast risk by community on the flight's day, at the same lead",
        """MATCH (x:Flight {id: $flight_id})
           OPTIONAL MATCH (x)-[:DEPARTS_FROM]->(:Airport)-[:IN_COMMUNITY]->(oc:Community)
           OPTIONAL MATCH (x)-[:ARRIVES_AT]->(:Airport)-[:IN_COMMUNITY]->(dc:Community)
           WITH x, oc.id AS o, dc.id AS d
           MATCH (f:Flight {in_replay: true, date: x.date})-[:DEPARTS_FROM]->(:Airport)-[:IN_COMMUNITY]->(c:Community)
           WITH c, o, d, f['p_miss_' + toString($lead_h) + 'h'] AS p
           RETURN c.id AS community, c.busiest[0..4] AS busiest,
                  CASE WHEN c.id = o AND c.id = d THEN 'trip origin and destination' WHEN c.id = o THEN 'trip origin'
                       WHEN c.id = d THEN 'trip destination' END AS trip_role, count(*) AS flights,
                  avg(p) AS mean_p_miss, sum(CASE WHEN p >= 0.5 THEN 1 ELSE 0 END) AS flights_p_miss_50
           ORDER BY community"""),
    "outcome": ("Hindsight: what happened, known only after the flight",
        """MATCH (f:Flight {id: $flight_id})
           RETURN f.id AS flight, f.outcome_known_at AS known_at, f.outcome_cancelled AS cancelled,
                  f.outcome_diverted AS diverted, f.outcome_arr_delay_min AS arr_delay_min,
                  f.outcome_weather_delay_min AS weather_delay_min, f.outcome_nas_delay_min AS nas_delay_min,
                  f.source_file AS source_file"""),
}


def ask(session, name: str, flight_id: str, lead_h: int, max_earlier_h: float = 24) -> list[dict]:
    """max_earlier_h: how much earlier than the trip an alternative may leave, the app's setting.
    24 means no limit: an alternative leaves at least 1 h after the forecast."""
    if name not in QUERIES:
        raise ValueError(f"unknown query {name!r}")
    if lead_h not in LEADS:
        raise ValueError(f"lead_h must be one of {LEADS}")
    if not 0 <= max_earlier_h <= 24:
        raise ValueError("max_earlier_h must be between 0 and 24")
    return session.run(QUERIES[name][1], flight_id=flight_id, lead_h=lead_h, max_earlier_h=float(max_earlier_h)).data()
