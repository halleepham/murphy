"""Build the Neo4j knowledge graph. 2023-25 only; reads no 2026 file.

Summary layer from 2023-24 (communities, routes, co-disruption links), and a slice of the 2025
replay: its forecasted flights in WINDOW, their alternatives, and weather reports at the 30
airports that have them. Uses the app's replay (data/forecast/app/replay.duckdb, final_network).

  PYTHONPATH=. python scripts/build_graph.py rows         build and count the rows, write nothing
  PYTHONPATH=. python scripts/build_graph.py load --clear clear the instance, then load

--clear deletes everything in the instance first.
"""
import sys, time
import duckdb, numpy as np, pandas as pd
from murphy import graph as kg
from murphy.forecast import network as nw

WINDOW = kg.WINDOW
WEATHER_FROM, WEATHER_TO = "2025-01-03", "2025-01-13"  # 24 h before the first departure
APP = kg.ROOT / "data" / "forecast" / "app" / "replay.duckdb"
ROWS = kg.ROOT / "data" / "forecast" / "rows.parquet"
GCN = kg.ROOT / "data" / "forecast" / "network" / "full" / "m.parquet"
COMMUNITIES = kg.ROOT / "data" / "forecast" / "network" / "communities.csv"
AIRPORTS = kg.ROOT / "data" / "airports.parquet"
WEATHER = kg.ROOT / "data" / "weather"
TRAIN = (2023, 2024)
log = lambda *a: print(*a, flush=True)


def replay():
    con = duckdb.connect(str(APP), read_only=True)
    w = f"flight_date BETWEEN '{WINDOW[0]}' AND '{WINDOW[1]}'"
    trips = con.sql(f"SELECT * FROM trips WHERE {w}").df()
    alts = con.sql(f"SELECT * FROM alts WHERE flight_id IN (SELECT flight_id FROM trips WHERE {w})").df()
    return trips, alts


def gcn_inputs(ids) -> pd.DataFrame:
    """Group m: the GCN's predicted disruption at origin and destination, as the model saw it."""
    r = pd.read_parquet(ROWS, columns=["flight_id", "lead_h"])
    r = r[r.flight_id.isin(set(ids))]
    m = pd.read_parquet(GCN).iloc[r.index]
    m.index = r.index
    d = pd.concat([r, m.drop(columns="rid")], axis=1)
    out = d[["flight_id"]].drop_duplicates().set_index("flight_id")
    for lead in kg.LEADS:
        x = d[d.lead_h == lead].set_index("flight_id")
        out[f"gcn_origin_{lead}h"], out[f"gcn_dest_{lead}h"] = x.gcn_o_next3h, x.gcn_d_next3h
    return out.reset_index().rename(columns={"flight_id": "id"})


def flights(ids) -> pd.DataFrame:
    f = duckdb.sql(f"SELECT * FROM read_parquet('{kg.ROOT}/data/flights/2025_01.parquet')").df()
    return f[f.flight_id.isin(set(ids))]


def summary():
    paths = nw.files(TRAIN)
    big = nw.big_airports(paths)
    co = nw.add_graphs([nw.co_graph(nw.excess_panel(nw.files([y]), big), big) for y in TRAIN], big)
    co = pd.DataFrame([(a, b, d["weight"]) for a, b, d in co.edges(data=True)], columns=["a", "b", "weight"])
    routes = nw.route_counts(paths).rename(columns={"n": "flights"})
    comm = pd.read_csv(COMMUNITIES)
    names = pd.read_parquet(AIRPORTS).drop_duplicates("airport").set_index("airport")
    vol = routes.groupby("a").flights.sum()
    airports = sorted(set(comm.airport) | set(routes.a) | set(routes.b))
    ap = pd.DataFrame({"iata": airports})
    ap["name"] = ap.iata.map(names.name)
    ap["lat"], ap["lon"] = ap.iata.map(names.lat), ap.iata.map(names.lon)
    ap["community"] = ap.iata.map(comm.set_index("airport").community).astype("Int64")
    ap["big"] = ap.iata.map(comm.set_index("airport").big).fillna(False).astype(bool)
    ap["departures_2023_24"] = ap.iata.map(vol).fillna(0).astype(int)
    cm = []
    for c, g in comm.groupby("community"):
        top = g[g.big].airport.map(vol).sort_values(ascending=False).index
        cm.append({"id": int(c), "airports": len(g), "big_airports": int(g.big.sum()),
                   "busiest": [g.airport[i] for i in top[:8]]})
    return ap, pd.DataFrame(cm), routes, co


def weather(airports):
    a = sorted(airports)
    m = duckdb.sql(f"""SELECT airport, obs_utc, known_at, metar, tmpf, sknt, gust, vsby, p01i, wxcodes,
                       skyc1, skyl1, source_file FROM '{WEATHER}/asos.parquet'
                       WHERE obs_utc >= '{WEATHER_FROM}' AND obs_utc < '{WEATHER_TO}'""").df()
    m = m[m.airport.isin(a)]
    m.insert(0, "id", m.airport + ":" + m.obs_utc.dt.strftime("%Y-%m-%dT%H:%M"))
    m = m.drop_duplicates("id")
    t = duckdb.sql(f"""SELECT product_id id, any_value(airport) airport, min(issued_utc) issued_utc,
                       min(known_at) known_at, bool_or(is_amendment) is_amendment,
                       string_agg(raw, ' ' ORDER BY period_start_utc) raw, any_value(source_file) source_file
                       FROM '{WEATHER}/taf.parquet'
                       WHERE issued_utc >= '{WEATHER_FROM}' AND issued_utc < '{WEATHER_TO}' GROUP BY 1""").df()
    return m, t[t.airport.isin(a)]


def build():
    t0 = time.time()
    trips, alts = replay()
    fc = kg.forecast_columns(trips)
    ids = set(trips.flight_id) | set(alts.alt)
    fl = kg.flight_facts(flights(ids))
    missing = ids - set(fl.id)
    log(f"trips {len(fc)}, alternative pairs {len(alts)}, flights {len(fl)}, ids not in BTS file {len(missing)}")
    fl = fl.merge(fc, on="id", how="left").merge(gcn_inputs(trips.flight_id), on="id", how="left")
    fl["in_replay"] = fl.id.isin(set(trips.flight_id))
    alt = kg.alternative_rows(alts)
    ap, cm, routes, co = summary()
    metar, taf = weather(set(ap.iata) & set(pd.read_parquet(WEATHER / "asos.parquet", columns=["airport"]).airport))
    carriers = sorted(set(fl.marketing) | set(fl.operating))
    log(f"airports {len(ap)}, communities {len(cm)}, carriers {len(carriers)}, routes {len(routes)}, "
        f"co-disruption links {len(co)}, METAR {len(metar)}, TAF {len(taf)}, {time.time() - t0:.0f}s")
    log("flights with GCN inputs at 24h:", int(fl.gcn_origin_24h.notna().sum()), "of", int(fl.in_replay.sum()))
    return dict(ap=ap, cm=cm, routes=routes, co=co, fl=fl, alt=alt, metar=metar, taf=taf,
                carriers=pd.DataFrame({"code": carriers}))


LOADS = [
    ("Community", "cm", "UNWIND $rows AS r CREATE (c:Community) SET c = r"),
    ("Airport", "ap", "UNWIND $rows AS r CREATE (a:Airport) SET a = r"),
    ("IN_COMMUNITY", "ap", "UNWIND $rows AS r MATCH (a:Airport {iata: r.iata}), (c:Community {id: r.community}) "
                           "CREATE (a)-[:IN_COMMUNITY]->(c)"),
    ("ROUTE_TO", "routes", "UNWIND $rows AS r MATCH (a:Airport {iata: r.a}), (b:Airport {iata: r.b}) "
                           "CREATE (a)-[:ROUTE_TO {flights: r.flights, years: '2023-24'}]->(b)"),
    ("CO_DISRUPTS", "co", "UNWIND $rows AS r MATCH (a:Airport {iata: r.a}), (b:Airport {iata: r.b}) "
                          "CREATE (a)-[:CO_DISRUPTS {weight: r.weight, years: '2023-24'}]->(b)"),
    ("Carrier", "carriers", "UNWIND $rows AS r CREATE (c:Carrier) SET c = r"),
    ("Flight", "fl", "UNWIND $rows AS r CREATE (f:Flight) SET f = r"),
    ("DEPARTS_FROM/ARRIVES_AT", "fl", "UNWIND $rows AS r MATCH (f:Flight {id: r.id}), (o:Airport {iata: r.origin}), "
                                      "(d:Airport {iata: r.dest}) CREATE (f)-[:DEPARTS_FROM]->(o), (f)-[:ARRIVES_AT]->(d)"),
    ("MARKETED_BY/OPERATED_BY", "fl", "UNWIND $rows AS r MATCH (f:Flight {id: r.id}), (m:Carrier {code: r.marketing}), "
                                      "(o:Carrier {code: r.operating}) CREATE (f)-[:MARKETED_BY]->(m), (f)-[:OPERATED_BY]->(o)"),
    ("ALTERNATIVE_FOR", "alt", "UNWIND $rows AS r MATCH (a:Flight {id: r.alt}), (t:Flight {id: r.trip}) "
                               "CREATE (a)-[:ALTERNATIVE_FOR {lead_h: r.lead_h, alt_lead_h: r.alt_lead_h, "
                               "arr_ahead_min: r.arr_ahead_min, p_miss: r.p_miss}]->(t)"),
    ("Metar", "metar", "UNWIND $rows AS r MATCH (a:Airport {iata: r.airport}) CREATE (m:Metar) SET m = r "
                       "CREATE (m)-[:REPORTED_AT]->(a)"),
    ("Taf", "taf", "UNWIND $rows AS r MATCH (a:Airport {iata: r.airport}) CREATE (t:Taf) SET t = r "
                   "CREATE (t)-[:REPORTED_AT]->(a)"),
]


def load(parts, clear: bool):
    driver, db = kg.connect()
    with driver, driver.session(database=db) as s:
        if clear:
            kg.clear(s)
            log("cleared")
        elif s.run("MATCH (n) RETURN count(n) AS n").single()["n"]:
            raise SystemExit("the instance is not empty; pass --clear after a backup")
        kg.schema(s)
        for name, key, cypher in LOADS:
            t0 = time.time()
            n = kg.run(s, cypher, kg.records(parts[key]))
            log(f"  {name}: {n} rows, {time.time() - t0:.0f}s")
        log(s.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n ORDER BY n DESC").data())
        log(s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC").data())


if __name__ == "__main__":
    parts = build()
    if sys.argv[1] == "load":
        load(parts, clear="--clear" in sys.argv)
