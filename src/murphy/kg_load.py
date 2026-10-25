"""Load the extracted records into Neo4j.

Every write is a parameterised MERGE on a key backed by a uniqueness constraint, so
the load is idempotent: running it twice leaves the same graph rather than a doubled
one. That matters more than it sounds -- iterating on a schema means loading many
times, and a loader that silently duplicates turns every count into a lie.

Run:  PYTHONPATH=src .venv/bin/python -m murphy.kg_load            # load
      PYTHONPATH=src .venv/bin/python -m murphy.kg_load --reset    # wipe, then load
      PYTHONPATH=src .venv/bin/python -m murphy.kg_load --dry-run  # counts only
"""

from __future__ import annotations

import argparse
import time

from . import kg, kg_build, kg_policy, kg_samples

BATCH = 5_000

CONSTRAINTS = [
    "CREATE CONSTRAINT flight_id   IF NOT EXISTS FOR (f:Flight)       REQUIRE f.flight_id    IS UNIQUE",
    "CREATE CONSTRAINT service_id  IF NOT EXISTS FOR (s:Service)      REQUIRE s.service_id   IS UNIQUE",
    "CREATE CONSTRAINT aircraft_id IF NOT EXISTS FOR (a:Aircraft)     REQUIRE a.tail_number  IS UNIQUE",
    "CREATE CONSTRAINT airport_id  IF NOT EXISTS FOR (p:Airport)      REQUIRE p.iata         IS UNIQUE",
    "CREATE CONSTRAINT airline_id  IF NOT EXISTS FOR (l:Airline)      REQUIRE l.carrier      IS UNIQUE",
    "CREATE CONSTRAINT weather_id  IF NOT EXISTS FOR (w:Weather)      REQUIRE w.weather_id   IS UNIQUE",
    "CREATE CONSTRAINT clause_id   IF NOT EXISTS FOR (c:PolicyClause) REQUIRE c.clause_id    IS UNIQUE",
    "CREATE CONSTRAINT trav_id     IF NOT EXISTS FOR (t:Traveler)     REQUIRE t.traveler_id  IS UNIQUE",
    "CREATE CONSTRAINT itin_id     IF NOT EXISTS FOR (i:Itinerary)    REQUIRE i.itinerary_id IS UNIQUE",
]

# Time is a property with a range index rather than a node (decision D5). Connection
# feasibility is a range scan over sched_dep_utc, so these indexes are what make the
# derived CONNECTS_TO query viable instead of a full scan.
INDEXES = [
    "CREATE INDEX flight_sched_dep IF NOT EXISTS FOR (f:Flight)  ON (f.sched_dep_utc)",
    "CREATE INDEX flight_sched_arr IF NOT EXISTS FOR (f:Flight)  ON (f.sched_arr_utc)",
    "CREATE INDEX weather_hour     IF NOT EXISTS FOR (w:Weather) ON (w.hour_utc)",
    "CREATE INDEX service_hour     IF NOT EXISTS FOR (s:Service) ON (s.sched_dep_hour)",
]

# PolicyClause is the only node type carrying prose, and prose is the only thing a
# traveller asks about in their own words. Without this index a question like "can I
# get a hotel?" can only be answered by someone who already knows the property is
# called commitment: "hotel".
FULLTEXT = [
    "CREATE FULLTEXT INDEX policy_text IF NOT EXISTS "
    "FOR (c:PolicyClause) ON EACH [c.text, c.commitment]",
]

# origin, dest, service_id and tail_number are carried on the extracted Flight record
# because the edges are built from them -- but they are not stored on the node.
# DEPARTS_FROM, ARRIVES_AT, INSTANCE_OF and FLIES are the single source of truth, and
# a duplicated value is a value that can drift (decision S6).
FLIGHT_EDGE_ONLY = {"origin", "dest", "service_id", "tail_number"}
SERVICE_EDGE_ONLY = {"origin", "dest", "carrier"}


def _strip(rows: list[dict], drop: set[str]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def _write(session, cypher: str, rows: list[dict], label: str) -> None:
    """Push rows in batches, reporting progress because a silent minute looks hung."""
    if not rows:
        print(f"  {label:<24} (nothing to write)")
        return
    started = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        session.run(cypher, rows=chunk)
    print(f"  {label:<24} {len(rows):>8,}  {time.time() - started:5.1f}s")


def load(days=kg_build.SEED_DAYS, reset: bool = False) -> tuple[int, int]:
    con = kg_build.connect()
    print(f"extracting {', '.join(days)} ...")
    flights = kg_build.flights(con, days)
    services = kg_build.services(con, days)
    aircraft = kg_build.aircraft(con, days)
    airports = kg_build.airports(con, days)
    airlines = kg_build.airlines(con)
    weather = kg_build.weather(con, days)
    precedes = kg_build.precedes(con, days)
    endpoints = kg_build.service_endpoints(con, days)

    cfg = kg.settings()
    with kg.driver() as drv:
        with drv.session(database=cfg["database"]) as s:
            if reset:
                print("resetting ...")
                # Deleted in chunks: a single DETACH DELETE over the whole graph
                # exceeds the transaction memory an Aura Free instance will allow.
                while True:
                    summary = s.run(
                        "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n "
                        "RETURN count(n) AS n"
                    ).single()
                    if not summary["n"]:
                        break

            print("constraints and indexes ...")
            for statement in CONSTRAINTS + INDEXES + FULLTEXT:
                s.run(statement)

            print("nodes:")
            _write(s, """
                UNWIND $rows AS row
                MERGE (f:Flight {flight_id: row.flight_id})
                SET f += row
            """, _strip(flights, FLIGHT_EDGE_ONLY), "Flight")

            _write(s, """
                UNWIND $rows AS row
                MERGE (s:Service {service_id: row.service_id})
                SET s += row
            """, _strip(services, SERVICE_EDGE_ONLY), "Service")

            _write(s, """
                UNWIND $rows AS row
                MERGE (a:Aircraft {tail_number: row.tail_number})
                SET a += row
            """, aircraft, "Aircraft")

            _write(s, """
                UNWIND $rows AS row
                MERGE (p:Airport {iata: row.iata})
                SET p += row
            """, airports, "Airport")

            _write(s, """
                UNWIND $rows AS row
                MERGE (l:Airline {carrier: row.carrier})
                SET l += row
            """, airlines, "Airline")

            _write(s, """
                UNWIND $rows AS row
                MERGE (w:Weather {weather_id: row.weather_id})
                SET w += row
            """, weather, "Weather")

            print("relationships:")
            # Cancelled flights have no tail number, so there is no aircraft to link.
            _write(s, """
                UNWIND $rows AS row
                MATCH (a:Aircraft {tail_number: row.tail_number})
                MATCH (f:Flight {flight_id: row.flight_id})
                MERGE (a)-[:FLIES]->(f)
            """, [{"tail_number": f["tail_number"], "flight_id": f["flight_id"]}
                  for f in flights if f["tail_number"]], "FLIES")

            _write(s, """
                UNWIND $rows AS row
                MATCH (a:Flight {flight_id: row.from_id})
                MATCH (b:Flight {flight_id: row.to_id})
                MERGE (a)-[r:PRECEDES]->(b)
                SET r.gap_min = row.gap_min
            """, precedes, "PRECEDES")

            _write(s, """
                UNWIND $rows AS row
                MATCH (f:Flight {flight_id: row.flight_id})
                MATCH (s:Service {service_id: row.service_id})
                MERGE (f)-[:INSTANCE_OF]->(s)
            """, [{"flight_id": f["flight_id"], "service_id": f["service_id"]}
                  for f in flights], "INSTANCE_OF")

            _write(s, """
                UNWIND $rows AS row
                MATCH (f:Flight {flight_id: row.flight_id})
                MATCH (p:Airport {iata: row.iata})
                MERGE (f)-[:DEPARTS_FROM]->(p)
            """, [{"flight_id": f["flight_id"], "iata": f["origin"]} for f in flights],
                 "DEPARTS_FROM (flight)")

            _write(s, """
                UNWIND $rows AS row
                MATCH (f:Flight {flight_id: row.flight_id})
                MATCH (p:Airport {iata: row.iata})
                MERGE (f)-[:ARRIVES_AT]->(p)
            """, [{"flight_id": f["flight_id"], "iata": f["dest"]} for f in flights],
                 "ARRIVES_AT (flight)")

            # Services reach airports through the same two relationships, which is what
            # lets route planning traverse Airport <- Service -> Airport without a
            # combinatorial CONNECTS_TO between services (decision S3).
            _write(s, """
                UNWIND $rows AS row
                MATCH (s:Service {service_id: row.service_id})
                MATCH (p:Airport {iata: row.iata})
                MERGE (s)-[:DEPARTS_FROM]->(p)
            """, [{"service_id": e["service_id"], "iata": e["origin"]} for e in endpoints],
                 "DEPARTS_FROM (service)")

            _write(s, """
                UNWIND $rows AS row
                MATCH (s:Service {service_id: row.service_id})
                MATCH (p:Airport {iata: row.iata})
                MERGE (s)-[:ARRIVES_AT]->(p)
            """, [{"service_id": e["service_id"], "iata": e["dest"]} for e in endpoints],
                 "ARRIVES_AT (service)")

            _write(s, """
                UNWIND $rows AS row
                MATCH (s:Service {service_id: row.service_id})
                MATCH (l:Airline {carrier: row.carrier})
                MERGE (s)-[:OPERATED_BY]->(l)
            """, [{"service_id": e["service_id"], "carrier": e["carrier"]} for e in endpoints],
                 "OPERATED_BY")

            _write(s, """
                UNWIND $rows AS row
                MATCH (w:Weather {weather_id: row.weather_id})
                MATCH (p:Airport {iata: row.iata})
                MERGE (w)-[:OBSERVED_AT]->(p)
            """, [{"weather_id": w["weather_id"], "iata": w["iata"]} for w in weather],
                 "OBSERVED_AT")

            print("policy:")
            policy = kg_policy.summary()
            _write(s, """
                UNWIND $rows AS row
                MERGE (c:PolicyClause {clause_id: row.clause_id})
                SET c += row
            """, policy["PolicyClause"], "PolicyClause")

            # A commitment a carrier does not make is simply an absent edge, which is
            # what a graph is for -- no "No" rows to filter out later.
            _write(s, """
                UNWIND $rows AS row
                MATCH (l:Airline {carrier: row.carrier})
                MATCH (c:PolicyClause {clause_id: row.clause_id})
                MERGE (l)-[:PUBLISHES]->(c)
            """, policy["PUBLISHES"], "PUBLISHES")

            print("samples:")
            _write(s, """
                UNWIND $rows AS row
                MERGE (t:Traveler {traveler_id: row.traveler_id})
                SET t += row
            """, kg_samples.traveler_rows(), "Traveler")

            _write(s, """
                UNWIND $rows AS row
                MERGE (i:Itinerary {itinerary_id: row.itinerary_id})
                SET i += row
            """, kg_samples.itinerary_rows(), "Itinerary")

            _write(s, """
                UNWIND $rows AS row
                MATCH (t:Traveler {traveler_id: row.traveler_id})
                MATCH (i:Itinerary {itinerary_id: row.itinerary_id})
                MERGE (t)-[:BOOKED]->(i)
            """, kg_samples.booked_rows(), "BOOKED")

            _write(s, """
                UNWIND $rows AS row
                MATCH (i:Itinerary {itinerary_id: row.itinerary_id})
                MATCH (f:Flight {flight_id: row.flight_id})
                MERGE (i)-[c:CONTAINS]->(f)
                SET c.leg_order = row.leg_order
            """, kg_samples.contains_rows(), "CONTAINS")

            # Itinerary endpoints and duration are derived from the legs rather than
            # hand-written, so they cannot drift away from the flights they describe.
            s.run("""
                MATCH (i:Itinerary)-[c:CONTAINS]->(f:Flight)
                WITH i, min(f.sched_dep_utc) AS dep, max(f.sched_arr_utc) AS arr
                SET i.total_sched_min = duration.between(dep, arr).minutes
            """)
            s.run("""
                MATCH (i:Itinerary)-[c:CONTAINS]->(:Flight)-[:DEPARTS_FROM]->(p:Airport)
                WITH i, c.leg_order AS leg, p.iata AS iata ORDER BY leg
                WITH i, collect(iata) AS origins
                SET i.origin = origins[0]
            """)
            s.run("""
                MATCH (i:Itinerary)-[c:CONTAINS]->(:Flight)-[:ARRIVES_AT]->(p:Airport)
                WITH i, c.leg_order AS leg, p.iata AS iata ORDER BY leg
                WITH i, collect(iata) AS dests
                SET i.final_dest = dests[-1]
            """)
            print("  itinerary endpoints derived from legs")

    nodes, rels = kg.counts()
    print(f"\ngraph now holds {nodes:,} nodes and {rels:,} relationships")
    return nodes, rels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true",
                    help="delete everything first. The graph is rebuilt from Parquet, "
                         "so nothing is lost that cannot be reloaded.")
    ap.add_argument("--dry-run", action="store_true",
                    help="extract and report counts without writing")
    a = ap.parse_args()

    if a.dry_run:
        kg_build.main()
        return
    load(reset=a.reset)


if __name__ == "__main__":
    main()
