"""Airport-hour congestion: a second label on `Weather`, and two new edges.

**Why this exists.** The graph scores flights one at a time, so correlated failure is
invisible -- the limitation named first in CLAUDE.md. Five alternatives leaving the same
airport in the same hour do not fail independently, and ranking them by individual risk
recommends the worst option with confidence. The project plan nominates a GNN as "the
proper instrument" for exactly this.

**Why no new node type.** `Weather` is already one node per airport-hour, keyed by
`iata` and `hour_utc`. It was built to carry observations; it is equally the natural
place to hang congestion. Adding the label `:AirportHour` names that second role without
renaming anything, so the existing loader, queries, schema diagram and screenshots all
stay valid. Neo4j nodes carry many labels; the course lab does the same with
`:Paper:Lab5542`.

**Why the counts are scheduled, never observed.** `n_sched_departures` and
`n_sched_arrivals` are known before the day starts, so they cannot leak an outcome.
Observed congestion -- how many flights in this hour actually ran late -- would leak,
because it is computed partly from the flight being scored and partly from flights that
had not yet departed when the prediction would have been made. The GNN reaches observed
congestion the legitimate way: by aggregating its neighbors' features, under the depth
limit in `spec_gnn.md` S5.4.

Cancelled flights are counted. A cancellation was still a scheduled movement competing
for the same gates and crew, and the count is a statement about the schedule, not about
what flew.

Run:  PYTHONPATH=src .venv/bin/python -m murphy.kg_congestion
      PYTHONPATH=src .venv/bin/python -m murphy.kg_congestion --dry-run
"""

from __future__ import annotations

import argparse
import time

from . import kg, kg_build

BATCH = 5_000

# The same strftime expression `kg_build.weather` used to mint `weather_id`. It must be
# character-identical or the edges join to nothing: DuckDB renders a TIMESTAMPTZ in the
# session timezone, so the key is only stable while the session timezone is. Reusing the
# expression rather than recomputing the hour is what keeps the two in step.
_WID = "iata || '-' || strftime(hour_utc, '%Y-%m-%dT%H')"

INDEXES = [
    "CREATE INDEX airporthour_iata IF NOT EXISTS FOR (h:AirportHour) ON (h.iata)",
]


def airport_hours(con, days=kg_build.SEED_DAYS) -> list[dict]:
    """Scheduled movement counts per airport-hour, both ends of every flight."""
    return kg_build._rows(con, f"""
        WITH slice AS (
            SELECT * FROM read_parquet('{kg_build.PARQUET}', hive_partitioning=true)
            WHERE carrier IN {kg_build._carriers_sql()}
              AND flight_date IN {kg_build._days_sql(days)}
        ),
        moves AS (
            SELECT origin AS iata, date_trunc('hour', dep_utc) AS hour_utc, 1 AS dep, 0 AS arr
            FROM slice
            UNION ALL
            SELECT dest, date_trunc('hour', arr_utc), 0, 1
            FROM slice
        )
        SELECT {_WID} AS weather_id,
               sum(dep)::INT AS n_sched_departures,
               sum(arr)::INT AS n_sched_arrivals
        FROM moves
        GROUP BY iata, hour_utc
    """)


def _flight_hour_edges(con, days, iata_col: str, time_col: str) -> list[dict]:
    return kg_build._rows(con, f"""
        WITH slice AS (
            SELECT flight_id, {iata_col} AS iata, date_trunc('hour', {time_col}) AS hour_utc
            FROM read_parquet('{kg_build.PARQUET}', hive_partitioning=true)
            WHERE carrier IN {kg_build._carriers_sql()}
              AND flight_date IN {kg_build._days_sql(days)}
              AND {time_col} IS NOT NULL
        )
        SELECT flight_id, {_WID} AS weather_id FROM slice
    """)


def departs_in(con, days=kg_build.SEED_DAYS) -> list[dict]:
    return _flight_hour_edges(con, days, "origin", "dep_utc")


def arrives_in(con, days=kg_build.SEED_DAYS) -> list[dict]:
    return _flight_hour_edges(con, days, "dest", "arr_utc")


def _write(session, cypher: str, rows: list[dict], label: str) -> int:
    if not rows:
        print(f"  {label:<24} (nothing to write)")
        return 0
    started = time.time()
    for i in range(0, len(rows), BATCH):
        session.run(cypher, rows=rows[i:i + BATCH])
    print(f"  {label:<24} {len(rows):>8,}  {time.time() - started:5.1f}s")
    return len(rows)


def load(days=kg_build.SEED_DAYS, dry_run: bool = False) -> dict:
    con = kg_build.connect()
    print(f"extracting {', '.join(days)} ...")
    hours = airport_hours(con, days)
    dep = departs_in(con, days)
    arr = arrives_in(con, days)
    print(f"  airport-hours {len(hours):,} | DEPARTS_IN {len(dep):,} | ARRIVES_IN {len(arr):,}")

    if dry_run:
        return {"airport_hours": len(hours), "departs_in": len(dep), "arrives_in": len(arr)}

    cfg = kg.settings()
    with kg.driver() as drv:
        with drv.session(database=cfg["database"]) as s:
            # Edges are MERGEd on the pair, so a second run leaves the same graph.
            # Only airport-hours that already exist as Weather nodes are labelled: the
            # ~0.35% of flights with no weather observation have no node to attach to,
            # and inventing one would be a node with no observation behind it.
            print("label and counts:")
            labelled = _write(s, """
                UNWIND $rows AS row
                MATCH (w:Weather {weather_id: row.weather_id})
                SET w:AirportHour,
                    w.n_sched_departures = row.n_sched_departures,
                    w.n_sched_arrivals   = row.n_sched_arrivals,
                    w.n_sched_movements  = row.n_sched_departures + row.n_sched_arrivals
            """, hours, "AirportHour")

            for statement in INDEXES:
                s.run(statement)

            print("edges:")
            _write(s, """
                UNWIND $rows AS row
                MATCH (f:Flight {flight_id: row.flight_id})
                MATCH (h:AirportHour {weather_id: row.weather_id})
                MERGE (f)-[:DEPARTS_IN]->(h)
            """, dep, "DEPARTS_IN")

            _write(s, """
                UNWIND $rows AS row
                MATCH (f:Flight {flight_id: row.flight_id})
                MATCH (h:AirportHour {weather_id: row.weather_id})
                MERGE (f)-[:ARRIVES_IN]->(h)
            """, arr, "ARRIVES_IN")

            got = s.run("""
                MATCH (h:AirportHour) WITH count(h) AS hours
                MATCH ()-[d:DEPARTS_IN]->() WITH hours, count(d) AS dep
                MATCH ()-[a:ARRIVES_IN]->() RETURN hours, dep, count(a) AS arr
            """).single().data()

    n, r = kg.counts()
    print(f"\nin graph: {got['hours']:,} AirportHour | "
          f"{got['dep']:,} DEPARTS_IN | {got['arr']:,} ARRIVES_IN")
    print(f"totals:   {n:,} nodes ({100*n/200_000:.0f}% of cap) | "
          f"{r:,} relationships ({100*r/400_000:.0f}% of cap)")
    return {**got, "nodes": n, "relationships": r}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="extract and count, write nothing")
    args = ap.parse_args()
    load(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
