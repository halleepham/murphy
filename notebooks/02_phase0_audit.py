"""Phase 0 audit for Challenge 3 — verify the data before designing the graph schema.

Challenge 2's expensive lesson was that a schema designed against assumptions costs
more than one designed against the file. Every check here feeds a specific schema
decision, named in the output. Reproduce with:

    .venv/bin/python notebooks/02_phase0_audit.py

Findings are written up in docs/challenge-3/phase0_findings.md.
"""

import duckdb

FULL = "read_parquet('data/processed/flights_full/**/*.parquet', hive_partitioning=true)"
Y2024 = "read_parquet('data/processed/flights/**/*.parquet', hive_partitioning=true)"

# The ten US majors the Challenge 3 scope keeps (decision D1). Operating carrier,
# i.e. the aircraft the traveller actually boards.
MAJORS = "('AA','DL','UA','WN','AS','B6','NK','F9','G4','HA')"


def show(con, title, sql):
    print(f"\n=== {title} ===")
    print(con.execute(sql).df().to_string(index=False))


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")

    show(con, "Carriers, 2023-2024 (carrier = BTS Reporting_Airline = operating)", f"""
        SELECT carrier, count(*) AS flights,
               round(100.0*count(*)/sum(count(*)) OVER (), 2) AS pct,
               carrier IN {MAJORS} AS major,
               count(DISTINCT tail_number) AS tails,
               count(DISTINCT origin) AS origins
        FROM {FULL} GROUP BY carrier ORDER BY flights DESC
    """)

    show(con, "D1 coverage cost: what majors-only scoping drops", f"""
        WITH f AS (SELECT *, carrier IN {MAJORS} AS major FROM {FULL}),
             ap AS (SELECT origin a, bool_or(major) keep FROM f GROUP BY origin),
             rt AS (SELECT origin||'-'||dest r, bool_or(major) keep FROM f GROUP BY r)
        SELECT (SELECT count(*) FROM f) AS flights_all,
               (SELECT count(*) FROM f WHERE major) AS flights_kept,
               (SELECT count(*) FROM ap) AS airports_all,
               (SELECT count(*) FROM ap WHERE keep) AS airports_kept,
               (SELECT count(*) FROM rt) AS routes_all,
               (SELECT count(*) FROM rt WHERE keep) AS routes_kept
    """)

    show(con, "Airports that disappear entirely under D1 (regional-only)", f"""
        WITH f AS (SELECT *, carrier IN {MAJORS} AS major FROM {FULL})
        SELECT origin, count(*) flights FROM f
        GROUP BY origin HAVING bool_or(major) = false
        ORDER BY flights DESC LIMIT 15
    """)

    # The PRECEDES edge is the backbone of the rotation and correlated-failure
    # queries. If tails are sparse or chains do not close spatially, much of the
    # graph's value disappears -- so this is the check that most needed running.
    show(con, "Tail numbers: null rate", f"""
        SELECT round(100.0*avg((tail_number IS NULL)::INT), 2) AS pct_null,
               round(100.0*avg((tail_number IS NULL AND cancelled=1)::INT)
                     / nullif(avg((tail_number IS NULL)::INT), 0), 1) AS pct_of_nulls_cancelled
        FROM {FULL}
    """)

    show(con, "Rotation chains: does leg N's dest equal leg N+1's origin?", f"""
        WITH ch AS (
          SELECT tail_number, dep_utc, dest,
                 lead(origin)  OVER w AS nxt_origin,
                 lead(dep_utc) OVER w AS nxt_dep
          FROM {FULL}
          WHERE tail_number IS NOT NULL AND cancelled = 0 AND diverted = 0
          WINDOW w AS (PARTITION BY tail_number ORDER BY dep_utc)
        )
        SELECT count(*) AS linked_pairs,
               round(100.0*avg((dest = nxt_origin)::INT), 2) AS pct_continuous,
               round(median(datediff('minute', dep_utc, nxt_dep)), 0) AS median_gap_min
        FROM ch WHERE nxt_origin IS NOT NULL
    """)

    show(con, "Rotation depth: legs per tail per day", f"""
        WITH d AS (SELECT tail_number, flight_date, count(*) legs FROM {FULL}
                   WHERE tail_number IS NOT NULL AND cancelled = 0 GROUP BY 1, 2)
        SELECT round(avg(legs), 2) mean_legs, median(legs) med_legs, max(legs) max_legs
        FROM d
    """)

    # The defect that forced the Aeolus replacement. Verified, not trusted.
    show(con, "Timestamp integrity", f"""
        SELECT sum((arr_utc <= dep_utc)::INT) AS arrives_before_departs,
               round(100.0*avg((datediff('minute', dep_utc, arr_utc)
                                = sched_duration_min)::INT), 2) AS pct_duration_matches_clock,
               sum((dep_utc IS NULL OR arr_utc IS NULL)::INT) AS null_timestamps
        FROM {FULL} WHERE cancelled = 0
    """)

    show(con, "Cancellations and causes", f"""
        SELECT round(100.0*avg(cancelled), 2) AS pct_cancelled,
               round(100.0*avg(diverted), 2) AS pct_diverted
        FROM {FULL}
    """)

    show(con, "Cancellation codes", f"""
        SELECT coalesce(cancellation_code, '(null)') AS code, count(*) AS n
        FROM {FULL} WHERE cancelled = 1 GROUP BY 1 ORDER BY n DESC
    """)

    # BTS only populates cause columns when arrival delay >= 15 minutes. Confirming
    # that convention holds stops us modelling a null as "no cause".
    show(con, "Delay causes follow the BTS >=15 min convention", f"""
        SELECT CASE WHEN arr_delay_min >= 15 THEN 'delayed >=15' ELSE 'under 15' END AS bucket,
               count(*) AS n,
               round(100.0*avg((delay_carrier_min IS NOT NULL)::INT), 1) AS pct_causes_present
        FROM {FULL} WHERE cancelled = 0 GROUP BY 1
    """)

    show(con, "Weather join coverage (null rates)", f"""
        SELECT round(100.0*avg((origin_visibility_m IS NULL)::INT), 2) AS origin_vis_null,
               round(100.0*avg((origin_temp_c IS NULL)::INT), 2) AS origin_temp_null,
               round(100.0*avg((dest_visibility_m IS NULL)::INT), 2) AS dest_vis_null
        FROM {FULL}
    """)

    # Airport nodes must be built from origin UNION dest: STC appears once as a
    # destination only, almost certainly a diversion.
    show(con, "Airports appearing as destination but never as origin", f"""
        SELECT dest, count(*) n FROM {FULL}
        WHERE dest NOT IN (SELECT DISTINCT origin FROM {FULL})
        GROUP BY 1 ORDER BY n DESC
    """)

    # Aura Free caps at 200k nodes / 400k relationships, so a full year does not
    # fit and the seed graph has to be a scoped slice.
    show(con, "Node counts if the whole of 2024 (majors) were loaded", f"""
        SELECT count(*) AS flight_nodes,
               count(DISTINCT origin||'-'||dest||'-'||carrier||'-'||sched_dep_hour) AS service_nodes,
               count(DISTINCT origin) AS airport_nodes,
               count(DISTINCT carrier) AS airline_nodes,
               count(DISTINCT tail_number) AS aircraft_nodes,
               count(DISTINCT origin||'-'||date_trunc('hour', dep_utc)) AS weather_nodes
        FROM {Y2024} WHERE carrier IN {MAJORS}
    """)

    show(con, "One hub, one day (ATL, 2024-11-18) -- candidate seed slice", f"""
        SELECT count(*) AS flights, count(DISTINCT tail_number) AS tails,
               count(DISTINCT carrier) AS carriers, count(DISTINCT dest) AS dests
        FROM {Y2024}
        WHERE carrier IN {MAJORS} AND origin = 'ATL' AND flight_date = DATE '2024-11-18'
    """)


if __name__ == "__main__":
    main()
