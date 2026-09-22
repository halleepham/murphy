"""Verify the 2024 flight CSV against the assumptions in the project plan.

Reads the raw CSV in place with DuckDB. Writes nothing, drops nothing, fixes
nothing. The point is to find out what is actually in the file before any
cleaning rule is chosen.

Run:  .venv/bin/python notebooks/01_schema_check.py
"""

from pathlib import Path

import duckdb

CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "flight_with_weather_2024.csv"

# Columns that describe what happened after the flight. Labels, never inputs.
OUTCOME_COLS = ["DEP_DELAY", "ARR_DELAY", "DEP_TIME", "ARR_TIME", "TAXI_OUT", "TAXI_IN",
                "WHEELS_OFF", "WHEELS_ON", "ACTUAL_ELAPSED_TIME", "AIR_TIME"]


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    if not CSV.exists():
        raise SystemExit(f"Missing: {CSV}")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW f AS SELECT * FROM read_csv_auto('{CSV}', header=true)")
    n_rows = con.execute("SELECT count(*) FROM f").fetchone()[0]

    section("1. Shape and inferred types")
    for name, dtype, *_ in con.execute("DESCRIBE f").fetchall():
        print(f"  {name:<22} {dtype}")
    print(f"\n{n_rows:,} rows")
    print("\nNOTE: the time columns are TIMESTAMPs, not the HHMM integers the plan")
    print("assumes. Kaggle preprocessed them. See section 5 for what that broke.")

    section("2. Date coverage")
    con.sql("""
        SELECT MONTH, count(*) AS n, count(DISTINCT FL_DATE) AS days
        FROM f GROUP BY MONTH ORDER BY MONTH
    """).show(max_rows=20)
    con.sql("""
        SELECT min(FL_DATE) AS first_date, max(FL_DATE) AS last_date,
               count(DISTINCT FL_DATE) AS distinct_dates, 366 - count(DISTINCT FL_DATE) AS missing
        FROM f
    """).show()

    section("3. Null rate per column")
    for name, *_ in con.execute("DESCRIBE f").fetchall():
        n_null = con.execute(f'SELECT count(*) - count("{name}") FROM f').fetchone()[0]
        flag = "  <-- has nulls" if n_null else ""
        print(f"  {name:<22} {n_null:>10,}  {100.0 * n_null / n_rows:>6.2f}%{flag}")

    section("4. Carrier codes")
    con.sql("SELECT OP_CARRIER, count(*) AS n FROM f GROUP BY 1 ORDER BY n DESC").show(max_rows=40)

    section("5. Time semantics: rollover and timezone")
    print("Every time column carries FL_DATE's calendar date. If rollover had been")
    print("applied, an overnight flight's arrival would sit on the following day.\n")
    con.sql("""
        SELECT sum(CASE WHEN CRS_DEP_TIME::DATE <> FL_DATE::DATE THEN 1 ELSE 0 END) AS dep_date_differs,
               sum(CASE WHEN CRS_ARR_TIME::DATE <> FL_DATE::DATE THEN 1 ELSE 0 END) AS arr_date_differs,
               sum(CASE WHEN ARR_TIME::DATE  <> FL_DATE::DATE THEN 1 ELSE 0 END) AS act_arr_date_differs
        FROM f
    """).show()

    print("Consequence: scheduled arrival lands before scheduled departure.")
    con.sql("""
        SELECT count(*) AS n, round(100.0 * count(*) / (SELECT count(*) FROM f), 3) AS pct
        FROM f WHERE CRS_ARR_TIME < CRS_DEP_TIME
    """).show()

    print("Clock duration minus stored CRS_ELAPSED_TIME. Whole-hour gaps are timezone")
    print("offsets; gaps near -1440 are the missing day rollover.")
    con.sql("""
        WITH d AS (SELECT date_diff('minute', CRS_DEP_TIME, CRS_ARR_TIME) - CRS_ELAPSED_TIME AS diff FROM f)
        SELECT diff, count(*) AS n, round(100.0 * count(*) / (SELECT count(*) FROM f), 2) AS pct
        FROM d GROUP BY diff ORDER BY n DESC LIMIT 12
    """).show(max_rows=15)

    con.sql("""
        WITH d AS (SELECT date_diff('minute', CRS_DEP_TIME, CRS_ARR_TIME) - CRS_ELAPSED_TIME AS diff FROM f)
        SELECT CASE WHEN diff % 60 = 0 THEN 'whole hour (timezone / rollover)'
                    ELSE 'NOT a whole hour (unexplained)' END AS kind, count(*) AS n
        FROM d GROUP BY kind ORDER BY n DESC
    """).show()

    section("6. Cancellations and diversions")
    print("There is no CANCELLED or DIVERTED column. If those flights were present")
    print("they would show as null actual times. They do not:")
    con.sql("""
        SELECT count(*) - count(DEP_TIME) AS null_dep_time,
               count(*) - count(ARR_TIME) AS null_arr_time,
               count(*) - count(ARR_DELAY) AS null_arr_delay
        FROM f
    """).show()
    print("They were already removed upstream. Every row is a flight that operated.")

    section("7. Is ARR_DELAY trustworthy as the label?")
    con.sql("""
        WITH d AS (SELECT ARR_DELAY, date_diff('minute', CRS_ARR_TIME, ARR_TIME) AS clock FROM f)
        SELECT count(*) AS n, sum(CASE WHEN ARR_DELAY = clock THEN 1 ELSE 0 END) AS agrees,
               round(100.0 * sum(CASE WHEN ARR_DELAY = clock THEN 1 ELSE 0 END) / count(*), 2) AS pct
        FROM d
    """).show()
    print("Disagreement is concentrated in the rollover rows, so ARR_DELAY is the")
    print("reliable column and the timestamps are the derived, broken ones.\n")
    con.sql("""
        SELECT min(ARR_DELAY) AS lo, max(ARR_DELAY) AS hi, round(avg(ARR_DELAY), 2) AS mean,
               quantile_cont(ARR_DELAY, 0.10) AS p10, quantile_cont(ARR_DELAY, 0.50) AS p50,
               quantile_cont(ARR_DELAY, 0.90) AS p90, quantile_cont(ARR_DELAY, 0.99) AS p99
        FROM f
    """).show()

    section("8. Duplicate flight keys")
    con.sql("""
        SELECT count(*) AS duplicated_keys, coalesce(sum(n), 0) AS rows_involved
        FROM (SELECT count(*) AS n FROM f
              GROUP BY FL_DATE, OP_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, DEST HAVING count(*) > 1)
    """).show()

    section("9. Columns of unclear meaning")
    con.sql("""
        SELECT count(DISTINCT FLIGHTS) AS flights_distinct, max(FLIGHTS) AS flights_max,
               count(DISTINCT ORIGIN) AS airports, count(DISTINCT ORIGIN_INDEX) AS origin_index_vals,
               min(ORIGIN_INDEX) AS idx_lo, max(ORIGIN_INDEX) AS idx_hi
        FROM f
    """).show()
    print("Is ORIGIN_INDEX just an integer encoding of ORIGIN?")
    con.sql("""
        SELECT count(*) AS airports_with_more_than_one_index
        FROM (SELECT ORIGIN FROM f GROUP BY ORIGIN HAVING count(DISTINCT ORIGIN_INDEX) > 1)
    """).show()
    print("Do MONTH / DAY_OF_MONTH agree with FL_DATE?")
    con.sql("""
        SELECT sum(CASE WHEN MONTH <> month(FL_DATE) THEN 1 ELSE 0 END) AS month_mismatch,
               sum(CASE WHEN DAY_OF_MONTH <> day(FL_DATE) THEN 1 ELSE 0 END) AS day_mismatch
        FROM f
    """).show()

    section("10. Comparable-flight counts (the retrieval question)")
    print("Match key: route + carrier + month + 2-hour scheduled-departure bin.\n")
    for origin, dest, carrier, label in [("ATL", "LAX", "DL", "trunk"),
                                         ("BOS", "ATL", "DL", "medium"),
                                         ("MCI", "DEN", "WN", "thin"),
                                         ("MCI", "MDW", "WN", "thin")]:
        row = con.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE MONTH = 10),
                   count(*) FILTER (WHERE MONTH = 10 AND hour(CRS_DEP_TIME) // 2 = 3)
            FROM f WHERE ORIGIN = '{origin}' AND DEST = '{dest}' AND OP_CARRIER = '{carrier}'
        """).fetchone()
        print(f"  {origin}->{dest} {carrier:<3} ({label:<6}) "
              f"year={row[0]:>6,}   october={row[1]:>5,}   oct + 06:00-07:59 bin={row[2]:>4,}")

    print("\nSize of every route+carrier+month+bin group in the file:")
    con.sql("""
        WITH g AS (SELECT count(*) AS n FROM f
                   GROUP BY ORIGIN, DEST, OP_CARRIER, MONTH, hour(CRS_DEP_TIME) // 2)
        SELECT count(*) AS n_groups, round(avg(n), 1) AS mean,
               quantile_cont(n, 0.10) AS p10, quantile_cont(n, 0.50) AS median,
               quantile_cont(n, 0.90) AS p90, max(n) AS largest,
               sum(CASE WHEN n < 10 THEN 1 ELSE 0 END) AS under_10,
               round(100.0 * sum(CASE WHEN n < 10 THEN 1 ELSE 0 END) / count(*), 1) AS pct_under_10
        FROM g
    """).show()

    print("What share of actual FLIGHTS fall into a group that thin?")
    con.sql("""
        WITH g AS (SELECT count(*) AS n FROM f
                   GROUP BY ORIGIN, DEST, OP_CARRIER, MONTH, hour(CRS_DEP_TIME) // 2)
        SELECT sum(CASE WHEN n < 10 THEN n ELSE 0 END) AS flights_in_thin_groups,
               round(100.0 * sum(CASE WHEN n < 10 THEN n ELSE 0 END) / sum(n), 1) AS pct_of_all_flights
        FROM g
    """).show()

    section("11. Outcome columns present (labels, never model inputs)")
    print("  " + ", ".join(OUTCOME_COLS))
    print("\nDone. Nothing was written.")


if __name__ == "__main__":
    main()
