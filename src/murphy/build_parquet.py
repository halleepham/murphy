"""Build the queryable flight table from the raw 2024 CSV.

Keeps only what the Challenge 2 slice needs, mints a stable flight ID, derives
the departure-hour bin the match key uses, and drops the 107 rows whose
schedule arithmetic cannot be explained (see notebooks/schema_findings.md).

Deliberately does NOT carry the timestamp columns forward. They are broken --
local times with no timezone, and no midnight rollover -- so the only sound
representation is the scheduled clock time as a display string plus ARR_DELAY
in minutes as the label.

Does carry the observed weather columns, which describe conditions on the day
each historical flight actually flew. They are shown as evidence, never used to
forecast: the app has no weather forecast for the traveller's own flight.

Run:  .venv/bin/python src/murphy/build_parquet.py
"""

import shutil
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "flight_with_weather_2024.csv"
OUT = ROOT / "data" / "processed" / "flights"

# Rows where clock duration minus scheduled duration is not a whole number of
# hours. A whole-hour gap is a timezone offset or the missing day rollover;
# anything else means the schedule fields disagree for a reason we cannot name.
UNEXPLAINED = "(date_diff('minute', CRS_DEP_TIME, CRS_ARR_TIME) - CRS_ELAPSED_TIME) % 60 <> 0"

BUILD = f"""
SELECT
    -- Natural key: unique across all 6,284,841 rows, verified zero duplicates.
    strftime(FL_DATE, '%Y-%m-%d') || '_' || OP_CARRIER
        || CAST(CAST(OP_CARRIER_FL_NUM AS INT) AS VARCHAR)
        || '_' || ORIGIN || '_' || DEST                       AS flight_id,

    FL_DATE::DATE                                             AS flight_date,
    OP_CARRIER                                                AS carrier,
    CAST(OP_CARRIER_FL_NUM AS INT)                            AS flight_number,
    ORIGIN                                                    AS origin,
    DEST                                                      AS dest,

    MONTH                                                     AS month,
    DAY_OF_WEEK                                               AS day_of_week,

    -- Scheduled departure as minutes past local midnight, plus the 2-hour bin
    -- the match key uses. Safe: this is a clock reading, not an instant.
    hour(CRS_DEP_TIME) * 60 + minute(CRS_DEP_TIME)            AS sched_dep_minutes,
    hour(CRS_DEP_TIME)                                        AS sched_dep_hour,
    hour(CRS_DEP_TIME) // 2                                   AS dep_hour_bin,
    strftime(CRS_DEP_TIME, '%H:%M')                           AS sched_dep_local,
    strftime(CRS_ARR_TIME, '%H:%M')                           AS sched_arr_local,

    CAST(CRS_ELAPSED_TIME AS INT)                             AS sched_duration_min,

    -- Observed weather on the day of the flight. Carried as a property of the
    -- evidence so the traveller can see which comparable flights ran in bad
    -- conditions -- not as a forecast input, and not as a causal claim.
    -- Units inferred from value ranges: Celsius, millimetres, km/h.
    round(O_TEMP, 1)                                          AS origin_temp_c,
    round(O_PRCP, 1)                                          AS origin_precip_mm,
    round(O_WSPD, 1)                                          AS origin_wind_kph,
    round(D_TEMP, 1)                                          AS dest_temp_c,
    round(D_PRCP, 1)                                          AS dest_precip_mm,
    round(D_WSPD, 1)                                          AS dest_wind_kph,

    -- Outcomes. Labels and evidence, never model inputs.
    CAST(ARR_DELAY AS INT)                                    AS arr_delay_min,
    CAST(DEP_DELAY AS INT)                                    AS dep_delay_min
FROM read_csv_auto('{CSV}', header=true)
WHERE NOT {UNEXPLAINED}
"""


def main():
    if not CSV.exists():
        raise SystemExit(f"Missing: {CSV}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv_auto('{CSV}', header=true)")
    n_raw = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    n_drop = con.execute(f"SELECT count(*) FROM raw WHERE {UNEXPLAINED}").fetchone()[0]

    print(f"Reading  {n_raw:,} rows")
    print(f"Dropping {n_drop:,} unexplained rows ({100.0 * n_drop / n_raw:.4f}%)")
    print(f"Writing  {OUT}/ partitioned by origin ...")

    # Clear the target first. OVERWRITE_OR_IGNORE leaves files from a previous
    # build in place and writes new ones beside them, so a schema change would
    # silently produce a directory holding two incompatible layouts.
    if OUT.exists():
        shutil.rmtree(OUT)

    con.execute(f"""
        COPY ({BUILD}) TO '{OUT}'
        (FORMAT PARQUET, PARTITION_BY (origin), OVERWRITE_OR_IGNORE, COMPRESSION ZSTD)
    """)

    con.execute(f"CREATE VIEW out AS SELECT * FROM read_parquet('{OUT}/**/*.parquet', hive_partitioning=true)")
    n_out = con.execute("SELECT count(*) FROM out").fetchone()[0]
    n_ids = con.execute("SELECT count(DISTINCT flight_id) FROM out").fetchone()[0]
    size_mb = sum(p.stat().st_size for p in OUT.rglob("*.parquet")) / 1e6

    print(f"\nWrote {n_out:,} rows across {len(list(OUT.iterdir()))} origin partitions, "
          f"{size_mb:,.0f} MB")
    print(f"flight_id unique: {n_ids == n_out} ({n_ids:,} distinct)")
    if n_out != n_raw - n_drop:
        print(f"WARNING: expected {n_raw - n_drop:,} rows, got {n_out:,}")

    print("\nSample:")
    con.sql("SELECT * FROM out USING SAMPLE 3 ROWS").show()


if __name__ == "__main__":
    main()
