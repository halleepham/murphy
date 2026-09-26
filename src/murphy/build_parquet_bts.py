"""Build the queryable flight table from BTS records and NOAA observations.

Replaces the Kaggle-preprocessed source. What changes, and why it matters:

  * Cancelled and diverted flights are KEPT. The old file had them stripped, so
    the application was blind to the outcome travellers care most about, on
    exactly the days weather is worst.
  * Times are converted here, correctly. BTS gives local HHMM integers; we hold
    a timezone for every airport, so real UTC instants can be constructed and
    midnight rollover handled rather than worked around. The old file's
    timestamps were converted without either, which is why 3.4% of its flights
    arrived before they departed.
  * Weather is observation, not reanalysis, and includes visibility, ceiling,
    fog and thunderstorm -- the conditions that actually close airports.

Column names match the previous table wherever the meaning is the same, so
retrieval, route planning, evaluation and the application keep working without
change. New columns are additions, never renames.

Writes to `flights_v2` so the existing table stays queryable until the new one
is verified.

Run:  .venv/bin/python src/murphy/build_parquet_bts.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
BTS = ROOT / "data" / "raw" / "bts" / "ontime" / "*.csv"
ISD = ROOT / "data" / "raw" / "isd" / "*.parquet"
TZ = ROOT / "data" / "raw" / "airport_timezones.csv"
# Two artifacts, because they serve different purposes.
#   FULL  -- both years. Model training and the L1/L2 comparison read this. Too
#            large to commit, and rebuildable from the fetch scripts.
#   APP   -- the most recent year only. This is what the application queries and
#            what is committed, so a clone runs and the deployment stays light.
FULL = ROOT / "data" / "processed" / "flights_full"
APP = ROOT / "data" / "processed" / "flights"
APP_YEAR = 2024


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSTALL icu; LOAD icu;")

    con.execute(f"CREATE OR REPLACE VIEW tz AS SELECT airport, timezone FROM read_csv_auto('{TZ}', header=true)")

    # ISD reports more often than hourly -- scheduled observations plus specials
    # issued when conditions change. Collapse to the hour a flight can be joined
    # to, taking the worst of each condition rather than the mean: a traveller
    # cares that visibility dropped during the hour, not that it averaged out.
    con.execute(f"""
        CREATE OR REPLACE VIEW wx AS
        SELECT airport,
               date_trunc('hour', obs_time) AS hour_utc,
               avg(temp_c)                  AS temp_c,
               max(wind_ms)                 AS wind_ms,
               min(visibility_m)            AS visibility_m,
               min(ceiling_m)               AS ceiling_m,
               sum(coalesce(precip_mm, 0))  AS precip_mm,
               max(fog)                     AS fog,
               max(thunderstorm)            AS thunderstorm
        FROM read_parquet('{ISD}')
        GROUP BY 1, 2
    """)

    # BTS gives local clock times as HHMM integers, with 2400 meaning midnight
    # at the end of the day. Build a real local timestamp first, then let the
    # airport's timezone turn it into an instant.
    con.execute(f"""
        CREATE OR REPLACE VIEW raw AS
        SELECT
            FlightDate::DATE                                  AS flight_date,
            Reporting_Airline                                 AS carrier,
            CAST(Flight_Number_Reporting_Airline AS INT)      AS flight_number,
            nullif(trim(Tail_Number), '')                     AS tail_number,
            Origin                                            AS origin,
            Dest                                              AS dest,
            CAST(CRSDepTime AS INT)                           AS crs_dep_hhmm,
            CAST(CRSArrTime AS INT)                           AS crs_arr_hhmm,
            CAST(CRSElapsedTime AS INT)                       AS sched_duration_min,
            CAST(Distance AS INT)                             AS distance_mi,
            CAST(Cancelled AS INT)                            AS cancelled,
            nullif(trim(CancellationCode), '')                AS cancellation_code,
            CAST(Diverted AS INT)                             AS diverted,
            TRY_CAST(ArrDelay AS INT)                         AS arr_delay_min,
            TRY_CAST(DepDelay AS INT)                         AS dep_delay_min,
            TRY_CAST(CarrierDelay AS INT)                     AS delay_carrier_min,
            TRY_CAST(WeatherDelay AS INT)                     AS delay_weather_min,
            TRY_CAST(NASDelay AS INT)                         AS delay_nas_min,
            TRY_CAST(SecurityDelay AS INT)                    AS delay_security_min,
            TRY_CAST(LateAircraftDelay AS INT)                AS delay_late_aircraft_min,
            CAST(Month AS INT)                                AS month,
            CAST(DayOfWeek AS INT)                            AS day_of_week
        FROM read_csv_auto('{BTS}', header=true, union_by_name=true, sample_size=200000)
        WHERE CRSDepTime IS NOT NULL
          AND CRSElapsedTime IS NOT NULL
          -- Three rows in two years carry a negative scheduled duration, which
          -- is a BTS data error rather than anything meaningful.
          AND CAST(CRSElapsedTime AS INT) > 0
          -- One row in two years has no flight number, which makes its record
          -- id null and breaks the key every downstream component relies on.
          AND Flight_Number_Reporting_Airline IS NOT NULL
    """)

    con.execute("""
        CREATE OR REPLACE VIEW timed AS
        SELECT r.*,
               (crs_dep_hhmm // 100) % 24 * 60 + crs_dep_hhmm % 100  AS sched_dep_minutes,
               (crs_dep_hhmm // 100) % 24                            AS sched_dep_hour,
               otz.timezone AS origin_tz,
               dtz.timezone AS dest_tz,
               -- local wall-clock departure, then the same instant in UTC
               (flight_date + INTERVAL 1 MINUTE
                  * ((crs_dep_hhmm // 100) % 24 * 60 + crs_dep_hhmm % 100)) AS dep_local_ts
        FROM raw r
        JOIN tz otz ON otz.airport = r.origin
        JOIN tz dtz ON dtz.airport = r.dest
    """)

    con.execute("""
        CREATE OR REPLACE VIEW utc AS
        SELECT t.*,
               timezone(origin_tz, dep_local_ts)                              AS dep_utc,
               timezone(origin_tz, dep_local_ts) + INTERVAL 1 MINUTE * sched_duration_min
                                                                              AS arr_utc
        FROM timed t
    """)

    FULL.parent.mkdir(parents=True, exist_ok=True)
    for path in (FULL, APP):
        if path.exists():
            shutil.rmtree(path)

    con.execute(f"""
        COPY (
            SELECT
                strftime(u.flight_date, '%Y-%m-%d') || '_' || u.carrier
                    || CAST(u.flight_number AS VARCHAR) || '_' || u.origin || '_' || u.dest
                                                              AS flight_id,
                u.flight_date, u.carrier, u.flight_number, u.tail_number, u.dest,
                u.month, u.day_of_week::TINYINT AS day_of_week,
                u.sched_dep_minutes::SMALLINT AS sched_dep_minutes, u.sched_dep_hour::TINYINT AS sched_dep_hour,
                (u.sched_dep_hour // 2)::TINYINT             AS dep_hour_bin,
                strftime(u.dep_local_ts, '%H:%M')             AS sched_dep_local,
                strftime(u.dep_local_ts + INTERVAL 1 MINUTE * u.sched_duration_min, '%H:%M')
                                                              AS sched_arr_local,
                u.sched_duration_min::SMALLINT AS sched_duration_min, u.distance_mi::SMALLINT AS distance_mi,

                -- real instants, which the previous table could not provide
                u.dep_utc, u.arr_utc, u.origin_tz, u.dest_tz,

                -- outcomes: labels and evidence, never model inputs
                u.arr_delay_min::SMALLINT AS arr_delay_min, u.dep_delay_min::SMALLINT AS dep_delay_min,
                u.cancelled::TINYINT AS cancelled, u.cancellation_code, u.diverted::TINYINT AS diverted,
                u.delay_carrier_min, u.delay_weather_min, u.delay_nas_min,
                u.delay_security_min, u.delay_late_aircraft_min,

                -- observed conditions at each end, joined on the hour of the event
                ow.temp_c::FLOAT AS origin_temp_c, ow.wind_ms::FLOAT AS origin_wind_ms,
                ow.precip_mm::FLOAT AS origin_precip_mm, ow.visibility_m::INT AS origin_visibility_m,
                ow.ceiling_m::INT AS origin_ceiling_m, ow.fog::TINYINT AS origin_fog,
                ow.thunderstorm::TINYINT AS origin_thunderstorm,
                dw.temp_c::FLOAT AS dest_temp_c, dw.wind_ms::FLOAT AS dest_wind_ms,
                dw.precip_mm::FLOAT AS dest_precip_mm, dw.visibility_m::INT AS dest_visibility_m,
                dw.ceiling_m::INT AS dest_ceiling_m, dw.fog::TINYINT AS dest_fog,
                dw.thunderstorm::TINYINT AS dest_thunderstorm,

                u.origin
            FROM utc u
            LEFT JOIN wx ow ON ow.airport = u.origin
                           AND ow.hour_utc = date_trunc('hour', u.dep_utc)
            LEFT JOIN wx dw ON dw.airport = u.dest
                           AND dw.hour_utc = date_trunc('hour', u.arr_utc)
            -- BTS carries a handful of duplicate records, all of them on the
            -- two spring-forward dates in this period. 189 rows out of 13.7
            -- million, but flight_id is used as a key downstream, so it has to
            -- actually be one.
            QUALIFY row_number() OVER (PARTITION BY flight_id ORDER BY u.dep_utc) = 1
            -- Sorting clusters the repeated strings -- airport, carrier, tail --
            -- so they dictionary-compress properly. Worth 34% of the file size.
            ORDER BY u.origin, u.carrier, u.flight_date, u.sched_dep_minutes
        )
        TO '{FULL}'
        (FORMAT PARQUET, PARTITION_BY (month), OVERWRITE_OR_IGNORE,
         COMPRESSION ZSTD, FILENAME_PATTERN 'flights_{{i}}')
    """)

    # The application's copy: one year, same schema, committed to the repository.
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{FULL}/**/*.parquet', hive_partitioning=true)
            WHERE year(flight_date) = {APP_YEAR}
            ORDER BY origin, carrier, flight_date, sched_dep_minutes
        )
        TO '{APP}'
        (FORMAT PARQUET, PARTITION_BY (month), OVERWRITE_OR_IGNORE,
         COMPRESSION ZSTD, FILENAME_PATTERN 'flights_{{i}}')
    """)


def main():
    ap = argparse.ArgumentParser(description="Build the flight table from BTS and NOAA data.")
    ap.parse_args()

    con = duckdb.connect()
    print("building ...")
    build(con)

    for label, path in (("full  ", FULL), ("app   ", APP)):
        con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('{path}/**/*.parquet', hive_partitioning=true)")
        rows = con.execute("SELECT count(*) FROM t").fetchone()[0]
        size = sum(p.stat().st_size for p in path.rglob("*.parquet")) / 1e6
        dupes = con.execute("SELECT count(*) - count(DISTINCT flight_id) FROM t").fetchone()[0]
        print(f"  {label} {rows:>12,} rows  {size:>6,.0f} MB  duplicate ids: {dupes}")

    con.execute(f"CREATE OR REPLACE VIEW out AS SELECT * FROM read_parquet('{FULL}/**/*.parquet', hive_partitioning=true)")
    con.sql("""
        SELECT count(DISTINCT origin) AS airports, count(DISTINCT carrier) AS carriers,
               count(DISTINCT tail_number) AS tails,
               min(flight_date) AS first, max(flight_date) AS last,
               round(100.0 * avg(cancelled), 2) AS pct_cancelled,
               round(100.0 * count(origin_visibility_m) / count(*), 1) AS pct_with_weather
        FROM out
    """).show()


if __name__ == "__main__":
    main()
