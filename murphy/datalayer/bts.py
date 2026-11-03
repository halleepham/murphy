"""One BTS monthly zip -> one Parquet file of in-scope flight legs.

Scope (docs/decisions/0004): marketing networks AA DL UA WN AS B6 F9, which
carries their regional partners. Spirit is excluded.

Times. BTS gives local "hhmm" clock times. Here:
  sched_dep_utc = FlightDate + CRSDepTime, read in the origin's time zone
  sched_arr_utc = CRSArrTime, read in the destination's time zone, on the
                  local date nearest to sched_dep_utc + CRSElapsedTime
  actual_*_utc  = sched_*_utc + DepDelay / ArrDelay minutes
BTS measures ArrDelay from CRSArrTime, so arrival is anchored on it.
CRSElapsedTime is sometimes wrong: on 11 Mar 2023 ~120 legs carry +60 min.
`arrival_check` lists legs where the two disagree.

When each outcome becomes known (used by murphy.datalayer.asof). Assumptions:
  dep_known_at: actual departure; for a cancellation, scheduled departure.
  arr_known_at: actual arrival; for a cancellation, scheduled departure;
                for a diversion, the later of scheduled arrival and actual
                departure (BTS gives no time the diversion was known).

Usage: python -m murphy.datalayer.bts data/raw/bts/mkt_2026_01.zip
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
AIRPORTS = ROOT / "data" / "airports.parquet"
OUT_DIR = ROOT / "data" / "flights"
CARRIERS = ("AA", "DL", "UA", "WN", "AS", "B6", "F9")


def _hhmm(col: str) -> str:
    """'0905' -> 545 minutes after midnight."""
    return f"(CAST(substr({col}, 1, 2) AS INT) * 60 + CAST(substr({col}, 3, 2) AS INT))"


SQL = f"""
WITH src AS (
  SELECT * FROM read_csv($csv, sample_size = -1, types = {{
    'CRSDepTime': 'VARCHAR', 'CRSArrTime': 'VARCHAR', 'Tail_Number': 'VARCHAR'}})
  WHERE Marketing_Airline_Network IN {CARRIERS}
),
t AS (
  SELECT s.*,
    timezone('UTC', timezone(o.tz, CAST(s.FlightDate AS TIMESTAMP)
        + to_minutes({_hhmm('s.CRSDepTime')}))) AS sched_dep_utc,
    o.tz AS origin_tz, d.tz AS dest_tz
  FROM src s
  LEFT JOIN read_parquet($airports) o ON o.airport = s.Origin
  LEFT JOIN read_parquet($airports) d ON d.airport = s.Dest
),
-- Arrival is anchored on CRSArrTime, because BTS measures ArrDelay from it.
-- Elapsed time only picks the arrival's local date (k days after FlightDate).
a0 AS (
  SELECT *,
    timezone('UTC', timezone(dest_tz, CAST(FlightDate AS TIMESTAMP)
        + to_minutes({_hhmm('CRSArrTime')}))) AS arr_same_day_utc,
    sched_dep_utc + to_minutes(CAST(CRSElapsedTime AS INT)) AS arr_by_elapsed_utc
  FROM t
),
u AS (
  SELECT *,
    timezone('UTC', timezone(dest_tz, CAST(FlightDate AS TIMESTAMP)
        + to_days(CAST(round(epoch(arr_by_elapsed_utc - arr_same_day_utc) / 86400) AS INT))
        + to_minutes({_hhmm('CRSArrTime')}))) AS sched_arr_utc
  FROM a0
),
v AS (
  SELECT *,
    sched_dep_utc + to_minutes(CAST(DepDelay AS INT)) AS actual_dep_utc,
    sched_arr_utc + to_minutes(CAST(ArrDelay AS INT)) AS actual_arr_utc
  FROM u
)
SELECT
  -- One source row (Aug 2024, F9 MIA-ATL) has no flight number: "NA".
  strftime(FlightDate, '%Y-%m-%d') || ':' || Marketing_Airline_Network
    || coalesce(CAST(Flight_Number_Marketing_Airline AS VARCHAR), 'NA')
    || ':' || Origin || '-' || Dest
    || ':' || CRSDepTime                          AS flight_id,
  $source                                         AS source_file,
  FlightDate                                      AS flight_date,
  Marketing_Airline_Network                       AS marketing_carrier,
  Operating_Airline                               AS operating_carrier,
  Flight_Number_Marketing_Airline                 AS flight_number,
  Tail_Number                                     AS tail_number,
  Origin AS origin, Dest AS dest, origin_tz, dest_tz,
  CRSDepTime AS crs_dep_local, CRSArrTime AS crs_arr_local,
  CAST(CRSElapsedTime AS INT)                     AS crs_elapsed_min,
  Distance                                        AS distance_mi,
  sched_dep_utc, sched_arr_utc, actual_dep_utc, actual_arr_utc,
  DepDelay AS dep_delay_min, ArrDelay AS arr_delay_min,
  Cancelled = 1 AS cancelled, CancellationCode AS cancellation_code,
  Diverted = 1 AS diverted,
  CarrierDelay AS carrier_delay_min, WeatherDelay AS weather_delay_min,
  NASDelay AS nas_delay_min, SecurityDelay AS security_delay_min,
  LateAircraftDelay AS late_aircraft_delay_min,
  Duplicate AS duplicate_flag,
  CASE WHEN Cancelled = 1 THEN sched_dep_utc ELSE actual_dep_utc END AS dep_known_at,
  CASE WHEN Cancelled = 1 THEN sched_dep_utc
       WHEN Diverted = 1 THEN greatest(sched_arr_utc, actual_dep_utc)
       ELSE actual_arr_utc END                    AS arr_known_at
FROM v
"""


def ingest_month(zip_path: Path, out_dir: Path = OUT_DIR, airports: Path = AIRPORTS) -> Path:
    zip_path = Path(zip_path)
    out = Path(out_dir) / (zip_path.stem.removeprefix("mkt_") + ".parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        csv = zf.extract(name, tmp)
        rel = con.sql(SQL, params={"csv": csv, "airports": str(airports),
                                   "source": zip_path.name})
        missing = con.sql(
            "SELECT DISTINCT origin FROM rel WHERE origin_tz IS NULL UNION "
            "SELECT DISTINCT dest FROM rel WHERE dest_tz IS NULL").fetchall()
        if missing:
            raise ValueError(f"No time zone for airports {sorted(m[0] for m in missing)}; "
                             "add them to murphy/datalayer/airports.py before ingesting.")
        n, ids = con.sql("SELECT count(*), count(DISTINCT flight_id) FROM rel").fetchall()[0]
        if n != ids:  # count(DISTINCT) skips NULLs, so this catches both
            raise ValueError(f"{zip_path.name}: {n - ids} legs lack a unique flight_id")
        rel.to_parquet(str(out))
    return out


def arrival_check(parquet: Path):
    """Rows where BTS's CRSElapsedTime disagrees with the departure and
    arrival clock times after time-zone conversion. A few are source errors;
    most of an airport's flights disagreeing means a wrong time zone."""
    return duckdb.sql(f"""
        SELECT flight_id, origin, dest, dest_tz, crs_dep_local, crs_arr_local,
               crs_elapsed_min,
               CAST(epoch(sched_arr_utc - sched_dep_utc) / 60 AS INT) AS derived_min
        FROM read_parquet('{parquet}')
        WHERE CAST(epoch(sched_arr_utc - sched_dep_utc) / 60 AS INT) != crs_elapsed_min
    """)


if __name__ == "__main__":
    for z in sys.argv[1:]:
        p = ingest_month(Path(z))
        bad = len(arrival_check(p).fetchall())
        n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{p}')").fetchall()[0][0]
        print(f"{p.name}: {n:,} legs, {bad} arrival-time mismatches")
