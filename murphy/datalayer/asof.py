"""Point-in-time access to flight legs: only what was known at `as_of`.

The replay rule from docs/decisions/0003: pick a past moment as "now" and use
only data available before it. Every data access in murphy goes through here.

What is visible at `as_of` (strictly before it). Owner-approved, 0006:
  schedule   always: carrier, flight number, airports, scheduled times.
  tail       from 24 h before scheduled departure (aircraft get swapped;
             the 24 h is an assumption, not a sourced figure).
  departure  once dep_known_at < as_of: actual_dep_utc, dep_delay_min,
             cancelled, cancellation_code.
  arrival    once arr_known_at < as_of: actual_arr_utc, arr_delay_min,
             diverted.
  causes     the five delay-cause minutes, only once BTS would have
             published that month: 00:00 UTC on the 21st of the second
             month after (observed: Jul 2026 data on 21 Sep, Aug on 19 Oct).
             A live status feed never carries causes.

Departure and arrival become known at their event time, as from a live
status feed. The *_known_at columns are never returned: they would leak the
actual times.

Each row carries flight_id and source_file (provenance) and the as_of used.
"""

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

FLIGHTS = Path(__file__).resolve().parents[2] / "data" / "flights"

SCHEDULE = [
    "flight_id", "source_file", "flight_date", "marketing_carrier",
    "operating_carrier", "flight_number", "origin", "dest",
    "crs_dep_local", "crs_arr_local", "crs_elapsed_min", "distance_mi",
    "sched_dep_utc", "sched_arr_utc",
]
DEPARTURE = ["actual_dep_utc", "dep_delay_min", "cancelled", "cancellation_code"]
ARRIVAL = ["actual_arr_utc", "arr_delay_min", "diverted"]
CAUSES = [
    "carrier_delay_min", "weather_delay_min", "nas_delay_min",
    "security_delay_min", "late_aircraft_delay_min",
]
TAIL_VISIBLE = "sched_dep_utc - INTERVAL 24 HOUR < $t"
PUBLISHED_AT = "date_trunc('month', flight_date) + INTERVAL 2 MONTH + INTERVAL 20 DAY"


def _utc(ts: datetime) -> datetime:
    """Stored times are naive UTC. Refuse naive input: it is ambiguous."""
    if ts.tzinfo is None:
        raise ValueError("as_of must be timezone-aware, e.g. datetime(..., tzinfo=timezone.utc)")
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def flights_as_of(
    as_of: datetime,
    start_date: str,
    end_date: str,
    origin: str | None = None,
    dest: str | None = None,
    tail: str | None = None,
    source: Path | str = FLIGHTS,
) -> pd.DataFrame:
    """Flight legs with flight_date in [start_date, end_date], masked to as_of."""
    t = _utc(as_of)
    dep = [f"CASE WHEN dep_known_at < $t THEN {c} END AS {c}" for c in DEPARTURE]
    arr = [f"CASE WHEN arr_known_at < $t THEN {c} END AS {c}" for c in ARRIVAL]
    causes = [f"CASE WHEN {PUBLISHED_AT} < $t THEN {c} END AS {c}" for c in CAUSES]
    tail_col = f"CASE WHEN {TAIL_VISIBLE} THEN tail_number END AS tail_number"
    status = """CASE
        WHEN arr_known_at < $t AND diverted THEN 'diverted'
        WHEN arr_known_at < $t AND NOT cancelled THEN 'arrived'
        WHEN dep_known_at < $t AND cancelled THEN 'cancelled'
        WHEN dep_known_at < $t THEN 'departed'
        ELSE 'scheduled' END AS status"""
    filters = ["flight_date BETWEEN CAST($start AS DATE) AND CAST($end AS DATE)"]
    params = {"t": t, "start": start_date, "end": end_date, "src": f"{source}/*.parquet"}
    for col, val in (("origin", origin), ("dest", dest)):
        if val is not None:
            filters.append(f"{col} = ${col}")
            params[col] = val
    if tail is not None:  # match only where the tail is already visible
        filters.append(f"tail_number = $tail AND {TAIL_VISIBLE}")
        params["tail"] = tail
    sql = f"""
        SELECT {", ".join(SCHEDULE + [tail_col] + dep + arr + causes)}, {status}, $t AS as_of_utc
        FROM read_parquet($src)
        WHERE {" AND ".join(filters)}
        ORDER BY sched_dep_utc, flight_id
    """
    return duckdb.sql(sql, params=params).df()


# --- Weather (docs/decisions/0007) -------------------------------------------
# ASOS: an observation is visible once known_at = obs time + METAR_LAG < as_of.
# TAF:  a forecast is visible once issued (known_at = issue time) < as_of.
# Only the 30 airports in weather.AIRPORTS have weather. Asking for another
# raises, so missing weather is never mistaken for calm weather.

WEATHER = Path(__file__).resolve().parents[2] / "data" / "weather"


def _check_airport(airport: str) -> None:
    from murphy.datalayer.weather import AIRPORTS
    if airport not in AIRPORTS:
        raise ValueError(f"no weather for {airport}: only the 30 airports in 0007 have it")


def asos_as_of(
    as_of: datetime, airport: str, lookback_hours: int = 24,
    source: Path | str = WEATHER,
) -> pd.DataFrame:
    """Observations at `airport` known before as_of, from the last lookback_hours."""
    _check_airport(airport)
    t = _utc(as_of)
    sql = """
        SELECT *, $t AS as_of_utc FROM read_parquet($src)
        WHERE airport = $a AND known_at < $t
          AND obs_utc >= $t - to_hours(CAST($h AS BIGINT))
        ORDER BY obs_utc
    """
    params = {"t": t, "a": airport, "h": lookback_hours, "src": f"{source}/asos.parquet"}
    return duckdb.sql(sql, params=params).df()


def taf_as_of(as_of: datetime, airport: str, source: Path | str = WEATHER) -> pd.DataFrame:
    """Every period of the latest TAF for `airport` issued before as_of.

    Empty if none was issued in the data window. The caller checks
    issued_utc: an old TAF is weak evidence, and that is the caller's call.
    In 71 of 374,113 products IEM holds two versions of a group; both are
    returned, so the caller sees the conflict.
    """
    _check_airport(airport)
    t = _utc(as_of)
    sql = """
        WITH taf AS (SELECT * FROM read_parquet($src) WHERE airport = $a AND known_at < $t)
        SELECT *, $t AS as_of_utc FROM taf
        WHERE known_at = (SELECT max(known_at) FROM taf)
        ORDER BY period_start_utc, period_type
    """
    params = {"t": t, "a": airport, "src": f"{source}/taf.parquet"}
    return duckdb.sql(sql, params=params).df()
