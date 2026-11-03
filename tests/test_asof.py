"""Tests for the BTS ingest and the as_of data layer.

Unit tests build a tiny synthetic month in a temp dir, so they run without the
gitignored data/. The rows are test fixtures, not data or results. Tests on
real data skip when data/flights is absent.

Cases: UTC conversion with and without DST, midnight rollover, masking at
three moments, the strict boundary, cancellation, leakage invariance,
two deliberate failures, and a comparison with a naive unmasked query.
"""

import csv
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from murphy.datalayer.asof import ARRIVAL, DEPARTURE, flights_as_of
from murphy.datalayer.bts import arrival_check, ingest_month

UTC = timezone.utc
COLS = [
    "FlightDate", "Marketing_Airline_Network", "Operating_Airline",
    "Flight_Number_Marketing_Airline", "Tail_Number", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "DepDelay",
    "ArrDelay", "Cancelled", "CancellationCode", "Diverted", "CarrierDelay",
    "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "Duplicate",
]


def leg(date, mkt, num, o, d, dep, arr, elapsed, dep_delay="", arr_delay="",
        cancelled=0, code="", diverted=0, op=None, tail="N1", late_aircraft=""):
    return [date, mkt, op or mkt, num, tail, o, d, dep, arr, elapsed, 500,
            dep_delay, arr_delay, cancelled, code, diverted, "", "", "", "", late_aircraft, "N"]


# ORD is UTC-6 in Jan (CST), UTC-5 in Jul (CDT). PHX is UTC-7 all year.
LEGS = [
    # 09:05 CST = 15:05 UTC. 120 min -> 17:05 UTC = 12:05 EST at ATL.
    leg("2023-01-15", "UA", 100, "ORD", "ATL", "0905", "1205", 120, 10, 5),
    # Same clock time in July: 14:05 UTC.
    leg("2023-07-15", "UA", 100, "ORD", "ATL", "0905", "1205", 120, 0, 0),
    # PHX has no DST: 09:05 MST = 16:05 UTC in July.
    leg("2023-07-15", "WN", 200, "PHX", "ORD", "0905", "1405", 180, 0, 0),
    # Overnight: 23:30 CST (05:30 UTC) + 150 min lands 03:00 EST next day.
    leg("2023-01-15", "AA", 300, "ORD", "ATL", "2330", "0300", 150, 20, 15,
        late_aircraft=15),
    # Cancelled, scheduled 12:00 CST = 18:00 UTC.
    leg("2023-01-15", "DL", 400, "ORD", "ATL", "1200", "1500", 120, cancelled=1, code="B"),
    # Real source error (11 Mar 2023, DL32): CRSElapsedTime 186 but the clock
    # times give 126. ArrDelay is measured from CRSArrTime, so it must win.
    leg("2023-03-11", "DL", 32, "DFW", "ATL", "1754", "2100", 186, 0, 0),
    # Spirit is out of scope (0004) and must be dropped.
    leg("2023-01-15", "NK", 500, "ORD", "ATL", "1000", "1300", 120, 0, 0),
]
AIRPORTS = pd.DataFrame({
    "airport": ["ORD", "ATL", "PHX", "DFW"],
    "tz": ["America/Chicago", "America/New_York", "America/Phoenix", "America/Chicago"],
})


def make_month(tmp: Path, legs, airports=AIRPORTS) -> Path:
    csv_path = tmp / "month.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(legs)
    z = tmp / "mkt_2023_01.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(csv_path, "month.csv")
    ap = tmp / "airports.parquet"
    airports.to_parquet(ap, index=False)
    return ingest_month(z, out_dir=tmp / "flights", airports=ap)


@pytest.fixture
def month(tmp_path):
    return make_month(tmp_path, LEGS)


def at(month, ts, start="2023-01-01", end="2023-12-31", **kw):
    return flights_as_of(ts, start, end, source=month.parent, **kw).set_index("flight_id")


def test_utc_conversion_handles_dst_and_zones(month):
    df = at(month, datetime(2023, 1, 1, tzinfo=UTC))
    assert df.loc["2023-01-15:UA100:ORD-ATL:0905", "sched_dep_utc"] == pd.Timestamp("2023-01-15 15:05")
    assert df.loc["2023-07-15:UA100:ORD-ATL:0905", "sched_dep_utc"] == pd.Timestamp("2023-07-15 14:05")
    assert df.loc["2023-07-15:WN200:PHX-ORD:0905", "sched_dep_utc"] == pd.Timestamp("2023-07-15 16:05")


def test_overnight_arrival_rolls_to_next_day_and_matches_source(month):
    df = at(month, datetime(2023, 1, 1, tzinfo=UTC))
    assert df.loc["2023-01-15:AA300:ORD-ATL:2330", "sched_arr_utc"] == pd.Timestamp("2023-01-16 08:00")


def test_wrong_elapsed_time_does_not_shift_arrival(month):
    """Failure case from real data: the arrival clock time wins, and the
    check flags exactly the one inconsistent leg."""
    df = at(month, datetime(2023, 1, 1, tzinfo=UTC))
    # 21:00 EST = 02:00 UTC next day, not 03:00 as departure + 186 min gives.
    assert df.loc["2023-03-11:DL32:DFW-ATL:1754", "sched_arr_utc"] == pd.Timestamp("2023-03-12 02:00")
    flagged = arrival_check(month).fetchall()
    assert [r[0] for r in flagged] == ["2023-03-11:DL32:DFW-ATL:1754"]


def test_out_of_scope_carrier_dropped(month):
    df = at(month, datetime(2023, 1, 1, tzinfo=UTC))
    assert "NK" not in set(df.marketing_carrier)
    assert len(df) == 6


def test_masking_before_during_after(month):
    fid = "2023-01-15:UA100:ORD-ATL:0905"  # dep 15:15 UTC actual, arr 17:10 UTC actual
    before = at(month, datetime(2023, 1, 15, 15, 0, tzinfo=UTC)).loc[fid]
    assert before.status == "scheduled"
    assert before[DEPARTURE + ARRIVAL].isna().all()

    during = at(month, datetime(2023, 1, 15, 16, 0, tzinfo=UTC)).loc[fid]
    assert during.status == "departed"
    assert during.dep_delay_min == 10 and not during.cancelled
    assert during[ARRIVAL].isna().all()

    after = at(month, datetime(2023, 1, 15, 18, 0, tzinfo=UTC)).loc[fid]
    assert after.status == "arrived"
    assert after.arr_delay_min == 5
    assert after.actual_arr_utc == pd.Timestamp("2023-01-15 17:10")


def test_boundary_is_strict(month):
    """An event at exactly as_of is not yet known: 'before' means before."""
    fid = "2023-01-15:UA100:ORD-ATL:0905"
    row = at(month, datetime(2023, 1, 15, 15, 15, tzinfo=UTC)).loc[fid]
    assert row.status == "scheduled" and pd.isna(row.dep_delay_min)


def test_tail_hidden_until_24h_before_departure(month):
    fid = "2023-01-15:UA100:ORD-ATL:0905"  # scheduled 15:05 UTC
    assert pd.isna(at(month, datetime(2023, 1, 14, 15, 0, tzinfo=UTC)).loc[fid].tail_number)
    assert at(month, datetime(2023, 1, 14, 15, 10, tzinfo=UTC)).loc[fid].tail_number == "N1"


def test_tail_filter_cannot_find_a_hidden_tail(month):
    """Searching by tail must not return flights whose tail is still hidden."""
    early = at(month, datetime(2023, 1, 1, tzinfo=UTC), tail="N1")
    assert early.empty
    t = datetime(2023, 1, 15, 6, 0, tzinfo=UTC)  # UA100 within 24 h; AA300 (Jan 16 05:30) too
    found = at(month, t, tail="N1")
    assert set(found.index) == {"2023-01-15:UA100:ORD-ATL:0905", "2023-01-15:AA300:ORD-ATL:2330",
                                "2023-01-15:DL400:ORD-ATL:1200"}


def test_delay_causes_hidden_until_bts_publishes(month):
    """Jan 2023 data counts as published 00:00 UTC on 21 Mar 2023. Before
    that the arrival delay is visible but its cause is not."""
    fid = "2023-01-15:AA300:ORD-ATL:2330"
    before = at(month, datetime(2023, 3, 20, 23, 59, tzinfo=UTC)).loc[fid]
    assert before.arr_delay_min == 15 and pd.isna(before.late_aircraft_delay_min)
    after = at(month, datetime(2023, 3, 21, 0, 1, tzinfo=UTC)).loc[fid]
    assert after.late_aircraft_delay_min == 15


def test_cancellation_hidden_until_scheduled_departure(month):
    fid = "2023-01-15:DL400:ORD-ATL:1200"
    early = at(month, datetime(2023, 1, 15, 17, 59, tzinfo=UTC)).loc[fid]
    assert pd.isna(early.cancelled)  # not False: "not cancelled" would be a leak too
    late = at(month, datetime(2023, 1, 15, 18, 1, tzinfo=UTC)).loc[fid]
    assert late.cancelled and late.status == "cancelled" and late.cancellation_code == "B"


def test_future_outcomes_cannot_change_the_answer(tmp_path):
    """Leakage check: rewrite a flight's outcome after as_of; the view must not move."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    changed = [list(r) for r in LEGS]
    changed[3][11], changed[3][12] = 240, 230  # AA300 (dep 05:30 UTC Jan 16) much later
    ma, mb = make_month(a, LEGS), make_month(b, changed)
    t = datetime(2023, 1, 16, 5, 0, tzinfo=UTC)
    pd.testing.assert_frame_equal(at(ma, t).drop(columns="source_file"),
                                  at(mb, t).drop(columns="source_file"))


def test_naive_as_of_is_rejected(month):
    with pytest.raises(ValueError, match="timezone-aware"):
        flights_as_of(datetime(2023, 1, 15, 12), "2023-01-15", "2023-01-15", source=month.parent)


def test_airport_without_time_zone_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="PHX"):
        make_month(tmp_path, LEGS, airports=AIRPORTS[AIRPORTS.airport != "PHX"])


def test_baseline_naive_query_leaks_and_as_of_does_not(month):
    """Compare with the simpler approach: read the Parquet directly. It shows
    the future arrival delay at a moment before departure; as_of does not."""
    t = datetime(2023, 1, 15, 12, 0, tzinfo=UTC)
    naive = duckdb.sql(f"SELECT flight_id, arr_delay_min FROM read_parquet('{month}') "
                       "WHERE flight_date = DATE '2023-01-15' AND NOT cancelled").df()
    masked = at(month, t, "2023-01-15", "2023-01-15")
    assert naive.arr_delay_min.notna().sum() == 2
    assert masked.arr_delay_min.notna().sum() == 0


REAL = Path(__file__).resolve().parents[1] / "data" / "flights" / "2023_01.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="real BTS data not ingested")
def test_real_ids_unique_and_times_consistent():
    """Every flight_id across all ingested months is non-null and unique
    (provenance). One month's elapsed-time disagreements stay rare."""
    n, distinct = duckdb.sql(
        f"SELECT count(*), count(DISTINCT flight_id) FROM read_parquet('{REAL.parent}/*.parquet')"
    ).fetchall()[0]
    assert n == distinct, f"{n - distinct} legs lack a unique flight_id"
    m = duckdb.sql(f"SELECT count(*) FROM read_parquet('{REAL}')").fetchall()[0][0]
    bad = len(arrival_check(REAL).fetchall())
    assert bad / m < 1e-4, f"{bad} of {m} legs: elapsed time disagrees with clock times"
