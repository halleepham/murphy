"""Tests for the IEM weather build and the weather as_of queries.

Unit tests write tiny synthetic IEM CSVs to a temp dir. The rows are test
fixtures, not data or results. The real-data test skips when data/weather
is absent.

Cases: the strict METAR boundary, the lookback window, latest TAF before
as_of, no TAF yet (thin evidence), trace vs missing precipitation, TAF
period labels, leakage invariance, two deliberate failures, and a comparison
with a naive "nearest observation" query, which leaks.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from murphy.datalayer.asof import WEATHER, asos_as_of, taf_as_of
from murphy.datalayer.weather import AIRPORTS, build

UTC = timezone.utc
ASOS_HEAD = (
    "station,valid,tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,skyc1,skyc2,"
    "skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,ice_accretion_1hr,ice_accretion_3hr,"
    "ice_accretion_6hr,peak_wind_gust,peak_wind_drct,peak_wind_time,feel,metar,snowdepth"
)
TAF_HEAD = (
    "station,valid,fx_valid,raw,is_tempo,fx_valid_end,sknt,drct,gust,visibility,"
    "presentwx,skyc,skyl,ws_level,ws_drct,ws_sknt,product_id,ftype,is_amendment"
)


def obs(valid, sknt, p01i="0.00", ice=""):
    return (f"ATL,{valid},50,40,60,240,{sknt},{p01i},29.9,1000,10,,FEW,,,,3000,,,,,"
            f"{ice},,,,,,50,KATL METAR {valid} {sknt}KT,")


def taf(issued, start, ftype, sknt, tag, pid_time=None):
    # Real product_ids start with the issue time, e.g. 202401100153-KFFC-...
    stamp = "".join(c for c in (pid_time or issued) if c.isdigit())
    return (f"KATL,{issued},{start},GROUP {sknt}KT,{ftype == 'Temporary'},,{sknt},240,0,6,"
            f"[],['FEW'],[3000],,,,{stamp}-KFFC-FTUS42-TAFATL-{tag},{ftype},False")


ASOS_ROWS = [
    obs("2024-01-10 00:52", 10),
    obs("2024-01-10 01:52", 12, p01i="T"),
    obs("2024-01-10 02:52", 30, p01i="", ice="T"),  # known at 03:02
]
TAF_ROWS = [
    taf("2024-01-09 23:20", "2024-01-09 23:20", "Observation", 8, "P1"),
    taf("2024-01-09 23:20", "2024-01-10 06:00", "Forecast", 9, "P1"),
    taf("2024-01-10 01:53", "2024-01-10 01:53", "Observation", 17, "P2"),
    taf("2024-01-10 01:53", "2024-01-10 07:00", "Forecast", 18, "P2"),
    taf("2024-01-10 01:53", "2024-01-10 04:00", "Temporary", 25, "P2"),
    taf("2024-01-10 03:30", "2024-01-10 03:30", "Observation", 30, "P3"),
]


def make(tmp: Path, asos=ASOS_ROWS, tafs=TAF_ROWS) -> Path:
    for kind, head, rows in (("asos", ASOS_HEAD, asos), ("taf", TAF_HEAD, tafs)):
        (tmp / "raw" / kind).mkdir(parents=True, exist_ok=True)
        (tmp / "raw" / kind / "ATL_2024.csv").write_text("\n".join([head, *rows]) + "\n")
    build(tmp / "raw", tmp / "out")
    return tmp / "out"


def at(h, m, s=0):
    return datetime(2024, 1, 10, h, m, s, tzinfo=UTC)


@pytest.fixture
def wx(tmp_path):
    return make(tmp_path)


def test_metar_hidden_until_lag_passes(wx):
    # 02:52 obs + 10 min lag = 03:02. Strictly before: hidden at 03:02:00.
    assert asos_as_of(at(3, 2), "ATL", source=wx).obs_utc.max() == datetime(2024, 1, 10, 1, 52)
    assert asos_as_of(at(3, 2, 1), "ATL", source=wx).obs_utc.max() == datetime(2024, 1, 10, 2, 52)


def test_lookback_window(wx):
    got = asos_as_of(at(3, 30), "ATL", lookback_hours=2, source=wx)
    assert list(got.sknt) == [12, 30]  # 00:52 is older than 2 h


def test_trace_is_zero_with_flag_missing_stays_null(wx):
    got = asos_as_of(at(4, 0), "ATL", source=wx).set_index("sknt")
    assert got.loc[12, "p01i"] == 0 and got.loc[12, "p01i_trace"]
    assert got.loc[30, "p01i"] != got.loc[30, "p01i"]  # NaN: missing, not zero
    assert got.loc[30, "ice_1hr"] == 0 and got.loc[30, "ice_1hr_trace"]


def test_latest_taf_before_as_of(wx):
    got = taf_as_of(at(3, 0), "ATL", source=wx)  # P3 (03:30) not issued yet
    assert set(got.product_id.str[-2:]) == {"P2"}
    assert list(got.period_type) == ["initial", "TEMPO", "FM"]
    assert set(taf_as_of(at(3, 30), "ATL", source=wx).product_id.str[-2:]) == {"P2"}  # strict
    assert set(taf_as_of(at(3, 31), "ATL", source=wx).product_id.str[-2:]) == {"P3"}


def test_taf_known_at_later_of_two_issue_times(tmp_path):
    # Real case: LAX, 10 Sep 2023, IEM 17:20 vs product_id 17:40.
    wx = make(tmp_path, tafs=TAF_ROWS[:2] + [
        taf("2024-01-10 01:20", "2024-01-10 01:20", "Observation", 17, "P2", "2024-01-10 01:40")])
    assert set(taf_as_of(at(1, 30), "ATL", source=wx).product_id.str[-2:]) == {"P1"}
    assert set(taf_as_of(at(1, 41), "ATL", source=wx).product_id.str[-2:]) == {"P2"}


def test_no_taf_issued_yet_is_empty(wx):
    # Thin evidence: before the first TAF there is nothing, not a guess.
    assert taf_as_of(at(0, 0) - timedelta(days=1), "ATL", source=wx).empty


def test_future_rows_do_not_change_the_past(tmp_path):
    base = make(tmp_path / "a")
    later = make(tmp_path / "b",
                 asos=ASOS_ROWS + [obs("2024-01-10 03:52", 45)],
                 tafs=TAF_ROWS + [taf("2024-01-10 05:00", "2024-01-10 05:00", "Observation", 40, "P4")])
    t = at(3, 10)
    for q in (asos_as_of, taf_as_of):
        a, b = q(t, "ATL", source=base), q(t, "ATL", source=later)
        assert a.drop(columns="source_file").equals(b.drop(columns="source_file"))


def test_failure_airport_without_weather(wx):
    with pytest.raises(ValueError, match="no weather for CAE"):
        asos_as_of(at(3, 0), "CAE", source=wx)
    with pytest.raises(ValueError, match="no weather for CAE"):
        taf_as_of(at(3, 0), "CAE", source=wx)


def test_failure_naive_datetime(wx):
    with pytest.raises(ValueError, match="timezone-aware"):
        asos_as_of(datetime(2024, 1, 10, 3, 0), "ATL", source=wx)


def test_baseline_nearest_observation_leaks(wx):
    # Simpler baseline: join each moment to the observation nearest in time.
    # At 02:50 it picks the 02:52 report, which did not exist yet.
    naive = duckdb.sql(
        "SELECT sknt FROM read_parquet($p) ORDER BY abs(epoch(obs_utc) - epoch($t)) LIMIT 1",
        params={"p": str(wx / "asos.parquet"), "t": datetime(2024, 1, 10, 2, 50)},
    ).fetchall()[0][0]
    masked = asos_as_of(at(2, 50), "ATL", source=wx).sknt.iloc[-1]
    assert (naive, masked) == (30, 12)


@pytest.mark.skipif(not (WEATHER / "taf.parquet").exists(), reason="data/weather not built")
def test_real_data_coverage_and_issue_times():
    a, t = str(WEATHER / "asos.parquet"), str(WEATHER / "taf.parquet")
    have = {r[0] for r in duckdb.sql(f"SELECT DISTINCT airport FROM '{a}'").fetchall()}
    assert have == set(AIRPORTS)
    have = {r[0] for r in duckdb.sql(f"SELECT DISTINCT airport FROM '{t}'").fetchall()}
    assert have == set(AIRPORTS)
    # known_at is never before either issue time, and every row of one
    # product shares it. The two times disagree rarely (84 products).
    early, split, differ = duckdb.sql(f"""SELECT
        count(*) FILTER (WHERE known_at < issued_utc
            OR strftime(known_at, '%Y%m%d%H%M') < substr(product_id, 1, 12)),
        (SELECT count(*) FROM (SELECT product_id FROM '{t}' GROUP BY 1
                               HAVING count(DISTINCT known_at) > 1)),
        count(DISTINCT product_id) FILTER (
            WHERE strftime(issued_utc, '%Y%m%d%H%M') <> substr(product_id, 1, 12))
          / count(DISTINCT product_id)
        FROM '{t}'""").fetchall()[0]
    assert (early, split) == (0, 0) and differ < 0.001
