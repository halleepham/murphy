"""Tests for the Step 2 model inputs.

Synthetic legs and weather, built through the real ingest. They are test
fixtures, not data or results. Bulk queries are checked against the as_of
functions the rest of the project trusts.

Cases: rates equal counts from flights_as_of; the 2 h state window and its
strict boundary; the aircraft's prior leg at each lead time, including blank
at 24 h; a later outcome changes nothing (leakage); a missing tail; weather
equals asos_as_of and taf_as_of; TEMPO worst case; a TAF too old; an airport
with no weather stays blank (thin evidence).
"""

from datetime import timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from murphy.datalayer.asof import flights_as_of, taf_as_of
from murphy.forecast import features as ft
from murphy.forecast.sample import build_rows
from tests.test_asof import leg, make_month
from tests.test_weather import ASOS_ROWS, TAF_ROWS, make

UTC = timezone.utc
O_A = ("ORD", "ATL", "0800", "1100", 120)  # dep 14:00 UTC, arr 16:00 UTC in Jan
A_O = ("ATL", "ORD", "1200", "1300", 120)  # dep 17:00 UTC, arr 19:00 UTC in Jan
D10 = "2023-01-10"
LEGS = [
    leg("2023-01-08", "UA", 1, *O_A, 0, -5, tail="N2"),
    leg("2023-01-09", "UA", 2, *O_A, 30, 40, tail="N3"),
    leg("2023-01-09", "DL", 3, *A_O, cancelled=1, code="A", tail="N4"),
    leg(D10, "UA", 4, *O_A, 20, 25, tail="N1"),  # prior leg of UA5
    leg(D10, "UA", 5, *A_O, 10, 10, tail="N1"),
    leg(D10, "AA", 6, "ORD", "ATL", "1000", "1300", 120, 16, 16, tail=""),  # no tail
]
UA5 = "2023-01-10:UA5:ATL-ORD:1200"
AA6 = "2023-01-10:AA6:ORD-ATL:1000"


def setup(tmp, legs=LEGS):
    month = make_month(tmp, legs)
    rows = tmp / "rows.parquet"
    build_rows(month.parent, rows, rate_per_10k=10000)
    r = pd.read_parquet(rows)
    return duckdb.connect(), rows, month.parent, r


def row_of(r, flight, lead):
    return r.index[(r.flight_id == flight) & (r.lead_h == lead)][0]


def test_rates_match_flights_as_of(tmp_path):
    con, rows, fl, r = setup(tmp_path)
    b = ft.rates(con, rows, fl)
    i = row_of(r, UA5, 2)
    known = flights_as_of(r.as_of_utc[i].to_pydatetime().replace(tzinfo=UTC), "2022-01-01",
                          "2023-12-31", source=fl)
    known = known[known.status.isin(["arrived", "cancelled", "diverted"])]
    late = (known.status != "arrived") | (known.arr_delay_min >= 15)
    g = late.mean()
    atl = known.origin == "ATL"
    want = (late[atl].sum() + ft.ALPHA * g) / (atl.sum() + ft.ALPHA)
    assert b.origin_p_late[i] == pytest.approx(want)
    cxl = known.status == "cancelled"
    want = (cxl[atl].sum() + ft.ALPHA * cxl.mean()) / (atl.sum() + ft.ALPHA)
    assert b.origin_p_cxl[i] == pytest.approx(want)


def test_state_window_and_strict_boundary(tmp_path):
    """UA4 leaves ORD at 14:20 UTC, AA6 at 16:16. Exactly 14:20 does not see UA4;
    a minute later does. At 16:20 both are in the window; at 16:21 UA4 is out."""
    con, _, fl, r = setup(tmp_path)
    q = pd.DataFrame({"origin": "ORD", "dest": "ATL", "as_of_utc": pd.to_datetime(
        ["2023-01-10 14:20", "2023-01-10 14:21", "2023-01-10 16:20", "2023-01-10 16:21"])})
    q.to_parquet(tmp_path / "q.parquet")
    c = ft.state(con, tmp_path / "q.parquet", fl)
    assert c.dep_2h_n.tolist()[:2] == [0, 1]
    assert c.dep_2h_share_late[1] == 1 and c.dep_2h_mean_delay[1] == 20
    assert c.dep_2h_n.tolist()[2:] == [2, 1]
    assert np.isnan(c.dep_2h_share_late[0])  # no flights: blank, not zero


def test_cancellation_counts_only_once_known(tmp_path):
    con, _, fl, r = setup(tmp_path)
    q = pd.DataFrame({"origin": "ATL", "dest": "ORD", "as_of_utc": pd.to_datetime(
        ["2023-01-09 17:00", "2023-01-09 17:01"])})
    q.to_parquet(tmp_path / "q.parquet")
    c = ft.state(con, tmp_path / "q.parquet", fl)
    assert c.dep_2h_cxl.tolist() == [0, 1]


def test_prior_leg_changes_with_lead_and_blank_at_24h(tmp_path):
    con, rows, fl, r = setup(tmp_path)
    d = ft.aircraft(con, rows, fl)
    at24, at6, at2 = (row_of(r, UA5, h) for h in (24, 6, 2))
    # Blank at 24 h: the tail is not visible yet.
    assert d.loc[at24, ["prior_turn_min", "prior_status", "prior_delay_so_far"]].isna().all()
    # UA5 leaves 17:00. Prior leg UA4 sched 14:00-16:00, actual 14:20-16:25.
    assert d.prior_turn_min[at6] == 60
    assert d.prior_status[at6] == ft.STATUS["scheduled"]  # as_of 11:00
    assert d.prior_delay_so_far[at6] == 0
    assert d.prior_status[at2] == ft.STATUS["departed"]  # as_of 15:00: UA4 departed 14:20
    # A later as_of shows departure, then arrival.
    q = pd.DataFrame({"flight_id": UA5, "sched_dep_utc": pd.Timestamp("2023-01-10 17:00"),
                      "as_of_utc": pd.to_datetime(["2023-01-10 14:10", "2023-01-10 15:00",
                                                   "2023-01-10 16:30"])})
    q.to_parquet(tmp_path / "q.parquet")
    e = ft.aircraft(con, tmp_path / "q.parquet", fl)
    assert e.prior_status.tolist() == [ft.STATUS["scheduled"], ft.STATUS["departed"], ft.STATUS["arrived"]]
    assert e.prior_delay_so_far.tolist() == [10, 20, 25]  # past schedule; dep delay; arr delay


def test_missing_tail_stays_blank(tmp_path):
    con, rows, fl, r = setup(tmp_path)
    d = ft.aircraft(con, rows, fl)
    assert d.loc[row_of(r, AA6, 2), ["prior_turn_min", "prior_status"]].isna().all()
    assert len(d) == len(r)


def test_later_outcomes_do_not_change_inputs(tmp_path):
    """Change UA5's own outcome and every leg after each forecast: inputs equal."""
    changed = LEGS[:4] + [leg(D10, "UA", 5, *A_O, 90, 95, tail="N1"), LEGS[5]]
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    con, ra, fa, r = setup(tmp_path / "a")
    _, rb, fb, _ = setup(tmp_path / "b", changed)
    i = row_of(r, UA5, 2)
    for fn in (ft.rates, ft.state, ft.aircraft):
        a, b = fn(con, ra, fa), fn(con, rb, fb)
        pd.testing.assert_series_equal(a.loc[i], b.loc[i])


@pytest.fixture
def wx(tmp_path):
    """Synthetic ATL weather from test_weather's raw CSVs: known times 2024-01-10."""
    tempo = ("KATL,2024-01-10 01:53,2024-01-10 04:00,GROUP,True,2024-01-10 06:00,25,240,0,2,"
             "['TSRA'],['BKN'],[900],,,,202401100153-KFFC-FTUS42-TAFATL-P2,Temporary,False")
    rows = [t for t in TAF_ROWS if "Temporary" not in t] + [tempo]
    return make(tmp_path, ASOS_ROWS, rows)


def wx_rows(tmp, origin, dest, as_of, dep, arr):
    q = pd.DataFrame({"origin": [origin], "dest": [dest], "as_of_utc": [pd.Timestamp(as_of)],
                      "sched_dep_utc": [pd.Timestamp(dep)], "sched_arr_utc": [pd.Timestamp(arr)]})
    q.to_parquet(tmp / "q.parquet")
    return tmp / "q.parquet"


def test_observation_strict_lag_and_no_weather(tmp_path, wx):
    con = duckdb.connect()
    # The 02:52 report is known at 03:02. At exactly 03:02 the 01:52 one is latest.
    q = wx_rows(tmp_path, "ATL", "ORD", "2024-01-10 03:02", "2024-01-10 05:00", "2024-01-10 07:00")
    o = ft.observation(con, q, wx)
    assert o.obs_sknt[0] == 12 and o.has_obs[0] == 1 and o.obs_age_min[0] == 70
    # Past the 24 h lookback: blank, like asos_as_of.
    q = wx_rows(tmp_path, "ATL", "ORD", "2024-01-12 03:02", "2024-01-12 05:00", "2024-01-12 07:00")
    o = ft.observation(con, q, wx)
    assert o.has_obs[0] == 0 and np.isnan(o.obs_sknt[0])


def test_uncovered_airport_is_blank_not_calm(tmp_path, wx):
    """Thin evidence: no weather at the airport gives blanks and a 0 flag."""
    con = duckdb.connect()
    q = wx_rows(tmp_path, "XNA", "ORD", "2024-01-10 03:02", "2024-01-10 05:00", "2024-01-10 07:00")
    o = ft.observation(con, q, wx)
    t = ft.taf(con, q, wx, "origin", "sched_dep_utc", "taf_dep")
    assert o.has_obs[0] == 0 and t.taf_dep_has[0] == 0
    assert np.isnan(o.obs_sknt[0]) and np.isnan(t.taf_dep_sknt[0])


def test_taf_matches_taf_as_of_and_takes_worst_group(tmp_path, wx):
    con = duckdb.connect()
    # At 03:00 the latest is P2 (issued 01:53). 05:00 falls in its opening group
    # (17 kt, 6 mi, 3000 ft) and its TEMPO (25 kt, 2 mi, TS, 900 ft): worst of both.
    q = wx_rows(tmp_path, "ATL", "ORD", "2024-01-10 03:00", "2024-01-10 05:00", "2024-01-10 07:00")
    t = ft.taf(con, q, wx, "origin", "sched_dep_utc", "taf_dep")
    got = taf_as_of(pd.Timestamp("2024-01-10 03:00", tz="UTC").to_pydatetime(), "ATL", source=wx)
    assert set(got.product_id.str[-2:]) == {"P2"}
    assert got[got.period_type == "initial"].sknt.iloc[0] == 17
    assert t.taf_dep_sknt[0] == 25 and t.taf_dep_visibility[0] == 2
    assert t.taf_dep_ceiling[0] == 900 and t.taf_dep_ts[0] == 1
    assert t.taf_dep_horizon_h[0] == pytest.approx((5 * 60 - (60 + 53)) / 60)
    # 07:00 is after the TEMPO, under the FM group of the same product (18 kt).
    q = wx_rows(tmp_path, "ATL", "ORD", "2024-01-10 03:00", "2024-01-10 07:00", "2024-01-10 09:00")
    t = ft.taf(con, q, wx, "origin", "sched_dep_utc", "taf_dep")
    assert t.taf_dep_sknt[0] == 18 and t.taf_dep_ts[0] == 0


def test_old_taf_is_blank(tmp_path, wx):
    con = duckdb.connect()
    q = wx_rows(tmp_path, "ATL", "ORD", "2024-01-12 03:00", "2024-01-12 05:00", "2024-01-12 07:00")
    t = ft.taf(con, q, wx, "origin", "sched_dep_utc", "taf_dep")
    assert t.taf_dep_has[0] == 0 and np.isnan(t.taf_dep_sknt[0])


def _aircraft():
    # rows 0-2 normal, 3 overlap, 4 no tail, 5 lead 24 (blank for every flight), 6 prior leg cancelled
    return pd.DataFrame({"prior_turn_min": [60.0, 70, 80, -30, np.nan, np.nan, 50],
                         "prior_status": [0.0, 1, 2, 0, np.nan, np.nan, 3],
                         "prior_delay_so_far": [5.0, 6, 7, 0, np.nan, np.nan, np.nan]})


def _hide(d, **kw):
    from murphy.forecast.features import hide_hindsight
    return hide_hindsight(d, ["tune"] * 7, kw.get("lead", [2, 2, 2, 2, 2, 24, 2]), [[9] * 7], **{k: v for k, v in kw.items() if k == "seed"})


def test_hide_replaces_overlap_and_no_tail_with_a_whole_real_row():
    d = _aircraft()
    out = _hide(d)
    real = set(map(tuple, d.iloc[[0, 1, 2, 6]].fillna(-1).to_numpy()))
    for i in (3, 4):
        assert tuple(out.iloc[i].fillna(-1)) in real
    assert out.prior_turn_min.drop(5).notna().all() and not (out.prior_turn_min < 0).any()


def test_hide_leaves_normal_rows_and_a_cancelled_prior_leg_alone():
    d, out = _aircraft(), _hide(_aircraft())
    pd.testing.assert_frame_equal(out.iloc[[0, 1, 2, 6]], d.iloc[[0, 1, 2, 6]])


def test_hide_leaves_24h_blank():
    assert _hide(_aircraft()).iloc[5].isna().all()


def test_hide_is_repeatable():
    pd.testing.assert_frame_equal(_hide(_aircraft(), seed=3), _hide(_aircraft(), seed=3))


def test_hide_never_borrows_across_splits():
    from murphy.forecast.features import hide_hindsight
    d = pd.DataFrame({"prior_turn_min": [-5.0, 60, 999], "prior_status": [0.0, 1, 2], "prior_delay_so_far": [0.0, 1, 2]})
    out = hide_hindsight(d, ["tune", "tune", "test"], [2, 2, 2], [[9, 9, 9]])
    assert out.prior_turn_min[0] == 60


def test_hide_falls_back_to_another_hour_when_the_hour_is_empty():
    from murphy.forecast.features import hide_hindsight
    d = pd.DataFrame({"prior_turn_min": [-5.0, 60], "prior_status": [0.0, 1], "prior_delay_so_far": [0.0, 1]})
    assert hide_hindsight(d, ["tune", "tune"], [2, 2], [[3, 9]]).prior_turn_min[0] == 60


def test_hide_prefers_the_closest_match():
    from murphy.forecast.features import hide_hindsight
    d = pd.DataFrame({"prior_turn_min": [-5.0, 60, 999], "prior_status": [0.0, 1, 2], "prior_delay_so_far": [0.0, 1, 2]})
    assert hide_hindsight(d, ["tune"] * 3, [2, 2, 2], [["AA", "AA", "UA"]]).prior_turn_min[0] == 60
