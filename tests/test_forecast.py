"""Tests for Step 2's sample, climatology baseline and scores.

Synthetic legs, built through the real BTS ingest. They are test fixtures,
not data or results.

Cases: label boundaries; climatology equals shares counted from
flights_as_of (global and per origin); the strict as_of boundary; leakage
invariance; the trailing window; no history gives no forecast (thin-evidence
failure case); a flight's lead-time rows share a split; Brier at its
extremes; calibration; climatology against a simpler uniform baseline.
"""

from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from murphy.datalayer.asof import flights_as_of
from murphy.forecast import climatology as clim
from murphy.forecast.sample import CLASSES, LEADS_H, build_rows
from murphy.forecast.score import PCOLS, calibration, multiclass_brier, report
from tests.test_asof import leg, make_month

UTC = timezone.utc
O_A = ("ORD", "ATL", "0800", "1100", 120)  # 08:00 CST = 14:00 UTC
A_O = ("ATL", "ORD", "0800", "0900", 120)  # 08:00 EST = 13:00 UTC
LEGS = [
    leg("2023-01-10", "UA", 1, *O_A, 0, -5),
    leg("2023-01-11", "UA", 2, *O_A, 0, 0),
    leg("2023-01-12", "UA", 3, *O_A, 15, 15),
    leg("2023-01-13", "DL", 4, *A_O, 180, 180),
    leg("2023-01-14", "DL", 5, *A_O, cancelled=1, code="A"),
    leg("2023-01-15", "AA", 6, *O_A, 0, "", diverted=1),
    leg("2023-01-16", "AA", 7, *A_O, 179, 179),
    leg("2023-01-17", "UA", 8, *O_A, 30, 30),
]


def py_label(r) -> str:
    """Independent of label_sql, to cross-check it."""
    if r.status == "cancelled":
        return "cancelled"
    if r.status == "diverted":
        return "diverted"
    d = r.arr_delay_min
    if d < 0:
        return "early"
    if d >= 180:
        return "180+"
    a = int(d // 15) * 15
    return f"{a}-{a + 15}"


def build(tmp, legs=LEGS, by=None):
    month = make_month(tmp, legs)
    rows = tmp / "rows.parquet"
    build_rows(month.parent, rows, rate_per_10k=10000)
    out = clim.climatology(rows, tmp / "clim.parquet", by=by, flights=month.parent)
    r = pd.read_parquet(rows).merge(pd.read_parquet(out), on=["flight_id", "lead_h"])
    return month, r


def expected_shares(month, as_of, origin=None):
    known = flights_as_of(as_of.to_pydatetime().replace(tzinfo=UTC), "2022-01-01",
                          "2023-12-31", origin=origin, source=month.parent)
    known = known[known.status.isin(["arrived", "cancelled", "diverted"])]
    labels = known.apply(py_label, axis=1)
    return labels.value_counts().reindex(CLASSES, fill_value=0)


def test_label_boundaries(tmp_path):
    _, r = build(tmp_path)
    got = r.drop_duplicates("flight_id").set_index("flight_id").outcome.to_dict()
    assert got == {
        "2023-01-10:UA1:ORD-ATL:0800": "early", "2023-01-11:UA2:ORD-ATL:0800": "0-15",
        "2023-01-12:UA3:ORD-ATL:0800": "15-30", "2023-01-13:DL4:ATL-ORD:0800": "180+",
        "2023-01-14:DL5:ATL-ORD:0800": "cancelled", "2023-01-15:AA6:ORD-ATL:0800": "diverted",
        "2023-01-16:AA7:ATL-ORD:0800": "165-180", "2023-01-17:UA8:ORD-ATL:0800": "30-45",
    }


def test_rows_three_leads_one_split(tmp_path):
    _, r = build(tmp_path)
    assert len(r) == len(LEGS) * len(LEADS_H)
    assert (r.groupby("flight_id").split.nunique() == 1).all()
    assert set(r.split) == {"fit"}
    assert ((r.sched_dep_utc - r.as_of_utc) == pd.to_timedelta(r.lead_h, unit="h")).all()


def test_global_climatology_matches_flights_as_of(tmp_path):
    month, r = build(tmp_path)
    checked = 0
    for _, row in r[r.n_history > 0].iterrows():
        n = expected_shares(month, row.as_of_utc)
        np.testing.assert_allclose(row[PCOLS].to_numpy(float), (n / n.sum()).to_numpy())
        checked += 1
    assert checked >= 15


def test_origin_climatology_shrinks_toward_global(tmp_path):
    month, r = build(tmp_path, by="origin")
    row = r[(r.flight_id == "2023-01-17:UA8:ORD-ATL:0800") & (r.lead_h == 2)].iloc[0]
    g = expected_shares(month, row.as_of_utc)
    o = expected_shares(month, row.as_of_utc, origin="ORD")
    want = (o + clim.ALPHA * g / g.sum()) / (o.sum() + clim.ALPHA)
    np.testing.assert_allclose(row[PCOLS].to_numpy(float), want.to_numpy())


def test_strict_boundary_excludes_outcome_known_at_as_of(tmp_path):
    """UA1 lands 15:55 UTC on 10 Jan (11:00 EST - 5 min). A forecast exactly
    then must not see it; one a minute later must."""
    month = make_month(tmp_path, LEGS)
    rows = pd.DataFrame({"flight_id": ["x", "y"], "lead_h": [0, 0], "as_of_utc": [
        pd.Timestamp("2023-01-10 15:55"), pd.Timestamp("2023-01-10 15:56")]})
    rows.to_parquet(tmp_path / "r.parquet")
    out = pd.read_parquet(clim.climatology(tmp_path / "r.parquet", tmp_path / "c.parquet",
                                           flights=month.parent))
    assert out.n_history.tolist() == [0, 1]


def test_later_outcomes_do_not_change_earlier_forecasts(tmp_path):
    changed = LEGS[:-1] + [leg("2023-01-17", "UA", 8, *O_A, 0, -20)]
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    _, ra = build(tmp_path / "a")
    _, rb = build(tmp_path / "b", legs=changed)
    early = ra.as_of_utc < pd.Timestamp("2023-01-17 15:00")
    pd.testing.assert_frame_equal(ra[early][PCOLS], rb[early][PCOLS])


def test_trailing_window_drops_old_outcomes(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    _, full = build(tmp_path / "a")
    monkeypatch.setattr(clim, "WINDOW_DAYS", 1)
    _, short = build(tmp_path / "b")
    assert full.n_history.max() == 7  # all but the last leg, before its own forecast
    assert short.n_history.max() <= 2  # one leg a day; arrival times vary


def test_no_history_gives_no_forecast(tmp_path):
    """Thin-evidence failure case: the first legs have nothing before them.
    The baseline must say 'unknown' (NaN), not invent a distribution."""
    _, r = build(tmp_path)
    first = r[r.n_history == 0]
    assert len(first) > 0 and first[PCOLS].isna().all().all()


def test_brier_extremes():
    outcome = pd.Series(["early", "cancelled"])
    perfect = (outcome.to_numpy()[:, None] == np.array(CLASSES)).astype(float)
    assert multiclass_brier(perfect, outcome) == 0
    assert multiclass_brier(perfect[::-1], outcome) == 2


def test_calibration_of_calibrated_forecast():
    rng = np.random.default_rng(0)
    n = 20000
    df = pd.DataFrame({c: 0.0 for c in PCOLS}, index=range(n))
    df["p_cancelled"], df["p_early"] = 0.3, 0.7
    df["outcome"] = np.where(rng.random(n) < 0.3, "cancelled", "early")
    t = calibration(df, "cancelled")
    assert len(t) == 1 and abs(t.observed[0] - 0.3) < 0.02


def uniform_vs_climatology(tmp, legs):
    _, r = build(tmp, legs=legs)
    r = r[r.n_history > 0]
    uniform = r.copy()
    uniform[PCOLS] = 1 / len(CLASSES)
    return report(r).iloc[0].brier_multi, report(uniform).iloc[0].brier_multi


def test_climatology_beats_uniform_baseline(tmp_path):
    """Comparison with a simpler baseline (every class equally likely), on
    a skewed mix like real traffic: most legs arrive 0-15 min late."""
    skewed = [leg(f"2023-01-{d:02d}", "UA", d, *O_A, 0, 5) for d in range(1, 21)]
    skewed[6] = leg("2023-01-07", "UA", 7, *O_A, 0, -5)
    skewed[13] = leg("2023-01-14", "UA", 14, *O_A, cancelled=1, code="A")
    c, u = uniform_vs_climatology(tmp_path, skewed)
    assert c < u


def test_uniform_wins_when_history_is_uninformative(tmp_path):
    """Failure case: all 8 legs have different outcomes, so past shares
    predict nothing and climatology scores worse than uniform. The baseline
    is not good by construction; it needs enough history."""
    c, u = uniform_vs_climatology(tmp_path, LEGS)
    assert c > u
