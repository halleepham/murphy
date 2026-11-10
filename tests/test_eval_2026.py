"""Tests for the 2026 evaluation script. Synthetic data, not results.

Cases: relabeled 2026 rows get real network inputs while the library still blanks 'test' (the
deliberate failure: blank inputs are refused before scoring); 2026 scores only once; the app's band
edges; the calibration gap; 2025 Jan-Aug keeps only those months; a forecast that knows something
beats climatology on the same rows, and a copy of climatology does not (simpler baseline).
"""
import numpy as np
import pandas as pd
import pytest

from murphy.forecast import network as nw
from murphy.forecast.sample import CLASSES
from scripts import eval_2026 as ev
from tests.test_network import MEMBER, MIN, events, row


def test_relabeled_2026_rows_get_inputs_the_library_would_blank():
    e = events(("BBB", MIN(30), 1), ("CCC", MIN(60), 0))
    sealed, _ = nw.state_inputs(row(split="test"), e, MEMBER)
    opened, _ = nw.state_inputs(ev.unsealed(row(split="test")), e, MEMBER)
    as_2025, _ = nw.state_inputs(row(split="tune"), e, MEMBER)
    assert sealed.drop(columns="rid").isna().all(axis=None)
    assert opened.comm_o_2h_n[0] == 2 and opened.comm_o_2h_share_late[0] == 0.5
    pd.testing.assert_frame_equal(opened, as_2025)  # the same rule as 2025


def test_unsealed_leaves_other_splits_alone():
    r = pd.concat([row(split="fit"), row(split="tune"), row(split="test")], ignore_index=True)
    assert ev.unsealed(r).split.tolist() == ["fit", "tune", "unsealed"]


def test_blank_network_inputs_are_refused():
    """The failure this guards against: final_network scored on 2026 with groups f and m blank."""
    with pytest.raises(SystemExit, match="run 'inputs' first"):
        ev.blank_check({"tune": 0.093, "test": 1.0}, "test")
    ev.blank_check({"tune": 0.093, "test": 0.10}, "test")  # a normal gap passes


def test_2026_is_scored_once(tmp_path):
    p = tmp_path / "scores.csv"
    ev.once(p)
    p.write_text("x")
    with pytest.raises(SystemExit, match="scored once"):
        ev.once(p)


def test_band_edges_match_the_app():
    assert ev.band(np.array([0.0, 0.3999, 0.40, 0.5999, 0.60, 1.0])).tolist() == \
        ["keep", "keep", "too_uncertain", "too_uncertain", "high", "high"]


def test_weighted_gap():
    cal = pd.DataFrame({"n": [100, 300], "mean_pred": [0.1, 0.5], "observed": [0.2, 0.5]})
    assert ev.weighted_gap(cal) == pytest.approx(0.025)


def test_jan_aug_keeps_only_those_months():
    r = pd.DataFrame({"flight_date": pd.to_datetime(["2025-01-01", "2025-08-31", "2025-09-01", "2025-12-31"])})
    assert ev.periods(r, "tune")["jan_aug"].tolist() == [True, True, False, False]
    assert list(ev.periods(r, "test")) == ["all"]  # 2026 is Jan-Aug already


def test_informed_forecast_beats_climatology_and_a_copy_does_not():
    rng = np.random.default_rng(0)
    outcome = pd.Series(rng.choice(["early", "0-15", "cancelled"], 2000, p=[0.5, 0.4, 0.1]))
    clim = np.zeros((2000, len(CLASSES)))
    clim[:, [CLASSES.index("early"), CLASSES.index("0-15"), CLASSES.index("cancelled")]] = [0.5, 0.4, 0.1]
    told = clim.copy()
    cxl = (outcome == "cancelled").to_numpy()
    told[cxl] = 0
    told[cxl, CLASSES.index("cancelled")] = 0.9
    told[cxl, CLASSES.index("early")] = 0.1
    base = ev.brier_rows(clim, outcome).mean()
    assert ev.brier_rows(told, outcome).mean() < base
    assert ev.brier_rows(clim.copy(), outcome).mean() == base


# ---- the advice replay ----

from scripts import eval_2026_replay as rp


def test_ratio_is_blank_without_needless_changes():
    assert rp.ratio(10, 4) == 2.5 and np.isnan(rp.ratio(3, 0))


def test_rates_sit_beside_counts():
    r = rp.add_rates({"trips": 2000, "changes": 10, "rescued": 6, "needless": 2, "worse": 1})
    assert r["changes_per_1000"] == 5 and r["rescued_per_needless"] == 3 and r["worse_share"] == 0.1 and r["changes"] == 10


def _table(lo60=0.5, mm_lo=(0.1, 0.1, 0.1), net60=1.0, rescued=30, needless=10, worse=1, changes=50):
    rows = []
    for buf, mlo in zip((30, 60, 120), mm_lo):
        rows.append({"rule": "murphy_firm60", "buffer": buf, "lo": lo60, "net": net60, "rescued": rescued, "needless": needless,
                     "worse": worse, "changes": changes})
        rows.append({"rule": "thumb", "buffer": buf, "mm_lo": mlo})
    return pd.DataFrame(rows)


def test_r_tests_pass_on_a_clear_win():
    assert all(rp.r_tests(_table(), net_2025_60=1.15).values())


def test_r_tests_fail_when_an_interval_reaches_zero():
    """The deliberate failure: a gain whose interval touches 0, no lead over the thumb at 60 or 120 min, too few rescues."""
    t = rp.r_tests(_table(lo60=0.0, mm_lo=(0.1, 0.0, -0.01), net60=0.5, rescued=15, worse=3), net_2025_60=1.15)
    assert not any(t.values())
