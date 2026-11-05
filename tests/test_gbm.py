"""Tests for the Step 2 gradient-boosted model (0008 Decision 6).

Small synthetic inputs: test fixtures, not data or results.

Cases: probabilities sum to 1 in CLASSES order; a class never seen in the fit
gets 0 and a correct column (failure case); blanks are accepted; early stopping
stays off; the quarter is fixed and by flight; checkpoints match a full predict; cutting to the best round; a split
loads only its own rows; a useful input beats the simpler no-input baseline.
"""

import numpy as np
import pandas as pd
import pytest

from murphy.forecast import gbm
from murphy.forecast.sample import CLASSES
from murphy.forecast.score import multiclass_brier


def data(n=2000, seed=0):
    g = np.random.default_rng(seed)
    x = g.uniform(0, 1, n)
    y = np.where(g.uniform(0, 1, n) < x, "cancelled", "early")
    y = np.where(g.uniform(0, 1, n) < 0.1, "15-30", y)
    X = pd.DataFrame({"carrier": g.integers(0, 3, n).astype(float), "x": x})
    return X, pd.Series(y)


def fitted(rounds=10):
    X, y = data()
    return gbm.make_model(rounds, learning_rate=0.1, max_leaf_nodes=7).fit(X, y), X, y


def test_probabilities_sum_to_one_in_class_order():
    m, X, _ = fitted()
    p = gbm.predict(m, X)
    assert p.shape == (len(X), len(CLASSES))
    np.testing.assert_allclose(p.sum(axis=1), 1)
    hi = X.x > 0.9
    assert p[hi, CLASSES.index("cancelled")].mean() > p[hi, CLASSES.index("early")].mean()


def test_class_absent_from_fit_gets_zero():
    m, X, _ = fitted()
    p = gbm.predict(m, X)
    assert (p[:, CLASSES.index("180+")] == 0).all()
    assert p[:, CLASSES.index("15-30")].mean() == pytest.approx(0.1, abs=0.03)


def test_blanks_are_accepted():
    m, X, _ = fitted()
    X = X.copy()
    X.loc[:50, "x"] = np.nan
    assert not np.isnan(gbm.predict(m, X)).any()


def test_early_stopping_is_off():
    assert gbm.make_model(5).early_stopping is False


def test_checkpoints_match_full_predict():
    m, X, y = fitted(rounds=10)
    s = gbm.staged_scores(m, X, y, every=5)
    assert s.rounds.tolist() == [5, 10]
    assert s.brier_multi.iloc[-1] == pytest.approx(multiclass_brier(gbm.predict(m, X), y))


def test_cutting_to_best_round_equals_a_shorter_fit():
    long, X, y = fitted(rounds=10)
    short, _, _ = fitted(rounds=5)
    long._predictors = long._predictors[:5]
    np.testing.assert_allclose(gbm.predict(long, X), gbm.predict(short, X))


def test_split_loads_only_its_rows_and_blanks_unseen_carrier(tmp_path):
    rows = pd.DataFrame({"flight_id": list("abc"), "lead_h": 2, "origin": "ORD", "dest": "ATL",
                         "outcome": "early", "split": ["fit", "tune", "test"]})
    rows.to_parquet(tmp_path / "r.parquet")
    pd.DataFrame({"rid": range(3), "carrier": ["UA", "NEW", "DL"], "x": [1.0, 2.0, 3.0]}
                 ).to_parquet(tmp_path / "a.parquet")
    Xf, _, carriers = gbm.load_split("fit", "a", tmp_path / "r.parquet", tmp_path)
    Xt, rt, _ = gbm.load_split("tune", "a", tmp_path / "r.parquet", tmp_path, carriers)
    assert carriers == ["UA"] and len(Xt) == 1 and rt.flight_id[0] == "b"
    assert np.isnan(Xt.carrier.iloc[0]) and Xt.x.iloc[0] == 2.0


def test_quarter_is_fixed_and_keeps_a_flights_rows_together():
    ids = pd.Series([f"f{i}" for i in range(4000)] * 3)
    a, b = gbm.in_quarter(ids), gbm.in_quarter(ids)
    assert (a == b).all() and 0.2 < a.mean() < 0.3
    assert (pd.Series(a).groupby(ids).nunique() == 1).all()


def test_load_split_quarter_keeps_only_quarter_rows(tmp_path):
    ids = [f"f{i}" for i in range(400)]
    pd.DataFrame({"flight_id": ids, "lead_h": 2, "origin": "ORD", "dest": "ATL",
                  "outcome": "early", "split": "fit"}).to_parquet(tmp_path / "r.parquet")
    pd.DataFrame({"rid": range(400), "carrier": "UA", "x": np.arange(400.0)}
                 ).to_parquet(tmp_path / "a.parquet")
    X, r, _ = gbm.load_split("fit", "a", tmp_path / "r.parquet", tmp_path, quarter=True)
    assert 0 < len(X) < 400 and len(X) == len(r)
    assert gbm.in_quarter(r.flight_id).all()
    assert X.x.astype(int).tolist() == [int(f[1:]) for f in r.flight_id]


def test_useful_input_beats_simpler_baseline():
    m, _, _ = fitted(rounds=50)
    Xn, yn = data(seed=1)
    share = yn.value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
    base = np.tile(share, (len(yn), 1))  # no inputs: class shares from the same sample
    assert multiclass_brier(gbm.predict(m, Xn), yn) < multiclass_brier(base, yn) - 0.05
