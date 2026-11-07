import numpy as np
import pytest
from murphy.forecast.sample import CLASSES
from murphy.recommend import made_it, p_make, should_change


def probs(**kw):
    p = np.zeros((1, len(CLASSES)))
    for k, v in kw.items():
        p[0, CLASSES.index(k.replace("_", "-").removeprefix("d"))] = v
    return p


P = probs(early=0.5, d0_15=0.2, d15_30=0.1, cancelled=0.2)


@pytest.mark.parametrize("slack,want", [(-1, 0.0), (0, 0.5), (14, 0.5), (15, 0.7), (30, 0.8), (29, 0.7)])
def test_chance_rounds_slack_down(slack, want):
    assert p_make(P, [slack])[0] == pytest.approx(want)


def test_cancelled_never_counts_however_long_the_slack():
    assert p_make(P, [10_000])[0] == pytest.approx(0.8)


def test_actual_outcome():
    got = made_it([10, 30, np.nan, np.nan], ["0-15", "30-45", "cancelled", "diverted"], [15, 30, 500, 500])
    assert got.tolist() == [True, False, False, False]


def test_rule_needs_both_conditions():
    # current 0.5 to make, so 0.5 to miss
    assert should_change([0.5], [0.7], 0.4, 0.15)[0]
    assert not should_change([0.5], [0.6], 0.4, 0.15)[0]   # gain too small
    assert not should_change([0.8], [0.99], 0.4, 0.15)[0]  # low risk: arriving sooner is no reason


def test_no_alternative_means_keep():
    assert not should_change([0.1], [np.nan], 0.3, 0.1)[0]
