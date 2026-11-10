import numpy as np
import pytest
from murphy.forecast.sample import CLASSES
from murphy.recommend import inbound_too_late, made_it, p_make, should_change, verdict


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


def test_inbound_rule_counts_the_turn():
    # 90 late with a 60-minute turn is 30 past departure: fits a 60-minute deadline, not a 15.
    assert inbound_too_late([90], [60], 15)[0]
    assert not inbound_too_late([90], [60], 60)[0]
    assert not inbound_too_late([120], [600], 60)[0]  # late hours ago, long turn: no reason to change


def test_inbound_rule_blank_never_fires():
    assert not inbound_too_late([np.nan, 120], [60, np.nan], 15).any()


@pytest.mark.parametrize("cur,best,cancelled,want", [
    (0.61, None, False, "keep"),            # 39% risk
    (0.60, 0.99, False, "too_uncertain"),   # exactly 40%
    (0.41, 0.99, False, "too_uncertain"),   # 59%: a good alternative does not firm it up
    (0.40, 0.55, False, "consider_changing"),  # exactly 60% and exactly 15 points
    (0.40, 0.54, False, "at_risk_no_better_flight"),
    (0.10, None, False, "at_risk_no_better_flight"),  # thin evidence: nothing to offer
    (0.90, 0.95, True, "cancelled"),
])
def test_verdict(cur, best, cancelled, want):
    assert verdict(cur, best, cancelled) == want
