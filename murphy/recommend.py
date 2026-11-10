"""The keep-or-change rule. Code, not the LLM.

Slack is the minutes between a flight's scheduled arrival and the deadline. The chance of
making it adds the outcome classes that end at or before the slack, rounded down to a
15-minute edge, so a chance is never overstated by rounding. Above 180 minutes the classes
stop, so it adds through 165-180 only.
"""

import numpy as np
import pandas as pd

from murphy.forecast.sample import CLASSES

STEP = 15
MAX_CLASSES = CLASSES.index("165-180") + 1  # early through 165-180
# Chosen on 2025 replay results.
MIN_MISS = 0.50
MIN_GAIN = 0.15
MAX_EARLIER_H = 2  # default; the traveler can change it
# The app's bands. MIN_MISS above stays as the replay's setting.
UNSURE_FROM = 0.40
FIRM_FROM = 0.60


def p_make(p: np.ndarray, slack_min) -> np.ndarray:
    """p: rows of class probabilities in CLASSES order. Negative slack gives 0: the
    'early' class has no detail, so arriving ahead of schedule cannot be counted on."""
    slack = np.asarray(slack_min, dtype=float)
    k = np.minimum(np.floor(slack / STEP).astype(int) + 1, MAX_CLASSES)
    cum = np.concatenate([np.zeros((len(p), 1)), p[:, :MAX_CLASSES].cumsum(axis=1)], axis=1)
    return cum[np.arange(len(p)), np.where(slack < 0, 0, k)]


def made_it(arr_delay_min, outcome, slack_min) -> np.ndarray:
    """What actually happened. Cancelled or diverted never makes it."""
    failed = pd.Series(outcome).isin(["cancelled", "diverted"]).to_numpy()
    delay = np.asarray(arr_delay_min, dtype=float)
    return ~failed & (delay < np.asarray(slack_min, dtype=float))


def should_change(p_make_current, p_make_best_alt, min_miss: float = MIN_MISS,
                  min_gain: float = MIN_GAIN) -> np.ndarray:
    """Change only if missing is likely enough and an alternative lowers it enough. A
    trip with no alternative has a blank best alternative and never changes."""
    cur, alt = np.asarray(p_make_current, dtype=float), np.asarray(p_make_best_alt, dtype=float)
    return ((1 - cur) >= min_miss) & (np.nan_to_num(alt - cur, nan=-1.0) >= min_gain)


def inbound_too_late(delay_so_far, turn_min, slack_min) -> np.ndarray:
    """The traveler's rule of thumb: the inbound aircraft's lateness outlasts its scheduled
    turn by more than the deadline allows. A blank delay or turn says nothing, so never fires."""
    return (np.asarray(delay_so_far, dtype=float) - np.asarray(turn_min, dtype=float)) > slack_min


def verdict(p_make_current: float, p_make_best_alt: float | None, known_cancelled: bool = False) -> str:
    """One trip's advice. A flight known to be cancelled is stated as that, not as a risk.
    A high risk with no flight that clears the gain is not advice to change."""
    if known_cancelled:
        return "cancelled"
    miss = 1 - p_make_current
    if miss < UNSURE_FROM:
        return "keep"
    if miss < FIRM_FROM:
        return "too_uncertain"
    if p_make_best_alt is not None and p_make_best_alt - p_make_current >= MIN_GAIN:
        return "consider_changing"
    return "at_risk_no_better_flight"
