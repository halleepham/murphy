"""Ranking airports before retrieval, scored against what happened.

A ranking picks the top K airports by a score. The truth is each airport's late-or-cancelled share
of departures scheduled in the next 3 hours. A pick is a hit if its true share is at least the
K-th largest true share among the eligible airports, so ties in the truth never count against a
ranking. Only airports with at least MIN_DEP scheduled departures in the window are eligible, and a
unit is scored only if it has more than K eligible airports, so there is a real choice to make.
"""
import functools

import numpy as np
import pandas as pd

K = 3            # airports retrieved per question (AI choice, untested)
K_NATIONAL = 10  # the national top list
MIN_DEP = 10     # departures in the 3-hour window for an airport to count (AI choice, untested)
MIN_AIRPORTS = 20  # eligible airports for an hour to count (AI choice, untested)
TIE_SEED = 2022  # its own seed, so tie order cannot match a data generator seeded 0 or 1


def top_k(score: np.ndarray, ok: np.ndarray, k: int, seed: int = TIE_SEED) -> np.ndarray:
    """Column indices of the k highest scores among ok cells, per row. Ties in the score are broken
    by a fixed random order, so a constant score ranks like random."""
    tie = np.random.default_rng(seed).random(score.shape)
    s = np.where(ok, score, -np.inf)
    return np.lexsort((tie, -s), axis=-1)[:, :k]


def score_rows(score, truth, ok, k, seed=TIE_SEED):
    """Per row: hit share of the top k, and the mean true share of the k picked. Blank if the row
    has k or fewer eligible airports."""
    pick = top_k(score, ok, k, seed)
    kth = -np.sort(-np.where(ok, truth, -np.inf), axis=-1)[:, k - 1]
    got = np.take_along_axis(truth, pick, axis=-1)
    valid = ok.sum(-1) > k
    hit = np.where(valid, (got >= kth[:, None]).mean(-1), np.nan)
    level = np.where(valid, got.mean(-1), np.nan)
    return hit, level


def neighbor_table(graph, airports):
    """For each airport, its co-disruption neighbors as column indices (padded with -1) and the link
    weights (padded with 0), heaviest first, the order the `why` query retrieves them in today."""
    ix = {a: i for i, a in enumerate(airports)}
    lists = [sorted(((d["weight"], ix[b]) for b, d in graph[a].items()), reverse=True) for a in airports]
    m = max(len(x) for x in lists)
    nb, wt = np.full((len(airports), m), -1), np.zeros((len(airports), m))
    for i, x in enumerate(lists):
        for j, (w, b) in enumerate(x):
            nb[i, j], wt[i, j] = b, w
    return nb, wt


def by_neighbors(a: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """(hours, airports) -> (hours * airports, neighbors): each airport's neighbors' values per hour."""
    out = a[:, np.maximum(nb, 0)]
    return out.reshape(-1, nb.shape[1])


# --- The app's use: look up the forecast's GCN score for one airport and time. ---

SCORES = "gcn_co_avg.npy"  # the 3-seed average the forecast uses
START, END = "2023-01-01", "2026-01-01"  # the hours the arrays cover (scripts/network.py)


@functools.cache
def gcn_scores():
    """(scores, airport index) or None when the gitignored arrays are not on this machine."""
    from murphy.forecast import network as nw
    path, groups = nw.OUT / SCORES, nw.OUT / "communities.csv"
    if not (path.exists() and groups.exists()):
        return None
    d = pd.read_csv(groups, keep_default_na=False)
    big = d[d.big.astype(str) == "True"].airport.tolist()  # the order the arrays were saved in
    p = np.load(path)
    assert p.shape == (len(pd.date_range(START, END, freq="h", inclusive="left")), len(big))
    return p, {a: i for i, a in enumerate(big)}


def gcn_lookup(airport: str, as_of) -> float | None:
    """The GCN's late share for the next 3 hours, from the top of the hour at or before as_of, so
    nothing after as_of is used (the hour's inputs end before it, gcn._past). None if unknown."""
    s = gcn_scores()
    if s is None or airport not in s[1] or as_of is None:
        return None
    if hasattr(as_of, "to_native"):  # neo4j DateTime
        as_of = as_of.to_native()
    h = hour_index(as_of)
    return float(s[0][h, s[1][airport]]) if 0 <= h < len(s[0]) else None


def hour_index(as_of) -> int:
    """Row of the score arrays for the top of the hour at or before as_of (UTC; naive means UTC)."""
    t = pd.Timestamp(as_of)
    t = (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t).floor("h")
    return int((t - pd.Timestamp(START)) / pd.Timedelta(hours=1))
