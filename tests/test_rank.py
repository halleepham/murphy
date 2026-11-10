"""The ranking demo's scoring."""
import networkx as nx
import numpy as np

from murphy.forecast import rank

TRUTH = np.array([[0.9, 0.7, 0.5, 0.3, 0.1]])
OK = np.ones_like(TRUTH, dtype=bool)


def test_perfect_ranking_hits_every_pick():
    hit, level = rank.score_rows(TRUTH, TRUTH, OK, 2)
    assert hit[0] == 1.0 and np.isclose(level[0], 0.8)


def test_reversed_ranking_misses_every_pick():
    hit, level = rank.score_rows(-TRUTH, TRUTH, OK, 2)
    assert hit[0] == 0.0 and np.isclose(level[0], 0.2)


def test_ties_in_the_truth_do_not_count_against_a_ranking():
    truth = np.array([[0.5, 0.5, 0.5, 0.0]])
    hit, _ = rank.score_rows(np.array([[0.0, 0.1, 0.2, 0.9]]), truth, np.ones_like(truth, bool), 2)
    assert hit[0] == 0.5  # one pick is a tied top airport, the other is the 0.0 one


def test_ineligible_airports_are_never_picked():
    ok = np.array([[False, True, True, True, True]])
    hit, level = rank.score_rows(TRUTH, TRUTH, ok, 2)  # the best airport is too thin to count
    assert hit[0] == 1.0 and np.isclose(level[0], 0.6)


def test_thin_row_is_blank_not_scored():
    ok = np.array([[True, True, False, False, False]])
    hit, level = rank.score_rows(TRUTH, TRUTH, ok, 2)  # only 2 eligible for a top 2: no choice to make
    assert np.isnan(hit[0]) and np.isnan(level[0])


def test_constant_score_ranks_like_random():
    rng = np.random.default_rng(0)
    truth = rng.random((4000, 8))
    ok = np.ones_like(truth, bool)
    const, _ = rank.score_rows(np.zeros_like(truth), truth, ok, 3)
    rand, _ = rank.score_rows(rng.random(truth.shape), truth, ok, 3)
    assert abs(const.mean() - 3 / 8) < 0.02 and abs(rand.mean() - 3 / 8) < 0.02


def test_neighbor_table_is_heaviest_first_and_padded():
    g = nx.Graph()
    g.add_weighted_edges_from([("A", "B", 0.2), ("A", "C", 0.6), ("B", "C", 0.4)])
    g.add_node("D")
    nb, wt = rank.neighbor_table(g, ["A", "B", "C", "D"])
    assert nb[0].tolist() == [2, 1] and wt[0].tolist() == [0.6, 0.2]
    assert nb[3].tolist() == [-1, -1] and wt[3].tolist() == [0, 0]


def test_by_neighbors_lines_up_hours_and_airports():
    a = np.array([[10, 11, 12], [20, 21, 22]])  # 2 hours, 3 airports
    nb = np.array([[1, 2], [0, -1], [0, 1]])
    out = rank.by_neighbors(a, nb)
    assert out.shape == (6, 2)
    assert out[0].tolist() == [11, 12] and out[4].tolist() == [20, 20] and out[5].tolist() == [20, 21]


# --- The app's switch ---

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from murphy import graphrag as gr  # noqa: E402

WHY = [
    {"airport": "ATL", "co_disruption_weight": None, "metar_id": "ATL:1", "as_of": "2025-01-10T15:30Z"},
    {"airport": "CLT", "co_disruption_weight": 1.06, "metar_id": "CLT:1", "as_of": "2025-01-10T15:30Z"},
    {"airport": "RDU", "co_disruption_weight": 1.02, "metar_id": "RDU:1", "as_of": "2025-01-10T15:30Z"},
    {"airport": "BNA", "co_disruption_weight": 0.98, "metar_id": None, "as_of": "2025-01-10T15:30Z"},
    {"airport": "MEM", "co_disruption_weight": 0.82, "metar_id": "MEM:1", "as_of": "2025-01-10T15:30Z"},
    {"airport": "XXX", "co_disruption_weight": 0.80, "metar_id": "XXX:1", "as_of": "2025-01-10T15:30Z"},
    {"airport": "TYS", "co_disruption_weight": 0.70, "metar_id": "TYS:1", "as_of": "2025-01-10T15:30Z"},
]
FAKE = {"CLT": 0.40, "RDU": 0.40, "BNA": 0.99, "MEM": 0.89, "TYS": 0.62}  # XXX has no score


class Fake:
    def __init__(self, rows):
        self.rows = rows

    def run(self, cypher, **_):
        from murphy import graph as kg
        name = next(n for n, (_, cy) in kg.QUERIES.items() if cy == cypher)
        return type("R", (), {"data": lambda _s: [dict(r) for r in self.rows.get(name, [])]})()


@pytest.fixture
def fake_scores(monkeypatch):
    monkeypatch.setattr(rank, "gcn_scores", lambda: ("fake", {}))
    monkeypatch.setattr(rank, "gcn_lookup", lambda a, t: FAKE.get(a))


def test_switch_is_off_by_default_and_evidence_is_unchanged(monkeypatch, fake_scores):
    monkeypatch.delenv("MURPHY_GCN_RANK", raising=False)
    ev = gr.evidence(Fake({"why": WHY}), "why", "F", 2)
    assert [e["airport"] for e in ev] == ["ATL", "CLT", "RDU", "MEM", "XXX", "TYS"]
    assert not any("gcn_late_share_next3h" in e for e in ev)
    assert gr.neighbor_order().startswith("co-disruption link weight")


def test_switch_on_keeps_origin_and_top_3_by_gcn(monkeypatch, fake_scores):
    monkeypatch.setenv("MURPHY_GCN_RANK", "1")
    ev = gr.evidence(Fake({"why": WHY}), "why", "F", 2)
    # BNA scores highest but has no weather, so it is not shown; ties keep link-weight order.
    assert [e["airport"] for e in ev] == ["ATL", "MEM", "TYS", "CLT"]
    assert ev[1]["gcn_late_share_next3h"] == "89%" and "gcn_late_share_next3h" not in ev[0]
    assert gr.neighbor_order().startswith("GCN")


def test_unscored_neighbors_come_last(monkeypatch, fake_scores):
    rows = [WHY[0], WHY[5], WHY[6]]  # XXX (no score) is heavier than TYS
    assert [r["airport"] for r in gr.gcn_neighbors(rows)] == ["ATL", "TYS", "XXX"]


def test_without_score_files_rows_are_unchanged(monkeypatch):
    monkeypatch.setattr(rank, "gcn_scores", lambda: None)
    monkeypatch.setenv("MURPHY_GCN_RANK", "1")
    assert gr.gcn_neighbors(WHY) == WHY
    assert gr.neighbor_order().startswith("co-disruption")


def test_ablation_ignores_the_switch(monkeypatch, fake_scores):
    monkeypatch.setenv("MURPHY_GCN_RANK", "1")
    ev = gr.evidence(Fake({"why": WHY}), "why", "F", 2, communities=False)
    assert [e["airport"] for e in ev] == ["ATL"]


real = pytest.mark.skipif(rank.gcn_scores() is None, reason="GCN arrays are gitignored")


@real
def test_lookup_matches_the_saved_array_and_never_uses_a_later_hour():
    p, ix = rank.gcn_scores()
    h = int((pd.Timestamp("2025-01-10 15:00") - pd.Timestamp(rank.START)) / pd.Timedelta(hours=1))
    assert rank.gcn_lookup("ATL", "2025-01-10T15:59Z") == float(p[h, ix["ATL"]])  # 15:00, not 16:00
    assert rank.gcn_lookup("ATL", "2025-01-10T10:59-05:00") == float(p[h, ix["ATL"]])  # same instant, other zone


@real
def test_lookup_is_blank_outside_the_arrays():
    assert rank.gcn_lookup("ATL", "2026-01-02T12:00Z") is None  # 2026 is sealed and not in the arrays
    assert rank.gcn_lookup("GST", "2025-01-10T15:00Z") is None  # a small airport has no score
