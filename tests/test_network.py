"""Tests for communities, regional inputs and the GCN. Synthetic data, not results.

Cases: window boundaries match features._windowed's rule; a later event changes nothing (leakage);
own airport left out; thin evidence stays blank; test rows stay blank; groups recovered from shared
shocks; a shared daily pattern alone makes no group (failure case); small airports join by flights;
hourly inputs ignore the hour itself; usual comes only from training hours; a GCN with edges beats
the same network with none when a neighbor holds the signal, and does not when none does (simpler
baseline).
"""
import networkx as nx
import numpy as np
import pandas as pd
import torch

from murphy.forecast import gcn, network as nw

T = pd.Timestamp("2025-03-01 12:00")
MIN = lambda m: T - pd.Timedelta(minutes=m)
MEMBER = {"AAA": 0, "BBB": 0, "CCC": 0, "DDD": 1}


def events(*rows):
    return pd.DataFrame(rows, columns=["a", "t", "l"]).astype({"t": "datetime64[ns]"})


def row(origin="AAA", dest="DDD", split="tune", as_of=T):
    return pd.DataFrame({"origin": [origin], "dest": [dest], "split": [split], "as_of_utc": [as_of]}).astype({"as_of_utc": "datetime64[ns]"})


def test_window_keeps_two_hours_back_and_drops_as_of_itself():
    e = events(("BBB", MIN(120), 1), ("BBB", MIN(121), 1), ("BBB", MIN(1), 0), ("BBB", T, 1))
    f, _ = nw.state_inputs(row(), e, MEMBER)
    assert f.comm_o_2h_n[0] == 2 and f.comm_o_2h_share_late[0] == 0.5


def test_a_later_event_changes_nothing():
    e = events(("BBB", MIN(30), 1), ("CCC", MIN(30), 0))
    later = pd.concat([e, events(("BBB", T + pd.Timedelta(minutes=1), 1), ("CCC", T + pd.Timedelta(hours=3), 1))])
    a, n1 = nw.state_inputs(row(), e, MEMBER)
    b, n2 = nw.state_inputs(row(), later, MEMBER)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(n1, n2)


def test_own_airport_is_left_out_but_counts_nationally():
    e = events(("AAA", MIN(30), 1), ("BBB", MIN(30), 0))
    f, n = nw.state_inputs(row(), e, MEMBER)
    assert f.comm_o_2h_n[0] == 1 and f.comm_o_2h_share_late[0] == 0
    assert n.nat_2h_n[0] == 2 and n.nat_2h_share_late[0] == 0.5


def test_thin_evidence_stays_blank_not_calm():
    e = events(("AAA", MIN(30), 1))  # only the origin itself flew
    f, _ = nw.state_inputs(row(), e, MEMBER)
    assert f.comm_o_2h_n[0] == 0 and np.isnan(f.comm_o_2h_share_late[0])
    f, _ = nw.state_inputs(row(origin="ZZZ"), e, MEMBER)  # no community at all
    assert np.isnan(f.comm_o_2h_n[0]) and np.isnan(f.comm_o_2h_share_late[0])


def test_2026_rows_stay_blank():
    f, n = nw.state_inputs(row(split="test"), events(("BBB", MIN(30), 1)), MEMBER)
    assert f.drop(columns="rid").isna().all(axis=None) and n.drop(columns="rid").isna().all(axis=None)


def _panel(shared, seed=0, blocks=800):
    rng = np.random.default_rng(seed)
    a = [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)]
    d = rng.normal(0, 1, (blocks, 10))
    return a, pd.DataFrame(d + shared(rng, blocks), columns=a)


def test_groups_found_from_shared_shocks():
    def shocks(rng, n):
        return np.hstack([np.repeat(rng.normal(0, 2, (n, 1)), 5, 1), np.repeat(rng.normal(0, 2, (n, 1)), 5, 1)])
    a, p = _panel(shocks)
    groups = nw.best_louvain(nw.co_graph(p, a, top=4), seeds=5)
    assert {frozenset(g) for g in groups} == {frozenset(a[:5]), frozenset(a[5:])}


def test_no_shared_shocks_gives_no_real_groups():
    a, p = _panel(lambda rng, n: 0)
    g = nw.co_graph(p, a, top=4)
    # Noise has weak links: nothing like the 0.8 that shared shocks give.
    assert max(d["weight"] for *_, d in g.edges(data=True)) < 0.2


def test_usual_pattern_is_removed_so_it_cannot_link_airports(tmp_path):
    rng = np.random.default_rng(1)
    t = pd.date_range("2024-01-01", periods=2880, freq="3h")
    rows = []
    for a in ("X", "Y"):
        evening = (t.hour >= 18).astype(float) * 0.4
        for when, s in zip(t, evening + rng.normal(0.2, 0.03, len(t))):
            late = int(np.clip(s, 0, 1) * 10)
            rows += [(a, when, 15 if i < late else 0) for i in range(10)]
    df = pd.DataFrame(rows, columns=["origin", "sched_dep_utc", "dep_delay_min"]).assign(cancelled=False)
    df["dep_delay_min"] = df.dep_delay_min.astype("float64")
    df.to_parquet(tmp_path / "f.parquet")
    p = nw.excess_panel([str(tmp_path / "f.parquet")], ["X", "Y"])
    assert abs(p.corr().loc["X", "Y"]) < 0.1  # raw shares would correlate strongly


def test_small_airports_join_the_group_they_fly_to_most():
    routes = pd.DataFrame({"a": ["S", "S", "A", "Q"], "b": ["A", "B", "B", "R"], "n": [5, 9, 100, 7]})
    m = nw.assign_small([{"A"}, {"B"}], routes)
    assert m["S"] == 1 and m["A"] == 0 and "Q" not in m and "R" not in m


def _tables(hours=6, nodes=2):
    hs = pd.date_range("2024-01-01", periods=hours, freq="h")
    z = np.zeros((hours, nodes), dtype="float32")
    return {"hours": hs, "airports": [f"N{i}" for i in range(nodes)], **{k: z.copy() for k in
            ("dep_n", "dep_late", "dep_cxl", "dep_delay", "arr_n", "arr_late", "sch_n", "sch_late")}}


def test_hour_inputs_ignore_the_hour_itself_and_the_future():
    t = _tables()
    t["dep_n"][3, 0], t["dep_late"][3, 0] = 10, 10
    x = gcn.node_inputs(t, gcn.usual_table(t, np.ones(6, bool)))
    i = gcn.INPUTS.index("dep_share_late")
    assert x[3, 0, i] == 0  # happened during hour 3: unknown at its top
    assert x[4, 0, i] == 1 and x[5, 0, i] == 1  # within the 2 h before
    y, w = gcn.targets(_tables())
    assert (w == 0).all()  # no flights: no target


def test_usual_comes_only_from_training_hours():
    t = _tables(hours=48)
    t["sch_n"][:] = 10
    t["sch_late"][24:] = 10  # only the second day is bad
    train = np.arange(48) < 24
    assert gcn.usual_table(t, train).max() == 0
    assert gcn.usual_table(t, ~train).max() == 1


def test_hour_lookup_uses_the_hour_at_or_before_and_blanks_the_rest():
    hs = pd.date_range("2024-01-01", periods=2, freq="h")
    preds = np.array([[.1, .2], [.3, .4]], dtype="float32")
    as_of = pd.Series(pd.to_datetime(["2024-01-01 00:59", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 00:10"]))
    out = gcn.hour_lookup(as_of, pd.Series(["A", "B", "A", "ZZZ"]), preds, hs, ["A", "B"])
    assert np.allclose(out[:2], [.1, .4]) and np.isnan(out[2:]).all()


def test_message_passing_beats_no_edges_only_when_a_neighbor_holds_the_signal():
    rng = np.random.default_rng(0)
    hours, nodes = 1500, 8
    x = rng.normal(0, 1, (hours, nodes, 4)).astype("float32")
    partner = np.arange(nodes) ^ 1
    g = nx.Graph([(f"N{i}", f"N{i ^ 1}") for i in range(0, nodes, 2)])
    for a, b, d in g.edges(data=True):
        d["weight"] = 1.0
    names = [f"N{i}" for i in range(nodes)]
    edges = gcn.edge_tensors(g, names)
    none = gcn.edge_tensors(None, names)
    w = np.ones((hours, nodes), dtype="float32")
    train = np.arange(hours) < 1200

    def err(y, e):
        m, mean, sd = gcn.fit(x, y, w, train, *e, epochs=10)
        return float(((gcn.predict(m, mean, sd, x[~train], *e) - y[~train]) ** 2).mean())

    from_neighbor = (0.5 + 0.1 * x[:, partner, 0]).astype("float32")
    assert err(from_neighbor, edges) < 0.5 * err(from_neighbor, none)
    from_self = (0.5 + 0.1 * x[:, :, 0]).astype("float32")
    assert abs(err(from_self, edges) - err(from_self, none)) < 0.003
