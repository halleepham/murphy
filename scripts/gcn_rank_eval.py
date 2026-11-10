"""The GCN ranking demo: does the GCN pick which airports to retrieve better than simpler orders?
2025 only; nothing reads 2026.

  neighbors  for each big airport and hour, rank its co-disruption neighbors and keep the top 3:
             the airports whose weather the `why` query would fetch. Today's order is link weight.
  national   rank all 100 big airports each hour and keep the top 10.

Each is scored twice: against the true late share (primary, fixed before scoring) and against the
true share minus that airport's usual for the hour and month (secondary). For the second, every
level score is turned into an excess by subtracting the usual.
Results: data/forecast/network/rank_*.csv. Usage: PYTHONPATH=. python scripts/gcn_rank_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import network as S  # noqa: E402  the GCN's own tables, graph and training years
from murphy.forecast import gcn, network as nw, rank  # noqa: E402
from murphy.forecast.score import day_ci  # noqa: E402

log = lambda *a: print(*a, flush=True)
EXAMPLE = ("ATL", "2025-01-10 15:00")  # chosen before scoring: the Atlanta snow day


def inputs():
    _, big = S.load_member()
    t = S.tables(big)
    y, _ = gcn.targets(t)
    n_next = gcn._next(t["sch_n"], gcn.HORIZON_H)
    train = S.train_mask(t["hours"], S.TRAIN_YEARS)
    usual = gcn.node_inputs(t, gcn.usual_table(t, train))[..., gcn.INPUTS.index("usual")]
    dep_n = gcn._past(t["dep_n"], gcn.STATE_H)
    last2 = np.divide(gcn._past(t["dep_late"] + t["dep_cxl"], gcn.STATE_H), dep_n, out=usual.astype("float64"), where=dep_n > 0)
    load = lambda k: np.load(nw.OUT / f"gcn_{k}.npy")
    scores = {"GCN, 3-seed average (the forecast's)": load("co_avg"), "GCN co-disruption, seed 0": load("co_final"),
              "GCN no edges, seed 0": load("none_final"), "usual for the hour": usual, "last 2 h at this airport": last2,
              "random": np.random.default_rng(1).random(y.shape)}
    test = (t["hours"].year == 2025) & ((t["hours"] + pd.Timedelta(hours=gcn.HORIZON_H - 1)).year == 2025)
    ok = n_next >= rank.MIN_DEP
    hours = test & (ok.sum(1) >= rank.MIN_AIRPORTS)
    log(f"2025 hours {test.sum():,}; with >= {rank.MIN_AIRPORTS} airports of >= {rank.MIN_DEP} departures: {hours.sum():,}")
    g = S.graph_for("co", S.TRAIN_YEARS, big)
    return t, big, y, usual, ok, scores, hours, g


def table(days, truth, ok, scores, k, ref):
    rows = []
    res = {name: rank.score_rows(s, truth, ok, k) for name, s in scores.items()}
    keep = ~np.isnan(res[ref][0])
    d = pd.Series(days[keep])
    for name, (hit, level) in res.items():
        assert (np.isnan(hit) == ~keep).all()
        h, lv, r = hit[keep], level[keep], res[ref][0][keep]
        ci, diff = day_ci(pd.Series(h), d, n_boot=500), day_ci(pd.Series(h - r), d, n_boot=500)
        rows.append({"ranking": name, "units": int(keep.sum()), "hit": ci["diff"], "hit_lo": ci["lo"], "hit_hi": ci["hi"],
                     "vs_gcn": diff["diff"], "vs_lo": diff["lo"], "vs_hi": diff["hi"], "picked_true_share": lv.mean()})
    return pd.DataFrame(rows)


def main():
    t, big, y, usual, ok, scores, hours, g = inputs()
    nb, wt = rank.neighbor_table(g, big)
    log(f"neighbors per airport: min {(nb >= 0).sum(1).min()}, median {np.median((nb >= 0).sum(1))}, max {(nb >= 0).sum(1).max()}")
    H = np.flatnonzero(hours)
    day_h = t["hours"][H].floor("D").to_numpy()
    ref = "GCN, 3-seed average (the forecast's)"
    for truth_name, truth, shift in (("late share", y, False), ("late share minus usual", y - usual, True)):
        adj = {k: (v - usual if shift and k != "random" else v) for k, v in scores.items()}
        # Neighbors: one unit per (hour, airport); padding is never eligible.
        nok = rank.by_neighbors(ok[H], nb) & np.tile(nb >= 0, (len(H), 1))
        ns = {k: rank.by_neighbors(v[H], nb) for k, v in adj.items()}
        ns["current order (link weight)"] = np.tile(wt, (len(H), 1))
        nt = table(np.repeat(day_h, len(big)), rank.by_neighbors(truth[H], nb), nok, ns, rank.K, ref)
        # National: one unit per hour.
        na = table(day_h, truth[H], ok[H], {k: v[H] for k, v in adj.items()}, rank.K_NATIONAL, ref)
        for name, d in (("neighbors", nt), ("national", na)):
            d.insert(0, "truth", truth_name)
            out = nw.OUT / f"rank_{name}.csv"
            d.to_csv(out, mode="a" if shift else "w", header=not shift, index=False)
            log(f"\n{name}, truth = {truth_name}, top {rank.K if name == 'neighbors' else rank.K_NATIONAL}")
            log(d.drop(columns="truth").round(4).to_string(index=False))
    example(t, big, y, usual, ok, scores, nb, wt)


def example(t, big, y, usual, ok, scores, nb, wt):
    a, when = EXAMPLE
    h, i = t["hours"].get_loc(pd.Timestamp(when)), big.index(a)
    cols = [j for j in nb[i] if j >= 0]
    d = pd.DataFrame({"airport": [big[j] for j in cols], "link_weight": wt[i, :len(cols)],
                      "gcn": scores["GCN, 3-seed average (the forecast's)"][h, cols], "usual": usual[h, cols],
                      "last_2h": scores["last 2 h at this airport"][h, cols], "true_late_share": y[h, cols],
                      "eligible": ok[h, cols]})
    d.to_csv(nw.OUT / "rank_example.csv", index=False)
    log(f"\nexample: {a} neighbors at {when} UTC (origin's own true share {y[h, i]:.3f})")
    log(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
