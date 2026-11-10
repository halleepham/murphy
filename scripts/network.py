"""Layer 4: communities and the GCN, each tested with and without.

  communities  find the groups on 2023-24; write communities.csv
  state        groups f and n for the fit and 2025 rows; check against group C on real data
  gcn          train no-edge, routes and co-disruption on 2023-24; compare on 2025 hours
  gnn_inputs <kind> <group>  group g (or h, the no-edge comparison) from one edge kind: cross-fit on 2023-24, so each year's input comes from
               a model that never saw that year; 2025 from a model on both years
  seeds        repeat no-edge and co-disruption with other seeds; sparse routes once
  hourcheck    hourly inputs against group C on real data; wiping the future changes nothing
  scan         leak checks on the new inputs
  fit <name>   one quarter-sized forecast fit: base, n, f, g, h or fg
  compare      all finished quick fits against base, on 2025
  average      group m: co-disruption averaged over three seeds
  fit_full     the one full refit, final_network
  final_compare  final_network against final_clean, after the replay

Nothing here reads a 2026 outcome. 2026 rows stay blank, and January 1 2026 departures are read
only as far as the last 2025 row's forecast time, the same as group C. Results: data/forecast/network/.
Usage: PYTHONPATH=. python scripts/network.py <step> [arg]
"""
import sys, time
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from murphy.forecast import gbm, gcn, network as nw
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.gbm import CLASSES
from murphy.forecast.score import PCOLS, day_ci

ALT = OUT_DIR / "alternatives"
SETS = {"fit": (ALT / "fit_rows.parquet", ALT / "clean" / "fit", ALT / "fit_features"),
        "full": (OUT_DIR / "rows.parquet", ALT / "clean" / "full", OUT_DIR / "features")}
TRAIN_YEARS = (2023, 2024)
ROUNDS = 225  # final_clean's best round
START, END = "2023-01-01", "2026-01-01"
GROUPS = {"base": "abcde", "n": "abcden", "f": "abcdef", "g": "abcdeg", "h": "abcdeh", "i": "abcdei", "k": "abcdek", "fg": "abcdefg", "fm": "abcdefm"}
KINDS = ("none", "routes", "co")
pd.set_option("display.width", 200, "display.max_columns", 40)
log = lambda *a: print(*a, flush=True)


def train_files():
    return nw.files(TRAIN_YEARS)


def communities():
    paths, out = train_files(), []
    big = nw.big_airports(paths)
    yearly = [nw.co_graph(nw.excess_panel(nw.files([y]), big), big) for y in TRAIN_YEARS]
    g = nw.add_graphs(yearly, big)
    groups = nw.best_louvain(g)
    member = nw.assign_small(groups, nw.route_counts(paths))
    vol = nw.duckdb.sql(f"SELECT origin a, count(*) n FROM read_parquet({paths!r}) GROUP BY 1").df().set_index("a").n
    log(f"big airports {len(big)}, groups {len(groups)}, modularity {nw.modularity(g, groups, weight='weight'):.4f}")
    for i, c in enumerate(groups):
        log(f"  {i} big={len(c)} all={sum(v == i for v in member.values())}:", " ".join(sorted(c, key=lambda a: -vol[a])[:14]))
    every = set(vol.index)
    log("airports with departures", len(every), "without a group", sorted(every - set(member)))
    nw.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"airport": list(member), "community": list(member.values()),
                  "big": [a in big for a in member]}).sort_values(["community", "airport"]).to_csv(nw.OUT / "communities.csv", index=False)


def load_member():
    d = pd.read_csv(nw.OUT / "communities.csv", keep_default_na=False)
    return dict(zip(d.airport, d.community)), d[d.big.astype(str) == "True"].airport.tolist()


def link(name):
    out = nw.OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for g in "abcde":
        p = out / f"{g}.parquet"
        if not p.exists():
            p.symlink_to((SETS[name][1] / f"{g}.parquet").resolve())
    return out


def state():
    member, _ = load_member()
    paths = nw.files((2023, 2024, 2025)) + [str(nw.FLIGHTS / "2026_01.parquet")]
    for name, (rows, *_ ) in SETS.items():
        t = time.time()
        r = pd.read_parquet(rows, columns=["origin", "dest", "split", "as_of_utc"])
        before = r[r.split != "test"].as_of_utc.max()
        ev = nw.load_events(paths, before)
        f, n = nw.state_inputs(r, ev, member)
        out = link(name)
        f.to_parquet(out / "f.parquet", index=False)
        n.to_parquet(out / "n.parquet", index=False)
        # Real-data check: the same windows on the origin alone must equal group C's departure count.
        c = pd.read_parquet(SETS[name][2] / "c.parquet", columns=["dep_2h_n"]).dep_2h_n.to_numpy()
        live = np.flatnonzero((r.split != "test").to_numpy())
        pick = np.random.default_rng(0).choice(live, 300_000, replace=False)
        mine, _ = nw.Events(ev.a.to_numpy(), nw.to_ns(ev.t), ev.l).window(r.origin.to_numpy()[pick], nw.to_ns(r.as_of_utc)[pick])
        log(f"{name}: {len(r):,} rows, {time.time() - t:.0f} s; events {len(ev):,}; own-airport window equals group C on {(mine == c[pick]).mean():.6f} of 300,000")
        for k in f.columns[1:]:
            log(f"   {k}: blank {f[k][live].isna().mean():.3f}, mean {f[k][live].mean():.3f}")
        log(f"   nat_2h_share_late: blank {n.nat_2h_share_late[live].isna().mean():.3f}, mean {n.nat_2h_share_late[live].mean():.3f}; test rows all blank {f.iloc[np.setdiff1d(np.arange(len(r)), live), 1:].isna().all(axis=None)}")


def graph_for(kind, years, big):
    if kind == "none":
        return None
    if kind == "co":
        return nw.add_graphs([nw.co_graph(nw.excess_panel(nw.files([y]), big), big) for y in years], big)
    d = nw.route_counts(nw.files(years))
    d = d[d.a.isin(big) & d.b.isin(big) & (d.a != d.b)]
    g = nw.nx.Graph()
    g.add_nodes_from(big)
    if kind == "routes8":  # each airport's 8 busiest routes only, as sparse as the co-disruption graph
        d = d.sort_values("n", ascending=False).groupby("a").head(8)
    for x in d.groupby([d[["a", "b"]].min(axis=1), d[["a", "b"]].max(axis=1)]).n.sum().items():
        g.add_edge(*x[0], weight=float(np.log1p(x[1])))
    return g


def train_mask(hours, years):
    """Hours whose whole target falls inside these years, so no fold's target reaches another year."""
    return np.isin(hours.year, years) & np.isin((hours + pd.Timedelta(hours=gcn.HORIZON_H - 1)).year, years)


def run_fold(t, y, w, kind, years, big, tag, seed=0):
    train = train_mask(t["hours"], years)
    x = gcn.node_inputs(t, gcn.usual_table(t, train))
    e = gcn.edge_tensors(graph_for(kind, years, big), big)
    t0 = time.time()
    m, mean, sd = gcn.fit(x, y, w, train, *e, seed=seed, log=lambda s: log(f"  {tag} {s}"))
    log(f"{tag}: {train.sum():,} hours, {e[0].shape[1]} directed edges, {time.time() - t0:.0f} s")
    return gcn.predict(m, mean, sd, x, *e)


def tables(big):
    return gcn.hourly(nw.files((2023, 2024, 2025)), big, START, END)


def gcn_compare():
    _, big = load_member()
    t = tables(big)
    y, w = gcn.targets(t)
    test = (t["hours"].year == 2025) & ((t["hours"] + pd.Timedelta(hours=gcn.HORIZON_H - 1)).year == 2025)
    train = train_mask(t["hours"], TRAIN_YEARS)
    days = pd.Series(np.repeat(t["hours"].floor("D")[test], len(big)))
    wt = w[test].ravel()
    err = lambda p: ((p[test] - y[test]) ** 2).ravel()
    log(f"2025 node-hours with a target: {(wt > 0).sum():,} of {len(wt):,}; train hours {train.sum():,}")
    usual = gcn.node_inputs(t, gcn.usual_table(t, train))[..., gcn.INPUTS.index("usual")]
    dep_n = gcn._past(t["dep_n"], gcn.STATE_H)
    now = np.divide(gcn._past(t["dep_late"] + t["dep_cxl"], gcn.STATE_H), dep_n, out=usual.copy().astype("float64"), where=dep_n > 0)
    res = {"usual for the hour": usual, "last 2 h at this airport": now}
    for k in KINDS:
        p = run_fold(t, y, w, k, TRAIN_YEARS, big, f"final_{k}")
        np.save(nw.OUT / f"gcn_{k}_final.npy", p)
        res[f"GCN {k}"] = p
    rows, ref = [], err(res["GCN none"])
    for name, p in res.items():
        e = err(p)
        ci = day_ci(pd.Series((e - ref) * wt / wt.mean()), days, n_boot=500)
        rows.append({"model": name, "mse": (e * wt).sum() / wt.sum(), "vs_no_edges": ci["diff"], "lo": ci["lo"], "hi": ci["hi"]})
    out = pd.DataFrame(rows)
    out.to_csv(nw.OUT / "gcn_compare.csv", index=False)
    log(out.round(6).to_string(index=False))


def gnn_inputs(kind, group="g"):
    member, big = load_member()
    t = tables(big)
    y, w = gcn.targets(t)
    p = np.load(nw.OUT / f"gcn_{kind}_final.npy").copy()  # the model on both years, right for 2025
    for train_y, predict_y in ((2024, 2023), (2023, 2024)):
        q = run_fold(t, y, w, kind, (train_y,), big, f"cross_{kind}_{train_y}")
        h = t["hours"].year == predict_y
        p[h] = q[h]
    np.save(nw.OUT / f"gcn_{kind}_crossfit.npy", p)
    write_group(p, t, big, group)


def write_group(p, t, big, group):
    for name, (rows, *_) in SETS.items():
        r = pd.read_parquet(rows, columns=["origin", "dest", "split", "as_of_utc"])
        live = (r.split != "test").to_numpy()
        g = {"rid": np.arange(len(r))}
        for side, col in (("o", "origin"), ("d", "dest")):
            v = gcn.hour_lookup(r.as_of_utc, r[col], p, t["hours"], big)
            v[~live] = np.nan
            g[f"gcn_{side}_next3h"] = v
        out = link(name)
        pd.DataFrame(g).to_parquet(out / f"{group}.parquet", index=False)
        log(f"{name}: origin blank {np.isnan(g['gcn_o_next3h'][live]).mean():.3f}, dest blank {np.isnan(g['gcn_d_next3h'][live]).mean():.3f}, mean {np.nanmean(g['gcn_o_next3h'][live]):.4f}")


def seeds():
    """Is the difference between edge kinds more than the luck of one training run? Seed 1 also
    becomes groups i (co) and k (no edges), so the forecast comparison can be repeated."""
    _, big = load_member()
    t = tables(big)
    y, w = gcn.targets(t)
    test = (t["hours"].year == 2025) & ((t["hours"] + pd.Timedelta(hours=gcn.HORIZON_H - 1)).year == 2025)
    wt = w[test].ravel()
    log(f"true 2025 late share {(y[test].ravel() * wt).sum() / wt.sum():.5f}")
    rows = []
    def score(kind, seed, p):
        e = ((p[test] - y[test]) ** 2).ravel()
        rows.append({"kind": kind, "seed": seed, "mse": (e * wt).sum() / wt.sum(), "mean_pred": (p[test].ravel() * wt).sum() / wt.sum()})
        log(rows[-1])
    for kind in ("none", "co"):
        score(kind, 0, np.load(nw.OUT / f"gcn_{kind}_final.npy"))
    for kind, seed in (("none", 1), ("none", 2), ("co", 1), ("co", 2), ("routes8", 0)):
        p = run_fold(t, y, w, kind, TRAIN_YEARS, big, f"final_{kind}_s{seed}", seed)
        score(kind, seed, p)
        if seed == 1:
            for train_y, predict_y in ((2024, 2023), (2023, 2024)):
                q = run_fold(t, y, w, kind, (train_y,), big, f"cross_{kind}_{train_y}_s1", 1)
                h = t["hours"].year == predict_y
                p[h] = q[h]
            write_group(p, t, big, "i" if kind == "co" else "k")
    pd.DataFrame(rows).to_csv(nw.OUT / "gcn_seeds.csv", index=False)


def average():
    """Group m: co-disruption averaged over three seeds, since one seed's level moves by up to 3 points."""
    _, big = load_member()
    t = tables(big)
    y, w = gcn.targets(t)
    all_p = [np.load(nw.OUT / "gcn_co_crossfit.npy")]
    for seed in (1, 2):
        p = run_fold(t, y, w, "co", TRAIN_YEARS, big, f"avg_final_s{seed}", seed)
        for train_y, predict_y in ((2024, 2023), (2023, 2024)):
            q = run_fold(t, y, w, "co", (train_y,), big, f"avg_cross_{train_y}_s{seed}", seed)
            h = t["hours"].year == predict_y
            p[h] = q[h]
        all_p.append(p)
    p = np.mean(all_p, axis=0)
    np.save(nw.OUT / "gcn_co_avg.npy", p)
    write_group(p, t, big, "m")


def alt_inputs(rows, feats):
    """Groups f and m for a replay's alternative flights, by the same rules as the trips."""
    member, big = load_member()
    r = pd.read_parquet(rows, columns=["origin", "dest", "split", "as_of_utc"])
    paths = nw.files((2023, 2024, 2025)) + [str(nw.FLIGHTS / "2026_01.parquet")]
    f, _ = nw.state_inputs(r, nw.load_events(paths, r.as_of_utc.max()), member)
    f.to_parquet(feats / "f.parquet", index=False)
    t = {"hours": pd.date_range(START, END, freq="h", inclusive="left")}
    p = np.load(nw.OUT / "gcn_co_avg.npy")
    pd.DataFrame({"rid": np.arange(len(r)), "gcn_o_next3h": gcn.hour_lookup(r.as_of_utc, r.origin, p, t["hours"], big),
                  "gcn_d_next3h": gcn.hour_lookup(r.as_of_utc, r.dest, p, t["hours"], big)}).to_parquet(feats / "m.parquet", index=False)


def fit_full():
    """The one full refit, same rows, rounds and settings as final_clean, plus groups f and m."""
    path = ALT / "final_network.joblib"
    assert not path.exists(), "already fitted"
    X, r, carriers = gbm.load_split("fit", GROUPS["fm"], rows=SETS["fit"][0], features=nw.OUT / "fit")
    assert not (X.prior_turn_min < 0).any()
    t = time.time()
    n_rows, n_cols = X.shape
    m = gbm.make_model(gbm.MAX_ROUNDS, **gbm.REFIT).fit(X, r.outcome)
    joblib.dump({"model": m, "carriers": carriers, "groups": GROUPS["fm"]}, path)
    log(f"final_network: {n_rows:,} rows, {n_cols} inputs, largest leaf {gbm.largest_leaf(m):.3f}, fit {time.time() - t:.0f} s")


def final_compare():
    """final_network against final_clean on every 2025 forecast, and the replay's net benefit."""
    keys = ["flight_id", "lead_h"]
    new, old = (pd.read_parquet(ALT / d / "current.parquet", columns=[*keys, "flight_date", "outcome", *PCOLS]) for d in ("replay_final_network", "replay_final_clean"))
    m = new.merge(old, on=[*keys, "flight_date", "outcome"], suffixes=("", "_old"))
    assert len(m) == len(new) == len(old)
    y = pd.get_dummies(m.outcome).reindex(columns=CLASSES, fill_value=0).to_numpy(dtype="float32")
    late = m.outcome.isin(CLASSES[2:]).to_numpy()
    multi = lambda c: ((m[c].to_numpy(dtype="float32") - y) ** 2).sum(axis=1)
    lt = lambda c: (m[c[2:]].sum(axis=1).to_numpy() - late) ** 2
    oldc = [f"{c}_old" for c in PCOLS]
    rows = []
    for name, w in [("all", np.ones(len(m), bool))] + [(f"{h} h", (m.lead_h == h).to_numpy()) for h in sorted(m.lead_h.unique())]:
        a, b = multi(PCOLS)[w], multi(oldc)[w]
        ci = day_ci(pd.Series(a - b), m.flight_date[w], n_boot=500)
        cl = day_ci(pd.Series(lt(PCOLS)[w] - lt(oldc)[w]), m.flight_date[w], n_boot=500)
        rows.append({"group": name, "forecasts": int(w.sum()), "new": a.mean(), "final_clean": b.mean(), "diff": ci["diff"], "lo": ci["lo"], "hi": ci["hi"],
                     "late_diff": cl["diff"], "late_lo": cl["lo"], "late_hi": cl["hi"], "pred_late": m.loc[w, PCOLS[2:]].sum(axis=1).mean(), "pred_late_old": m.loc[w, oldc[2:]].sum(axis=1).mean(), "true_late": late[w].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(nw.OUT / "final_compare.csv", index=False)
    log(t.round(5).to_string(index=False))
    a, b = (pd.read_csv(ALT / d / "baselines.csv") for d in ("replay_final_network", "replay_final_clean"))
    for name, x in (("final_network", a), ("final_clean", b)):
        log(name); log(x[(x.scope == "all") & (x.rule == "murphy")][["buffer", "changes", "rescued", "needless", "worse", "both_missed", "net", "lo", "hi", "flagged"]].round(3).to_string(index=False))


def hourcheck():
    """Real data. The hourly inputs must match group C at top-of-hour forecasts, and wiping everything
    after an hour must not change that hour's inputs."""
    _, big = load_member()
    t = tables(big)
    rows, _, raw = SETS["full"]
    r = pd.read_parquet(rows, columns=["origin", "split", "as_of_utc"])
    c = pd.read_parquet(raw / "c.parquet", columns=["dep_2h_n", "dep_2h_share_late", "dep_2h_cxl", "arr_2h_n"])
    a = {x: i for i, x in enumerate(big)}
    pos = pd.Series(np.arange(len(t["hours"])), index=t["hours"])
    pick = (r.split == "tune") & (r.as_of_utc.dt.minute == 0) & (r.as_of_utc.dt.second == 0) & r.origin.isin(big) & (r.as_of_utc < END)
    i = np.flatnonzero(pick.to_numpy())
    h, n = pos.reindex(r.as_of_utc.iloc[i]).to_numpy().astype(int), r.origin.iloc[i].map(a).to_numpy()
    dep = gcn._past(t["dep_n"], gcn.STATE_H)[h, n]
    late = gcn._past(t["dep_late"], gcn.STATE_H)[h, n]
    log(f"top-of-hour rows {len(i):,}; dep count equals group C {(dep == c.dep_2h_n.iloc[i].to_numpy()).mean():.6f}")
    s = c.dep_2h_share_late.iloc[i].to_numpy()
    ok = dep > 0
    log(f"late share within 1e-5 of group C {(np.abs(late[ok] / dep[ok] - s[ok]) < 1e-5).mean():.6f}")
    train = train_mask(t["hours"], TRAIN_YEARS)
    u = gcn.usual_table(t, train)
    x0 = gcn.node_inputs(t, u)
    for T in (int(np.flatnonzero(t["hours"] == pd.Timestamp("2025-07-15 18:00"))[0]), int(np.flatnonzero(t["hours"] == pd.Timestamp("2025-01-10 06:00"))[0])):
        wiped = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()}
        for k in ("dep_n", "dep_late", "dep_cxl", "dep_delay", "arr_n", "arr_late", "sch_late"):
            wiped[k][T:] = 0
        x1 = gcn.node_inputs(wiped, u)
        log(f"hour {t['hours'][T]}: inputs unchanged after wiping the future: {np.array_equal(x0[T], x1[T])}")
    # The own-airport control for the top-tenth table in scan.
    keep = np.flatnonzero((r.split == "tune").to_numpy() & (pd.read_parquet(rows, columns=["lead_h"]).lead_h < 24).to_numpy())
    keep = np.random.default_rng(0).choice(keep, 500_000, replace=False)
    o = pd.read_parquet(rows, columns=["outcome"]).outcome.iloc[keep]
    bad, lt = o.isin(["cancelled", "diverted"]).to_numpy(), o.isin(CLASSES[2:]).to_numpy()
    p = pd.read_parquet(nw.OUT / "quick_base.parquet").to_numpy()
    tune = np.flatnonzero((r.split == "tune").to_numpy())
    m = pd.Series(np.arange(len(tune)), index=tune).reindex(keep).to_numpy()
    def line(name, v):
        hi = v >= np.nanquantile(v, .9)
        log(f"  {name:32s} top10-failed {bad[hi].mean():.4f} top10-late {lt[hi].mean():.4f}")
    line("base forecast's own P(late)", p[m, 2:].sum(axis=1))
    line("group C: origin late share", c.dep_2h_share_late.iloc[keep].to_numpy(dtype=float))
    line("GCN co, origin", pd.read_parquet(nw.OUT / "full" / "g.parquet", columns=["gcn_o_next3h"]).gcn_o_next3h.iloc[keep].to_numpy(dtype=float))


def scan(n=500_000):
    """Would a leak show? Cancelled or diverted share in the top tenth of each new input, at 2 and 6 h,
    against a control that peeks: the same window moved 2 h ahead. Then the hindsight probe of refit_clean.py."""
    member, _ = load_member()
    rows, _, _ = SETS["full"]
    r = pd.read_parquet(rows, columns=["origin", "split", "as_of_utc", "outcome", "lead_h"])
    keep = np.flatnonzero((r.split == "tune").to_numpy() & (r.lead_h < 24).to_numpy())
    keep = np.random.default_rng(0).choice(keep, min(n, len(keep)), replace=False)
    d = nw.OUT / "full"
    X = pd.concat([pd.read_parquet(d / f"{g}.parquet").drop(columns="rid").iloc[keep].reset_index(drop=True) for g in "fng"], axis=1)
    bad = r.outcome.iloc[keep].isin(["cancelled", "diverted"]).to_numpy()
    late = r.outcome.iloc[keep].isin(CLASSES[2:]).to_numpy()
    log(f"overall: failed {bad.mean():.4f}, late {late.mean():.4f}, rows {len(keep):,}")
    def line(name, v):
        v = np.asarray(v, dtype=float)
        hi = v >= np.nanquantile(v, .9)
        log(f"  {name:28s} blank {np.isnan(v).mean():.3f} blank-failed {bad[np.isnan(v)].mean() if np.isnan(v).any() else float('nan'):.4f} top10-failed {bad[hi].mean():.4f} top10-late {late[hi].mean():.4f}")
    for c in X.columns:
        line(c, X[c])
    ev = nw.load_events(nw.files((2023, 2024, 2025)), r.as_of_utc[r.split == "tune"].max() + pd.Timedelta(hours=4))
    ee = ev[ev.a.map(member).notna()]
    e = nw.Events(ee.a.map(member).astype(int).to_numpy(), nw.to_ns(ee.t), ee.l)
    g = r.origin.iloc[keep].map(member)
    ok = g.notna().to_numpy()
    a, b = e.window(g[ok].astype(int).to_numpy(), nw.to_ns(r.as_of_utc.iloc[keep])[ok] + nw.WINDOW_NS)
    peek = np.full(len(keep), np.nan)
    peek[ok] = np.where(a > 0, b / np.maximum(a, 1), np.nan)
    line("CONTROL peeks 2 h ahead", peek)
    # Hindsight probe: can hidden flights be told apart better once the new inputs are in? Fit rows, under 24 h.
    fit_rows, clean, raw = SETS["fit"]
    d0 = pd.read_parquet(raw / "d.parquet", columns=["prior_turn_min"]).prior_turn_min
    a0 = pd.read_parquet(raw / "a.parquet", columns=["lead_h"]).lead_h
    hid = ((a0 < 24) & (d0.isna() | (d0 < 0))).to_numpy()
    base, rf, _ = gbm.load_split("fit", rows=fit_rows, features=nw.OUT / "fit")
    extra, _, _ = gbm.load_split("fit", "afng", rows=fit_rows, features=nw.OUT / "fit")
    extra = extra[["comm_o_2h_n", "comm_o_2h_share_late", "comm_d_2h_n", "comm_d_2h_share_late", "nat_2h_n", "nat_2h_share_late", "gcn_o_next3h", "gcn_d_next3h"]]
    short = base.lead_h.to_numpy() < 24
    y = hid[short]
    base, extra = base[short].reset_index(drop=True), extra[short].reset_index(drop=True)
    rng = np.random.default_rng(0)
    i = np.unique(np.concatenate([rng.choice(len(y), min(400_000, len(y)), replace=False), np.flatnonzero(y)]))
    i = rng.permutation(i)
    tr, te = i[:len(i) * 7 // 10], i[len(i) * 7 // 10:]
    log(f"hidden flights in probe {y[i].sum():,} of {len(i):,}")
    for name, Z in (("no aircraft inputs (floor)", base.drop(columns=["prior_turn_min", "prior_status", "prior_delay_so_far"])), ("hidden", base), ("hidden + new inputs", pd.concat([base, extra], axis=1))):
        m = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=0).fit(Z.iloc[tr], y[tr])
        log(f"  {name:30s} AUC {roc_auc_score(y[te], m.predict_proba(Z.iloc[te])[:, 1]):.4f}")


def fit(name):
    path = nw.OUT / f"quick_{name}.parquet"
    if path.exists():
        log(name, "done"); return
    groups = GROUPS[name]
    t = time.time()
    X, r, carriers = gbm.load_split("fit", groups, rows=SETS["fit"][0], features=nw.OUT / "fit", quarter=True)
    assert not (X.prior_turn_min < 0).any()
    n_rows, n_cols = X.shape
    m = gbm.make_model(ROUNDS, **gbm.REFIT).fit(X, r.outcome)
    fit_s = time.time() - t
    del X, r
    Xt, rt, _ = gbm.load_split("tune", groups, rows=SETS["full"][0], features=nw.OUT / "full", carriers=carriers)
    pd.DataFrame(gbm.predict(m, Xt), columns=PCOLS).astype("float32").to_parquet(path, index=False)
    log(f"{name}: {n_rows:,} rows, {n_cols} inputs, largest leaf {gbm.largest_leaf(m):.3f}, fit {fit_s:.0f} s, total {time.time() - t:.0f} s")


def compare():
    r = pd.read_parquet(SETS["full"][0], columns=["flight_date", "outcome", "split", "lead_h", "as_of_utc"])
    r = r[r.split == "tune"].reset_index(drop=True)
    y = pd.get_dummies(r.outcome).reindex(columns=CLASSES, fill_value=0).to_numpy(dtype="float32")
    late = r.outcome.isin(CLASSES[2:]).to_numpy()
    def scores(name):
        p = pd.read_parquet(nw.OUT / f"quick_{name}.parquet").to_numpy()
        return ((p - y) ** 2).sum(axis=1), (p[:, 2:].sum(axis=1) - late) ** 2, p[:, 2:].sum(axis=1)
    bm, bl, bp = scores("base")
    rows = []
    for name in GROUPS:
        if not (nw.OUT / f"quick_{name}.parquet").exists():
            continue
        m, l, p = scores(name)
        ci, cl = day_ci(pd.Series(m - bm), r.flight_date), day_ci(pd.Series(l - bl), r.flight_date)
        rows.append({"inputs": name, "multi": m.mean(), "vs_base": ci["diff"], "lo": ci["lo"], "hi": ci["hi"],
                     "late": l.mean(), "late_vs_base": cl["diff"], "late_lo": cl["lo"], "late_hi": cl["hi"], "pred_late": p.mean(), "true_late": late.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(nw.OUT / "quick_compare.csv", index=False)
    log(out.round(5).to_string(index=False))
    for a, b in (("f", "n"), ("g", "h"), ("i", "k"), ("g", "i"), ("h", "k"), ("g", "f"), ("fg", "g"), ("fm", "fg")):
        if all((nw.OUT / f"quick_{x}.parquet").exists() for x in (a, b)):
            ci = day_ci(pd.Series(scores(a)[0] - scores(b)[0]), r.flight_date)
            log(f"{a} minus {b}, multiclass: {ci['diff']:+.5f} ({ci['lo']:+.5f} to {ci['hi']:+.5f})")
    # The gap this was meant to close: is the forecast still low when the region is disrupted?
    f = pd.read_parquet(nw.OUT / "full" / "f.parquet")[(pd.read_parquet(SETS["full"][0], columns=["split"]).split == "tune").to_numpy()]
    s = f.comm_o_2h_share_late.to_numpy()
    hour = r.as_of_utc.dt.hour.to_numpy()
    exc = s - pd.Series(s).groupby(hour).transform("mean").to_numpy()
    q = np.nanquantile(exc, [.5, .9, .99])
    band = np.digitize(exc, q)
    for name in [x for x in ("base", "f", "g", "h", "i", "k", "fg", "fm") if (nw.OUT / f"quick_{x}.parquet").exists()]:
        _, _, p = scores(name)
        log(name, "true minus predicted late, calm to worst:", [round(float(late[(band == b) & ~np.isnan(exc)].mean() - p[(band == b) & ~np.isnan(exc)].mean()), 4) for b in range(4)])


if __name__ == "__main__":
    step, *args = sys.argv[1:]
    {"communities": communities, "state": state, "gcn": gcn_compare, "gnn_inputs": gnn_inputs, "average": average, "fit_full": fit_full, "final_compare": final_compare, "scan": scan, "seeds": seeds, "hourcheck": hourcheck, "fit": fit, "compare": compare}[step](*args)
