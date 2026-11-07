"""Refit with hindsight hidden from the aircraft inputs (docs/decisions/0014).

  prepare  write hidden copies of group D; the other groups are links to the originals.
  probe    can hidden flights still be told apart from normal ones?
  scan     does any other input carry a hindsight signal?
  mini     a small fit, hidden and not hidden, by kind of flight. A check, not a result.
  fit      the full fit, same rows and settings as final_leads.
  score    2025 scores, after replay_cutoffs has forecast the trips with it.

Trains on 2023-24; checks and scores on 2025. 2026 rows are rewritten with the rest
but no outcome is read, and they lend to and borrow from only each other.
Usage: PYTHONPATH=. python scripts/refit_clean.py <step> [rounds]
"""
import json, sys, time
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from murphy.forecast import features as ft
from murphy.forecast import gbm
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.features import hide_hindsight
from murphy.forecast.score import PCOLS, day_ci
from murphy.recommend import made_it, p_make

OUT = OUT_DIR / "alternatives"
C = OUT / "clean"
SETS = {"full": (OUT_DIR / "rows.parquet", ft.OUT), "fit": (OUT / "fit_rows.parquet", OUT / "fit_features"),
        "grid": (OUT / "grid_rows.parquet", OUT / "grid_features"), "unseen": (OUT / "unseen_rows.parquet", OUT / "unseen_features")}
KEYS = ["flight_id", "lead_h"]
SPREAD = (1, 2, 3, 4, 6, 9, 12, 18, 24)  # as in refit_leads
MATCH = lambda a, r: [a.carrier, a.local_hour, r.origin]  # the closest of 14 tried by the probe (0014)
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)


def hide_set(rows, feats, out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for g in "abce":
        link = out / f"{g}.parquet"
        if not link.exists():
            link.symlink_to((feats / f"{g}.parquet").resolve())
    a = pd.read_parquet(feats / "a.parquet", columns=["lead_h", "local_hour", "carrier"])
    d = pd.read_parquet(feats / "d.parquet")
    r = pd.read_parquet(rows, columns=["split", "origin"])
    assert len(a) == len(d) == len(r), "rows and inputs differ in length"
    h = hide_hindsight(d, r.split, a.lead_h, MATCH(a, r))
    moved = (d.prior_turn_min.fillna(-1e9) != h.prior_turn_min.fillna(-1e9)).sum()
    short = a.lead_h.to_numpy() < 24
    print(f"{out.name}: rows {len(d):,}, values changed {moved:,}; after: overlaps {(h.prior_turn_min < 0).sum():,}, "
          f"blank turn under 24 h {h.prior_turn_min[short].isna().sum():,}", flush=True)
    assert not (h.prior_turn_min < 0).any() and h.prior_turn_min[short].notna().all()
    h.to_parquet(out / "d.parquet", index=False)


def prepare() -> None:
    for name, (rows, feats) in SETS.items():
        hide_set(rows, feats, C / name)


def hidden_mask(feats) -> np.ndarray:
    d = pd.read_parquet(feats / "d.parquet", columns=["prior_turn_min"]).prior_turn_min
    a = pd.read_parquet(feats / "a.parquet", columns=["lead_h"]).lead_h
    return ((a < 24) & (d.isna() | (d < 0))).to_numpy()


def probe(n=400_000) -> None:
    """A classifier tries to pick out hidden flights. Removed aircraft inputs set the floor: bad days
    show in the other inputs, which is fair. Hidden should sit at that floor; original far above."""
    rows, feats = SETS["fit"]
    X0, r, _ = gbm.load_split("fit", rows=rows, features=feats)
    Xh, _, _ = gbm.load_split("fit", rows=rows, features=C / "fit")
    keep = X0.lead_h.to_numpy() < 24
    y = hidden_mask(feats)[keep]
    X0, Xh = X0[keep].reset_index(drop=True), Xh[keep].reset_index(drop=True)
    print(f"short-lead fit rows {len(y):,}, hidden {y.sum():,} ({y.mean():.4f})", flush=True)
    rng = np.random.default_rng(0)
    i = rng.choice(len(y), min(n, len(y)), replace=False)
    # Oversample hidden rows so a rare class can be learned.
    i = np.unique(np.concatenate([i, np.flatnonzero(y)]))
    cut = len(i) * 7 // 10
    i = rng.permutation(i)
    tr, te = i[:cut], i[cut:]
    d = ["prior_turn_min", "prior_status", "prior_delay_so_far"]
    for name, X in (("no aircraft inputs (floor)", Xh.drop(columns=d)), ("hidden", Xh), ("original (should be near 1)", X0)):
        m = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=0).fit(X.iloc[tr], y[tr])
        print(f"{name:30s} AUC {roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]):.4f}", flush=True)


def scan(n=500_000) -> None:
    """2025 at 2 and 6 h, all inputs after hiding: blank versus filled, and the top tenth, by cancelled share."""
    rows, _ = SETS["full"]
    X, r, _ = gbm.load_split("tune", rows=rows, features=C / "full")
    keep = (X.lead_h.to_numpy() < 24)
    X, r = X[keep].reset_index(drop=True), r[keep].reset_index(drop=True)
    i = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
    X, can = X.iloc[i], r.outcome.iloc[i].isin(["cancelled", "diverted"]).to_numpy()
    out = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=float)
        blank = np.isnan(v)
        hi = v >= np.nanquantile(v, 0.9) if (~blank).any() else blank
        lo = v <= np.nanquantile(v, 0.1) if (~blank).any() else blank
        out.append({"input": c, "blank_share": blank.mean(), "fail_blank": can[blank].mean() if blank.any() else np.nan,
                    "fail_filled": can[~blank].mean() if (~blank).any() else np.nan, "fail_top10": can[hi].mean(), "fail_low10": can[lo].mean()})
    t = pd.DataFrame(out)
    t["worst"] = t[["fail_blank", "fail_top10", "fail_low10"]].max(axis=1)
    t.to_csv(C / "scan.csv", index=False)
    print(f"overall cancelled or diverted {can.mean():.4f}")
    print(t.sort_values("worst", ascending=False).round(4).head(25).to_string(index=False))


def kinds(d) -> np.ndarray:
    return np.select([d.prior_turn_min.isna(), d.prior_delay_so_far.isna(), d.prior_turn_min < 0], ["no tail", "delay blank", "overlap"], "normal")


def mini(rounds=60, every=10) -> None:
    """Same rows for both, a tenth of the fit. The control must show the leak, or this check proves nothing."""
    rows, feats = SETS["full"]
    r0 = pd.read_parquet(rows, columns=[*gbm.INFO, "arr_delay_min"])
    tune = (r0.split == "tune").to_numpy()
    d0 = pd.read_parquet(feats / "d.parquet")[tune].reset_index(drop=True)
    r = r0[tune].reset_index(drop=True)
    pick = np.flatnonzero((pd.util.hash_pandas_object(r.flight_id, index=False).to_numpy() % 5 == 0) & (r.lead_h < 24).to_numpy())
    for name, f_fit, f_tune in (("not hidden", SETS["fit"][1], feats), ("hidden", C / "fit", C / "full")):
        t = time.time()
        X, rf, carriers = gbm.load_split("fit", rows=SETS["fit"][0], features=f_fit)
        sel = pd.util.hash_pandas_object(rf.flight_id, index=False).to_numpy() % every == 0
        m = gbm.make_model(rounds, **gbm.REFIT).fit(X[sel], rf.outcome[sel])
        del X
        Xt, _, _ = gbm.load_split("tune", rows=rows, features=f_tune, carriers=carriers)
        p = gbm.predict(m, Xt.iloc[pick])
        risk = 1 - p_make(p, np.full(len(p), 60))
        miss = ~made_it(r.arr_delay_min.iloc[pick], r.outcome.iloc[pick], np.full(len(p), 60))
        res = pd.DataFrame({"lead_h": r.lead_h.iloc[pick].to_numpy(), "kind": kinds(d0.iloc[pick]), "risk": risk, "miss": miss, "flag": risk >= 0.5})
        print(f"--- {name}: {sel.sum():,} fit rows, {time.time() - t:.0f} s", flush=True)
        print(res.groupby(["lead_h", "kind"]).agg(n=("miss", "size"), pred=("risk", "mean"), true=("miss", "mean"), flagged=("flag", "sum")).round(3).to_string(), flush=True)


def fit(rounds=gbm.MAX_ROUNDS) -> None:
    tag = "final_clean" if rounds == gbm.MAX_ROUNDS else f"final_clean_dry{rounds}"
    assert not (OUT / f"{tag}.joblib").exists(), f"{tag} already exists"
    X, r, carriers = gbm.load_split("fit", rows=SETS["fit"][0], features=C / "fit")
    assert not (X.prior_turn_min < 0).any()
    print(f"fit rows: {len(X):,}", flush=True)
    t = time.time()
    model = gbm.make_model(rounds, **gbm.REFIT).fit(X, r.outcome)
    fit_s = time.time() - t
    del X, r
    joblib.dump({"model": model, "carriers": carriers, "groups": "abcde"}, OUT / f"{tag}.joblib")
    Xg, rg, _ = gbm.load_split("tune", rows=SETS["grid"][0], features=C / "grid", carriers=carriers)
    keep = rg.lead_h.isin(SPREAD).to_numpy()
    curve = gbm.staged_scores(model, Xg[keep], rg.outcome[keep])
    info = {"fit_s": round(fit_s), "largest_leaf": gbm.largest_leaf(model),
            "best_rounds": int(curve.loc[curve.brier_multi.idxmin()].rounds), "curve": curve.to_dict("list")}
    (OUT / f"{tag}.json").write_text(json.dumps(info))
    print(f"saved {tag}: {info}", flush=True)


def score(tag="final_clean") -> None:
    """All 2025 forecasts: the hidden model against final_leads and by-origin climatology, overall,
    by forecast time and by kind of flight. final_leads saw the leak, so its score is flattered."""
    new, old = (pd.read_parquet(OUT / d / "current.parquet", columns=[*KEYS, "flight_date", "outcome", *PCOLS, *(c for c in ([f"c_{k}" for k in PCOLS] if d == "replay" else []))])
                for d in (f"replay_{tag}", "replay"))
    m = new.merge(old, on=[*KEYS, "flight_date", "outcome"], suffixes=("", "_old"))
    assert len(m) == len(new) == len(old)
    raw = pd.read_parquet(SETS["full"][1] / "d.parquet")[(pd.read_parquet(SETS["full"][0], columns=["split"]).split == "tune").to_numpy()]
    r = pd.read_parquet(SETS["full"][0], columns=[*KEYS, "split"])
    r = r[r.split == "tune"].reset_index(drop=True)
    k = r[KEYS].assign(kind=kinds(raw.reset_index(drop=True)))
    m = m.merge(k, on=KEYS)
    y = pd.get_dummies(m.outcome).reindex(columns=gbm.CLASSES, fill_value=0).to_numpy(dtype="float32")
    brier = lambda cols: ((m[cols].to_numpy(dtype="float32") - y) ** 2).sum(axis=1)
    b_new, b_old, b_clim = brier(PCOLS), brier([f"{c}_old" for c in PCOLS]), brier([f"c_{c}" for c in PCOLS])
    late = m.outcome.isin(gbm.CLASSES[2:]).to_numpy()
    late_cols = PCOLS[2:]
    rows = []
    groups = [("all", np.ones(len(m), bool))] + [(f"{h} h", (m.lead_h == h).to_numpy()) for h in sorted(m.lead_h.unique())] + \
             [(f"{x}, under 24 h", ((m.kind == x) & (m.lead_h < 24)).to_numpy()) for x in ("normal", "overlap", "no tail", "delay blank")]
    for name, w in groups:
        ci_old = day_ci(pd.Series(b_new[w] - b_old[w]), m.flight_date[w], n_boot=300)
        ci_clim = day_ci(pd.Series(b_new[w] - b_clim[w]), m.flight_date[w], n_boot=300)
        rows.append({"group": name, "forecasts": int(w.sum()), "hidden": b_new[w].mean(), "final_leads": b_old[w].mean(), "climatology": b_clim[w].mean(),
                     "vs_old": ci_old["diff"], "old_lo": ci_old["lo"], "old_hi": ci_old["hi"], "vs_clim": ci_clim["diff"], "clim_lo": ci_clim["lo"], "clim_hi": ci_clim["hi"],
                     "late_true": late[w].mean(), "late_pred": m.loc[w, late_cols].sum(axis=1).mean(), "late_pred_old": m.loc[w, [f"{c}_old" for c in late_cols]].sum(axis=1).mean()})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"scores_{tag}.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    step = sys.argv[1]
    args = [int(a) if a.isdigit() else a for a in sys.argv[2:]]
    {"prepare": prepare, "probe": probe, "scan": scan, "mini": mini, "fit": fit, "score": score}[step](*args)
