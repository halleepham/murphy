"""The 2026 test, forecast only. New files only; the app and 2025 results are untouched.

  inputs       groups f and m for the 2026 rows. Refits the three co-disruption GCNs on 2023-24 (seeds 0-2,
               as in scripts/network.py) and checks they reproduce the saved 2025 predictions first.
  score tune   2025, by the same code. Must reproduce the published 2025 figures. Rerunnable.
  score test   2026, once. Refuses if its results exist or the 2025 run has not passed.
  compare      2026 beside 2025 and 2025 Jan-Aug, and the pass/fail tests G1-G7.

Models are fixed: no refit, no setting changes. Results: data/forecast/eval_2026/ (gitignored).
Usage: PYTHONPATH=. python scripts/eval_2026.py <step> [split]
"""
import json, sys, time
import numpy as np, pandas as pd
from murphy.forecast import gbm
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.gbm import CLASSES
from murphy.forecast.score import PCOLS, calibration, day_ci, report
from murphy.recommend import FIRM_FROM, UNSURE_FROM, made_it, p_make

OUT = OUT_DIR / "eval_2026"
FEATS = OUT / "features"            # a-e linked to the hidden set, plus f and m with 2026 filled
ALT, NET = OUT_DIR / "alternatives", OUT_DIR / "network"
ROWS = OUT_DIR / "rows.parquet"
END_2026 = "2026-09-01"             # the last 2026 file is August
DEADLINES = (30, 60, 120)
# name: (saved model, rounds, input directory). Rounds frozen before the test.
MODELS = {"final_network": (ALT / "final_network.joblib", 300, FEATS),
          "final_clean": (ALT / "final_clean.joblib", 300, ALT / "clean" / "full"),
          "schedule": (gbm.OUT / "l2abl_a.joblib", None, OUT_DIR / "features"),
          "schedule_history": (gbm.OUT / "l2abl_ab.joblib", None, OUT_DIR / "features")}
PAIRS = [("final_network", "clim_origin"), ("final_network", "schedule"), ("final_network", "final_clean"),
         ("final_network", "schedule_history"), ("final_clean", "clim_origin"), ("schedule", "clim_origin")]
# What the 2025 run must reproduce: the published 2025 scores and the app's track record.
PUBLISHED_2025 = {"final_network": 0.5602, "final_clean": 0.5612, "clim_origin": 0.6026,
                  ("high", 60): 0.820, ("too_uncertain", 60): 0.533, ("keep", 60): 0.085}
MAX_BLANK_GAP = 0.05                # network inputs may be blank this much more often than in 2025, no more
log = lambda *a, **k: print(*a, flush=True, **k)


# ---- small pieces, tested in tests/test_eval_2026.py ----

def unsealed(rows: pd.DataFrame) -> pd.DataFrame:
    """network.state_inputs leaves any row with split 'test' blank. A relabeled copy lets the same
    code build 2026 inputs without changing the library."""
    return rows.assign(split=np.where(rows.split == "test", "unsealed", rows.split))


def blank_check(blank_share: dict, split: str, ref: str = "tune", gap: float = MAX_BLANK_GAP) -> None:
    """Refuse to score a split whose network inputs are blank far more often than 2025's."""
    if blank_share[split] > blank_share[ref] + gap:
        raise SystemExit(f"network inputs blank on {blank_share[split]:.1%} of {split} rows against "
                         f"{blank_share[ref]:.1%} for {ref}: run 'inputs' first")


def once(path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: 2026 is scored once")


def band(risk: np.ndarray) -> np.ndarray:
    """The app's verdict bands (murphy/app/server.py)."""
    return np.where(risk < UNSURE_FROM, "keep", np.where(risk < FIRM_FROM, "too_uncertain", "high"))


def weighted_gap(cal: pd.DataFrame) -> float:
    """Mean |predicted - observed| over calibration bins, weighted by forecasts."""
    return float(((cal.mean_pred - cal.observed).abs() * cal.n).sum() / cal.n.sum())


def periods(r: pd.DataFrame, split: str) -> dict:
    """Which rows each reported period uses. 2025 also gets Jan-Aug, the months 2026 has."""
    out = {"all": np.ones(len(r), bool)}
    if split == "tune":
        out["jan_aug"] = (pd.to_datetime(r.flight_date).dt.month <= 8).to_numpy()
    return out


def brier_rows(p: np.ndarray, outcome: pd.Series) -> np.ndarray:
    y = (outcome.to_numpy()[:, None] == np.array(CLASSES)[None, :])
    return ((p - y) ** 2).sum(axis=1)


# ---- inputs ----

def inputs() -> None:
    from murphy.forecast import gcn, network as nw
    from scripts import network as sn
    FEATS.mkdir(parents=True, exist_ok=True)
    for g in "abcde":
        link = FEATS / f"{g}.parquet"
        if not link.exists():
            link.symlink_to((ALT / "clean" / "full" / f"{g}.parquet").resolve())
    member, big = sn.load_member()
    r = pd.read_parquet(ROWS, columns=["origin", "dest", "split", "as_of_utc"])
    test = (r.split == "test").to_numpy()

    # GCN: the three co-disruption seeds on 2023-24, as in scripts/network.py average(), now run forward to Aug 2026.
    t0 = time.time()
    t = gcn.hourly(nw.files((2023, 2024, 2025, 2026)), big, sn.START, END_2026)
    y, w = gcn.targets(t)
    log(f"hourly tables {len(t['hours']):,} hours x {len(big)} airports, {time.time() - t0:.0f} s")
    preds = [sn.run_fold(t, y, w, "co", sn.TRAIN_YEARS, big, f"eval_co_s{s}", s) for s in (0, 1, 2)]
    old_hours = pd.date_range(sn.START, sn.END, freq="h", inclusive="left")
    n_old = len(old_hours)
    in_2025 = (old_hours.year == 2025) & (old_hours < pd.Timestamp(sn.END) - pd.Timedelta(hours=gcn.HORIZON_H))
    seed0 = np.abs(preds[0][:n_old] - np.load(NET / "gcn_co_final.npy"))
    avg = np.mean(preds, axis=0)
    avg_gap = np.abs(avg[:n_old][in_2025] - np.load(NET / "gcn_co_avg.npy")[in_2025])
    check = {"seed0_max_diff_2023_25": float(seed0[:n_old].max()), "seed0_max_diff_2025": float(seed0[in_2025].max()),
             "avg_max_diff_2025": float(avg_gap.max()), "avg_mean_diff_2025": float(avg_gap.mean())}
    log("reproduction:", check)
    (OUT / "gcn_reproduction.json").write_text(json.dumps(check))
    if check["avg_max_diff_2025"] > 1e-4:
        raise SystemExit("the refit GCNs do not reproduce the saved 2025 predictions: stop and ask")
    np.save(OUT / "gcn_co_avg_to_2026_08.npy", avg)

    # Group m: copy the saved inputs, fill the 2026 rows from the refit average.
    m = pd.read_parquet(NET / "full" / "m.parquet")
    for side, col in (("o", "origin"), ("d", "dest")):
        v = gcn.hour_lookup(r.as_of_utc[test], r[col][test], avg, t["hours"], big)
        m.loc[test, f"gcn_{side}_next3h"] = v
    m.to_parquet(FEATS / "m.parquet", index=False)

    # Group f: same windows, 2026 departures known before the last 2026 forecast.
    t0 = time.time()
    rt = unsealed(r[test].reset_index(drop=True))
    ev = nw.load_events(nw.files((2025, 2026)), rt.as_of_utc.max())  # the 2 h window never reaches back further
    f_new, _ = nw.state_inputs(rt, ev, member)
    f = pd.read_parquet(NET / "full" / "f.parquet")
    for c in f.columns[1:]:
        f.loc[test, c] = f_new[c].to_numpy()
    f.to_parquet(FEATS / "f.parquet", index=False)
    log(f"group f for {test.sum():,} rows, {time.time() - t0:.0f} s")
    share = {s: float(m.gcn_o_next3h[(r.split == s).to_numpy()].isna().mean()) for s in ("fit", "tune", "test")}
    share_f = {s: float(f.comm_o_2h_share_late[(r.split == s).to_numpy()].isna().mean()) for s in ("fit", "tune", "test")}
    log("m blank share:", share, "f share-late blank share:", share_f)
    (OUT / "input_blanks.json").write_text(json.dumps({"m": share, "f": share_f}))


# ---- scoring ----

def forecasts(split: str):
    """Every model's forecast for one split, in row order, with the row info."""
    r = pd.read_parquet(ROWS, columns=[*gbm.INFO, "flight_date", "arr_delay_min"])
    r = r[r.split == split].reset_index(drop=True)
    clim = r[["flight_id", "lead_h"]].merge(pd.read_parquet(OUT_DIR / "clim_origin.parquet"), on=["flight_id", "lead_h"], how="left")
    glob = r[["flight_id", "lead_h"]].merge(pd.read_parquet(OUT_DIR / "clim_global.parquet"), on=["flight_id", "lead_h"], how="left")
    p = {"clim_origin": clim[PCOLS].to_numpy(dtype="float32"), "clim_global": glob[PCOLS].to_numpy(dtype="float32"),
         "uniform": np.full((len(r), len(CLASSES)), 1 / len(CLASSES), dtype="float32")}
    for name, (path, rounds, feats) in MODELS.items():
        t0 = time.time()
        rounds = rounds or json.loads(path.with_suffix(".json").read_text())["best_rounds"]
        m, saved = gbm.truncated(path, rounds)
        X, rx, _ = gbm.load_split(split, saved["groups"], features=feats, carriers=saved["carriers"])
        assert (rx.flight_id.to_numpy() == r.flight_id.to_numpy()).all() and (rx.lead_h.to_numpy() == r.lead_h.to_numpy()).all()
        p[name] = gbm.predict(m, X).astype("float32")
        log(f"{name}: {rounds} rounds, {len(X):,} forecasts, {X.shape[1]} inputs, {time.time() - t0:.0f} s")
        del X
    return r, p


def score(split: str) -> None:
    assert split in ("tune", "test")
    out = OUT / split
    if split == "test":
        once(out / "scores.csv")
        if not (OUT / "tune" / "passed.json").exists():
            raise SystemExit("the 2025 run has not reproduced the published figures yet")
    blanks = json.loads((OUT / "input_blanks.json").read_text())["m"]
    blank_check(blanks, split)
    out.mkdir(parents=True, exist_ok=True)
    r, p = forecasts(split)
    raw_d = pd.read_parquet(OUT_DIR / "features" / "d.parquet")[(pd.read_parquet(ROWS, columns=["split"]).split == split).to_numpy()]
    from scripts.refit_clean import kinds
    kind = kinds(raw_d.reset_index(drop=True))
    loss = {k: brier_rows(v, r.outcome) for k, v in p.items()}
    scores, diffs, cals, bands, by_kind = [], [], [], [], []
    for period, keep in periods(r, split).items():
        rk = r[keep].reset_index(drop=True)
        for name, v in p.items():
            df = pd.concat([rk, pd.DataFrame(v[keep], columns=PCOLS)], axis=1)
            scores.append(report(df).assign(model=name, period=period,
                                            pred_late=float(v[keep][:, 2:].sum(axis=1).mean()),
                                            true_late=float(rk.outcome.isin(CLASSES[2:]).mean())))
            for e in ("late", "cancelled"):
                cals.append(calibration(df, e).assign(model=name, event=e, period=period))
            for k in ("normal", "overlap", "no tail", "delay blank"):
                w = (kind[keep] == k) & (rk.lead_h < 24).to_numpy()
                by_kind.append({"period": period, "model": name, "kind": k, "forecasts": int(w.sum()), "brier": float(loss[name][keep][w].mean())})
        groups = [("all", np.ones(len(rk), bool))] + [(f"lead {h}h", (rk.lead_h == h).to_numpy()) for h in (2, 6, 24)]
        for a, b in PAIRS:
            for g, w in groups:
                d = day_ci(pd.Series((loss[a][keep] - loss[b][keep])[w]), rk.flight_date[w])
                diffs.append({"period": period, "model": a, "against": b, "group": g, "forecasts": int(w.sum()), **d})
        for dl in DEADLINES:
            for name in ("final_network", "final_clean"):
                risk = 1 - p_make(p[name][keep].astype(float), np.full(len(rk), dl))
                miss = ~made_it(rk.arr_delay_min, rk.outcome, np.full(len(rk), dl))
                t = pd.DataFrame({"band": band(risk), "risk": risk, "miss": miss}).groupby("band")
                bands.append(t.agg(forecasts=("miss", "size"), predicted_miss=("risk", "mean"), true_miss=("miss", "mean"))
                             .reset_index().assign(period=period, model=name, deadline_min=dl))
    s, dfs, c, b, k = (pd.concat(scores), pd.DataFrame(diffs), pd.concat(cals), pd.concat(bands), pd.DataFrame(by_kind))
    for name, x in (("scores", s), ("diffs", dfs), ("calibration", c), ("bands", b), ("kinds", k)):
        x.to_csv(out / f"{name}.csv", index=False)
    log(f"wrote {out}")
    if split == "tune":
        reproduce(s, b)


def reproduce(s: pd.DataFrame, b: pd.DataFrame) -> None:
    """Same code, same 2025 numbers as published, to the rounding they were published at."""
    got = {m: float(s[(s.model == m) & (s.group == "all") & (s.period == "all")].brier_multi.iloc[0])
           for m in ("final_network", "final_clean", "clim_origin")}
    nb = b[(b.model == "final_network") & (b.period == "all")]
    for (bd, dl) in [k for k in PUBLISHED_2025 if isinstance(k, tuple)]:
        got[(bd, dl)] = float(nb[(nb.band == bd) & (nb.deadline_min == dl)].true_miss.iloc[0])
    bad = {str(k): (round(got[k], 4), v) for k, v in PUBLISHED_2025.items()
           if abs(got[k] - v) > (0.00005 if isinstance(k, str) else 0.0005)}
    log("2025 reproduction:", {str(k): round(v, 4) for k, v in got.items()})
    if bad:
        raise SystemExit(f"2025 does not reproduce (got, published): {bad}")
    (OUT / "tune" / "passed.json").write_text(json.dumps({str(k): v for k, v in got.items()}))
    log("2025 reproduced; 2026 may be scored")


# ---- side by side, and the pass/fail tests ----

def compare() -> None:
    pd.set_option("display.width", 220, "display.max_columns", 30)
    load = lambda split, name: pd.read_csv(OUT / split / f"{name}.csv")
    cols = {"2026": ("test", "all"), "2025": ("tune", "all"), "2025 Jan-Aug": ("tune", "jan_aug")}
    s = {k: load(sp, "scores").query("period == @per") for k, (sp, per) in cols.items()}
    d = {k: load(sp, "diffs").query("period == @per") for k, (sp, per) in cols.items()}
    c = {k: load(sp, "calibration").query("period == @per") for k, (sp, per) in cols.items()}
    b = {k: load(sp, "bands").query("period == @per") for k, (sp, per) in cols.items()}
    for m in ("brier_multi", "brier_late", "brier_cancelled"):
        log(f"\n{m}, all forecasts")
        log(pd.DataFrame({k: v[v.group == "all"].set_index("model")[m] for k, v in s.items()}).round(4).to_string())
    log("\ndifferences, 95% interval by day")
    for k, v in d.items():
        log(k); log(v[["model", "against", "group", "diff", "lo", "hi"]].round(5).to_string(index=False))
    gaps = {k: v.groupby(["model", "event"]).apply(weighted_gap).unstack() for k, v in c.items()}
    log("\ncalibration, weighted mean gap"); [log(k, g.round(4).to_string(), sep="\n") for k, g in gaps.items()]
    log("\nbands"); [log(k, v.round(3).to_string(index=False), sep="\n") for k, v in b.items()]
    pick = lambda k, a, against, g: d[k][(d[k].model == a) & (d[k].against == against) & (d[k].group == g)].iloc[0]
    g1 = pick("2026", "final_network", "clim_origin", "all")
    nb = b["2026"][(b["2026"].model == "final_network") & (b["2026"].deadline_min == 60)].set_index("band").true_miss
    tests = {
        "G1 beats climatology, all": g1.hi < 0,
        "G2 beats climatology at 2, 6, 24 h": all(pick("2026", "final_network", "clim_origin", f"lead {h}h").hi < 0 for h in (2, 6, 24)),
        "G3 gap at least half of 2025's": g1["diff"] <= 0.5 * pick("2025", "final_network", "clim_origin", "all")["diff"],
        "G4 beats schedule-only": pick("2026", "final_network", "schedule", "all").hi < 0,
        "G5 late calibration no worse than climatology": gaps["2026"].loc["final_network", "late"] <= gaps["2026"].loc["clim_origin", "late"],
        "G6 60%+ band truly misses at least 60%": nb["high"] >= 0.60,
        "G7 40-60% band truly misses 40-70%": 0.40 <= nb["too_uncertain"] <= 0.70}
    log("\npass/fail"); [log(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in tests.items()]
    (OUT / "tests.json").write_text(json.dumps({k: bool(v) for k, v in tests.items()}))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "inputs":
        inputs()
    elif step == "score" and len(sys.argv) == 3:
        score(sys.argv[2])
    elif step == "compare":
        compare()
    else:
        raise SystemExit(__doc__)
