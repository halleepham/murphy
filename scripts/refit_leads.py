"""Refit with leads spread across 1-24 h, and compare with final_l2.

Same flights (2023-24), settings and row count as final_l2. Each flight keeps 3 rows;
its 3 leads are chosen by a fixed hash from SPREAD. Scored on 2025 only, on the same
30,000 flights at each lead: those it trained on, those it did not (UNSEEN), and 30/36 h.
Finished steps are skipped when rerun.

Usage: PYTHONPATH=. python scripts/refit_leads.py [rounds]   (default 300; other values
are saved under a separate name, for a dry run)
"""
import json, sys, time
from pathlib import Path
import duckdb, joblib, numpy as np, pandas as pd
from scipy.special import softmax
from murphy.forecast import features as ft
from murphy.forecast import gbm
from murphy.forecast.climatology import climatology
from murphy.forecast.score import PCOLS, calibration, day_ci, event_brier, event_prob, multiclass_brier, row_brier
from scripts.alternatives_measure import OUT, ROWS, make_rows

SPREAD = (1, 2, 3, 4, 6, 9, 12, 18, 24)
UNSEEN = (5, 8, 15, 21)
pd.set_option("display.width", 200, "display.max_columns", 30)


def fit_rows(path: Path) -> None:
    leads = ", ".join(f"({h})" for h in SPREAD)
    duckdb.sql(f"""
        COPY (WITH f AS (SELECT * EXCLUDE (lead_h, as_of_utc) FROM read_parquet('{ROWS}')
                         WHERE split = 'fit' AND lead_h = 2)
              SELECT f.flight_id, source_file, flight_date, marketing_carrier, operating_carrier,
                     origin, dest, sched_dep_utc, sched_arr_utc, distance_mi, l.lead_h,
                     sched_dep_utc - to_hours(CAST(l.lead_h AS BIGINT)) AS as_of_utc,
                     outcome, arr_delay_min, split
              FROM f, (VALUES {leads}) l(lead_h)
              QUALIFY row_number() OVER (PARTITION BY f.flight_id ORDER BY md5(f.flight_id || '|' || l.lead_h)) <= 3
              ORDER BY as_of_utc, f.flight_id, l.lead_h) TO '{path}' (FORMAT parquet)""")
    # hash(id, lead) gave half the flights the same 3 leads, so check the spread.
    sets = duckdb.sql(f"""SELECT count(*), min(n), max(n) FROM (SELECT s, count(*) n FROM
        (SELECT flight_id, list_sort(list(lead_h)) s FROM read_parquet('{path}') GROUP BY 1) GROUP BY 1)""").fetchall()[0]
    assert sets[0] == 84 and sets[2] < 1.1 * sets[1], f"leads not evenly spread: {sets}"


def build(rows: Path, feats: Path) -> None:
    if (feats / "e.parquet").exists():
        return
    feats.mkdir(parents=True, exist_ok=True)
    clim = feats / "clim_origin.parquet"
    climatology(rows, clim, by="origin")
    ft.build(rows=rows, out=feats, clim_origin=clim)


def scored(model, X, r, rounds=None) -> pd.DataFrame:
    if rounds:
        model._predictors = model._predictors[:rounds]
    return pd.concat([r, pd.DataFrame(gbm.predict(model, X), columns=PCOLS)], axis=1)


def main(rounds: int) -> None:
    tag = "final_leads" if rounds == gbm.MAX_ROUNDS else f"final_leads_dry{rounds}"
    rows, feats = OUT / "fit_rows.parquet", OUT / "fit_features"
    if not rows.exists():
        fit_rows(rows)
    print(duckdb.sql(f"SELECT lead_h, count(*) n FROM read_parquet('{rows}') GROUP BY 1 ORDER BY 1").df().to_string(index=False), flush=True)
    t = time.time(); build(rows, feats); print(f"fit inputs: {time.time() - t:.0f} s", flush=True)

    unseen_rows, unseen_feats = OUT / "unseen_rows.parquet", OUT / "unseen_features"
    if not unseen_rows.exists():
        make_rows(unseen_rows, 30000, UNSEEN)
    t = time.time(); build(unseen_rows, unseen_feats); print(f"unseen inputs: {time.time() - t:.0f} s", flush=True)

    path = OUT / f"{tag}.joblib"
    if path.exists():
        saved = joblib.load(path)
        info = json.loads((OUT / f"{tag}.json").read_text())
    else:
        X, r, carriers = gbm.load_split("fit", rows=rows, features=feats)
        print(f"fit rows: {len(X):,}", flush=True)
        t = time.time()
        model = gbm.make_model(rounds, **gbm.REFIT).fit(X, r.outcome)
        fit_s = time.time() - t
        del X, r
        saved = {"model": model, "carriers": carriers, "groups": "abcde"}
        joblib.dump(saved, path)
        # Rounds are picked at the trained leads only, as final_l2's were on its leads.
        Xg, rg, _ = gbm.load_split("tune", rows=OUT / "grid_rows.parquet", features=OUT / "grid_features", carriers=carriers)
        keep = rg.lead_h.isin(SPREAD).to_numpy()
        curve = gbm.staged_scores(model, Xg[keep], rg.outcome[keep])
        info = {"fit_s": round(fit_s), "largest_leaf": gbm.largest_leaf(model),
                "best_rounds": int(curve.loc[curve.brier_multi.idxmin()].rounds), "curve": curve.to_dict("list")}
        (OUT / f"{tag}.json").write_text(json.dumps(info))
    print({k: v for k, v in info.items()}, flush=True)

    frames = []
    for rw, fe in ((OUT / "grid_rows.parquet", OUT / "grid_features"), (unseen_rows, unseen_feats)):
        new = saved["model"]
        Xn, r, _ = gbm.load_split("tune", rows=rw, features=fe, carriers=saved["carriers"])
        old = joblib.load(gbm.OUT / "final_l2.joblib")
        Xo, _, _ = gbm.load_split("tune", rows=rw, features=fe, carriers=old["carriers"])
        days = pd.read_parquet(rw, columns=["flight_id", "flight_date"]).drop_duplicates().set_index("flight_id").flight_date
        fn, fo = scored(new, Xn, r, info["best_rounds"]), scored(old["model"], Xo, r)
        for lead in sorted(r.lead_h.unique()):
            a, b = fn[fn.lead_h == lead], fo[fo.lead_h == lead]
            both = pd.concat([row_brier(a).rename("a"), row_brier(b).rename("b")], axis=1, join="inner")
            ci = day_ci(both.a - both.b, days.reindex(both.index.get_level_values(0)))
            gap = lambda g: float((lambda c: ((c.mean_pred - c.observed).abs() * c.n).sum() / c.n.sum())(calibration(g, "late")))
            frames.append({"lead_h": lead, "in_new_fit": lead in SPREAD, "in_old_fit": lead in (2, 6, 24),
                           "new": multiclass_brier(a[PCOLS].to_numpy(), a.outcome), "old": multiclass_brier(b[PCOLS].to_numpy(), b.outcome),
                           "diff": ci["diff"], "lo": ci["lo"], "hi": ci["hi"],
                           "late_new": event_brier(a, "late"), "late_old": event_brier(b, "late"),
                           "obs_late": a.outcome.isin(gbm.CLASSES[2:]).mean(),
                           "mean_late_new": event_prob(a, "late").mean(), "mean_late_old": event_prob(b, "late").mean(),
                           "gap_new": gap(a), "gap_old": gap(b)})
    res = pd.DataFrame(frames).sort_values("lead_h")
    res.to_csv(OUT / f"compare_{tag}.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else gbm.MAX_ROUNDS)
