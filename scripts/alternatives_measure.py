"""What forecasts for alternative flights take. 2025 only.

  counts  how many alternatives a trip has: same origin and dest, any carrier,
          departing at least 1 h after the forecast, arriving no later than the
          current flight's scheduled arrival + X h. Schedule only, no outcomes.
  grid    the same sampled flights at many lead times, because an alternative's lead
          is rarely the 24, 6 or 2 h the model trained on. Times the input build,
          checks it against the saved inputs at 24/6/2 h, and scores by lead.

Results: data/forecast/alternatives/ (gitignored).
Usage: PYTHONPATH=. python scripts/alternatives_measure.py counts|grid [n_flights]
"""
import sys, time
import duckdb, joblib, numpy as np, pandas as pd
from murphy.forecast import features as ft
from murphy.forecast import gbm
from murphy.forecast.climatology import OUT_DIR, climatology
from murphy.forecast.sample import ROOT
from murphy.forecast.score import PCOLS, calibration, day_ci, event_brier, event_prob, multiclass_brier, row_brier

OUT = OUT_DIR / "alternatives"
ROWS = OUT_DIR / "rows.parquet"
F2025 = str(ROOT / "data" / "flights" / "2025_*.parquet")  # never reads 2026 outcomes
LEADS = (1, 2, 3, 4, 6, 9, 12, 18, 24, 30, 36)
TRAINED = (2, 6, 24)
pd.set_option("display.width", 200, "display.max_columns", 30)


def make_rows(path, n_flights: int, lead_set=LEADS) -> None:
    leads = ", ".join(f"({h})" for h in lead_set)
    duckdb.sql(f"""
        COPY (WITH f AS (SELECT * EXCLUDE (lead_h, as_of_utc) FROM read_parquet('{ROWS}')
                         WHERE split = 'tune' AND lead_h = 2
                         ORDER BY hash(flight_id) LIMIT {n_flights})
              SELECT f.flight_id, source_file, flight_date, marketing_carrier, operating_carrier,
                     origin, dest, sched_dep_utc, sched_arr_utc, distance_mi, l.lead_h,
                     sched_dep_utc - to_hours(CAST(l.lead_h AS BIGINT)) AS as_of_utc,
                     outcome, arr_delay_min, split
              FROM f, (VALUES {leads}) l(lead_h)
              ORDER BY as_of_utc, f.flight_id, l.lead_h) TO '{path}' (FORMAT parquet)""")


def timed_build(rows, out) -> dict:
    t = time.time()
    clim = out / "clim_origin.parquet"
    out.mkdir(parents=True, exist_ok=True)
    climatology(rows, clim, by="origin")
    times = {"clim": time.time() - t}
    for g in "abcde":
        t = time.time()
        ft.build(rows=rows, out=out, clim_origin=clim, groups=g)
        times[g] = time.time() - t
    return times


def counts(n_flights: int) -> None:
    sql = f"""
      WITH t AS (SELECT flight_id, origin, dest, as_of_utc, lead_h, sched_arr_utc FROM read_parquet('{ROWS}')
                 WHERE split = 'tune' AND flight_id IN
                   (SELECT flight_id FROM read_parquet('{ROWS}') WHERE split = 'tune' AND lead_h = 2
                    ORDER BY hash(flight_id) LIMIT {n_flights})),
           a AS (SELECT flight_id, origin, dest, sched_dep_utc, sched_arr_utc FROM read_parquet('{F2025}'))
      SELECT t.flight_id, t.lead_h, a.flight_id AS alt, date_diff('minute', t.as_of_utc, a.sched_dep_utc) / 60.0 AS alt_lead,
             date_diff('minute', t.sched_arr_utc, a.sched_arr_utc) / 60.0 AS arr_after
      FROM t JOIN a ON t.origin = a.origin AND t.dest = a.dest AND t.flight_id <> a.flight_id
       AND a.sched_dep_utc >= t.as_of_utc + INTERVAL 1 HOUR
       AND a.sched_arr_utc <= t.sched_arr_utc + INTERVAL 12 HOUR"""
    t0 = time.time()
    d = duckdb.sql(sql).df()
    print(f"{n_flights:,} flights, {len(d):,} pairs up to +12 h, {time.time() - t0:.0f} s")
    trips = duckdb.sql(f"SELECT count(*) FROM read_parquet('{ROWS}') WHERE split = 'tune' AND lead_h = 2").fetchall()[0][0]
    print(f"sampled 2025 flights: {trips:,}")
    rows = []
    for x in (0, 3, 6, 12):
        k = d[d.arr_after <= x]
        for lead in TRAINED:
            c = k[k.lead_h == lead].groupby("flight_id").size()
            per = np.zeros(n_flights); per[:len(c)] = c.to_numpy()
            rows.append({"arrive_by_plus_h": x, "lead_h": lead, "mean_alts": per.mean(),
                         "median": np.median(per), "p90": np.quantile(per, 0.9),
                         "share_none": (per == 0).mean(), "pairs": int(per.sum())})
    r = pd.DataFrame(rows)
    r["est_pairs_all_sampled"] = (r.pairs * trips / n_flights).round().astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "counts.csv", index=False)
    print(r.round(3).to_string(index=False))
    k = d[d.arr_after <= 6]
    bins = [0, 1.5, 2.5, 5, 7, 23, 25, 1000]
    names = ["1-1.5", "1.5-2.5 (trained 2)", "2.5-5", "5-7 (trained 6)", "7-23", "23-25 (trained 24)", "25+"]
    print("alternative's own lead, arriving by +6 h:")
    print((pd.cut(k.alt_lead, bins, labels=names, right=False).value_counts(normalize=True, sort=False)).round(3).to_string())


def grid(n_flights: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, feats = OUT / "grid_rows.parquet", OUT / "grid_features"
    make_rows(rows, n_flights)
    times = timed_build(rows, feats)
    print("build seconds:", {k: round(v, 1) for k, v in times.items()}, flush=True)

    # The new build must equal the saved inputs where the two overlap.
    con = duckdb.connect()
    for g in "abcde":
        new = con.sql(f"""
          SELECT o.old_rid, n.* FROM (SELECT file_row_number AS rid, flight_id, lead_h FROM read_parquet('{rows}', file_row_number = true)
                                      WHERE lead_h IN {TRAINED}) x
          JOIN (SELECT file_row_number AS old_rid, flight_id, lead_h FROM read_parquet('{ROWS}', file_row_number = true)) o USING (flight_id, lead_h)
          JOIN read_parquet('{feats / (g + ".parquet")}') n ON n.rid = x.rid ORDER BY o.old_rid""").df()
        old = con.sql(f"SELECT * FROM read_parquet('{ft.OUT / (g + '.parquet')}') WHERE rid IN (SELECT unnest($ids))",
                      params={"ids": new.old_rid.tolist()}).df().sort_values("rid").reset_index(drop=True)
        cols = [c for c in old.columns if c != "rid"]
        bad = {c: int((~((new[c] == old[c]) | (new[c].isna() & old[c].isna()))).sum()) for c in cols}
        print(f"group {g}: {len(old):,} rows compared, columns differing: {({c: n for c, n in bad.items() if n}) or 'none'}", flush=True)

    t = time.time()
    saved = joblib.load(gbm.OUT / "final_l2.joblib")
    m = saved["model"]
    print("lead_h split points the model can use:",
          m._bin_mapper.bin_thresholds_[list(m.feature_names_in_).index("lead_h")])
    X, r, _ = gbm.load_split("tune", saved["groups"], rows=rows, features=feats, carriers=saved["carriers"])
    p = gbm.predict(m, X)
    print(f"predict {len(X):,} rows: {time.time() - t:.1f} s (includes loading)", flush=True)
    f = pd.concat([r, pd.DataFrame(p, columns=PCOLS)], axis=1)
    clim = r.merge(pd.read_parquet(feats / "clim_origin.parquet"), on=["flight_id", "lead_h"]).dropna(subset=PCOLS)
    days = pd.read_parquet(rows, columns=["flight_id", "flight_date"]).drop_duplicates().set_index("flight_id").flight_date
    out = []
    for lead, g in f.groupby("lead_h"):
        c = clim[clim.lead_h == lead]
        both = pd.concat([row_brier(g).rename("a"), row_brier(c).rename("b")], axis=1, join="inner")
        ci = day_ci(both.a - both.b, days.reindex(both.index.get_level_values(0)))
        cal = calibration(g, "late")
        out.append({"lead_h": lead, "n": len(g), "trained": lead in TRAINED,
                    "model": multiclass_brier(g[PCOLS].to_numpy(), g.outcome),
                    "clim": multiclass_brier(c[PCOLS].to_numpy(), c.outcome),
                    "diff": ci["diff"], "lo": ci["lo"], "hi": ci["hi"],
                    "late_model": event_brier(g, "late"), "late_clim": event_brier(c, "late"),
                    "mean_p_late": event_prob(g, "late").mean(),
                    "obs_late": g.outcome.isin(gbm.CLASSES[2:]).mean(),
                    "cal_gap": float(((cal.mean_pred - cal.observed).abs() * cal.n).sum() / cal.n.sum())})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "grid_scores.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"counts": counts, "grid": grid}[cmd](int(sys.argv[2]) if len(sys.argv) > 2 else {"counts": 20000, "grid": 30000}[cmd])
