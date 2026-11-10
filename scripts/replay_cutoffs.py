"""Replay 2025 trips under candidate cutoffs for the change rule.

A trip is a sampled 2025 flight at 24, 6 or 2 h before departure, with a deadline a set
number of minutes after its scheduled arrival (no real deadlines exist). An alternative:
same route, leaves 1 h or more after the forecast and within 24 h, scheduled to arrive by
the deadline, not already known to be cancelled. The traveler takes the recommendation.
Past schedules do not show seats, so an alternative may not have been bookable.

Steps are saved and skipped when rerun. 2025 only.
Usage: PYTHONPATH=. python scripts/replay_cutoffs.py        final_leads, as first measured
       MURPHY_MODEL=final_clean PYTHONPATH=. python ...     hindsight hidden, in replay_final_clean/
       MURPHY_MODEL=final_network PYTHONPATH=. python ...   plus communities and the GCN
"""
import os, time
import duckdb, joblib, numpy as np, pandas as pd
from murphy.forecast import features as ft
from murphy.forecast import gbm
from murphy.forecast.climatology import OUT_DIR, climatology
from murphy.forecast.features import hide_hindsight
from murphy.forecast.sample import ROOT
from murphy.forecast.score import PCOLS, day_ci
from murphy.recommend import made_it, p_make, should_change

OUT = OUT_DIR / "alternatives"
TAG = os.environ.get("MURPHY_MODEL", "final_leads")
HIDDEN = TAG != "final_leads"  # the original model saw the aircraft inputs unhidden
R = OUT / ("replay" if not HIDDEN else f"replay_{TAG}")
MODEL = OUT / f"{TAG}.joblib"
FEATS = OUT_DIR / "network" / "full" if TAG == "final_network" else OUT / "clean" / "full" if HIDDEN else ft.OUT
ROWS = OUT_DIR / "rows.parquet"
F2025 = str(ROOT / "data" / "flights" / "2025_*.parquet")
BUFFERS = (30, 60, 120)
MISS = (0.3, 0.4, 0.5)
GAIN = (0.10, 0.15, 0.20)
NEAR_H = 2  # an AI assumption: how much earlier a traveler could leave
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)


def forecasts(rows, feats) -> pd.DataFrame:
    saved = joblib.load(MODEL)
    X, r, _ = gbm.load_split("tune", saved["groups"], rows=rows, features=feats, carriers=saved["carriers"])
    return pd.concat([r, pd.DataFrame(gbm.predict(saved["model"], X), columns=PCOLS).astype("float32")], axis=1)


def current() -> pd.DataFrame:
    path = R / "current.parquet"
    if not path.exists():
        t = time.time()
        f = forecasts(ROWS, FEATS)
        c = pd.read_parquet(OUT_DIR / "clim_origin.parquet", columns=["flight_id", "lead_h", *PCOLS])
        c = c.rename(columns={k: f"c_{k}" for k in PCOLS})
        f = f.merge(c, on=["flight_id", "lead_h"], how="left")
        info = pd.read_parquet(ROWS, columns=["flight_id", "lead_h", "split", "flight_date", "arr_delay_min", "sched_arr_utc", "as_of_utc"])
        f = f.merge(info[info.split == "tune"].drop(columns="split"), on=["flight_id", "lead_h"])
        f.to_parquet(path, index=False)
        print(f"current forecasts: {len(f):,}, {time.time() - t:.0f} s", flush=True)
    return pd.read_parquet(path)


def build_alternatives(cur: pd.DataFrame) -> None:
    pairs, rows, feats = R / "pairs.parquet", R / "alt_rows.parquet", R / "alt_features"
    if not pairs.exists():
        at_risk = np.zeros(len(cur), bool)
        for prefix in ("", "c_"):
            at_risk |= (1 - p_make(cur[[prefix + k for k in PCOLS]].fillna(0).to_numpy(), np.full(len(cur), min(BUFFERS)))) >= min(MISS)
        print(f"trips at risk at the widest setting: {at_risk.sum():,} of {len(cur):,}", flush=True)
        t = cur.loc[at_risk, ["flight_id", "lead_h", "origin", "dest", "as_of_utc", "sched_arr_utc"]]
        con = duckdb.connect()
        con.register("t", t)
        con.execute(f"""COPY (SELECT t.flight_id, t.lead_h, a.flight_id AS alt, t.as_of_utc,
              date_diff('minute', t.as_of_utc, a.sched_dep_utc) / 60.0 AS alt_lead,
              date_diff('minute', a.sched_arr_utc, t.sched_arr_utc) AS arr_ahead_min
            FROM t JOIN read_parquet('{F2025}') a ON t.origin = a.origin AND t.dest = a.dest AND t.flight_id <> a.flight_id
             AND a.sched_dep_utc >= t.as_of_utc + INTERVAL 1 HOUR
             AND a.sched_dep_utc <= t.as_of_utc + INTERVAL 24 HOUR
             AND a.sched_arr_utc <= t.sched_arr_utc + to_minutes({max(BUFFERS)})
             AND NOT (a.cancelled AND a.dep_known_at < t.as_of_utc)) TO '{pairs}' (FORMAT parquet)""")
        con.execute(f"""COPY (SELECT a.flight_id, a.source_file, a.flight_date, a.marketing_carrier, a.operating_carrier,
              a.origin, a.dest, a.sched_dep_utc, a.sched_arr_utc, a.distance_mi, p.alt_lead AS lead_h, p.as_of_utc,
              a.outcome, a.arr_delay_min, 'tune' AS split
            FROM (SELECT DISTINCT alt, as_of_utc, alt_lead FROM read_parquet('{pairs}')) p
            JOIN (SELECT *, {ft_label()} AS outcome FROM read_parquet('{F2025}')) a ON a.flight_id = p.alt
            WHERE a.outcome IS NOT NULL AND a.arr_known_at IS NOT NULL
            ORDER BY p.as_of_utc, a.flight_id) TO '{rows}' (FORMAT parquet)""")
        print(duckdb.sql(f"SELECT (SELECT count(*) FROM read_parquet('{pairs}')) pairs, (SELECT count(*) FROM read_parquet('{rows}')) alt_rows").df().to_string(index=False), flush=True)
    if not (feats / "e.parquet").exists():
        t0 = time.time()
        feats.mkdir(parents=True, exist_ok=True)
        climatology(rows, feats / "clim_origin.parquet", by="origin")
        ft.build(rows=rows, out=feats, clim_origin=feats / "clim_origin.parquet")
        print(f"alternative inputs: {time.time() - t0:.0f} s", flush=True)
    if HIDDEN and not (feats / "d_raw.parquet").exists():
        hide_alternatives(rows, feats)
    if TAG == "final_network" and not (feats / "m.parquet").exists():  # layer 4's regional inputs
        from scripts.network import alt_inputs
        alt_inputs(rows, feats)
    path = R / "alt_forecasts.parquet"
    if not path.exists():
        t0 = time.time()
        f = forecasts(rows, feats)
        c = pd.read_parquet(feats / "clim_origin.parquet", columns=["flight_id", "lead_h", *PCOLS])
        f = f.merge(c.rename(columns={k: f"c_{k}" for k in PCOLS}), on=["flight_id", "lead_h"], how="left")
        info = pd.read_parquet(rows, columns=["flight_id", "lead_h", "arr_delay_min"])
        f.merge(info, on=["flight_id", "lead_h"]).rename(columns={"flight_id": "alt", "lead_h": "alt_lead"}).drop(
            columns=["origin", "dest", "split"]).to_parquet(path, index=False)
        print(f"alternative forecasts: {time.time() - t0:.0f} s", flush=True)


def hide_alternatives(rows, feats) -> None:
    """Same treatment as the fit and the trips. The original stays as d_raw."""
    d = pd.read_parquet(feats / "d.parquet")
    d.to_parquet(feats / "d_raw.parquet", index=False)
    x = pd.read_parquet(feats / "a.parquet", columns=["lead_h", "local_hour", "carrier"])
    r = pd.read_parquet(rows, columns=["split", "origin"])
    h = hide_hindsight(d, r.split, x.lead_h, [x.carrier, x.local_hour, r.origin])
    assert not (h.prior_turn_min < 0).any() and h.prior_turn_min[x.lead_h.to_numpy() < 24].notna().all()
    h.to_parquet(feats / "d.parquet", index=False)


def ft_label() -> str:
    from murphy.forecast.sample import label_sql
    return label_sql()


def evaluate(cur: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_parquet(R / "pairs.parquet")
    alts = pd.read_parquet(R / "alt_forecasts.parquet")
    pairs = pairs.merge(alts, on=["alt", "alt_lead"], how="inner", suffixes=("", "_a"))
    trips_out, rows = [], []
    for buf in BUFFERS:
        made_cur = made_it(cur.arr_delay_min, cur.outcome, np.full(len(cur), buf))
        pk = pairs[pairs.arr_ahead_min + buf >= 0]
        slack = (pk.arr_ahead_min + buf).to_numpy()
        for name, prefix in (("model", ""), ("climatology", "c_")):
            pc = p_make(cur[[prefix + k for k in PCOLS]].fillna(0).to_numpy(), np.full(len(cur), buf))
            pa = p_make(pk[[prefix + k for k in PCOLS]].fillna(0).to_numpy(), slack)
            m = pd.DataFrame({"flight_id": pk.flight_id.to_numpy(), "lead_h": pk.lead_h.to_numpy(), "pa": pa,
                              "alt_made": made_it(pk.arr_delay_min, pk.outcome, slack)})
            best = m.sort_values("pa", ascending=False, kind="stable").drop_duplicates(["flight_id", "lead_h"])
            t = cur[["flight_id", "lead_h", "flight_date"]].assign(pc=pc, made=made_cur).merge(best, on=["flight_id", "lead_h"], how="left")
            for miss in MISS:
                for gain in GAIN:
                    ch = should_change(t.pc, t.pa, miss, gain)
                    f = t[ch]
                    alt_made = f.alt_made.astype(bool)
                    diff = pd.Series(np.where(ch, t.alt_made.fillna(False).astype(bool).astype(int) - t.made.astype(int), 0))
                    ci = day_ci(diff, t.flight_date, n_boot=300)
                    rows.append({"buffer": buf, "scorer": name, "min_miss": miss, "min_gain": gain, "trips": len(t),
                                 "changes": int(ch.sum()), "per_1000": 1000 * ch.mean(),
                                 "current_missed": (~f.made).mean() if len(f) else np.nan,
                                 "alt_made": alt_made.mean() if len(f) else np.nan,
                                 "saved": int((~f.made & alt_made).sum()), "needless_ok": int((f.made & alt_made).sum()),
                                 "hurt": int((f.made & ~alt_made).sum()), "both_missed": int((~f.made & ~alt_made).sum()),
                                 "net_per_1000": 1000 * ci["diff"], "lo": 1000 * ci["lo"], "hi": 1000 * ci["hi"]})
    res = pd.DataFrame(rows)
    res.to_csv(R / "cutoffs.csv", index=False)
    return res


def breakdown(cur: pd.DataFrame, buf: int = 60, gain: float = 0.15) -> pd.DataFrame:
    """One deadline, by forecast time and by how the alternative is picked:
    best      highest chance of making it, whenever it leaves;
    near      highest chance, leaving no more than NEAR_H before the current flight;
    latest    the last-leaving flight that clears the gain, the smallest change of plan."""
    pairs = pd.read_parquet(R / "pairs.parquet").merge(
        pd.read_parquet(R / "alt_forecasts.parquet"), on=["alt", "alt_lead"], suffixes=("", "_a"))
    pk = pairs[pairs.arr_ahead_min + buf >= 0]
    slack = (pk.arr_ahead_min + buf).to_numpy()
    m = pd.DataFrame({"flight_id": pk.flight_id.to_numpy(), "lead_h": pk.lead_h.to_numpy(), "pa": p_make(pk[PCOLS].to_numpy(), slack),
                      "alt_made": made_it(pk.arr_delay_min, pk.outcome, slack), "ahead_h": pk.arr_ahead_min.to_numpy() / 60,
                      "dep_rel_h": (pk.alt_lead - pk.lead_h).to_numpy()})
    trips = cur[["flight_id", "lead_h", "flight_date"]].assign(
        pc=p_make(cur[PCOLS].to_numpy(), np.full(len(cur), buf)),
        made=made_it(cur.arr_delay_min, cur.outcome, np.full(len(cur), buf)))
    m = m.merge(trips[["flight_id", "lead_h", "pc"]], on=["flight_id", "lead_h"])
    picks = {"best": m.sort_values("pa", ascending=False, kind="stable"),
             "near": m[m.dep_rel_h >= -NEAR_H].sort_values("pa", ascending=False, kind="stable"),
             "latest": m[m.pa - m.pc >= gain].sort_values("dep_rel_h", ascending=False, kind="stable")}
    rows = []
    for pick, ordered in picks.items():
        t = trips.merge(ordered.drop(columns="pc").drop_duplicates(["flight_id", "lead_h"]), on=["flight_id", "lead_h"], how="left")
        for miss in MISS:
            for lead, g in list(t.groupby("lead_h")) + [("all", t)]:
                risky = g[(1 - g.pc) >= miss]
                ch = should_change(g.pc, g.pa, miss, gain)
                f = g[ch]
                am = f.alt_made.astype(bool)
                diff = np.where(ch, g.alt_made.fillna(False).astype(bool).astype(int) - g.made.astype(int), 0)
                ci = day_ci(pd.Series(diff), g.flight_date, n_boot=300)
                rows.append({"pick": pick, "min_miss": miss, "lead_h": lead, "trips": len(g), "flagged": len(risky),
                             "pred_miss": (1 - risky.pc).mean(), "true_miss": (~risky.made).mean(),
                             "changes": len(f), "no_change_share": 1 - len(f) / max(len(risky), 1),
                             "saved": int((~f.made & am).sum()), "needless": int((f.made & am).sum()),
                             "hurt": int((f.made & ~am).sum()), "both_miss": int((~f.made & ~am).sum()),
                             "alt_p": f.pa.mean(), "alt_true": am.mean(), "net_per_1000": 1000 * ci["diff"],
                             "lo": 1000 * ci["lo"], "hi": 1000 * ci["hi"], "med_arr_ahead_h": f.ahead_h.median(),
                             "med_dep_rel_h": f.dep_rel_h.median(), "leaves_2h_earlier": (f.dep_rel_h <= -2).mean()})
    b = pd.DataFrame(rows)
    b.to_csv(R / f"breakdown_{buf}.csv", index=False)
    return b


def main() -> None:
    R.mkdir(parents=True, exist_ok=True)
    cur = current()
    build_alternatives(cur)
    for buf in BUFFERS:
        made = made_it(cur.arr_delay_min, cur.outcome, np.full(len(cur), buf))
        print(f"buffer {buf}: {made.mean():.4f} of trips make the deadline by keeping the flight")
    if not (R / "cutoffs.csv").exists():
        print(evaluate(cur).round(3).to_string(index=False))
    print(breakdown(cur).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
