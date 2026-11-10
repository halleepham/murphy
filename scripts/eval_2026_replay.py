"""The 2026 advice replay. New files only; the 2025 replay scripts are untouched.

Same method as scripts/replay_cutoffs.py and replay_baselines.py, which hard-code 2025. This reuses their
year-free parts (chances, pick, tally, hide_alternatives) and takes the split as an argument.

  check               tally the SAVED 2025 replay with this file's loader; must equal baselines_firm60.csv
  build tune 2025-01  rebuild alternatives for one 2025 month; must equal the saved replay for those trips
  build test          trips, alternatives, their inputs and forecasts for 2026
  tally test          2026, once; then R1-R6 beside 2025

Results: data/forecast/eval_2026/replay_*/ (gitignored).
Usage: PYTHONPATH=. python scripts/eval_2026_replay.py <step> [split] [month]
"""
import os
os.environ["MURPHY_MODEL"] = "final_network"  # the app's model; read by scripts.replay_cutoffs at import
import json, sys, time
import duckdb, joblib, numpy as np, pandas as pd
from murphy.forecast import features as ft, gbm
from murphy.forecast.climatology import OUT_DIR, climatology
from murphy.forecast.sample import ROOT
from murphy.forecast.score import PCOLS, day_ci
from murphy.recommend import FIRM_FROM, MIN_MISS, inbound_too_late, should_change
from scripts.eval_2026 import END_2026, FEATS, OUT as EVAL, ROWS, once, unsealed
from scripts.replay_baselines import KEYS, chances, pick, tally
from scripts.replay_cutoffs import BUFFERS, MISS, ft_label, hide_alternatives
from murphy.recommend import p_make

ALT = OUT_DIR / "alternatives"
SAVED_2025 = ALT / "replay_final_network"
MODEL = ALT / "final_network.joblib"
SCHEDULE = "l2abl_a"
YEAR = {"tune": 2025, "test": 2026}
log = lambda *a, **k: print(*a, flush=True, **k)
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)


def out_dir(split, month=None):
    return EVAL / (f"replay_{split}_{month}" if month else f"replay_{split}")


def flights(split) -> str:
    return str(ROOT / "data" / "flights" / f"{YEAR[split]}_*.parquet")


# ---- small pieces, tested in tests/test_eval_2026.py ----

def ratio(rescued, needless) -> float:
    """Rescued per needless change. Blank when there were no needless changes."""
    return rescued / needless if needless else float("nan")


def add_rates(res: dict) -> dict:
    """Counts beside rates, since 2026 has fewer trips than 2025."""
    return {**res, "changes_per_1000": 1000 * res["changes"] / res["trips"], "rescued_per_needless": ratio(res["rescued"], res["needless"]),
            "worse_share": res["worse"] / res["changes"] if res["changes"] else float("nan")}


def r_tests(t: pd.DataFrame, net_2025_60: float) -> dict:
    """R1-R6 from a tally table with rules murphy_firm60 and thumb."""
    m = t[t.rule == "murphy_firm60"].set_index("buffer")
    th = t[t.rule == "thumb"].set_index("buffer")
    return {"R1 Murphy's net at 60 min above 0": m.loc[60, "lo"] > 0,
            "R2 Murphy minus thumb at 60 min above 0": th.loc[60, "mm_lo"] > 0,
            "R3 Murphy minus thumb at 30 and 120 min above 0": bool(th.loc[30, "mm_lo"] > 0 and th.loc[120, "mm_lo"] > 0),
            "R4 net at 60 min at least half of 2025's": m.loc[60, "net"] >= 0.5 * net_2025_60,
            "R5 rescued at least twice needless at 60 min": m.loc[60, "rescued"] >= 2 * m.loc[60, "needless"],
            "R6 made worse at most 5% of changes at 60 min": m.loc[60, "worse"] <= 0.05 * m.loc[60, "changes"]}


# ---- forecasts ----

def predict(name_or_path, split, rows, feats, keep=None) -> pd.DataFrame:
    if name_or_path == SCHEDULE:
        rounds = json.loads((gbm.OUT / f"{SCHEDULE}.json").read_text())["best_rounds"]
        model, saved = gbm.truncated(gbm.OUT / f"{SCHEDULE}.joblib", rounds)
    else:
        saved = joblib.load(name_or_path)
        model = saved["model"]
    X, r, _ = gbm.load_split(split, saved["groups"], rows=rows, features=feats, carriers=saved["carriers"])
    if keep is not None:
        k = keep(r).to_numpy()  # by position: X keeps the row numbers of the whole file
        X, r = X[k].reset_index(drop=True), r[k].reset_index(drop=True)
    return pd.concat([r, pd.DataFrame(gbm.predict(model, X), columns=PCOLS).astype("float32")], axis=1)


# ---- build ----

def build(split, month=None) -> None:
    R = out_dir(split, month)
    R.mkdir(parents=True, exist_ok=True)
    info = pd.read_parquet(ROWS, columns=["flight_id", "lead_h", "split", "flight_date", "arr_delay_min", "sched_arr_utc", "as_of_utc"])
    info = info[info.split == split].drop(columns="split")
    in_month = (lambda r: r.flight_id.isin(set(info.flight_id[info.flight_date.astype(str).str[:7] == month]))) if month else None
    t0 = time.time()
    if not (R / "current.parquet").exists():
        f = predict(MODEL, split, ROWS, FEATS, in_month)
        c = pd.read_parquet(OUT_DIR / "clim_origin.parquet", columns=["flight_id", "lead_h", *PCOLS]).rename(columns={k: f"c_{k}" for k in PCOLS})
        f.merge(c, on=KEYS, how="left").merge(info, on=KEYS).to_parquet(R / "current.parquet", index=False)
        s = predict(SCHEDULE, split, ROWS, OUT_DIR / "features", in_month)
        s[KEYS].join(s[PCOLS].add_prefix("s_")).to_parquet(R / "schedule_current.parquet", index=False)
        log(f"current forecasts {len(f):,}, {time.time() - t0:.0f} s")
    cur = pd.read_parquet(R / "current.parquet")
    pairs, rows, feats = R / "pairs.parquet", R / "alt_rows.parquet", R / "alt_features"
    if not pairs.exists():  # as replay_cutoffs.build_alternatives, with this split's flights and label
        at_risk = np.zeros(len(cur), bool)
        for prefix in ("", "c_"):
            at_risk |= (1 - p_make(cur[[prefix + k for k in PCOLS]].fillna(0).to_numpy(), np.full(len(cur), min(BUFFERS)))) >= min(MISS)
        log(f"trips at risk at the widest setting: {at_risk.sum():,} of {len(cur):,}")
        con = duckdb.connect()
        con.register("t", cur.loc[at_risk, ["flight_id", "lead_h", "origin", "dest", "as_of_utc", "sched_arr_utc"]])
        F = flights(split)
        con.execute(f"""COPY (SELECT t.flight_id, t.lead_h, a.flight_id AS alt, t.as_of_utc,
              date_diff('minute', t.as_of_utc, a.sched_dep_utc) / 60.0 AS alt_lead,
              date_diff('minute', a.sched_arr_utc, t.sched_arr_utc) AS arr_ahead_min
            FROM t JOIN read_parquet('{F}') a ON t.origin = a.origin AND t.dest = a.dest AND t.flight_id <> a.flight_id
             AND a.sched_dep_utc >= t.as_of_utc + INTERVAL 1 HOUR
             AND a.sched_dep_utc <= t.as_of_utc + INTERVAL 24 HOUR
             AND a.sched_arr_utc <= t.sched_arr_utc + to_minutes({max(BUFFERS)})
             AND NOT (a.cancelled AND a.dep_known_at < t.as_of_utc)) TO '{pairs}' (FORMAT parquet)""")
        con.execute(f"""COPY (SELECT a.flight_id, a.source_file, a.flight_date, a.marketing_carrier, a.operating_carrier,
              a.origin, a.dest, a.sched_dep_utc, a.sched_arr_utc, a.distance_mi, p.alt_lead AS lead_h, p.as_of_utc,
              a.outcome, a.arr_delay_min, '{split}' AS split
            FROM (SELECT DISTINCT alt, as_of_utc, alt_lead FROM read_parquet('{pairs}')) p
            JOIN (SELECT *, {ft_label()} AS outcome FROM read_parquet('{F}')) a ON a.flight_id = p.alt
            WHERE a.outcome IS NOT NULL AND a.arr_known_at IS NOT NULL
            ORDER BY p.as_of_utc, a.flight_id) TO '{rows}' (FORMAT parquet)""")
        log(duckdb.sql(f"SELECT (SELECT count(*) FROM read_parquet('{pairs}')) pairs, (SELECT count(*) FROM read_parquet('{rows}')) alt_rows").df().to_string(index=False))
    if not (feats / "e.parquet").exists():
        t0 = time.time()
        feats.mkdir(parents=True, exist_ok=True)
        climatology(rows, feats / "clim_origin.parquet", by="origin")
        ft.build(rows=rows, out=feats, clim_origin=feats / "clim_origin.parquet")
        log(f"alternative inputs A-E: {time.time() - t0:.0f} s")
    if not (feats / "d_raw.parquet").exists():
        hide_alternatives(rows, feats)
    if not (feats / "m.parquet").exists():
        network_inputs(rows, feats)
    if not (R / "alt_forecasts.parquet").exists():
        t0 = time.time()
        f = predict(MODEL, split, rows, feats)
        c = pd.read_parquet(feats / "clim_origin.parquet", columns=["flight_id", "lead_h", *PCOLS]).rename(columns={k: f"c_{k}" for k in PCOLS})
        ar = pd.read_parquet(rows, columns=["flight_id", "lead_h", "arr_delay_min"])
        f.merge(c, on=KEYS, how="left").merge(ar, on=KEYS).rename(columns={"flight_id": "alt", "lead_h": "alt_lead"}).drop(
            columns=["origin", "dest", "split"]).to_parquet(R / "alt_forecasts.parquet", index=False)
        s = predict(SCHEDULE, split, rows, feats)
        s[KEYS].join(s[PCOLS].add_prefix("s_")).to_parquet(R / "schedule_alt.parquet", index=False)
        log(f"alternative forecasts: {time.time() - t0:.0f} s")
    if month:
        same_as_saved(R)


def network_inputs(rows, feats) -> None:
    """Groups f and m for alternatives, as scripts/network.py alt_inputs, but with 2026 flights and the GCN run to Aug 2026."""
    from murphy.forecast import gcn, network as nw
    from scripts import network as sn
    member, big = sn.load_member()
    r = pd.read_parquet(rows, columns=["origin", "dest", "split", "as_of_utc"])
    f, _ = nw.state_inputs(unsealed(r), nw.load_events(nw.files((2024, 2025, 2026)), r.as_of_utc.max()), member)
    f.to_parquet(feats / "f.parquet", index=False)
    hours = pd.date_range(sn.START, END_2026, freq="h", inclusive="left")
    p = np.load(EVAL / "gcn_co_avg_to_2026_08.npy")
    pd.DataFrame({"rid": np.arange(len(r)), "gcn_o_next3h": gcn.hour_lookup(r.as_of_utc, r.origin, p, hours, big),
                  "gcn_d_next3h": gcn.hour_lookup(r.as_of_utc, r.dest, p, hours, big)}).to_parquet(feats / "m.parquet", index=False)


def same_as_saved(R) -> None:
    """Gate 2: the rebuilt month equals the saved 2025 replay for the same trips."""
    new_p = pd.read_parquet(R / "pairs.parquet")
    old_p = pd.read_parquet(SAVED_2025 / "pairs.parquet")
    old_p = old_p[old_p.flight_id.isin(set(pd.read_parquet(R / "current.parquet").flight_id))]
    k = ["flight_id", "lead_h", "alt", "alt_lead"]
    same_pairs = new_p.sort_values(k).reset_index(drop=True)[k].equals(old_p.sort_values(k).reset_index(drop=True)[k])
    a_new = pd.read_parquet(R / "alt_forecasts.parquet")
    a_old = pd.read_parquet(SAVED_2025 / "alt_forecasts.parquet").merge(a_new[["alt", "alt_lead"]], on=["alt", "alt_lead"])
    m = a_new.merge(a_old, on=["alt", "alt_lead"], suffixes=("", "_old"))
    # Hidden alternatives borrow aircraft inputs at random from the pool of alternatives, and one month's
    # pool is smaller than the year's, so only those not hidden must match exactly.
    raw = pd.read_parquet(R / "alt_features" / "d_raw.parquet", columns=["prior_turn_min"]).prior_turn_min
    ar = pd.read_parquet(R / "alt_rows.parquet", columns=["flight_id", "lead_h", "as_of_utc"]).rename(columns={"flight_id": "alt", "lead_h": "alt_lead"})
    ar["hidden"] = ((ar.alt_lead < 24) & (raw.isna() | (raw < 0))).to_numpy()
    m = m.merge(ar, on=["alt", "alt_lead"])
    d = np.abs(m[PCOLS].to_numpy() - m[[f"{c}_old" for c in PCOLS]].to_numpy()).max(axis=1)
    # The saved 2025 replay looked up 2024 hours in a cross-fit GCN array (a model on 2023 alone); this build uses the
    # 2023-24 model at every hour, which is the one for 2025 and 2026. So only forecast times in 2025 must match.
    same_year = (m.as_of_utc.dt.year == 2025).to_numpy()
    exact = ~m.hidden.to_numpy() & same_year
    gap, hidden_gap = float(d[exact].max()), float(d[m.hidden.to_numpy()].max()) if m.hidden.any() else 0.0
    prior_gap = float(d[~m.hidden.to_numpy() & ~same_year].max()) if (~same_year).any() else 0.0
    c_new = pd.read_parquet(R / "current.parquet")
    c_old = pd.read_parquet(SAVED_2025 / "current.parquet").merge(c_new[KEYS], on=KEYS)
    mc = c_new.merge(c_old, on=KEYS, suffixes=("", "_old"))
    cgap = float(np.abs(mc[PCOLS].to_numpy() - mc[[f"{c}_old" for c in PCOLS]].to_numpy()).max())
    res = {"pairs_new": len(new_p), "pairs_saved": len(old_p), "pairs_equal": bool(same_pairs),
           "alt_forecasts": len(a_new), "matched": len(m), "hidden": int(m.hidden.sum()), "compared_exactly": int(exact.sum()),
           "alt_max_diff": gap, "prior_year_not_hidden": int((~m.hidden.to_numpy() & ~same_year).sum()), "prior_year_max_diff": prior_gap,
           "hidden_max_diff": hidden_gap, "trips": len(c_new), "trip_max_diff": cgap}
    log("gate 2:", res)
    (R / "gate.json").write_text(json.dumps(res))
    if not (same_pairs and len(m) == len(a_new) and gap < 1e-5 and cgap < 1e-5):
        raise SystemExit("the rebuilt month does not match the saved 2025 replay: stop and ask")
    log("gate 2 passed")


# ---- tally ----

def load(R, split):
    """As replay_baselines.load, for any split: trips with the inputs the rule of thumb reads, and alternatives."""
    cur = pd.read_parquet(R / "current.parquet", columns=[*KEYS, "flight_date", "outcome", "arr_delay_min", *PCOLS, *[f"c_{k}" for k in PCOLS]])
    d = ["prior_delay_so_far", "prior_turn_min"]
    X, r, _ = gbm.load_split(split, "ad")  # as recorded
    cur = cur.merge(pd.concat([r[KEYS], X[["prior_turn_min"]].add_prefix("raw_").reset_index(drop=True)], axis=1), on=KEYS, how="left")
    X, r, _ = gbm.load_split(split, "ad", features=FEATS)  # as the model and the rule of thumb see them
    cur = cur.merge(pd.concat([r[KEYS], X[d].reset_index(drop=True)], axis=1), on=KEYS, how="left")
    cur["clean"] = ~(cur.raw_prior_turn_min < 0)
    cur = cur.merge(pd.read_parquet(R / "schedule_current.parquet"), on=KEYS, how="left")
    X, r, _ = gbm.load_split(split, "a", rows=R / "alt_rows.parquet", features=R / "alt_features")
    raw = pd.read_parquet(R / "alt_features" / "d_raw.parquet", columns=["prior_turn_min"]).prior_turn_min.to_numpy()
    turn = pd.DataFrame({"alt": r.flight_id, "alt_lead": r.lead_h, "alt_clean": ~(raw < 0)})
    sa = pd.read_parquet(R / "schedule_alt.parquet").rename(columns={"flight_id": "alt", "lead_h": "alt_lead"})
    pairs = pd.read_parquet(R / "pairs.parquet").merge(pd.read_parquet(R / "alt_forecasts.parquet"), on=["alt", "alt_lead"], suffixes=("", "_a"))
    pairs = pairs.merge(sa, on=["alt", "alt_lead"], how="left").merge(turn, on=["alt", "alt_lead"], how="left")
    pairs["dep_rel_h"] = pairs.alt_lead - pairs.lead_h
    return cur, pairs


def rules(cur, pairs) -> pd.DataFrame:
    """The app's rule against the rule of thumb and the schedule-only model, and the replay's own 50% rule."""
    rec = []
    for buf in BUFFERS:
        t, a = chances(cur, pairs, buf)
        m = pick(t, a, "pa", False)
        res, base = tally(m, should_change(m.pc, m.pa, FIRM_FROM), {"buffer": buf, "rule": "murphy_firm60"})
        rec.append(add_rates({**res, "flagged": int(((1 - m.pc) >= FIRM_FROM).sum())}))
        res, _ = tally(m, should_change(m.pc, m.pa, MIN_MISS), {"buffer": buf, "rule": "murphy_50"})
        rec.append(add_rates({**res, "flagged": int(((1 - m.pc) >= MIN_MISS).sum())}))
        ts, als = chances(cur, pairs, buf, "s_")
        s = pick(ts, als, "pa", False)
        res, diff = tally(s, should_change(s.pc, s.pa, FIRM_FROM), {"buffer": buf, "rule": "schedule_firm60"})
        ci = day_ci(pd.Series(base - diff), m.flight_date, n_boot=300)
        rec.append(add_rates({**res, "flagged": int(((1 - s.pc) >= FIRM_FROM).sum()), "murphy_minus": 1000 * ci["diff"], "mm_lo": 1000 * ci["lo"], "mm_hi": 1000 * ci["hi"]}))
        th = pick(t, a, "alt_lead", True)
        flag = inbound_too_late(cur.prior_delay_so_far, cur.prior_turn_min, buf)
        res, diff = tally(th, flag & th.pa.notna().to_numpy(), {"buffer": buf, "rule": "thumb"})
        ci = day_ci(pd.Series(base - diff), m.flight_date, n_boot=300)
        rec.append(add_rates({**res, "flagged": int(flag.sum()), "no_alternatives": int((flag & th.pa.isna().to_numpy()).sum()),
                              "murphy_minus": 1000 * ci["diff"], "mm_lo": 1000 * ci["lo"], "mm_hi": 1000 * ci["hi"]}))
    return pd.DataFrame(rec)


COLS = ["buffer", "rule", "trips", "flagged", "changes", "changes_per_1000", "rescued", "needless", "worse", "both_missed",
        "rescued_per_needless", "worse_share", "net", "lo", "hi", "murphy_minus", "mm_lo", "mm_hi"]


def check() -> None:
    """Gate 1: this file's loader and tally on the saved 2025 replay equal baselines_firm60.csv, and R6's 2025 value."""
    cur, pairs = load(SAVED_2025, "tune")
    t = rules(cur, pairs)
    (EVAL / "replay_2025").mkdir(parents=True, exist_ok=True)
    t.to_csv(EVAL / "replay_2025" / "rules.csv", index=False)
    old = pd.read_csv(SAVED_2025 / "baselines_firm60.csv")
    new = t[t.rule.isin(["murphy_firm60", "thumb"])].reset_index(drop=True)
    cols = ["buffer", "trips", "changes", "rescued", "needless", "worse", "both_missed", "net", "lo", "hi", "flagged", "murphy_minus", "mm_lo", "mm_hi"]
    a, b = new[cols].astype(float).to_numpy(), old[cols].astype(float).to_numpy()
    same = np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
    log(new[COLS].round(3).to_string(index=False))
    m60 = new[(new.rule == "murphy_firm60") & (new.buffer == 60)].iloc[0]
    log(f"R6 2025 recomputed: worse {int(m60.worse)} of {int(m60.changes)} changes = {m60.worse_share:.4f}")
    if not same:
        raise SystemExit("the new loader does not reproduce baselines_firm60.csv: stop")
    log("gate 1 passed: baselines_firm60.csv reproduced exactly")


def tally_split(split) -> None:
    R = out_dir(split)
    assert split == "test"
    once(R / "rules.csv")
    for gate in (EVAL / "replay_2025" / "rules.csv", out_dir("tune", "2025-01") / "gate.json"):
        if not gate.exists():
            raise SystemExit(f"{gate} missing: run the gates first")
    cur, pairs = load(R, split)
    t = rules(cur, pairs)
    t.to_csv(R / "rules.csv", index=False)
    old = pd.read_csv(EVAL / "replay_2025" / "rules.csv")
    for name, x in (("2026", t), ("2025", old)):
        log(name); log(x[COLS].round(3).to_string(index=False))
    net60 = float(old[(old.rule == "murphy_firm60") & (old.buffer == 60)].net.iloc[0])
    tests = r_tests(t, net60)
    log("\npass/fail"); [log(f"  {'PASS' if v else 'FAIL'}  {k}") for k, v in tests.items()]
    (R / "tests.json").write_text(json.dumps({k: bool(v) for k, v in tests.items()}))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["check"]:
        check()
    elif args[:1] == ["build"] and len(args) in (2, 3):
        build(args[1], args[2] if len(args) == 3 else None)
    elif args == ["tally", "test"]:
        tally_split("test")
    else:
        raise SystemExit(__doc__)
