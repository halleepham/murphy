"""Too-uncertain band and simple baselines that act, on the saved 2025 replay.

Baselines, each deciding keep or change at the same forecast moments as murphy:
  schedule  the schedule-only model (l2abl_a at its best round) under murphy's rule.
  thumb     the traveler's rule: the inbound aircraft's lateness outlasts its turn by more than the deadline
            allows, so take the next allowed flight. It has no forecast, so it checks no gain.
Steps are saved and skipped when rerun. 2025 only.
Usage: PYTHONPATH=. python scripts/replay_baselines.py    add MURPHY_MODEL=final_clean for hidden inputs
"""
import json
import numpy as np, pandas as pd
from murphy.forecast import gbm
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.score import PCOLS, day_ci
from scripts.replay_cutoffs import FEATS, HIDDEN, R
from murphy.recommend import MAX_EARLIER_H, MIN_MISS, inbound_too_late, made_it, p_make, should_change

OUT = OUT_DIR / "alternatives"
KEYS = ["flight_id", "lead_h"]
BUFFERS = (30, 60, 120)
NAME = "l2abl_a"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)


def schedule_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    cur_path, alt_path = R / "schedule_current.parquet", R / "schedule_alt.parquet"
    if not cur_path.exists():
        rounds = json.loads((gbm.OUT / f"{NAME}.json").read_text())["best_rounds"]
        model, saved = gbm.truncated(gbm.OUT / f"{NAME}.joblib", rounds)
        for path, kw in ((cur_path, {}), (alt_path, {"rows": R / "alt_rows.parquet", "features": R / "alt_features"})):
            X, r, _ = gbm.load_split("tune", "a", carriers=saved["carriers"], **kw)
            s = pd.DataFrame(gbm.predict(model, X), columns=[f"s_{k}" for k in PCOLS]).astype("float32")
            pd.concat([r[KEYS], s], axis=1).to_parquet(path, index=False)
    return pd.read_parquet(cur_path), pd.read_parquet(alt_path).rename(columns={"flight_id": "alt", "lead_h": "alt_lead"})


def load():
    """Trips and alternatives, with `clean` false wherever a tail's previous leg is scheduled to
    land after this flight leaves. That overlap cannot be a plan, so it may be hindsight."""
    cols = [*KEYS, "flight_date", "outcome", "arr_delay_min", *PCOLS, *[f"c_{k}" for k in PCOLS]]
    cur = pd.read_parquet(R / "current.parquet", columns=cols)
    d = ["prior_delay_so_far", "prior_turn_min", "prior_status"]
    X, r, _ = gbm.load_split("tune", "ad")  # as recorded, for telling kinds of flight apart
    cur = cur.merge(pd.concat([r[KEYS], X[d].add_prefix("raw_").reset_index(drop=True)], axis=1), on=KEYS, how="left")
    X, r, _ = gbm.load_split("tune", "ad", features=FEATS)  # as the model and the rule of thumb see them
    cur = cur.merge(pd.concat([r[KEYS], X[d].reset_index(drop=True)], axis=1), on=KEYS, how="left")
    cur["clean"] = ~(cur.raw_prior_turn_min < 0)
    sc, sa = schedule_forecasts()
    cur = cur.merge(sc, on=KEYS, how="left")
    X, r, _ = gbm.load_split("tune", "ad", rows=R / "alt_rows.parquet", features=R / "alt_features")
    raw = pd.read_parquet(R / "alt_features" / ("d_raw.parquet" if HIDDEN else "d.parquet"), columns=["prior_turn_min"]).prior_turn_min.to_numpy()
    turn = pd.DataFrame({"alt": r.flight_id, "alt_lead": r.lead_h, "alt_clean": ~(raw < 0)})
    pairs = pd.read_parquet(R / "pairs.parquet").merge(pd.read_parquet(R / "alt_forecasts.parquet"), on=["alt", "alt_lead"], suffixes=("", "_a"))
    pairs = pairs.merge(sa, on=["alt", "alt_lead"], how="left").merge(turn, on=["alt", "alt_lead"], how="left")
    pairs["dep_rel_h"] = pairs.alt_lead - pairs.lead_h
    return cur, pairs


def chances(cur, pairs, buf, prefix=""):
    """Each trip with its chance and result, and each allowed alternative with its own."""
    cols = [prefix + k for k in PCOLS]
    t = cur[[*KEYS, "flight_date", "clean"]].assign(
        pc=p_make(cur[cols].fillna(0).to_numpy(), np.full(len(cur), buf)),
        made=made_it(cur.arr_delay_min, cur.outcome, np.full(len(cur), buf)))
    pk = pairs[(pairs.arr_ahead_min + buf >= 0) & (pairs.dep_rel_h >= -MAX_EARLIER_H)]
    slack = (pk.arr_ahead_min + buf).to_numpy()
    a = pd.DataFrame({"flight_id": pk.flight_id.to_numpy(), "lead_h": pk.lead_h.to_numpy(), "alt_lead": pk.alt_lead.to_numpy(),
                      "alt_clean": pk.alt_clean.to_numpy(), "pa": p_make(pk[cols].fillna(0).to_numpy(), slack),
                      "alt_made": made_it(pk.arr_delay_min, pk.outcome, slack)})
    return t, a


def tally(t, change, label) -> dict:
    """change: trips that switch; t.alt_made is the picked alternative's result. By whole days."""
    f = t[change]
    am = f.alt_made.astype(bool)
    diff = np.where(change, t.alt_made.fillna(False).astype(bool).astype(int) - t.made.astype(int), 0)
    ci = day_ci(pd.Series(diff), t.flight_date, n_boot=300)
    return {**label, "trips": len(t), "changes": len(f), "rescued": int((~f.made & am).sum()), "needless": int((f.made & am).sum()),
            "worse": int((f.made & ~am).sum()), "both_missed": int((~f.made & ~am).sum()),
            "net": 1000 * ci["diff"], "lo": 1000 * ci["lo"], "hi": 1000 * ci["hi"]}, diff


def pick(t, a, order, ascending):
    best = a.sort_values(order, ascending=ascending, kind="stable").drop_duplicates(KEYS)
    return t.merge(best.drop(columns="alt_clean"), on=KEYS, how="left")


def run(cur, pairs) -> None:
    rec, band = [], []
    for buf in BUFFERS:
        for scope in ("all",) if HIDDEN else ("all", "clean"):
            keep = cur.clean.to_numpy() if scope == "clean" else np.ones(len(cur), bool)
            c = cur[keep].reset_index(drop=True)
            p = pairs[pairs.alt_clean] if scope == "clean" else pairs
            t, a = chances(c, p, buf)
            m = pick(t, a, "pa", False)
            res, base = tally(m, should_change(m.pc, m.pa), {"buffer": buf, "scope": scope, "rule": "murphy"})
            res["flagged"] = int(((1 - m.pc) >= MIN_MISS).sum())
            rec.append(res)
            # Schedule-only model, same rule, its own forecasts.
            ts, als = chances(c, p, buf, "s_")
            s = pick(ts, als, "pa", False)
            res, diff = tally(s, should_change(s.pc, s.pa), {"buffer": buf, "scope": scope, "rule": "schedule"})
            res["flagged"] = int(((1 - s.pc) >= MIN_MISS).sum())
            ci = day_ci(pd.Series(base - diff), m.flight_date, n_boot=300)
            rec.append({**res, "murphy_minus": 1000 * ci["diff"], "mm_lo": 1000 * ci["lo"], "mm_hi": 1000 * ci["hi"]})
            # Rule of thumb: the inbound's lateness outlasts the turn by more than the deadline allows.
            th = pick(t, a, "alt_lead", True)
            flag = inbound_too_late(c.prior_delay_so_far, c.prior_turn_min, buf)
            res, diff = tally(th, flag & th.pa.notna().to_numpy(), {"buffer": buf, "scope": scope, "rule": "thumb"})
            res["flagged"] = int(flag.sum())
            ci = day_ci(pd.Series(base - diff), m.flight_date, n_boot=300)
            rec.append({**res, "murphy_minus": 1000 * ci["diff"], "mm_lo": 1000 * ci["lo"], "mm_hi": 1000 * ci["hi"],
                        "flag_true_miss": (~th.made[flag]).mean(), "both_flag": int((flag & ((1 - m.pc) >= MIN_MISS).to_numpy()).sum())})
            # The band: every risk range, with the change that would be offered there at the same gain.
            bin_ = pd.cut(1 - m.pc, [-1, 0.4, 0.5, 0.6, 2], right=False, labels=["<40", "40-50", "50-60", "60+"])
            for lead, gl in list(m.groupby("lead_h")) + [("all", m)]:
                ch = should_change(gl.pc, gl.pa, 0.0)
                for b in bin_.cat.categories:
                    w = (bin_[gl.index] == b).to_numpy()
                    r, _ = tally(gl[w], ch[w], {"buffer": buf, "scope": scope, "lead": lead, "band": b})
                    band.append({**r, "pred_miss": (1 - gl.pc[w]).mean(), "true_miss": (~gl.made[w]).mean()})
    rec, band = pd.DataFrame(rec), pd.DataFrame(band)
    rec.to_csv(R / "baselines.csv", index=False)
    band.to_csv(R / "band.csv", index=False)
    print(rec.round(3).to_string(index=False))
    print(band.round(3).to_string(index=False))


def turn_table(cur, buf=60) -> None:
    """How forecasts and results vary with the scheduled turn, and a rule that ignores the turn."""
    miss = ~made_it(cur.arr_delay_min, cur.outcome, np.full(len(cur), buf))
    risk = 1 - p_make(cur[PCOLS].to_numpy(), np.full(len(cur), buf))
    a = pd.DataFrame({"lead_h": cur.lead_h, "turn": cur.raw_prior_turn_min, "status": cur.raw_prior_status, "miss": miss, "risk": risk, "flag": risk >= MIN_MISS})
    known = a[a.turn.notna()].copy()
    known["bin"] = pd.cut(known.turn, [-1e9, -60, -15, -0.5, 14.5, 29.5, 44.5, 59.5, 119.5, 1e9],
                          labels=["<-60", "-60 to -16", "-15 to -1", "0-14", "15-29", "30-44", "45-59", "60-119", "120+"], right=False)
    t = known.groupby("bin", observed=True).agg(forecasts=("miss", "size"), pred_miss=("risk", "mean"), true_miss=("miss", "mean"),
                                                flagged=("flag", "sum"), inbound_not_departed=("status", lambda s: (s == 0).mean()))
    t.to_csv(R / "turn.csv")
    print(t.round(3).to_string())
    print(pd.DataFrame({"flagged": a.groupby("lead_h").flag.sum(), "flagged_overlap": a[a.turn < 0].groupby("lead_h").flag.sum(),
                        "overlap": a[a.turn < 0].groupby("lead_h").size()}).fillna(0).astype(int).to_string())
    kind = np.select([cur.raw_prior_turn_min.isna(), cur.raw_prior_delay_so_far.isna(), cur.raw_prior_turn_min < 0],
                     ["no tail", "delay blank", "overlap"], "normal")
    k = a.assign(kind=kind, cancelled=(cur.outcome == "cancelled").to_numpy()).groupby(["lead_h", "kind"]).agg(
        forecasts=("miss", "size"), pred_miss=("risk", "mean"), true_miss=("miss", "mean"), cancelled=("cancelled", "mean"), flagged=("flag", "sum"))
    k.to_csv(R / "aircraft_kinds.csv")
    print(k.round(3).to_string())
    naive = cur.raw_prior_delay_so_far > buf
    print(f"ignoring the turn: {naive.sum():,} flags, true miss {miss[naive].mean():.3f}", flush=True)


def main() -> None:
    cur, pairs = load()
    print(f"trips {len(cur):,}; overlapping turn {(~cur.clean).sum():,}; alternatives {pairs.alt_clean.mean():.4f} clean", flush=True)
    wide = (1 - p_make(cur[PCOLS].to_numpy(), np.full(len(cur), 30)) >= 0.3) | (1 - p_make(cur[[f"c_{k}" for k in PCOLS]].fillna(0).to_numpy(), np.full(len(cur), 30)) >= 0.3)
    for buf in BUFFERS:
        s = 1 - p_make(cur[[f"s_{k}" for k in PCOLS]].to_numpy(), np.full(len(cur), buf)) >= MIN_MISS
        f = inbound_too_late(cur.prior_delay_so_far, cur.prior_turn_min, buf)
        print(f"deadline {buf}: no saved alternatives for {(s & ~wide).sum():,} of {s.sum():,} schedule flags, {(f & ~wide).sum():,} of {f.sum():,} thumb flags", flush=True)
    turn_table(cur)
    run(cur, pairs)


if __name__ == "__main__":
    main()
