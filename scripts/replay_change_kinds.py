"""What kind of change does the chosen rule advise (docs/decisions/0015)? 2025 only.

The rule of 0014 Decision 4 at a 60-minute deadline: firm from 60% risk, 15-point gain,
highest chance leaving at most 2 h earlier. Splits the changes by whether the alternative
has the same marketing carrier, and reports the gain in chance. Prints 4,489 as its check.
Usage: PYTHONPATH=. python scripts/replay_change_kinds.py
"""
import duckdb, numpy as np, pandas as pd
from murphy.forecast.score import PCOLS
from murphy.recommend import made_it, p_make, should_change

R = "data/forecast/alternatives/replay_final_clean/"
BUF, MISS, GAIN, NEAR_H = 60, 0.60, 0.15, 2

cur = pd.read_parquet(R + "current.parquet", columns=["flight_id", "lead_h", "origin", "dest", "outcome", "arr_delay_min", *PCOLS])
pairs = pd.read_parquet(R + "pairs.parquet").merge(
    pd.read_parquet(R + "alt_forecasts.parquet", columns=["alt", "alt_lead", "outcome", "arr_delay_min", *PCOLS]), on=["alt", "alt_lead"])
pk = pairs[pairs.arr_ahead_min + BUF >= 0]
slack = (pk.arr_ahead_min + BUF).to_numpy()
alts = pd.DataFrame({"flight_id": pk.flight_id.to_numpy(), "lead_h": pk.lead_h.to_numpy(), "alt": pk.alt.to_numpy(),
                     "pa": p_make(pk[PCOLS].to_numpy(), slack), "alt_made": made_it(pk.arr_delay_min, pk.outcome, slack),
                     "dep_rel_h": (pk.alt_lead - pk.lead_h).to_numpy()})
best = alts[alts.dep_rel_h >= -NEAR_H].sort_values("pa", ascending=False, kind="stable").drop_duplicates(["flight_id", "lead_h"])
t = cur[["flight_id", "lead_h", "origin", "dest"]].assign(
    pc=p_make(cur[PCOLS].to_numpy(), np.full(len(cur), BUF)),
    made=made_it(cur.arr_delay_min, cur.outcome, np.full(len(cur), BUF))).merge(best, on=["flight_id", "lead_h"], how="left")
f = t[should_change(t.pc, t.pa, MISS, GAIN)].copy()
print("changes (expect 4489):", len(f))

c = duckdb.sql("SELECT flight_id, marketing_carrier AS mk FROM read_parquet('data/flights/2025_*.parquet')").df()
f = f.merge(c, on="flight_id").merge(c.rename(columns={"flight_id": "alt", "mk": "alt_mk"}), on="alt")
f["same"] = f.mk == f.alt_mk
am = f.alt_made.astype(bool)
f["result"] = np.select([~f.made & am, f.made & am, f.made & ~am], ["rescued", "needless", "worse"], "both_missed")
print(pd.crosstab(f.same, f.result, margins=True))
print(f.groupby("lead_h").same.agg(["size", "mean"]).round(4))
gain = f.pa - f.pc
print("gain:", gain.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("break-even, as a multiple of cost:", (1 / gain).quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
print("routes:", f.groupby(["origin", "dest"]).ngroups, "route and carrier:", f.groupby(["origin", "dest", "alt_mk"]).ngroups)
