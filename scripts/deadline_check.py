"""Does the verdict band hold at every deadline the app offers? 2025 only.

The band (keep below 40% risk, too uncertain 40-60%, firm advice from 60%) was checked at 30, 60
and 120 minutes. The app's deadline chart offers 0 to 120 minutes in 15-minute
steps, so this repeats the same band tally at each step, on the saved replay of the app's model.
It first reproduces the saved 30/60/120 rows of band.csv as its check.
Writes deadline_band.csv beside band.csv; never overwrites band.csv.
Usage: MURPHY_MODEL=final_network PYTHONPATH=. python scripts/deadline_check.py
"""
import os, time
import numpy as np, pandas as pd

assert os.environ.get("MURPHY_MODEL") == "final_network", "run with MURPHY_MODEL=final_network (the app's model)"
from scripts.replay_baselines import R, chances, load, pick, tally
from murphy.recommend import should_change

STEPS = tuple(range(0, 121, 15))
BINS, LABELS = [-1, 0.4, 0.5, 0.6, 2], ["<40", "40-50", "50-60", "60+"]


def main() -> None:
    t0 = time.time()
    cur, pairs = load()
    print(f"loaded {len(cur):,} trips, {len(pairs):,} pairs in {time.time() - t0:.0f} s", flush=True)
    rows = []
    for buf in STEPS:
        t, a = chances(cur, pairs, buf)
        m = pick(t, a, "pa", False)
        bin_ = pd.cut(1 - m.pc, BINS, right=False, labels=LABELS)
        ch = should_change(m.pc, m.pa, 0.0)  # the change that would be offered there, at the same gain
        for b in LABELS:
            w = (bin_ == b).to_numpy()
            r, _ = tally(m[w], ch[w], {"buffer": buf, "band": b})
            rows.append({**r, "pred_miss": (1 - m.pc[w]).mean(), "true_miss": (~m.made[w]).mean()})
        print(f"deadline {buf:>3} min done, {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(R / "deadline_band.csv", index=False)

    # Check: the saved 30/60/120 rows (lead 'all', scope 'all') must come back exactly in their counts.
    saved = pd.read_csv(R / "band.csv").query("scope == 'all' and lead == 'all'")
    cols = ["trips", "changes", "rescued", "needless", "worse", "both_missed"]
    got = out[out.buffer.isin([30, 60, 120])].set_index(["buffer", "band"])[cols].sort_index()
    want = saved.set_index(["buffer", "band"])[cols].sort_index()
    print("reproduces band.csv:", got.equals(want.astype(got.dtypes.to_dict())))
    print(out[["buffer", "band", "trips", "changes", "rescued", "needless", "net", "lo", "hi", "pred_miss", "true_miss"]].round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
