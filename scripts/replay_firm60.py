"""Murphy against the rule of thumb under the app's rule: firm "consider changing" from 60%.

replay_baselines.py scores Murphy with the replay's 50% cutoff. This reuses its loading and tallying,
changes only Murphy's cutoff to FIRM_FROM, and writes baselines_firm60.csv beside baselines.csv. The
rule of thumb does not use Murphy's cutoff, so its row is the same as there; it is recomputed here
for the paired difference. The app is not touched.
Usage: MURPHY_MODEL=final_network PYTHONPATH=. python scripts/replay_firm60.py
"""
import numpy as np, pandas as pd
from murphy.forecast.score import day_ci
from murphy.recommend import FIRM_FROM, inbound_too_late, should_change
from scripts.replay_baselines import BUFFERS, R, chances, load, pick, tally


def main() -> None:
    cur, pairs = load()
    rec = []
    for buf in BUFFERS:
        t, a = chances(cur, pairs, buf)
        m = pick(t, a, "pa", False)
        res, base = tally(m, should_change(m.pc, m.pa, FIRM_FROM), {"buffer": buf, "rule": "murphy_firm60"})
        res["flagged"] = int(((1 - m.pc) >= FIRM_FROM).sum())
        rec.append(res)
        th = pick(t, a, "alt_lead", True)
        flag = inbound_too_late(cur.prior_delay_so_far, cur.prior_turn_min, buf)
        res, diff = tally(th, flag & th.pa.notna().to_numpy(), {"buffer": buf, "rule": "thumb"})
        res["flagged"] = int(flag.sum())
        ci = day_ci(pd.Series(base - diff), m.flight_date, n_boot=300)
        rec.append({**res, "murphy_minus": 1000 * ci["diff"], "mm_lo": 1000 * ci["lo"], "mm_hi": 1000 * ci["hi"]})
    out = pd.DataFrame(rec)
    out.to_csv(R / "baselines_firm60.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
