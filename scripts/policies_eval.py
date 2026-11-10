"""Compare embeddings with BM25 for finding policy passages.

Each situation question is searched for Delta and for Southwest (pool: that carrier plus the
federal rules). A hit is a top-3 passage from a section marked relevant below. The marks are
AI marks, written before any search was run, from reading the sections; the author checks them.
A section is relevant if it states what the traveler may get or must do in that situation.

  embed   bge-small cosine
  bm25    BM25 over section label + text
  random  three passages drawn at random from the pool: the exact chance of a hit

Thin evidence, deliberate: Southwest's contract says nothing about hotels or meals, so
delay_care for WN can only hit the federal sections. Out of scope, deliberate: questions
no policy answers; their top scores are compared with the in-scope ones, to see whether a
score cutoff could say "no passage found".

Writes data/policies/eval.csv. Usage: python scripts/policies_eval.py
"""

from math import comb

import numpy as np
import pandas as pd

from murphy.policies import OUT, SITUATIONS, Index
from murphy.pastdays import ROOT

# situation -> {carrier: section substrings}; "federal" applies to both carriers
RELEVANT = {
    "cancelled": {"federal": ["14 CFR 260.6"],
                  "DL": ["RULE 19: FLIGHT DELAYS/CANCELLATIONS — A.", "RULE 22: REFUNDS — A."],
                  "WN": ["9. Flight Changes, Cancellations, Delays, and Diversions — b.", "4. Tickets — c."]},
    "delay_refund": {"federal": ["14 CFR 260.6"],
                     "DL": ["RULE 19: FLIGHT DELAYS/CANCELLATIONS — A.", "RULE 22: REFUNDS — A."],
                     "WN": ["9. Flight Changes, Cancellations, Delays, and Diversions — b.", "4. Tickets — c."]},
    "delay_care": {"federal": ["14 CFR 259.5", "14 CFR 259.9"],
                   "DL": ["RULE 19: FLIGHT DELAYS/CANCELLATIONS — B."],
                   "WN": []},  # thin evidence: nothing in Southwest's contract
    "schedule_change": {"federal": ["14 CFR 260.6"],
                        "DL": ["RULE 19: FLIGHT DELAYS/CANCELLATIONS — A.", "RULE 2: SCHEDULES"],
                        "WN": ["9. Flight Changes, Cancellations, Delays, and Diversions — a.",
                               "9. Flight Changes, Cancellations, Delays, and Diversions — b."]},
    "change_myself": {"federal": [],
                      "DL": ["RULE 21: REROUTING — A", "RULE 21: REROUTING — C.", "RULE 16: FARES"],
                      "WN": ["4. Tickets — c.", "3. Fares — e.", "3. Fares — a."]},  # 3a added 8 Nov 2026: its (2) covers customer changes
    "missed_connection": {"federal": [],
                          "DL": ["RULE 19: FLIGHT DELAYS/CANCELLATIONS — A."],
                          "WN": ["9. Flight Changes, Cancellations, Delays, and Diversions — a.",
                                 "9. Flight Changes, Cancellations, Delays, and Diversions — b."]},
    "tarmac": {"federal": ["14 CFR 259.4"],
               "DL": ["RULE 19: FLIGHT DELAYS/CANCELLATIONS — C."],
               "WN": []},
    "bumped": {"federal": ["14 CFR 250.5", "14 CFR 250.6", "14 CFR 250.8", "14 CFR 250.2b", "14 CFR 250.9"],  # 250.6 added 8 Nov 2026
               "DL": ["RULE 20: DENIED BOARDING"],
               "WN": ["9. Flight Changes, Cancellations, Delays, and Diversions — c."]},
}

OUT_OF_SCOPE = [
    "What will the weather be at my destination tomorrow?",
    "Which airport lounge has the best food?",
    "How early should I leave home to drive to the airport?",
]

CARRIERS = ("DL", "WN")


def relevant(p: pd.DataFrame, situation: str, carrier: str) -> np.ndarray:
    keys = RELEVANT[situation]
    ok = np.zeros(len(p), bool)
    for c in ("federal", carrier):
        for k in keys[c]:
            ok |= (p.carrier == c).to_numpy() & p.section.str.contains(k, regex=False).to_numpy()
    return ok


def main():
    p = pd.read_parquet(OUT / "passages.parquet")
    ix = Index(p, np.load(OUT / "embed.npy"))
    # every mark must match at least one section, or a typo would silently count as "no hit"
    for sit, by in RELEVANT.items():
        for c, keys in by.items():
            for k in keys:
                assert p[p.carrier == c].section.str.contains(k, regex=False).any(), (sit, c, k)

    rows = []
    for sit, (_, q) in SITUATIONS.items():
        for c in CARRIERS:
            rel = relevant(p, sit, c)
            pool = p.carrier.isin({"federal", c}).to_numpy()
            n, r = int(pool.sum()), int((rel & pool).sum())
            rows.append({"situation": sit, "carrier": c, "method": "random",
                         "hit3": 1 - comb(n - r, 3) / comb(n, 3), "first_rank": None, "top": "", "top_score": None})
            for m in ("embed", "bm25"):
                got = ix.search(q, c, m, k=len(p))["passages"]
                ids = [g["id"] for g in got]
                rel_ids = set(p.id[rel])
                ranks = [i + 1 for i, x in enumerate(ids) if x in rel_ids]
                rows.append({"situation": sit, "carrier": c, "method": m,
                             "hit3": float(bool(ranks) and ranks[0] <= 3),
                             "first_rank": ranks[0] if ranks else None,
                             "top": " | ".join(f"{g['section'][:60]} ({g['part']})" for g in got[:3]),
                             "top_score": got[0]["score"]})
    df = pd.DataFrame(rows)
    out = ROOT / "data" / "policies"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eval.csv", index=False)

    print("Hit in top 3, mean over", len(SITUATIONS) * len(CARRIERS), "situation-carrier pairs:")
    print(df.groupby("method").hit3.mean().round(3).to_string())
    print("\nPer pair (1 = hit; first relevant rank):")
    w = df[df.method != "random"].pivot_table(index=["situation", "carrier"], columns="method",
                                              values=["hit3", "first_rank"], aggfunc="first")
    print(w.to_string())

    e = df[df.method == "embed"].hit3.to_numpy(); b = df[df.method == "bm25"].hit3.to_numpy()
    only_e, only_b = int(((e == 1) & (b == 0)).sum()), int(((e == 0) & (b == 1)).sum())
    n = only_e + only_b
    p_two = min(1.0, 2 * sum(comb(n, i) for i in range(max(only_e, only_b), n + 1)) / 2 ** n) if n else 1.0
    print(f"\nPairs only embed hits: {only_e}; only BM25 hits: {only_b}; exact sign test, two-sided p = {p_two:.3f}")

    print("\nTop-1 embed score, in scope vs out of scope (Delta pool):")
    ins = df[(df.method == "embed")].top_score
    oos = [ix.search(q, "DL", "embed", k=1)["passages"][0]["score"] for q in OUT_OF_SCOPE]
    print(f"  in scope: min {ins.min():.4f}, median {ins.median():.4f}, max {ins.max():.4f}")
    print(f"  out of scope: {', '.join(f'{s:.4f}' for s in oos)}")
    for q in OUT_OF_SCOPE:
        g = ix.search(q, "DL", "embed", k=1)["passages"][0]
        print(f"  {q!r} -> {g['section'][:70]}")


if __name__ == "__main__":
    main()
