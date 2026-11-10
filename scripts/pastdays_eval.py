"""Compare five ways of finding similar past airport-days.

Every 2025 airport-day is a query. Each method picks earlier days at the same
airport; its prediction is their mean late share (15+ min late or cancelled).
Scored against the query day's actual late share: mean absolute error, and
Spearman rank correlation pooled and within airport. A paired bootstrap over
query days gives a 95% interval for each method's MAE minus the embeddings'.

  embed    bge-small, top 10 by cosine
  bm25     BM25 over the same sentences, top 10
  numeric  nearest neighbors on standardized numbers, top 10
  month    mean of all earlier days at the airport in the same calendar month
  random   10 random earlier days (seed 0)

Writes data/pastdays/eval_predictions.parquet and eval_summary.csv.
Usage: python scripts/pastdays_eval.py [--sample N]
"""

import argparse
import time
from datetime import date

import numpy as np
import pandas as pd

from murphy.pastdays import OUT, Index, K, numeric

METHODS = ("embed", "bm25", "numeric", "month", "random")


def predictions(ix: Index, queries: pd.DataFrame, rng) -> pd.DataFrame:
    c, rows = ix.c, []
    month = np.array([d.month for d in c.date])
    for qi, q in queries.iterrows():
        cand = (c.airport == q.airport).to_numpy() & (ix._ord < q.date.toordinal())
        row = {"qi": qi, "airport": q.airport, "date": q.date, "actual": q.late_share}
        for m in ("embed", "bm25", "numeric"):
            hit = ix.search(q.airport, q.date, q.text, numeric(q), m, qvec=ix.embed[qi])
            row[m] = hit.late_share.mean()
            row[f"{m}_score"] = hit.score.mean()
            row[f"{m}_days"] = ",".join(str(d) for d in hit.date)
        same = cand & (month == q.date.month)
        row["month"] = c.late_share[same].mean() if same.any() else np.nan
        pick = rng.choice(np.flatnonzero(cand), size=K, replace=False)
        row["random"] = c.late_share.iloc[pick].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summary(p: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    err = {m: (p[m] - p.actual).abs().to_numpy() for m in METHODS}
    boots = rng.integers(0, len(p), size=(n_boot, len(p)))
    out = []
    for m in METHODS:
        d = err[m] - err["embed"]
        b = d[boots].mean(1)
        within = p.groupby("airport")[[m, "actual"]].corr(method="spearman").xs(m, level=1)["actual"]
        out.append({"method": m, "mae": err[m].mean(),
                    "spearman_pooled": p[m].corr(p.actual, method="spearman"),
                    "spearman_within_airport": within.mean(),
                    "mae_minus_embed": d.mean(),
                    "ci_low": np.quantile(b, 0.025), "ci_high": np.quantile(b, 0.975)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="query this many 2025 days (seed 0)")
    a = ap.parse_args()
    corpus = pd.read_parquet(OUT / "corpus.parquet")
    ix = Index(corpus, np.load(OUT / "embed.npy"))
    q = ix.c[[d.year == 2025 for d in ix.c.date]]
    if a.sample:
        q = q.sample(a.sample, random_state=0)
    t = time.time()
    p = predictions(ix, q, np.random.default_rng(0))
    print(f"{len(p)} queries in {time.time() - t:.0f}s")
    s = summary(p)
    p.to_parquet(OUT / "eval_predictions.parquet")
    s.to_csv(OUT / "eval_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
