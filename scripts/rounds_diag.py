"""Why does the 2025 score worsen with more rounds? (docs/decisions/0010 Finding 1)
Saved models only; no fitting.

Usage: PYTHONPATH=. python scripts/rounds_diag.py grid_lr0.1_leaves63,grid_lr0.05_leaves63,final 0
The last number caps rows per set (0 = all); an optional third argument sets the output folder.
About 4 minutes for those three models. Results: data/forecast/gbm/diag/.

Never loads the 2026 rows. Tests three explanations:
 (1) ordinary overfit: train keeps improving, unseen rows worsen, gradually
 (2) time shift: unseen 2023-24 flights keep improving, only 2025 worsens
 (3) blow-up: a few rows get extreme wrong probabilities (rare class, huge leaf)
"""
from pathlib import Path
import sys, joblib, numpy as np, pandas as pd
from scipy.special import softmax
from murphy.forecast import features as ft
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.gbm import OUT, INFO, aligned
from murphy.forecast.sample import CLASSES

MODELS = sys.argv[1].split(",")
N = int(sys.argv[2])  # rows cap per set, for a dry run
DIAG = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT / "diag"
DIAG.mkdir(exist_ok=True)
CKPT = [25, 50, 75, 100, 150, 200, 275]

r = pd.read_parquet(OUT_DIR / "rows.parquet", columns=INFO)
h = pd.util.hash_pandas_object(r.flight_id, index=False).to_numpy()
masks = {"train_q": (r.split == "fit").to_numpy() & (h % 16 == 0),   # in the quarter
         "unseen_2324": (r.split == "fit").to_numpy() & (h % 16 == 1),  # not in the quarter
         "tune_2025": (r.split == "tune").to_numpy() & (h % 8 == 0)}
for k in masks:
    idx = np.flatnonzero(masks[k])[:N] if N else np.flatnonzero(masks[k])
    m = np.zeros(len(r), bool); m[idx] = True; masks[k] = m
union = np.logical_or.reduce(list(masks.values()))
assert (r.split[union] != "test").all()
X = pd.concat([pd.read_parquet(ft.OUT / f"{g}.parquet").drop(columns="rid")[union] for g in "abcde"], axis=1)
ri = r[union].reset_index(drop=True)
sets = {k: masks[k][union] for k in masks}
print({k: int(v.sum()) for k, v in sets.items()}, flush=True)
del r
rare = pd.Series(ri.outcome[sets["train_q"]]).value_counts(normalize=True).reindex(CLASSES).fillna(0)
print("class share, train sample:\n", rare.round(5).to_string(), flush=True)
Y = (ri.outcome.to_numpy()[:, None] == np.array(CLASSES)[None, :]).astype(float)

rows, leaves, tails = [], [], []
for name in MODELS:
    saved = joblib.load(OUT / f"{name}.joblib")
    m = saved["model"]
    # (a) leaf sizes, no data needed. Value already includes the learning rate.
    for i, pred in enumerate(m._predictors, start=1):
        for k, tree in enumerate(pred):
            v = tree.nodes["value"][tree.nodes["is_leaf"].astype(bool)]
            leaves.append((name, i, m.classes_[k], float(np.abs(v).max())))
    Xm = X.copy()
    Xm["carrier"] = pd.Categorical(Xm.carrier, categories=saved["carriers"]).codes.astype("float32")
    Xm.loc[Xm.carrier < 0, "carrier"] = np.nan
    first = {}
    for i, raw in enumerate(m.staged_decision_function(Xm), start=1):
        if i not in CKPT and i != m.n_iter_:
            continue
        p = aligned(softmax(raw, axis=1), m.classes_)
        per = (p - Y) ** 2
        loss = per.sum(axis=1)
        for s, keep in sets.items():
            if name == "final" and s != "tune_2025":
                # final was fit on all 2023-24, so both samples are train rows
                s = "train_q" if s == "train_q" else None
                if s is None: continue
                keep = sets["train_q"]
            l, pp, yy = loss[keep], p[keep], Y[keep]
            top = pp.argmax(axis=1)
            row = {"model": name, "set": s, "round": i, "brier": l.mean(),
                   "logloss": -np.log(np.clip((pp * yy).sum(axis=1), 1e-15, 1)).mean(),
                   "sharp_p>.9": (pp.max(axis=1) > .9).mean(),
                   "hit_when_sharp": (yy[np.arange(len(yy)), top][pp.max(axis=1) > .9]).mean() if (pp.max(axis=1) > .9).any() else np.nan,
                   "worst_row": l.max()}
            row.update({f"c:{c}": v for c, v in zip(CLASSES, per[keep].mean(axis=0))})
            rows.append(row)
            if s not in first:
                first[s] = (i, l.copy())
            else:
                d = np.sort(l - first[s][1])[::-1]
                tot = d.sum()
                n = len(d)
                tails.append({"model": name, "set": s, "from": first[s][0], "to": i, "rise": tot / n,
                              "top0.1%_rows": d[:max(1, n // 1000)].sum() / n,
                              "top1%_rows": d[:max(1, n // 100)].sum() / n,
                              "rest": d[n // 100:].sum() / n,
                              "rows_worse_by>.5": (d > .5).mean()})
    print("done", name, flush=True)

pd.DataFrame(rows).to_csv(DIAG / "rounds_scores.csv", index=False)
pd.DataFrame(leaves, columns=["model", "round", "cls", "max_abs_leaf"]).to_csv(DIAG / "rounds_leaves.csv", index=False)
pd.DataFrame(tails).to_csv(DIAG / "rounds_tails.csv", index=False)
