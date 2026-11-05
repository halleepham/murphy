"""The gradient-boosted forecast (docs/decisions/0008 Decision 6).

One HistGradientBoostingClassifier over the 16 outcome classes. Fit on
2023-24; settings and number of rounds chosen on 2025; 2026 scored once.
Early stopping is off: it holds out random rows, which breaks Decision 4.2.
Rounds are chosen by scoring 2025 at checkpoints during one fit, not by refitting.
The grid and ablation fit on a fixed quarter of the 2023-24 flights; the best
settings are then refit on all of them (Decision 7).

Commands (results in data/forecast/gbm/, resumable: finished runs are skipped):
  python -m murphy.forecast.gbm time     seconds per round on the real fit set
  python -m murphy.forecast.gbm grid     the settings below, all inputs, on 2025
  python -m murphy.forecast.gbm ablate   best settings on smaller input sets, on 2025
  python -m murphy.forecast.gbm final    best settings, every 2023-24 flight
  python -m murphy.forecast.gbm tune     scores and calibration on 2025
  python -m murphy.forecast.gbm test     2026, once, for models picked on 2025
"""

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from murphy.forecast import features as ft
from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.sample import CLASSES
from murphy.forecast.score import PCOLS, calibration, multiclass_brier, report

OUT = OUT_DIR / "gbm"
GROUPS = ("a", "ab", "abcd", "abcde")  # the ablation, simplest first
GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves}
        for lr in (0.05, 0.1) for leaves in (15, 31, 63)]
MAX_ROUNDS = 300
EVERY = 25
INFO = ["flight_id", "lead_h", "origin", "dest", "outcome", "split"]


def in_quarter(flight_id: pd.Series) -> np.ndarray:
    """A fixed quarter of flights. By flight, so lead-time rows stay together."""
    return pd.util.hash_pandas_object(flight_id, index=False).to_numpy() % 4 == 0


def load_split(split: str, groups: str = "abcde", rows=OUT_DIR / "rows.parquet",
               features=ft.OUT, carriers: list | None = None, quarter: bool = False):
    """Inputs and row info for one split only, so tuning cannot touch 2026."""
    r = pd.read_parquet(rows, columns=INFO)
    keep = (r.split == split).to_numpy()
    if quarter:
        keep = keep & in_quarter(r.flight_id)
    X = pd.concat([pd.read_parquet(features / f"{g}.parquet").drop(columns="rid")[keep]
                   for g in groups], axis=1)
    carriers = carriers or sorted(X.carrier.dropna().unique())
    X["carrier"] = pd.Categorical(X.carrier, categories=carriers).codes.astype("float32")
    X.loc[X.carrier < 0, "carrier"] = np.nan  # a carrier unseen in 2023-24: blank
    return X, r[keep].reset_index(drop=True), carriers


def make_model(rounds: int, **settings) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=rounds, early_stopping=False, categorical_features=["carrier"],
        random_state=0, **settings)


def aligned(p: np.ndarray, model_classes) -> np.ndarray:
    """Columns in CLASSES order. A class absent from the fit gets probability 0."""
    out = np.zeros((len(p), len(CLASSES)))
    for j, c in enumerate(model_classes):
        out[:, CLASSES.index(c)] = p[:, j]
    return out


def predict(model, X) -> np.ndarray:
    return aligned(model.predict_proba(X), model.classes_)


def staged_scores(model, X, outcome, every: int = EVERY) -> pd.DataFrame:
    """Multiclass Brier at each checkpoint round. One pass over the trees."""
    out = []
    for i, raw in enumerate(model.staged_decision_function(X), start=1):
        if i % every == 0 or i == model.n_iter_:
            out.append({"rounds": i, "brier_multi": multiclass_brier(
                aligned(softmax(raw, axis=1), model.classes_), outcome)})
    return pd.DataFrame(out)


def run(name: str, groups: str, settings: dict, rounds: int = MAX_ROUNDS,
        quarter: bool = True) -> dict:
    """Fit on 2023-24, score on 2025, save. Skipped if already done."""
    OUT.mkdir(exist_ok=True)
    done = OUT / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    X, r, carriers = load_split("fit", groups, quarter=quarter)
    t = time.time()
    model = make_model(rounds, **settings).fit(X, r.outcome)
    fit_s, n_fit = time.time() - t, len(X)
    del X, r
    Xt, rt, _ = load_split("tune", groups, carriers=carriers)
    scores = staged_scores(model, Xt, rt.outcome)
    best = scores.loc[scores.brier_multi.idxmin()]
    joblib.dump({"model": model, "carriers": carriers, "groups": groups}, OUT / f"{name}.joblib")
    result = {"name": name, "groups": groups, **settings, "fit_rows": n_fit, "fit_s": round(fit_s),
              "best_rounds": int(best.rounds), "tune_brier": float(best.brier_multi),
              "last_rounds": int(scores.rounds.iloc[-1]),
              "last_brier": float(scores.brier_multi.iloc[-1]),
              "curve": scores.to_dict("list")}
    done.write_text(json.dumps(result))
    print({k: v for k, v in result.items() if k != "curve"}, flush=True)
    return result


def results() -> pd.DataFrame:
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))
                         ]).drop(columns="curve")


def best_grid() -> pd.Series:
    r = results()
    return r[r.name.str.startswith("grid")].sort_values("tune_brier").iloc[0]


def truncated(path, rounds: int):
    """The saved model cut back to its best round on 2025."""
    saved = joblib.load(path)
    m = saved["model"]
    m._predictors = m._predictors[:rounds]
    return m, saved


def forecast_frame(name: str, split: str, rounds: int) -> pd.DataFrame:
    m, saved = truncated(OUT / f"{name}.joblib", rounds)
    X, r, _ = load_split(split, saved["groups"], carriers=saved["carriers"])
    return pd.concat([r, pd.DataFrame(predict(m, X), columns=PCOLS)], axis=1)


def score_all(split: str) -> None:
    res = results()
    best = best_grid()["name"]
    names = res[res.name.str.startswith("ablate") | res.name.isin([best, "final"])]
    clim = pd.read_parquet(OUT_DIR / "clim_origin.parquet")
    r = pd.read_parquet(OUT_DIR / "rows.parquet", columns=INFO)
    base = r[r.split == split].merge(clim, on=["flight_id", "lead_h"])
    out = [report(base.dropna(subset=PCOLS)).assign(model="clim_origin")]
    cal = []
    for _, row in names.iterrows():
        f = forecast_frame(row["name"], split, int(row.best_rounds))
        label = "abcde_full" if row["name"] == "final" else row.groups
        out.append(report(f).assign(model=label))
        for e in ("late", "cancelled"):
            cal.append(calibration(f, e).assign(model=label, event=e))
    for e in ("late", "cancelled"):
        cal.append(calibration(base.dropna(subset=PCOLS), e).assign(model="clim_origin", event=e))
    s = pd.concat(out)
    s.to_csv(OUT / f"scores_{split}.csv", index=False)
    pd.concat(cal).to_csv(OUT / f"calibration_{split}.csv", index=False)
    pd.set_option("display.width", 200)
    for m in ("brier_multi", "brier_late", "brier_cancelled"):
        print(m, s.pivot(index="group", columns="model", values=m).round(4), sep="\n")


def main(cmd: str) -> None:
    if cmd == "time":
        X, r, _ = load_split("fit")
        print(f"{X.shape[0]:,} rows, {X.shape[1]} inputs", flush=True)
        t = time.time()
        make_model(10, learning_rate=0.1, max_leaf_nodes=31).fit(X, r.outcome)
        print(f"10 rounds, 31 leaves: {time.time() - t:.0f} s", flush=True)
    elif cmd == "grid":
        for s in GRID:
            run(f"grid_lr{s['learning_rate']}_leaves{s['max_leaf_nodes']}", "abcde", s)
    elif cmd == "ablate":
        b = best_grid()
        s = {"learning_rate": float(b.learning_rate), "max_leaf_nodes": int(b.max_leaf_nodes)}
        for g in GROUPS[:-1]:  # the best grid model is already the all-inputs case
            run(f"ablate_{g}", g, s, rounds=int(b.last_rounds))
    elif cmd == "final":
        b = best_grid()
        s = {"learning_rate": float(b.learning_rate), "max_leaf_nodes": int(b.max_leaf_nodes)}
        run("final", "abcde", s, rounds=int(b.best_rounds), quarter=False)
    elif cmd in ("tune", "test"):
        score_all(cmd)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
