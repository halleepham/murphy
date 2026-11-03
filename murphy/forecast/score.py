"""Scores for Step 2 (docs/decisions/0003 Decision 1, 0008): Brier and calibration.

multiclass Brier  mean over forecasts of sum_k (p_k - y_k)^2. 0 is perfect;
                  2 is the worst. Lower is better.
event Brier       mean (p - y)^2 for one yes/no event built by adding classes.
                  "late" = the DOT definition of not on time: arriving
                  15+ minutes late, cancelled, or diverted.
calibration       forecasts binned by predicted probability; in each bin,
                  mean predicted vs observed share. Close = calibrated.
coverage          (0008) weather at both ends, one end, or neither, using
                  the 30 airports in murphy.datalayer.weather.AIRPORTS.
"""

import numpy as np
import pandas as pd

from murphy.datalayer.weather import AIRPORTS
from murphy.forecast.climatology import col
from murphy.forecast.sample import CLASSES

PCOLS = [col(c) for c in CLASSES]
EVENTS = {
    "late": [c for c in CLASSES if c not in ("early", "0-15")],
    "cancelled": ["cancelled"],
}


def multiclass_brier(p: np.ndarray, outcome: pd.Series) -> float:
    y = (outcome.to_numpy()[:, None] == np.array(CLASSES)[None, :]).astype(float)
    return float(((p - y) ** 2).sum(axis=1).mean())


def event_prob(df: pd.DataFrame, event: str) -> np.ndarray:
    return df[[col(c) for c in EVENTS[event]]].sum(axis=1).to_numpy()


def event_brier(df: pd.DataFrame, event: str) -> float:
    y = df["outcome"].isin(EVENTS[event]).to_numpy()
    return float(((event_prob(df, event) - y) ** 2).mean())


def calibration(df: pd.DataFrame, event: str, bins: int = 10) -> pd.DataFrame:
    p = event_prob(df, event)
    y = df["outcome"].isin(EVENTS[event]).to_numpy()
    b = np.minimum((p * bins).astype(int), bins - 1)
    t = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin")
    return t.agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()


def coverage(df: pd.DataFrame) -> pd.Series:
    n = df["origin"].isin(AIRPORTS).astype(int) + df["dest"].isin(AIRPORTS).astype(int)
    return n.map({2: "both", 1: "one", 0: "neither"})


def report(df: pd.DataFrame) -> pd.DataFrame:
    """Scores overall and by lead time and weather coverage.

    df: one row per forecast, with outcome, lead_h, origin, dest and PCOLS.
    """
    df = df.assign(coverage=coverage(df))
    groups = [("all", df)]
    groups += [(f"lead {h}h", g) for h, g in df.groupby("lead_h")]
    groups += [(f"weather {c}", g) for c, g in df.groupby("coverage")]
    rows = []
    for name, g in groups:
        rows.append({"group": name, "n": len(g),
                     "brier_multi": multiclass_brier(g[PCOLS].to_numpy(), g["outcome"]),
                     **{f"brier_{e}": event_brier(g, e) for e in EVENTS}})
    return pd.DataFrame(rows)


def score_files(rows, forecasts: dict, split: str = "test") -> pd.DataFrame:
    """Score each forecast file on one split. Missing probabilities (no
    history) are counted and dropped, never filled."""
    r = pd.read_parquet(rows, columns=["flight_id", "lead_h", "origin", "dest",
                                       "outcome", "split"])
    r = r[r.split == split]
    out = []
    for name, path in forecasts.items():
        f = r.merge(pd.read_parquet(path), on=["flight_id", "lead_h"], how="left")
        if name == "uniform":
            f[PCOLS] = 1 / len(CLASSES)
        missing = int(f[PCOLS].isna().any(axis=1).sum())
        out.append(report(f.dropna(subset=PCOLS)).assign(model=name, missing=missing))
    return pd.concat(out)


if __name__ == "__main__":
    from murphy.forecast.climatology import OUT_DIR
    rows = OUT_DIR / "rows.parquet"
    files = {"uniform": OUT_DIR / "clim_global.parquet",
             "clim_global": OUT_DIR / "clim_global.parquet",
             "clim_origin": OUT_DIR / "clim_origin.parquet"}
    s = score_files(rows, files)
    s.to_csv(OUT_DIR / "scores_baselines.csv", index=False)
    pd.set_option("display.width", 160)
    print(s.pivot(index="group", columns="model", values=["brier_multi", "brier_late"]).round(4))
    print(s.groupby("model")[["n", "missing"]].first())
