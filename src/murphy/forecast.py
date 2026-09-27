"""Serve the L2 model for a single flight the traveller is actually taking.

The model was trained on flights whose weather was known, because it had already
happened. A traveller asking about a flight in November has no such thing -- and
no forecast exists months ahead either. So the prediction is made under *typical*
conditions for that airport and month, and the application says so rather than
implying it knows the weather.

That is a real limitation, not a workaround to hide: on a day that turns out
badly, this prediction will be optimistic, and the honest place to fix it is a
weather forecast at serving time, which is Challenge 3 work.

Route distance and scheduled duration come from history too, since a pasted
confirmation carries neither.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murphy.model import CATEGORICAL, FEATURES, NUMERIC

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "processed" / "model"


class ModelUnavailable(RuntimeError):
    """The trained model is not on disk. Run src/murphy/model.py to build it."""


@lru_cache(maxsize=1)
def _loaded():
    import xgboost as xgb

    model_file = MODEL_DIR / "l2_quantile.json"
    if not model_file.exists():
        raise ModelUnavailable(
            "No trained model found. Build it with: "
            ".venv/bin/python src/murphy/model.py"
        )
    booster = xgb.Booster()
    booster.load_model(model_file)
    categories = json.loads((MODEL_DIR / "categories.json").read_text())
    codes = {name: {v: i for i, v in enumerate(values)}
             for name, values in categories.items()}

    con = duckdb.connect()
    weather = {(r[0], r[1]): r[2:] for r in con.execute(
        f"SELECT * FROM read_parquet('{MODEL_DIR / 'typical_weather.parquet'}')"
    ).fetchall()}
    norms = {(r[0], r[1]): (r[2], r[3]) for r in con.execute(
        f"SELECT * FROM read_parquet('{MODEL_DIR / 'route_norms.parquet'}')"
    ).fetchall()}
    return booster, codes, weather, norms


def available() -> bool:
    try:
        _loaded()
        return True
    except Exception:
        return False


def predict(carrier: str, origin: str, dest: str, month: int,
            day_of_week: int, hour: int) -> dict | None:
    """Predicted arrival-delay quantiles under typical conditions.

    Returns None when the route has no history to draw a distance and duration
    from -- the same reticence the retrieval layer shows.
    """
    import xgboost as xgb

    booster, codes, weather, norms = _loaded()

    route = norms.get((origin, dest))
    if route is None:
        return None
    duration, distance = route

    wx = weather.get((origin, month))
    if wx is None:
        return None
    temp, wind, precip, visibility = wx

    row = {
        "carrier": codes["carrier"].get(carrier, -1),
        "origin": codes["origin"].get(origin, -1),
        "dest": codes["dest"].get(dest, -1),
        "month": month, "day_of_week": day_of_week, "sched_dep_hour": hour,
        "sched_duration_min": duration, "distance_mi": distance,
        "origin_temp_c": temp, "origin_wind_ms": wind,
        "origin_precip_mm": precip, "origin_visibility_m": visibility,
    }
    x = np.array([[float(row[f]) for f in FEATURES]], dtype=np.float32)
    types = ["c"] * len(CATEGORICAL) + ["q"] * len(NUMERIC)
    matrix = xgb.DMatrix(x, feature_names=FEATURES, feature_types=types,
                         enable_categorical=True)
    p10, p50, p90 = booster.predict(matrix)[0]

    return {"p10": round(float(p10)), "p50": round(float(p50)),
            "p90": round(float(p90)),
            "unseen_carrier": carrier not in codes["carrier"],
            "unseen_route": origin not in codes["origin"] or dest not in codes["dest"]}
