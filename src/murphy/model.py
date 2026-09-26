"""L2 -- a learned quantile forecaster, and the comparison against looking up.

Challenge 2's stated question is whether a trained model beats retrieving
similar flights. L1 is the empirical distribution over comparable flights, which
is what the application already shows. L2 is gradient-boosted quantile
regression over the same information, expressed as features.

Both are judged on identical test rows with identical metrics, and both may only
learn from 2023. Nothing that would be unknown before departure is used as a
feature: no departure delay, no taxi time, no actual times, no delay causes.
Those describe what happened, and a forecaster does not have them.

Run:  .venv/bin/python src/murphy/model.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "flights_full" / "**" / "*.parquet"
MODEL_DIR = ROOT / "data" / "processed" / "model"

TRAIN_YEAR, TEST_YEAR = 2023, 2024
QUANTILES = [0.10, 0.50, 0.90]

CATEGORICAL = ["carrier", "origin", "dest"]
NUMERIC = [
    "month", "day_of_week", "sched_dep_hour", "sched_duration_min", "distance_mi",
    "origin_temp_c", "origin_wind_ms", "origin_precip_mm", "origin_visibility_m",
]
FEATURES = CATEGORICAL + NUMERIC

# Only flights that operated carry an arrival delay, so only those can teach a
# model about lateness. Cancellation is a different question and a later one.
USABLE = "cancelled = 0 AND arr_delay_min IS NOT NULL"


def load(con, year: int, sample: int | None = None) -> dict:
    """Pull one year of usable flights, optionally a random sample of them.

    Training on all 6.6 million rows at depth 8 took over an hour on this
    machine without finishing, which makes the model impossible to iterate on.
    A two-million-row sample trains in minutes and is far more data than the
    model needs to separate 300 airports and 15 carriers -- the constraint is
    signal, not volume. The sample is random across the whole year, so seasonal
    and weekly structure is preserved.
    """
    cols = ", ".join(FEATURES)
    limit = ""
    if sample:
        limit = f"USING SAMPLE {sample} ROWS (reservoir, 0)"
    return con.execute(f"""
        SELECT {cols}, arr_delay_min
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE year(flight_date) = {year} AND {USABLE}
        {limit}
    """).fetchnumpy()


def encode(train: dict, test: dict) -> tuple[np.ndarray, np.ndarray]:
    """Turn the three categorical columns into codes the booster can split on.

    Categories are fixed from the training year. A carrier or airport that only
    appears in the test year becomes -1, which XGBoost treats as its own branch
    -- the honest answer for something the model has never seen.
    """
    train_cols, test_cols = [], []
    for name in CATEGORICAL:
        levels = {v: i for i, v in enumerate(sorted(set(train[name].tolist())))}
        train_cols.append(np.array([levels[v] for v in train[name].tolist()], dtype=np.float32))
        test_cols.append(np.array([levels.get(v, -1) for v in test[name].tolist()], dtype=np.float32))
    for name in NUMERIC:
        train_cols.append(np.asarray(train[name], dtype=np.float32))
        test_cols.append(np.asarray(test[name], dtype=np.float32))
    return np.column_stack(train_cols), np.column_stack(test_cols)


def lookup_baseline(con) -> dict:
    """L1: the empirical quantiles the application already computes.

    Applied here exactly as the app applies them, including the fallback when a
    match is too thin -- route and carrier and hour band first, then dropping
    the hour band, then the carrier, and finally the route itself.
    """
    for name, keys in [("g_exact", "origin, dest, carrier, dep_hour_bin"),
                       ("g_route_carrier", "origin, dest, carrier"),
                       ("g_route", "origin, dest"),
                       ("g_all", "1")]:
        group = keys if keys != "1" else ""
        select_keys = f"{keys}," if keys != "1" else ""
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT {select_keys}
                   quantile_cont(arr_delay_min, 0.10) AS p10,
                   quantile_cont(arr_delay_min, 0.50) AS p50,
                   quantile_cont(arr_delay_min, 0.90) AS p90,
                   count(*) AS n
            FROM read_parquet('{PARQUET}', hive_partitioning=true)
            WHERE year(flight_date) = {TRAIN_YEAR} AND {USABLE}
            {"GROUP BY " + group if group else ""}
            {"HAVING count(*) >= 10" if group else ""}
        """)

    return con.execute(f"""
        SELECT
            coalesce(e.p10, rc.p10, r.p10, a.p10) AS p10,
            coalesce(e.p50, rc.p50, r.p50, a.p50) AS p50,
            coalesce(e.p90, rc.p90, r.p90, a.p90) AS p90
        FROM read_parquet('{PARQUET}', hive_partitioning=true) t
        LEFT JOIN g_exact e ON e.origin = t.origin AND e.dest = t.dest
                           AND e.carrier = t.carrier AND e.dep_hour_bin = t.dep_hour_bin
        LEFT JOIN g_route_carrier rc ON rc.origin = t.origin AND rc.dest = t.dest
                                    AND rc.carrier = t.carrier
        LEFT JOIN g_route r ON r.origin = t.origin AND r.dest = t.dest
        CROSS JOIN g_all a
        WHERE year(t.flight_date) = {TEST_YEAR} AND {USABLE.replace("cancelled", "t.cancelled").replace("arr_delay_min", "t.arr_delay_min")}
    """).fetchnumpy()


def pinball(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    diff = actual - predicted
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score(name: str, actual, p10, p50, p90) -> dict:
    return {
        "method": name,
        "pinball_p10": pinball(actual, p10, 0.10),
        "pinball_p50": pinball(actual, p50, 0.50),
        "pinball_p90": pinball(actual, p90, 0.90),
        "pinball_mean": (pinball(actual, p10, 0.10) + pinball(actual, p50, 0.50)
                         + pinball(actual, p90, 0.90)) / 3,
        "coverage_80": float(np.mean((actual >= p10) & (actual <= p90))),
        "median_abs_error": float(np.median(np.abs(actual - p50))),
        "interval_width": float(np.mean(p90 - p10)),
    }


def main():
    ap = argparse.ArgumentParser(description="Train L2 and compare it against L1.")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--sample", type=int, default=2_000_000,
                    help="training rows to sample; 0 uses all of them")
    a = ap.parse_args()

    con = duckdb.connect()
    print(f"loading: train {TRAIN_YEAR}, test {TEST_YEAR}")
    train = load(con, TRAIN_YEAR, a.sample or None)
    test = load(con, TEST_YEAR)
    y_train = np.asarray(train["arr_delay_min"], dtype=np.float32)
    y_test = np.asarray(test["arr_delay_min"], dtype=np.float32)
    X_train, X_test = encode(train, test)
    print(f"  train {X_train.shape[0]:,} rows  test {X_test.shape[0]:,} rows  "
          f"{X_train.shape[1]} features")

    print("\nL1: empirical quantiles over comparable flights")
    l1 = lookup_baseline(con)
    l1_p10 = np.asarray(l1["p10"], dtype=np.float32)
    l1_p50 = np.asarray(l1["p50"], dtype=np.float32)
    l1_p90 = np.asarray(l1["p90"], dtype=np.float32)
    assert len(l1_p50) == len(y_test), (len(l1_p50), len(y_test))

    print(f"\nL2: gradient-boosted quantile regression, {a.rounds} rounds")
    types = ["c"] * len(CATEGORICAL) + ["q"] * len(NUMERIC)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES,
                         feature_types=types, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, feature_names=FEATURES,
                        feature_types=types, enable_categorical=True)

    started = time.time()
    booster = xgb.train(
        {"objective": "reg:quantileerror", "quantile_alpha": QUANTILES,
         "tree_method": "hist", "max_depth": 6, "eta": 0.1,
         "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8,
         "max_bin": 128, "max_cat_to_onehot": 1, "seed": 0, "nthread": 0},
        dtrain, num_boost_round=a.rounds,
        evals=[(dtrain, "train")], verbose_eval=50,
    )
    print(f"  trained in {time.time() - started:.0f}s")

    pred = booster.predict(dtest)
    l2_p10, l2_p50, l2_p90 = pred[:, 0], pred[:, 1], pred[:, 2]

    results = [score("L1 lookup", y_test, l1_p10, l1_p50, l1_p90),
               score("L2 model", y_test, l2_p10, l2_p50, l2_p90)]

    print(f"\n{'':<12}{'pinball':>9}{'p10':>8}{'p50':>8}{'p90':>8}"
          f"{'cover80':>9}{'medAE':>8}{'width':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<12}{r['pinball_mean']:>9.3f}{r['pinball_p10']:>8.3f}"
              f"{r['pinball_p50']:>8.3f}{r['pinball_p90']:>8.3f}"
              f"{r['coverage_80']:>8.1%}{r['median_abs_error']:>8.1f}"
              f"{r['interval_width']:>8.1f}")

    better = (results[0]["pinball_mean"] - results[1]["pinball_mean"]) / results[0]["pinball_mean"]
    print(f"\nL2 pinball loss is {better:+.1%} versus L1 "
          f"({'lower is better -- L2 wins' if better > 0 else 'L1 wins'})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(MODEL_DIR / "l2_quantile.json")
    (MODEL_DIR / "comparison.json").write_text(json.dumps(results, indent=2))

    # The category codes are part of the model. Without them a saved booster is
    # unusable, because the integers it split on mean nothing on their own.
    (MODEL_DIR / "categories.json").write_text(json.dumps(
        {name: sorted(set(train[name].tolist())) for name in CATEGORICAL}, indent=2))

    # The model expects weather, and a traveller's flight has none -- there is no
    # forecast for a date months away. So the application predicts under typical
    # conditions for that airport and month, taken from history, and says so.
    # An assumption stated is worth more than a number that pretends otherwise.
    con.execute(f"""
        COPY (
            SELECT origin, month,
                   median(origin_temp_c)        AS origin_temp_c,
                   median(origin_wind_ms)       AS origin_wind_ms,
                   median(origin_precip_mm)     AS origin_precip_mm,
                   median(origin_visibility_m)  AS origin_visibility_m
            FROM read_parquet('{PARQUET}', hive_partitioning=true)
            WHERE year(flight_date) = {TRAIN_YEAR} AND {USABLE}
            GROUP BY 1, 2
        ) TO '{MODEL_DIR / "typical_weather.parquet"}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT origin, dest,
                   CAST(median(sched_duration_min) AS INT) AS sched_duration_min,
                   CAST(median(distance_mi) AS INT)        AS distance_mi
            FROM read_parquet('{PARQUET}', hive_partitioning=true)
            WHERE year(flight_date) = {TRAIN_YEAR}
            GROUP BY 1, 2
        ) TO '{MODEL_DIR / "route_norms.parquet"}' (FORMAT PARQUET)
    """)
    print(f"\nsaved model, categories, typical weather and route norms to {MODEL_DIR}")

    gain = booster.get_score(importance_type="gain")
    print("\nwhat the model leans on (gain):")
    for name, value in sorted(gain.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:22} {value:,.0f}")


if __name__ == "__main__":
    main()
