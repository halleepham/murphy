"""Which structural changes actually add signal? Tested before building the model.

The baseline diagnostic (06) established that delay propagates and that one hop of
flow-weighted graph convolution adds 22% at the six-hour horizon. This script asks
which *changes to that structure* are worth paying for, using the same ridge probe so
every answer is comparable to the baseline and to each other.

Four candidates, each a hypothesis about why the simple version is leaving signal on
the table:

  P1  travel-time edges   the flat version compares A at 2pm against B at 3pm whether
                          the flight takes one hour or five. Physically wrong. Shift
                          each neighbour by its actual flight time.
  P2  arrivals            the panel is departure delay only. Lateness *landing* at an
                          airport is the other half of propagation and is missing.
  P3  rotation            flow counts say "A sends many planes to B". The PRECEDES
                          chain says "*this aircraft* goes A to B to C", which is a
                          far sharper dependency and is already built and validated.
  P4  weather + traffic   both sit in the table already and neither is an outcome.

Every feature is strictly past-observed relative to the target hour, except the
scheduled departure count, which is known in advance by construction.

Run:  PYTHONPATH=src .venv/bin/python notebooks/07_stgnn_structure_probes.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

import duckdb
import numpy as np

FLIGHTS = "data/processed/flights_full/**/*.parquet"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "stgnn_structure_probes.json"

TOP_AIRPORTS = 100
LATE_MIN = 15
MIN_DEPARTURES = 5
LAGS = 3
HORIZONS = (1, 3, 6)
TRAIN_THROUGH = "2024-01-01"


def ridge(X, y, lam=1.0):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    P = np.eye(X1.shape[1]) * lam
    P[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + P, X1.T @ y)


def r2(y, pred):
    return 1.0 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main() -> None:
    con = duckdb.connect()
    airports = [r[0] for r in con.execute(f"""
        SELECT origin, count(*) n FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 GROUP BY origin ORDER BY n DESC LIMIT {TOP_AIRPORTS}
    """).fetchall()]
    inlist = ",".join("'" + a + "'" for a in airports)
    a_index = {a: i for i, a in enumerate(airports)}

    print("building the airport-hour panel ...")
    # Departure lateness, arrival lateness, weather and the schedule, in one pass each.
    dep = con.execute(f"""
        SELECT origin AS ap, date_trunc('hour', dep_utc) AS h, count(*) AS n,
               avg(CASE WHEN dep_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND dep_delay_min IS NOT NULL AND origin IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()
    arr = con.execute(f"""
        SELECT dest AS ap, date_trunc('hour', arr_utc) AS h,
               avg(CASE WHEN arr_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND arr_delay_min IS NOT NULL AND dest IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()
    wx = con.execute(f"""
        SELECT origin AS ap, date_trunc('hour', dep_utc) AS h,
               avg(origin_wind_ms) AS wind, avg(origin_precip_mm) AS precip,
               avg(origin_visibility_m) AS vis,
               avg(CAST(origin_thunderstorm AS DOUBLE)) AS tstorm
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE origin IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()
    sched = con.execute(f"""
        SELECT origin AS ap, date_trunc('hour', dep_utc) AS h, count(*) AS n
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE origin IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()

    # P3: share of this hour's departures whose inbound aircraft had already landed
    # late *before the hour began*. The guard is what keeps it an input rather than a
    # peek: an inbound still in the air at the as-of time is not observable yet.
    rot = con.execute(f"""
        WITH legs AS (
            SELECT tail_number, origin, dep_utc, arr_utc, arr_delay_min,
                   LAG(arr_utc)        OVER w AS prev_arr_utc,
                   LAG(arr_delay_min)  OVER w AS prev_arr_delay
            FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
            WHERE cancelled = 0 AND tail_number IS NOT NULL
            WINDOW w AS (PARTITION BY tail_number ORDER BY dep_utc)
        )
        SELECT origin AS ap, date_trunc('hour', dep_utc) AS h,
               avg(CASE WHEN prev_arr_delay >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM legs
        WHERE origin IN ({inlist})
          AND prev_arr_utc IS NOT NULL
          AND prev_arr_utc + INTERVAL (prev_arr_delay) MINUTE < date_trunc('hour', dep_utc)
        GROUP BY 1, 2
    """).fetchall()

    hours = sorted({r[1] for r in dep})
    h_index = {h: i for i, h in enumerate(hours)}
    S = (len(airports), len(hours))

    def grid(rows, col=2, default=np.nan):
        M = np.full(S, default)
        for r in rows:
            if r[0] in a_index and r[1] in h_index and r[col] is not None:
                M[a_index[r[0]], h_index[r[1]]] = r[col]
        return M

    share = grid(dep, 3)
    ndep = grid(dep, 2, 0.0)
    share[ndep < MIN_DEPARTURES] = np.nan
    arrivals = grid(arr, 2)
    rotation = grid(rot, 2)
    schedule = grid(sched, 2, 0.0)
    wind, precip, vis, tstorm = (grid(wx, c) for c in (2, 3, 4, 5))

    # Flow adjacency and mean flight time per ordered pair, both from training data only.
    pairs = con.execute(f"""
        SELECT origin, dest, count(*) AS n, avg(sched_duration_min) AS dur
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND flight_date < DATE '{TRAIN_THROUGH}'
          AND origin IN ({inlist}) AND dest IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()
    A = np.zeros((len(airports), len(airports)))
    lag_h = np.zeros_like(A, dtype=int)
    for origin, dest, n, dur in pairs:
        i, j = a_index[dest], a_index[origin]
        if i == j:
            continue
        A[i, j] = n
        lag_h[i, j] = max(1, int(round((dur or 120) / 60.0)))
    tot = A.sum(axis=1, keepdims=True)
    A = np.divide(A, tot, out=np.zeros_like(A), where=tot > 0)

    def aggregate(M, shift=None):
        """Weighted mean of neighbours, optionally each at its own travel-time lag."""
        out = np.full(S, np.nan)
        src = np.nan_to_num(M, nan=0.0)
        seen = (~np.isnan(M)).astype(float)
        if shift is None:
            num, den = A @ src, A @ seen
        else:
            num, den = np.zeros(S), np.zeros(S)
            for i in range(len(airports)):
                nz = np.nonzero(A[i])[0]
                for j in nz:
                    k = lag_h[i, j]
                    num[i, k:] += A[i, j] * src[j, :-k]
                    den[i, k:] += A[i, j] * seen[j, :-k]
        return np.divide(num, den, out=out, where=den > 0.2)

    flow_flat = aggregate(share)
    flow_timed = aggregate(share, shift=True)
    print(f"  panel {S[0]} airports x {S[1]} hours; "
          f"mean travel lag {lag_h[A > 0].mean():.1f} h")

    hod = np.array([h.hour for h in hours])
    cut = int(np.searchsorted(hours, dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)))

    # Each probe is the baseline plus one block of columns, so gains are comparable.
    BLOCKS = {
        "P0 baseline (own history + time of day)": [],
        "P1a flow neighbours, flat (06's version)": [("nb", flow_flat)],
        "P1b flow neighbours, travel-time lagged": [("nb", flow_timed)],
        "P2  flat + arrivals landing here": [("nb", flow_flat), ("arr", arrivals)],
        "P3  flat + inbound aircraft already late": [("nb", flow_flat), ("rot", rotation)],
        "P4  flat + weather and scheduled traffic": [("nb", flow_flat), ("wind", wind),
                                                     ("precip", precip), ("vis", vis),
                                                     ("tstorm", tstorm), ("sched", schedule)],
        "P5  everything, on the flat base": [("nb", flow_flat), ("arr", arrivals),
                           ("rot", rotation), ("wind", wind), ("precip", precip),
                           ("vis", vis), ("tstorm", tstorm), ("sched", schedule)],
    }

    results: dict = {"horizons": list(HORIZONS), "airports": len(airports)}
    for H in HORIZONS:
        print(f"\nhorizon +{H}h   (train 2023, test 2024)")
        rows: dict = {}
        for name, block in BLOCKS.items():
            X, y, t = [], [], []
            for i in range(len(airports)):
                for j in range(LAGS, S[1] - H):
                    tgt = share[i, j + H]
                    own = share[i, j - LAGS + 1: j + 1]
                    if np.isnan(tgt) or np.isnan(own).any():
                        continue
                    # "sched" is the known schedule at the target hour; everything else
                    # is read at the as-of hour j and is therefore already observed.
                    extra = [schedule[i, j + H] if k == "sched" else M[i, j]
                             for k, M in block]
                    if any(np.isnan(v) for v in extra):
                        continue
                    tod = [np.sin(2 * np.pi * hod[j + H] / 24),
                           np.cos(2 * np.pi * hod[j + H] / 24)]
                    X.append(np.concatenate([own, tod, extra]))
                    y.append(tgt); t.append(j)
            X, y, t = np.array(X), np.array(y), np.array(t)
            tr, te = t < cut, t >= cut
            w = ridge(X[tr], y[tr])
            score = r2(y[te], np.hstack([np.ones((int(te.sum()), 1)), X[te]]) @ w)
            rows[name] = {"r2": round(score, 4), "test_rows": int(te.sum())}
            base = rows.get("P0 baseline (own history + time of day)", {}).get("r2")
            gain = f"{score - base:+.4f}" if base is not None and name[:2] != "P0" else "  --  "
            print(f"    {name:<42} R2 {score:+.4f}   gain {gain}   n={te.sum():,}")
        results[f"horizon_{H}h"] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwritten {OUT.name}")


if __name__ == "__main__":
    main()
