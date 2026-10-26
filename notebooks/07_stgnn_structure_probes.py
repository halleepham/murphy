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
  P6  geographic proximity a flight-count graph only sees airports that exchange
                          planes. Two airports under one weather system move together
                          whether or not they do. D3 is the evidence: *unconnected*
                          pairs still correlated at +0.242, which is shared structure
                          the flow graph cannot represent. STPN carries geography and
                          schedule as separate relations for this reason.
  P7  both relations      flow and geography together, which is STPN's multi-graph.
  P9  no graph at all     the control the set was missing. Every "neighbour" feature
                          above is a weighted mean over other airports, and a national
                          mean is the same thing with the structure removed. If that
                          scores as well, the gain was never about *which* airports
                          are neighbours and no graph is justified. This is the
                          cheapest way to falsify the whole model.

Every feature is strictly past-observed relative to the target hour, except the
scheduled departure count, which is known in advance by construction.

**Every probe is also scored on bad days alone.** A pooled R2 over a whole year is
dominated by ordinary hours, and this project exists for the days that are not
ordinary. A structure worth building is one that helps while the network is breaking,
so each block is scored again on the worst 5% and 25% of test days by national late
share, on the calm half, and on the seed day itself. RMSE is reported alongside R2
because R2 is normalised by the variance of whichever stratum it is computed on and so
cannot be compared across them; gains *within* a stratum are comparable either way.

**Every probe is scored on exactly the same rows.** A feature that is missing more
often shrinks its own test set, and R2 is normalised by the variance of whatever set it
is computed on, so blocks evaluated on different rows are not comparable. The valid-row
mask is therefore built once from *all* candidate features and applied to every block,
including the baseline.

Run:  PYTHONPATH=src .venv/bin/python notebooks/07_stgnn_structure_probes.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

import csv

import duckdb
import numpy as np

FLIGHTS = "data/processed/flights_full/**/*.parquet"
COORDS = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "airport_coords.csv"
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


def rmse(y, pred):
    return float(np.sqrt(((y - pred) ** 2).mean()))


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

    coords = {}
    with COORDS.open() as f:
        for row in csv.DictReader(f):
            coords[row["airport"]] = (float(row["lat"]), float(row["lon"]))
    missing = [a for a in airports if a not in coords]
    if missing:
        raise SystemExit(f"no coordinates for {missing}")

    def km(a, b):
        (la1, lo1), (la2, lo2) = coords[a], coords[b]
        return float(np.hypot((la1 - la2) * 111,
                              (lo1 - lo2) * 111 * np.cos(np.radians((la1 + la2) / 2))))

    D = np.array([[km(a, b) for b in airports] for a in airports])

    def geo_adjacency(sigma_km):
        """Gaussian kernel on distance, cut at 3 sigma, row-normalised.

        The standard construction in this literature. Sigma sets what counts as
        "the same weather", so it is swept rather than guessed.
        """
        G = np.exp(-((D / sigma_km) ** 2))
        G[D > 3 * sigma_km] = 0.0
        np.fill_diagonal(G, 0.0)
        tot = G.sum(axis=1, keepdims=True)
        return np.divide(G, tot, out=np.zeros_like(G), where=tot > 0)

    def aggregate(M, shift=None, W=None):
        """Weighted mean of neighbours, optionally each at its own travel-time lag."""
        out = np.full(S, np.nan)
        W = A if W is None else W
        src = np.nan_to_num(M, nan=0.0)
        seen = (~np.isnan(M)).astype(float)
        if shift is None:
            num, den = W @ src, W @ seen
        else:
            num, den = np.zeros(S), np.zeros(S)
            for i in range(len(airports)):
                nz = np.nonzero(W[i])[0]
                for j in nz:
                    k = lag_h[i, j]
                    num[i, k:] += W[i, j] * src[j, :-k]
                    den[i, k:] += W[i, j] * seen[j, :-k]
        return np.divide(num, den, out=out, where=den > 0.2)

    flow_flat = aggregate(share)
    flow_timed = aggregate(share, shift=True)
    geo_300 = aggregate(share, W=geo_adjacency(300))
    geo_750 = aggregate(share, W=geo_adjacency(750))
    # 750 km won the first sweep and was its widest setting, so the maximum was not
    # bracketed. 1500 km brackets it from above.
    geo_1500 = aggregate(share, W=geo_adjacency(1500))

    # The no-graph control: every other airport weighted equally.
    U = 1.0 - np.eye(len(airports))
    nat_mean = aggregate(share, W=U / U.sum(axis=1, keepdims=True))
    print(f"  panel {S[0]} airports x {S[1]} hours; "
          f"mean travel lag {lag_h[A > 0].mean():.1f} h")

    hod = np.array([h.hour for h in hours])
    cut = int(np.searchsorted(hours, dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)))

    # Day strata for the bad-day evaluation. "Bad" is a property of the whole network,
    # not of one airport, so the ranking is the national late share across every
    # airport that day. Ranked within the test year only, so thresholds describe 2024.
    day_of = np.array([h.date() for h in hours])
    test_days = sorted({d for d in day_of[cut:].tolist()})
    nat = {d: float(np.nanmean(share[:, day_of == d])) for d in test_days}
    vals = np.array([nat[d] for d in test_days])
    p95, p75, p50 = (float(np.nanpercentile(vals, q)) for q in (95, 75, 50))
    SEED_DAY = dt.date(2024, 7, 19)
    STRATA = {
        "all test days": set(test_days),
        "worst 5% of days": {d for d in test_days if nat[d] >= p95},
        "worst 25% of days": {d for d in test_days if nat[d] >= p75},
        "calm half of days": {d for d in test_days if nat[d] < p50},
        "seed day 19 Jul 2024": {SEED_DAY},
    }
    print(f"  national late share: calm half below {p50:.3f}, worst 25% above "
          f"{p75:.3f}, worst 5% above {p95:.3f}; "
          f"seed day {nat.get(SEED_DAY, float('nan')):.3f}")

    # Each probe is the baseline plus one block of columns, so gains are comparable.
    # Keys must be unique per *matrix*, not per role: the shared-feature dictionary is
    # keyed by name, so two variants sharing a key silently collapse to one.
    BLOCKS = {
        "P0 baseline (own history + time of day)": [],
        "P1a flow neighbours, flat (06's version)": [("flow", flow_flat)],
        "P1b flow neighbours, travel-time lagged": [("flow_timed", flow_timed)],
        "P2  flat + arrivals landing here": [("flow", flow_flat), ("arr", arrivals)],
        "P3  flat + inbound aircraft already late": [("flow", flow_flat), ("rot", rotation)],
        "P4  flat + weather and scheduled traffic": [("flow", flow_flat), ("wind", wind),
                                                     ("precip", precip), ("vis", vis),
                                                     ("tstorm", tstorm), ("sched", schedule)],
        "P6a geographic neighbours only, 300 km": [("geo300", geo_300)],
        "P6b geographic neighbours only, 750 km": [("geo750", geo_750)],
        "P6c geographic neighbours only, 1500 km": [("geo1500", geo_1500)],
        "P7a flat flow + geography, 300 km": [("flow", flow_flat), ("geo300", geo_300)],
        "P7b flat flow + geography, 750 km": [("flow", flow_flat), ("geo750", geo_750)],
        "P7c flat flow + geography, 1500 km": [("flow", flow_flat), ("geo1500", geo_1500)],
        "P8  flow + geography + arrivals": [("flow", flow_flat), ("geo750", geo_750),
                                            ("arr", arrivals)],
        "P9a national mean only (no graph)": [("nat", nat_mean)],
        "P9b flat flow + national mean": [("flow", flow_flat), ("nat", nat_mean)],
        "P9c geography 750 km + national mean": [("geo750", geo_750), ("nat", nat_mean)],
        "P5  everything, on the flat base": [("flow", flow_flat), ("arr", arrivals),
                           ("rot", rotation), ("geo750", geo_750), ("nat", nat_mean),
                           ("wind", wind), ("precip", precip), ("vis", vis),
                           ("tstorm", tstorm), ("sched", schedule)],
    }

    results: dict = {"horizons": list(HORIZONS), "airports": len(airports)}
    ALL = {k: M for block in BLOCKS.values() for k, M in block}

    for H in HORIZONS:
        print(f"\nhorizon +{H}h   (train 2023, test 2024, identical rows for every row)")
        # One pass: collect own-history, time of day and every candidate feature, and
        # keep a row only where all of them exist.
        own_l, extra_l, y_l, t_l = [], [], [], []
        keys = list(ALL)
        for i in range(len(airports)):
            for j in range(LAGS, S[1] - H):
                tgt = share[i, j + H]
                own = share[i, j - LAGS + 1: j + 1]
                if np.isnan(tgt) or np.isnan(own).any():
                    continue
                extra = [schedule[i, j + H] if k == "sched" else ALL[k][i, j]
                         for k in keys]
                if any(np.isnan(v) for v in extra):
                    continue
                tod = [np.sin(2 * np.pi * hod[j + H] / 24),
                       np.cos(2 * np.pi * hod[j + H] / 24)]
                own_l.append(np.concatenate([own, tod]))
                extra_l.append(extra); y_l.append(tgt); t_l.append(j)
        base_X = np.array(own_l)
        extras = np.array(extra_l)
        y, t = np.array(y_l), np.array(t_l)
        tr, te = t < cut, t >= cut
        col = {k: n for n, k in enumerate(keys)}
        print(f"    common rows: train {tr.sum():,}  test {te.sum():,}")

        # The target hour's day decides the stratum, since that is the hour forecast.
        tgt_day = day_of[t + H]
        masks = {nm: te & np.isin(tgt_day, list(days)) for nm, days in STRATA.items()}

        # Fit once per block on the full training year, then score each stratum. The
        # question is whether a structure helps more when it matters, not whether a
        # model fitted only to bad days would do better.
        preds: dict = {}
        for name, block in BLOCKS.items():
            X = base_X if not block else np.hstack(
                [base_X, extras[:, [col[k] for k, _ in block]]])
            w = ridge(X[tr], y[tr])
            preds[name] = np.hstack([np.ones((len(X), 1)), X]) @ w

        rows: dict = {}
        for nm, m in masks.items():
            if m.sum() < 500:
                print(f"    -- {nm}: only {m.sum():,} rows, skipped")
                continue
            print(f"    -- {nm}  n={m.sum():,}  mean late share {y[m].mean():.3f}")
            base, out = None, {}
            for name in BLOCKS:
                score, err = r2(y[m], preds[name][m]), rmse(y[m], preds[name][m])
                if base is None:
                    base = score
                out[name] = {"r2": round(score, 4), "gain": round(score - base, 4),
                             "rmse": round(err, 4)}
                gain = "  --  " if name.startswith("P0") else f"{score - base:+.4f}"
                print(f"       {name:<42} R2 {score:+.4f}  gain {gain}  RMSE {err:.4f}")
            out["_rows"] = int(m.sum())
            out["_mean_late_share"] = round(float(y[m].mean()), 4)
            rows[nm] = out
        rows["_common_test_rows"] = int(te.sum())
        results[f"horizon_{H}h"] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwritten {OUT.name}")


if __name__ == "__main__":
    main()
