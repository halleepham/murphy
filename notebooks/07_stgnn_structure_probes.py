"""Which structural changes actually add signal? Tested before building the model.

The baseline diagnostic (06) established that delay propagates and that one hop of
flow-weighted graph convolution helps at six hours. This script asks which *changes to
that structure* are worth paying for, using a ridge probe so every answer is comparable
to the baseline and to the others.

THE AS-OF CORRECTION (T11)
--------------------------
An earlier version of this script keyed every feature to the hour a flight was
*scheduled* in. That reads the future. `dep_utc` is the scheduled instant, so a flight
scheduled at 14:50 running thirty minutes late does not push back until 15:20 -- after
the as-of instant at 15:00 -- and its final delay is not knowable then. Measured over
2023-24, only **32.9% of late departures and 31.4% of late arrivals have actually
happened by the end of their scheduled hour**, against 98% of on-time ones. The leak
was therefore selective on exactly the signal being predicted.

Fixed by splitting the two roles that one matrix was serving:

  TARGET   keyed to the *scheduled* hour. "Of the flights scheduled to depart in hour
           j+H, what share will be 15+ minutes late." That is the question a traveller
           asks, and a target carries no leak by construction.

  FEATURES keyed to *actual* event hours, so a value at hour j is complete at the end
           of hour j. "Of the flights that actually departed in hour j, what share left
           15+ late." This is what the rotation feature was already doing, and it was
           the only one.

The old leaky definitions are kept as the X-blocks below, so the inflation is measured
rather than assumed.

THE HYPOTHESES
--------------
  H1  flow neighbours     one hop of flow-weighted convolution, the 06 structure.
  H2  arrivals            lateness *landing* at an airport, the other half of
                          propagation. Now keyed to actual landing time.
  H3a rotation, as-of     share of this hour's departures whose inbound had already
                          landed late. T8's version, keyed to the as-of hour.
  H3b rotation, target    the same mechanism pointed forward: of the flights scheduled
                          in the *target* hour, what share have an inbound that already
                          landed late by the as-of instant. Strictly observable, and
                          the version a forecaster would actually use. New in T12.
  H4  weather + traffic   both sit in the table already and neither is an outcome.
  H6  geography           a flight-count graph only sees airports that exchange planes.
                          Two airports under one weather system move together whether
                          or not they do. D3: *unconnected* pairs still correlated at
                          +0.242. STPN carries geography as a separate relation.
  H7  both relations      flow and geography together, STPN's multi-graph.
  H9  no graph at all     the control. Every neighbour feature is a weighted mean over
                          other airports, and a national mean is the same object with
                          the structure removed. If that scores as well, no graph is
                          justified.

Every feature is complete at the as-of instant, with one declared exception: the
scheduled departure count at the target hour, which is known in advance by
construction and is not an outcome.

SCORING
-------
**Every probe is scored on exactly the same rows.** A feature that is missing more
often shrinks its own test set, and R2 is normalised by the variance of whatever set it
is computed on, so blocks evaluated on different rows are not comparable. The valid-row
mask is built once from *all* candidate features and applied to every block.

**Every probe is also scored on bad days alone.** A pooled R2 over a year is dominated
by ordinary hours, and this project exists for the hours that are not. RMSE is reported
alongside R2 because R2 is normalised per stratum and cannot be compared across them;
gains *within* a stratum are comparable either way.

Run:  PYTHONPATH=src .venv/bin/python notebooks/07_stgnn_structure_probes.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

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
GEO_SIGMA_KM = 750.0


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

    # ---- TARGET: keyed to the scheduled hour. The forecast quantity. ----------------
    tgt = con.execute(f"""
        SELECT origin AS ap, date_trunc('hour', dep_utc) AS h, count(*) AS n,
               avg(CASE WHEN dep_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND dep_delay_min IS NOT NULL AND origin IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()

    # ---- FEATURES: keyed to ACTUAL event hours, so complete at the hour's end ------
    dep_obs = con.execute(f"""
        SELECT origin AS ap,
               date_trunc('hour', dep_utc + INTERVAL (dep_delay_min) MINUTE) AS h,
               count(*) AS n,
               avg(CASE WHEN dep_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND dep_delay_min IS NOT NULL AND origin IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()
    arr_obs = con.execute(f"""
        SELECT dest AS ap,
               date_trunc('hour', arr_utc + INTERVAL (arr_delay_min) MINUTE) AS h,
               count(*) AS n,
               avg(CASE WHEN arr_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND arr_delay_min IS NOT NULL AND dest IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()

    # ---- The old leaky definitions, kept to measure what the leak was worth --------
    arr_sched = con.execute(f"""
        SELECT dest AS ap, date_trunc('hour', arr_utc) AS h,
               avg(CASE WHEN arr_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS v
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND arr_delay_min IS NOT NULL AND dest IN ({inlist})
        GROUP BY 1, 2
    """).fetchall()

    # ---- Not outcomes: observable at the as-of hour, or known in advance -----------
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

    # ---- Rotation. One query serves every horizon: `lead_min` is how long before the
    # target hour began the inbound actually landed, so a horizon keeps only the
    # inbounds that had landed by its own as-of instant. For target hour j+H with
    # as-of at the end of hour j, that cutoff is (H-1) hours before the target hour.
    rot = con.execute(f"""
        WITH legs AS (
            SELECT tail_number, origin, dep_utc,
                   LAG(arr_utc)       OVER w AS prev_arr_utc,
                   LAG(arr_delay_min) OVER w AS prev_arr_delay
            FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
            WHERE cancelled = 0 AND tail_number IS NOT NULL
            WINDOW w AS (PARTITION BY tail_number ORDER BY dep_utc)
        ), landed AS (
            SELECT origin AS ap, date_trunc('hour', dep_utc) AS h,
                   prev_arr_delay >= {LATE_MIN} AS inbound_late,
                   date_diff('minute',
                             prev_arr_utc + INTERVAL (prev_arr_delay) MINUTE,
                             date_trunc('hour', dep_utc)) AS lead_min
            FROM legs
            WHERE origin IN ({inlist}) AND prev_arr_utc IS NOT NULL
              AND prev_arr_delay IS NOT NULL
        )
        SELECT ap, h,
               sum(CASE WHEN lead_min >= 0   THEN 1 ELSE 0 END) AS n1,
               sum(CASE WHEN lead_min >= 0   AND inbound_late THEN 1 ELSE 0 END) AS l1,
               sum(CASE WHEN lead_min >= 120 THEN 1 ELSE 0 END) AS n3,
               sum(CASE WHEN lead_min >= 120 AND inbound_late THEN 1 ELSE 0 END) AS l3,
               sum(CASE WHEN lead_min >= 300 THEN 1 ELSE 0 END) AS n6,
               sum(CASE WHEN lead_min >= 300 AND inbound_late THEN 1 ELSE 0 END) AS l6
        FROM landed GROUP BY 1, 2
    """).fetchall()

    hours = sorted({r[1] for r in tgt} | {r[1] for r in dep_obs})
    h_index = {h: i for i, h in enumerate(hours)}
    S = (len(airports), len(hours))

    def grid(rows, col, default=np.nan):
        M = np.full(S, default)
        for r in rows:
            if r[0] in a_index and r[1] in h_index and r[col] is not None:
                M[a_index[r[0]], h_index[r[1]]] = r[col]
        return M

    # Target, and the leaky share that used to double as own history.
    sched_late = grid(tgt, 3)
    sched_late[grid(tgt, 2, 0.0) < MIN_DEPARTURES] = np.nan

    # Honest own history: keyed to the hour flights actually left.
    dep_late = grid(dep_obs, 3)
    dep_late[grid(dep_obs, 2, 0.0) < MIN_DEPARTURES] = np.nan

    arrivals_obs = grid(arr_obs, 3)
    arrivals_obs[grid(arr_obs, 2, 0.0) < MIN_DEPARTURES] = np.nan
    arrivals_leaky = grid(arr_sched, 2)

    schedule = grid(sched, 2, 0.0)
    wind, precip, vis, tstorm = (grid(wx, c) for c in (2, 3, 4, 5))

    # Rotation, one matrix per horizon, each using only inbounds observable by then.
    rot_by_h = {}
    for H, (nc, lc) in zip(HORIZONS, [(2, 3), (4, 5), (6, 7)]):
        n, l = grid(rot, nc, 0.0), grid(rot, lc, 0.0)
        rot_by_h[H] = np.divide(l, n, out=np.full(S, np.nan), where=n >= MIN_DEPARTURES)
    rot_asof = rot_by_h[HORIZONS[0]]   # T8's version: keyed to the as-of hour

    # Missingness profile, reported rather than assumed (spec U1).
    hod_all = np.array([h.hour for h in hours])
    print("  usable cells: "
          f"target {np.isfinite(sched_late).mean()*100:.1f}%, "
          f"own-history {np.isfinite(dep_late).mean()*100:.1f}%, "
          f"arrivals {np.isfinite(arrivals_obs).mean()*100:.1f}%")
    by_hour = [(h, float(np.isfinite(dep_late[:, hod_all == h]).mean())) for h in range(24)]
    worst = sorted(by_hour, key=lambda x: x[1])[:6]
    print("  least-observed hours of day (UTC): " +
          ", ".join(f"{h:02d}h {v*100:.0f}%" for h, v in worst))

    # ---- Adjacency ----------------------------------------------------------------
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
        """Gaussian kernel on distance, cut at 3 sigma, row-normalised."""
        G = np.exp(-((D / sigma_km) ** 2))
        G[D > 3 * sigma_km] = 0.0
        np.fill_diagonal(G, 0.0)
        s = G.sum(axis=1, keepdims=True)
        return np.divide(G, s, out=np.zeros_like(G), where=s > 0)

    A_geo = geo_adjacency(GEO_SIGMA_KM)
    U = 1.0 - np.eye(len(airports))
    A_nat = U / U.sum(axis=1, keepdims=True)

    def aggregate(M, W, shift=None):
        """Weighted mean over neighbours, optionally each at its own travel-time lag."""
        out = np.full(S, np.nan)
        src = np.nan_to_num(M, nan=0.0)
        seen = (~np.isnan(M)).astype(float)
        if shift is None:
            num, den = W @ src, W @ seen
        else:
            num, den = np.zeros(S), np.zeros(S)
            for i in range(len(airports)):
                for j in np.nonzero(W[i])[0]:
                    k = lag_h[i, j]
                    num[i, k:] += W[i, j] * src[j, :-k]
                    den[i, k:] += W[i, j] * seen[j, :-k]
        return np.divide(num, den, out=out, where=den > 0.2)

    # Honest aggregates, from the actual-hour matrix.
    flow_flat = aggregate(dep_late, A)
    flow_timed = aggregate(dep_late, A, shift=True)
    geo_750 = aggregate(dep_late, A_geo)
    nat_mean = aggregate(dep_late, A_nat)
    # Leaky aggregates, from the scheduled-hour matrix, for the X-blocks only.
    flow_leaky = aggregate(sched_late, A)
    geo_leaky = aggregate(sched_late, A_geo)
    print(f"  panel {S[0]} airports x {S[1]} hours; "
          f"mean travel lag {lag_h[A > 0].mean():.1f} h")

    hod = np.array([h.hour for h in hours])
    cut = int(np.searchsorted(hours, dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)))

    day_of = np.array([h.date() for h in hours])
    test_days = sorted({d for d in day_of[cut:].tolist()})
    nat = {d: float(np.nanmean(sched_late[:, day_of == d])) for d in test_days}
    vals = np.array([nat[d] for d in test_days])
    p95, p75, p50 = (float(np.nanpercentile(vals, q)) for q in (95, 75, 50))
    STRATA = {
        "all test days": set(test_days),
        "worst 5% of days": {d for d in test_days if nat[d] >= p95},
        "worst 25% of days": {d for d in test_days if nat[d] >= p75},
        "calm half of days": {d for d in test_days if nat[d] < p50},
    }
    print(f"  national late share: calm half below {p50:.3f}, worst 25% above "
          f"{p75:.3f}, worst 5% above {p95:.3f}")

    # Keys must be unique per *matrix*, not per role: the shared feature dictionary is
    # keyed by name, so two variants sharing a key silently collapse to one.
    def BLOCKS_FOR(H):
        rot_tgt = rot_by_h[H]
        return {
            # ---- honest: every feature complete at the as-of instant --------------
            "H0 own history (actual departures) + time of day": [],
            "H9  + national mean, no graph": [("nat", nat_mean)],
            "H1a + flow neighbours, flat": [("flow", flow_flat)],
            "H1b + flow neighbours, travel-time lagged": [("flow_t", flow_timed)],
            "H6  + geography 750 km": [("geo", geo_750)],
            "H7  + flow + geography": [("flow", flow_flat), ("geo", geo_750)],
            "H2  + flow + arrivals (actual landings)": [("flow", flow_flat),
                                                        ("arr", arrivals_obs)],
            "H3a + flow + rotation at as-of hour": [("flow", flow_flat),
                                                    ("rot_a", rot_asof)],
            "H3b + flow + rotation at target hour": [("flow", flow_flat),
                                                     ("rot_t", rot_tgt)],
            "H4  + flow + weather and traffic": [("flow", flow_flat), ("wind", wind),
                                                 ("precip", precip), ("vis", vis),
                                                 ("tstorm", tstorm), ("sched", schedule)],
            "H8  + flow + geography + arrivals": [("flow", flow_flat), ("geo", geo_750),
                                                  ("arr", arrivals_obs)],
            "H5  everything honest": [("flow", flow_flat), ("geo", geo_750),
                                      ("arr", arrivals_obs), ("rot_t", rot_tgt),
                                      ("nat", nat_mean), ("wind", wind),
                                      ("precip", precip), ("vis", vis),
                                      ("tstorm", tstorm), ("sched", schedule)],
            # ---- leaky: what T8 and T9 actually measured -------------------------
            "X0 LEAKY own history (scheduled hour)": [("lk_own1", None)],
            "X2 LEAKY own history + leaky arrivals": [("lk_own1", None),
                                                      ("arr_lk", arrivals_leaky)],
            "X8 LEAKY flow + geography + arrivals": [("lk_own1", None),
                                                     ("flow_lk", flow_leaky),
                                                     ("geo_lk", geo_leaky),
                                                     ("arr_lk", arrivals_leaky)],
        }

    results: dict = {"horizons": list(HORIZONS), "airports": len(airports),
                     "geo_sigma_km": GEO_SIGMA_KM}

    for H in HORIZONS:
        BLOCKS = BLOCKS_FOR(H)
        ALL = {k: M for b in BLOCKS.values() for k, M in b if M is not None}
        keys = list(ALL)
        print(f"\nhorizon +{H}h   (train 2023, test 2024, identical rows for every block)")

        own_l, lk_l, extra_l, y_l, t_l = [], [], [], [], []
        for i in range(len(airports)):
            for j in range(LAGS, S[1] - H):
                tgt_v = sched_late[i, j + H]
                own = dep_late[i, j - LAGS + 1: j + 1]
                lk = sched_late[i, j - LAGS + 1: j + 1]
                if np.isnan(tgt_v) or np.isnan(own).any() or np.isnan(lk).any():
                    continue
                extra = [schedule[i, j + H] if k == "sched" else ALL[k][i, j]
                         for k in keys]
                if any(np.isnan(v) for v in extra):
                    continue
                tod = [np.sin(2 * np.pi * hod[j + H] / 24),
                       np.cos(2 * np.pi * hod[j + H] / 24)]
                own_l.append(np.concatenate([own, tod]))
                lk_l.append(lk)
                extra_l.append(extra); y_l.append(tgt_v); t_l.append(j)

        base_X = np.array(own_l)            # honest own history + time of day
        leaky_own = np.array(lk_l)          # the scheduled-hour lags, X-blocks only
        extras = np.array(extra_l)
        y, t = np.array(y_l), np.array(t_l)
        tr, te = t < cut, t >= cut
        col = {k: n for n, k in enumerate(keys)}
        print(f"    common rows: train {tr.sum():,}  test {te.sum():,}")

        tgt_day = day_of[t + H]
        masks = {nm: te & np.isin(tgt_day, list(d)) for nm, d in STRATA.items()}

        preds: dict = {}
        for name, block in BLOCKS.items():
            if name.startswith("X"):
                # Leaky blocks replace honest own history with the scheduled-hour lags.
                cols = [leaky_own, base_X[:, LAGS:]]
                cols += [extras[:, [col[k]]] for k, M in block if M is not None]
                X = np.hstack(cols)
            else:
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
                sc, err = r2(y[m], preds[name][m]), rmse(y[m], preds[name][m])
                if base is None:
                    base = sc
                out[name] = {"r2": round(sc, 4), "gain": round(sc - base, 4),
                             "rmse": round(err, 4)}
                g = "  --  " if name.startswith("H0") else f"{sc - base:+.4f}"
                print(f"       {name:<50} R2 {sc:+.4f}  vs H0 {g}  RMSE {err:.4f}")
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
