"""Does airport-hour delay actually propagate? Measured before any model is built.

The GraphSAGE result (decision N13) was that message passing added nothing, because
`outcome_*` properties are excluded from node features, so no delay information sat on
the graph for it to carry. An STGNN is a different proposition: its node features *are*
a time series of past observed delay per airport-hour, which reintroduces that signal
legitimately, because upstream outcomes observed before the prediction time are already
permitted as inputs (decision N6).

That argument is only worth acting on if the signal exists. This script asks four
questions of the data, in increasing order of how much they commit us:

  D1  Is the panel dense enough to be a time series at all?
  D2  Does an airport's own delay persist hour to hour? (temporal signal)
  D3  Does an upstream airport's delay lead a downstream airport's? (spatial signal)
  D4  Do neighbours add anything *beyond* an airport's own history? (the decisive one)
      Measured at every horizon Murphy serves, because the horizon at which a question
      is asked and the horizon at which the graph helps are different things. Travelers
      do ask about the next hour -- a connecting passenger on a late inbound is the
      canonical case -- but at one hour out an airport's own state is already strongly
      predictive, and the rotation chain built in Challenge 3 already serves that case.
      The graph has to earn its place where own-persistence has decayed.

D4 is the STGNN question written as linear regression. One hop of graph convolution
with a fixed adjacency is exactly "weight each neighbour by how many flights it sends
here, and average." If that cannot beat an airport's own recent history, a learned
model over the same structure is unlikely to find something linear regression missed.
That is the same reasoning as the N13 feature ablation, applied before the build rather
than after it.

Run:  PYTHONPATH=src .venv/bin/python notebooks/06_stgnn_diagnostic.py
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
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "stgnn_diagnostic.json"

# Network state is measured from *all* traffic in the table, regionals included. The
# nine-major scope governs which flights Murphy forecasts for, not which flights make an
# airport congested: a SkyWest regional occupies a gate and a runway slot exactly like a
# mainline flight. Decision T1 in stgnn_log.md, with a sensitivity check in D2.
TOP_AIRPORTS = 100
LATE_MIN = 15          # the BTS convention already used throughout the project
MIN_DEPARTURES = 5     # an airport-hour below this is too thin to carry a rate
LAGS = 3               # hours of own history offered to both models in D4


def panel(con) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Airport x hour matrices of (share late, departures), plus the hour index."""
    airports = [r[0] for r in con.execute(f"""
        SELECT origin, count(*) AS n FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0
        GROUP BY origin ORDER BY n DESC LIMIT {TOP_AIRPORTS}
    """).fetchall()]

    rows = con.execute(f"""
        SELECT origin,
               date_trunc('hour', dep_utc)                              AS hour_utc,
               count(*)                                                 AS n_dep,
               avg(CASE WHEN dep_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS share_late
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND dep_delay_min IS NOT NULL
          AND origin IN ({','.join("'" + a + "'" for a in airports)})
        GROUP BY origin, hour_utc
    """).fetchall()

    hours = sorted({r[1] for r in rows})
    h_index = {h: i for i, h in enumerate(hours)}
    a_index = {a: i for i, a in enumerate(airports)}

    share = np.full((len(airports), len(hours)), np.nan)
    ndep = np.zeros((len(airports), len(hours)))
    for origin, hour, n, late in rows:
        i, j = a_index[origin], h_index[hour]
        share[i, j] = late
        ndep[i, j] = n

    # An hour with too few departures is noise, not a rate.
    share[ndep < MIN_DEPARTURES] = np.nan
    return share, ndep, airports, np.array(hours)


def adjacency(con, airports: list[str], through: str) -> np.ndarray:
    """A[i, j] = flights from airport j into airport i, row-normalised.

    Row-normalised so a neighbour aggregate is a weighted mean rather than a sum, which
    keeps it on the same scale as the airport's own share-late and makes the D4
    comparison a fair one.
    """
    a_index = {a: i for i, a in enumerate(airports)}
    rows = con.execute(f"""
        SELECT origin, dest, count(*) AS n
        FROM read_parquet('{FLIGHTS}', hive_partitioning=true)
        WHERE cancelled = 0 AND flight_date < DATE '{through}'
          AND origin IN ({','.join("'" + a + "'" for a in airports)})
          AND dest   IN ({','.join("'" + a + "'" for a in airports)})
        GROUP BY origin, dest
    """).fetchall()

    A = np.zeros((len(airports), len(airports)))
    for origin, dest, n in rows:
        A[a_index[dest], a_index[origin]] = n      # into dest, from origin
    np.fill_diagonal(A, 0.0)
    totals = A.sum(axis=1, keepdims=True)
    return np.divide(A, totals, out=np.zeros_like(A), where=totals > 0)


def autocorr(series: np.ndarray, lag: int) -> float | None:
    a, b = series[:-lag], series[lag:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 200:
        return None
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge with an intercept, so this needs no new dependency."""
    X1 = np.hstack([np.ones((len(X), 1)), X])
    P = np.eye(X1.shape[1]) * lam
    P[0, 0] = 0.0                                   # never penalise the intercept
    return np.linalg.solve(X1.T @ X1 + P, X1.T @ y)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main() -> None:
    con = duckdb.connect()
    print("building the airport-hour panel ...")
    share, ndep, airports, hours = panel(con)
    A = adjacency(con, airports, through="2024-01-01")   # training period only
    out: dict = {"airports": len(airports), "hours": len(hours),
                 "late_threshold_min": LATE_MIN, "min_departures": MIN_DEPARTURES}

    # ---- D1 density -------------------------------------------------------------
    usable = int((~np.isnan(share)).sum())
    out["d1_density"] = {
        "cells": int(share.size),
        "usable_cells": usable,
        "usable_pct": round(100 * usable / share.size, 1),
        "mean_departures_per_usable_hour": round(float(ndep[~np.isnan(share)].mean()), 1),
        "overall_share_late": round(float(np.nanmean(share)), 4),
    }
    print(f"\nD1  {usable:,} usable airport-hours of {share.size:,} "
          f"({out['d1_density']['usable_pct']}%), "
          f"mean share late {out['d1_density']['overall_share_late']:.3f}")

    # ---- D2 does an airport's own delay persist? --------------------------------
    print("\nD2  own-history autocorrelation of share-late")
    print("    raw, then with each airport's hour-of-day profile removed")

    hod = np.array([h.hour for h in hours])
    deseasoned = np.full_like(share, np.nan)
    for i in range(len(airports)):
        for h in range(24):
            sel = hod == h
            m = np.nanmean(share[i, sel]) if np.any(~np.isnan(share[i, sel])) else np.nan
            deseasoned[i, sel] = share[i, sel] - m

    d2 = {}
    for lag in (1, 2, 3, 6, 12, 24):
        vals = [c for c in (autocorr(share[i], lag) for i in range(len(airports)))
                if c is not None]
        dvals = [c for c in (autocorr(deseasoned[i], lag) for i in range(len(airports)))
                 if c is not None]
        d2[f"lag_{lag}h"] = {"median": round(float(np.median(vals)), 3),
                             "p10": round(float(np.percentile(vals, 10)), 3),
                             "p90": round(float(np.percentile(vals, 90)), 3),
                             "airports": len(vals),
                             "deseasoned_median": round(float(np.median(dvals)), 3)}
        print(f"    lag {lag:>2}h   raw {d2[f'lag_{lag}h']['median']:+.3f}   "
              f"deseasoned {d2[f'lag_{lag}h']['deseasoned_median']:+.3f}")
    out["d2_temporal_autocorrelation"] = d2

    # ---- D3 does an upstream airport lead a downstream one? ---------------------
    # Connected pairs are compared against unconnected ones. Without that control a
    # positive correlation proves only that bad days are bad everywhere.
    print("\nD3  upstream -> downstream lead correlation, against an unconnected control")
    rng = np.random.default_rng(7)
    connected = [(i, j) for i in range(len(airports)) for j in range(len(airports))
                 if A[i, j] > 0.02]
    unconnected = [(i, j) for i in range(len(airports)) for j in range(len(airports))
                   if i != j and A[i, j] == 0]
    rng.shuffle(unconnected)
    unconnected = unconnected[:len(connected)]

    d3 = {"connected_pairs": len(connected), "control_pairs": len(unconnected)}
    for lag in (1, 2, 3):
        def pair_corrs(pairs):
            vals = []
            for dest, origin in pairs:
                a, b = share[origin][:-lag], share[dest][lag:]
                ok = ~np.isnan(a) & ~np.isnan(b)
                if ok.sum() < 200 or a[ok].std() == 0 or b[ok].std() == 0:
                    continue
                vals.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
            return vals
        c, u = pair_corrs(connected), pair_corrs(unconnected)
        d3[f"lag_{lag}h"] = {"connected_median": round(float(np.median(c)), 3),
                             "control_median": round(float(np.median(u)), 3),
                             "gap": round(float(np.median(c) - np.median(u)), 3)}
        print(f"    lag {lag}h   connected {np.median(c):+.3f}   "
              f"control {np.median(u):+.3f}   gap {np.median(c) - np.median(u):+.3f}")
    out["d3_spatial_lead"] = d3

    # ---- D4 do neighbours add anything beyond own history? ----------------------
    # One hop of graph convolution, written as a feature. Train on 2023, test on 2024.
    print("\nD4  neighbour aggregate against own history alone (ridge, train 2023, test 2024)")
    neighbour = A @ np.nan_to_num(share, nan=0.0)          # weighted mean of inbound
    observed = A @ (~np.isnan(share)).astype(float)        # how much of it was real
    neighbour = np.divide(neighbour, observed, out=np.full_like(neighbour, np.nan),
                          where=observed > 0.2)

    cut = int(np.searchsorted(hours, dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)))
    sweep = {}
    for H in (1, 2, 3, 6, 12, 24):   # T-2h, T-6h and T-24h are the plan's horizons
        Xo, Xn, y, t = [], [], [], []
        for i in range(len(airports)):
            for j in range(LAGS, share.shape[1] - H):
                tgt = share[i, j + H]
                own = share[i, j - LAGS + 1: j + 1]
                nb = neighbour[i, j]
                if np.isnan(tgt) or np.isnan(own).any() or np.isnan(nb):
                    continue
                tod = [np.sin(2 * np.pi * hod[j + H] / 24),
                       np.cos(2 * np.pi * hod[j + H] / 24)]
                base = np.concatenate([own, tod])
                Xo.append(base); Xn.append(np.append(base, nb))
                y.append(tgt); t.append(j)
        Xo, Xn, y, t = np.array(Xo), np.array(Xn), np.array(y), np.array(t)
        tr, te = t < cut, t >= cut

        r = {"test_rows": int(te.sum())}
        for name, X in (("own", Xo), ("own_plus_neighbours", Xn)):
            w = ridge(X[tr], y[tr])
            r[name] = round(r2(y[te], np.hstack([np.ones((int(te.sum()), 1)), X[te]]) @ w), 4)
        r["persistence"] = round(r2(y[te], Xo[te][:, LAGS - 1]), 4)
        r["neighbour_gain"] = round(r["own_plus_neighbours"] - r["own"], 4)
        r["relative_gain_pct"] = round(100 * r["neighbour_gain"] / r["own"], 1)
        sweep[f"horizon_{H}h"] = r
        print(f"    +{H}h   persistence {r['persistence']:+.3f}   own {r['own']:+.3f}   "
              f"+neighbours {r['own_plus_neighbours']:+.3f}   "
              f"gain {r['neighbour_gain']:+.4f} ({r['relative_gain_pct']:+.1f}%)")
    out["d4_incremental_value"] = sweep

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwritten {OUT.name}")


if __name__ == "__main__":
    main()
