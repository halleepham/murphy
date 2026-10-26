"""Is the weak graph result about flight delay, or about how we built the graph?

T12 measured graph structure at +0.0071 beyond a national scalar at six hours. Before
accepting that, two explanations had to be ruled out -- both of which would be defects
of ours rather than facts about the data.

  Q1  IS THE ADJACENCY TOO DIFFUSE?  A row-normalised annual flight-count matrix,
      applied identically at every hour, might be close to a global average by
      construction. Measured by effective neighbour count, by how far an
      hour-specific adjacency departs from the annual one, and by correlating the
      resulting features.

  Q2  IS MEAN-POOLING THE BOTTLENECK?  Each relation gave the model ONE number per
      hour: a weighted mean over neighbours. That is the weakest possible graph
      operator. If neighbour identity matters, handing the model its top-k
      neighbours as SEPARATE columns should beat the mean. Also tested: a longer
      history window, since this literature typically uses ~12 input steps and the
      probes used 3.

Both answers are in results/stgnn_aggregation.json and in T13 of the log.

Run:  PYTHONPATH=src .venv/bin/python notebooks/08_stgnn_aggregation_probe.py
"""
import csv, datetime as dt, json, pathlib, warnings
warnings.filterwarnings("ignore")
import duckdb, numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "stgnn_aggregation.json"
RESULTS = {}
F = "data/processed/flights_full/**/*.parquet"
COORDS = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "airport_coords.csv"
con = duckdb.connect()
aps = [r[0] for r in con.execute(f"""
    SELECT origin, count(*) n FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 GROUP BY origin ORDER BY n DESC LIMIT 100""").fetchall()]
inl = ",".join("'" + a + "'" for a in aps)
ai = {a: i for i, a in enumerate(aps)}
N = len(aps)

def rownorm(M):
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)

def eff_neighbours(W):
    """exp(entropy): 1 = one neighbour carries everything, N-1 = perfectly uniform."""
    out = []
    for r in W:
        p = r[r > 0]
        if len(p) == 0:
            continue
        out.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(out))

def top_mass(W, k=5):
    return float(np.mean([np.sort(r)[::-1][:k].sum() for r in W if r.sum() > 0]))

# --- static annual adjacency, training rows only -------------------------------
pairs = con.execute(f"""
    SELECT origin, dest, count(*) n FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 AND flight_date < DATE '2024-01-01'
      AND origin IN ({inl}) AND dest IN ({inl}) GROUP BY 1,2""").fetchall()
A = np.zeros((N, N))
for o, d, n in pairs:
    if ai[d] != ai[o]:
        A[ai[d], ai[o]] = n
A = rownorm(A)

coords = {r["airport"]: (float(r["lat"]), float(r["lon"]))
          for r in csv.DictReader(COORDS.open())}
def km(a, b):
    (y1, x1), (y2, x2) = coords[a], coords[b]
    return float(np.hypot((y1-y2)*111, (x1-x2)*111*np.cos(np.radians((y1+y2)/2))))
D = np.array([[km(a, b) for b in aps] for a in aps])
G = np.exp(-((D/750.0)**2)); G[D > 2250] = 0.0; np.fill_diagonal(G, 0.0)
A_geo = rownorm(G)
U = rownorm(1.0 - np.eye(N))

print("How concentrated is each adjacency? (N-1 = 99 means a plain global average)")
for nm, W in [("flow, annual", A), ("geography 750 km", A_geo), ("uniform", U)]:
    print(f"  {nm:<20} effective neighbours {eff_neighbours(W):6.1f}   "
          f"top-5 mass {top_mass(W)*100:5.1f}%")

# --- hour-of-day specific adjacency -------------------------------------------
dyn = con.execute(f"""
    SELECT date_part('hour', dep_utc) hr, origin, dest, count(*) n
    FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 AND flight_date < DATE '2024-01-01'
      AND origin IN ({inl}) AND dest IN ({inl}) GROUP BY 1,2,3""").fetchall()
Ah = {h: np.zeros((N, N)) for h in range(24)}
for hr, o, d, n in dyn:
    if ai[d] != ai[o]:
        Ah[int(hr)][ai[d], ai[o]] = n
Ah = {h: rownorm(M) for h, M in Ah.items()}

def cos_rows(W1, W2):
    num = (W1*W2).sum(axis=1)
    den = np.linalg.norm(W1, axis=1)*np.linalg.norm(W2, axis=1)
    v = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return float(v[den > 0].mean())

sims = [cos_rows(Ah[h], A) for h in range(24)]
print(f"\nHour-specific adjacency vs the annual one: mean row cosine "
      f"{np.mean(sims):.3f}  (min {np.min(sims):.3f} at {int(np.argmin(sims)):02d}h)")
print(f"Hour-specific vs uniform:                  mean row cosine "
      f"{np.mean([cos_rows(Ah[h], U) for h in range(24)]):.3f}")
print(f"Annual vs uniform:                         mean row cosine {cos_rows(A, U):.3f}")

# --- do the resulting FEATURES differ? ----------------------------------------
dep = con.execute(f"""
    SELECT origin ap, date_trunc('hour', dep_utc + INTERVAL (dep_delay_min) MINUTE) h,
           count(*) n, avg(CASE WHEN dep_delay_min >= 15 THEN 1 ELSE 0 END) v
    FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 AND dep_delay_min IS NOT NULL AND origin IN ({inl})
    GROUP BY 1,2""").fetchall()
hours = sorted({r[1] for r in dep}); hi = {h: i for i, h in enumerate(hours)}
S = (N, len(hours))
late = np.full(S, np.nan); nd = np.zeros(S)
for ap, h, n, v in dep:
    late[ai[ap], hi[h]] = v; nd[ai[ap], hi[h]] = n
late[nd < 5] = np.nan

def agg(W):
    src = np.nan_to_num(late, nan=0.0); seen = (~np.isnan(late)).astype(float)
    num, den = W @ src, W @ seen
    return np.divide(num, den, out=np.full(S, np.nan), where=den > 0.2)

f_flow, f_geo, f_nat = agg(A), agg(A_geo), agg(U)
# hour-specific aggregate: each column uses its own hour-of-day adjacency
f_dyn = np.full(S, np.nan)
hod = np.array([h.hour for h in hours])
for h in range(24):
    cols = hod == h
    src = np.nan_to_num(late[:, cols], nan=0.0)
    seen = (~np.isnan(late[:, cols])).astype(float)
    num, den = Ah[h] @ src, Ah[h] @ seen
    f_dyn[:, cols] = np.divide(num, den, out=np.full(src.shape, np.nan), where=den > 0.2)

ok = np.isfinite(f_flow) & np.isfinite(f_geo) & np.isfinite(f_nat) & np.isfinite(f_dyn)
print(f"\nCorrelation between the neighbour FEATURES themselves (n={ok.sum():,} cells):")
for nm, X, Y in [("flow      vs national mean", f_flow, f_nat),
                 ("geography vs national mean", f_geo, f_nat),
                 ("flow      vs geography    ", f_flow, f_geo),
                 ("flow      vs hour-specific", f_flow, f_dyn),
                 ("hour-spec vs national mean", f_dyn, f_nat)]:
    print(f"  {nm}  r = {np.corrcoef(X[ok], Y[ok])[0,1]:+.4f}")

RESULTS["q1_adjacency"] = {
    "effective_neighbours": {"flow_annual": round(eff_neighbours(A), 1),
                             "geography_750km": round(eff_neighbours(A_geo), 1),
                             "uniform": round(eff_neighbours(U), 1)},
    "top5_mass": {"flow_annual": round(top_mass(A), 4),
                  "geography_750km": round(top_mass(A_geo), 4),
                  "uniform": round(top_mass(U), 4)},
    "hour_specific_vs_annual_cosine": round(float(np.mean(sims)), 3),
    "hour_specific_vs_annual_cosine_min": round(float(np.min(sims)), 3),
    "annual_vs_uniform_cosine": round(cos_rows(A, U), 3),
    "feature_correlations": {
        "flow_vs_national_mean": round(float(np.corrcoef(f_flow[ok], f_nat[ok])[0, 1]), 4),
        "geography_vs_national_mean": round(float(np.corrcoef(f_geo[ok], f_nat[ok])[0, 1]), 4),
        "flow_vs_hour_specific": round(float(np.corrcoef(f_flow[ok], f_dyn[ok])[0, 1]), 4),
    },
    "cells": int(ok.sum()),
}
print()
F = "data/processed/flights_full/**/*.parquet"
LATE, MIND, HORIZONS = 15, 5, (1, 3, 6)
con = duckdb.connect()
aps = [r[0] for r in con.execute(f"""
    SELECT origin, count(*) n FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 GROUP BY origin ORDER BY n DESC LIMIT 100""").fetchall()]
inl = ",".join("'" + a + "'" for a in aps); ai = {a: i for i, a in enumerate(aps)}; N = len(aps)

tgt = con.execute(f"""
    SELECT origin ap, date_trunc('hour', dep_utc) h, count(*) n,
           avg(CASE WHEN dep_delay_min >= {LATE} THEN 1 ELSE 0 END) v
    FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 AND dep_delay_min IS NOT NULL AND origin IN ({inl})
    GROUP BY 1,2""").fetchall()
dep = con.execute(f"""
    SELECT origin ap, date_trunc('hour', dep_utc + INTERVAL (dep_delay_min) MINUTE) h,
           count(*) n, avg(CASE WHEN dep_delay_min >= {LATE} THEN 1 ELSE 0 END) v
    FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 AND dep_delay_min IS NOT NULL AND origin IN ({inl})
    GROUP BY 1,2""").fetchall()
pairs = con.execute(f"""
    SELECT origin, dest, count(*) n FROM read_parquet('{F}', hive_partitioning=true)
    WHERE cancelled = 0 AND flight_date < DATE '2024-01-01'
      AND origin IN ({inl}) AND dest IN ({inl}) GROUP BY 1,2""").fetchall()

hours = sorted({r[1] for r in tgt} | {r[1] for r in dep})
hi = {h: i for i, h in enumerate(hours)}; S = (N, len(hours))
def grid(rows, c, d=np.nan):
    M = np.full(S, d)
    for r in rows:
        if r[0] in ai and r[1] in hi and r[c] is not None: M[ai[r[0]], hi[r[1]]] = r[c]
    return M
y_mat = grid(tgt, 3); y_mat[grid(tgt, 2, 0.0) < MIND] = np.nan
late = grid(dep, 3);  late[grid(dep, 2, 0.0) < MIND] = np.nan

A = np.zeros((N, N))
for o, d, n in pairs:
    if ai[d] != ai[o]: A[ai[d], ai[o]] = n
order = np.argsort(-A, axis=1)                     # neighbours by flight volume
s = A.sum(axis=1, keepdims=True); An = np.divide(A, s, out=np.zeros_like(A), where=s > 0)
U = 1.0 - np.eye(N); U = U / U.sum(axis=1, keepdims=True)

def agg(W):
    src = np.nan_to_num(late, nan=0.0); seen = (~np.isnan(late)).astype(float)
    num, den = W @ src, W @ seen
    return np.divide(num, den, out=np.full(S, np.nan), where=den > 0.2)
flow_mean, nat_mean = agg(An), agg(U)

K = 5
topk = [np.array([late[order[i, k]] for i in range(N)]) for k in range(K)]

def ridge(X, y, lam=1.0):
    X1 = np.hstack([np.ones((len(X), 1)), X]); P = np.eye(X1.shape[1])*lam; P[0, 0] = 0
    return np.linalg.solve(X1.T@X1 + P, X1.T@y)
def r2(y, p): return 1.0 - ((y-p)**2).sum()/((y-y.mean())**2).sum()

hod = np.array([h.hour for h in hours])
cut = int(np.searchsorted(hours, dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)))
MAXLAG = 6

RESULTS["q2_aggregation"] = {}
for H in HORIZONS:
    rows = []
    for i in range(N):
        for j in range(MAXLAG, S[1]-H):
            t = y_mat[i, j+H]
            own6 = late[i, j-MAXLAG+1:j+1]
            nb = [flow_mean[i, j], nat_mean[i, j]] + [topk[k][i, j] for k in range(K)]
            if np.isnan(t) or np.isnan(own6).any() or any(np.isnan(v) for v in nb):
                continue
            rows.append((i, j, t, own6, nb))
    y = np.array([r[2] for r in rows]); tt = np.array([r[1] for r in rows])
    own6 = np.array([r[3] for r in rows]); nb = np.array([r[4] for r in rows])
    tod = np.stack([np.sin(2*np.pi*hod[tt+H]/24), np.cos(2*np.pi*hod[tt+H]/24)], 1)
    tr, te = tt < cut, tt >= cut
    own3 = own6[:, -3:]
    base3 = np.hstack([own3, tod]); base6 = np.hstack([own6, tod])
    BL = {
        "own history 3 lags (what T12 used)":        base3,
        "own history 6 lags":                        base6,
        "3 lags + national mean":                    np.hstack([base3, nb[:, 1:2]]),
        "3 lags + flow MEAN (T12's H1a)":            np.hstack([base3, nb[:, 0:1]]),
        "3 lags + top-5 neighbours, SEPARATE":       np.hstack([base3, nb[:, 2:2+K]]),
        "3 lags + flow mean + top-5 separate":       np.hstack([base3, nb[:, 0:1], nb[:, 2:2+K]]),
        "6 lags + flow mean + top-5 separate":       np.hstack([base6, nb[:, 0:1], nb[:, 2:2+K]]),
    }
    print(f"\n+{H}h   train {tr.sum():,}  test {te.sum():,}")
    b, got = None, {}
    for nm, X in BL.items():
        w = ridge(X[tr], y[tr])
        sc = r2(y[te], np.hstack([np.ones((int(te.sum()), 1)), X[te]]) @ w)
        if b is None: b = sc
        got[nm] = {"r2": round(float(sc), 4), "gain": round(float(sc - b), 4)}
        print(f"   {nm:<42} R2 {sc:+.4f}   vs base {sc-b:+.4f}")
    got["_test_rows"] = int(te.sum())
    RESULTS["q2_aggregation"][f"horizon_{H}h"] = got


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(RESULTS, indent=2, default=float))
print(f"\nwritten {OUT.name}")
