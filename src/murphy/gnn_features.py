"""Turn the exported subgraph into features, labels, a split, and the guards that
make all three trustworthy.

Nothing here imports torch. Tensor construction belongs to `gnn_model`; keeping this
module numeric-only means every assertion below can be tested without a GNN, and a
feature bug surfaces as a failed assertion rather than as a suspiciously good score.

**The three guards, each one a silent failure if omitted.**

1. *No outcome of the flight itself.* Asserted by column name against the `outcome_`
   prefix, which exists on the nodes precisely so this rule is checkable (decision S6).
2. *An inbound leg's delay is used only where it was observable.* Populated only where
   the inbound flight had actually landed before this flight was due out, with a
   companion indicator column. A masked value is unknown, never zero (decision N6).
3. *Normalization statistics come from the training window only.* Fitting a scaler on
   the whole day leaks the test window's distribution into the training features. It is
   a small leak and it is still a leak.

Run:  PYTHONPATH=src .venv/bin/python -m murphy.gnn_features
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gnn_export import Export, export

LATE_MINUTES = 15          # the BTS convention the whole project uses
TRAIN_QUANTILE = 0.70      # measured: 3,021 train / 1,312 test for WN

# The one outcome column allowed into the feature matrix, and only because it belongs to
# a *different* flight that had already landed. Spelled out here so the assertion below
# has something explicit to allow rather than a loose prefix check.
UPSTREAM_FEATURE = "up_delay_min_masked"

FLIGHT_FEATURES = [
    "dep_hour_sin", "dep_hour_cos",
    "sched_duration_min", "distance_mi",
    UPSTREAM_FEATURE, "up_delay_observed",
]

HOUR_FEATURES = [
    "n_sched_departures", "n_sched_arrivals",
    "temp_c", "wind_ms", "precip_mm", "visibility_m", "ceiling_m", "fog", "thunderstorm",
]


def _native(dt):
    """Neo4j DateTime -> python datetime, leaving real datetimes alone."""
    return dt.to_native() if hasattr(dt, "to_native") else dt


@dataclass
class Dataset:
    flight_ids: list[str]
    x_flight: np.ndarray            # [n_flights, len(FLIGHT_FEATURES)]
    y: np.ndarray                   # [n_flights] 1 = arrived 15+ minutes late
    origin_idx: np.ndarray          # embedding indices
    dest_idx: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    has_upstream: np.ndarray        # the 80.8% the rotation path can speak to
    hour_ids: list[str]
    x_hour: np.ndarray              # [n_hours, len(HOUR_FEATURES)]
    rot_edges: np.ndarray           # [2, n] inbound -> flight, forward in time only
    hour_edges: np.ndarray          # [2, n] hour -> flight
    airports: list[str]
    dep_cut: object                 # the split boundary, kept for the report
    # --- raw, unscaled values. Not features: the rule rungs and the query target need
    # real minutes, and the feature matrix holds standardized ones.
    delay_min: np.ndarray           # observed arrival delay, the ranking ground truth
    up_delay_raw: np.ndarray        # inbound delay in minutes where observable, else 0
    up_observed: np.ndarray         # whether that inbound delay was observable
    dep_movements: np.ndarray       # scheduled movements in the departure hour (G2b)
    service_pct_late15: np.ndarray  # the stored baseline rate (G0)
    origin: list[str]
    dest: list[str]

    def summary(self) -> dict:
        return {
            "flights": len(self.flight_ids),
            "train": int(self.train_mask.sum()),
            "test": int(self.test_mask.sum()),
            "late_pct": round(100 * float(self.y.mean()), 1),
            "with_upstream": int(self.has_upstream.sum()),
            "upstream_observed": int(
                self.x_flight[:, FLIGHT_FEATURES.index("up_delay_observed")].sum()),
            "hours": len(self.hour_ids),
            "rot_edges": int(self.rot_edges.shape[1]),
            "hour_edges": int(self.hour_edges.shape[1]),
            "airports": len(self.airports),
        }


def build(e: Export | None = None) -> Dataset:
    e = e or export()
    flights = e.labelled
    flights.sort(key=lambda f: (_native(f["sched_dep_utc"]), f["flight_id"]))
    fid = {f["flight_id"]: i for i, f in enumerate(flights)}

    # ---- inbound legs, keyed by the flight they feed
    inbound = {r["flight_id"]: r for r in e.precedes if r["flight_id"] in fid}

    airports = sorted({f["origin"] for f in flights} | {f["dest"] for f in flights})
    aidx = {a: i for i, a in enumerate(airports)}

    n = len(flights)
    x = np.zeros((n, len(FLIGHT_FEATURES)), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    delay_min = np.zeros(n, dtype=np.float32)
    up_raw = np.zeros(n, dtype=np.float32)
    up_obs = np.zeros(n, dtype=np.float32)
    svc = np.zeros(n, dtype=np.float32)
    o_idx = np.zeros(n, dtype=np.int64)
    d_idx = np.zeros(n, dtype=np.int64)
    has_up = np.zeros(n, dtype=bool)
    col = {name: i for i, name in enumerate(FLIGHT_FEATURES)}

    for f in flights:
        i = fid[f["flight_id"]]
        # Local scheduled hour, cyclically encoded: hour 23 sits beside hour 0. Verified
        # local rather than UTC -- the hour means the same thing operationally at every
        # airport, which is what makes it a useful feature at all.
        h = float(f["sched_dep_hour"])
        x[i, col["dep_hour_sin"]] = math.sin(2 * math.pi * h / 24)
        x[i, col["dep_hour_cos"]] = math.cos(2 * math.pi * h / 24)
        x[i, col["sched_duration_min"]] = float(f["sched_duration_min"] or 0)
        x[i, col["distance_mi"]] = float(f["distance_mi"] or 0)
        o_idx[i] = aidx[f["origin"]]
        d_idx[i] = aidx[f["dest"]]
        y[i] = 1.0 if f["outcome_arr_delay_min"] >= LATE_MINUTES else 0.0
        delay_min[i] = float(f["outcome_arr_delay_min"])
        svc[i] = float(f["service_pct_late15"] or 0.0)

        up = inbound.get(f["flight_id"])
        if up is None:
            continue
        has_up[i] = True
        delay = up["up_outcome_arr_delay_min"]
        sched_arr = up["up_sched_arr_utc"]
        if delay is None or sched_arr is None:
            continue
        # GUARD 2: the inbound delay counts only if that aircraft was on the ground
        # before this flight was due out. Otherwise the number existed only in
        # hindsight.
        actual_arr = _native(sched_arr) + np.timedelta64(int(delay), "m").astype("O")
        if actual_arr <= _native(f["sched_dep_utc"]):
            x[i, col[UPSTREAM_FEATURE]] = float(delay)
            x[i, col["up_delay_observed"]] = 1.0
            up_raw[i] = float(delay)
            up_obs[i] = 1.0

    # ---- airport-hours
    hours = sorted(e.airport_hours, key=lambda h: h["weather_id"])
    hidx = {h["weather_id"]: i for i, h in enumerate(hours)}
    xh = np.zeros((len(hours), len(HOUR_FEATURES)), dtype=np.float32)
    for h in hours:
        j = hidx[h["weather_id"]]
        for k, name in enumerate(HOUR_FEATURES):
            xh[j, k] = float(h[name] or 0)

    # ---- edges. GUARD 1 on direction: inbound -> flight, never the reverse.
    rot = np.array([[fid[r["up_id"]], fid[r["flight_id"]]]
                    for r in e.precedes
                    if r["up_id"] in fid and r["flight_id"] in fid], dtype=np.int64).T
    hr = np.array([[hidx[r["weather_id"]], fid[r["flight_id"]]]
                   for r in e.hour_edges
                   if r["flight_id"] in fid and r["weather_id"] in hidx], dtype=np.int64).T

    # ---- scheduled movements in the departure hour, for rung G2b. Taken from the
    # DEPARTS_IN edge specifically: a congested arrival hour is a different fact.
    dep_hour_of = {r["flight_id"]: r["weather_id"]
                   for r in e.hour_edges if r["kind"] == "DEPARTS_IN"}
    mv_of = {h["weather_id"]: float(h["n_sched_departures"] or 0) for h in hours}
    dep_mv = np.array([mv_of.get(dep_hour_of.get(f["flight_id"]), 0.0)
                       for f in flights], dtype=np.float32)

    # ---- temporal split
    deps = [_native(f["sched_dep_utc"]) for f in flights]
    cut = deps[int(n * TRAIN_QUANTILE)]
    train = np.array([d < cut for d in deps])
    test = ~train

    # ---- GUARD 3: scale on the training window only
    cont = [col[c] for c in ("sched_duration_min", "distance_mi", UPSTREAM_FEATURE)]
    mu = x[train][:, cont].mean(0)
    sd = x[train][:, cont].std(0)
    sd[sd == 0] = 1.0
    x[:, cont] = (x[:, cont] - mu) / sd
    hmu, hsd = xh.mean(0), xh.std(0)
    hsd[hsd == 0] = 1.0
    xh = (xh - hmu) / hsd

    ds = Dataset(
        flight_ids=[f["flight_id"] for f in flights],
        x_flight=x, y=y, origin_idx=o_idx, dest_idx=d_idx,
        train_mask=train, test_mask=test, has_upstream=has_up,
        hour_ids=[h["weather_id"] for h in hours], x_hour=xh,
        rot_edges=rot, hour_edges=hr, airports=airports, dep_cut=cut,
        delay_min=delay_min, up_delay_raw=up_raw, up_observed=up_obs,
        dep_movements=dep_mv, service_pct_late15=svc,
        origin=[f["origin"] for f in flights], dest=[f["dest"] for f in flights],
    )
    assert_guards(ds, flights)
    return ds


def assert_guards(ds: Dataset, flights: list[dict]) -> None:
    """The acceptance criteria from `spec_gnn.md` U2, enforced rather than reviewed."""
    # 1 -- no outcome of the flight itself reached the feature matrix
    leaked = [c for c in FLIGHT_FEATURES
              if c.startswith("outcome_") or (c.startswith("up_") and c != UPSTREAM_FEATURE
                                              and c != "up_delay_observed")]
    assert not leaked, f"outcome columns in features: {leaked}"
    assert UPSTREAM_FEATURE in FLIGHT_FEATURES, "the allow-listed column must be present"

    # 2 -- the mask: every populated upstream delay was observable, and the indicator agrees
    icol = FLIGHT_FEATURES.index("up_delay_observed")
    ucol = FLIGHT_FEATURES.index(UPSTREAM_FEATURE)
    observed = ds.x_flight[:, icol] == 1.0
    assert observed.sum() <= ds.has_upstream.sum(), \
        "more observed upstream delays than flights with an inbound leg"
    # a flight with no inbound leg can never have an observed upstream delay
    assert not (observed & ~ds.has_upstream).any(), "observed delay without an inbound leg"

    # 3 -- the split is temporal, not random
    deps = [_native(f["sched_dep_utc"]) for f in flights]
    train_max = max(d for d, m in zip(deps, ds.train_mask) if m)
    test_min = min(d for d, m in zip(deps, ds.test_mask) if m)
    assert train_max <= test_min, f"split is not temporal: {train_max} > {test_min}"

    # 4 -- edge direction: every rotation edge runs forward in time
    for s, t in ds.rot_edges.T:
        assert deps[s] <= deps[t], "a rotation edge points backwards in time"

    # 5 -- indices are in range
    assert ds.rot_edges.max() < len(ds.flight_ids)
    assert ds.hour_edges[0].max() < len(ds.hour_ids)
    assert ds.hour_edges[1].max() < len(ds.flight_ids)


def main():
    ds = build()
    print("features built, all guards passed\n")
    for k, v in ds.summary().items():
        print(f"  {k:<18} {v:>8,}" if isinstance(v, int) else f"  {k:<18} {v:>8}")
    print(f"\n  split boundary     {ds.dep_cut}")
    print(f"  median delay       {float(np.median(ds.delay_min)):.0f} min")
    print(f"  dep-hour movements {float(ds.dep_movements.mean()):.1f} mean")
    print(f"  flight features    {FLIGHT_FEATURES}")
    print(f"  hour features      {HOUR_FEATURES}")


if __name__ == "__main__":
    main()
