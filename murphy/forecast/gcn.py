"""A two-layer GCN over airports, one graph per hour.

Each airport's state at the top of an hour predicts its share of late or cancelled departures
over the next 3 hours. With no edges, each airport sees only itself, so the same code is the
comparison that shows what message passing adds. Each airport's own inputs also go straight to
the last layer, so a graph can only add to them.

Nothing from at or after the hour enters the inputs, except the schedule. Blank target: under
MIN_TARGET departures. `usual` comes only from the years a fold trains on.
"""
import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv

from murphy.forecast.features import LATE_MIN
from murphy.forecast.network import LATE_SQL

MIN_TARGET = 3
HORIZON_H = 3
STATE_H = 2
INPUTS = ["log_dep_n", "dep_share_late", "dep_mean_delay", "dep_share_cxl", "log_arr_n", "arr_share_late",
          "log_sched_next", "usual", "hour_sin", "hour_cos", "year_sin", "year_cos"]
BATCH, EPOCHS, HIDDEN, LR = 64, 20, 32, 0.005  # AI choices


def hourly(paths, airports, start, end):
    """Tables hour by airport: departures and arrivals known in each hour, and what was scheduled."""
    con = duckdb.connect()
    con.execute(f"""CREATE TEMP VIEW f AS SELECT * FROM read_parquet({paths!r}) WHERE origin IN {tuple(airports)!r} OR dest IN {tuple(airports)!r}""")
    span = (start, end)
    dep = con.execute(f"""SELECT origin a, date_trunc('hour', dep_known_at) h, count(*) n,
        sum(CASE WHEN NOT cancelled AND dep_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) late,
        sum(CASE WHEN cancelled THEN 1 ELSE 0 END) cxl,
        sum(CASE WHEN cancelled THEN 0 ELSE coalesce(dep_delay_min, 0) END) delay
        FROM f WHERE dep_known_at >= ? AND dep_known_at < ? GROUP BY 1, 2""", span).df()
    arr = con.execute(f"""SELECT dest a, date_trunc('hour', arr_known_at) h, count(*) n,
        sum(CASE WHEN cancelled OR diverted OR arr_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) late
        FROM f WHERE arr_known_at >= ? AND arr_known_at < ? GROUP BY 1, 2""", span).df()
    sch = con.execute(f"""SELECT origin a, date_trunc('hour', sched_dep_utc) h, count(*) n,
        sum(CASE WHEN {LATE_SQL} THEN 1 ELSE 0 END) late
        FROM f WHERE sched_dep_utc >= ? AND sched_dep_utc < ? GROUP BY 1, 2""", span).df()
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    def table(d, col):
        d = d[d.a.isin(airports)]
        return d.pivot(index="h", columns="a", values=col).reindex(index=hours, columns=airports).fillna(0).to_numpy(dtype="float32")
    return {"hours": hours, "airports": list(airports), "dep_n": table(dep, "n"), "dep_late": table(dep, "late"),
            "dep_cxl": table(dep, "cxl"), "dep_delay": table(dep, "delay"), "arr_n": table(arr, "n"),
            "arr_late": table(arr, "late"), "sch_n": table(sch, "n"), "sch_late": table(sch, "late")}


def _past(a, width):
    """Sum over the `width` hours before each hour; the hour itself is not included."""
    c = np.vstack([np.zeros((1, a.shape[1])), a.cumsum(0)])
    i = np.arange(len(a))
    return c[i] - c[np.maximum(i - width, 0)]


def _next(a, width):
    """Sum over this hour and the next width - 1."""
    c = np.vstack([np.zeros((1, a.shape[1])), a.cumsum(0)])
    i = np.arange(len(a))
    return c[np.minimum(i + width, len(a))] - c[i]


def targets(t):
    n, late = _next(t["sch_n"], HORIZON_H), _next(t["sch_late"], HORIZON_H)
    y = np.divide(late, n, out=np.zeros_like(late), where=n > 0)
    return y.astype("float32"), np.where(n >= MIN_TARGET, n, 0).astype("float32")


def usual_table(t, train: np.ndarray) -> np.ndarray:
    """Late share per airport, hour of day and month, from training hours only."""
    h, m = t["hours"].hour.to_numpy(), t["hours"].month.to_numpy() - 1
    n, late = np.zeros((12, 24, t["sch_n"].shape[1])), np.zeros((12, 24, t["sch_n"].shape[1]))
    np.add.at(n, (m[train], h[train]), t["sch_n"][train])
    np.add.at(late, (m[train], h[train]), t["sch_late"][train])
    return np.divide(late, n, out=np.zeros_like(late), where=n > 0)


def node_inputs(t, usual: np.ndarray) -> np.ndarray:
    dep_n, arr_n = _past(t["dep_n"], STATE_H), _past(t["arr_n"], STATE_H)
    share = lambda a: np.divide(a, dep_n, out=np.zeros_like(a), where=dep_n > 0)
    h, m = t["hours"].hour.to_numpy(), t["hours"].month.to_numpy() - 1
    doy = t["hours"].dayofyear.to_numpy()
    wave = lambda x, period, fn: np.broadcast_to(fn(2 * np.pi * x / period)[:, None], dep_n.shape)
    x = np.stack([np.log1p(dep_n), share(_past(t["dep_late"], STATE_H)), share(_past(t["dep_delay"], STATE_H)) / 60,
                  share(_past(t["dep_cxl"], STATE_H)), np.log1p(arr_n),
                  np.divide(_past(t["arr_late"], STATE_H), arr_n, out=np.zeros_like(arr_n), where=arr_n > 0),
                  np.log1p(_next(t["sch_n"], HORIZON_H)), usual[m, h],
                  wave(h, 24, np.sin), wave(h, 24, np.cos), wave(doy, 365.25, np.sin), wave(doy, 365.25, np.cos)], axis=-1)
    assert x.shape[-1] == len(INPUTS)
    return x.astype("float32")


class AirportGCN(torch.nn.Module):
    def __init__(self, n_inputs=len(INPUTS), hidden=HIDDEN):
        super().__init__()
        self.a, self.b = GCNConv(n_inputs, hidden), GCNConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden + n_inputs, 1)

    def forward(self, x, edges, weights):
        h = self.a(x, edges, weights).relu()
        h = self.b(h, edges, weights).relu()
        return self.head(torch.cat([h, x], dim=1)).squeeze(1)


def edge_tensors(graph, airports):
    """Both directions of each link, or nothing for the no-edge run."""
    if graph is None or graph.number_of_edges() == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0)
    ix = {a: i for i, a in enumerate(airports)}
    pairs = [(ix[a], ix[b], d["weight"]) for a, b, d in graph.edges(data=True)]
    s, d, w = map(np.array, zip(*pairs))
    return (torch.tensor(np.stack([np.r_[s, d], np.r_[d, s]]), dtype=torch.long),
            torch.tensor(np.r_[w, w], dtype=torch.float32))


def _batch_edges(edges, weights, k, nodes):
    shift = (torch.arange(k) * nodes).repeat_interleave(edges.shape[1])
    return edges.repeat(1, k) + shift, weights.repeat(k)


def fit(x, y, w, train, edges, weights, epochs=EPOCHS, seed=0, log=None):
    """Weighted squared error. x is standardized from training hours; returns (model, mean, sd)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, sd = x[train].mean((0, 1)), x[train].std((0, 1)) + 1e-6
    nodes = x.shape[1]
    model, idx = AirportGCN(x.shape[-1]), np.flatnonzero(train)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    full = _batch_edges(edges, weights, BATCH, nodes)
    for e in range(epochs):
        order, total, weight = rng.permutation(idx), 0.0, 0.0
        for i in range(0, len(order) - BATCH + 1, BATCH):
            b = order[i:i + BATCH]
            bx = torch.from_numpy((x[b] - mean) / sd).reshape(-1, x.shape[-1])
            by, bw = torch.from_numpy(y[b]).reshape(-1), torch.from_numpy(w[b]).reshape(-1)
            loss = ((model(bx, *full) - by) ** 2 * bw).sum() / bw.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            total, weight = total + loss.item(), weight + 1
        if log:
            log(f"epoch {e + 1}: train {total / max(weight, 1):.5f}")
    return model, mean, sd


@torch.no_grad()
def predict(model, mean, sd, x, edges, weights, batch=256):
    model.eval()
    nodes, out = x.shape[1], np.zeros(x.shape[:2], dtype="float32")
    for i in range(0, len(x), batch):
        b = x[i:i + batch]
        ee = _batch_edges(edges, weights, len(b), nodes)
        p = model(torch.from_numpy((b - mean) / sd).reshape(-1, x.shape[-1]), *ee)
        out[i:i + batch] = p.reshape(len(b), nodes).clamp(0, 1).numpy()
    return out


def hour_lookup(as_of: pd.Series, airport: pd.Series, preds: np.ndarray, hours, airports) -> np.ndarray:
    """The prediction made at the top of the hour at or before as_of. Blank if no such hour or airport."""
    i = pd.DatetimeIndex(as_of).floor("h")
    h = pd.Series(np.arange(len(hours)), index=hours).reindex(i).to_numpy()
    a = airport.map({x: k for k, x in enumerate(airports)}).to_numpy()
    out, ok = np.full(len(as_of), np.nan, dtype="float32"), ~np.isnan(h) & ~np.isnan(a)
    out[ok] = preds[h[ok].astype(int), a[ok].astype(int)]
    return out
