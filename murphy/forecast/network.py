"""Airport communities and the regional inputs built from them.

A community is a group of airports that go bad together. For each airport and 3-hour block:
the share of departures 15+ minutes late or cancelled, minus that airport's usual for the hour
and month. Correlate between airports, link each to its TOP most correlated, find groups.

Group f  community state: last 2 h of departures at the other airports in the community of the
         origin, and of the destination. Own airports are left out: group C already has them.
Group n  the whole country over the same window. The simpler thing to compare with: no graph.

Windows are [as_of - 2 h, as_of), the same as features._windowed. 2026 rows stay blank.
"""
import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

from murphy.forecast.climatology import OUT_DIR
from murphy.forecast.features import LATE_MIN, STATE_HOURS
from murphy.forecast.sample import FLIGHTS

OUT = OUT_DIR / "network"
MIN_DEPARTURES = 20_000  # about 27 a day over 2023-24
TOP = 8  # an AI choice, untested
SEEDS = 30
LATE_SQL = f"(cancelled OR dep_delay_min >= {LATE_MIN})"
WINDOW_NS = STATE_HOURS * 3600 * 10**9


def files(years, flights=FLIGHTS) -> list[str]:
    return [str(p) for y in years for p in sorted(flights.glob(f"{y}_*.parquet"))]


def big_airports(paths, minimum=MIN_DEPARTURES) -> list[str]:
    rows = duckdb.sql(f"SELECT origin FROM read_parquet({paths!r}) GROUP BY 1 HAVING count(*) >= {minimum} ORDER BY 1")
    return [r[0] for r in rows.fetchall()]


def excess_panel(paths, airports) -> pd.DataFrame:
    """Blocks by airports: late share minus that airport's usual for the hour and month."""
    d = duckdb.sql(f"""SELECT origin a, time_bucket(INTERVAL 3 HOUR, sched_dep_utc) t,
                              avg(CASE WHEN {LATE_SQL} THEN 1.0 ELSE 0.0 END) s, count(*) n
                       FROM read_parquet({paths!r}) GROUP BY 1, 2""").df()
    d = d[d.a.isin(airports) & (d.n >= 3)].copy()
    d["x"] = d.s - d.groupby(["a", d.t.dt.hour, d.t.dt.month]).s.transform("mean")
    return d.pivot(index="t", columns="a", values="x")


def co_graph(panel: pd.DataFrame, airports, top=TOP, min_periods=300) -> nx.Graph:
    corr = panel.corr(min_periods=min_periods).reindex(index=airports, columns=airports).to_numpy().copy()
    np.fill_diagonal(corr, np.nan)
    g = nx.Graph()
    g.add_nodes_from(airports)
    for i, a in enumerate(airports):
        for j in np.argsort(-np.nan_to_num(corr[i], nan=-9))[:top]:
            if corr[i][j] > 0:  # blank or negative is no link
                g.add_edge(a, airports[j], weight=float(corr[i][j]))
    return g


def add_graphs(graphs, airports) -> nx.Graph:
    out = nx.Graph()
    out.add_nodes_from(airports)
    for g in graphs:
        for a, b, d in g.edges(data=True):
            out.add_edge(a, b, weight=out[a][b]["weight"] + d["weight"] if out.has_edge(a, b) else d["weight"])
    return out


def best_louvain(g: nx.Graph, seeds=SEEDS) -> list[set]:
    """Louvain depends on its seed, so run many and keep the best modularity."""
    runs = [louvain_communities(g, weight="weight", seed=s) for s in range(seeds)]
    return sorted(max(runs, key=lambda c: modularity(g, c, weight="weight")), key=len, reverse=True)


def route_counts(paths) -> pd.DataFrame:
    return duckdb.sql(f"SELECT origin a, dest b, count(*) n FROM read_parquet({paths!r}) GROUP BY 1, 2").df()


def assign_small(groups: list[set], routes: pd.DataFrame) -> dict:
    """Big airports keep their group. Any other joins the group it flies to most. No flights to one: no group."""
    member = {a: i for i, g in enumerate(groups) for a in g}
    both = pd.concat([routes, routes.rename(columns={"a": "b", "b": "a"})])
    both = both[~both.a.isin(member) & both.b.isin(member)]
    both = both.assign(c=both.b.map(member)).groupby(["a", "c"]).n.sum().reset_index()
    both = both.sort_values(["a", "n", "c"], ascending=[True, False, True]).drop_duplicates("a")
    return member | dict(zip(both.a, both.c.astype(int)))


class Events:
    """Departures as they became known, per key, for fast window counts."""

    def __init__(self, keys, t_ns, late):
        order = np.lexsort((t_ns, keys))
        keys, self.t, late = np.asarray(keys)[order], np.asarray(t_ns)[order], np.asarray(late, dtype=float)[order]
        self.cum = np.concatenate([[0.0], late.cumsum()])
        self.uniq, self.start = np.unique(keys, return_index=True)
        self.end = np.append(self.start[1:], len(keys))

    def window(self, keys, as_of_ns, width=WINDOW_NS):
        """n and late in [as_of - width, as_of), per row. A key with no events gives 0."""
        keys, as_of_ns = np.asarray(keys), np.asarray(as_of_ns)
        n, late = np.zeros(len(keys)), np.zeros(len(keys))
        where = {k: (s, e) for k, s, e in zip(self.uniq, self.start, self.end)}
        for k in pd.unique(keys):
            if k not in where:
                continue
            s, e = where[k]
            i = np.flatnonzero(keys == k)
            lo = s + np.searchsorted(self.t[s:e], as_of_ns[i] - width, side="left")
            hi = s + np.searchsorted(self.t[s:e], as_of_ns[i], side="left")
            n[i], late[i] = hi - lo, self.cum[hi] - self.cum[lo]
        return n, late


def to_ns(s: pd.Series) -> np.ndarray:
    return s.astype("datetime64[ns]").astype("int64").to_numpy()


def load_events(paths, before) -> pd.DataFrame:
    """Departures known before `before`. A later one could never reach these rows, and is not read."""
    return duckdb.execute(f"""SELECT origin a, dep_known_at t, CASE WHEN {LATE_SQL} THEN 1 ELSE 0 END l
                              FROM read_parquet({paths!r}) WHERE dep_known_at IS NOT NULL AND dep_known_at < ?""",
                          [before]).df()


def state_inputs(rows: pd.DataFrame, events: pd.DataFrame, member: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groups f and n, in row order. Needs origin, dest, as_of_utc, split. Test rows stay blank."""
    t = to_ns(events.t)
    by_air = Events(events.a.to_numpy(), t, events.l)
    groups = events.a.map(member)
    known = groups.notna().to_numpy()
    by_group = Events(groups[known].astype(int).to_numpy(), t[known], events.l[known])
    everywhere = Events(np.zeros(len(events), dtype=int), t, events.l)

    live = (rows.split != "test").to_numpy()
    as_of = to_ns(rows.as_of_utc)
    f, n = {}, {}
    for side, col in (("o", "origin"), ("d", "dest")):
        g = rows[col].map(member)
        nn, ll = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        ok = live & g.notna().to_numpy()
        a, b = by_group.window(g[ok].astype(int).to_numpy(), as_of[ok])
        c, d = by_air.window(rows[col].to_numpy()[ok], as_of[ok])
        nn[ok], ll[ok] = a - c, b - d
        f[f"comm_{side}_2h_n"] = nn
        with np.errstate(divide="ignore", invalid="ignore"):
            f[f"comm_{side}_2h_share_late"] = np.where(nn > 0, ll / nn, np.nan)
    nn, ll = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    nn[live], ll[live] = everywhere.window(np.zeros(live.sum(), dtype=int), as_of[live])
    n["nat_2h_n"] = nn
    with np.errstate(divide="ignore", invalid="ignore"):
        n["nat_2h_share_late"] = np.where(nn > 0, ll / nn, np.nan)
    rid = {"rid": np.arange(len(rows))}
    return pd.DataFrame(rid | f).astype({k: "float32" for k in f}), pd.DataFrame(rid | n).astype({k: "float32" for k in n})
