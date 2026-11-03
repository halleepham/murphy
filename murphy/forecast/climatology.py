"""Climatology baseline (docs/decisions/0003, 0008): outcome shares in the past.

For a forecast at as_of, the probability of each outcome class is its share
among legs whose outcome was known strictly before as_of (arr_known_at <
as_of, the rule flights_as_of uses for status), over the trailing
WINDOW_DAYS. This is 0008 Decision 4.1: no leg's own or later outcome is used.

Two versions:
  global  all legs.
  origin  legs from the same origin airport, shrunk toward global:
          p = (n_k + ALPHA * p_global_k) / (n + ALPHA).
          ALPHA = 100 legs is an AI choice, not tuned. It keeps a small
          airport from getting 0% or 100% from a handful of legs.

How: count each class per arr_known_at, take running totals, then an ASOF
JOIN finds the latest total strictly before as_of. The total 365 days
earlier is subtracted. One query, not one per forecast.
"""

from pathlib import Path

import duckdb

from murphy.forecast.sample import CLASSES, FLIGHTS, ROOT, label_sql

WINDOW_DAYS = 365
ALPHA = 100
OUT_DIR = ROOT / "data" / "forecast"


def col(c: str) -> str:
    """Class name -> column name: '15-30' -> 'p_15_30', '180+' -> 'p_180p'."""
    return "p_" + c.replace("-", "_").replace("+", "p")


def _totals_sql(by: str | None) -> str:
    """Running count of each class, at every distinct time an outcome became known."""
    key = f"{by}, " if by else ""
    counts = ", ".join(f"count(*) FILTER (outcome = '{c}') AS n_{i}" for i, c in enumerate(CLASSES))
    running = ", ".join(f"sum(n_{i}) OVER w AS n_{i}" for i in range(len(CLASSES)))
    part = f"PARTITION BY {by} " if by else ""
    return f"""
        SELECT {key}t, {running}
        FROM (SELECT {key}arr_known_at AS t, {counts}
              FROM (SELECT *, {label_sql()} AS outcome FROM read_parquet($src))
              WHERE outcome IS NOT NULL AND arr_known_at IS NOT NULL
              GROUP BY ALL)
        WINDOW w AS ({part}ORDER BY t)
    """


def _window_counts(alias: str, by: str | None) -> tuple[str, str]:
    """SQL joins and expressions for class counts inside the trailing window."""
    on = f"r.{by} = {alias}_now.{by} AND " if by else ""
    on_old = f"r.{by} = {alias}_old.{by} AND " if by else ""
    joins = f"""
        ASOF LEFT JOIN {alias} {alias}_now ON {on}r.as_of_utc > {alias}_now.t
        ASOF LEFT JOIN {alias} {alias}_old
          ON {on_old}r.as_of_utc - to_days({WINDOW_DAYS}) > {alias}_old.t"""
    n = [f"(coalesce({alias}_now.n_{i}, 0) - coalesce({alias}_old.n_{i}, 0))"
         for i in range(len(CLASSES))]
    return joins, n


def climatology(rows: Path | str, out: Path | str, by: str | None = None,
                flights: Path | str = FLIGHTS) -> Path:
    """Write flight_id, lead_h, and one probability column per class."""
    g_join, g = _window_counts("g", None)
    g_total = " + ".join(g)
    g_p = [f"{x} / nullif({g_total}, 0)" for x in g]
    joins, probs = g_join, g_p
    ctes = f"g AS ({_totals_sql(None)})"
    if by:
        o_join, o = _window_counts("o", by)
        o_total = " + ".join(o)
        probs = [f"({x} + {ALPHA} * {p}) / ({o_total} + {ALPHA})" for x, p in zip(o, g_p)]
        joins += o_join
        ctes += f", o AS ({_totals_sql(by)})"
    select = ", ".join(f"{p} AS {col(c)}" for p, c in zip(probs, CLASSES))
    sql = f"""
        WITH {ctes}
        SELECT r.flight_id, r.lead_h, {g_total} AS n_history, {select}
        FROM read_parquet($rows) r {joins}
        ORDER BY r.flight_id, r.lead_h
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql(sql, params={"src": f"{flights}/*.parquet", "rows": str(rows)}).to_parquet(str(out))
    return out


if __name__ == "__main__":
    rows = OUT_DIR / "rows.parquet"
    for by in (None, "origin"):
        p = climatology(rows, OUT_DIR / f"clim_{by or 'global'}.parquet", by=by)
        print(f"-> {p}")
