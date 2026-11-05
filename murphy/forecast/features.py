"""Model inputs for Step 2 (docs/decisions/0008 Decision 6), one row per forecast.

Every input is computed as of the forecast, using the same rules as
murphy.datalayer.asof, but in bulk: one query per group, not one per row.

  A schedule   carrier, local hour, weekday, month, distance, duration, lead.
  B rates      trailing 365 days, known before as_of: P(late), P(cancel) and
               mean delay by origin, dest, carrier, route, carrier x origin,
               shrunk toward all airports with climatology's ALPHA. Plus the
               by-origin climatology.
  C state      last 2 h of departures at the origin and arrivals at the dest:
               how many, share late, mean delay, cancellations.
  D aircraft   the tail's previous leg by schedule: scheduled turn time, its
               status and delay as of the forecast. Blank at 24 h lead, since
               the tail is visible only once sched_dep - 24 h < as_of (0006).
  E weather    latest observation at the origin; the TAF in force at departure
               (origin) and at arrival (dest). Blank outside the 30 airports,
               with has-weather flags. Blank never means calm.

A blank stays blank: the model handles missing values natively.

Usage: python -m murphy.forecast.features  -> data/forecast/features/{a..e}.parquet
"""

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from murphy.datalayer.weather import AIRPORTS
from murphy.forecast.climatology import ALPHA, OUT_DIR, WINDOW_DAYS, col
from murphy.forecast.sample import CLASSES, FLIGHTS, ROOT

WEATHER = ROOT / "data" / "weather"
OUT = OUT_DIR / "features"
LATE_MIN = 15  # DOT: 15+ minutes late
STATE_HOURS = 2
OBS_MAX_AGE_MIN = 24 * 60  # same lookback as asos_as_of
TAF_MAX_HORIZON_H = 30  # US TAFs run 24 or 30 h; the data has no end time
STATUS = {"scheduled": 0, "departed": 1, "arrived": 2, "cancelled": 3, "diverted": 4}
RATE_KEYS = {"origin": ["origin"], "dest": ["dest"], "carrier": ["operating_carrier"],
             "route": ["origin", "dest"], "carrier_origin": ["operating_carrier", "origin"]}

ROWS_SQL = "(SELECT file_row_number AS rid, * FROM read_parquet($rows, file_row_number = true))"
# A leg with a known outcome, the same test as sample.py and climatology.py.
RESOLVED = "(cancelled OR diverted OR arr_delay_min IS NOT NULL) AND arr_known_at IS NOT NULL"


def _windowed(con, rows, flights, events: str, by: list[str], width: str) -> pd.DataFrame:
    """Sum event measures per forecast over [as_of - width, as_of), one query.

    events: SQL with columns {by}, t, and numeric measures. Running totals per
    key, then two ASOF JOINs (strict, as in climatology) and a subtraction.
    """
    measures = [c for c in con.sql(f"SELECT * FROM ({events}) LIMIT 0",
                                   params={"src": f"{flights}/*.parquet"}).columns
                if c not in by and c != "t"]
    key = "".join(f"{k}, " for k in by)
    part = f"PARTITION BY {', '.join(by)} " if by else ""
    grouped = ", ".join(f"sum({m}) AS {m}" for m in measures)
    running = ", ".join(f"sum({m}) OVER w AS {m}" for m in measures)
    on = lambda a: "".join(f"r.{k} = {a}.{k} AND " for k in by)
    diff = ", ".join(f"coalesce(a.{m}, 0) - coalesce(b.{m}, 0) AS {m}" for m in measures)
    sql = f"""
        WITH e AS (SELECT {key}t, {running}
                   FROM (SELECT {key}t, {grouped} FROM ({events}) GROUP BY ALL)
                   WINDOW w AS ({part}ORDER BY t))
        SELECT r.rid, {diff}
        FROM {ROWS_SQL} r
        ASOF LEFT JOIN e a ON {on('a')} r.as_of_utc > a.t
        ASOF LEFT JOIN e b ON {on('b')} r.as_of_utc - {width} > b.t
        ORDER BY r.rid
    """
    return con.sql(sql, params={"src": f"{flights}/*.parquet", "rows": str(rows)}).df()


def _outcome_events(keys: str) -> str:
    """One event per leg, when its final outcome became known."""
    return f"""SELECT {keys} arr_known_at AS t, 1.0 AS n,
          CASE WHEN cancelled OR diverted OR arr_delay_min >= {LATE_MIN} THEN 1.0 ELSE 0 END AS n_late,
          CASE WHEN cancelled THEN 1.0 ELSE 0 END AS n_cxl,
          CASE WHEN cancelled OR diverted THEN 0 ELSE 1.0 END AS n_arr,
          CASE WHEN cancelled OR diverted THEN 0 ELSE arr_delay_min END AS sum_delay
        FROM read_parquet($src) WHERE {RESOLVED}"""


def _departure_events() -> str:
    """One event per leg, when its departure or cancellation became known."""
    return f"""SELECT origin, dep_known_at AS t, 1.0 AS n,
          CASE WHEN NOT cancelled AND dep_delay_min >= {LATE_MIN} THEN 1.0 ELSE 0 END AS n_late,
          CASE WHEN cancelled THEN 1.0 ELSE 0 END AS n_cxl,
          CASE WHEN cancelled THEN 0 ELSE 1.0 END AS n_arr,
          CASE WHEN cancelled THEN 0 ELSE coalesce(dep_delay_min, 0) END AS sum_delay
        FROM read_parquet($src) WHERE dep_known_at IS NOT NULL"""


def schedule(con, rows, flights) -> pd.DataFrame:
    """Group A. Hour, weekday and month are the origin's local schedule."""
    return con.sql(f"""
        SELECT r.rid, f.operating_carrier AS carrier,
               CAST(f.crs_dep_local AS INTEGER) // 100 AS local_hour,
               isodow(r.flight_date) AS weekday, month(r.flight_date) AS month,
               r.distance_mi, date_diff('minute', r.sched_dep_utc, r.sched_arr_utc) AS duration_min,
               r.lead_h
        FROM {ROWS_SQL} r JOIN read_parquet($src) f USING (flight_id)
        ORDER BY r.rid""", params={"src": f"{flights}/*.parquet", "rows": str(rows)}).df()


def rates(con, rows, flights, clim_origin=None) -> pd.DataFrame:
    """Group B. p = (n_k + ALPHA * p_all) / (n + ALPHA), as in climatology."""
    width = f"to_days({WINDOW_DAYS})"
    g = _windowed(con, rows, flights, _outcome_events(""), [], width)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_late, g_cxl = g.n_late / g.n, g.n_cxl / g.n
        g_delay = g.sum_delay / g.n_arr
    out = {"rid": g.rid.to_numpy()}
    for name, keys in RATE_KEYS.items():
        w = _windowed(con, rows, flights, _outcome_events("".join(f"{k}, " for k in keys)),
                      keys, width)
        out[f"{name}_p_late"] = ((w.n_late + ALPHA * g_late) / (w.n + ALPHA)).to_numpy()
        out[f"{name}_p_cxl"] = ((w.n_cxl + ALPHA * g_cxl) / (w.n + ALPHA)).to_numpy()
        out[f"{name}_mean_delay"] = ((w.sum_delay + ALPHA * g_delay) / (w.n_arr + ALPHA)).to_numpy()
    df = pd.DataFrame(out)
    if clim_origin is not None:
        cols = ", ".join(f"c.{col(k)} AS clim_{col(k)}" for k in CLASSES)
        c = con.sql(f"""SELECT r.rid, {cols} FROM {ROWS_SQL} r
                        LEFT JOIN read_parquet($clim) c USING (flight_id, lead_h)
                        ORDER BY r.rid""", params={"rows": str(rows), "clim": str(clim_origin)}).df()
        df = pd.concat([df, c.drop(columns="rid")], axis=1)
    return df


def state(con, rows, flights) -> pd.DataFrame:
    """Group C. n counts everything known in the window; shares are blank at n = 0."""
    width = f"to_hours({STATE_HOURS})"
    out = {}
    for prefix, w in (("dep", _windowed(con, rows, flights, _departure_events(), ["origin"], width)),
                      ("arr", _windowed(con, rows, flights, _outcome_events("dest, "), ["dest"], width))):
        out["rid"] = w.rid.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{prefix}_2h_n"] = w.n.to_numpy()
            out[f"{prefix}_2h_share_late"] = (w.n_late / w.n).to_numpy()
            out[f"{prefix}_2h_mean_delay"] = (w.sum_delay / w.n_arr).to_numpy()
            out[f"{prefix}_2h_cxl"] = w.n_cxl.to_numpy()
    return pd.DataFrame(out)


def aircraft(con, rows, flights) -> pd.DataFrame:
    """Group D. The previous leg is chosen by schedule alone, so choosing it
    reveals no outcome; its status follows flights_as_of's rules. Delay so far:
    arrival delay once arrived, departure delay once departed, otherwise how far
    past its scheduled departure as_of is. Blank if cancelled or diverted."""
    a = "p.arr_known_at < r.as_of_utc"
    d = "p.dep_known_at < r.as_of_utc"
    s = STATUS
    df = con.sql(f"""
        WITH p AS (SELECT tail_number, sched_dep_utc, sched_arr_utc, dep_known_at, arr_known_at,
                          cancelled, diverted, dep_delay_min, arr_delay_min
                   FROM read_parquet($src) WHERE tail_number IS NOT NULL
                   QUALIFY row_number() OVER (PARTITION BY tail_number, sched_dep_utc
                                              ORDER BY flight_id) = 1),
             r AS (SELECT r.rid, r.as_of_utc, r.sched_dep_utc, f.tail_number
                   FROM {ROWS_SQL} r JOIN read_parquet($src) f USING (flight_id)
                   WHERE r.sched_dep_utc - INTERVAL 24 HOUR < r.as_of_utc
                     AND f.tail_number IS NOT NULL)
        SELECT r.rid,
          date_diff('minute', p.sched_arr_utc, r.sched_dep_utc) AS prior_turn_min,
          CASE WHEN p.sched_dep_utc IS NULL THEN NULL
               WHEN {a} AND p.diverted THEN {s['diverted']}
               WHEN {a} AND NOT p.cancelled THEN {s['arrived']}
               WHEN {d} AND p.cancelled THEN {s['cancelled']}
               WHEN {d} THEN {s['departed']}
               ELSE {s['scheduled']} END AS prior_status,
          CASE WHEN p.sched_dep_utc IS NULL THEN NULL
               WHEN {a} AND p.diverted THEN NULL
               WHEN {a} AND NOT p.cancelled THEN p.arr_delay_min
               WHEN {d} AND p.cancelled THEN NULL
               WHEN {d} THEN p.dep_delay_min
               ELSE greatest(0, date_diff('minute', p.sched_dep_utc, r.as_of_utc)) END
            AS prior_delay_so_far
        FROM r ASOF LEFT JOIN p ON r.tail_number = p.tail_number
                               AND r.sched_dep_utc > p.sched_dep_utc
        """, params={"src": f"{flights}/*.parquet", "rows": str(rows)}).df()
    n = con.sql("SELECT count(*) FROM read_parquet($rows)", params={"rows": str(rows)}).fetchall()[0][0]
    return df.set_index("rid").reindex(pd.RangeIndex(n, name="rid")).reset_index()


def _wx_flags(c: str) -> str:
    return (f"CASE WHEN regexp_matches({c}, 'TS') THEN 1 ELSE 0 END AS ts, "
            f"CASE WHEN regexp_matches({c}, 'SN|PL|FZ|GR|IC') THEN 1 ELSE 0 END AS snow_ice, "
            f"CASE WHEN regexp_matches({c}, 'FG') THEN 1 ELSE 0 END AS fog")


def observation(con, rows, weather) -> pd.DataFrame:
    """Latest observation known before as_of at the origin, if under 24 h old."""
    layers = ", ".join(f"CASE WHEN trim(skyc{i}) IN ('BKN', 'OVC', 'VV') THEN TRY_CAST(skyl{i} AS DOUBLE) END"
                       for i in range(1, 5))
    fields = ["tmpf", "sknt", "gust", "vsby", "ceiling", "p01i", "ts", "snow_ice", "fog"]
    df = con.sql(f"""
        WITH o AS (SELECT airport, known_at, obs_utc, tmpf, sknt, gust, vsby, p01i,
                          least({layers}) AS ceiling,
                          {_wx_flags("coalesce(wxcodes, '')")}
                   FROM read_parquet($asos)
                   QUALIFY row_number() OVER (PARTITION BY airport, known_at ORDER BY metar) = 1)
        SELECT r.rid, date_diff('minute', o.obs_utc, r.as_of_utc) AS age_min,
               {", ".join(f"o.{c}" for c in fields)}
        FROM {ROWS_SQL} r ASOF LEFT JOIN o ON r.origin = o.airport AND r.as_of_utc > o.known_at
        ORDER BY r.rid""", params={"asos": f"{weather}/asos.parquet", "rows": str(rows)}).df()
    df = df.astype({c: "float64" for c in ["age_min"] + fields})  # an all-blank column has no type
    df.loc[df.age_min > OBS_MAX_AGE_MIN, ["age_min"] + fields] = np.nan
    df.insert(1, "has_obs", df.age_min.notna().astype("float32"))
    return df.rename(columns={c: f"obs_{c}" for c in ["age_min"] + fields})


def taf(con, rows, weather, airport: str, at: str, prefix: str) -> pd.DataFrame:
    """The latest TAF known before as_of, read at time `at`: the base group in
    force, then the worst of it and any TEMPO/PROB group covering `at`."""
    sky = "str_split(regexp_replace(skyc, '[^A-Z,]', '', 'g'), ',')"
    lvl = "str_split(regexp_replace(skyl, '[^0-9,]', '', 'g'), ',')"
    ceil = (f"list_min(list_transform(list_filter(range(1, len({sky}) + 1), "
            f"lambda i: {sky}[i] IN ('BKN', 'OVC', 'VV')), lambda i: TRY_CAST({lvl}[i] AS DOUBLE)))")
    groups = f"""SELECT airport, known_at, period_start_utc, period_end_utc, period_type,
                        sknt, gust, visibility, {ceil} AS ceiling,
                        {_wx_flags("coalesce(presentwx, '')")}
                 FROM read_parquet($taf)"""
    df = con.sql(f"""
        WITH g AS ({groups}),
             base AS (SELECT * FROM g WHERE period_type IN ('initial', 'FM')
                      QUALIFY row_number() OVER (PARTITION BY airport, known_at, period_start_utc
                                                 ORDER BY sknt, visibility) = 1),
             products AS (SELECT DISTINCT airport, known_at FROM g),
             r AS (SELECT rid, {airport} AS airport, as_of_utc, {at} AS t FROM {ROWS_SQL}
                   WHERE {airport} IN {tuple(AIRPORTS)}),
             latest AS (SELECT r.*, p.known_at FROM r
                        ASOF JOIN products p ON r.airport = p.airport AND r.as_of_utc > p.known_at),
             b AS (SELECT l.rid, l.t, l.known_at, x.sknt, x.gust, x.visibility, x.ceiling,
                          x.ts, x.snow_ice, x.fog
                   FROM latest l ASOF LEFT JOIN base x
                     ON l.airport = x.airport AND l.known_at = x.known_at
                    AND l.t >= x.period_start_utc),
             w AS (SELECT l.rid, min(x.visibility) AS visibility, min(x.ceiling) AS ceiling,
                          max(x.sknt) AS sknt, max(x.gust) AS gust, max(x.ts) AS ts,
                          max(x.snow_ice) AS snow_ice, max(x.fog) AS fog
                   FROM latest l JOIN g x ON l.airport = x.airport AND l.known_at = x.known_at
                   WHERE x.period_type IN ('TEMPO', 'PROB30')
                     AND x.period_start_utc <= l.t AND l.t < x.period_end_utc
                   GROUP BY l.rid)
        SELECT b.rid, date_diff('minute', b.known_at, b.t) / 60.0 AS horizon_h,
               greatest(b.sknt, w.sknt) AS sknt, greatest(b.gust, w.gust) AS gust,
               least(b.visibility, w.visibility) AS visibility,
               least(b.ceiling, w.ceiling) AS ceiling,
               greatest(b.ts, w.ts) AS ts, greatest(b.snow_ice, w.snow_ice) AS snow_ice,
               greatest(b.fog, w.fog) AS fog
        FROM b LEFT JOIN w USING (rid)""",
                 params={"taf": f"{weather}/taf.parquet", "rows": str(rows)}).df()
    df = df.astype({c: "float64" for c in df.columns if c != "rid"})
    df = df[df.horizon_h <= TAF_MAX_HORIZON_H]
    n = con.sql("SELECT count(*) FROM read_parquet($rows)", params={"rows": str(rows)}).fetchall()[0][0]
    df = df.set_index("rid").reindex(pd.RangeIndex(n, name="rid")).reset_index()
    df.insert(1, "has", df.horizon_h.notna().astype("float32"))
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "rid"})


def weather_inputs(con, rows, weather) -> pd.DataFrame:
    """Group E."""
    parts = [observation(con, rows, weather),
             taf(con, rows, weather, "origin", "sched_dep_utc", "taf_dep"),
             taf(con, rows, weather, "dest", "sched_arr_utc", "taf_arr")]
    return pd.concat([parts[0]] + [p.drop(columns="rid") for p in parts[1:]], axis=1)


def _save(df: pd.DataFrame, path: Path) -> None:
    """float32 keeps the five files small; carrier stays text."""
    num = [c for c in df.columns if c not in ("rid", "carrier")]
    df[num] = df[num].astype("float32")
    df.to_parquet(path, index=False)


def build(rows=OUT_DIR / "rows.parquet", out=OUT, flights=FLIGHTS, weather=WEATHER,
          clim_origin=OUT_DIR / "clim_origin.parquet", groups="abcde") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB'")
    makers = {"a": lambda: schedule(con, rows, flights),
              "b": lambda: rates(con, rows, flights, clim_origin),
              "c": lambda: state(con, rows, flights),
              "d": lambda: aircraft(con, rows, flights),
              "e": lambda: weather_inputs(con, rows, weather)}
    for g in groups:
        t = time.time()
        df = makers[g]()
        _save(df, out / f"{g}.parquet")
        print(f"{g}: {df.shape[0]:,} rows, {df.shape[1] - 1} inputs, {time.time() - t:.0f} s", flush=True)


def load(out=OUT, groups="abcde") -> pd.DataFrame:
    """The chosen groups side by side, in rows.parquet order."""
    parts = [pd.read_parquet(Path(out) / f"{g}.parquet") for g in groups]
    for p in parts:
        assert (p.rid.to_numpy() == np.arange(len(p))).all(), "feature file out of row order"
    return pd.concat([p.drop(columns="rid") for p in parts], axis=1)


if __name__ == "__main__":
    build(groups=sys.argv[1] if len(sys.argv) > 1 else "abcde")
