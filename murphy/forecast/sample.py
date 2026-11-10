"""Forecast rows for Step 2: one row per (leg, lead time).

Outcome classes, from arrival delay in minutes:
  early (< 0), 0-15, 15-30, ..., 165-180, 180+, cancelled, diverted.
A range [a, b) includes a and excludes b.

Sample: a fixed hash of flight_id picks about 2M legs from
2023-2025, and the same rate from 2026 for the test set. Hashing, not
random(), so the same legs come back on every run. Each leg gives three rows,
with as_of = scheduled departure minus 24 h, 6 h, 2 h. The three rows share
flight_id, so a split by flight_id keeps them together.

Legs with no recorded outcome (4 of 26,332,839) are left out.

Usage: python -m murphy.forecast.sample   -> data/forecast/rows.parquet
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
FLIGHTS = ROOT / "data" / "flights"
OUT = ROOT / "data" / "forecast" / "rows.parquet"

EDGES = list(range(0, 181, 15))  # 0, 15, ..., 180
CLASSES = (["early"] + [f"{a}-{a + 15}" for a in EDGES[:-1]]
           + ["180+", "cancelled", "diverted"])
LEADS_H = (24, 6, 2)
# 942 / 10,000 of the 21,239,759 legs in 2023-2025 is about 2.0M.
RATE_PER_10K = 942


def label_sql(col: str = "arr_delay_min") -> str:
    """SQL CASE giving the outcome class of a leg."""
    ranges = "\n".join(f"WHEN {col} < {a + 15} THEN '{a}-{a + 15}'" for a in EDGES[:-1])
    return f"""CASE
        WHEN cancelled THEN 'cancelled'
        WHEN diverted THEN 'diverted'
        WHEN {col} IS NULL THEN NULL
        WHEN {col} < 0 THEN 'early'
        {ranges}
        ELSE '180+' END"""


def build_rows(flights: Path | str = FLIGHTS, out: Path | str = OUT,
               rate_per_10k: int = RATE_PER_10K) -> int:
    """Write the forecast rows. Returns the row count."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    leads = ", ".join(f"({h})" for h in LEADS_H)
    sql = f"""
        WITH legs AS (
          SELECT *, {label_sql()} AS outcome
          FROM read_parquet($src)
          WHERE hash(flight_id) % 10000 < $rate
        )
        SELECT flight_id, source_file, flight_date, marketing_carrier,
               operating_carrier, origin, dest, sched_dep_utc, sched_arr_utc,
               distance_mi, l.lead_h,
               sched_dep_utc - to_hours(CAST(l.lead_h AS BIGINT)) AS as_of_utc,
               outcome, arr_delay_min,
               CASE WHEN year(flight_date) = 2026 THEN 'test'
                    WHEN year(flight_date) = 2025 THEN 'tune'
                    ELSE 'fit' END AS split
        FROM legs, (VALUES {leads}) l(lead_h)
        WHERE outcome IS NOT NULL AND arr_known_at IS NOT NULL
        ORDER BY as_of_utc, flight_id, lead_h
    """
    con = duckdb.connect()
    rel = con.sql(sql, params={"src": f"{flights}/*.parquet", "rate": rate_per_10k})
    rel.to_parquet(str(out))
    return con.sql(f"SELECT count(*) FROM read_parquet('{out}')").fetchall()[0][0]


if __name__ == "__main__":
    print(f"{build_rows():,} rows -> {OUT}")
