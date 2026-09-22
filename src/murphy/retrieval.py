"""R1 -- comparable-flight retrieval, and the empirical range computed over it.

This is ladder rung L1: the distribution over retrieved comparables is the
baseline any model has to beat, and it is also the evidence the traveler
inspects. Same object, two jobs.

Every number here comes from SQL over real rows. Nothing is modelled, nothing is
interpolated, and when the evidence is too thin the answer is a refusal rather
than a range.

Run:  .venv/bin/python src/murphy/retrieval.py --origin BOS --dest ATL \
          --carrier DL --month 11 --hour 6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "processed" / "flights" / "**" / "*.parquet"

# How many comparable flights we need before we are willing to answer.
COMFORTABLE = 20   # answer plainly
MINIMUM = 10       # answer, but say the evidence is limited
FLOOR = 5          # below this, after the whole ladder, refuse

# How far either side of the scheduled hour a "widened" match reaches.
WIDE_HOURS = 3


@dataclass
class Query:
    origin: str
    dest: str
    carrier: str
    month: int
    sched_dep_hour: int

    def __post_init__(self):
        self.origin = self.origin.strip().upper()
        self.dest = self.dest.strip().upper()
        self.carrier = self.carrier.strip().upper()


@dataclass
class Result:
    ok: bool
    query: dict
    rung: str                      # which step of the ladder answered
    rung_index: int
    match_description: str         # plain English, for the traveler
    n: int
    confidence: str                # "ok" | "limited" | "thin" | "refused"
    message: str
    p10: int | None = None
    p50: int | None = None
    p90: int | None = None
    evidence: list = field(default_factory=list)
    ladder_trace: list = field(default_factory=list)
    sql: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _ladder(q: Query) -> list[tuple[str, str, str]]:
    """The fallback ladder, in order. Each rung drops one more constraint.

    Order comes from the project plan: widen the hour bin, drop carrier, drop
    month, refuse.
    """
    bin_lo, bin_hi = (q.sched_dep_hour // 2) * 2, (q.sched_dep_hour // 2) * 2 + 1
    wide_lo, wide_hi = q.sched_dep_hour - WIDE_HOURS, q.sched_dep_hour + WIDE_HOURS
    route = f"origin = '{q.origin}' AND dest = '{q.dest}'"

    return [
        (
            "exact",
            f"{route} AND carrier = '{q.carrier}' AND month = {q.month} "
            f"AND sched_dep_hour BETWEEN {bin_lo} AND {bin_hi}",
            f"{q.carrier} {q.origin}→{q.dest}, same month, departing "
            f"{bin_lo:02d}:00–{bin_hi:02d}:59",
        ),
        (
            "wider departure window",
            f"{route} AND carrier = '{q.carrier}' AND month = {q.month} "
            f"AND sched_dep_hour BETWEEN {wide_lo} AND {wide_hi}",
            f"{q.carrier} {q.origin}→{q.dest}, same month, departing within "
            f"{WIDE_HOURS}h of {q.sched_dep_hour:02d}:00",
        ),
        (
            "any carrier",
            f"{route} AND month = {q.month} "
            f"AND sched_dep_hour BETWEEN {wide_lo} AND {wide_hi}",
            f"any carrier {q.origin}→{q.dest}, same month, departing within "
            f"{WIDE_HOURS}h of {q.sched_dep_hour:02d}:00",
        ),
        (
            "any carrier, any month",
            f"{route} AND sched_dep_hour BETWEEN {wide_lo} AND {wide_hi}",
            f"any carrier {q.origin}→{q.dest}, any month, departing within "
            f"{WIDE_HOURS}h of {q.sched_dep_hour:02d}:00",
        ),
    ]


def retrieve(q: Query, con: duckdb.DuckDBPyConnection | None = None,
             evidence_limit: int = 50) -> Result:
    """Walk the ladder until the evidence is thick enough, then compute the range."""
    con = con or duckdb.connect()
    trace, chosen = [], None

    for i, (name, where, description) in enumerate(_ladder(q)):
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{PARQUET}', hive_partitioning=true) "
            f"WHERE {where}"
        ).fetchone()[0]
        trace.append({"rung": name, "n": n})
        if chosen is None and n >= MINIMUM:
            chosen = (i, name, where, description, n)
            break

    # Nothing reached MINIMUM. Fall back to the widest rung and see if it clears
    # the floor; if not, refuse.
    if chosen is None:
        i, (name, where, description) = len(trace) - 1, _ladder(q)[len(trace) - 1]
        n = trace[-1]["n"]
        if n < FLOOR:
            return Result(
                ok=False, query=asdict(q), rung=name, rung_index=i,
                match_description=description, n=n, confidence="refused",
                message=(
                    f"Only {n} comparable flights after widening the match as far as it "
                    f"goes. That is not enough to quote a range, so Murphy will not "
                    f"guess one."
                ),
                ladder_trace=trace,
            )
        chosen = (i, name, where, description, n)

    i, name, where, description, n = chosen
    sql = f"""
SELECT quantile_cont(arr_delay_min, 0.10) AS p10,
       quantile_cont(arr_delay_min, 0.50) AS p50,
       quantile_cont(arr_delay_min, 0.90) AS p90
FROM read_parquet('{PARQUET}', hive_partitioning=true)
WHERE {where}
""".strip()
    p10, p50, p90 = con.execute(sql).fetchone()

    evidence = con.execute(f"""
        SELECT flight_id, flight_date, carrier, flight_number, origin, dest,
               sched_dep_local, sched_arr_local, arr_delay_min
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE {where}
        ORDER BY flight_date
        LIMIT {evidence_limit}
    """).fetchall()
    cols = ["flight_id", "flight_date", "carrier", "flight_number", "origin", "dest",
            "sched_dep_local", "sched_arr_local", "arr_delay_min"]

    if n >= COMFORTABLE:
        confidence = "ok"
        message = f"Based on {n} comparable flights."
    elif n >= MINIMUM:
        confidence = "limited"
        message = (f"Based on {n} comparable flights. That is a small sample, so treat "
                   f"the range as indicative.")
    else:
        confidence = "thin"
        message = (f"Only {n} comparable flights even after widening the match. The "
                   f"range is shown, but it rests on very little evidence.")

    if i > 0:
        message += f" The exact match was too thin, so it was widened to: {description}."

    return Result(
        ok=True, query=asdict(q), rung=name, rung_index=i,
        match_description=description, n=n, confidence=confidence, message=message,
        p10=round(p10), p50=round(p50), p90=round(p90),
        evidence=[dict(zip(cols, row)) for row in evidence],
        ladder_trace=trace, sql=sql,
    )


def main():
    ap = argparse.ArgumentParser(description="Retrieve comparable flights and a delay range.")
    ap.add_argument("--origin", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--carrier", required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--hour", type=int, required=True, help="scheduled departure hour, 0-23")
    ap.add_argument("--json", action="store_true", help="print the full payload")
    a = ap.parse_args()

    r = retrieve(Query(a.origin, a.dest, a.carrier, a.month, a.hour))

    if a.json:
        print(r.to_json())
        return

    q = r.query
    print(f"\n{q['carrier']} {q['origin']}→{q['dest']}, month {q['month']}, "
          f"departing {q['sched_dep_hour']:02d}:00")
    print("-" * 68)

    if not r.ok:
        print(f"REFUSED. {r.message}")
        print(f"\nLadder: " + "  ".join(f"{t['rung']}={t['n']}" for t in r.ladder_trace))
        return

    print(f"Arrival delay    p10 {r.p10:+d} min   p50 {r.p50:+d} min   p90 {r.p90:+d} min")
    print(f"Evidence         {r.n} flights  ({r.confidence})")
    print(f"Matched on       {r.match_description}")
    print(f"\n{r.message}")
    print(f"\nLadder: " + "  ".join(f"{t['rung']}={t['n']}" for t in r.ladder_trace))
    print(f"\nFirst {min(5, len(r.evidence))} of {r.n} comparable flights:")
    for e in r.evidence[:5]:
        print(f"  {e['flight_id']:<28} {e['sched_dep_local']}→{e['sched_arr_local']}  "
              f"{e['arr_delay_min']:+4d} min")


if __name__ == "__main__":
    main()
