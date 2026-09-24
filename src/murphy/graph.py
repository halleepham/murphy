"""Top-k route planning over the flight network.

The traveller already has a booking. This answers the next question: given where
they are going, what are the alternatives, and how does their own route compare?

Airports are nodes. A "service" is a route flown by one carrier in a given
departure hour, aggregated across the month -- that is the granularity the
retrieval layer already matches on, so routes and evidence line up exactly.

Nothing here invents a number. Delay ranges come from `retrieval.retrieve()`
unchanged, and connection risk is the observed share of comparable flights whose
arrival delay would have eaten the transfer.

Two things the data cannot support, stated rather than faked:
  * no fare, so there is no cost objective
  * no cancellations, so every route is conditional on its flights operating

Run:  .venv/bin/python src/murphy/graph.py --origin BOS --dest MCI --month 11 --hour 6
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# Allow running this file directly as well as importing it as part of the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murphy.retrieval import MINIMUM, PARQUET, Query, retrieve

# One global minimum connection time, stated as an assumption. Real minimums
# vary by airport and terminal; we have no data for that, and a tiered table
# would be invented numbers dressed as evidence.
MIN_CONNECTION_MIN = 45

# A layover longer than this is not a connection anyone would book.
MAX_LAYOVER_MIN = 360

# A service needs this many flights in the month before it counts as real.
MIN_SERVICE_FLIGHTS = 8

# How far either side of the requested hour a departure may sit.
DEPARTURE_WINDOW_HOURS = 4


@dataclass
class Leg:
    origin: str
    dest: str
    carrier: str
    dep_hour: int
    dep_local: str
    arr_local: str
    duration_min: int
    n_flights: int
    p10: int | None = None
    p50: int | None = None
    p90: int | None = None
    evidence_n: int | None = None
    confidence: str | None = None
    match_quality: str | None = None


@dataclass
class Route:
    legs: list[Leg]
    hub: str | None                 # None for a direct flight
    layover_min: int | None
    scheduled_total_min: int
    typical_arrival_delay: int      # p50 of the final leg, against its own schedule
    tail_arrival_delay: int         # p90 of the final leg, against its own schedule
    connection_risk: float | None   # share of comparable inbound flights that miss it
    connection_sample: int | None   # how many flights that share was drawn from
    labels: list[str] = field(default_factory=list)
    is_travellers_route: bool = False

    # Delay is measured against each itinerary's own schedule, so it cannot be
    # compared across routes: a two-stop trip "arriving 20 minutes early" can
    # still land two hours after a nonstop. Door-to-door time is the comparable
    # quantity, so ranking uses these instead.
    @property
    def typical_total_min(self) -> int:
        return self.scheduled_total_min + self.typical_arrival_delay

    @property
    def tail_total_min(self) -> int:
        return self.scheduled_total_min + self.tail_arrival_delay

    @property
    def stops(self) -> int:
        return len(self.legs) - 1

    @property
    def carriers(self) -> str:
        return "/".join(sorted({leg.carrier for leg in self.legs}))

    def describe(self) -> str:
        path = self.legs[0].origin + "".join(f"→{leg.dest}" for leg in self.legs)
        return f"{self.carriers} {path}"


def _window(hour: int) -> tuple[int, int]:
    return hour - DEPARTURE_WINDOW_HOURS, hour + DEPARTURE_WINDOW_HOURS


def _services(con, where: str) -> list[Leg]:
    """Aggregate flights into services: one row per route + carrier + hour."""
    rows = con.execute(f"""
        SELECT origin, dest, carrier, sched_dep_hour,
               count(*) AS n,
               CAST(median(sched_dep_minutes) AS INT) AS dep_min,
               CAST(median(CAST(split_part(sched_arr_local, ':', 1) AS INT) * 60
                         + CAST(split_part(sched_arr_local, ':', 2) AS INT)) AS INT) AS arr_min,
               CAST(median(sched_duration_min) AS INT) AS dur
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE {where}
        GROUP BY 1, 2, 3, 4
        HAVING count(*) >= {MIN_SERVICE_FLIGHTS}
    """).fetchall()

    legs = []
    for origin, dest, carrier, hour, n, dep_min, arr_min, dur in rows:
        legs.append(Leg(
            origin=origin, dest=dest, carrier=carrier, dep_hour=hour,
            dep_local=f"{dep_min // 60:02d}:{dep_min % 60:02d}",
            arr_local=f"{arr_min // 60:02d}:{arr_min % 60:02d}",
            duration_min=dur, n_flights=n,
        ))
        legs[-1]._dep_min, legs[-1]._arr_min = dep_min, arr_min
    return legs


def _candidate_hubs(con, origin: str, dest: str, month: int, limit: int = 12) -> list[str]:
    """Airports reachable from the origin that also reach the destination."""
    return [r[0] for r in con.execute(f"""
        WITH out AS (
            SELECT dest AS hub, count(*) AS n
            FROM read_parquet('{PARQUET}', hive_partitioning=true)
            WHERE origin = '{origin}' AND month = {month} AND dest <> '{dest}'
            GROUP BY 1
        ), inn AS (
            SELECT origin AS hub, count(*) AS n
            FROM read_parquet('{PARQUET}', hive_partitioning=true)
            WHERE dest = '{dest}' AND month = {month} AND origin <> '{origin}'
            GROUP BY 1
        )
        SELECT out.hub
        FROM out JOIN inn USING (hub)
        ORDER BY least(out.n, inn.n) DESC
        LIMIT {limit}
    """).fetchall()]


def _connection_risk(con, leg: Leg, month: int, slack_min: int) -> tuple[float, int] | None:
    """Share of this service's real flights that arrived too late to connect.

    Grounded, not modelled: it counts rows whose observed arrival delay exceeded
    the slack the traveller actually has.
    """
    lo, hi = leg.dep_hour - 1, leg.dep_hour + 1
    row = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE arr_delay_min > {slack_min})
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE origin = '{leg.origin}' AND dest = '{leg.dest}'
          AND carrier = '{leg.carrier}' AND month = {month}
          AND sched_dep_hour BETWEEN {lo} AND {hi}
    """).fetchone()
    if not row[0]:
        return None
    return round(row[1] / row[0], 3), row[0]


def plan_routes(origin: str, dest: str, month: int, hour: int,
                k: int = 5, con: duckdb.DuckDBPyConnection | None = None,
                travellers_route: tuple[str, str | None] | None = None) -> list[Route]:
    """Enumerate direct and one-stop itineraries, score them, return the best k.

    `travellers_route` identifies the itinerary they already booked, as
    (carrier, hub) -- hub is None for a nonstop. Matching on carrier as well as
    hub matters: without it, a booked nonstop would mark every nonstop
    alternative as "yours".
    """
    origin, dest = origin.strip().upper(), dest.strip().upper()
    con = con or duckdb.connect()
    lo, hi = _window(hour)

    direct = _services(con, f"origin = '{origin}' AND dest = '{dest}' AND month = {month} "
                            f"AND sched_dep_hour BETWEEN {lo} AND {hi}")

    hubs = _candidate_hubs(con, origin, dest, month)
    first_legs, second_legs = [], []
    if hubs:
        hub_list = ", ".join(f"'{h}'" for h in hubs)
        first_legs = _services(con, f"origin = '{origin}' AND dest IN ({hub_list}) "
                                    f"AND month = {month} "
                                    f"AND sched_dep_hour BETWEEN {lo} AND {hi}")
        second_legs = _services(con, f"origin IN ({hub_list}) AND dest = '{dest}' "
                                     f"AND month = {month}")

    # --- assemble candidate itineraries
    candidates: list[tuple[list[Leg], str | None, int | None]] = [
        ([leg], None, None) for leg in direct
    ]
    by_hub: dict[str, list[Leg]] = {}
    for leg in second_legs:
        by_hub.setdefault(leg.origin, []).append(leg)

    for first in first_legs:
        for second in by_hub.get(first.dest, []):
            gap = second._dep_min - first._arr_min
            if gap < 0:
                gap += 1440                      # departs the following day
            if gap < MIN_CONNECTION_MIN or gap > MAX_LAYOVER_MIN:
                continue
            candidates.append(([first, second], first.dest, gap))

    if not candidates:
        return []

    # Rank roughly first so only plausible itineraries get the expensive scoring.
    candidates.sort(key=lambda c: sum(l.duration_min for l in c[0]) + (c[2] or 0))
    candidates = candidates[:16]

    routes: list[Route] = []
    for legs, hub, layover in candidates:
        scored, usable = [], True
        for leg in legs:
            result = retrieve(Query(leg.origin, leg.dest, leg.carrier, month, leg.dep_hour),
                              con=con, evidence_limit=0)
            if not result.ok or result.n < MINIMUM:
                usable = False
                break
            leg.p10, leg.p50, leg.p90 = result.p10, result.p50, result.p90
            leg.evidence_n = result.n
            leg.confidence = result.confidence
            leg.match_quality = result.match_quality
            scored.append(leg)
        if not usable:
            continue

        risk = None
        if hub is not None:
            risk = _connection_risk(con, scored[0], month, layover - MIN_CONNECTION_MIN)

        mine = False
        if travellers_route is not None:
            want_carrier, want_hub = travellers_route
            mine = (hub == want_hub
                    and want_carrier.strip().upper() in {l.carrier for l in scored})

        routes.append(Route(
            legs=scored, hub=hub, layover_min=layover,
            scheduled_total_min=sum(l.duration_min for l in scored) + (layover or 0),
            typical_arrival_delay=scored[-1].p50,
            tail_arrival_delay=scored[-1].p90,
            connection_risk=risk[0] if risk else None,
            connection_sample=risk[1] if risk else None,
            is_travellers_route=mine,
        ))

    if not routes:
        return []

    # --- diversity. The feedback is explicit that top-k must not be k near
    # duplicates, so keep only the best itinerary for each (hub, carriers)
    # combination: two United nonstops an hour apart are one option, not two.
    seen, diverse = set(), []
    for route in sorted(routes, key=_reliability_key):
        key = (route.hub, route.carriers)
        if key in seen:
            continue
        seen.add(key)
        diverse.append(route)

    _label(diverse)
    return sorted(diverse, key=_reliability_key)[:k]


# Minutes of trip time a traveller should be willing to trade to avoid a 100%
# chance of missing a connection. Deliberately blunt and deliberately visible:
# a missed connection costs far more than the delay itself, and we have no data
# on what it actually costs, so this is a stated preference, not a measurement.
MISSED_CONNECTION_PENALTY_MIN = 240


def _reliability_key(route: Route):
    """Lower is better: worst-case door-to-door time, penalised by connection risk."""
    return route.tail_total_min + (route.connection_risk or 0.0) * MISSED_CONNECTION_PENALTY_MIN


def _fastest_key(route: Route):
    return route.typical_total_min


def _label(routes: list[Route]) -> None:
    """Tag the standout routes so the traveller sees why each one is listed."""
    if not routes:
        return
    min(routes, key=_fastest_key).labels.append("fastest")
    min(routes, key=_reliability_key).labels.append("most reliable")
    fewest = min(routes, key=lambda r: (r.stops, _reliability_key(r)))
    if fewest.stops == 0:
        fewest.labels.append("nonstop")


def main():
    ap = argparse.ArgumentParser(description="Plan top-k routes between two airports.")
    ap.add_argument("--origin", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--hour", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--booked-carrier", help="the carrier the traveller already booked")
    ap.add_argument("--via", help="the hub they booked through; omit for a nonstop")
    a = ap.parse_args()

    booked = (a.booked_carrier, a.via) if a.booked_carrier else None
    routes = plan_routes(a.origin, a.dest, a.month, a.hour, k=a.k, travellers_route=booked)

    print(f"\n{a.origin} → {a.dest}, month {a.month}, departing near {a.hour:02d}:00")
    print("-" * 78)
    if not routes:
        print("No route found with enough evidence to score. Murphy will not guess one.")
        return

    for i, r in enumerate(routes, 1):
        tags = f"  [{', '.join(r.labels)}]" if r.labels else ""
        mine = "  ← your booked route" if r.is_travellers_route else ""
        print(f"\n{i}. {r.describe()}{tags}{mine}")
        print(f"   {r.stops} stop{'s' if r.stops != 1 else ''}"
              + (f" via {r.hub}, {r.layover_min} min layover" if r.hub else "")
              + f"   scheduled {r.scheduled_total_min // 60}h{r.scheduled_total_min % 60:02d}m")
        print(f"   door to door   typically {r.typical_total_min // 60}h"
              f"{r.typical_total_min % 60:02d}m, "
              f"1 in 10 worse than {r.tail_total_min // 60}h{r.tail_total_min % 60:02d}m")
        if r.connection_risk is not None:
            print(f"   connection     {r.connection_risk:.0%} of {r.connection_sample} "
                  f"comparable inbound flights arrived too late to make it")
        for leg in r.legs:
            print(f"     {leg.carrier} {leg.origin}→{leg.dest} {leg.dep_local}–{leg.arr_local}"
                  f"   p50 {leg.p50:+d}  p90 {leg.p90:+d}   "
                  f"{leg.evidence_n} flights ({leg.confidence})")


if __name__ == "__main__":
    main()
