"""Does risk-aware route ranking beat ranking by scheduled time?

The comparison the instructor feedback asks for, run on real 2024 outcomes.

  Baseline  -- rank itineraries by scheduled duration. What a booking site does.
  Murphy    -- rank by what comparable flights historically did door to door,
               penalised by the share of inbound flights that missed the
               transfer.

Both planners choose an itinerary for a real origin, destination and date. We
then look up what actually happened to the itinerary each one chose.

Leakage guard: the risk-aware ranking may only use flights from the TRAINING
months. Test-month outcomes are used for scoring and never for ranking. The
split is temporal, never random.

Run:  .venv/bin/python src/murphy/evaluate.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murphy.graph import (
    MAX_LAYOVER_MIN, MIN_CONNECTION_MIN, MISSED_CONNECTION_PENALTY_MIN,
)  # noqa: F401
from murphy.retrieval import PARQUET

TRAIN_MONTHS = [1, 2, 4, 5, 6, 7, 8, 9]      # March is absent from the source data
TEST_MONTH = 10

# Chosen so the routing decision is real. The first six are long-haul pairs with
# roughly one nonstop a day and dozens of connecting options, so a traveller who
# cannot take that one nonstop has to choose a connection. The last four are
# well-served pairs, included to check the method does not connect needlessly.
#
# An earlier pair set was all trunk routes. Both methods simply picked the
# nonstop every time, disagreeing about nothing structural, and the comparison
# measured nothing. That was a flaw in the experiment, not a finding.
PAIRS = [
    ("LAS", "DCA"), ("IAH", "JFK"), ("FLL", "PHX"),
    ("DAL", "SEA"), ("SLC", "DCA"), ("DCA", "AUS"),
    ("SFO", "BOS"), ("ORD", "LAX"), ("BOS", "MCI"), ("SEA", "ATL"),
]

# A traveller wants to leave at roughly a particular time. Comparing a 06:00
# departure against a 15:00 one is not comparing substitutes, so each decision
# is made among itineraries leaving near one of these target hours.
TARGET_HOURS = [7, 12, 17]
DEPARTURE_SLACK_HOURS = 3

MIN_HISTORY = 10          # a service needs this much history to be rankable


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _hubs(con, origin, dest, limit=8):
    return [r[0] for r in con.execute(f"""
        WITH out AS (SELECT dest AS hub, count(*) n
                     FROM read_parquet('{PARQUET}', hive_partitioning=true)
                     WHERE origin = ? AND dest <> ? GROUP BY 1),
             inn AS (SELECT origin AS hub, count(*) n
                     FROM read_parquet('{PARQUET}', hive_partitioning=true)
                     WHERE dest = ? AND origin <> ? GROUP BY 1)
        SELECT hub FROM out JOIN inn USING (hub)
        ORDER BY least(out.n, inn.n) DESC LIMIT {limit}
    """, [origin, dest, dest, origin]).fetchall()]


def _history(con, routes: list[tuple[str, str]]) -> dict:
    """Per service, what comparable flights did during the TRAINING months."""
    pairs = ", ".join(f"('{o}','{d}')" for o, d in routes)
    months = ", ".join(str(m) for m in TRAIN_MONTHS)
    rows = con.execute(f"""
        SELECT origin, dest, carrier, sched_dep_hour, count(*) AS n,
               quantile_cont(arr_delay_min, 0.50) AS p50,
               quantile_cont(arr_delay_min, 0.90) AS p90
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE (origin, dest) IN ({pairs}) AND month IN ({months})
        GROUP BY 1, 2, 3, 4
        HAVING count(*) >= {MIN_HISTORY}
    """).fetchall()
    return {(o, d, c, h): (n, p50, p90) for o, d, c, h, n, p50, p90 in rows}


def _miss_rate(con, routes: list[tuple[str, str]]) -> dict:
    """For each service and each plausible slack, the share that arrived too late.

    Counted from real training-month flights, not modelled.
    """
    pairs = ", ".join(f"('{o}','{d}')" for o, d in routes)
    months = ", ".join(str(m) for m in TRAIN_MONTHS)
    rows = con.execute(f"""
        SELECT origin, dest, carrier, sched_dep_hour,
               count(*) AS n, list(arr_delay_min) AS delays
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE (origin, dest) IN ({pairs}) AND month IN ({months})
        GROUP BY 1, 2, 3, 4
        HAVING count(*) >= {MIN_HISTORY}
    """).fetchall()
    return {(o, d, c, h): delays for o, d, c, h, n, delays in rows}


def _flights(con, routes: list[tuple[str, str]], month: int) -> dict:
    """Actual flights in the test month, keyed by date and route."""
    pairs = ", ".join(f"('{o}','{d}')" for o, d in routes)
    rows = con.execute(f"""
        SELECT flight_date, origin, dest, carrier, flight_number,
               sched_dep_minutes, sched_arr_local, sched_duration_min,
               sched_dep_hour, arr_delay_min, dep_delay_min
        FROM read_parquet('{PARQUET}', hive_partitioning=true)
        WHERE (origin, dest) IN ({pairs}) AND month = {month}
    """).fetchall()
    out = defaultdict(list)
    for r in rows:
        out[(r[0], r[1], r[2])].append({
            "carrier": r[3], "number": r[4], "dep_min": r[5],
            "arr_min": _minutes(r[6]), "dur": r[7], "hour": r[8],
            "arr_delay": r[9], "dep_delay": r[10],
        })
    return out


def _itineraries(day, origin, dest, hubs, flights):
    """Every itinerary a traveller could actually have flown that day."""
    out = []
    for f in flights.get((day, origin, dest), []):
        out.append({"legs": [f], "hub": None, "layover": 0,
                    "sched_total": f["dur"]})
    for hub in hubs:
        firsts = flights.get((day, origin, hub), [])
        seconds = flights.get((day, hub, dest), [])
        for a in firsts:
            for b in seconds:
                gap = b["dep_min"] - a["arr_min"]
                if gap < 0:
                    gap += 1440
                if MIN_CONNECTION_MIN <= gap <= MAX_LAYOVER_MIN:
                    out.append({"legs": [a, b], "hub": hub, "layover": gap,
                                "sched_total": a["dur"] + gap + b["dur"]})
    return out


def _rankable(itin, origin, dest, history):
    """An itinerary Murphy can score: every leg needs training history."""
    keys, prev = [], origin
    last = len(itin["legs"]) - 1
    for i, leg in enumerate(itin["legs"]):
        to = dest if i == last else itin["hub"]
        key = (prev, to, leg["carrier"], leg["hour"])
        if key not in history:
            return None
        keys.append(key)
        prev = to
    return keys


def _murphy_score(itin, keys, history, delays):
    """Expected worst-case door-to-door time, penalised by connection risk."""
    _, _, p90_final = history[keys[-1]]
    tail_total = itin["sched_total"] + p90_final

    risk = 0.0
    if itin["hub"] is not None:
        slack = itin["layover"] - MIN_CONNECTION_MIN
        sample = delays.get(keys[0], [])
        if sample:
            risk = sum(1 for d in sample if d > slack) / len(sample)
    return tail_total + risk * MISSED_CONNECTION_PENALTY_MIN, risk


def _realized(itin):
    """What actually happened. None if the connection did not hold."""
    legs = itin["legs"]
    if len(legs) == 2:
        actual_arr = legs[0]["arr_min"] + legs[0]["arr_delay"]
        actual_dep = legs[1]["dep_min"] + legs[1]["dep_delay"]
        gap = actual_dep - actual_arr
        if gap < 0:
            gap += 1440
        if gap < MIN_CONNECTION_MIN:
            return None                      # missed the transfer
    return itin["sched_total"] + legs[-1]["arr_delay"]


def run(month: int = TEST_MONTH, verbose: bool = True) -> dict:
    con = duckdb.connect()
    results = {"baseline": [], "murphy": [], "agree": 0, "decisions": 0,
               "baseline_missed": 0, "murphy_missed": 0, "disagreements": [],
               "baseline_penalised": [], "murphy_penalised": []}

    for origin, dest in PAIRS:
        hubs = _hubs(con, origin, dest)
        routes = [(origin, dest)] + [(origin, h) for h in hubs] + [(h, dest) for h in hubs]
        history = _history(con, routes)
        delays = _miss_rate(con, routes)
        flights = _flights(con, routes, month)

        days = sorted({d for (d, o, x) in flights.keys()})
        for day in days:
            options = _itineraries(day, origin, dest, hubs, flights)
            rankable = []
            for itin in options:
                keys = _rankable(itin, origin, dest, history)
                if keys is None:
                    continue
                score, risk = _murphy_score(itin, keys, history, delays)
                rankable.append((itin, score, risk))

            for target in TARGET_HOURS:
                lo = (target - DEPARTURE_SLACK_HOURS) * 60
                hi = (target + DEPARTURE_SLACK_HOURS) * 60
                scored = [r for r in rankable if lo <= r[0]["legs"][0]["dep_min"] <= hi]
                if len(scored) < 2:
                    continue                  # no real choice at this time of day

                baseline = min(scored, key=lambda s: s[0]["sched_total"])[0]
                murphy = min(scored, key=lambda s: s[1])[0]

                results["decisions"] += 1
                agreed = baseline is murphy
                if agreed:
                    results["agree"] += 1

                got = {}
                for name, choice in (("baseline", baseline), ("murphy", murphy)):
                    got[name] = _realized(choice)
                    if got[name] is None:
                        results[f"{name}_missed"] += 1
                        # A missed connection must not simply vanish from the
                        # time statistics -- dropping it would delete a method's
                        # own failures and flatter it. Charge a stated penalty.
                        results[f"{name}_penalised"].append(
                            choice["sched_total"] + MISSED_CONNECTION_PENALTY_MIN)
                    else:
                        results[name].append(got[name])
                        results[f"{name}_penalised"].append(got[name])

                if not agreed:
                    results["disagreements"].append({
                        "pair": f"{origin}-{dest}", "day": day, "target": target,
                        "baseline": got["baseline"], "murphy": got["murphy"],
                        "baseline_stops": len(baseline["legs"]) - 1,
                        "murphy_stops": len(murphy["legs"]) - 1,
                    })

        if verbose:
            print(f"  {origin}→{dest}: {len(days)} days")

    return results


def report(results: dict) -> None:
    n = results["decisions"]
    print(f"\n{'=' * 66}\nRESULTS — {n} planning decisions, "
          f"{len(PAIRS)} routes, month {TEST_MONTH} 2024\n{'=' * 66}")
    print(f"\nThe two methods chose the same itinerary {results['agree']} times "
          f"({results['agree'] / n:.0%}).")
    print(f"They differed on {n - results['agree']} decisions — those are where "
          f"the comparison lives.\n")

    def p90(v):
        return sorted(v)[int(len(v) * 0.9)]

    print("Completed trips only — missed connections excluded\n")
    print(f"{'':<12}{'median':>10}{'mean':>10}{'p90':>10}{'missed':>14}")
    print("-" * 56)
    for name, label in (("baseline", "Scheduled"), ("murphy", "Murphy")):
        vals, missed = results[name], results[f"{name}_missed"]
        print(f"{label:<12}{statistics.median(vals):>9.0f}m{statistics.mean(vals):>9.0f}m"
              f"{p90(vals):>9.0f}m{missed:>9} ({missed / n:.1%})")

    print(f"\nWith missed connections charged {MISSED_CONNECTION_PENALTY_MIN} minutes")
    print("(a stated assumption, not a measurement — we have no data on what")
    print("rebooking actually costs)\n")
    print(f"{'':<12}{'median':>10}{'mean':>10}{'p90':>10}")
    print("-" * 42)
    for name, label in (("baseline", "Scheduled"), ("murphy", "Murphy")):
        vals = results[f"{name}_penalised"]
        print(f"{label:<12}{statistics.median(vals):>9.0f}m{statistics.mean(vals):>9.0f}m"
              f"{p90(vals):>9.0f}m")

    print("\nDoor-to-door minutes, from scheduled departure to actual arrival.")

    # On most days both methods pick the same nonstop, so the marginal
    # distributions above are dominated by decisions where no choice was made.
    # The paired view compares the two picks on the same day, and only on the
    # days they actually differed.
    dis = results["disagreements"]
    both = [d for d in dis if d["baseline"] is not None and d["murphy"] is not None]
    print(f"\n{'=' * 66}\nPAIRED — same route, same day, only where they disagreed"
          f"\n{'=' * 66}\n")

    if not both:
        print("No comparable disagreements.")
        return

    diffs = [d["murphy"] - d["baseline"] for d in both]
    wins = sum(1 for x in diffs if x < 0)
    losses = sum(1 for x in diffs if x > 0)
    ties = sum(1 for x in diffs if x == 0)

    print(f"{len(both)} paired decisions where the two methods chose differently.\n")
    print(f"  Murphy arrived earlier   {wins:>4}  ({wins / len(both):.0%})")
    print(f"  Murphy arrived later     {losses:>4}  ({losses / len(both):.0%})")
    print(f"  Same arrival             {ties:>4}  ({ties / len(both):.0%})")
    print(f"\n  Mean difference        {statistics.mean(diffs):>+6.1f} min "
          f"(negative favours Murphy)")
    print(f"  Median difference      {statistics.median(diffs):>+6.1f} min")
    worst = sorted(diffs)
    print(f"  Worst case for Murphy  {worst[-1]:>+6.0f} min")
    print(f"  Best case for Murphy   {worst[0]:>+6.0f} min")

    only_b = sum(1 for d in dis if d["baseline"] is None and d["murphy"] is not None)
    only_m = sum(1 for d in dis if d["murphy"] is None and d["baseline"] is not None)
    print(f"\n  Connections missed by the scheduled-time pick but not Murphy: {only_b}")
    print(f"  Connections missed by Murphy but not the scheduled-time pick: {only_m}")

    conn = [d for d in dis if d["baseline_stops"] != d["murphy_stops"]]
    print(f"\n  Decisions where the two disagreed about whether to connect at all: "
          f"{len(conn)} of {len(dis)}")


def main():
    ap = argparse.ArgumentParser(description="Compare risk-aware ranking with scheduled-time ranking.")
    ap.add_argument("--month", type=int, default=TEST_MONTH)
    a = ap.parse_args()
    print(f"Training on months {TRAIN_MONTHS}, testing on month {a.month}.\n")
    report(run(a.month))


if __name__ == "__main__":
    main()
