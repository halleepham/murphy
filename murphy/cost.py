"""The estimated cost of a new ticket. Shown only: it never changes the advice.

The estimate is the median fare people paid on a route and airline in one quarter. No flight has
its own fare, so every flight on that route and airline gets the same figure.
"""

import pandas as pd

MIN_TICKETS = 20  # an AI choice; fewer rows than this is too thin to show
SOURCE = "BTS DB1BMarket 2025 Q2"


def estimated_fare(fares: pd.DataFrame, origin: str, dest: str, carrier: str) -> dict:
    """fares: one row per origin, dest, carrier, with a blank carrier for the whole route.
    Falls back from the airline to the route, then to no estimate."""
    on_route = fares[(fares.origin == origin) & (fares.dest == dest) & (fares.tickets >= MIN_TICKETS)]
    airline, route = on_route[on_route.carrier == carrier], on_route[on_route.carrier.isna()]
    for basis, found in (("route and airline", airline), ("route, all airlines", route)):
        if len(found):
            row = found.iloc[0]
            return {"fare": float(row.fare), "basis": basis, "tickets": int(row.tickets), "source": SOURCE}
    return {"fare": None, "basis": "no estimate", "tickets": 0, "source": SOURCE}
