"""Sample travellers and itineraries — the Session provenance tier.

These are the only hand-authored nodes in the graph. Everything else is extracted
from BTS rows; these are invented people, so nothing is ever computed from them.
They exist because the thesis is that risk is personal *because the threshold is
personal*, and the schema has to show where that constraint enters.

Their legs are real flights from the seed day, chosen so the demo queries have
something worth finding. Each one is a different shape of connection risk:

  * **cascade** — the headline. A connection scheduled with 115 minutes of slack that
    came down to 13, because the inbound aircraft had been running late since its
    third leg within the loaded day. BTS attributes that leg's delay to "carrier",
    which names a category -- causes inside the airline's control -- and never an
    incident, so nothing here claims a particular cause.
  * **correlated** — both legs ran ~120 minutes late *together*, so the connection
    survived. Correlated delay is not always bad news, and a system that scores legs
    independently cannot tell the two cases apart.
  * **nonstop** — a control with no connection at all.
  * **no-escape** — the correlated-failure case. A 45-minute connection at Atlanta
    onto Newark, where every one of the nine alternatives that day was flown by an
    aircraft already running hours late. Switching flights was not an escape, and a
    system that scores legs independently cannot see that.
"""

from datetime import datetime, timezone, timedelta

CDT = timezone(timedelta(hours=-5))
EDT = timezone(timedelta(hours=-4))

TRAVELERS = [
    {
        "traveler_id": "sample-cascade",
        "arrive_by_utc": datetime(2024, 7, 20, 0, 30, tzinfo=EDT),
        "ranking_preference": "reliability",
        "note": "Connection looked safe on paper; the inbound aircraft had other ideas.",
    },
    {
        "traveler_id": "sample-correlated",
        "arrive_by_utc": datetime(2024, 7, 20, 1, 0, tzinfo=CDT),
        "ranking_preference": "reliability",
        "note": "Both legs late together, so the connection held.",
    },
    {
        "traveler_id": "sample-no-escape",
        "arrive_by_utc": datetime(2024, 7, 19, 22, 0, tzinfo=EDT),
        "ranking_preference": "reliability",
        "note": "Nine ways to Newark, all broken by the same cause.",
    },
    {
        "traveler_id": "sample-nonstop",
        "arrive_by_utc": datetime(2024, 7, 19, 23, 0, tzinfo=CDT),
        "ranking_preference": "fastest",
        "note": "Control: no connection to miss.",
    },
]

ITINERARIES = [
    {
        "itinerary_id": "itin-cascade",
        "traveler_id": "sample-cascade",
        "legs": ["2024-07-19_WN2679_HOU_MDW", "2024-07-19_WN2594_MDW_BOS"],
    },
    {
        "itinerary_id": "itin-correlated",
        "traveler_id": "sample-correlated",
        "legs": ["2024-07-19_DL1051_BNA_MSP", "2024-07-19_DL2683_MSP_IAH"],
    },
    {
        "itinerary_id": "itin-no-escape",
        "traveler_id": "sample-no-escape",
        "legs": ["2024-07-19_DL1246_MKE_ATL", "2024-07-19_DL2498_ATL_EWR"],
    },
    {
        "itinerary_id": "itin-nonstop",
        "traveler_id": "sample-nonstop",
        "legs": ["2024-07-19_WN4458_HOU_MDW"],
    },
]


def traveler_rows() -> list[dict]:
    # `name` is a display caption -- see the note in kg_build.flights().
    return [dict(t, name=t["traveler_id"].replace("sample-", "")) for t in TRAVELERS]


def itinerary_rows() -> list[dict]:
    return [{"itinerary_id": i["itinerary_id"],
             "name": i["itinerary_id"].replace("itin-", ""),
             "n_stops": len(i["legs"]) - 1}
            for i in ITINERARIES]


def booked_rows() -> list[dict]:
    return [{"traveler_id": i["traveler_id"], "itinerary_id": i["itinerary_id"]}
            for i in ITINERARIES]


def contains_rows() -> list[dict]:
    return [
        {"itinerary_id": i["itinerary_id"], "flight_id": flight_id, "leg_order": n}
        for i in ITINERARIES
        for n, flight_id in enumerate(i["legs"], start=1)
    ]
