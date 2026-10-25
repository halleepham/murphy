"""U1 acceptance: extraction produces the shape the spec predicts, without Neo4j.

The load is the expensive, slow, networked half. Everything checkable before that
point is checked here, so a wrong count surfaces in a second rather than after
117,000 writes.
"""

import pytest

from murphy import kg_build as kb

# Counts for the seed day, 2024-07-19, nine majors. Kept in step with
# docs/challenge-3/spec_challenge3.md section 2 -- if one changes, so does the other.
EXPECTED = {
    "Flight": 15_224,
    "Service": 15_158,
    "Aircraft": 3_596,
    "Airport": 202,
    "Airline": 9,
    "Weather": 2_843,
    "PRECEDES": 8_982,
}


@pytest.fixture(scope="module")
def batches():
    return kb.summary(kb.connect())


def test_counts_match_the_spec(batches):
    actual = {name: len(batches[name]) for name in EXPECTED}
    assert actual == EXPECTED


def test_extraction_is_deterministic():
    """Two extractions must agree exactly.

    Three aircraft on the seed day have two flights stamped at the same instant, so
    ordering by departure time alone leaves lead() free to choose either. That would
    make the graph unreproducible in a way nothing else would catch.
    """
    con = kb.connect()
    first = {(r["from_id"], r["to_id"]) for r in kb.precedes(con)}
    second = {(r["from_id"], r["to_id"]) for r in kb.precedes(con)}
    assert first == second
    assert len(first) == EXPECTED["PRECEDES"]


def test_node_keys_are_unique(batches):
    keys = {
        "Flight": "flight_id", "Service": "service_id", "Aircraft": "tail_number",
        "Airport": "iata", "Airline": "carrier", "Weather": "weather_id",
    }
    for node, key in keys.items():
        values = [r[key] for r in batches[node]]
        assert len(values) == len(set(values)), f"{node}.{key} is not unique"
        assert all(v is not None for v in values), f"{node}.{key} has nulls"


def test_only_major_carriers(batches):
    assert {r["carrier"] for r in batches["Airline"]} == set(kb.MAJORS)
    assert {r["carrier"] for r in batches["Flight"]} <= set(kb.MAJORS)


def test_outcome_columns_are_prefixed(batches):
    """Outcomes are labels, never inputs. The prefix is how that rule stays visible."""
    flight = batches["Flight"][0]
    for field in ("arr_delay_min", "dep_delay_min", "cancelled", "diverted",
                  "delay_carrier_min", "delay_late_aircraft_min"):
        assert f"outcome_{field}" in flight
        assert field not in flight, f"{field} must only appear prefixed"


def test_causes_reported_tracks_the_bts_convention(batches):
    """A null cause means "delay under 15 minutes", not "no cause".

    BTS populates cause columns only at 15 minutes or more, so the flag has to agree
    with the delay rather than merely exist.
    """
    operated = [f for f in batches["Flight"]
                if not f["outcome_cancelled"] and f["outcome_arr_delay_min"] is not None]
    for f in operated:
        late = f["outcome_arr_delay_min"] >= 15
        assert f["causes_reported"] == late, f["flight_id"]


def test_services_are_year_aggregates_not_day_slices(batches):
    """A Service averages many flights because it spans the year.

    At day grain Service was 1:1 with Flight and carried no information -- the mistake
    the Phase 0 audit caught. This guards the fix.
    """
    counts = [s["n_flights"] for s in batches["Service"] if s["n_flights"]]
    assert sum(counts) / len(counts) > 50


def test_service_quantiles_are_ordered(batches):
    for s in batches["Service"]:
        if s["p10_arr_delay"] is not None:
            assert s["p10_arr_delay"] <= s["p50_arr_delay"] <= s["p90_arr_delay"], \
                s["service_id"]


def test_every_flight_points_at_a_service_that_exists(batches):
    services = {s["service_id"] for s in batches["Service"]}
    assert {f["service_id"] for f in batches["Flight"]} <= services


def test_precedes_endpoints_are_loaded_flights(batches):
    """A dangling edge would fail at load time; cheaper to catch it here."""
    flights = {f["flight_id"] for f in batches["Flight"]}
    for edge in batches["PRECEDES"]:
        assert edge["from_id"] in flights and edge["to_id"] in flights


def test_airports_include_destination_only_stations(batches):
    """Built from origin UNION dest.

    One airport in 2024 appears only as a destination, so building from origins alone
    would leave an ARRIVES_AT edge pointing at a node that was never created.
    """
    airports = {a["iata"] for a in batches["Airport"]}
    assert {f["dest"] for f in batches["Flight"]} <= airports
    assert {f["origin"] for f in batches["Flight"]} <= airports


def test_weather_covers_both_ends(batches):
    """Departure-side-only extraction misses arrival weather entirely.

    Getting this wrong loses arrival weather, which is exactly what the shared-hub
    correlated-failure query needs.
    """
    # hour_utc is a datetime, not a string -- it has to be, or Neo4j silently
    # concatenates rather than adding when a duration is applied to it.
    assert not isinstance(batches["Weather"][0]["hour_utc"], str)
    arrival_airports = {w["iata"] for w in batches["Weather"]}
    assert len(batches["Weather"]) > 2_500
    assert {f["dest"] for f in batches["Flight"]} & arrival_airports


def test_rotation_edges_never_span_more_than_a_day(batches):
    """Guards a latent corruption rather than a live one.

    With a single day loaded there are no cross-day gaps, so this passes trivially
    today. The moment SEED_DAYS holds two non-adjacent days it stops being trivial:
    without the cap, an aircraft's last leg on one day chains to its first leg on the
    next, and spatial continuity does not catch it because the aircraft genuinely was
    at that airport both times. Measured at 209 such edges -- 2.3% of the total --
    for 19 July plus 29 October, invisible in every count.
    """
    assert batches["PRECEDES"], "no rotation edges at all"
    worst = max(e["gap_min"] for e in batches["PRECEDES"])
    assert worst <= kb.MAX_ROTATION_GAP_HOURS * 60, f"{worst} minutes between legs"
