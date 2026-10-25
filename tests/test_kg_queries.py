"""U5 acceptance: each demo query returns a hand-checked, non-empty result.

The spec's criterion is that every query answers against the seed data and that the
two headline queries show what they claim to show -- not merely that they parse.
That distinction matters here more than usual: three separate Cypher mistakes during
this build returned wrong answers rather than raising, so "it ran" is not evidence.
"""

import pytest

from murphy import kg, kg_queries


def _reachable() -> bool:
    try:
        with kg.driver() as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="no reachable Neo4j instance")


@pytest.mark.parametrize("name", sorted(kg_queries.QUERIES))
def test_every_query_returns_rows(name):
    assert kg_queries.run(name), f"{name} returned nothing"


def test_q1_counts_every_node_label():
    """Every node type the schema defines is present and counted.

    Membership, not set equality: `:AirportHour` was added as a second label on
    `Weather` for the GNN work (decision N3), and a test that treats a new label as a
    failure is testing the schema's size rather than its correctness.
    """
    counts = {r["label"]: r["nodes"] for r in kg_queries.run("q1")}
    assert counts["Flight"] == 15_224
    assert {"Flight", "Service", "Aircraft", "Weather", "Airport",
            "Airline", "PolicyClause", "Traveler", "Itinerary"} <= set(counts)


def test_q2_counts_every_relationship_type():
    counts = {r["relationship"]: r["edges"] for r in kg_queries.run("q2")}
    assert counts["PRECEDES"] == 8_982
    assert "CONTAINS" in counts and "BOOKED" in counts


def test_q3_returns_ordered_legs_with_the_deadline():
    rows = kg_queries.run("q3")
    assert [r["leg"] for r in rows] == [1, 2]
    assert rows[0]["to"] == rows[1]["from"], "legs should meet at the hub"
    assert rows[0]["traveler_deadline"] is not None, "the personal threshold is the point"


def test_q4_returns_the_whole_aircraft_day_not_one_flight():
    rows = kg_queries.run("q4")
    assert len({r["aircraft"] for r in rows}) == 1
    assert len(rows) == 5, "N928WN flew five legs on the seed day"
    # The legs must chain: each arrival is the next departure.
    for earlier, later in zip(rows, rows[1:]):
        assert earlier["to"] == later["from"]


def test_q5_shows_the_cascade_building_along_the_chain():
    """The multi-hop query has to demonstrate propagation, not just reachability.

    The aircraft starts the day fine, breaks with a carrier delay, and the following
    legs inherit it -- with BTS independently attributing the inheritance to "late
    aircraft." The traversal and the cause codes agree, which is what makes the path
    an explanation rather than a coincidence.
    """
    rows = kg_queries.run("q5")
    assert max(r["hops"] for r in rows) >= 3, "needs real depth"

    blamed = [r["blamed_on_late_aircraft"] for r in rows
              if r["blamed_on_late_aircraft"] is not None]
    assert max(blamed) > 100, "the downstream legs should be blamed on late aircraft"

    delays = [r["arrived_late_by"] for r in rows]
    assert delays[0] < 15 and delays[-1] > 100, "delay should grow along the chain"


def test_q6_finds_alternatives_that_share_a_failure_cause():
    """The project-specific query, and the argument for using a graph at all.

    Every ATL->EWR option that day was flown by an aircraft already running late.
    A forecaster scoring legs independently sees nine separate moderate risks; the
    graph shows one cause behind all of them.
    """
    rows = kg_queries.run("q6")
    assert {r["leaving_from"] for r in rows} == {"ATL"}
    assert {r["heading_to"] for r in rows} == {"EWR"}
    assert sum(r["is_my_booked_flight"] for r in rows) == 1, "the booked flight is ranked among them"

    compromised = [r for r in rows if r["already_late_by"] is not None]
    assert len(compromised) >= 8
    assert all(r["already_late_by"] >= 45 for r in compromised), \
        "every option with a traceable upstream leg was already compromised"
    assert all(r["option_arrived_late_by"] > 100 for r in compromised)


def test_connection_feasibility_respects_the_window():
    rows = kg_queries.run("connection")
    assert all(45 <= r["scheduled_slack_min"] <= 240 for r in rows)
    assert all(r["hub"] == "MDW" for r in rows)
    # The point of the query: scheduled slack looks safe, real slack does not.
    tightest = min(r["slack_after_inbound_delay"] for r in rows)
    assert tightest < 0, "the inbound's delay should wipe out the smaller connections"


def test_queries_are_parameterised_not_interpolated():
    """No query may carry a literal from the sample data.

    Interpolating would work and then quietly break the moment the seed day changes;
    worse, it is the shape that invites a model to generate Cypher instead of filling
    parameters in a fixed template.
    """
    for name, (_, cypher, params) in kg_queries.QUERIES.items():
        for value in params.values():
            assert str(value) not in cypher, f"{name} hardcodes {value!r}"
