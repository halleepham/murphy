"""U2 acceptance: the loaded graph matches the schema the spec describes.

These tests need a reachable Neo4j instance and skip without one, so the suite still
runs on a machine that has only the Parquet data.
"""

import pytest

from murphy import kg

# Aura Free's ceilings. Worth asserting because exceeding them fails a load silently
# late in the run, and because every schema addition has to be checked against them.
NODE_CAP = 200_000
REL_CAP = 400_000

# Counts that are facts about the seed day's data, not about any assignment's limits.
# Asserted individually rather than as one frozen dictionary: an added label or
# relationship type is iteration working, and a test that fails on it tests the wrong
# thing. The loader's real invariant -- that it does not duplicate -- is carried by the
# uniqueness constraints and by these counts not growing.
NODE_COUNTS = {
    "Flight": 15_224, "Service": 15_158, "Aircraft": 3_596,
    "Weather": 2_843, "Airport": 202, "Airline": 9,
    "Traveler": 4, "Itinerary": 4, "PolicyClause": 19,
}

REL_COUNTS = {
    "FLIES": 14_412, "PRECEDES": 8_982, "INSTANCE_OF": 15_224,
    "DEPARTS_FROM": 30_382, "ARRIVES_AT": 30_382,
    "OPERATED_BY": 15_158, "OBSERVED_AT": 2_843,
    "BOOKED": 4, "CONTAINS": 7, "PUBLISHES": 95,
}


def _reachable() -> bool:
    try:
        with kg.driver() as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(), reason="no reachable Neo4j instance"
)


def test_the_graph_fits_inside_the_aura_caps():
    """The operational constraint that actually binds, unlike a count of types."""
    nodes, rels = kg.counts()
    assert nodes <= NODE_CAP, f"{nodes:,} nodes exceeds the {NODE_CAP:,} cap"
    assert rels <= REL_CAP, f"{rels:,} relationships exceeds the {REL_CAP:,} cap"


def test_known_node_labels_have_their_expected_counts():
    """Each label checked on its own, so a new label is not a failure.

    A duplicate-inserting loader shows up here as a count that grew.
    """
    rows = kg.run("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")
    actual = {r["label"]: r["n"] for r in rows}
    for label, expected in NODE_COUNTS.items():
        assert actual.get(label) == expected, f"{label}: {actual.get(label)} != {expected}"


def test_known_relationship_types_have_their_expected_counts():
    rows = kg.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")
    actual = {r["t"]: r["n"] for r in rows}
    for rel, expected in REL_COUNTS.items():
        assert actual.get(rel) == expected, f"{rel}: {actual.get(rel)} != {expected}"


def test_flights_do_not_duplicate_their_airports():
    """origin and dest live on the edges, not as properties (decision S6)."""
    row = kg.run("MATCH (f:Flight) RETURN f LIMIT 1")[0]
    props = dict(row["f"])
    for leaked in ("origin", "dest", "service_id", "tail_number"):
        assert leaked not in props


def test_outcomes_are_prefixed_on_the_stored_node():
    row = kg.run("MATCH (f:Flight) WHERE f.outcome_arr_delay_min IS NOT NULL "
                 "RETURN f LIMIT 1")[0]
    props = dict(row["f"])
    assert "arr_delay_min" not in props
    assert "outcome_arr_delay_min" in props


def test_every_flight_reaches_its_service_and_airports():
    # The per-row subqueries have to be bound in a WITH before aggregating; placing
    # them beside count(*) in one RETURN silently collapses them to a single row.
    row = kg.run("""
        MATCH (f:Flight)
        WITH count { (f)-[:INSTANCE_OF]->(:Service) }  AS svc,
             count { (f)-[:DEPARTS_FROM]->(:Airport) } AS org,
             count { (f)-[:ARRIVES_AT]->(:Airport) }   AS dst
        RETURN count(*) AS total,
               sum(CASE WHEN svc > 0 THEN 1 ELSE 0 END) AS with_service,
               sum(CASE WHEN org > 0 THEN 1 ELSE 0 END) AS with_origin,
               sum(CASE WHEN dst > 0 THEN 1 ELSE 0 END) AS with_dest
    """)[0]
    assert row["with_service"] == row["total"]
    assert row["with_origin"] == row["total"]
    assert row["with_dest"] == row["total"]


def test_only_cancelled_flights_lack_an_aircraft():
    """FLIES is short of the flight count by exactly the cancelled flights.

    BTS assigns a tail number only when an aircraft actually operated, so a missing
    one is information rather than a gap.
    """
    row = kg.run("""
        MATCH (f:Flight) WHERE NOT (f)<-[:FLIES]-(:Aircraft)
        RETURN count(*) AS n, sum(f.outcome_cancelled) AS cancelled
    """)[0]
    assert row["n"] == row["cancelled"] == 812


def test_precedes_chains_are_spatially_continuous():
    """An edge means the same airframe really continued, not just a shared tail."""
    row = kg.run("""
        MATCH (a:Flight)-[:PRECEDES]->(b:Flight)
        MATCH (a)-[:ARRIVES_AT]->(x:Airport)
        MATCH (b)-[:DEPARTS_FROM]->(y:Airport)
        RETURN count(*) AS total, sum(CASE WHEN x = y THEN 1 ELSE 0 END) AS continuous
    """)[0]
    assert row["continuous"] == row["total"]


def test_services_carry_their_precomputed_quantiles():
    """The LLM cannot derive a quantile from a subgraph, so it must be stored."""
    row = kg.run("""
        MATCH (s:Service)
        RETURN count(*) AS total,
               count(s.p50_arr_delay) AS with_median,
               count(s.n_flights)     AS with_sample_size
    """)[0]
    assert row["with_median"] > row["total"] * 0.99
    assert row["with_sample_size"] == row["total"]


def test_multi_hop_rotation_chain_exists():
    """Q5 needs real depth to traverse, not a graph of two-leg stubs."""
    row = kg.run("""
        MATCH path = (:Flight)-[:PRECEDES*3]->(:Flight)
        RETURN count(path) AS n
    """)[0]
    assert row["n"] > 0


def test_timestamps_are_temporal_types_not_strings():
    """The bug this guards failed silently, which is why it needs a test.

    Stored as strings, `sched_arr_utc + duration({minutes: 45})` returns the two
    concatenated -- '2024-07-19 09:15:00-05PT45M' -- with no error. Every
    connection-feasibility query would have run and been wrong.
    """
    row = kg.run("""
        MATCH (f:Flight) WITH f LIMIT 1
        RETURN f.sched_dep_utc AS dep,
               (f.sched_arr_utc + duration({minutes: 45})) AS plus45
    """)[0]
    assert not isinstance(row["dep"], str), "timestamps regressed to strings"
    assert not isinstance(row["plus45"], str), "duration arithmetic degraded to concat"


def test_connection_feasibility_computes_real_slack():
    """The derived CONNECTS_TO of decision S3, end to end."""
    rows = kg.run("""
        MATCH (a:Flight {flight_id: $fid})-[:ARRIVES_AT]->(hub:Airport)
              <-[:DEPARTS_FROM]-(b:Flight)
        WHERE b.sched_dep_utc >= a.sched_arr_utc + duration({minutes: 45})
          AND b.sched_dep_utc <= a.sched_arr_utc + duration({hours: 4})
          AND b.carrier = a.carrier
        RETURN duration.between(a.sched_arr_utc, b.sched_dep_utc).minutes AS slack
    """, fid="2024-07-19_WN2679_HOU_MDW")
    assert rows, "no connections found for the cascade itinerary's inbound"
    assert all(45 <= r["slack"] <= 240 for r in rows)


def test_sample_itineraries_resolve_to_real_flights():
    rows = kg.run("""
        MATCH (t:Traveler)-[:BOOKED]->(i:Itinerary)-[c:CONTAINS]->(f:Flight)
        RETURN i.itinerary_id AS id, count(f) AS legs,
               i.origin AS origin, i.final_dest AS dest, i.total_sched_min AS mins
        ORDER BY id
    """)
    assert len(rows) == 4
    for r in rows:
        assert r["origin"] and r["dest"] and r["mins"] > 0
    by_id = {r["id"]: r for r in rows}
    assert by_id["itin-cascade"]["legs"] == 2
    assert by_id["itin-nonstop"]["legs"] == 1


def test_the_cascade_itinerary_actually_cascades():
    """The headline demo: the inbound is late because its own aircraft was late."""
    row = kg.run("""
        MATCH (:Itinerary {itinerary_id: 'itin-cascade'})-[c:CONTAINS {leg_order: 1}]->(f:Flight)
        MATCH (upstream:Flight)-[:PRECEDES]->(f)
        RETURN f.outcome_arr_delay_min AS delay,
               f.outcome_delay_late_aircraft_min AS blamed,
               upstream.outcome_arr_delay_min AS upstream_delay
    """)[0]
    assert row["delay"] > 60
    assert row["blamed"] > 60, "BTS should attribute most of this to late aircraft"
    assert row["upstream_delay"] > 60, "the upstream leg should itself be late"


def test_the_only_orphans_are_commitments_nobody_makes():
    """An isolated node is normally a loader bug -- an edge whose MATCH failed
    silently. There are exactly two legitimate exceptions, and they are a finding
    rather than a defect: **no US carrier commits to cash compensation** for a
    controllable delay or cancellation, so those two clauses have no publisher.

    Keeping the nodes preserves the answer. Dropping them would make "nobody offers
    this" indistinguishable from "we never asked."
    """
    rows = kg.run("""
        MATCH (n) WHERE NOT (n)--()
        RETURN labels(n)[0] AS label, n.clause_id AS clause_id
    """)
    assert {(r["label"], r["clause_id"]) for r in rows} == {
        ("PolicyClause", "dot-cancellation-cash_compensation"),
        ("PolicyClause", "dot-delay-cash_compensation"),
    }
