"""GraphRAG acceptance: retrieval is correct and unambiguous before any LLM sees it.

Split deliberately. The retrieval tests need Neo4j and run every time; the LLM test
needs a network call and an API key, so it is marked and skipped unless asked for.
Most of what can go wrong here goes wrong in retrieval -- the Q3 failure during the
build was an ambiguous *field name*, not a model fault, and a test over rows would
have caught it.
"""

import os

import pytest

from murphy import graphrag as g, kg


def _reachable() -> bool:
    try:
        with kg.driver() as d:
            d.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="no reachable Neo4j instance")

ITINERARY = "itin-no-escape"
CASCADE_START = "2024-07-19_WN4854_LBB_DAL"
LONG_DELAY = "2024-07-19_DL2498_ATL_EWR"   # 186 min late, Delta
SHORT_DELAY = "2024-07-19_DL1246_MKE_ATL"  # 56 min late, Delta


# ------------------------------------------------------------------- retrieval

def test_itinerary_returns_both_legs_in_order():
    r = g.retrieve_itinerary(ITINERARY)
    assert [row["leg"] for row in r.rows] == [1, 2]
    assert r.rows[0]["to"] == r.rows[1]["from"], "legs should meet at the hub"
    assert all(row["traveler_deadline"] is not None for row in r.rows), \
        "the personal threshold is the point of the whole project"


def test_itinerary_carries_service_context():
    """A delay is meaningless without knowing what the route normally does."""
    r = g.retrieve_itinerary(ITINERARY)
    for row in r.rows:
        assert row["typical_delay_for_this_service"] is not None
        assert row["service_sample_size"] > 50


def test_cascade_is_genuinely_multi_hop():
    r = g.retrieve_cascade(CASCADE_START)
    hops = [row["hops_downstream"] for row in r.rows]
    assert hops == sorted(hops)
    assert max(hops) >= 3, "needs real depth to be a multi-hop demonstration"
    assert len({row["aircraft"] for row in r.rows}) == 1, "one airframe throughout"


def test_cascade_carries_the_cause_attribution():
    """The path is only an explanation because BTS attributes the delay separately."""
    r = g.retrieve_cascade(CASCADE_START)
    blamed = [row["blamed_on_late_aircraft_min"] for row in r.rows
              if row["blamed_on_late_aircraft_min"] is not None]
    assert max(blamed) > 100


# ---------------------------------------------------- the ambiguity regression

def test_route_risk_field_names_cannot_be_misread():
    """Guards the exact failure found during the build.

    The first version returned one field called `avg_arrival_delay_min`, computed over
    every option. The model attributed it to the compromised subset and was wrong by
    11 minutes. Every aggregate must now name its own scope.
    """
    r = g.retrieve_route_risk("ATL", "EWR")
    row = r.rows[0]
    averages = [k for k in row if k.startswith("avg_")]
    assert len(averages) == 3, "one unscoped average is what caused the misreading"
    for key in averages:
        assert any(scope in key for scope in ("ALL", "COMPROMISED", "OTHER")), \
            f"{key} does not say which set it covers"
    assert row["avg_delay_among_COMPROMISED_options"] != row["avg_delay_across_ALL_options"]


def test_route_risk_distinguishes_traceable_from_compromised():
    """Two different counts, and conflating them would overstate the finding."""
    row = g.retrieve_route_risk("ATL", "EWR").rows[0]
    assert row["options_with_traceable_upstream"] <= row["total_options"]
    assert row["options_whose_aircraft_was_already_late"] <= row["options_with_traceable_upstream"]


# ------------------------------------------------------- the GraphRAG narrowing

def test_graph_context_narrows_policy_to_one_carrier():
    """The whole argument for GraphRAG over plain text search."""
    narrowed = g.retrieve_policy("can I get a hotel or a meal", flight_id=LONG_DELAY)
    open_search = g.retrieve_policy("can I get a hotel or a meal")

    assert {r["airline"] for r in narrowed.rows} == {"Delta Air Lines"}
    assert len({r["airline"] for r in open_search.rows}) > 1, \
        "unnarrowed search should span carriers -- that is the point of the contrast"
    assert "Delta Air Lines" not in {r["airline"] for r in open_search.rows}


def test_threshold_gate_respects_the_actual_delay():
    """A 3-hour commitment must not surface on a 56-minute delay."""
    long_delay = {r["commitment"] for r in
                  g.retrieve_policy("hotel meal voucher", flight_id=LONG_DELAY, limit=20).rows}
    short_delay = {r["commitment"] for r in
                   g.retrieve_policy("hotel meal voucher", flight_id=SHORT_DELAY, limit=20).rows}
    assert "meal" in long_delay
    assert "meal" not in short_delay
    assert short_delay < long_delay, "the short delay qualifies for strictly fewer"


def test_every_clause_is_traceable_to_a_source():
    for row in g.retrieve_policy("hotel", flight_id=LONG_DELAY).rows:
        assert row["url"].startswith("https://")
        assert row["clause_text"] and row["source"]


# ------------------------------------------------------------------- provenance

@pytest.mark.parametrize("make", [
    lambda: g.retrieve_itinerary(ITINERARY),
    lambda: g.retrieve_cascade(CASCADE_START),
    lambda: g.retrieve_route_risk("ATL", "EWR"),
    lambda: g.retrieve_policy("hotel", flight_id=LONG_DELAY),
])
def test_retrieval_carries_its_own_query(make):
    """An answer the traveller cannot check is not grounded, only fluent."""
    r = make()
    assert r.rows and r.cypher and r.tool
    assert "MATCH" in r.cypher or "CALL" in r.cypher


def test_no_query_interpolates_its_parameters():
    """Fixed templates, never string-built -- the reason Text2Cypher was rejected."""
    for cypher in (g.ITINERARY, g.CASCADE, g.ROUTE_RISK, g.POLICY_SEARCH, g.POLICY_CONTEXT):
        assert "$" in cypher, "parameters must be bound, not substituted"
        for literal in ("itin-no-escape", "2024-07-19", "'ATL'", "'EWR'"):
            assert literal not in cypher


def test_empty_context_is_explicit():
    """A retriever that finds nothing must say so, not hand the model a blank."""
    empty = g.retrieve_itinerary("no-such-itinerary")
    assert empty.empty
    assert "no rows" in empty.as_context()


# -------------------------------------------------------------------- the LLM

@pytest.mark.skipif(not os.environ.get("MURPHY_LLM_TESTS"),
                    reason="set MURPHY_LLM_TESTS=1 to spend API calls")
def test_answer_is_grounded_in_the_evidence():
    r = g.retrieve_route_risk("ATL", "EWR")
    a = g.answer(r, question="How many options had an aircraft already running late?")
    assert str(int(r.rows[0]["options_whose_aircraft_was_already_late"])) in a.answer
    assert a.evidence == r.rows, "the evidence must travel with the answer"


@pytest.mark.skipif(not os.environ.get("MURPHY_LLM_TESTS"),
                    reason="set MURPHY_LLM_TESTS=1 to spend API calls")
def test_out_of_scope_question_is_refused_not_invented():
    """Routing has no verification; grounding is the only backstop. Check it holds."""
    r = g.retrieve_itinerary(ITINERARY)
    a = g.answer(r, question="What will the weather be in Denver next Tuesday?")
    assert "does not contain" in a.answer.lower()
