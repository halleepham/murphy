"""GraphRAG. Offline: a fake Neo4j session, no LLM call.
The live comparison is scripts/graphrag_eval.py."""
import pytest

from murphy import graph as kg, graphrag as gr


class FakeSession:
    """Answers each fixed query with canned rows."""
    def __init__(self, rows):
        self.rows = rows

    def run(self, cypher, **_):
        name = next(n for n, (_, cy) in kg.QUERIES.items() if cy == cypher)
        return type("R", (), {"data": lambda _s: [dict(r) for r in self.rows.get(name, [])]})()


WHY = [
    {"airport": "ATL", "co_disruption_weight": None, "community": 4, "metar_id": "ATL:1", "metar": "KATL FZRA"},
    {"airport": "CLT", "co_disruption_weight": 1.0612, "community": 4, "metar_id": "CLT:1", "metar": "KCLT CLR"},
    {"airport": "RDU", "co_disruption_weight": 1.021, "community": 4, "metar_id": None, "metar": None},
]


def test_every_question_runs_known_queries():
    for _, (_, queries) in gr.QUESTIONS.items():
        assert set(queries) <= set(kg.QUERIES)


def test_code_formats_numbers_before_the_llm_sees_them():
    row = gr.fmt({"alt_p_miss": 0.156, "trip_p_miss": 0.4189, "co_disruption_weight": 1.06123, "dep_local": "1124"})
    assert row == {"alt_p_miss": "16%", "trip_p_miss": "42%", "co_disruption_weight": 1.061, "dep_local": "1124"}


def test_evidence_drops_neighbors_without_weather_and_the_ablation_drops_communities():
    s = FakeSession({"why": WHY})
    with_c = gr.evidence(s, "why", "F", 24)
    assert [e["airport"] for e in with_c] == ["ATL", "CLT"]  # RDU has no METAR: nothing known at as_of
    without = gr.evidence(s, "why", "F", 24, communities=False)
    assert [e["airport"] for e in without] == ["ATL"]
    assert not gr.COMMUNITY_KEYS & set(without[0])


def test_unknown_flight_gives_no_evidence_and_says_so():
    ev = gr.evidence(FakeSession({}), "better", "XX", 24)
    assert ev == []
    assert "(no evidence found)" in gr.messages("Is there a better flight?", "XX", 24, ev)[-1]["content"]


def test_bad_lead_is_refused_before_any_query():
    with pytest.raises(ValueError):
        gr.evidence(FakeSession({}), "chance", "F", 12)


def test_glossary_is_sent_for_fields_in_the_evidence_only():
    ev = [{"tag": "E1", "query": "alternatives", "alt_p_miss": "16%", "arr_ahead_min": 174}]
    msg = gr.messages("q", "F", 24, ev)[-1]["content"]
    assert "alt_p_miss:" in msg and "arr_ahead_min:" in msg and "co_disruption_weight:" not in msg


EV = [{"tag": "E1", "metar": "KMDW 19012G20KT", "p_miss": "15%"}]
CONTEXT = "Trip flight: F. " + gr.as_text(EV)


def test_check_passes_copied_numbers_and_odd_tag_spelling():
    c = gr.check("A 15% chance [​E1] and 【E1】 [ E1 ].", CONTEXT, EV)
    assert c == {"invented": [], "bad_tags": [], "cited": 1}


def test_check_catches_a_decoded_number_the_evidence_does_not_hold():
    # Run 1: the model read 19012G20KT as "190 kt". It is 190 degrees at 12 kt.
    assert gr.check("Winds near 190 kt [E1].", CONTEXT, EV)["invented"] == ["190"]
    assert gr.check("Ceiling at 5 000 ft [E1].", CONTEXT, EV)["invented"] == ["5000"]


def test_check_catches_a_citation_to_missing_evidence():
    assert gr.check("Snow is coming [E3].", CONTEXT, EV)["bad_tags"] == ["E3"]


def test_bm25_finds_the_trip_and_hindsight_is_counted():
    docs = [{"kind": "flight", "id": "a", "text": "Flight 2025-01-10:AA3104:ATL-DFW. Outcome: cancelled", "known_at": "2025-01-10T19:10Z"},
            {"kind": "metar", "id": "b", "text": "METAR ORD clear", "known_at": "2025-01-09T10:00Z"},
            {"kind": "metar", "id": "c", "text": "METAR SEA rain", "known_at": "2025-01-09T11:00Z"}]
    top = gr.bm25(docs, "how likely 2025-01-10:AA3104:ATL-DFW", k=2)
    assert top[0]["text"].startswith("Flight 2025-01-10:AA3104")
    assert gr.hindsight(top, "2025-01-09T19:10Z") == 1  # the outcome, learned after as_of
