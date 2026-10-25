"""U3 acceptance: the Documented tier is real, sourced, and discriminating.

Every clause must trace to a published document. Nothing here is paraphrased, and
nothing is invented to fill a gap -- a carrier the source does not cover gets no
edges, and that absence is reported rather than patched.
"""

import pytest

from murphy import kg_policy


@pytest.fixture(scope="module")
def policy():
    return kg_policy.summary()


def test_dashboard_yields_sixteen_clauses(policy):
    """Eight commitments across two disruption types."""
    dot = [c for c in policy["PolicyClause"] if c["clause_id"].startswith("dot-")]
    assert len(dot) == 16
    assert {c["disruption_type"] for c in dot} == {"cancellation", "delay"}


def test_every_clause_has_its_filter_keys(policy):
    """These keys are what the graph hands a vector retriever next stage.

    Without them the document tier is undifferentiated text and GraphRAG collapses
    into plain vector search.
    """
    for c in policy["PolicyClause"]:
        assert c["disruption_type"] in {"cancellation", "delay", "both"}
        assert isinstance(c["controllable"], bool)
        assert c["domestic"] is True
        assert c["threshold_hours"] is None or isinstance(c["threshold_hours"], int)


def test_every_clause_is_traceable_to_a_published_document(policy):
    for c in policy["PolicyClause"]:
        assert c["url"].startswith("https://")
        assert c["source"] and c["section"]
        assert len(c["text"]) > 40, f"{c['clause_id']} has no substantive text"


def test_a_commitment_not_made_is_an_absent_edge(policy):
    """Frontier declines hotels; the graph says so by having no edge, not a 'No'."""
    hotel = "dot-cancellation-hotel"
    carriers = {p["carrier"] for p in policy["PUBLISHES"] if p["clause_id"] == hotel}
    assert "DL" in carriers
    assert "F9" not in carriers
    assert all("answer" not in c and "value" not in c for c in policy["PolicyClause"])


def test_every_carrier_in_scope_is_covered_by_the_dashboard(policy):
    """Scope and source line up exactly, and that is not a coincidence.

    Spirit was the one carrier in scope the dashboard did not list. It ceased
    operations on 2 May 2026, which is also why a dashboard describing carriers that
    currently fly had no row for it. Dropping Spirit from scope closed the gap.
    """
    from murphy import kg_build
    assert kg_policy.NOT_ON_DASHBOARD == ()
    assert set(kg_policy.NAME_TO_CARRIER.values()) == set(kg_build.MAJORS)
    assert "NK" not in {p["carrier"] for p in policy["PUBLISHES"]}


def test_cash_compensation_is_offered_by_nobody(policy):
    """A finding, not a bug: no US carrier commits to cash for a controllable delay."""
    for disruption in ("cancellation", "delay"):
        clause = f"dot-{disruption}-cash_compensation"
        assert not [p for p in policy["PUBLISHES"] if p["clause_id"] == clause]


def test_contract_clauses_come_from_the_pdf_not_a_transcription(policy):
    """Anchors locate the text in the document, so a revision fails loudly."""
    coc = [c for c in policy["PolicyClause"] if c["clause_id"].startswith("coc-")]
    assert len(coc) == 3
    assert any("force majeure" in c["section"].lower() for c in coc)
    for c in coc:
        assert "(p" in c["section"], "a clause must say which page it came from"


def test_contracts_supply_the_uncontrollable_half(policy):
    """The dashboard covers controllable disruptions and assumes the other half.

    Delta's force majeure list is what defines the boundary, so the graph carries
    both sides rather than only the promises.
    """
    uncontrollable = [c for c in policy["PolicyClause"] if c["controllable"] is False]
    assert uncontrollable
    assert "acts of God" in uncontrollable[0]["text"]
