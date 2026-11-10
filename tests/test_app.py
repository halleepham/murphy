import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from murphy.app.server import MODEL, create_app
from murphy.forecast.sample import ROOT
from murphy.forecast.score import PCOLS


def probs(make):
    """Chance `make` of arriving within 15 minutes, the rest after three hours."""
    return {k: 0.0 for k in PCOLS} | {"p_0_15": make, "p_180p": 1 - make}


def trip(fid, make, cancelled=False):
    return {"flight_id": fid, "lead_h": 2, "flight_date": "2025-03-01", "as_of_utc": "2025-03-01 10:00:00", "carrier": "DL",
            "flight_number": 1, "origin": "LGA", "dest": "ORD", "crs_dep_local": "0700", "crs_arr_local": "0900",
            "sched_dep_utc": pd.Timestamp("2025-03-01 12:00"), "sched_arr_utc": pd.Timestamp("2025-03-01 15:00"),
            "known_cancelled": cancelled, **probs(make)}


def alt(fid, aid, make, carrier, earlier_h=0.0):
    return {"flight_id": fid, "lead_h": 2, "alt": aid, "alt_lead": 2 - earlier_h, "arr_ahead_min": 0, "carrier": carrier,
            "flight_number": 9, "flight_date": "2025-03-01", "crs_dep_local": "0800", "crs_arr_local": "1000",
            "sched_dep_utc": pd.Timestamp("2025-03-01 13:00"), "sched_arr_utc": pd.Timestamp("2025-03-01 16:00"), **probs(make)}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("app") / "t.duckdb"
    con = duckdb.connect(str(path))
    trips = pd.DataFrame([trip("low", .9), trip("mid", .5), trip("high", .2), trip("gone", .2, True), trip("alone", .2), trip("low2", .9),
                          trip("jan", .2) | {"flight_date": "2025-01-05"}])
    alts = pd.DataFrame([alt("high", "a1", .9, "UA"), alt("high", "a2", .8, "DL"), alt("high", "a3", .7, "XX"),
                         alt("high", "a4", .95, "UA", earlier_h=3), alt("mid", "a5", .9, "UA"), alt("low2", "a6", .5, "UA"),
                         alt("jan", "a7", .9, "UA")])
    fares = pd.DataFrame({"origin": ["LGA", "LGA"], "dest": ["ORD", "ORD"], "carrier": ["UA", "DL"], "tickets": [100, 100], "fare": [300.0, 200.0]})
    track = pd.DataFrame([{"deadline_min": d, "band": b, "forecasts": 10, "predicted_miss": .5, "true_miss": .6}
                          for d in range(0, 121, 15) for b in ("keep", "too_uncertain", "high")])
    typical = pd.DataFrame([{"lead_h": 2, "deadline_min": d, "chance": .8} for d in range(0, 181, 15)])
    for name in ("trips", "alts", "fares", "track", "typical"):
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}")
    con.close()
    return TestClient(create_app(path))


def get(client, fid, **kw):
    return client.get("/api/trip", params={"flight_id": fid, **kw})


def test_low_risk_keeps(client):
    assert get(client, "low").json()["answer"]["verdict"] == "keep"


def test_high_risk_changes_marks_best_and_traces_the_cost(client):
    d = get(client, "high").json()
    assert d["answer"]["verdict"] == "consider_changing"
    assert [(a["flight_id"], a["best"]) for a in d["alternatives"]] == [("a1", True), ("a2", False), ("a3", False)]
    assert d["alternatives"][0]["cost"]["fare"] == 300.0 and "DB1B" in d["alternatives"][0]["cost"]["source"]
    assert d["provenance"]["model"] == MODEL == "final_network" and d["uncertainty"]["limits"]


def test_too_uncertain_still_lists_alternatives(client):
    d = get(client, "mid").json()
    assert d["answer"]["verdict"] == "too_uncertain" and len(d["alternatives"]) == 1


def test_earlier_limit_is_the_travelers_control(client):
    assert get(client, "high", max_earlier_h=3).json()["alternatives"][0]["flight_id"] == "a4"


def test_price_sort_reorders_but_cost_never_changes_the_advice(client):
    by_chance, by_price = get(client, "high").json(), get(client, "high", sort="price").json()
    assert [a["flight_id"] for a in by_price["alternatives"]] == ["a2", "a1", "a3"]  # no estimate last
    assert by_price["alternatives"][2]["cost"]["fare"] is None
    assert [a["flight_id"] for a in by_price["alternatives"] if a["best"]] == ["a1"]
    assert by_price["answer"] == by_chance["answer"]


def test_a_worse_flight_is_never_marked_best(client):
    d = get(client, "mid", deadline_min=60).json()
    assert d["alternatives"][0]["best"] is True  # 90% against 50%
    assert all(not a["best"] for a in get(client, "low2").json()["alternatives"])


def test_known_cancelled_is_stated(client):
    assert get(client, "gone").json()["answer"]["verdict"] == "cancelled"


def test_high_risk_with_nothing_to_offer(client):
    d = get(client, "alone").json()
    assert d["answer"]["verdict"] == "at_risk_no_better_flight" and d["alternatives"] == []


def test_unknown_flight_and_bad_settings_are_refused(client):
    assert get(client, "nope").status_code == 404
    assert get(client, "low", deadline_min=50).status_code == 422  # not a 15-minute step
    assert get(client, "low", deadline_min=135).status_code == 422  # past what was checked
    assert get(client, "low", sort="vibes").status_code == 422


def test_no_outcome_reaches_the_screen(client):
    text = get(client, "high").text
    assert "outcome" not in text and "arr_delay" not in text


def test_examples_put_flights_the_questions_cover_first(client):
    rows = client.get("/api/examples").json()
    assert rows[0]["flight_id"] == "jan" and rows[0]["askable"] and not any(r["askable"] for r in rows[1:])
    assert len(rows) <= 6
    assert get(client, "jan").json()["askable"] and not get(client, "high").json()["askable"]


def test_find_a_flight(client):
    rows = client.get("/api/flights", params={"date": "2025-03-01", "carrier": "dl", "number": 1}).json()
    assert len(rows) == 6
    assert client.get("/api/flights", params={"date": "not-a-date", "carrier": "DL", "number": 1}).json() == []


# --- Suggested questions: a fake graph, no Groq call -------------------------------------

from contextlib import contextmanager
from murphy import graphrag as gr
from tests.test_graphrag import FakeSession

FLIGHT = [{"flight": "F", "forecasts": [{"lead_h": 2, "as_of": None, "p_miss": 0.42}]}]


@pytest.fixture
def asker(tmp_path, monkeypatch):
    seen = {}
    class Recording(FakeSession):
        def run(self, cypher, **params):
            seen.update(params)
            return super().run(cypher, **params)
    rows = {"flight": FLIGHT, "alternatives": [{"alternative": "A", "alt_p_miss": 0.16, "trip_p_miss": 0.42}]}
    graph = {"rows": rows}

    @contextmanager
    def session():
        if graph.get("down"):
            raise OSError("unreachable")
        yield Recording(graph["rows"])
    duckdb.connect(str(tmp_path / "e.duckdb")).close()
    c = TestClient(create_app(tmp_path / "e.duckdb", graph_session=session))
    return c, graph, seen


def ask(c, question="chance", **kw):
    return c.get("/api/ask", params={"flight_id": "F", "question": question, "lead_h": 2, **kw})


def test_a_question_returns_the_explanation_its_evidence_and_checks(asker, monkeypatch):
    c, _, _ = asker
    monkeypatch.setattr(gr, "llm", lambda msgs, wait=True: "A 42% chance of being late [E1].")
    d = ask(c).json()
    assert d["status"] == "answered" and d["evidence"][0]["tag"] == "E1"
    assert d["checks"] == {"invented": [], "bad_tags": [], "cited": 1}
    assert d["provenance"]["queries"] == ["flight"]


def test_other_deadlines_are_refused_not_shown_beside_a_different_number(asker):
    c, _, _ = asker
    r = ask(c, deadline_min=30)
    assert r.status_code == 422 and "60-minute" in r.json()["detail"]


def test_at_the_daily_limit_the_evidence_still_shows(asker, monkeypatch):
    c, _, _ = asker
    def limited(msgs, wait=True):
        raise gr.Paused("400")
    monkeypatch.setattr(gr, "llm", limited)
    d = ask(c, question="better").json()
    assert d["status"] == "paused" and d["explanation"] is None and len(d["evidence"]) == 1


def test_the_travelers_earlier_limit_reaches_the_query(asker, monkeypatch):
    c, _, seen = asker
    monkeypatch.setattr(gr, "llm", lambda msgs, wait=True: "ok")
    ask(c, question="better", max_earlier_h=3)
    assert seen["max_earlier_h"] == 3.0


def test_a_flight_outside_the_graph_calls_no_llm(asker, monkeypatch):
    c, graph, _ = asker
    graph["rows"] = {}
    monkeypatch.setattr(gr, "llm", lambda msgs, wait=True: pytest.fail("LLM called"))
    assert ask(c).json()["status"] == "not_in_graph"


def test_graph_down_is_said_plainly(asker):
    c, graph, _ = asker
    graph["down"] = True
    r = ask(c)
    assert r.status_code == 503 and "not reachable" in r.json()["detail"]


@pytest.fixture
def past_client(client, tmp_path):
    from tests.test_pastdays import fake_taf, trip_index
    from datetime import date
    path = tmp_path / "p.duckdb"
    con = duckdb.connect(str(path))
    trips = pd.DataFrame([trip("high", .2) | {"origin": "ORD"}, trip("far", .2) | {"origin": "ABQ"}])
    con.execute("CREATE TABLE trips AS SELECT * FROM trips")
    con.close()
    return TestClient(create_app(path, past_index=lambda: trip_index(80), taf_as_of=fake_taf(date(2025, 3, 1))))


@pytest.mark.skipif(not (ROOT / "data" / "airports.parquet").exists(), reason="data/airports.parquet not downloaded")
def test_similar_past_days_for_a_trip(past_client):
    d = past_client.get("/api/pastdays", params={"flight_id": "high"}).json()
    assert d["status"] == "ok" and d["answer"]["days"] == 10
    assert all(x["date"] < "2025-03-01" for x in d["days"])      # the trip's own day never shows
    assert "decisions" not in d["provenance"] and d["uncertainty"]["thin_above"] == 3.16


@pytest.mark.skipif(not (ROOT / "data" / "airports.parquet").exists(), reason="data/airports.parquet not downloaded")
def test_similar_past_days_refuses_unknown_and_says_no_weather(past_client):
    assert past_client.get("/api/pastdays", params={"flight_id": "nope"}).status_code == 404
    assert past_client.get("/api/pastdays", params={"flight_id": "far"}).json()["status"] == "no_weather"


class FakePolicies:
    """Stands in for murphy.policies.Index: records what was searched."""
    def __init__(self):
        self.calls = []

    def search(self, question, carrier, method="embed", k=3):
        from murphy.policies import CONTRACTS
        self.calls.append((question, carrier, method))
        p = {"id": "x:0001", "carrier": "federal", "section": "14 CFR 260.6 Refunds.", "part": "1/1", "text": "Refund it.",
             "url": "https://example.test", "collected": "2026-11-07", "sha256": "ab" * 32, "score": 0.8}
        return {"carrier": carrier, "note": None if carrier in CONTRACTS else f"{carrier}: federal rules only.", "passages": [p]}


@pytest.fixture
def pol_client(tmp_path):
    path = tmp_path / "pol.duckdb"
    con = duckdb.connect(str(path))
    trips = pd.DataFrame([trip("dl", .2), trip("ua", .2) | {"carrier": "UA"}])
    con.execute("CREATE TABLE trips AS SELECT * FROM trips")
    con.close()
    fake = FakePolicies()
    return TestClient(create_app(path, policy_ix=lambda: fake)), fake


def test_policies_search_the_trips_airline_and_trace_the_source(pol_client):
    c, fake = pol_client
    d = c.get("/api/policies", params={"flight_id": "dl", "situation": "cancelled"}).json()
    assert fake.calls[-1][1:] == ("DL", "embed") and "cancelled my flight" in fake.calls[-1][0]
    x = d["passages"][0]
    assert x["section"] and x["url"] and x["collected"] and x["sha256"] == "ab" * 6
    assert d["note"] is None and "decisions" not in d["provenance"]


def test_policies_for_an_airline_not_collected_say_so(pol_client):  # thin evidence
    c, _ = pol_client
    assert "federal rules only" in c.get("/api/policies", params={"flight_id": "ua", "situation": "tarmac"}).json()["note"]


def test_policies_refuse_unknown_situation_flight_and_missing_index(pol_client, tmp_path):
    c, _ = pol_client
    assert c.get("/api/policies", params={"flight_id": "dl", "situation": "weather"}).status_code == 422
    assert c.get("/api/policies", params={"flight_id": "nope", "situation": "cancelled"}).status_code == 404
    path = tmp_path / "pol.duckdb"
    assert TestClient(create_app(path, policy_ix=lambda: None)).get(
        "/api/policies", params={"flight_id": "dl", "situation": "cancelled"}).status_code == 503


def test_the_default_situation_follows_the_advice(client):
    assert get(client, "gone").json()["policy_situation"] == "cancelled"
    assert get(client, "low").json()["policy_situation"] == "delay_refund"


def test_the_three_pages_and_shared_files_are_served(client):
    for path, marker in [("/", 'id="find"'), ("/trip", 'id="assistant"'), ("/about", "About Murphy")]:
        r = client.get(path)
        assert r.status_code == 200 and marker in r.text and 'class="topbar"' in r.text
    assert "--accent" in client.get("/static/app.css").text
    assert "async function get" in client.get("/static/common.js").text
    assert client.get("/static/missing.js").status_code == 404


def test_the_about_page_states_every_limit_the_app_has(client):  # limits moved off the trip page
    from murphy.app.server import LIMITS
    about = client.get("/about").text
    assert all(x in about for x in LIMITS)


# --- The deadline what-if curve ------------------------------------------------------------

def test_every_checked_deadline_step_is_offered(client):
    for d in range(0, 121, 15):
        assert get(client, "high", deadline_min=d).status_code == 200


def test_the_curve_matches_the_chance_at_each_deadline_and_never_falls(client):
    c = get(client, "high").json()["curve"]
    assert c["deadline_min"] == list(range(0, 181, 15)) and c["checked_to"] == 120
    assert all(b >= a for a, b in zip(c["chance"], c["chance"][1:]))
    for i, d in enumerate(c["deadline_min"][:9]):
        assert c["chance"][i] == get(client, "high", deadline_min=d).json()["answer"]["chance"]


def test_the_curve_shows_the_best_alternative_and_the_typical_flight(client):
    c = get(client, "high").json()["curve"]
    assert c["best_alt"]["flight"] == "UA 9" and c["best_alt"]["chance"][4] == pytest.approx(.9)
    assert c["typical"] == [.8] * 13  # the simpler line to compare against


def test_no_best_alternative_curve_when_nothing_is_better(client):  # thin evidence: nothing to compare
    assert get(client, "alone").json()["curve"]["best_alt"] is None
    assert get(client, "low").json()["curve"]["best_alt"] is None


def test_scheduled_utc_times_reach_the_timeline(client):
    d = get(client, "high").json()
    assert d["flight"]["arrives_utc"] == "2025-03-01 15:00:00"
    assert all(a["departs_utc"] == "2025-03-01 13:00:00" and a["schedule_ok"] for a in d["alternatives"])


# --- Human feedback: validate or correct an answer; no Groq call in tests ----------------

def fb(c, verdict, note="", question="chance", **kw):
    return c.post("/api/feedback", json={"flight_id": "F", "question": question, "verdict": verdict, "note": note, **kw})


@pytest.fixture
def feedback_log(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(gr, "FEEDBACK", path)
    return path


def test_looks_right_is_saved_without_an_llm_call(asker, feedback_log, monkeypatch):
    c, _, _ = asker
    monkeypatch.setattr(gr, "llm", lambda msgs, wait=True: pytest.fail("LLM called"))
    assert fb(c, "right").json() == {"status": "saved"}
    assert '"verdict": "right"' in feedback_log.read_text()


def test_something_wrong_returns_one_checked_correction(asker, feedback_log, monkeypatch):
    c, _, _ = asker
    calls = []
    def fake(msgs, wait=True):
        calls.append(msgs)
        return "A 42% chance [E1]." if len(msgs) == 2 else "The evidence says 42%, not 10% [E1]."
    monkeypatch.setattr(gr, "llm", fake)
    d = fb(c, "wrong", note="I think it is 10%").json()
    assert d["status"] == "refined" and d["first"] == "A 42% chance [E1]."
    assert calls[-1][-2] == {"role": "assistant", "content": "A 42% chance [E1]."} and "10%" in calls[-1][-1]["content"]
    assert d["checks"]["invented"] == ["10"]       # a number from the note alone is not evidence
    assert '"refined": true' in feedback_log.read_text()


def test_wrong_needs_a_note_and_the_daily_cap_holds(asker, feedback_log, monkeypatch):
    c, _, _ = asker
    monkeypatch.setattr(gr, "llm", lambda msgs, wait=True: "ok [E1].")
    assert fb(c, "wrong").status_code == 422
    monkeypatch.setattr(gr, "MAX_REFINES_PER_DAY", 1)
    assert fb(c, "wrong", note="no").json()["status"] == "refined"
    assert fb(c, "wrong", note="still no").json()["status"] == "limit"


def test_feedback_at_the_groq_limit_says_so(asker, feedback_log, monkeypatch):
    c, _, _ = asker
    def limited(msgs, wait=True):
        if len(msgs) > 2:
            raise gr.Paused("60")
        return "A 42% chance [E1]."
    monkeypatch.setattr(gr, "llm", limited)
    assert fb(c, "wrong", note="no").json()["status"] == "paused"
