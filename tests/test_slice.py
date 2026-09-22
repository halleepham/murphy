"""Test cases for the Challenge 2 slice.

Covers the path the traveller actually takes -- paste, parse, correct, retrieve,
inspect -- including the cases where the system is supposed to refuse.

Retrieval, validation and refusal tests run entirely offline against the Parquet
table. Parser tests need an API key and skip cleanly without one, so the suite
is runnable by anyone who clones the repository.

Run:  .venv/bin/pytest -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from murphy.config import load_env                                    # noqa: E402
from murphy.parser import (                                           # noqa: E402
    FlightLeg, Itinerary, parse, to_query_args, validate,
)
from murphy.retrieval import COMFORTABLE, MINIMUM, Query, retrieve    # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

HAS_KEY = bool(load_env("GROQ_API_KEY") or load_env("GEMINI_API_KEY"))
needs_api = pytest.mark.skipif(
    not HAS_KEY,
    reason="no API key in .env -- see README, parser tests skipped",
)

# The retrieval tests query the Parquet table, which is built from a 1.6 GB CSV
# that is not in the repository. Without it they should say so, not fail with a
# DuckDB error a reader has to decode.
HAS_DATA = (ROOT / "data" / "processed" / "flights").exists()
needs_data = pytest.mark.skipif(
    not HAS_DATA,
    reason="no Parquet table -- run src/murphy/build_parquet.py first",
)


# ---------------------------------------------------------------------------
# 1-4. Retrieval: the four evidence bands
# ---------------------------------------------------------------------------

@needs_data
def test_1_exact_match_is_confident():
    """A busy route in its own month answers from an exact match."""
    r = retrieve(Query("BOS", "ATL", "DL", 11, 6))
    assert r.ok
    assert r.rung_index == 0
    assert r.match_quality == "exact"
    assert r.n >= COMFORTABLE
    assert r.confidence == "ok"
    assert r.p10 < r.p50 < r.p90          # a range, not a point
    assert len(r.evidence) > 0
    assert all(e["origin"] == "BOS" and e["dest"] == "ATL" for e in r.evidence)


@needs_data
def test_2_small_sample_is_flagged():
    """Between 10 and 19 flights still answers, but says the sample is small."""
    r = retrieve(Query("MCI", "DEN", "WN", 11, 7))
    assert r.ok
    assert MINIMUM <= r.n < COMFORTABLE
    assert r.confidence == "limited"
    assert "indicative" in r.message


@needs_data
def test_3_missing_month_widens_and_says_so():
    """March is absent from the source data, so the ladder must widen."""
    r = retrieve(Query("BOS", "ATL", "DL", 3, 6))
    assert r.ok
    assert r.rung_index > 0
    assert r.ladder_trace[0]["n"] == 0            # exact match found nothing
    assert r.query["carrier"] == "DL"
    assert "any month" in r.match_description
    # Widening kept the traveller's airline rather than dropping it first.
    assert "any carrier" not in r.match_description


@needs_data
def test_4_no_evidence_refuses_rather_than_guessing():
    """FAILURE CASE. A route with no service must not produce a range."""
    r = retrieve(Query("BOS", "ANC", "DL", 11, 6))
    assert not r.ok
    assert r.confidence == "refused"
    assert r.n == 0
    assert r.p10 is None and r.p50 is None and r.p90 is None
    assert "will not" in r.message
    assert all(step["n"] == 0 for step in r.ladder_trace)   # every rung tried


# ---------------------------------------------------------------------------
# 5. Confidence reflects match quality, not just sample size
# ---------------------------------------------------------------------------

@needs_data
def test_5_large_but_loose_match_lowers_confidence():
    """A big sample of poorly matched flights must not read as high confidence."""
    loose = retrieve(Query("BOS", "HNL", "DL", 11, 6))
    exact = retrieve(Query("BOS", "ATL", "DL", 11, 6))

    assert loose.ok and exact.ok
    assert loose.n > exact.n                  # far more rows...
    assert loose.match_quality == "loose"
    assert loose.confidence == "limited"      # ...yet lower confidence
    assert exact.confidence == "ok"
    assert "loose" in loose.message


# ---------------------------------------------------------------------------
# 6. Weather is reported as evidence, and withheld when it would be noise
# ---------------------------------------------------------------------------

@needs_data
def test_6_weather_split_only_when_both_sides_are_big_enough():
    r = retrieve(Query("BOS", "ATL", "DL", 11, 6))
    wx = r.weather
    assert wx["wet"] + wx["dry"] + wx["unknown"] == r.n
    if wx["comparable"]:
        assert wx["wet"] >= 3 and wx["dry"] >= 3
    else:
        assert wx["wet"] < 3 or wx["dry"] < 3
    assert "origin_precip_mm" in r.evidence[0]


# ---------------------------------------------------------------------------
# 7-8. Validation: the parse is a proposal, not a fact
# ---------------------------------------------------------------------------

@needs_data
def test_7_unknown_codes_are_reported_not_passed_through():
    """FAILURE CASE. Codes absent from the data are flagged, not queried."""
    leg = FlightLeg(carrier="ZZ", flight_number=1, origin="BOS", dest="QQQ",
                    departure_date="2026-11-18", scheduled_departure_local="06:00")
    problems = validate(Itinerary(legs=[leg]))[0]

    assert not problems["ready"]
    assert any("ZZ" in p for p in problems["unusable"])
    assert any("QQQ" in p for p in problems["unusable"])


@needs_data
def test_8_incomplete_leg_cannot_reach_retrieval():
    """A leg missing a required field yields no query at all."""
    leg = FlightLeg(carrier="DL", flight_number=1422, origin=None, dest="ATL",
                    departure_date="2026-11-18", scheduled_departure_local="06:00")
    problems = validate(Itinerary(legs=[leg]))[0]

    assert "origin" in problems["missing"]
    assert to_query_args(leg) is None


# ---------------------------------------------------------------------------
# 9-11. The parser. Needs an API key; skipped without one.
# ---------------------------------------------------------------------------

@needs_api
def test_9_multi_leg_confirmation_parses_correctly():
    """Both legs, and the regional operator does not overwrite the carrier."""
    expected = json.loads((FIXTURES / "expected_synthetic.json").read_text())
    itinerary = parse((FIXTURES / "confirmation_synthetic.txt").read_text())

    assert len(itinerary.legs) == len(expected["legs"]) == 2
    for got, want in zip(itinerary.legs, expected["legs"]):
        assert got.carrier == want["carrier"]
        assert got.flight_number == want["flight_number"]
        assert got.origin == want["origin"]
        assert got.dest == want["dest"]
        assert got.departure_date == want["departure_date"]
        assert got.scheduled_departure_local == want["scheduled_departure_local"]

    # The second leg is flown by Endeavor but ticketed as Delta. The forecast
    # must follow the marketing carrier, which is what the data is keyed on.
    assert itinerary.legs[1].carrier == "DL"
    assert "Endeavor" in (itinerary.legs[1].operated_by or "")


@needs_api
@needs_data
def test_10_missing_values_are_null_never_guessed():
    """FAILURE CASE. An incomplete confirmation yields nulls, not inventions."""
    text = "UNITED AIRLINES\nTrip confirmation\n\nUA 328   Denver to Chicago\n   Depart 7:45 AM\n"
    itinerary = parse(text)

    assert len(itinerary.legs) == 1
    leg = itinerary.legs[0]
    assert leg.carrier == "UA" and leg.flight_number == 328
    # "Denver" and "Chicago" are city names, not IATA codes. Converting them
    # would be a guess, so the parser leaves them null and the UI asks.
    assert leg.origin is None
    assert leg.dest is None
    assert leg.departure_date is None
    assert validate(itinerary)[0]["missing"] == ["origin", "dest", "departure_date"]


@needs_api
def test_11_text_with_no_flights_yields_no_itinerary():
    """FAILURE CASE. Unrelated text must not produce a fabricated trip."""
    itinerary = parse("Thanks for dinner last night, it was great catching up. "
                      "Let me know when you are free again.")
    assert itinerary.legs == []


# ---------------------------------------------------------------------------
# 12. End to end
# ---------------------------------------------------------------------------

@needs_api
@needs_data
def test_12_confirmation_to_range_end_to_end():
    """The whole slice: text in, evidence-backed range out, for every leg."""
    itinerary = parse((FIXTURES / "confirmation_synthetic.txt").read_text())

    for leg in itinerary.legs:
        args = to_query_args(leg)
        assert args is not None
        result = retrieve(Query(**args))

        assert result.ok
        assert result.p10 <= result.p50 <= result.p90
        assert result.n > 0
        assert result.sql.startswith("SELECT")
        assert "data/processed" in result.sql      # relative path, not absolute
        # Every number traces to rows the traveller can look at.
        assert len(result.evidence) > 0
        assert all("flight_id" in row for row in result.evidence)
