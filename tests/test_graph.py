"""Tests for the knowledge graph rows and fixed queries. Synthetic data, not results.

Cases: a flight's chance of missing matches the app's rule at each lead; an alternative that
arrives earlier gets that much more slack; a cancelled flight's outcome is known at departure;
blanks and times are cleaned for Neo4j; an unknown query or lead is refused (failure case); no
query before the flight returns an outcome field, and every weather read is limited to as_of
(leakage). The simpler-baseline comparison for this layer is GraphRAG against no retrieval and
BM25, which is tested with GraphRAG.
"""
import re

import numpy as np
import pandas as pd
import pytest

from murphy import graph as kg
from murphy.forecast.score import PCOLS
from murphy.recommend import p_make

T = pd.Timestamp("2025-01-10 12:00")


def probs(*hot):
    """One row of class probabilities: equal weight on the given column indexes."""
    p = np.zeros(len(PCOLS))
    p[list(hot)] = 1 / len(hot)
    return p


def trips():
    rows = []
    for lead, p in ((24, probs(0)), (6, probs(0, 5)), (2, probs(len(PCOLS) - 2))):  # on time; half 60-75; cancelled
        rows.append({"flight_id": "F1", "lead_h": lead, "as_of_utc": T - pd.Timedelta(hours=lead), **dict(zip(PCOLS, p))})
    return pd.DataFrame(rows)


def test_chance_of_missing_matches_the_app_rule_at_each_lead():
    f = kg.forecast_columns(trips()).iloc[0]
    assert f.p_miss_24h == pytest.approx(0.0)
    assert f.p_miss_6h == pytest.approx(0.5)  # 60-75 ends after a 60-minute deadline
    assert f.p_miss_2h == pytest.approx(1.0)
    assert f.as_of_6h == T - pd.Timedelta(hours=6)


def test_an_earlier_alternative_gets_more_slack():
    p = probs(5)  # all on 60-75 late
    alts = pd.DataFrame([{"flight_id": "F1", "alt": a, "lead_h": 6, "alt_lead": 7.0, "arr_ahead_min": ahead,
                          **dict(zip(PCOLS, p))} for a, ahead in (("A0", 0), ("A30", 30))])
    r = kg.alternative_rows(alts).set_index("alt")
    assert r.p_miss["A0"] == pytest.approx(1.0)
    assert r.p_miss["A30"] == pytest.approx(0.0)  # 90 minutes of slack covers 60-75
    assert r.p_miss["A30"] == pytest.approx(1 - p_make(p[None], [90])[0])


def test_cancelled_flight_outcome_known_at_departure():
    f = pd.DataFrame([{"flight_id": "F1", "flight_date": pd.Timestamp("2025-01-10").date(), "origin": "ATL", "dest": "DFW",
                       "marketing_carrier": "AA", "operating_carrier": "AA", "flight_number": 1, "crs_dep_local": "1410",
                       "crs_arr_local": "1549", "sched_dep_utc": T, "sched_arr_utc": T + pd.Timedelta(hours=2),
                       "distance_mi": 731.0, "source_file": "x.zip", "arr_delay_min": np.nan, "cancelled": True,
                       "cancellation_code": "B", "diverted": False, "weather_delay_min": np.nan, "nas_delay_min": np.nan,
                       "carrier_delay_min": np.nan, "late_aircraft_delay_min": np.nan,
                       "arr_known_at": pd.NaT, "dep_known_at": pd.NaT}])
    r = kg.flight_facts(f).iloc[0]
    assert r.outcome_known_at == T and r.outcome_cancelled


def test_records_clean_blanks_and_times_for_neo4j():
    df = pd.DataFrame({"a": [np.nan], "t": [T], "n": [pd.NaT], "p": [np.array([0.25, 0.75])], "k": [np.int64(3)],
                       "c": pd.array([pd.NA], dtype="Int64")})
    r = kg.records(df)[0]
    assert r["a"] is None and r["n"] is None and r["c"] is None
    assert r["t"].tzinfo is not None and r["t"].hour == 12
    assert r["p"] == [0.25, 0.75] and r["k"] == 3 and type(r["k"]) is int


def test_unknown_query_or_lead_is_refused():
    with pytest.raises(ValueError):
        kg.ask(None, "drop_everything", "F1", 24)
    with pytest.raises(ValueError):
        kg.ask(None, "flight", "F1", 12)


def test_no_query_before_the_flight_returns_an_outcome():
    for name, (_, cypher) in kg.QUERIES.items():
        if name != "outcome":
            assert "outcome_" not in cypher, name


def test_every_weather_read_stops_at_as_of():
    for name, (_, cypher) in kg.QUERIES.items():
        for label in re.findall(r"\((\w+):(Metar|Taf)\)", cypher):
            assert f"{label[0]}.known_at <=" in cypher, name
