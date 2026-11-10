import pandas as pd
import pytest
from murphy.cost import MIN_TICKETS, estimated_fare

FARES = pd.DataFrame([
    ("LGA", "ORD", "DL", 100, 250.0),
    ("LGA", "ORD", "F9", MIN_TICKETS - 1, 90.0),
    ("LGA", "ORD", "UA", MIN_TICKETS, 270.0),
    ("LGA", "ORD", None, 500, 240.0),
    ("AAA", "BBB", "DL", 3, 400.0),
    ("AAA", "BBB", None, 3, 400.0),
], columns=["origin", "dest", "carrier", "tickets", "fare"])


def test_uses_the_route_and_airline_and_says_where_it_came_from():
    got = estimated_fare(FARES, "LGA", "ORD", "DL")
    assert got == {"fare": 250.0, "basis": "route and airline", "tickets": 100, "source": "BTS DB1BMarket 2025 Q2"}


def test_exactly_the_minimum_is_enough():
    assert estimated_fare(FARES, "LGA", "ORD", "UA")["fare"] == 270.0


@pytest.mark.parametrize("carrier", ["F9", "WN"])  # too thin; not on the route at all
def test_falls_back_to_the_route(carrier):
    got = estimated_fare(FARES, "LGA", "ORD", carrier)
    assert (got["fare"], got["basis"]) == (240.0, "route, all airlines")


def test_thin_route_gives_no_number():
    got = estimated_fare(FARES, "AAA", "BBB", "DL")
    assert (got["fare"], got["basis"]) == (None, "no estimate")


def test_unknown_route_gives_no_number():
    assert estimated_fare(FARES, "ZZZ", "YYY", "DL")["fare"] is None


def test_direction_matters():
    assert estimated_fare(FARES, "ORD", "LGA", "DL")["fare"] is None
