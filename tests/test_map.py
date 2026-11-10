"""The map's data: as of the forecast hour, thin cases said plainly, GCN order vs link weight."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from murphy import mapdata
from murphy.forecast import rank

BIG = {"AAA": 0, "BBB": 1, "CCC": 2, "DDD": 3}
AS_OF = "2023-01-02 05:30:00"  # hour row 29 of the arrays (they start 2023-01-01 00:00)
H = 29


def scores():
    s = np.full((40, 4), 0.1)
    s[H] = [0.2, 0.3, 0.9, 0.5]   # at the forecast hour CCC is worst, then DDD, then BBB
    s[H + 1] = 99.0               # the hour after: must never appear
    return s


PLACES = pd.DataFrame({"airport": ["AAA", "BBB", "CCC", "DDD", "SSS", "NOC"], "community": [0, 0, 1, 1, 1, 0],
                       "big": [True, True, True, True, False, False], "name": list("abcdef"),
                       "lat": [40.0, 41.0, 30.0, 31.0, 32.0, np.nan], "lon": [-74.0, -75.0, -90.0, -91.0, -92.0, np.nan]})
# AAA's links: heaviest is BBB, then DDD, then CCC; one link stored the other way round (undirected)
LINKS = pd.DataFrame({"a": ["AAA", "AAA", "CCC"], "b": ["BBB", "DDD", "AAA"], "weight": [0.9, 0.6, 0.3], "years": "2023-24"})


def wx(airport, as_of, around=None):
    return {"airport": airport, "limited": airport != "AAA"}


def build(origin="AAA", dest="SSS", as_of=AS_OF, s=None):
    return mapdata.build(origin, dest, as_of, scores() if s is None else s, BIG, PLACES, LINKS, wx)


def test_slider_ends_at_the_forecast_hour_and_never_shows_a_later_one():
    d = build()
    assert d["hours_utc"][-1] == "2023-01-02 05:00:00" and len(d["hours_utc"]) == mapdata.HOURS_BACK + 1
    assert all(v is None or v < 99 for vals in d["gcn"].values() for v in vals)
    assert d["gcn"]["CCC"][-1] == pytest.approx(0.9)


def test_region_risk_is_the_mean_over_its_big_airports():
    r = {g["community"]: g for g in build()["regions"]}
    assert r[0]["risk"][-1] == pytest.approx((0.2 + 0.3) / 2) and r[0]["airports_scored"] == 2 and r[0]["airports"] == 3
    assert r[1]["risk"][-1] == pytest.approx((0.9 + 0.5) / 2)  # SSS is small, so it does not count


def test_gcn_order_differs_from_the_simpler_link_weight_order():
    lk = build()["links"]
    assert {x["airport"] for x in lk["neighbors"]} == {"BBB", "CCC", "DDD"}  # CCC found from the reversed row
    assert lk["top_by_weight"] == ["BBB", "DDD", "CCC"]
    assert lk["top_by_gcn"][-1] == ["CCC", "DDD", "BBB"]
    assert lk["top_by_gcn"][0] == ["BBB", "DDD", "CCC"]  # all tied at 0.1: falls back to link weight


def test_small_origin_is_limited_not_empty_handed():
    lk = build(origin="SSS", dest="AAA")["links"]
    assert lk["limited"] and lk["neighbors"] == [] and "100 busiest" in lk["reason"]


def test_hours_before_the_arrays_are_blank_not_zero():
    d = build(as_of="2023-01-01 03:10:00")
    assert d["gcn"]["AAA"][:21] == [None] * 21 and d["gcn"]["AAA"][21] == pytest.approx(0.1)
    assert d["regions"][0]["risk"][:21] == [None] * 21


def test_airport_without_coordinates_is_left_off_and_counted():
    d = build(dest="NOC")
    assert d["route"]["dest"] is None and "NOC" not in {a["airport"] for a in d["airports"]}
    assert any("lack of coordinates: 1" in x for x in d["uncertainty"]["limits"])


def test_weather_outside_the_30_airports_is_limited():
    w = mapdata.weather_at("SSS", pd.Timestamp(AS_OF))
    assert w["limited"] and "30 large airports" in w["reason"]


def test_hour_rule_matches_the_why_query():
    assert rank.hour_index(AS_OF) == H == rank.hour_index(pd.Timestamp(AS_OF, tz="UTC"))
    assert rank.hour_index(pd.Timestamp("2023-01-02 00:30:00", tz="America/New_York")) == H


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import duckdb
    from murphy.app.server import create_app
    path = tmp_path_factory.mktemp("map") / "t.duckdb"
    con = duckdb.connect(str(path))
    trips = pd.DataFrame({"flight_id": ["f1"], "lead_h": [2], "origin": ["AAA"], "dest": ["SSS"],
                          "as_of_utc": [pd.Timestamp(AS_OF)],
                          "sched_dep_utc": [pd.Timestamp("2023-01-02 07:30")], "sched_arr_utc": [pd.Timestamp("2023-01-02 10:30")]})
    con.execute("CREATE TABLE trips AS SELECT * FROM trips")
    con.close()
    m = {"scores": scores(), "big": BIG, "places": PLACES, "links": LINKS, "weather": wx}
    return lambda files=True: TestClient(create_app(path, map_inputs=lambda: m if files else None))


def test_endpoint_returns_layers_with_provenance(client):
    d = client().get("/api/map", params={"flight_id": "f1"}).json()
    assert d["provenance"]["forecast_hour_utc"] == "2023-01-02 05:00:00" and d["links"]["top_by_gcn"][-1][0] == "CCC"


def test_endpoint_refuses_unknown_flight_and_missing_files(client):
    assert client().get("/api/map", params={"flight_id": "nope"}).status_code == 404
    assert client().get("/api/map", params={"flight_id": "f1", "lead_h": 5}).status_code == 422
    assert client(files=False).get("/api/map", params={"flight_id": "f1"}).status_code == 503


def test_regions_are_named_by_their_busiest_airports_and_hours_are_local():
    d = mapdata.build("AAA", "SSS", AS_OF, scores(), BIG, PLACES.assign(tz="America/New_York"), LINKS, wx,
                      names={0: ["BBB", "AAA", "NOC", "ZZZ"]})
    r = {g["community"]: g["name"] for g in d["regions"]}
    assert r == {0: "BBB, AAA, NOC area", 1: "Region 1"}       # no list for region 1: said plainly
    assert d["hours_local"][-1] == "Mon 12:00 am"              # 05:00 UTC in New York, 2 Jan 2023
    assert build()["hours_local"][-1] == "Mon 5:00 am UTC"      # no time zone known: says UTC
