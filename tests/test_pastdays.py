"""Tests for similar past airport-days.

Unit tests build tiny synthetic TAF periods and corpora; they are fixtures, not
data or results. Real-data tests skip when data/pastdays is not built.

Cases: weather-code words, lowest ceiling and TEMPO handling, the sentence,
period ends, no leakage, most-recent tie-break, no earlier days (thin
evidence, deliberate), an unknown method (deliberate failure), a comparison
with random days (simpler baseline), and the real corpus against
asof.taf_as_of and a direct count of departures.
"""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from murphy import pastdays as pd_
from murphy.pastdays import Index, day_weather, describe, numeric, with_ends, wx_words

T0 = datetime(2025, 1, 10, 0, 0)


def periods(*rows):
    cols = ["period_type", "period_start_utc", "period_end_utc", "sknt", "gust",
            "visibility", "skyc", "skyl", "presentwx", "issued_utc"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_wx_words():
    assert wx_words("-SHRA") == "light showers of rain"
    assert wx_words("VCTS") == "nearby thunderstorms"
    assert wx_words("+TSRA") == "heavy thunderstorms with rain"
    assert wx_words("FZFG") == "freezing fog"
    assert wx_words("XX") == "XX"          # unknown code is kept, not dropped


def test_lowest_ceiling_and_tempo():
    p = periods(
        ("initial", T0, None, 10, None, 6.01, "['FEW', 'BKN']", "[500, 3000]", "['BR']", T0),
        ("FM", T0, None, 25, 35, 1.0, "['OVC']", "[800]", "['-SN']", T0),
        ("TEMPO", T0, T0, 30, 40, 0.25, "['VV']", "[100]", "['+SN', 'BR']", T0),
    )
    w = day_weather(p)
    assert w["ceil_min"] == 800            # FEW 500 is not a ceiling; TEMPO VV 100 is not prevailing
    assert w["vis_min"] == 1.0 and w["wind_max"] == 25 and w["gust_max"] == 35
    assert w["wx"] == ["-SN", "BR"] and w["at_times"] == ["+SN"]


def test_describe():
    w = {"vis_min": 1.0, "ceil_min": 800, "wind_max": 25.0, "gust_max": 35.0,
         "wx": ["-SN"], "at_times": ["+TSRA"]}
    assert describe("ORD", date(2025, 1, 7), w) == (
        "ORD, a Tuesday in January. Light snow. At times heavy thunderstorms with rain. "
        "Low clouds at 800 ft. Poor visibility 1 mile. Strong wind up to 29 mph, gusts 40 mph.")


def test_prevailing_period_ends():
    h = lambda n: T0 + timedelta(hours=n)
    p = with_ends(periods(("initial", h(0), None, 5, None, 6, "[]", "[]", "[]", h(0)),
                          ("FM", h(4), None, 5, None, 6, "[]", "[]", "[]", h(0)),
                          ("TEMPO", h(1), h(3), 5, None, 6, "[]", "[]", "[]", h(0))))
    assert list(p.period_end_utc) == [h(4), h(pd_.TAF_MAX_HORIZON_H), h(3)]


def corpus(days):
    """days: list of (date, snowy, late_share) at airport ORD."""
    rows = []
    for d, snowy, late in days:
        w = {"vis_min": 1.0 if snowy else 6.01, "ceil_min": 800 if snowy else 12000,
             "wind_max": 20.0 if snowy else 8.0, "gust_max": 0.0,
             "wx": ["-SN"] if snowy else [], "at_times": []}
        rows.append({"airport": "ORD", "date": d, "has_taf": True, **w,
                     "text": describe("ORD", d, w), "late_share": late})
    return pd.DataFrame(rows)


def days(n, start=date(2024, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def test_no_leakage():
    ds = days(30)
    ix = Index(corpus([(d, i % 3 == 0, 0.5 if i % 3 == 0 else 0.1) for i, d in enumerate(ds)]))
    q = ix.c.iloc[20]
    for m in ("bm25", "numeric"):
        hit = ix.search("ORD", q.date, q.text, numeric(q), m)
        assert len(hit) == pd_.K and (hit.date < q.date).all()


def test_ties_go_to_the_most_recent_day():
    ds = days(15)
    ix = Index(corpus([(d, False, 0.1) for d in ds]))   # all texts differ only by weekday/month
    q = ix.c.iloc[14]
    hit = ix.search("ORD", q.date, q.text, numeric(q), "numeric", k=2)
    assert list(hit.date) == [ds[7], ds[0]]             # same weekday, newest first


def test_thin_evidence_no_earlier_days():
    ix = Index(corpus([(d, False, 0.1) for d in days(5)]))
    q = ix.c.iloc[0]
    assert ix.search("ORD", q.date, q.text, numeric(q), "bm25").empty
    assert len(ix.search("ORD", ix.c.date.iloc[3], q.text, numeric(q), "bm25")) == 3  # < K


def test_unknown_method_fails():
    ix = Index(corpus([(d, False, 0.1) for d in days(5)]))
    q = ix.c.iloc[4]
    with pytest.raises(ValueError, match="unknown method"):
        ix.search("ORD", q.date, q.text, numeric(q), "llm")


def test_beats_random_days_when_weather_matters():
    ds = days(120)
    rng = np.random.default_rng(0)
    snowy = rng.random(len(ds)) < 0.2
    ix = Index(corpus([(d, s, 0.5 if s else 0.1) for d, s in zip(ds, snowy)]))
    q = ix.c[ix.c.date == ds[-1]].iloc[0].copy()
    q_snow = q.copy()
    for k, v in {"vis_min": 1.0, "ceil_min": 800, "wind_max": 20.0, "wx": ["-SN"]}.items():
        q_snow[k] = v
    q_snow["text"] = describe("ORD", q.date, q_snow)
    for m in ("bm25", "numeric"):
        hit = ix.search("ORD", q.date, q_snow.text, numeric(q_snow), m)
        rand = ix.c.late_share[ix.c.date < q.date].sample(pd_.K, random_state=0).mean()
        assert hit.late_share.mean() == 0.5 > rand


REAL = pd_.OUT / "corpus.parquet"


@pytest.mark.skipif(not (REAL.exists() and any(pd_.FLIGHTS.glob("*.parquet"))), reason="data/pastdays or data/flights not built")
def test_real_corpus_matches_taf_as_of_and_flight_counts():
    import duckdb
    from murphy.datalayer.asof import taf_as_of
    c = pd.read_parquet(REAL).sample(20, random_state=0)
    for r in c.itertuples():
        t = taf_as_of(r.as_of.to_pydatetime().replace(tzinfo=timezone.utc), r.airport)
        assert t.known_at.iloc[0] == r.taf_known_at
    r = c.iloc[0]
    y, m = r.date.year, r.date.month
    n = duckdb.sql(f"""SELECT count(*) FROM read_parquet('{pd_.FLIGHTS}/{y}_{m:02d}.parquet')
                      WHERE origin = '{r.airport}' AND flight_date = DATE '{r.date}'
                        AND duplicate_flag = 'N'""").fetchall()[0][0]
    assert n == r.n_dep


# ---------------------------------------------------------------- one trip (the app)

def fake_taf(day, empty=False):
    """A TAF issued the evening before `day`, light snow all day. Fixture, not data."""
    t = datetime.combine(day, datetime.min.time()) - timedelta(hours=6)
    p = periods(("initial", t, None, 20, None, 1.0, "['OVC']", "[800]", "['-SN']", t))
    p = p.assign(product_id="FAKE", known_at=t)
    f = lambda as_of, airport: p.iloc[[]] if empty else p
    return f


def trip_index(n, snowy=True):
    """n ORD days from 1 Jan 2025, all snowy (close to the fake TAF's light snow) or all calm (far)."""
    c = corpus([(d, snowy, 0.1 + 0.01 * i) for i, d in enumerate(days(n, date(2025, 1, 1)))])
    c["n_dep"], c["n_late"], c["product_id"] = 100, 10, "P"
    return Index(c)  # the app searches with plain numbers: no embeddings


DAY, AS_OF, TZ = date(2025, 1, 20), datetime(2025, 1, 20, 12), "America/Chicago"


def test_trip_answer_uses_only_earlier_days_and_shows_the_range():
    r = pd_.similar(trip_index(30), "ORD", DAY, AS_OF, TZ, fake_taf(DAY))
    assert r["status"] == "ok" and r["answer"]["days"] == pd_.K
    assert all(d["date"] < str(DAY) for d in r["days"])          # never the trip's own day
    shares = [d["late_share"] for d in r["days"]]
    assert r["answer"]["range"] == [min(shares), max(shares)]
    assert r["query"]["text"].startswith("ORD, a Monday in January. Light snow.")


def test_trip_thin_when_no_close_days_shows_no_percent():
    r = pd_.similar(trip_index(30, snowy=False), "ORD", DAY, AS_OF, TZ, fake_taf(DAY))
    assert r["status"] == "thin" and r["answer"] is None and len(r["days"]) == pd_.K


def test_trip_thin_when_too_few_days_or_no_taf():
    assert pd_.similar(trip_index(5), "ORD", DAY, AS_OF, TZ, fake_taf(DAY))["status"] == "thin"
    r = pd_.similar(trip_index(30), "ORD", DAY, AS_OF, TZ, fake_taf(DAY, empty=True))
    assert r["status"] == "thin" and r["query"] is None and "TAF" in r["uncertainty"]["reason"]


def test_trip_outside_the_30_airports_says_so():
    assert pd_.similar(trip_index(30), "ABQ", DAY, AS_OF, None, fake_taf(DAY))["status"] == "no_weather"
