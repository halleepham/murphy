"""Weather in plain words (murphy/wxtext.py): fixed rules, nothing dropped, flags by threshold."""
import numpy as np
import pandas as pd

from murphy import wxtext as wt


def test_codes_become_words_and_unknown_letters_are_kept():
    assert wt.phrase("-FZRAPL") == "light freezing rain and ice pellets"
    assert wt.phrase("VCSH") == "nearby showers"
    assert wt.phrase("-XX") == "light XX"


def test_wind_visibility_and_sky():
    assert wt.wind(310, 24, 39) == "wind from the northwest at 28 mph, gusts to 45 mph"
    assert wt.wind(np.nan, np.nan, np.nan, "FM170000 VRB04KT P6SM FEW250") == "wind variable direction at 5 mph"
    assert wt.wind(0, 0, np.nan) == "calm wind" and wt.wind(np.nan, np.nan, np.nan) == "wind not reported"
    assert wt.visibility(6.01) == "visibility over 6 miles" and wt.visibility(1) == "visibility 1 mile"
    assert wt.sky(["CLR"], [np.nan]) == "clear sky"
    assert wt.sky(["FEW", "OVC"], [500, 1200.0]) == "a few clouds at 500 ft, overcast at 1,200 ft"


def item(**kw):
    return {"codes": [], "skyc": [], "skyl": [], "sknt": None, "gust": None, "vis": None} | kw


def test_flags_follow_the_thresholds_and_not_lookalike_codes():
    assert wt.notable([item(gust=29.0), item(sknt=19.0), item(vis=3.0)]) == []
    assert wt.notable([item(gust=30.0, vis=2.9)]) == ["strong gusts (up to 35 mph)", "poor visibility (down to 2.9 miles)"]
    assert wt.notable([item(skyc=["BKN"], skyl=[900])]) == ["low clouds (down to 900 ft)"]
    assert wt.notable([item(skyc=["SCT"], skyl=[500])]) == []          # scattered is not a ceiling
    assert wt.notable([item(codes=["VCSH", "BR"])]) == []             # showers nearby, mist
    assert wt.notable([item(codes=["-FZRA", "BCFG"])]) == ["freezing rain or drizzle", "fog"]


def taf(airport="ATL"):
    t0 = pd.Timestamp("2025-01-09 18:00")
    return pd.DataFrame({
        "airport": airport, "issued_utc": t0,
        "period_type": ["initial", "TEMPO", "FM"],
        "period_start_utc": [t0, t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=5)],
        "period_end_utc": [pd.NaT, t0 + pd.Timedelta(hours=3), pd.NaT],
        "sknt": [8.0, np.nan, 10.0], "drct": [330.0, np.nan, 90.0], "gust": [0.0, np.nan, 0.0],
        "visibility": [6.01, 2.0, 3.0], "presentwx": ["[]", "['-SN']", "['FZRA']"],
        "skyc": ["['SKC']", "[]", "['OVC']"], "skyl": ["[]", "[]", "[1200]"], "raw": ["", "", ""]}), t0


def test_forecast_periods_near_the_flight_are_marked():
    p, t0 = taf()
    near = [x["near"] for x in wt.taf_lines(p, "America/New_York", t0 + pd.Timedelta(hours=4))]
    assert near == [True, True, True]          # initial runs to the FM at +5 h; TEMPO ends at +3 h
    near = [x["near"] for x in wt.taf_lines(p, "America/New_York", t0 + pd.Timedelta(hours=10))]
    assert near == [False, False, True]
    lines = wt.taf_lines(p, "America/New_York", t0 + pd.Timedelta(hours=10))
    assert lines[1]["when"].startswith("At times") and lines[1]["text"] == "Light snow; visibility 2 miles."


def test_a_flight_past_the_forecast_span_is_not_covered():
    p, t0 = taf()
    assert not any(x["near"] for x in wt.taf_lines(p, None, t0 + pd.Timedelta(hours=32)))   # 30-hour TAF
    p, t0 = taf("DAL")
    assert not any(x["near"] for x in wt.taf_lines(p, None, t0 + pd.Timedelta(hours=26)))   # 24-hour TAF


def test_observation_with_blanks_says_not_reported():
    r = pd.Series({"wxcodes": np.nan, "tmpf": np.nan, "drct": np.nan, "sknt": np.nan, "gust": np.nan, "vsby": np.nan,
                   "obs_utc": pd.Timestamp("2025-01-09 18:52"), **{f"skyc{i}": np.nan for i in range(1, 5)},
                   **{f"skyl{i}": np.nan for i in range(1, 5)}})
    o = wt.obs_text(r, "America/New_York")
    assert o["text"] == "Wind not reported; visibility not reported; sky not reported."
    assert o["when"] == "Observed Thu 1:52 pm"


def test_short_line_keeps_only_what_a_traveler_needs():
    r = pd.Series({"wxcodes": "-SN BR", "tmpf": 33.0, "drct": 310.0, "sknt": 11.0, "gust": np.nan, "vsby": 2.0,
                   "obs_utc": pd.Timestamp("2025-03-16 18:11"), "skyc1": "FEW", "skyc2": "BKN", "skyc3": np.nan, "skyc4": np.nan,
                   "skyl1": 700.0, "skyl2": 1600.0, "skyl3": np.nan, "skyl4": np.nan})
    assert wt.obs_text(r, "America/Chicago")["short"] == "light snow, mist, 33°F, wind 13 mph, cloudy, visibility 2 miles"
