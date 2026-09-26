"""Murphy -- will I land on time?

    traveller gives an itinerary, by pasting a confirmation or by hand
      -> an LLM parses text into validated fields, guessing nothing
      -> the traveller corrects anything wrong
      -> SQL retrieval finds comparable historical flights
      -> a trained model answers the same question a second way
      -> alternative routes, with the traveller's own route ranked among them

Run:  .venv/bin/streamlit run app/app.py
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from murphy.parser import (  # noqa: E402
    ParserUnavailable, parse_detailed, to_query_args, validate, Itinerary, FlightLeg,
)
from murphy.retrieval import COMFORTABLE, MINIMUM, Query, retrieve  # noqa: E402
from murphy.graph import MIN_CONNECTION_MIN, SORTS, plan_routes  # noqa: E402
from murphy import forecast  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

st.set_page_config(page_title="Murphy", page_icon="🛫", layout="wide")
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:420px;max-width:420px;}</style>",
    unsafe_allow_html=True,
)

SAMPLES = {
    "Delta — two legs via Atlanta": "confirmation_synthetic.txt",
    "American — forwarded email": "confirmation_forwarded.txt",
    "Alaska — single leg": "confirmation_alaska.txt",
    "United — missing airports": "confirmation_united_partial.txt",
    "Not a confirmation at all": "not_a_confirmation.txt",
}

EDITABLE = [
    ("carrier", "Airline", "DL"),
    ("flight_number", "Flight no.", "1422"),
    ("origin", "From", "BOS"),
    ("dest", "To", "ATL"),
    ("departure_date", "Date", "2026-11-18"),
    ("scheduled_departure_local", "Departs", "06:00"),
    ("scheduled_arrival_local", "Arrives", "09:07"),
]

EMPTY_LEG = {field: None for field, _, _ in EDITABLE} | {"operated_by": None}


def reset():
    """Clear the itinerary AND the field widgets holding the old values.

    Streamlit keeps every widget's value under its key for the life of the
    session, and a stored value wins over the one passed in. Clearing only the
    itinerary therefore left the previous flight's text in the boxes, which then
    overwrote whatever had just been parsed -- so loading a second sample showed
    results for the first.
    """
    for key in list(st.session_state):
        if key.startswith("leg") or key in ("legs", "confirmation_code",
                                            "parse_error", "used_model"):
            st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def cached_routes(origin, dest, month, hour, carrier, hub):
    booked = (carrier, hub) if carrier else None
    return plan_routes(origin, dest, month, hour, k=None,
                       travellers_route=booked, evidence_limit=8)


def hhmm(minutes: int) -> str:
    return f"{minutes // 60}h{minutes % 60:02d}m"


def landing_window(scheduled_arrival, p10, p90):
    """Clock times for the range, built from the traveller's own schedule."""
    if not scheduled_arrival:
        return None
    try:
        base = datetime.strptime(scheduled_arrival, "%H:%M")
    except ValueError:
        return None
    return (f"{(base + timedelta(minutes=p10)).strftime('%H:%M')} – "
            f"{(base + timedelta(minutes=p90)).strftime('%H:%M')}")


def interval_chart(rows: list[dict]):
    """Both forecasts as intervals on one axis: easier to compare than six numbers."""
    data = alt.Data(values=rows)
    order = [r["method"] for r in rows]
    # Each row gets its own band with generous padding, and every mark is drawn
    # smaller than the band. Without that the tick -- which is taller than the
    # bar so the midpoint stays visible -- reaches into the neighbouring row and
    # the two intervals read as if they overlap.
    y = alt.Y("method:N", title=None, sort=order,
              scale=alt.Scale(paddingInner=0.55, paddingOuter=0.4),
              axis=alt.Axis(labelLimit=320, labelFontSize=13))

    band = alt.Chart(data).mark_bar(height=26, cornerRadius=4, opacity=0.9).encode(
        x=alt.X("p10:Q", title="Minutes against the scheduled arrival  ·  negative is early"),
        x2="p90:Q",
        y=y,
        color=alt.Color("method:N", sort=order, legend=None,
                        scale=alt.Scale(range=["#1f77b4", "#8c6bb1"])),
        tooltip=[alt.Tooltip("method:N", title="Worked out by"),
                 alt.Tooltip("p10:Q", title="1 in 10 land by"),
                 alt.Tooltip("p50:Q", title="Half land by"),
                 alt.Tooltip("p90:Q", title="1 in 10 land later than")],
    )
    mid = alt.Chart(data).mark_tick(thickness=3, size=30, color="white").encode(
        x="p50:Q", y=y)
    on_time = alt.Chart(alt.Data(values=[{"zero": 0}])).mark_rule(
        strokeDash=[4, 4], color="#444").encode(x="zero:Q")
    return (band + mid + on_time).properties(height=170).configure_view(strokeWidth=0)


def clean_leg(leg_dict) -> FlightLeg:
    clean = dict(leg_dict)
    if clean.get("flight_number"):
        try:
            clean["flight_number"] = int(clean["flight_number"])
        except ValueError:
            clean["flight_number"] = None
    return FlightLeg(**clean)


# ============================================================ input sidebar

with st.sidebar:
    st.title("🛫 Murphy")
    st.caption("Will I land on time?")

    how = st.radio("How would you like to start?",
                   ["Paste a confirmation", "Enter flights by hand"],
                   label_visibility="collapsed")

    if how == "Paste a confirmation":
        sample = st.selectbox("Try a sample", list(SAMPLES))
        if st.button("Load sample", use_container_width=True):
            st.session_state["raw_text"] = (FIXTURES / SAMPLES[sample]).read_text()
            reset()

        raw_text = st.text_area("Booking confirmation", key="raw_text", height=170,
                                placeholder="Paste the confirmation email here…")

        if st.button("Read it", type="primary", use_container_width=True,
                     disabled=not raw_text.strip()):
            reset()
            with st.spinner("Reading your confirmation…"):
                try:
                    itinerary, used_model = parse_detailed(raw_text)
                except ParserUnavailable as exc:
                    st.session_state["parse_error"] = str(exc)
                else:
                    st.session_state["legs"] = [l.model_dump() for l in itinerary.legs]
                    st.session_state["confirmation_code"] = itinerary.confirmation_code
                    st.session_state["used_model"] = used_model
    else:
        if st.button("Start a new itinerary", type="primary", use_container_width=True):
            reset()
            st.session_state["legs"] = [dict(EMPTY_LEG)]
            st.session_state["used_model"] = "entered by hand"

    if err := st.session_state.get("parse_error"):
        st.error(err)

    # ---------------------------------------------------- the flight editor
    if st.session_state.get("legs"):
        st.divider()
        code = st.session_state.get("confirmation_code")
        st.subheader(f"Your flights{f' · {code}' if code else ''}")
        st.caption("Blank means it was not found — never guessed. Edit and press Enter.")

        for i, leg in enumerate(st.session_state["legs"]):
            title = (f"Flight {i + 1} · {leg.get('carrier') or '—'} "
                     f"{leg.get('flight_number') or ''} · "
                     f"{leg.get('origin') or '???'} → {leg.get('dest') or '???'}")
            with st.expander(title, expanded=(i == 0)):
                if leg.get("operated_by"):
                    st.caption(f"Operated by {leg['operated_by']}")
                for start in (0, 2, 4, 6):
                    row = EDITABLE[start:start + 2]
                    cols = st.columns(len(row))
                    for col, (field, label, placeholder) in zip(cols, row):
                        value = leg.get(field)
                        leg[field] = col.text_input(
                            label, value="" if value is None else str(value),
                            key=f"leg{i}_{field}", placeholder=placeholder,
                        ).strip() or None

        add, remove = st.columns(2)
        if add.button("＋ Add a flight", use_container_width=True):
            st.session_state["legs"].append(dict(EMPTY_LEG))
            st.rerun()
        if remove.button("− Remove last", use_container_width=True,
                         disabled=len(st.session_state["legs"]) <= 1):
            st.session_state["legs"].pop()
            for field, _, _ in EDITABLE:
                st.session_state.pop(f"leg{len(st.session_state['legs'])}_{field}", None)
            st.rerun()


# ============================================================ welcome state

if "legs" not in st.session_state:
    st.header("What this does")
    a, b, c = st.columns(3)
    a.subheader("1 · Give it a flight")
    a.write("Paste a booking confirmation and an AI reads it into fields, or type "
            "the details yourself. Anything it cannot find is left blank, never "
            "guessed, and you can correct any of it.")
    b.subheader("2 · Will I land on time?")
    b.write("Not one guess, but the range comparable flights actually landed in — "
            "worked out two independent ways, with the flights themselves shown.")
    c.subheader("3 · Better routes")
    c.write("Other ways to reach the same destination, ranked by reliability or "
            "speed, with your own route placed among them.")

    st.divider()
    st.subheader("Two ways of answering, and why you get both")
    left, right = st.columns(2)
    left.markdown("**Retrieval** reads the flights that actually flew your route this "
                  "month at this hour and reports what they did. Every one is listed, "
                  "with its record number, so you can check the answer.")
    right.markdown("**A trained model** — quantile regression fitted on 6.6 million "
                   "flights from 2023. Across 2024 it was **11% more accurate**. It "
                   "cannot show you its reasoning.")
    st.caption("They usually agree. When they do not, that disagreement is worth more "
               "than either number alone.")

    st.divider()
    st.info("**Start on the left.**")
    st.caption("6.9 million US flights from 2024, Bureau of Transportation Statistics, "
               "with NOAA airport weather.")
    st.stop()

if not st.session_state["legs"]:
    st.warning("No flights found in that text. Nothing was parsed and no itinerary "
               "was invented.")
    st.stop()


# ============================================================ result tabs

tab_forecast, tab_routes = st.tabs(["Will I land on time?", "Better routes"])

# -------------------------------------------------------------- 1. forecast

with tab_forecast:
    legs = st.session_state["legs"]
    for i, leg_dict in enumerate(legs):
        if i:
            st.divider()

        leg = clean_leg(leg_dict)
        label = (f"{leg.carrier or '??'} {leg.flight_number or '??'} · "
                 f"{leg.origin or '???'} → {leg.dest or '???'}")
        st.subheader(f"Flight {i + 1} of {len(legs)} — {label}" if len(legs) > 1
                     else label)

        checks = validate(Itinerary(legs=[leg]))[0]
        if checks["missing"]:
            st.info("Needs " + ", ".join(f.replace("_", " ") for f in checks["missing"])
                    + " — fill it in on the left.")
            continue
        if checks["unusable"]:
            for problem in checks["unusable"]:
                st.error(problem)
            continue

        with st.spinner(f"Finding flights comparable to {label}…"):
            result = retrieve(Query(**to_query_args(leg)))

        if not result.ok:
            st.error(f"**Murphy will not forecast this flight.** {result.message}")
            st.caption("Tried: " + " · ".join(
                f"{t['rung']} = {t['n']}" for t in result.ladder_trace))
            continue

        model = None
        if forecast.available():
            with st.spinner("Asking the model…"):
                try:
                    model = forecast.predict(
                        leg.carrier, leg.origin, leg.dest,
                        date.fromisoformat(leg.departure_date).month,
                        date.fromisoformat(leg.departure_date).isoweekday(),
                        int(leg.scheduled_departure_local.split(":")[0]),
                    )
                except Exception:
                    model = None

        if model:
            st.altair_chart(
                interval_chart([
                    {"method": f"Retrieval — {result.n} real flights",
                     "p10": result.p10, "p50": result.p50, "p90": result.p90},
                    {"method": "Model — 6.6 million flights",
                     "p10": model["p10"], "p50": model["p50"], "p90": model["p90"]},
                ]),
                use_container_width=True,
            )
            left, right = st.columns(2)
            left.metric("Retrieval says", f"{result.p50:+d} min")
            left.caption(f"8 in 10 landed between {result.p10:+d} and {result.p90:+d} · "
                         f"from {result.n} flights you can read below")
            right.metric("The model says", f"{model['p50']:+d} min")
            right.caption(f"8 in 10 expected between {model['p10']:+d} and "
                          f"{model['p90']:+d} · assumes typical weather")

            spread = abs(model["p50"] - result.p50)
            if spread >= 10:
                st.warning(f"**They disagree by {spread} minutes.** Retrieval reports "
                           f"what happened to flights like yours; the model generalises "
                           f"across millions. Trust the one you can check.")
            else:
                st.success(f"**Both agree to within {spread} minutes.**")
        else:
            a, b, c = st.columns(3)
            a.metric("1 in 10 land by", f"{result.p10:+d} min")
            b.metric("Half land by", f"{result.p50:+d} min")
            c.metric("1 in 10 land later than", f"{result.p90:+d} min")

        window = landing_window(leg.scheduled_arrival_local, result.p10, result.p90)
        if window:
            st.markdown(f"Scheduled to land **{leg.scheduled_arrival_local}**. "
                        f"8 in 10 comparable flights landed **{window}**.")

        {"ok": st.success, "limited": st.warning}.get(
            result.confidence, st.error)(result.message)

        cx = result.cancellation
        if cx and cx["cancelled"]:
            st.caption(f"Also: {cx['cancelled']} of {cx['scheduled']} comparable flights "
                       f"({cx['rate']:.1%}) were cancelled and never flew. They are not "
                       f"in the range above.")

        with st.expander(f"The {result.n} flights behind this"):
            st.dataframe(
                result.evidence, use_container_width=True, hide_index=True,
                column_config={
                    "flight_id": "Record ID", "flight_date": "Date",
                    "arr_delay_min": st.column_config.NumberColumn("Arrived (min)", format="%+d"),
                    "origin_precip_mm": st.column_config.NumberColumn("Rain (mm)", format="%.1f"),
                    "origin_wind_kph": st.column_config.NumberColumn("Wind (km/h)", format="%.0f"),
                },
            )
            wx = result.weather
            if wx.get("comparable"):
                st.markdown("**Weather on the day**")
                st.table({
                    "At origin": ["Rain or snow", "Dry"],
                    "Flights": [wx["wet"], wx["dry"]],
                    "Typical": [f"{wx['wet_p50']:+d} min", f"{wx['dry_p50']:+d} min"],
                    "Worst 1 in 10": [f"{wx['wet_p90']:+d} min", f"{wx['dry_p90']:+d} min"],
                })
                st.caption("Conditions on the days these flights flew. A description of "
                           "these rows, not a claim that weather caused the difference.")

        with st.expander("Where these numbers came from"):
            source = st.session_state.get("used_model", "the language model")
            st.markdown(f"""
**Data** — US Bureau of Transportation Statistics On-Time Performance 2024 with
NOAA airport observations. 6,971,908 flights, 305 airports, 15 carriers,
including cancellations.

**Matched on** — {result.match_description} · match quality **{result.match_quality}**

**Comparable flights** — {result.n} · first rung reaching {MINIMUM} answers,
{COMFORTABLE}+ reported without caveat

**Fallback** — {' · '.join(f"{t['rung']} = {t['n']}" for t in result.ladder_trace)}

**Range** — empirical 10th, 50th and 90th percentiles of observed arrival delay.
No smoothing, no interpolation.

**Year** — matching uses month and departure hour, not the calendar year, so a
2026 flight is compared against 2024 flights on the same route.
""")
            st.code(result.sql, language="sql")
            st.caption(
                ("Entered by hand. " if source == "entered by hand"
                 else f"Parsed by {source}, which reads text only. ")
                + "Every number above is computed by SQL over the rows listed."
            )

# ---------------------------------------------------------- 2. better routes

with tab_routes:
    complete = [leg for leg in map(clean_leg, st.session_state["legs"])
                if to_query_args(leg) is not None]

    if not complete:
        st.info("Fill in the flight details on the left first.")
    else:
        first, last = complete[0], complete[-1]
        trip_origin, trip_dest = first.origin, last.dest
        booked_hub = first.dest if len(complete) > 1 else None

        st.subheader(f"Other ways to get {trip_origin} → {trip_dest}")
        st.caption(f"Built from flights that operated in this month of 2024, assuming a "
                   f"{MIN_CONNECTION_MIN}-minute minimum transfer. Schedules change, so "
                   f"treat these as routings that tend to work.")

        order = st.radio("Rank by", list(SORTS), horizontal=True)

        if trip_origin == trip_dest:
            st.info("Origin and destination are the same.")
        else:
            with st.spinner(f"Searching routes from {trip_origin} to {trip_dest}…"):
                routes = cached_routes(
                    trip_origin, trip_dest,
                    date.fromisoformat(first.departure_date).month,
                    int(first.scheduled_departure_local.split(":")[0]),
                    first.carrier, booked_hub,
                )

            if not routes:
                st.error(f"**No alternative Murphy can stand behind.** It could not build "
                         f"a route from {trip_origin} to {trip_dest} with enough evidence "
                         f"to rank.")
            else:
                routes = sorted(routes, key=SORTS[order])
                for position, route in enumerate(routes, 1):
                    route._position = position

                shown = routes[:5]
                mine = next((r for r in routes if r.is_travellers_route), None)
                trailing = mine if mine is not None and mine not in shown else None

                if mine is not None:
                    place, total = mine._position, len(routes)
                    message = (f"**Your route ranks {place} of {total} on "
                               f"{order.lower()}.**")
                    if place == 1:
                        st.success(message + " Nothing here beats it.")
                    elif place <= 3:
                        st.info(message)
                    else:
                        st.warning(message + " The routes below did better.")

                for route in shown + ([trailing] if trailing else []):
                    if trailing is not None and route is trailing:
                        st.divider()
                        st.caption("Your own route:")
                    tag = " · **yours**" if route.is_travellers_route else ""
                    tags = f" · {', '.join(route.labels)}" if route.labels else ""
                    st.markdown(f"**{route._position}. {route.describe()}**{tag}{tags}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Typically", hhmm(route.typical_total_min))
                    c2.metric("1 in 10 worse than", hhmm(route.tail_total_min))
                    if route.connection_risk is not None:
                        c3.metric("Connection missed", f"{route.connection_risk:.0%}")
                        st.caption(f"{route.layover_min} min in {route.hub} · "
                                   f"{route.connection_risk:.0%} of "
                                   f"{route.connection_sample} comparable inbound "
                                   f"flights arrived too late")
                    else:
                        c3.metric("Stops", "Nonstop")

                    with st.expander(f"Evidence · option {route._position}"):
                        for leg in route.legs:
                            st.markdown(f"**{leg.carrier} {leg.origin}→{leg.dest}** "
                                        f"{leg.dep_local}–{leg.arr_local} · "
                                        f"p50 {leg.p50:+d}, p90 {leg.p90:+d} · "
                                        f"{leg.evidence_n} flights ({leg.confidence})")
                            if leg.evidence:
                                st.dataframe(
                                    leg.evidence, use_container_width=True, hide_index=True,
                                    column_config={
                                        "flight_id": "Record ID", "flight_date": "Date",
                                        "arr_delay_min": st.column_config.NumberColumn(
                                            "Arrived (min)", format="%+d"),
                                    },
                                )
                        st.caption("Connections pair a real arrival with a real departure "
                                   "under the transfer rule. Feasible, not fares an "
                                   "airline sells.")
