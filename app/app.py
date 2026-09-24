"""Murphy -- will I land on time?

The Challenge 2 vertical slice, end to end:

    traveller pastes a confirmation
      -> LLM parses it to validated JSON
      -> traveller reviews and corrects the parsed fields
      -> SQL retrieval finds comparable historical flights
      -> empirical p10/p50/p90 over those flights, with the evidence shown
      -> alternative routes, with the traveller's own route ranked among them

Run:  .venv/bin/streamlit run app/app.py
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from murphy.parser import (  # noqa: E402
    ParserUnavailable, parse_detailed, to_query_args, validate, Itinerary, FlightLeg,
)
from murphy.retrieval import COMFORTABLE, MINIMUM, Query, retrieve  # noqa: E402
from murphy.graph import MIN_CONNECTION_MIN, SORTS, plan_routes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

st.set_page_config(page_title="Murphy", page_icon="🛫", layout="wide")

EDITABLE = [
    ("carrier", "Airline", "DL"),
    ("flight_number", "Flight no.", "1422"),
    ("origin", "From", "BOS"),
    ("dest", "To", "ATL"),
    ("departure_date", "Departure date", "2026-11-18"),
    ("scheduled_departure_local", "Scheduled departure", "06:00"),
    ("scheduled_arrival_local", "Scheduled arrival", "09:07"),
]


def reset():
    for key in ("legs", "confirmation_code", "parse_error"):
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def cached_routes(origin, dest, month, hour, carrier, hub):
    """Planning scores dozens of legs, so cache it against the trip's identity."""
    booked = (carrier, hub) if carrier else None
    # k=None returns every distinct candidate. The app slices for display, which
    # keeps "your route ranks 14th of 15" true under whichever ordering the
    # traveller picks rather than only the default one.
    return plan_routes(origin, dest, month, hour, k=None,
                       travellers_route=booked, evidence_limit=8)


def hhmm(minutes: int) -> str:
    return f"{minutes // 60}h{minutes % 60:02d}m"


def landing_window(scheduled_arrival, p10, p90):
    """Turn a delay range into clock times, using the traveller's own schedule.

    The only place a timestamp is constructed. Built from the scheduled arrival
    the traveller gave us plus a delay offset in minutes -- never from the
    dataset's broken timestamp columns.
    """
    if not scheduled_arrival:
        return None
    try:
        base = datetime.strptime(scheduled_arrival, "%H:%M")
    except ValueError:
        return None
    return (f"{(base + timedelta(minutes=p10)).strftime('%H:%M')} – "
            f"{(base + timedelta(minutes=p90)).strftime('%H:%M')}")


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
    st.markdown(
        "Paste a booking confirmation. Murphy finds real flights like yours and "
        "shows what actually happened to them."
    )
    st.divider()

    if st.button("Use a sample itinerary", use_container_width=True):
        st.session_state["raw_text"] = (FIXTURES / "confirmation_synthetic.txt").read_text()
        reset()

    raw_text = st.text_area(
        "Booking confirmation", key="raw_text", height=220,
        placeholder="Paste the whole confirmation email here…",
    )

    if st.button("Read my confirmation", type="primary",
                 disabled=not raw_text.strip(), use_container_width=True):
        reset()
        with st.spinner("Reading…"):
            try:
                itinerary, used_model = parse_detailed(raw_text)
            except ParserUnavailable as exc:
                st.session_state["parse_error"] = str(exc)
            else:
                st.session_state["legs"] = [leg.model_dump() for leg in itinerary.legs]
                st.session_state["confirmation_code"] = itinerary.confirmation_code
                st.session_state["used_model"] = used_model

    if err := st.session_state.get("parse_error"):
        st.error(f"Could not read this confirmation.\n\n{err}")

    if st.session_state.get("legs"):
        code = st.session_state.get("confirmation_code")
        st.success(f"{len(st.session_state['legs'])} flight(s) found"
                   + (f" · {code}" if code else ""))

    st.divider()
    st.caption(
        "Every number comes from SQL over real 2024 flight records. The language "
        "model only reads your confirmation — it never produces a figure."
    )


# ============================================================ welcome state

if "legs" not in st.session_state:
    st.header("What this does")
    a, b, c = st.columns(3)
    with a:
        st.subheader("1 · Check the details")
        st.write(
            "An AI reads your confirmation into structured fields. Anything it "
            "cannot find is left blank rather than guessed, and you can correct "
            "any of it."
        )
    with b:
        st.subheader("2 · Will I land on time?")
        st.write(
            "For each flight, the arrival delay of comparable historical flights — "
            "not a single guess, but the range they actually landed in, with the "
            "flights themselves shown."
        )
    with c:
        st.subheader("3 · Better routes")
        st.write(
            "Other ways to reach the same destination, ranked by reliability or "
            "speed, with your own booked route placed among them."
        )

    st.divider()
    st.info("**Start on the left** — paste a confirmation, or use the sample itinerary.")

    st.caption(
        "Built on 6,284,734 US flight records from 2024. Cancelled and diverted "
        "flights are absent from that data, so Murphy cannot speak to cancellation "
        "risk, and every range assumes the flight operates."
    )
    st.stop()

if not st.session_state["legs"]:
    st.warning(
        "No flights found in that text. Nothing was parsed, and no itinerary was "
        "invented. Check that you pasted a booking confirmation."
    )
    st.stop()


# ============================================================ result tabs

tab_check, tab_forecast, tab_routes = st.tabs(
    ["1 · Check the details", "2 · Will I land on time?", "3 · Better routes"]
)

# ---------------------------------------------------------- 1. check details

with tab_check:
    st.caption(
        "Anything the parser could not find is blank — it is never guessed. "
        "Edit any field and press Enter; the other tabs update."
    )
    for i, leg in enumerate(st.session_state["legs"]):
        st.markdown(f"**Flight {i + 1}**")
        if leg.get("operated_by"):
            st.caption(f"Operated by {leg['operated_by']} — the forecast uses the "
                       f"marketing carrier shown below.")
        for row_start in (0, 4):
            cols = st.columns(4)
            for col, (field, label, placeholder) in zip(cols, EDITABLE[row_start:row_start + 4]):
                value = leg.get(field)
                leg[field] = col.text_input(
                    label, value="" if value is None else str(value),
                    key=f"leg{i}_{field}", placeholder=placeholder,
                ).strip() or None

# ------------------------------------------------------------- 2. forecast

with tab_forecast:
    for leg_dict in st.session_state["legs"]:
        leg = clean_leg(leg_dict)
        st.markdown(f"#### {leg.carrier or '??'} {leg.flight_number or '??'}  "
                    f"{leg.origin or '???'} → {leg.dest or '???'}")

        checks = validate(Itinerary(legs=[leg]))[0]
        if checks["missing"]:
            st.info("Still needed before this leg can be forecast: **"
                    + ", ".join(f.replace("_", " ") for f in checks["missing"])
                    + "** — add it in *Check the details*.")
            continue
        if checks["unusable"]:
            for problem in checks["unusable"]:
                st.error(problem)
            continue

        with st.spinner("Finding comparable flights…"):
            result = retrieve(Query(**to_query_args(leg)))

        if not result.ok:
            st.error(f"**Murphy will not forecast this flight.**\n\n{result.message}")
            st.caption("Ladder tried: " + " · ".join(
                f"{t['rung']} = {t['n']}" for t in result.ladder_trace))
            continue

        a, b, c = st.columns(3)
        a.metric("Usually earlier than", f"{result.p10:+d} min")
        b.metric("Typical", f"{result.p50:+d} min")
        c.metric("Occasionally as late as", f"{result.p90:+d} min")

        window = landing_window(leg.scheduled_arrival_local, result.p10, result.p90)
        if window:
            st.markdown(f"Scheduled to land **{leg.scheduled_arrival_local}**. "
                        f"8 in 10 comparable flights landed between **{window}**.")

        {"ok": st.success, "limited": st.warning}.get(
            result.confidence, st.error)(result.message)

        wx = result.weather
        if wx.get("wet"):
            st.caption(f"{wx['wet']} of these {result.n} flights had rain or snow at "
                       f"{leg.origin} on the day they flew.")
        st.caption("This range covers flights that operated. Cancelled and diverted "
                   "flights are not in the dataset, so Murphy cannot speak to "
                   "cancellation risk.")

        with st.expander(f"See the {result.n} flights this is based on"):
            st.dataframe(
                result.evidence, use_container_width=True, hide_index=True,
                column_config={
                    "flight_id": "Record ID", "flight_date": "Date",
                    "arr_delay_min": st.column_config.NumberColumn("Arrived (min)", format="%+d"),
                    "origin_precip_mm": st.column_config.NumberColumn("Rain at origin (mm)", format="%.1f"),
                    "origin_wind_kph": st.column_config.NumberColumn("Wind (km/h)", format="%.0f"),
                },
            )
            if result.n > len(result.evidence):
                st.caption(f"Showing the first {len(result.evidence)} of {result.n}. "
                           f"The range is computed over all {result.n}.")
            if wx.get("comparable"):
                st.markdown("**Weather on the day**")
                st.table({
                    "Conditions at origin": ["Rain or snow", "Dry"],
                    "Flights": [wx["wet"], wx["dry"]],
                    "Typical arrival": [f"{wx['wet_p50']:+d} min", f"{wx['dry_p50']:+d} min"],
                    "Worst 1 in 10": [f"{wx['wet_p90']:+d} min", f"{wx['dry_p90']:+d} min"],
                })
                st.caption(
                    "Observed conditions on the days these flights flew, shown so you "
                    "can see what is inside the range. Murphy has no weather forecast "
                    "for your own flight, and this is a description of these rows, not "
                    "a claim that weather caused the difference — season, time of day "
                    "and traffic all move with it."
                )

        with st.expander("Where this number came from"):
            st.markdown(f"""
**Dataset** — MFDD / Aeolus 2024 flight records (BTS-derived), 6,284,734 rows
after cleaning, 305 airports, 15 carriers.

**Matched on** — {result.match_description}

**Match quality** — {result.match_quality} (a loose match means the carrier or
month had to be dropped to find enough flights, which lowers confidence however
many rows come back)

**Comparable flights found** — {result.n}

**Fallback ladder** — {' · '.join(f"{t['rung']} = {t['n']}" for t in result.ladder_trace)}
(the first rung reaching {MINIMUM} flights answers; {COMFORTABLE}+ is reported
without caveat)

**How the range was computed** — empirical 10th, 50th and 90th percentiles of
observed arrival delay across those rows. No model, no smoothing, no
interpolation.

**About the date** — the evidence is 2024 flights, whatever year you are
flying. Matching uses the month and departure hour, not the calendar year, so a
November 2026 flight is compared against November 2024 flights on the same
route.
""")
            st.code(result.sql, language="sql")
            st.caption(
                f"Parsed by {st.session_state.get('used_model', 'the language model')}, "
                f"which reads the confirmation text only. Every number above is "
                f"computed by SQL over the rows listed — the model never produces a "
                f"figure. Known data limits: March 2024 is absent from the source "
                f"file, and the source's timestamp columns are unusable for "
                f"arithmetic, so all figures are in delay-minutes."
            )

# --------------------------------------------------------- 3. better routes

with tab_routes:
    complete = [leg for leg in map(clean_leg, st.session_state["legs"])
                if to_query_args(leg) is not None]

    if not complete:
        st.info("Fill in the missing flight details first, in *Check the details*.")
    else:
        first, last = complete[0], complete[-1]
        trip_origin, trip_dest = first.origin, last.dest
        booked_hub = first.dest if len(complete) > 1 else None

        st.caption(
            f"Other ways to get from **{trip_origin}** to **{trip_dest}** around the "
            f"same time, built from flights that operated in this month of 2024. "
            f"Connections assume a {MIN_CONNECTION_MIN}-minute minimum transfer. "
            f"Schedules change, so treat these as routings that tend to work rather "
            f"than flights you can book today."
        )

        order = st.radio(
            "Rank by", list(SORTS), horizontal=True,
            help="The same routes, ordered by what matters to you. Reliability counts "
                 "the worst-case journey time and the share of inbound flights that "
                 "arrived too late to connect.",
        )

        if trip_origin == trip_dest:
            st.info("Origin and destination are the same — nothing to compare.")
        else:
            with st.spinner("Searching the flight network…"):
                routes = cached_routes(
                    trip_origin, trip_dest,
                    date.fromisoformat(first.departure_date).month,
                    int(first.scheduled_departure_local.split(":")[0]),
                    first.carrier, booked_hub,
                )

            if not routes:
                st.error(
                    f"**No alternative found with enough evidence to rank.** Murphy "
                    f"could not build a route from {trip_origin} to {trip_dest} that "
                    f"it can stand behind, so it is not offering one."
                )
            else:
                routes = sorted(routes, key=SORTS[order])
                for position, route in enumerate(routes, 1):
                    route._position = position

                shown = routes[:5]
                mine = next((r for r in routes if r.is_travellers_route), None)
                trailing = mine if mine is not None and mine not in shown else None

                if mine is not None:
                    place, total = mine._position, len(routes)
                    if place == 1:
                        st.success(f"**Your route is the best of {total} on "
                                   f"{order.lower()}.** Nothing here beats it.")
                    elif place <= 3:
                        st.info(f"**Your route ranks {place} of {total} on "
                                f"{order.lower()}.** The options above it are close.")
                    else:
                        st.warning(f"**Your route ranks {place} of {total} on "
                                   f"{order.lower()}.** The routes below did better "
                                   f"on the evidence available.")

                for route in shown + ([trailing] if trailing else []):
                    if trailing is not None and route is trailing:
                        st.divider()
                        st.caption("Your own route, for comparison:")
                    tag_mine = " · **your route**" if route.is_travellers_route else ""
                    tags = f" · {', '.join(route.labels)}" if route.labels else ""
                    st.markdown(f"**{route._position}. {route.describe()}**{tag_mine}{tags}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Typically", hhmm(route.typical_total_min))
                    c2.metric("1 in 10 worse than", hhmm(route.tail_total_min))
                    if route.connection_risk is not None:
                        c3.metric("Connection missed", f"{route.connection_risk:.0%}")
                        st.caption(
                            f"{route.layover_min} min in {route.hub}. "
                            f"{route.connection_risk:.0%} of {route.connection_sample} "
                            f"comparable inbound flights arrived too late to make it."
                        )
                    else:
                        c3.metric("Stops", "Nonstop")

                    with st.expander(f"Evidence behind {route.describe()} "
                                     f"(option {route._position})"):
                        for leg in route.legs:
                            st.markdown(
                                f"**{leg.carrier} {leg.origin}→{leg.dest}** "
                                f"{leg.dep_local}–{leg.arr_local} · "
                                f"p50 {leg.p50:+d} min, p90 {leg.p90:+d} min · "
                                f"{leg.evidence_n} comparable flights ({leg.confidence})"
                            )
                            if leg.evidence:
                                st.dataframe(
                                    leg.evidence, use_container_width=True, hide_index=True,
                                    column_config={
                                        "flight_id": "Record ID", "flight_date": "Date",
                                        "arr_delay_min": st.column_config.NumberColumn(
                                            "Arrived (min)", format="%+d"),
                                        "origin_precip_mm": st.column_config.NumberColumn(
                                            "Rain (mm)", format="%.1f"),
                                        "origin_wind_kph": st.column_config.NumberColumn(
                                            "Wind (km/h)", format="%.0f"),
                                    },
                                )
                        st.caption(
                            "These itineraries are built by pairing a real arrival with "
                            "a real departure under the transfer rule. They are feasible "
                            "connections, not fares an airline sells. Cancelled and "
                            "diverted flights are absent from the data, so every route "
                            "is conditional on its flights operating."
                        )
