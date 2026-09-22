"""Murphy -- will I land on time?

The Challenge 2 vertical slice, end to end:

    traveller pastes a confirmation
      -> LLM parses it to validated JSON
      -> traveller reviews and corrects the parsed fields
      -> SQL retrieval finds comparable historical flights
      -> empirical p10/p50/p90 over those flights, with the evidence shown
      -> traveller corrects something and re-runs

Run:  .venv/bin/streamlit run app/app.py
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from murphy.parser import (  # noqa: E402
    ParserUnavailable, parse, to_query_args, validate, Itinerary, FlightLeg, MODEL,
)
from murphy.retrieval import COMFORTABLE, MINIMUM, Query, retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

st.set_page_config(page_title="Murphy", page_icon="🛫", layout="centered")

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


def landing_window(scheduled_arrival: str | None, p10: int, p90: int) -> str | None:
    """Turn a delay range into clock times, using the traveller's own schedule.

    This is the only place a timestamp is constructed. It is built from the
    scheduled arrival the traveller gave us plus a delay offset in minutes --
    never from the dataset's broken timestamp columns.
    """
    if not scheduled_arrival:
        return None
    try:
        base = datetime.strptime(scheduled_arrival, "%H:%M")
    except ValueError:
        return None
    early = (base + timedelta(minutes=p10)).strftime("%H:%M")
    late = (base + timedelta(minutes=p90)).strftime("%H:%M")
    return f"{early} – {late}"


# ---------------------------------------------------------------- 1. input

st.title("🛫 Murphy")
st.caption("Paste a booking confirmation. Get an arrival range built from real "
           "historical flights — and see exactly which ones.")

with st.expander("Start from a sample confirmation", expanded=False):
    st.caption("Useful for trying the app without digging out your own booking.")
    if st.button("Load the sample Delta itinerary"):
        st.session_state["raw_text"] = (FIXTURES / "confirmation_synthetic.txt").read_text()
        reset()

raw_text = st.text_area(
    "Booking confirmation",
    key="raw_text",
    height=200,
    placeholder="Paste the whole confirmation email here…",
)

if st.button("Parse confirmation", type="primary", disabled=not raw_text.strip()):
    reset()
    with st.spinner("Reading the confirmation…"):
        try:
            itinerary = parse(raw_text)
        except ParserUnavailable as exc:
            st.session_state["parse_error"] = str(exc)
        else:
            st.session_state["legs"] = [leg.model_dump() for leg in itinerary.legs]
            st.session_state["confirmation_code"] = itinerary.confirmation_code

if err := st.session_state.get("parse_error"):
    st.error(f"Could not read this confirmation.\n\n{err}")

if "legs" in st.session_state and not st.session_state["legs"]:
    st.warning("No flights found in that text. Nothing was parsed, and no itinerary "
               "was invented. Check that you pasted a booking confirmation.")

# ------------------------------------------------- 2. review and correct

if st.session_state.get("legs"):
    st.divider()
    st.subheader("Check what was read")
    code = st.session_state.get("confirmation_code")
    st.caption(
        (f"Confirmation {code}. " if code else "")
        + "Anything the parser could not find is blank — it is never guessed. "
          "Correct any field and the forecast below updates."
    )

    for i, leg in enumerate(st.session_state["legs"]):
        st.markdown(f"**Flight {i + 1}**")
        for row_start in (0, 4):
            row = EDITABLE[row_start:row_start + 4]
            cols = st.columns(4)
            for col, (field, label, placeholder) in zip(cols, row):
                value = leg.get(field)
                leg[field] = col.text_input(
                    label,
                    value="" if value is None else str(value),
                    key=f"leg{i}_{field}",
                    placeholder=placeholder,
                ).strip() or None
        if leg.get("operated_by"):
            st.caption(f"Operated by {leg['operated_by']} — the forecast uses the "
                       f"marketing carrier shown above.")

# ---------------------------------------------------------- 3. forecast

    st.divider()
    st.subheader("Will I land on time?")

    for i, leg_dict in enumerate(st.session_state["legs"]):
        clean = dict(leg_dict)
        if clean.get("flight_number"):
            try:
                clean["flight_number"] = int(clean["flight_number"])
            except ValueError:
                clean["flight_number"] = None
        leg = FlightLeg(**clean)

        label = (f"{leg.carrier or '??'} {leg.flight_number or '??'}  "
                 f"{leg.origin or '???'} → {leg.dest or '???'}")
        st.markdown(f"#### {label}")

        checks = validate(Itinerary(legs=[leg]))[0]
        if checks["missing"]:
            st.info("Still needed before this leg can be forecast: **"
                    + ", ".join(f.replace("_", " ") for f in checks["missing"]) + "**")
            continue
        if checks["unusable"]:
            for problem in checks["unusable"]:
                st.error(problem)
            continue

        args = to_query_args(leg)
        with st.spinner("Finding comparable flights…"):
            result = retrieve(Query(**args))

        if not result.ok:
            st.error(f"**Murphy will not forecast this flight.**\n\n{result.message}")
            st.caption("Ladder tried: " + " · ".join(
                f"{t['rung']} = {t['n']}" for t in result.ladder_trace))
            continue

        window = landing_window(leg.scheduled_arrival_local, result.p10, result.p90)
        a, b, c = st.columns(3)
        a.metric("Usually earlier than", f"{result.p10:+d} min")
        b.metric("Typical", f"{result.p50:+d} min")
        c.metric("Occasionally as late as", f"{result.p90:+d} min")

        if window:
            st.markdown(f"Scheduled to land **{leg.scheduled_arrival_local}**. "
                        f"8 in 10 comparable flights landed between **{window}**.")

        if result.confidence == "ok":
            st.success(result.message)
        elif result.confidence == "limited":
            st.warning(result.message)
        else:
            st.error(result.message)

        st.caption("This range covers flights that operated. Cancelled and diverted "
                   "flights are not in the dataset, so Murphy cannot speak to "
                   "cancellation risk.")

        with st.expander(f"See the {result.n} flights this is based on"):
            st.dataframe(
                result.evidence,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "flight_id": "Record ID",
                    "flight_date": "Date",
                    "arr_delay_min": st.column_config.NumberColumn(
                        "Arrived (min)", format="%+d"),
                },
            )
            if result.n > len(result.evidence):
                st.caption(f"Showing the first {len(result.evidence)} of {result.n}. "
                           f"The range is computed over all {result.n}.")

        with st.expander("Where this number came from"):
            st.markdown(f"""
**Dataset** — MFDD / Aeolus 2024 flight records (BTS-derived), 6,284,734 rows
after cleaning, 305 airports, 15 carriers.

**Matched on** — {result.match_description}

**Comparable flights found** — {result.n}

**Fallback ladder** — {' · '.join(f"{t['rung']} = {t['n']}" for t in result.ladder_trace)}
(the first rung reaching {MINIMUM} flights answers; {COMFORTABLE}+ is reported
without caveat)

**How the range was computed** — empirical 10th, 50th and 90th percentiles of
observed arrival delay across those rows. No model, no smoothing, no
interpolation.
""")
            st.code(result.sql, language="sql")
            st.caption(
                f"Parsing by {MODEL}, which reads the confirmation text only. Every "
                f"number above is computed by SQL over the rows listed — the model "
                f"never produces a figure. Known data limits: March 2024 is absent "
                f"from the source file, and the source's timestamp columns are "
                f"unusable for arithmetic, so all figures are in delay-minutes."
            )
