"""C1 -- turn a pasted booking confirmation into validated itinerary JSON.

This is the only place an LLM appears in the Challenge 2 slice, and its job is
narrow on purpose: read messy text, return structured fields. It never computes
anything, never estimates a delay, and never fills in a value it could not find.

Two rules are enforced in code rather than trusted to the prompt:

  1. Missing means null. A field the model cannot locate comes back None and is
     surfaced to the traveller to fill in, never guessed.
  2. Parsed values are checked against the real dataset. An airport or carrier
     that does not exist in the 2024 file is reported as a problem, not passed
     silently into retrieval.

The passenger name is deliberately not extracted. The forecast does not need it,
so the parser does not ask for it.

Run:  .venv/bin/python src/murphy/parser.py tests/fixtures/confirmation_synthetic.txt
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from functools import lru_cache
from datetime import date
from pathlib import Path

import duckdb
from pydantic import BaseModel, Field

from murphy.config import ROOT, load_env

MODEL = "gemini-3.6-flash"
PARQUET = ROOT / "data" / "processed" / "flights" / "**" / "*.parquet"

# The free tier is unreliable in two distinct ways:
#   503 -- the model is busy. Frequent, intermittent, and the identical request
#          usually succeeds seconds later.
#   429 -- the per-minute request quota is spent (5 requests/minute).
# Neither is a reason to fail the traveller's request, so we retry hard, and if
# one model stays unavailable we move to the next.
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash"]

# The free tier allows 20 requests per day PER MODEL. That budget is small
# enough that retrying aggressively is self-defeating -- a few retries across a
# few models can spend the whole day's allowance on one confirmation. So: one
# attempt per model, move on quickly, and cache every success to disk so the
# same confirmation is never paid for twice.
RETRIES_PER_MODEL = 1
BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT_MS = 30_000
MAX_WAIT_SECONDS = 35
CACHE_DIR = ROOT / ".cache" / "parses"


class ParserUnavailable(RuntimeError):
    """The model could not be reached. Distinct from a bad parse."""

INSTRUCTIONS = """\
You extract flight details from booking confirmations.

Return one entry in `legs` for every individual flight in the confirmation, in
travel order. A trip with a connection has two legs.

Rules you must follow exactly:

- If a value is not stated in the text, return null for it. Never infer, never
  estimate, never fill in a plausible value. A null is a correct answer; a
  guess is not.
- Use IATA codes for airports (BOS, ATL, MCI). If the text gives only a city
  name and no code, return null rather than converting it yourself.
- `carrier` is the two-character code of the airline whose flight number is
  shown (the marketing carrier). "DL 3391" means carrier "DL", number 3391.
- If the text says a flight is operated by a different airline, put that text in
  `operated_by` verbatim. Do not use it to change `carrier`.
- Times are local clock times at the airport, as printed. Convert to 24-hour
  HH:MM. Do not adjust for time zones.
- Dates are YYYY-MM-DD. A leg inherits the most recent date heading above it.
- Do not extract the passenger name.
"""


class FlightLeg(BaseModel):
    carrier: str | None = Field(None, description="Two-character marketing carrier code, e.g. DL")
    flight_number: int | None = Field(None, description="Flight number as an integer")
    origin: str | None = Field(None, description="Three-letter IATA origin code")
    dest: str | None = Field(None, description="Three-letter IATA destination code")
    departure_date: str | None = Field(None, description="Departure date, YYYY-MM-DD")
    scheduled_departure_local: str | None = Field(None, description="Local departure time, HH:MM")
    scheduled_arrival_local: str | None = Field(None, description="Local arrival time, HH:MM")
    operated_by: str | None = Field(None, description="Operating airline, verbatim, if stated")


class Itinerary(BaseModel):
    legs: list[FlightLeg] = Field(default_factory=list)
    confirmation_code: str | None = Field(None, description="Booking reference, if stated")


def _cache_path(text: str) -> Path:
    digest = hashlib.sha256(text.strip().encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def _is_daily_quota(exc) -> bool:
    """True when a 429 is the per-day allowance rather than the per-minute one."""
    return "PerDay" in str(getattr(exc, "details", "")) or "PerDay" in str(exc)


def _retry_delay(message: str, default: float = 30.0) -> float:
    """Pull the server's suggested retry delay out of a quota error message."""
    match = re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", message)
    return float(match.group(1)) + 1 if match else default


def parse(text: str, model: str | None = None, use_cache: bool = True) -> Itinerary:
    """Send the confirmation to Gemini and get back a validated Itinerary.

    Successful parses are cached on disk by the hash of the input text, so
    re-running the same confirmation costs no quota. Set use_cache=False to
    force a fresh call.
    """
    cache_file = _cache_path(text)
    if use_cache and cache_file.exists():
        return Itinerary.model_validate_json(cache_file.read_text())

    key = load_env("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "No GEMINI_API_KEY found in .env\n"
            "Get one at https://aistudio.google.com -> Get API key, then add:\n"
            "  GEMINI_API_KEY=your-key-here"
        )

    from google import genai
    from google.genai import errors as genai_errors, types as genai_types

    # attempts=1 turns off the SDK's own backoff so a failure surfaces
    # immediately and this function controls all the waiting.
    client = genai.Client(
        api_key=key,
        http_options=genai_types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=genai_types.HttpRetryOptions(attempts=1),
        ),
    )

    models = [model] if model else list(FALLBACK_MODELS)
    if model and model not in FALLBACK_MODELS:
        models += [m for m in FALLBACK_MODELS if m != model]

    last_error = None

    for candidate in models:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=f"{INSTRUCTIONS}\n\nConfirmation:\n\n{text}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": Itinerary,
                        "temperature": 0,
                    },
                )
            except genai_errors.ClientError as exc:
                last_error = exc
                if exc.code == 429:
                    # Per-day allowance is gone for this model: waiting will not
                    # help, so move straight to the next one.
                    if _is_daily_quota(exc):
                        break
                    wait = _retry_delay(str(exc))
                    if attempt < RETRIES_PER_MODEL - 1 and wait <= MAX_WAIT_SECONDS:
                        time.sleep(wait)
                        continue
                    break          # quota spent on this model; try the next one
                raise ParserUnavailable(
                    f"The model rejected the request: {exc}"
                ) from exc
            except Exception as exc:                      # 503, 504, timeouts
                last_error = exc
                if attempt < RETRIES_PER_MODEL - 1:
                    time.sleep(BACKOFF_SECONDS * (2 ** attempt))
                continue

            if response.parsed is None:
                raise ParserUnavailable(
                    "The model did not return usable JSON. Nothing was parsed, and "
                    "no itinerary has been invented in its place."
                )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(response.parsed.model_dump_json(indent=2))
            return response.parsed

    daily = last_error is not None and _is_daily_quota(last_error)
    if daily:
        raise ParserUnavailable(
            "The Gemini free tier allows 20 requests per day per model, and "
            "today's allowance is spent on all of them. Confirmations parsed "
            "earlier today still work -- they are cached and cost nothing. "
            "A fresh confirmation will need to wait for the quota to reset."
        )
    raise ParserUnavailable(
        f"Could not reach any model (tried {', '.join(models)}). The Gemini free "
        f"tier is returning errors right now; this is on their side, not yours. "
        f"Wait a moment and press Parse again.\n\n(last error: {last_error})"
    )


# --------------------------------------------------------------------------
# Validation. The parse is a proposal; these checks decide whether to trust it.
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _known() -> tuple[frozenset[str], frozenset[str]]:
    """Airport and carrier codes present in the data. Cached; the file is static."""
    con = duckdb.connect()
    airports = frozenset(r[0] for r in con.execute(
        f"SELECT DISTINCT origin FROM read_parquet('{PARQUET}', hive_partitioning=true)"
    ).fetchall())
    carriers = frozenset(r[0] for r in con.execute(
        f"SELECT DISTINCT carrier FROM read_parquet('{PARQUET}', hive_partitioning=true)"
    ).fetchall())
    return airports, carriers


def validate(itinerary: Itinerary) -> list[dict]:
    """Return one entry per leg describing what is missing or unusable.

    Missing fields and unknown codes are reported separately: a null means the
    traveller has to supply it, an unknown code means the parse produced
    something the data cannot answer for.
    """
    airports, carriers = _known()
    report = []

    for i, leg in enumerate(itinerary.legs):
        missing, unusable = [], []

        for field_name in ("carrier", "flight_number", "origin", "dest",
                           "departure_date", "scheduled_departure_local"):
            if getattr(leg, field_name) is None:
                missing.append(field_name)

        if leg.origin and leg.origin not in airports:
            unusable.append(f"origin {leg.origin} does not appear in the 2024 data")
        if leg.dest and leg.dest not in airports:
            unusable.append(f"destination {leg.dest} does not appear in the 2024 data")
        if leg.carrier and leg.carrier not in carriers:
            unusable.append(f"carrier {leg.carrier} does not appear in the 2024 data")
        if leg.departure_date:
            try:
                date.fromisoformat(leg.departure_date)
            except ValueError:
                unusable.append(f"departure date {leg.departure_date!r} is not a real date")
        if leg.scheduled_departure_local:
            try:
                h, m = leg.scheduled_departure_local.split(":")
                if not (0 <= int(h) <= 23 and 0 <= int(m) <= 59):
                    raise ValueError
            except ValueError:
                unusable.append(
                    f"departure time {leg.scheduled_departure_local!r} is not a valid HH:MM")

        report.append({
            "leg": i,
            "ready": not missing and not unusable,
            "missing": missing,
            "unusable": unusable,
        })
    return report


def to_query_args(leg: FlightLeg) -> dict | None:
    """Convert a validated leg into the arguments retrieval needs.

    Returns None if the leg is not complete enough. Retrieval matches on month
    and departure hour, so the full date and time are narrowed here rather than
    inside the retrieval layer.
    """
    required = (leg.carrier, leg.origin, leg.dest, leg.departure_date,
                leg.scheduled_departure_local)
    if any(v is None for v in required):
        return None
    return {
        "origin": leg.origin,
        "dest": leg.dest,
        "carrier": leg.carrier,
        "month": date.fromisoformat(leg.departure_date).month,
        "sched_dep_hour": int(leg.scheduled_departure_local.split(":")[0]),
    }


def main():
    ap = argparse.ArgumentParser(description="Parse a booking confirmation into itinerary JSON.")
    ap.add_argument("file", nargs="?", help="path to a confirmation text file; omit to read stdin")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--json", action="store_true", help="print raw JSON only")
    a = ap.parse_args()

    text = Path(a.file).read_text() if a.file else sys.stdin.read()
    try:
        itinerary = parse(text, model=a.model)
    except ParserUnavailable as exc:
        print(f"\nCould not parse this confirmation.\n\n  {exc}\n")
        raise SystemExit(1)

    if not itinerary.legs:
        print("\nNo flights found in this text. Nothing was parsed, and no "
              "itinerary was invented. Check that you pasted a booking "
              "confirmation.")
        return

    if a.json:
        print(itinerary.model_dump_json(indent=2))
        return

    checks = validate(itinerary)
    print(f"\nConfirmation code: {itinerary.confirmation_code or '(not found)'}")
    print(f"Legs parsed: {len(itinerary.legs)}")

    for leg, check in zip(itinerary.legs, checks):
        print(f"\n--- Leg {check['leg'] + 1} " + "-" * 52)
        print(f"  {leg.carrier or '??'} {leg.flight_number or '??'}   "
              f"{leg.origin or '???'} -> {leg.dest or '???'}")
        print(f"  {leg.departure_date or '(no date)'}   "
              f"dep {leg.scheduled_departure_local or '--:--'}   "
              f"arr {leg.scheduled_arrival_local or '--:--'}")
        if leg.operated_by:
            print(f"  operated by: {leg.operated_by}")

        if check["ready"]:
            args = to_query_args(leg)
            print(f"  READY -> retrieval query: {args}")
        else:
            if check["missing"]:
                print(f"  ASK THE TRAVELLER: {', '.join(check['missing'])}")
            for problem in check["unusable"]:
                print(f"  PROBLEM: {problem}")


if __name__ == "__main__":
    main()
