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
import json
import re
import sys
import time
from functools import lru_cache
from datetime import date
from pathlib import Path

import duckdb
from pydantic import BaseModel, Field

from murphy.config import ROOT, load_env

# Two providers, tried in order. Groq is the working path: its free tier allows
# thousands of requests a day. Gemini stays as an automatic backup, but its free
# tier is 20 requests per day PER MODEL, so it cannot carry normal use.
#
# Retrying hard is self-defeating against a daily allowance -- a few retries can
# spend the whole budget on one confirmation -- so each model gets one attempt,
# and every success is cached to disk by input hash.

GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-flash-latest"]
MODEL = GROQ_MODELS[0]

PARQUET = ROOT / "data" / "processed" / "flights" / "**" / "*.parquet"
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


def _schema_hint() -> str:
    """The JSON shape we want, spelled out for providers without schema mode."""
    return json.dumps(Itinerary.model_json_schema(), indent=2)


def _call_groq(text: str, model: str, key: str) -> Itinerary:
    """One Groq completion. Raises on transport or quota problems."""
    from groq import Groq

    client = Groq(api_key=key, timeout=REQUEST_TIMEOUT_MS / 1000, max_retries=0)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": f"{INSTRUCTIONS}\n\nReturn JSON matching this schema "
                        f"exactly:\n{_schema_hint()}"},
            {"role": "user", "content": f"Confirmation:\n\n{text}"},
        ],
    )
    return Itinerary.model_validate_json(completion.choices[0].message.content)


def _call_gemini(text: str, model: str, key: str) -> Itinerary:
    """One Gemini completion. Raises on transport or quota problems."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(
        api_key=key,
        http_options=genai_types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=genai_types.HttpRetryOptions(attempts=1),
        ),
    )
    response = client.models.generate_content(
        model=model,
        contents=f"{INSTRUCTIONS}\n\nConfirmation:\n\n{text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": Itinerary,
            "temperature": 0,
        },
    )
    if response.parsed is None:
        raise ParserUnavailable(
            "The model did not return usable JSON. Nothing was parsed, and no "
            "itinerary has been invented in its place."
        )
    return response.parsed


def parse(text: str, model: str | None = None, use_cache: bool = True) -> Itinerary:
    """Turn confirmation text into a validated Itinerary. See parse_detailed."""
    itinerary, _ = parse_detailed(text, model=model, use_cache=use_cache)
    return itinerary


def parse_detailed(text: str, model: str | None = None,
                   use_cache: bool = True) -> tuple[Itinerary, str]:
    """Parse, and report which provider/model actually produced the result.

    The model that ran is not always the one configured first -- a provider can
    be down or out of quota -- and the app shows it as provenance, so it has to
    be the truth rather than the default.

    Tries Groq, then Gemini. Successful parses are cached on disk by the hash of
    the input text, so re-running the same confirmation costs no quota at all.
    Set use_cache=False to force a fresh call.
    """
    cache_file = _cache_path(text)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return Itinerary.model_validate(cached["itinerary"]), cached["model"]

    groq_key = load_env("GROQ_API_KEY")
    gemini_key = load_env("GEMINI_API_KEY")
    if not groq_key and not gemini_key:
        raise SystemExit(
            "No API key found in .env\n"
            "Get a free Groq key at https://console.groq.com -> API Keys, then add:\n"
            "  GROQ_API_KEY=your-key-here"
        )

    attempts = []
    if groq_key:
        chosen = [model] if model and model in GROQ_MODELS else GROQ_MODELS
        attempts += [("groq", m, groq_key) for m in chosen]
    if gemini_key:
        chosen = [model] if model and model in GEMINI_MODELS else GEMINI_MODELS
        attempts += [("gemini", m, gemini_key) for m in chosen]

    callers = {"groq": _call_groq, "gemini": _call_gemini}
    last_error = None
    daily_quota_hit = False

    for provider, candidate, key in attempts:
        try:
            itinerary = callers[provider](text, candidate, key)
        except ParserUnavailable:
            raise
        except Exception as exc:
            last_error = f"{provider}/{candidate}: {exc}"
            if _is_daily_quota(exc):
                daily_quota_hit = True
            continue

        used = f"{provider}/{candidate}"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(
            {"model": used, "itinerary": itinerary.model_dump()}, indent=2, default=str))
        return itinerary, used

    if daily_quota_hit:
        raise ParserUnavailable(
            "The daily request allowance is spent on every configured model. "
            "Confirmations parsed earlier still work -- they are cached and cost "
            "nothing. A new confirmation will have to wait for the quota to reset."
        )
    raise ParserUnavailable(
        f"Could not reach any model. Tried: "
        f"{', '.join(f'{p}/{m}' for p, m, _ in attempts)}.\n\n"
        f"Last error: {last_error}"
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
