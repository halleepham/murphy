"""The Documented provenance tier: what each airline commits to, and where it says so.

Two sources, both real, neither paraphrased:

  * The **DOT Airline Customer Service Dashboard**, which records what each carrier
    commits to when a delay or cancellation is within its control. Saved by hand to
    docs/challenge-3/sources/ because transportation.gov returns 403 to every
    automated client, browser user-agent included.
  * **Contracts of carriage**, the authoritative text behind those commitments.

Modelled so that a clause is the *commitment*, not the carrier's answer to it. One
PolicyClause per commitment, and PUBLISHES edges only from the airlines that actually
make it -- so "Delta does not offer cash compensation" is the absence of an edge
rather than a property saying No. Asking which airlines offer hotels becomes a
traversal in either direction.

Every clause carries the four filter keys the graph hands a vector retriever next
stage: disruption_type, controllable, threshold_hours, domestic. Without them,
GraphRAG degrades into plain vector search over undifferentiated text.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "docs" / "challenge-3" / "sources" / "dot_dashboard.html"
DASHBOARD_URL = ("https://www.transportation.gov/airconsumer/"
                 "airline-cancellation-delay-dashboard")

# The dashboard names carriers in full; the graph keys them by BTS code.
NAME_TO_CARRIER = {
    "Alaska": "AS", "Allegiant": "G4", "American": "AA", "Delta": "DL",
    "Frontier": "F9", "Hawaiian": "HA", "JetBlue": "B6", "Southwest": "WN",
    "United": "UA",
}

# Every carrier in scope appears on the dashboard. Spirit was the one exception and
# was dropped from scope when it ceased operations on 2 May 2026 -- which is also why
# the dashboard, reflecting carriers that currently fly, had never listed it.
NOT_ON_DASHBOARD = ()

# Short, stable keys for the eight commitments, so a clause can be found by what it
# promises rather than by matching its prose.
COMMITMENT_KEYS = [
    (r"same airline", "rebook_same_airline"),
    (r"partner airline", "rebook_partner_airline"),
    (r"^Meal", "meal"),
    (r"hotel accommodations", "hotel"),
    (r"ground transportation", "ground_transport"),
    (r"Cash compensation", "cash_compensation"),
    (r"Credit or travel voucher", "credit_or_voucher"),
    (r"Frequent flyer miles", "frequent_flyer_miles"),
]


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _commitment_key(text: str) -> str:
    for pattern, key in COMMITMENT_KEYS:
        if re.search(pattern, text, flags=re.I):
            return key
    raise ValueError(f"unrecognised commitment: {text!r}")


def parse_dashboard(path: Path = DASHBOARD) -> tuple[list[dict], list[dict]]:
    """Return (clauses, publishes) from the saved dashboard page."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. transportation.gov blocks automated fetching, so the "
            "page has to be saved from a browser -- see docs/challenge-3/spec_challenge3.md U3."
        )
    source = path.read_text(encoding="utf-8", errors="ignore")
    tables = re.findall(r"<table.*?</table>", source, flags=re.S | re.I)
    if len(tables) != 2:
        raise ValueError(f"expected 2 commitment tables, found {len(tables)}")

    clauses: list[dict] = []
    publishes: list[dict] = []

    for table in tables:
        caption = _text(re.search(r"<caption.*?</caption>", table, flags=re.S | re.I).group())
        disruption = "cancellation" if "cancellation" in caption.lower() else "delay"

        rows = re.findall(r"<tr.*?</tr>", table, flags=re.S | re.I)
        header = [_text(c) for c in
                  re.findall(r"<t[hd].*?</t[hd]>", rows[0], flags=re.S | re.I)]
        airlines = header[1:]

        for row in rows[1:]:
            cells = [_text(c) for c in
                     re.findall(r"<t[hd].*?</t[hd]>", row, flags=re.S | re.I)]
            if not cells:
                continue
            commitment = cells[0]
            key = _commitment_key(commitment)
            clause_id = f"dot-{disruption}-{key}"

            # "3 hours or more" appears in the commitment text itself, which is where
            # the threshold belongs -- it is part of what was promised.
            threshold = 3 if re.search(r"3 hours or more", commitment, flags=re.I) else None

            clauses.append({
                "clause_id": clause_id,
                "name": f"{key} ({disruption})",
                "source": "DOT Airline Customer Service Dashboard",
                "section": caption,
                "commitment": key,
                "text": commitment,
                "url": DASHBOARD_URL,
                "disruption_type": disruption,
                "controllable": True,       # both tables are explicitly "controllable"
                "threshold_hours": threshold,
                "domestic": True,
            })

            for airline, cell in zip(airlines, cells[1:]):
                # Cells repeat the airline name for responsive layout; strip it off.
                answer = cell.replace(airline, "").strip().lower()
                if answer == "yes":
                    publishes.append({
                        "carrier": NAME_TO_CARRIER[airline],
                        "clause_id": clause_id,
                    })

    return clauses, publishes


# ------------------------------------------------------------------ contracts
# The dashboard records what a carrier promises for disruptions within its control.
# The contract of carriage is the authoritative text behind that promise, and it
# supplies the half the dashboard assumes: what counts as *outside* a carrier's
# control in the first place.
#
# Each entry locates its text by an anchor in the PDF rather than carrying a
# transcription, so the clause stored in the graph is what the document says and a
# re-run against a revised contract either finds the span or fails loudly.

CONTRACTS = {
    "DL": {
        "file": "coc_DL.pdf",
        "title": "Delta Domestic General Rules Tariff",
        "url": "https://www.delta.com/us/en/legal/contract-of-carriage-dgr",
    },
    "WN": {
        "file": "coc_WN.pdf",
        "title": "Southwest Airlines Contract of Carriage",
        "url": "https://www.southwest.com/assets/pdfs/corporate-commitments/contract-of-carriage.pdf",
    },
}

CONTRACT_CLAUSES = [
    {
        "clause_id": "coc-DL-force-majeure",
        "carrier": "DL",
        "section": "Force majeure exclusions",
        "anchor": r"Weather conditions or acts of God.{0,700}",
        "disruption_type": "both",
        "controllable": False,
        "threshold_hours": None,
        "domestic": True,
    },
    {
        "clause_id": "coc-DL-amenities-4h",
        "carrier": "DL",
        "section": "Amenities after a 4-hour interruption",
        "anchor": r"when a passenger.s travel is interrupted for more than 4 hours.{0,900}",
        "disruption_type": "both",
        "controllable": True,
        "threshold_hours": 4,
        "domestic": True,
    },
    {
        "clause_id": "coc-WN-section-9b",
        "carrier": "WN",
        "section": "9(b) Cancellations and Significantly Delayed or Changed Flights",
        "anchor": r"In the event Southwest cancels a flight.{0,900}",
        "disruption_type": "both",
        "controllable": True,
        "threshold_hours": None,
        "domestic": True,
    },
]


def parse_contracts(folder: Path = None) -> tuple[list[dict], list[dict]]:
    """Pull the named clauses out of the contract PDFs."""
    from pypdf import PdfReader

    folder = folder or DASHBOARD.parent
    clauses, publishes = [], []

    for spec in CONTRACT_CLAUSES:
        contract = CONTRACTS[spec["carrier"]]
        path = folder / contract["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Download it from {contract['url']}."
            )
        found = None
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "")
            match = re.search(spec["anchor"], text, flags=re.I)
            if match:
                found = (number, match.group(0).strip())
                break
        if not found:
            raise ValueError(
                f"{spec['clause_id']}: anchor not found in {contract['file']}. "
                "The document was probably revised -- re-locate the clause rather "
                "than storing text that is no longer in it."
            )
        page_number, text = found
        clauses.append({
            "clause_id": spec["clause_id"],
            "name": f"{spec['carrier']} {spec['section'].split('(')[0].strip()}",
            "source": contract["title"],
            "section": f"{spec['section']} (p{page_number})",
            "commitment": "contract_terms",
            "text": text,
            "url": contract["url"],
            "disruption_type": spec["disruption_type"],
            "controllable": spec["controllable"],
            "threshold_hours": spec["threshold_hours"],
            "domestic": spec["domestic"],
        })
        publishes.append({"carrier": spec["carrier"], "clause_id": spec["clause_id"]})

    return clauses, publishes


def summary() -> dict:
    clauses, publishes = parse_dashboard()
    contract_clauses, contract_publishes = parse_contracts()
    return {
        "PolicyClause": clauses + contract_clauses,
        "PUBLISHES": publishes + contract_publishes,
    }


def main():
    clauses, publishes = parse_dashboard()
    print(f"{len(clauses)} clauses, {len(publishes)} PUBLISHES edges\n")
    by_carrier: dict[str, int] = {}
    for p in publishes:
        by_carrier[p["carrier"]] = by_carrier.get(p["carrier"], 0) + 1
    for carrier, n in sorted(by_carrier.items(), key=lambda kv: -kv[1]):
        print(f"  {carrier}  {n:>2} commitments")
    for carrier in NOT_ON_DASHBOARD:
        print(f"  {carrier}   0 commitments -- not listed on the dashboard")
    print()
    for c in clauses[:4]:
        print(f"  [{c['clause_id']}] threshold={c['threshold_hours']}")
        print(f"      {c['text'][:92]}")


if __name__ == "__main__":
    main()
