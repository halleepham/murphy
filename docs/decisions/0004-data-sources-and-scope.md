# 0004 — Data sources, carriers, and replay window

**Date:** 2 Nov 2026
**Status:** accepted

Research notes, with a source URL for every claim, are in
`private/research/research-data-sources.md`. They are gitignored because they
are working notes, not because they are sensitive. Claims marked "verified"
below were checked directly by the coordinating agent on 2 Nov 2026.

## Timing changed

The owner, 2 Nov 2026: the project ends within weeks, and **the bulk of the
build happens in the next two days.** So we use only the data that exists now.
Nothing that requires collecting data going forward.

## Decision 1 — Carriers

**In scope:** American (AA), Delta (DL), United (UA), Southwest (WN), Alaska
(AS), JetBlue (B6), Frontier (F9), plus the regional partners flying under the
AA, DL, UA and AS brands. Those are assigned by **marketing** carrier, through
BTS's marketing-carrier fields.

**Excluded: Spirit.** The owner's call. Supporting fact: Airways reported Spirit
preparing to cease operations on 2 May 2026
([airwaysmag.com](https://www.airwaysmag.com/new-post/spirit-nears-end-as-rescue-talks-collapse)).
Its status after that is not confirmed.

**Why regional partners are included:** a ticket that says "United" is often
flown by a regional airline. Those aircraft also carry delays down the line
through tail-number rotations.

**Why Frontier stays:** as a low-cost carrier with different policies, it gives
the recommendation engine a real policy contrast. Whether Frontier has
interline (rebook-on-another-airline) agreements is **not yet verified**; check
its contract of carriage.

## Decision 2 — Replay window

Train on **2023–2025**. Test and replay on **2026** months, which BTS has
published through Aug 2026 (verified: BTS release page, entry dated 19 Oct
2026). The model never sees the test period, so the evaluation is fair.

## Decision 3 — Sources

| Need | Source | Note |
|---|---|---|
| Flights and outcomes | BTS Reporting Carrier On-Time Performance | tail numbers, delay causes, cancellations |
| Weather observations and forecasts as issued | Iowa Environmental Mesonet (IEM) ASOS and TAF archives | public domain; TAF query returns the last forecast issued before a time |
| Forecast text (retrieval) | IEM NWS text-product archive | Area Forecast Discussion coverage UNVERIFIED |
| Cost | BTS DB1B, latest Q2 2025 (verified) | historical fare distribution, labeled as an estimate and not a quote |
| Policy | DOT Airline Customer Service Dashboard; contracts of carriage | contract text stays in `private/`, not committed |
| Satisfaction and performance | DOT Air Travel Consumer Report; ACSI | ACSI reuse terms UNVERIFIED |
| Airports | OurAirports | public domain |

**Rejected or unavailable:**
- **Live bookable fares.** Amadeus Self-Service shut down 17 Jul 2026 (verified,
  [PhocusWire](https://www.phocuswire.com/amadeus-shut-down-self-service-apis-portal-developers)).
  No free replacement was found.
- **Historical FAA ground stops and delay programs.** Advisories are kept about
  14 days ([NBAA](https://nbaa.org/aircraft-operations/airspace/tfm/resources/advisories/),
  a secondary source). A forward-looking collector was rejected for the timing
  reason above. Stand-in: BTS NAS-delay minutes from *earlier* hours only, so
  `as_of` is respected.
- **Skytrax-derived review datasets.** The license is unclear.
- **Minimum connection times.** Proprietary (IATA, OAG, Cirium). Murphy uses a
  stated, labeled assumption.

## Decision 4 — Vector retrieval covers three corpora

1. Policy passages (DOT commitments, contracts of carriage).
2. **Historical analog days:** similar past airport-days, retrieved with their
   record IDs as evidence.
3. Weather forecast text, if the IEM archive has it.

Baseline for each: BM25 keyword search.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Major carriers plus regionals; Spirit and Frontier left open | modified: exclude Spirit, keep Frontier | Frontier adds policy contrast |
| Start an FAA advisory collector now | rejected | project ends in weeks; use existing data only |
| Train 2023–25, test 2026 | accepted | honest out-of-time evaluation |
| Three retrieval corpora | accepted | analogs give traceable evidence |
