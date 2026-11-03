# 0006 — The flight data slice and the as_of data layer

**Date:** 2 Nov 2026
**Status:** accepted by the owner, 3 Nov 2026 (all four choices below).

Code: `scripts/fetch_bts.py`, `murphy/datalayer/airports.py`, `murphy/datalayer/bts.py`,
`murphy/datalayer/asof.py`. Tests: `tests/test_asof.py`.

## What was built

1. **Fetch.** 44 monthly zips, Jan 2023 – Aug 2026 (the `0004` window), from
   BTS *Marketing Carrier* On-Time Performance. Each is logged in
   `data/raw/bts/MANIFEST.csv` with URL, SHA-256, size and download time.
2. **Slice.** One Parquet file per month in `data/flights/`: the seven `0004`
   marketing networks (AA DL UA WN AS B6 F9, regionals included), all
   airports, 34 columns of the source's 120. Nothing under `data/` is
   committed (`.gitignore`). A licensing decision is still needed first.
3. **Times in UTC.** Scheduled departure is read in the origin's time zone
   and scheduled arrival in the destination's. `CRSElapsedTime` only picks
   the arrival's date. Actual times = scheduled + delay minutes. See
   Decision 4.
4. **`flights_as_of(as_of, start, end, …)`.** Returns legs with outcome
   columns NULL until they were known. Each row carries `flight_id`,
   `source_file` and `as_of_utc` for provenance.

## Facts verified against the data

- Field names, previously UNVERIFIED in the research notes:
  `Marketing_Airline_Network`, `Operating_Airline`, `Tail_Number`. The file
  has 120 columns, including a `Duplicate` flag ("marked Y if the flight is
  swapped based on Form-3A data"). It is kept but not filtered on.
- Jan 2026: 602,953 legs, 359 origin airports. **Spirit (NK) still appears**,
  with 11,915 flights. `0004` excludes it regardless.
- Jan 2023, after the slice: 536,689 legs, all `flight_id`s unique. 15 legs
  (0.003%) have a `CRSElapsedTime` that disagrees with the clock times.
  They are scattered (worst: LBF, 3 of 83), so they are source errors, not
  time-zone errors. A wrong zone would break almost every flight at that
  airport.
- Mar 2023: 128 such legs, 122 of them on the evening of 11 Mar, the night
  before DST began. Example: DL32 DFW→ATL, 17:54 CST → 21:00 EST is 126
  minutes, but `CRSElapsedTime` says 186.
- **Full slice:** 44 months, 26,332,839 legs, 1 Jan 2023 – 31 Aug 2026,
  1.3 GB Parquet (zips 1.4 GB). By marketing network: AA 7,038,324;
  DL 5,763,502; WN 5,177,601; UA 5,155,790; AS 1,553,803; B6 912,168;
  F9 731,651.
- Elapsed-time disagreements are 4–134 legs per month, except Jul 2024
  (311, 299 of them Southwest, varied sizes on many days). Same pattern as
  Jun 2024 (134): schedule noise in the source, not a time-zone error.
- One source row has no flight number (F9 MIA→ATL, 25 Aug 2024). Its ID uses
  `NA`. The ingest now refuses a month with a NULL or duplicate ID.
- **Spirit in the raw files:** 10,469 legs in Apr 2026, 354 in May, 0 in
  Jun and Aug. That fits the early-May shutdown cited in `0004`.
- The first download hung for 30+ minutes on a dropped connection
  (`urlretrieve` has no timeout). The fetcher now times out after 60 s and
  retries 3 times; Jun 2025 needed one retry.

## Decision 1 — Use the Marketing Carrier table, not Reporting Carrier

`0004`'s source table says "Reporting Carrier", but its Decision 1 assigns
regionals by marketing carrier. Only the Marketing Carrier table has that
field (research notes §1b). It also has the tail and delay-cause fields.

## Decision 2 — Time zones from IEM + OurAirports

BTS has no time zones. **Chosen:** IEM ASOS station `tzname`, joined via
OurAirports (IATA → FAA local code / ICAO). Both are `0004` sources and
public domain, and no dependency is added.

Alternatives: `timezonefinder` from lat/lon (a new dependency); a
hand-written table (unsourced); OpenFlights `airports.dat` (ODbL license,
not a `0004` source).

Four overrides, each with its reason in `airports.py`:
- **STT, STX:** IEM says `Atlantic/Bermuda`, which observes DST. The USVI
  does not.
- **PBI:** OurAirports now lists the airport as DJT / KDJT, while BTS still
  codes it PBI.
- **SPN:** there is no IEM station.

Any airport without a zone makes the ingest fail and name the airport.

## Decision 3 — When each piece of information counts as known

Owner's decision, 3 Nov 2026. The AI first built the "first build" column,
then proposed the changes; the owner accepted them.

| Information | First build | Accepted |
|---|---|---|
| Schedule (carrier, flight, airports, times) | always | always |
| Tail number | always | **24 h before scheduled departure** |
| Departure, cancelled = false | actual departure | same |
| Cancellation | scheduled departure | same |
| Arrival delay, diversion | actual arrival | same |
| Delay causes (5 BTS minutes) | actual arrival | **when BTS publishes the month** |

Rules:
- The boundary is strict: an event at exactly `as_of` is not yet known.
- **Publication date:** 00:00 UTC on the 21st of the second month after
  the flight month. Observed releases (research notes §1): Jul 2026 data on
  21 Sep, Aug data on 19 Oct. The rule matches July and is two days late
  (the safe side) for August.
- **Tail filter:** `flights_as_of(..., tail=)` matches only flights whose
  tail is already visible, so a search cannot reveal a hidden tail.
- **Diversion:** known at the later of scheduled arrival and actual
  departure. BTS gives no time.

Why, per option:
- **Causes (3a).** Alternatives: reveal at arrival (optimistic; a live
  status feed never gives causes, so the model could not run on the planned
  live adapter); never use causes. Chosen: publication lag. Same-day risk
  comes from earlier flights' delays and cancellations, which a live feed
  does give. **This revises `0004`'s ground-stop stand-in** from "NAS-delay
  minutes from earlier hours" to "delays and cancellations from earlier
  hours".
- **Tail (3b).** Alternative: always known. Airlines swap aircraft, and the
  24 h window makes a day-out forecast differ from a 2-hours-out one, which
  is the owner's "dynamic" goal (`0003`). **The 24 h is an assumption**, not
  a sourced figure.
- **Cancellation (3c).** BTS has no announcement time; any earlier time
  would be invented. Real notice is often earlier, so this is optimistic
  for a "warn before" claim on cancellations.

**Not modeled:** publication lag for the training history. The fixed split
(train 2023–25, test 2026) covers it. The climatology baseline is fitted on
2023–25 only.

## Decision 4 — Anchor arrival on `CRSArrTime`, not elapsed time

The first build computed arrival as departure + `CRSElapsedTime`. The 11 Mar
2023 legs showed that breaks: BTS measures `ArrDelay` from `CRSArrTime`, so
a wrong elapsed time put actual arrivals an hour off, and the `as_of` mask
with them. Now the arrival clock time is authoritative. Elapsed time chooses
only the date (same day, next day…). The check lists legs where the two
disagree. Regression test: `test_wrong_elapsed_time_does_not_shift_arrival`.

## Tests (`tests/test_asof.py`, 15 cases)

UTC with and without DST; midnight rollover; a wrong elapsed time (real
11 Mar 2023 row); carrier filter; masking before,
during and after a flight; the strict boundary; cancellation hidden until
departure; tail hidden until 24 h out; the tail filter cannot find a hidden
tail; causes hidden until publication; leakage invariance (changing a future outcome leaves the answer
unchanged); **failures:** a naive `as_of` is rejected, and an airport with no
zone stops the ingest; **baseline comparison:** a direct Parquet query shows
2 future arrival delays at a moment where `as_of` shows 0; a real-data check
(IDs unique across all 44 months; Jan 2023 mismatches < 0.01%).

Run: `.venv/bin/python -m pytest -q tests/` → `22 passed` (7 environment,
15 here), 3 Nov 2026.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Marketing Carrier table | accepted | the only table with the regional → brand field |
| IEM + OurAirports time zones, four overrides | accepted | sourced, no new dependency |
| Causes hidden until BTS publishes (revising the first build) | accepted | a live feed has no causes; keeps the live adapter possible |
| Tail visible 24 h before departure (revising the first build) | accepted | aircraft swaps; makes lead time matter |
| Cancellation known at scheduled departure | accepted | no source for an earlier time |
| Download all 44 months, every airport | accepted | tail rotations leave hub airports; ~1.4 GB zips |
