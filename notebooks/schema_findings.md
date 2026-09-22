# Schema findings — 2024 flight data

What `01_schema_check.py` found in `data/raw/flight_with_weather_2024.csv`.
Verified September 21, 2026. Re-run the script to reproduce every number here.

Source: [MFDD multi-modal flight delay dataset](https://www.kaggle.com/datasets/flnny123/mfddmulti-modal-flight-delay-dataset)
(Kaggle), 2024 file, downloaded unmodified.

## Shape

| | |
|---|---|
| Rows | 6,284,841 |
| Columns | 34 |
| Airports | 305 |
| Carriers | 15 |
| Duplicate flight keys | 0 |

Carriers are real IATA codes: WN (1.28M), DL (915K), AA (885K), UA (681K), OO,
YX, MQ, NK, AS, B6, OH, F9, 9E, HA, G4.

Nulls are confined to weather: 513 rows missing origin weather, 750 missing
destination weather. Every other column is complete.

## March 2024 is missing

335 distinct dates across 11 months. January has 31 days, February 29, then
April. There is no partial March — the month is absent entirely.

This is a hole in the retrieval match key, which filters on month. A March query
returns zero comparables and falls straight to the fallback ladder.

It is also a discrepancy against the published dataset worth noting in the
report, since the Human Design lists "my copy matches Aeolus's published
statistics" as a success criterion for A1.

## The time columns are broken

The project plan assumed BTS-style HHMM integers. They are not: Kaggle already
converted them to `TIMESTAMP`. The conversion was done badly, in two ways.

**1. Rollover was never applied.** Every time column carries `FL_DATE`'s calendar
date, arrivals included. Zero rows have an arrival dated to the following day. So
a red-eye scheduled 23:45 → 06:30 is stored as landing seventeen hours before it
took off.

Scheduled arrival precedes scheduled departure in **213,203 rows (3.39%)**.

**2. The times are local, with no timezone column.** Comparing clock duration to
the stored `CRS_ELAPSED_TIME`:

| clock − scheduled | rows | share |
|---|---|---|
| 0 | 3,144,978 | 50.04% |
| −60 | 980,908 | 15.61% |
| +60 | 972,191 | 15.47% |
| −120 | 325,591 | 5.18% |
| +120 | 291,704 | 4.64% |
| −180 | 210,845 | 3.35% |
| +180 | 144,832 | 2.30% |
| ≈ −1440 | ~200,000 | the missing rollover |

Only half the file agrees. The gap is a whole number of hours in 6,284,734 of
6,284,841 rows — that is the timezone offset. **107 rows are off by something
other than a whole hour and remain unexplained.**

### What to do about it

Do not do arithmetic on these timestamps. Work in delay-minutes instead:
`ARR_DELAY` is already the answer, in minutes, on the correct basis.

`ARR_DELAY` agrees with naive clock subtraction in 97.86% of rows, and the
disagreement is concentrated in exactly the rollover cases — so the label came
through from BTS intact and the timestamps are the derived, broken artifact.

This simplifies the Phase 1 path considerably: retrieval matches on
route/carrier/month/hour-bin and computes quantiles over `ARR_DELAY`, with no
cross-timezone arithmetic anywhere. Timestamps only reappear when rendering a
landing time for the traveler, which is their own scheduled arrival plus a
delay offset.

### Arrival delay distribution

| min | p10 | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| −126 | −23 | **−6** | **+43** | +214 | 3803 | 7.11 |

The median flight arrives six minutes early.

## Cancelled and diverted flights were already removed

There is no `CANCELLED` or `DIVERTED` column, and there are zero nulls in
`DEP_TIME`, `ARR_TIME` and `ARR_DELAY`. Had those flights been present they would
appear as null actual times. They were stripped upstream.

Two consequences, both of which belong in the app, not just the README:

- The app cannot say anything about cancellation risk.
- Every range it shows is **conditional on the flight operating at all**.

## Comparable-flight counts

Match key: route + carrier + month + 2-hour scheduled-departure bin. Across all
358,426 groups in the file:

| | |
|---|---|
| median group | **17 flights** |
| p10 / p90 | 2 / 31 |
| largest | 136 |
| groups with fewer than 10 | 36.1% |
| **flights landing in such a group** | **8.2%** |

That split is the useful part. A third of groups are too thin, but they are thin
precisely because few flights use them. A typical traveler's flight lands in a
group of 17–31.

Spot checks for October, 06:00–07:59 departures:

| route | year | October | + hour bin |
|---|---|---|---|
| ATL→LAX DL | 3,650 | 348 | 31 |
| BOS→ATL DL | 3,309 | 323 | 31 |
| MCI→DEN WN | 2,208 | 209 | 27 |
| MCI→MDW WN | 1,945 | 198 | 19 |

The ceiling of 31 is a once-daily flight across 31 days, so the hour bin is
barely constraining anything on these routes.

**Read on the match key: usable.** The thin-evidence path fires on roughly 8% of
queries — often enough to demo honestly, rare enough not to dominate.

## Columns that carry no information

- `FLIGHTS` is always 1.
- `ORIGIN_INDEX` / `DEST_INDEX` are integer encodings of `ORIGIN` / `DEST`. No
  airport maps to more than one index. Redundant.
- `MONTH` and `DAY_OF_MONTH` agree with `FL_DATE` in every row.

## Outcome columns

Labels, never model inputs: `ARR_DELAY`, `DEP_DELAY`, `DEP_TIME`, `ARR_TIME`,
`TAXI_OUT`, `TAXI_IN`, `WHEELS_OFF`, `WHEELS_ON`, `ACTUAL_ELAPSED_TIME`,
`AIR_TIME`.

There are no delay-cause columns and no tail number in this file, so the Phase 2
aircraft-rotation work (A3) needs the raw BTS join as planned.
