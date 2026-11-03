# 0008 — Forecast design: coverage, delay ranges, training sample

**Date:** 3 Nov 2026
**Status:** accepted by the owner, 3 Nov 2026. Decision 1 was revised and
then reverted the same day (see its history). This is Step 2 of `docs/PLAN.md`.
Built so far: the sample and the climatology baseline (Decision 5).

Given by earlier decisions: per-leg arrival-delay distribution plus
cancellation probability; climatology baseline; Brier score and calibration
(`0003` Decision 1). Model library: scikit-learn 1.9.1, already pinned.

## Decision 1 — Legs with no weather: missing values plus a flag (30 airports)

**Final:** weather stays at `0007`'s 30 airports. Every flight is forecast;
weather inputs are blank where an airport has none, with a "has weather"
flag. scikit-learn's `HistGradientBoosting` handles blanks natively.

**History, all 3 Nov 2026:**

1. *First choice.* Options: (a) blanks plus a flag; (b) borrow the nearest
   of the 30 stations; (c) forecast only flights with weather at both ends.
   **AI recommended (a)**: (b) uses stations that can be hundreds of miles
   away, which is weak evidence presented as data; (c) leaves most
   connections without a forecast. **Owner: accepted.**
2. *Revised.* The owner asked whether to forecast only covered airports, so
   the forecast is consistent with no missing weather, and whether to add
   medium airports. Measured first (BTS Jan 2023 – Aug 2026, duplicates
   excluded), share of flights with weather at both ends:

   | Weather at the top… | Both ends | At least one end |
   |---|---|---|
   | 30 | 33.9% | 95.3% |
   | 50 | 56.3% | 99.3% |
   | 75 | 72.3% | 99.8% |
   | 100 | 82.0% | 99.9% |
   | 150 | 92.0% | 99.9% |

   Of ranks 31–150, all but two returned TAFs for 1–2 Aug 2026 (PBI is
   filed under KPBI, then KDJT after its rename; VPS has none, also none for
   1–7 Mar 2024). **AI recommended** weather at 149 airports and
   forecasting covered flights only. **Owner: accepted.**
3. *Reverted.* The download ran at about one airport a minute, so it needed
   roughly 1.5–2 more hours. **The owner reverted to the 30 airports for
   lack of time.** The partial download (243 files, 24 airports) was
   deleted; the 300 original files still match their manifest hashes.
   Reverting means (a) again, since covered-only at 30 would forecast
   33.9% of flights. The owner asked which to pick. **AI recommended (a):**
   covered-only at 30 leaves two thirds of flights without a forecast;
   95.3% have weather at one end at least; route, carrier, time and
   aircraft history apply everywhere; the output can state "no weather
   data" as evidence; and scoring by coverage will show if (a) is weak.
   **Owner: confirmed (a).**

**Known for a later widening** (not built): HNL, SJU, ANC, OGG, KOA, LIH and
FAI use ICAO station IDs at IEM, not IATA; PBI's ASOS history is under DJT.

## Decision 2 — Delay as probabilities over 15-minute ranges

**Options:** (a) P(cancel) plus probabilities over arrival-delay ranges;
(b) predicted quantiles of delay.

The owner asked why ranges beat quantiles, given how the result is shown to
a user. The answer: both reach the user as one number ("72% chance you make
your connection"), so the question is which produces that number more
reliably.
- Ranges give "chance of more than *k* minutes late" by adding ranges.
  Quantiles must be interpolated, and extrapolated past the top one, which
  is where missed connections are.
- A cancelled flight has no delay, so quantiles need a separate model and a
  rule to combine them. Ranges treat cancellation as one more outcome.
- Brier score and calibration (`0003`) apply range by range. Quantiles need
  pinball loss or CRPS.
- scikit-learn fits one model per quantile, so quantiles can cross. Range
  probabilities always sum to 1.
- Against ranges: edges are arbitrary, and long-delay ranges have few
  flights, so their probabilities can be noisy. The usual remedy is to model
  "chance of more than *k*" for each *k*, which keeps answers ordered.

The first ranges proposed (0–15, 15–30, 30–60, …) could not answer a
45-minute connection exactly. **AI recommended 15-minute steps:** early,
0–15, 15–30, …, 165–180, 180+, plus cancelled and diverted.
**Owner: accepted.**

## Decision 3 — Train on a sample of about 2M legs, at three lead times

The data has 26,332,839 non-duplicate legs (`0007`), and this machine has
16 GB of RAM. The course says to avoid processing the whole dataset to prove
a point.
**Options:** (a) a random sample of about 2M legs from 2023–2025;
(b) every 2023–2025 leg at the 30 airports; (c) everything.
**AI recommended (a).** Each sampled leg is a forecast made at 24 h, 6 h and
2 h before scheduled departure, through `flights_as_of`, `asos_as_of` and
`taf_as_of`. The sample size and the lead times are AI proposals, not
measured optima. **Owner: accepted.**

**Test set:** 2026 flights (Jan–Aug). Skill is reported separately for
flights with weather at both ends, at one end, and at neither, so the value
of weather is visible rather than averaged away.

## Decision 4 — Leakage guards for the training sample

The owner asked whether a random sample leaks. The sample itself does not:
train is 2023–2025, test is 2026, and every input goes through `as_of`.
Three things built on top of it could, so each is a rule:

1. **Historical rates are computed as of the forecast.** Climatology, and
   any route, carrier or airport rate fed to the model, use only flights
   whose outcome was known before that forecast's `as_of`. One rate over all
   of 2023–2025 would put each flight's own outcome, and later ones, into its
   input.
2. **Tuning uses a time block, not a random split.** Fit on 2023–2024, tune
   on 2025. A random split puts flights from the same storm on both sides
   and flatters the score. 2026 is scored once, at the end.
3. **A flight's three lead-time rows stay together.** They share one
   outcome, so they always fall on the same side of any split.

Each gets a test when Step 2 is built. Also noted: when departure and
arrival become known is an assumption (`0006`); its effect can be checked.
**Owner: accepted, 3 Nov 2026.**

## Decision 5 — How the sample and climatology are built

**Status:** built 3 Nov 2026. AI proposals; **owner accepted all four,
3 Nov 2026.**

- **As-of in bulk.** Calling `flights_as_of` once per forecast would mean
  ~7.4M calls. Instead one DuckDB query applies the same rule
  (`arr_known_at < as_of`) with an ASOF JOIN on running class counts. A test
  checks the result equals shares counted through `flights_as_of` itself.
- **Sample by hash.** `hash(flight_id) % 10000 < 942`, so the same legs
  return on every run. Same rate for 2026 test legs.
- **Climatology window: trailing 365 days** of all legs, not only sampled
  ones. Untuned.
- **Two climatologies:** global, and by origin airport shrunk toward global
  with ALPHA = 100 legs. Untuned.
- **No history, no forecast.** A forecast with nothing before it is blank,
  not filled. None of the 2026 test forecasts are blank.

Code: `murphy/forecast/{sample,climatology,score}.py`. Tests:
`tests/test_forecast.py` (12 cases).

**Results on 2026** (1,439,445 forecasts = 479,815 legs × 3 lead times;
`data/forecast/scores_baselines.csv`, gitignored). Lower is better.

| Model | Multiclass Brier | Brier, late* |
|---|---|---|
| uniform (all 16 classes equal) | 0.9375 | 0.5770 |
| climatology, global | 0.6172 | 0.1885 |
| climatology, by origin | 0.6145 | 0.1870 |

\*late = 15+ minutes late, cancelled, or diverted (the DOT definition).
By weather coverage, by-origin climatology scores 0.6434 (both ends),
0.6014 (one end), 0.5889 (neither). Lead time changes nothing, as expected
for climatology. Sample built in 6 s; both climatologies in 5 min 17 s.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Missing weather left blank, with a flag | accepted (final) | native in the pinned library; no proxy evidence |
| Weather at 149 airports; forecast covered flights only | accepted, then reverted by the owner | 92.0% vs 33.9% coverage, but ~2 h more download; no time |
| Score 2026 separately by weather coverage (both / one / neither end) | not asked | shows what weather adds |
| Delay as 15-minute range probabilities plus P(cancel), not quantiles | accepted | owner asked why; direct threshold answers, cancellations, Brier |
| ~2M-leg sample, lead times 24 h / 6 h / 2 h | accepted | fast on 16 GB; avoids processing everything |
| Three leakage guards (as-of rates, time-block tuning, rows kept together) | accepted | owner asked if the random sample leaks; it does not, but these three could |
| Bulk as-of via ASOF JOIN; hash sample; 365-day window; ALPHA = 100 | accepted | one query instead of ~7.4M; equivalence tested |
