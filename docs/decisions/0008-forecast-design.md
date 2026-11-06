# 0008 — Forecast design: coverage, delay ranges, training sample

**Date:** 3 Nov 2026
**Status:** accepted by the owner, 3 Nov 2026. Decision 1 was revised and
then reverted the same day (see its history). This is Step 2 of `docs/PLAN.md`.
Built so far: the sample, the climatology baseline (Decision 5), and the
gradient-boosted model, scored on 2025 only (Decision 8). 2026 is not scored yet.

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

## Decision 6 — The gradient-boosted model and its inputs

**Status:** AI proposal; **owner accepted, 3 Nov 2026.** Built 5 Nov 2026;
see Decisions 7 and 8.

Measured in the sample before proposing: 356 origin airports, 18 operating
carriers, 6,416 routes; 0.28% of legs have no tail number.

- **Model.** One `HistGradientBoostingClassifier` over the 16 classes, so
  the probabilities sum to 1 and are scored like the baselines. Fit on
  2023–24, a grid of at most 6 settings scored on 2025, and 2026 scored
  once. The library's own early stopping is off: it holds out random rows,
  which breaks Decision 4.2.
- **Inputs, each as of the forecast** (same ASOF-join pattern as
  climatology):
  - A. *Schedule:* carrier (categorical), local hour, weekday, month,
    distance, scheduled duration, lead time.
  - B. *Trailing 365-day rates:* P(late), P(cancel), mean delay by origin,
    destination, carrier, route, carrier × origin; plus the by-origin
    climatology. Airports are rates, not categories: the library allows
    at most 255 categories and there are 356 airports.
  - C. *Current state:* the last 2 h of departures at the origin and
    arrivals at the destination (share late, mean delay, cancellations).
  - D. *The aircraft's prior leg:* scheduled turn time, its delay so far,
    whether it has departed. Blank at the 24 h lead time, because the tail
    is visible only once `sched_dep − 24 h < as_of` (`0006`).
  - E. *Weather:* latest observation at the origin; the TAF for departure
    at the origin and for arrival at the destination; has-weather flags.
- **Ablation:** A; A+B; A+B+C+D; all. Each is scored by weather coverage.
  This is the comparison against simpler models.
- **Calibration:** reliability is measured on 2025 first. It is fixed only
  if it needs fixing, because per-class calibration breaks "sums to 1".

**Timing measured before building:** with 45 synthetic features and 16
classes, 50 boosting rounds took 11.5 s on 200k rows and 27.2 s on 800k
(8 cores).

## Decision 7 — Tune on a quarter of the flights, then refit

**Status:** AI proposal; **owner chose it, 5 Nov 2026**, after the timing
below replaced an estimate. Changes Decision 6's grid and ablation, which
fit on all of 2023–24.

The estimate given to the owner was 30–100 minutes. It stretched a synthetic
timing (45 inputs) about 5× and was wrong. Measured on the real inputs:

| Fit rows | Inputs | Leaves | Rounds | Seconds |
|---|---|---|---|---|
| 3,931,131 | 78 | 31 | 10 | 110 |
| 982,293 | 78 | 31 | 2 / 12 | 8.6 / 29.1 |
| 982,293 | 78 | 15 | 12 | 23.6 |
| 982,293 | 78 | 63 | 12 | 32.8 |

That is about 2 s per round on a quarter and about 9 s on everything.

**Options** (projected from the table, 300 rounds a fit):
(a) grid and ablation on a fixed quarter, best settings refit on everything,
about 2–2.5 h; (b) the same with no refit, about 1.5 h, but the final model
sees a quarter of the data at an unmeasured cost; (c) everything on all rows,
as first designed, about 6–7 h. **AI recommended (a). Owner: accepted.**

The quarter is a fixed hash of `flight_id`, so a flight's three rows stay
together (Decision 4.3). 2025 is still scored in full, and 2026 stays unused.
The refit keeps the number of rounds the quarter chose, so it may stop early
for a model with four times the data. Its 2025 curve will show that.

**What the build added to Decision 6.** The 2 h state also has the number of
flights, since a share from one flight is weak evidence. Prior-leg "whether
departed" is a status: scheduled, departed, arrived, cancelled, diverted.
78 inputs in all: A 7, B 31, C 8, D 3, E 29.

**Checks on the real table before any fit** (`tests/test_features.py` covers
the rules on synthetic data):
- 40 random rows against `flights_as_of`, `asos_as_of` and `taf_as_of`: no
  observation or TAF mismatch; departures in the 2 h window agree within 1
  (the check allowed 1, because `dep_known_at` is not returned).
- 20 random prior legs against `flights_as_of(tail=)`: all statuses agree.
- Single-input AUC against "late" on 2023–24, by lead time, to catch a leak:
  the highest is 0.671 (aircraft's delay so far, 2 h lead). Nothing close to
  1. This screen cannot rule out a subtle leak.
- Prior-leg inputs are 100% blank at 24 h and 0.3% blank at 6 h and 2 h.
  Weather inputs are blank on 35.4% of rows, which matches the 64.6% of
  departures at the 30 airports.

**Odd values seen in the source and kept as they are:** visibility up to 21,005
miles, ceiling up to 256,000 ft, wind up to 170 kt in observations and 210 kt
in TAFs, scheduled duration down to −14 min, and turn times from −474 min to
1,153,247 min (a tail that last appeared over two years earlier). Trees split
on order, so size alone does little harm. They are not cleaned yet.

## Decision 8 — Results on 2025, and calibration

**Status:** results measured 5 Nov 2026. **Owner chose, 5 Nov 2026:** make no
calibration change (options: keep as is; a monthly correction; refit on
2023–25). AI recommended keeping it, since the tested fix did nothing and the
others cannot be checked before 2026. **The owner put scoring 2026 on hold**
while checking the project against its research claim. 2026 has not been
touched.

Files (gitignored): `data/forecast/gbm/*.json` (each fit and its 2025 curve),
`scores_tune.csv`, `calibration_tune.csv`. Code: `murphy/forecast/gbm.py`.

**Grid, on the quarter (982,293 rows), all inputs, 300 rounds.** Multiclass
Brier on 2025 at each model's best checkpoint; lower is better.

| Learning rate | Leaves | Best | At round | At round 300 |
|---|---|---|---|---|
| 0.05 | 15 | 0.5633 | 150 | 0.5660 |
| 0.05 | 31 | 0.5639 | 150 | 0.5640 |
| 0.05 | 63 | **0.5616** | 275 | 0.5616 |
| 0.1 | 15 | 0.5698 | 25 | 0.5968 |
| 0.1 | 31 | 0.5736 | 25 | 0.6319 |
| 0.1 | 63 | 0.5848 | 25 | 0.6866 |

**Refit on all 3,931,131 rows** (0.05, 63 leaves, 275 rounds, 1,931 s): best
at round **75**, 0.5607, then worsening to 0.5681 at 275. Decision 7 expected
the refit might stop too early; the opposite happened, and the rounds the
quarter chose were far too many. The saved model is used at round 75. Scores
that rise with rounds are unexplained. Choosing the round on 2025 handles it,
but also makes 2025 scores somewhat flattering.

**Ablation and baseline on 2025** (2,070,453 forecasts). The ablation fits use
the quarter and settings tuned for all inputs.

| Group | Climatology | A | A+B | A+B+C+D | All | All, refit |
|---|---|---|---|---|---|---|
| Multiclass, all | 0.6026 | 0.5947 | 0.5967 | 0.5678 | 0.5616 | **0.5607** |
| Lead 24 h | 0.6026 | 0.5947 | 0.5967 | 0.5883 | 0.5819 | 0.5808 |
| Lead 6 h | 0.6026 | 0.5947 | 0.5967 | 0.5730 | 0.5663 | 0.5655 |
| Lead 2 h | 0.6026 | 0.5947 | 0.5967 | 0.5421 | 0.5365 | 0.5359 |
| Weather at both ends | 0.6246 | 0.6154 | 0.6185 | 0.5895 | 0.5823 | 0.5807 |
| Weather at one end | 0.5965 | 0.5889 | 0.5908 | 0.5616 | 0.5555 | 0.5547 |
| Weather at neither | 0.5239 | 0.5223 | 0.5171 | 0.4937 | 0.4935 | 0.4957 |
| Brier, late | 0.1793 | 0.1740 | 0.1741 | 0.1508 | 0.1460 | 0.1465 |
| Brier, cancelled | 0.0153 | 0.0153 | 0.0154 | 0.0122 | 0.0120 | 0.0119 |

Read: current state and the prior leg give most of the gain, and more at
shorter lead. Weather adds a smaller gain where it exists, and none where it
does not, as it should. Trailing rates (B) did not help on their own. The
refit is barely better than the quarter fit and slightly worse on "late". No
confidence intervals are computed, so small gaps may be noise.

**Calibration of the refit on 2025, event "late".** Predicted 22.42% overall,
observed 23.61%. Forecasts from 0.2 to 0.9 run low, by 0.02–0.06. Weighted
mean gap 0.0160, about climatology's 0.0157, though the model covers a far
wider range. "Cancelled" has a weighted gap of 0.0015, with most bins within
0.03 and the largest gap 0.055.

| Predicted | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean predicted | 0.079 | 0.145 | 0.243 | 0.344 | 0.445 | 0.546 | 0.646 | 0.751 | 0.849 | 0.938 |
| Observed | 0.068 | 0.145 | 0.264 | 0.383 | 0.500 | 0.607 | 0.707 | 0.796 | 0.904 | 0.919 |

**Diagnosis.** Temperature scaling, which keeps the sum at 1, was fit on
Jan–Jun 2025 and checked on Jul–Dec: T = 0.996, multiclass 0.5632 → 0.5633.
So this is not over- or under-confidence. It looks like a shift in level:
by month, predicted vs observed late is close for Jan–May and for Aug–Sep, but
low for Jun (0.272 / 0.300), Jul (0.287 / 0.309), Oct (0.177 / 0.202), Nov
(0.196 / 0.228) and Dec (0.235 / 0.280). The check script was not kept in the
repo.

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
| One 16-class gradient-boosted model; input groups A–E; ablation; measure calibration before fixing | accepted | probabilities sum to 1; airports as rates (356 > 255 categories); ablation shows what each group adds |
| Grid and ablation on a fixed quarter of flights, refit on all | accepted | measured 9 s a round on all rows; full design projected at 6–7 h against 2–2.5 h |
