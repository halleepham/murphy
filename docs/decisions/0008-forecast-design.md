# 0008 — Forecast design: missing weather, delay bins, training sample

**Date:** 3 Nov 2026
**Status:** accepted by the owner, 3 Nov 2026 (all three choices below).
Nothing is built yet. This is Step 2 of `docs/PLAN.md`.

Given by earlier decisions: per-leg arrival-delay distribution plus
cancellation probability; climatology baseline; Brier score and calibration
(`0003` Decision 1). Model library: scikit-learn 1.9.1, already pinned.

## Decision 1 — Legs with no weather: missing values plus a flag

Weather covers the 30 airports of `0007`, 64.6% of departures.

**Options:** (a) leave weather features missing, add a "has weather" flag;
(b) borrow weather from the nearest of the 30; (c) forecast only legs with
weather at both ends.
**AI recommended (a).** scikit-learn's `HistGradientBoosting` handles
missing values natively, every leg gets a forecast, and skill can be
reported with and without weather. (b) uses stations that can be hundreds
of miles away, which is weak evidence presented as data. (c) leaves most
connections through smaller airports without a forecast.
**Owner: accepted.**

## Decision 2 — Delay as probabilities over bins

**Options:** (a) P(cancel) plus probabilities over arrival-delay bins;
(b) predicted quantiles of delay.
**AI recommended (a).** Brier score and calibration apply bin by bin, as
`0003` planned, and the Step 3 connection math is a sum over bins.
(b) needs other metrics (pinball loss, CRPS), and its tails are hard to get
right. Proposed bins, to be confirmed while building: early, 0–15, 15–30,
30–60, 60–120, 120+ minutes, plus diverted. **Owner: accepted.**

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

**Test set, as the accepted option stated:** 2026 legs (Jan–Aug) at the 30
airports. **Open point:** Decision 1 also gives every other 2026 leg a
forecast. Scoring those too, reported separately, would show skill without
weather. To be raised when Step 2 is scored.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Missing weather left blank, with a flag | accepted | native in the pinned library; no proxy evidence |
| Delay as bin probabilities plus P(cancel) | accepted | Brier and calibration per bin; simple connection math |
| ~2M-leg sample, lead times 24 h / 6 h / 2 h | accepted | fast on 16 GB; avoids processing everything |
