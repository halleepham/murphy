# 0013 — The keep-or-change rule: options from a 2025 replay

Date: 6 Nov 2026. Model: `final_leads`. Code: `murphy/recommend.py`,
`scripts/replay_cutoffs.py`. Results (gitignored): `data/forecast/alternatives/replay/`.
2025 only; 2026 stays sealed.

## How the replay works

A trip is one of 690,151 sampled 2025 flights, forecast at 24, 6 and 2 h before departure
(2,070,453 forecasts, so a flight can count up to three times). An alternative is on the
same route, leaves at least 1 h after the forecast and within 24 h, is scheduled to arrive
by the deadline, and is not already known to be cancelled. Chance of making it comes from
the outcome classes, rounded down to a 15-minute edge. The traveler is assumed to take
the recommendation.

**Assumptions and limits:** there are no real deadlines, so these are set at 30, 60 and
120 minutes after scheduled arrival. Past schedules do not show seats, so an alternative
may not have been bookable. Connections, nearby airports and cost are left out. Cutoffs
chosen here are chosen on 2025. Results are for forecasts, not real dollars or real
travelers.

## Finding 1 — What the required gain does

The gain, how much an alternative must raise the chance, barely matters between 10 and 20
points. At a 60-minute deadline and 40% risk, it gives 33,294, 32,603 and 31,834 changes for
10, 15 and 20 points, with net benefit 8.36, 8.31 and 8.23 per 1,000 forecasts. Larger gains
were not tried. The table below uses 15.

## Finding 2 — The chosen flight is usually much earlier

Picking the highest chance tends to pick a far earlier flight. At 50% risk, 60-minute
deadline: the median change leaves 9.9 h earlier from a 24 h forecast and 3.3 h earlier
from a 6 h one. From 2 h it cannot leave more than 1 h earlier. That explains the high
success rates, and many travelers could not do it.

## Finding 3 — The options

60-minute deadline, 15-point gain. Rates are shares of changes. Net is trips rescued minus
trips made worse, per 1,000 forecasts, with a 95% interval resampling whole days.

**How the alternative is picked, at 50% risk:**

| Pick | Changes | Rescued | Needless | Made worse | Both missed | Net | Median leaves earlier by | Leaves 2 h+ earlier | Flagged, no change offered |
|---|---|---|---|---|---|---|---|---|---|
| Highest chance, any time | 21,005 | 60.2% | 25.4% | 1.6% | 12.8% | 5.95 (5.43 to 6.42) | 2.3 h | 54.4% | 57.5% |
| Highest chance, leaving at most 2 h earlier | 14,829 | 55.8% | 23.2% | 2.2% | 18.8% | 3.84 (3.58 to 4.07) | 0.75 h | 2.5% | 70.0% |
| Latest flight that clears the gain | 21,005 | 52.8% | 24.3% | 2.7% | 20.2% | 5.09 (4.69 to 5.42) | 0.9 h | 30.2% | 57.5% |

**How high the risk must be, leaving at most 2 h earlier (an AI assumption):**

| Risk of missing | Flagged per 1,000 | Changes per 1,000 | Rescued | Needless | Made worse | Both missed | Net |
|---|---|---|---|---|---|---|---|
| 30% | 54.5 | 15.6 | 42.7% | 40.9% | 2.8% | 13.6% | 6.23 (5.79 to 6.63) |
| 40% | 34.7 | 10.2 | 50.2% | 30.7% | 2.6% | 16.5% | 4.85 (4.53 to 5.14) |
| 50% | 23.9 | 7.2 | 55.8% | 23.2% | 2.2% | 18.8% | 3.84 (3.58 to 4.07) |

"Needless" means the original flight would have made it too, and the alternative did. A
needless change costs money but does not lose the deadline.

## Finding 4 — Forecasts run low on risky flights

This checks a limit in `0011`. Flagged at 50%, predicted miss averages 70.3% and the true rate
is 74.7%. The gap is 1.0 to 5.3 points depending on lead and cutoff, and mostly larger than the
average across all flights.

## Finding 5 — Climatology is not a usable baseline here

By-origin climatology almost never reaches these cutoffs: 398 changes at most, at a
30-minute deadline and 30% risk, and none at 60 or 120 minutes. So "better than climatology"
would be trivially true for this rule. A simple baseline that actually acts is needed.

## Decision 1 — The cutoffs

**Human design:** the owner required that murphy change a flight only when really needed
(`0010` Decision 2), and asked for the options laid out rather than choosing without them.

**AI recommended, owner accepted, 6 Nov 2026:**

| Setting | Chosen | Why | Rejected, and why |
|---|---|---|---|
| Risk of missing | 50% | Changes only when really needed; 23% of changes are needless at 50% against 41% at 30%; reads simply as "more likely than not" | 30% and 40%: higher net benefit (6.23 and 4.85 against 3.84 per 1,000) but more needless, costly changes |
| Gain | 15 points | 10 to 20 points barely changes the results (Finding 1) | 10 and 20: no meaningful difference. Larger gains not tried |
| Alternative marked best | Highest chance, leaving no earlier than the traveler says, default 2 h | Realistic: the median change leaves 0.75 h earlier. The traveler controls the limit, and earlier flights are still listed | Any time: more rescued, but typically leaves 10 h earlier from a 24 h forecast. Latest flight: smallest change of plan, but 30% still leave 2 h or more earlier |
| Low forecasts on risky flights | No correction; show the track record | The user sees how such forecasts really turned out, and it adds no second adjustment fit on 2025 | Correcting now: not ruled out later if it is shown to help |

**Costs accepted:** net benefit falls from 5.95 to 3.84 per 1,000 against the most aggressive
pick, and 70% of flagged trips are offered no change. The app must say so plainly: high risk,
no good alternative within the time allowed. All of this was chosen on 2025 with invented
deadlines. The 2 h default is an AI assumption.

**Still to propose:** what "too uncertain to say" means, and a simple baseline that acts.

## Decision 2 — Only flights that are still ahead

**The owner asked** whether results take the time of looking into account, since showing
flights that have already left makes no sense.

**Built:** yes, in `scripts/replay_cutoffs.py`. An alternative must leave at least 1 h after the
forecast (an AI assumption for time to rebook) and within 24 h, and must not already be known to
be cancelled. Inputs use only what was known at that moment.

**Owner, 6 Nov 2026:** the app must keep this. **Not built:** the app itself. Forecasts are
precomputed for chosen moments (`0011` Decision 2), so it must use the latest moment at or before
the time being replayed and show which one it used. **Known gap:** the current flight's own
cancellation is not a model input, so a flight already known to be cancelled is not handled
directly. The app needs a check for it.

## Finding 4 detail — Risky flights by forecast time

Risky means flagged: predicted chance of missing at or above the cutoff. Here the cutoff is 50%
and the deadline 60 minutes after scheduled arrival, so missing is arriving 60+ minutes late, cancelled or
diverted. Flagged forecasts are 49,439 of 2,070,453, or 2.4%.

| Forecast made | Flagged | Predicted miss | True miss |
|---|---|---|---|
| 2 | 30,625 | 71.6% | 77.0% |
| 6 | 16,166 | 69.8% | 73.0% |
| 24 | 2,648 | 57.6% | 58.6% |
| all | 49,439 | 70.3% | 74.7% |

Read: the warnings are real, not exaggerated, but some flights with a true risk just above 50%
read below it and are not flagged.

## Appendix — All results

The raw files are gitignored, so every figure is recorded here. Keeping the original flight makes
the deadline in 83.60% of trips at 30 minutes, 90.06% at 60 and 94.82% at 120.

**A. All cutoffs, picking the highest chance at any time.** 2,070,453 forecasts. Net is per
1,000 forecasts, with a 95% interval resampling whole days. Climatology rows are omitted where it made
no changes: at every 40% and 50% setting, and at every 60- and 120-minute setting.

| Deadline | Scorer | Risk | Gain | Changes | Per 1,000 | Current missed | Alternative made | Rescued | Needless | Made worse | Both missed | Net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | model | 30% | 10 | 135,935 | 65.65 | 51.9% | 93.6% | 63,393 | 63,901 | 1,530 | 7,111 | 29.88 (27.36 to 32.26) |
| 30 min | model | 30% | 15 | 133,879 | 64.66 | 51.8% | 94.1% | 62,844 | 63,167 | 1,362 | 6,506 | 29.69 (27.20 to 32.08) |
| 30 min | model | 30% | 20 | 131,105 | 63.32 | 51.8% | 94.6% | 61,988 | 62,063 | 1,193 | 5,861 | 29.36 (26.90 to 31.71) |
| 30 min | model | 40% | 10 | 75,424 | 36.43 | 62.6% | 90.9% | 41,296 | 27,297 | 935 | 5,896 | 19.49 (17.75 to 21.09) |
| 30 min | model | 40% | 15 | 74,360 | 35.91 | 62.5% | 91.6% | 41,040 | 27,050 | 849 | 5,421 | 19.41 (17.68 to 21.00) |
| 30 min | model | 40% | 20 | 73,123 | 35.32 | 62.4% | 92.2% | 40,664 | 26,753 | 763 | 4,943 | 19.27 (17.57 to 20.84) |
| 30 min | model | 50% | 10 | 44,229 | 21.36 | 72.5% | 87.8% | 27,225 | 11,599 | 557 | 4,848 | 12.88 (11.80 to 13.90) |
| 30 min | model | 50% | 15 | 43,582 | 21.05 | 72.4% | 88.6% | 27,088 | 11,507 | 514 | 4,473 | 12.83 (11.75 to 13.84) |
| 30 min | model | 50% | 20 | 42,873 | 20.71 | 72.3% | 89.4% | 26,903 | 11,411 | 470 | 4,089 | 12.77 (11.71 to 13.76) |
| 30 min | climatology | 30% | 10 | 398 | 0.19 | 30.4% | 94.5% | 107 | 269 | 8 | 14 | 0.05 (0.03 to 0.06) |
| 30 min | climatology | 30% | 15 | 377 | 0.18 | 29.4% | 94.7% | 99 | 258 | 8 | 12 | 0.04 (0.03 to 0.06) |
| 30 min | climatology | 30% | 20 | 252 | 0.12 | 26.6% | 96.0% | 61 | 181 | 4 | 6 | 0.03 (0.02 to 0.04) |
| 60 min | model | 30% | 10 | 57,719 | 27.88 | 52.8% | 91.0% | 26,302 | 26,248 | 969 | 4,200 | 12.24 (11.10 to 13.22) |
| 60 min | model | 30% | 15 | 56,318 | 27.20 | 52.8% | 91.8% | 25,947 | 25,730 | 858 | 3,783 | 12.12 (10.99 to 13.09) |
| 60 min | model | 30% | 20 | 54,450 | 26.30 | 52.9% | 92.5% | 25,407 | 24,942 | 728 | 3,373 | 11.92 (10.82 to 12.88) |
| 60 min | model | 40% | 10 | 33,294 | 16.08 | 64.3% | 87.7% | 17,907 | 11,293 | 598 | 3,496 | 8.36 (7.60 to 9.03) |
| 60 min | model | 40% | 15 | 32,603 | 15.75 | 64.2% | 88.6% | 17,751 | 11,139 | 540 | 3,173 | 8.31 (7.56 to 8.98) |
| 60 min | model | 40% | 20 | 31,834 | 15.38 | 64.1% | 89.4% | 17,516 | 10,948 | 481 | 2,889 | 8.23 (7.49 to 8.89) |
| 60 min | model | 50% | 10 | 21,445 | 10.36 | 73.1% | 84.5% | 12,736 | 5,394 | 364 | 2,951 | 5.98 (5.45 to 6.44) |
| 60 min | model | 50% | 15 | 21,005 | 10.15 | 73.0% | 85.7% | 12,652 | 5,339 | 330 | 2,684 | 5.95 (5.43 to 6.42) |
| 60 min | model | 50% | 20 | 20,582 | 9.94 | 72.9% | 86.6% | 12,545 | 5,271 | 299 | 2,467 | 5.91 (5.41 to 6.37) |
| 120 min | model | 30% | 10 | 22,454 | 10.84 | 53.7% | 86.7% | 9,736 | 9,729 | 656 | 2,333 | 4.39 (3.94 to 4.85) |
| 120 min | model | 30% | 15 | 21,562 | 10.41 | 53.8% | 87.8% | 9,533 | 9,406 | 552 | 2,071 | 4.34 (3.89 to 4.80) |
| 120 min | model | 30% | 20 | 20,472 | 9.89 | 53.9% | 89.0% | 9,232 | 8,982 | 447 | 1,811 | 4.24 (3.81 to 4.70) |
| 120 min | model | 40% | 10 | 13,904 | 6.72 | 64.4% | 82.2% | 6,904 | 4,529 | 423 | 2,048 | 3.13 (2.81 to 3.48) |
| 120 min | model | 40% | 15 | 13,467 | 6.50 | 64.3% | 83.6% | 6,812 | 4,440 | 368 | 1,847 | 3.11 (2.79 to 3.46) |
| 120 min | model | 40% | 20 | 12,952 | 6.26 | 64.1% | 85.0% | 6,676 | 4,338 | 307 | 1,631 | 3.08 (2.76 to 3.41) |
| 120 min | model | 50% | 10 | 9,098 | 4.39 | 73.5% | 78.1% | 4,957 | 2,145 | 269 | 1,727 | 2.26 (2.02 to 2.50) |
| 120 min | model | 50% | 15 | 8,851 | 4.27 | 73.4% | 79.4% | 4,912 | 2,113 | 242 | 1,584 | 2.26 (2.01 to 2.50) |
| 120 min | model | 50% | 20 | 8,532 | 4.12 | 73.2% | 81.0% | 4,830 | 2,082 | 205 | 1,415 | 2.23 (1.99 to 2.47) |

**B. By pick and forecast time.** 60-minute deadline, 15-point gain. Earlier and ahead are medians, in
hours; a negative "leaves" means before the current flight. "No change" is the share of flagged trips with
no change offered.

| Pick | Risk | Forecast | Flagged | Predicted miss | True miss | Changes | No change | Rescued | Needless | Made worse | Both missed | Alternative predicted | Alternative true | Net | Arrives ahead | Leaves | Leaves 2 h+ earlier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best | 30% | 2 | 55,596 | 56.9% | 61.9% | 11,549 | 79.2% | 5,905 | 3,745 | 277 | 1,622 | 85.9% | 83.6% | 8.15 (7.68 to 8.60) | 0.42 | -0.42 | 0.0% |
| best | 30% | 6 | 36,511 | 52.1% | 55.9% | 25,215 | 30.9% | 12,221 | 10,816 | 370 | 1,808 | 93.8% | 91.4% | 17.17 (15.73 to 18.47) | 3.47 | -3.45 | 83.5% |
| best | 30% | 24 | 20,807 | 39.6% | 42.0% | 19,554 | 6.0% | 7,821 | 11,169 | 211 | 353 | 96.0% | 97.1% | 11.03 (9.37 to 12.56) | 9.98 | -9.92 | 99.4% |
| best | 30% | all | 112,914 | 52.2% | 56.3% | 56,318 | 50.1% | 25,947 | 25,730 | 858 | 3,783 | 93.0% | 91.8% | 12.12 (10.99 to 13.09) | 3.88 | -3.87 | 71.9% |
| best | 40% | 2 | 40,789 | 64.9% | 70.3% | 9,215 | 77.4% | 5,153 | 2,347 | 215 | 1,500 | 84.1% | 81.4% | 7.15 (6.71 to 7.55) | 0.38 | -0.37 | 0.0% |
| best | 40% | 6 | 23,317 | 62.1% | 65.8% | 16,081 | 31.0% | 9,085 | 5,262 | 240 | 1,494 | 92.8% | 89.2% | 12.82 (11.68 to 13.77) | 3.40 | -3.38 | 81.4% |
| best | 40% | 24 | 7,759 | 48.7% | 50.9% | 7,307 | 5.8% | 3,513 | 3,530 | 85 | 179 | 95.1% | 96.4% | 4.97 (4.06 to 5.88) | 10.02 | -9.92 | 99.5% |
| best | 40% | all | 71,865 | 62.3% | 66.7% | 32,603 | 54.6% | 17,751 | 11,139 | 540 | 3,173 | 90.8% | 88.6% | 8.31 (7.56 to 8.98) | 3.12 | -3.10 | 62.4% |
| best | 50% | 2 | 30,625 | 71.6% | 77.0% | 7,363 | 76.0% | 4,390 | 1,456 | 161 | 1,356 | 82.6% | 79.4% | 6.13 (5.72 to 6.49) | 0.37 | -0.33 | 0.0% |
| best | 50% | 6 | 16,166 | 69.8% | 73.0% | 11,176 | 30.9% | 6,906 | 2,879 | 146 | 1,245 | 92.2% | 87.6% | 9.79 (8.97 to 10.50) | 3.34 | -3.33 | 80.3% |
| best | 50% | 24 | 2,648 | 57.6% | 58.6% | 2,466 | 6.9% | 1,356 | 1,004 | 23 | 83 | 94.2% | 95.7% | 1.93 (1.48 to 2.37) | 10.03 | -9.92 | 99.4% |
| best | 50% | all | 49,439 | 70.3% | 74.7% | 21,005 | 57.5% | 12,652 | 5,339 | 330 | 2,684 | 89.1% | 85.7% | 5.95 (5.43 to 6.42) | 2.35 | -2.33 | 54.4% |
| near | 30% | 2 | 55,596 | 56.9% | 61.9% | 11,549 | 79.2% | 5,905 | 3,745 | 277 | 1,622 | 85.9% | 83.6% | 8.15 (7.68 to 8.60) | 0.42 | -0.42 | 0.0% |
| near | 30% | 6 | 36,511 | 52.1% | 55.9% | 13,722 | 62.4% | 5,944 | 5,500 | 370 | 1,908 | 88.9% | 83.4% | 8.08 (7.48 to 8.60) | 1.40 | -1.38 | 5.1% |
| near | 30% | 24 | 20,807 | 39.6% | 42.0% | 7,005 | 66.3% | 1,949 | 3,945 | 257 | 854 | 86.1% | 84.1% | 2.45 (2.08 to 2.81) | 1.55 | -1.52 | 6.0% |
| near | 30% | all | 112,914 | 52.2% | 56.3% | 32,276 | 71.4% | 13,798 | 13,190 | 904 | 4,384 | 87.2% | 83.6% | 6.23 (5.79 to 6.63) | 1.03 | -1.00 | 3.5% |
| near | 40% | 2 | 40,789 | 64.9% | 70.3% | 9,215 | 77.4% | 5,153 | 2,347 | 215 | 1,500 | 84.1% | 81.4% | 7.15 (6.71 to 7.55) | 0.38 | -0.37 | 0.0% |
| near | 40% | 6 | 23,317 | 62.1% | 65.8% | 9,253 | 60.3% | 4,606 | 2,871 | 236 | 1,540 | 87.7% | 80.8% | 6.33 (5.85 to 6.75) | 1.33 | -1.33 | 5.0% |
| near | 40% | 24 | 7,759 | 48.7% | 50.9% | 2,640 | 66.0% | 838 | 1,257 | 99 | 446 | 81.0% | 79.4% | 1.07 (0.90 to 1.23) | 1.52 | -1.50 | 6.2% |
| near | 40% | all | 71,865 | 62.3% | 66.7% | 21,108 | 70.6% | 10,597 | 6,475 | 550 | 3,486 | 85.3% | 80.9% | 4.85 (4.53 to 5.14) | 0.88 | -0.88 | 3.0% |
| near | 50% | 2 | 30,625 | 71.6% | 77.0% | 7,363 | 76.0% | 4,390 | 1,456 | 161 | 1,356 | 82.6% | 79.4% | 6.13 (5.72 to 6.49) | 0.37 | -0.33 | 0.0% |
| near | 50% | 6 | 16,166 | 69.8% | 73.0% | 6,659 | 58.8% | 3,623 | 1,649 | 139 | 1,248 | 87.0% | 79.2% | 5.05 (4.66 to 5.37) | 1.30 | -1.30 | 4.8% |
| near | 50% | 24 | 2,648 | 57.6% | 58.6% | 807 | 69.5% | 265 | 328 | 29 | 185 | 76.2% | 73.5% | 0.34 (0.25 to 0.43) | 1.55 | -1.50 | 6.4% |
| near | 50% | all | 49,439 | 70.3% | 74.7% | 14,829 | 70.0% | 8,278 | 3,433 | 329 | 2,789 | 84.2% | 79.0% | 3.84 (3.58 to 4.07) | 0.77 | -0.75 | 2.5% |
| latest | 30% | 2 | 55,596 | 56.9% | 61.9% | 11,549 | 79.2% | 5,623 | 3,671 | 351 | 1,904 | 83.0% | 80.5% | 7.64 (7.20 to 8.05) | 0.18 | -0.17 | 0.0% |
| latest | 30% | 6 | 36,511 | 52.1% | 55.9% | 25,215 | 30.9% | 10,686 | 10,479 | 707 | 3,343 | 88.4% | 83.9% | 14.46 (13.28 to 15.46) | 1.90 | -1.88 | 47.0% |
| latest | 30% | 24 | 20,807 | 39.6% | 42.0% | 19,554 | 6.0% | 6,441 | 10,805 | 575 | 1,733 | 88.5% | 88.2% | 8.50 (7.28 to 9.63) | 2.67 | -2.67 | 65.3% |
| latest | 30% | all | 112,914 | 52.2% | 56.3% | 56,318 | 50.1% | 22,750 | 24,955 | 1,633 | 6,980 | 87.3% | 84.7% | 10.20 (9.33 to 11.01) | 1.73 | -1.72 | 43.7% |
| latest | 40% | 2 | 40,789 | 64.9% | 70.3% | 9,215 | 77.4% | 4,883 | 2,284 | 278 | 1,770 | 80.7% | 77.8% | 6.67 (6.27 to 7.03) | 0.12 | -0.10 | 0.0% |
| latest | 40% | 6 | 23,317 | 62.1% | 65.8% | 16,081 | 31.0% | 7,866 | 5,045 | 457 | 2,713 | 86.2% | 80.3% | 10.74 (9.83 to 11.51) | 1.77 | -1.77 | 43.8% |
| latest | 40% | 24 | 7,759 | 48.7% | 50.9% | 7,307 | 5.8% | 2,775 | 3,388 | 227 | 917 | 84.4% | 84.3% | 3.69 (3.08 to 4.30) | 2.68 | -2.67 | 65.1% |
| latest | 40% | all | 71,865 | 62.3% | 66.7% | 32,603 | 54.6% | 15,524 | 10,717 | 962 | 5,400 | 84.2% | 80.5% | 7.03 (6.45 to 7.55) | 1.25 | -1.23 | 36.2% |
| latest | 50% | 2 | 30,625 | 71.6% | 77.0% | 7,363 | 76.0% | 4,141 | 1,415 | 202 | 1,605 | 78.8% | 75.5% | 5.71 (5.34 to 6.02) | 0.08 | -0.08 | 0.0% |
| latest | 50% | 6 | 16,166 | 69.8% | 73.0% | 11,176 | 30.9% | 5,915 | 2,739 | 286 | 2,236 | 84.6% | 77.4% | 8.16 (7.52 to 8.70) | 1.68 | -1.67 | 41.7% |
| latest | 50% | 24 | 2,648 | 57.6% | 58.6% | 2,466 | 6.9% | 1,030 | 957 | 70 | 409 | 80.7% | 80.6% | 1.39 (1.08 to 1.70) | 2.95 | -2.93 | 68.2% |
| latest | 50% | all | 49,439 | 70.3% | 74.7% | 21,005 | 57.5% | 11,086 | 5,111 | 558 | 4,250 | 82.1% | 77.1% | 5.08 (4.69 to 5.42) | 0.90 | -0.92 | 30.2% |

Pick names: `best` is the highest chance at any time; `near` is the highest chance leaving at most 2 h
earlier; `latest` is the last-leaving flight that clears the gain.

**C. Run facts.** Trips at risk at the widest setting: 242,557 of 2,070,453. Trip and alternative pairs:
733,505. Alternative rows: 732,747. Forecasting current flights took 309 s, building alternative inputs
953 s, and forecasting alternatives 102 s.

## Options that were open, now decided in `0014`

`0014` also found a possible leak that affects every result in this record (its Finding 1).

**What "too uncertain to say" means:** (a) close to the cutoff, such as a risk from 40% to 60%;
(b) thin evidence, such as few past flights on that route and carrier; (c) the advice flips when the
cutoffs or cost range change, which is the "holds up" check in `docs/PLAN.md`. AI recommended (a) and (c)
together. Band width and cost range are not chosen, and (c) needs the cost estimate first.

**A simple baseline that acts:** (a) the schedule-only model under the same rule, which may rarely reach
50%; (b) a traveler's rule of thumb: if the inbound aircraft is already late by more than the deadline
allows, take the next flight. AI recommended trying both, since neither needs new inputs.
