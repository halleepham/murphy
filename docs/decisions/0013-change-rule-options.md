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
