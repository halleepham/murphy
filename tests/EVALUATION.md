# Evaluation — does risk-aware ranking beat ranking by scheduled time?

The comparison the instructor feedback asks for, run on real 2024 outcomes.
Reproduce with:

```
.venv/bin/python src/murphy/evaluate.py
```

## Method

**Baseline** ranks itineraries by scheduled duration — what a booking site shows.
**Murphy** ranks by what comparable flights historically did door to door, penalised
by the share of inbound flights that arrived too late to make the transfer.

Both choose an itinerary for a real origin, destination, date and target departure
time. We then look up what actually happened to each one's choice.

- **Temporal split, never random.** Ranking uses January–September 2024 only.
  October is used for scoring and never for ranking.
- **Ten routes**, six with roughly one nonstop a day and dozens of connecting
  options, four well served — the latter check that the method does not connect
  needlessly.
- **Three target departure times a day** (07:00, 12:00, 17:00), candidates within
  three hours. Comparing a 06:00 departure against a 15:00 one is not comparing
  substitutes.

```
RESULTS — 923 planning decisions, 10 routes, month 10 2024
==================================================================

The two methods chose the same itinerary 515 times (56%).
They differed on 408 decisions — those are where the comparison lives.

Completed trips only — missed connections excluded

                median      mean       p90        missed
--------------------------------------------------------
Scheduled         297m      301m      395m       51 (5.5%)
Murphy            305m      306m      405m       18 (2.0%)

With missed connections charged 240 minutes
(a stated assumption, not a measurement — we have no data on what
rebooking actually costs)

                median      mean       p90
------------------------------------------
Scheduled         302m      318m      410m
Murphy            309m      312m      412m

Door-to-door minutes, from scheduled departure to actual arrival.

==================================================================
PAIRED — same route, same day, only where they disagreed
==================================================================

365 paired decisions where the two methods chose differently.

  Murphy arrived earlier    133  (36%)
  Murphy arrived later      225  (62%)
  Same arrival                7  (2%)

  Mean difference          +4.4 min (negative favours Murphy)
  Median difference        +6.0 min
  Worst case for Murphy    +248 min
  Best case for Murphy     -545 min

  Connections missed by the scheduled-time pick but not Murphy: 36
  Connections missed by Murphy but not the scheduled-time pick: 3

  Decisions where the two disagreed about whether to connect at all: 0 of 408
```

## What this shows

**Risk-aware ranking buys reliability with time.** It is not a clean win, and
saying so is more useful than claiming one.

| | Scheduled-time | Murphy | Difference |
|---|---|---|---|
| Missed connections | 51 (5.5%) | **18 (2.0%)** | **−65%** |
| Median door-to-door | 297 min | 305 min | +8 min |
| Mean, failures charged | 318 min | **312 min** | −6 min |
| p90, failures charged | 410 min | 412 min | +2 min |

On 923 planning decisions the two methods agreed 56% of the time — when a
nonstop was available in the departure window, both took it. Murphy does not
connect needlessly.

On the 365 decisions where they differed, Murphy arrived **later** 62% of the
time, by a median of 6 minutes. In exchange it missed 36 connections the
scheduled-time ranking missed, while missing only 3 the baseline made.

So the honest answer to *"does the improved approach provide useful value
compared with a simpler solution?"* is: **yes for reliability, no for typical
speed.** A traveller with a deadline should prefer it; a traveller who only
wants to be home sooner on an average day should not. That trade-off is the
kind of thing the traveller should be choosing, which is why the application
exposes the objectives rather than picking one.

## Limitations of this evaluation

- **A missed connection's real cost is unknown.** The penalised view charges a
  flat 240 minutes. That is a stated assumption, not a measurement.
- **Cancellations are absent from the source data**, so no itinerary is ever
  scored as cancelled. Both methods benefit equally, but real reliability
  differences are understated.
- **Connections are constructed, not published.** A one-stop itinerary here is
  an arrival paired with a departure under a 45-minute rule, not a fare an
  airline sells.
- **One month, ten routes.** Enough to see a consistent effect on connection
  reliability; not enough to claim a general result.
- **No fare data**, so cost never enters the ranking, and a cheaper itinerary
  is never preferred.
