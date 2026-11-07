# 0014 — "Too uncertain", a baseline that acts, and a possible leak

Date: 6 Nov 2026. Code: `scripts/replay_baselines.py`, `inbound_too_late` in `murphy/recommend.py`.
Results (gitignored): `data/forecast/alternatives/replay/{baselines,band,turn}.csv`. 2025 only;
2026 stays sealed. Replay method and limits: `0013`. It reproduces `0013` exactly at the chosen
rule: 49,439 flagged, 14,829 changes, 8,278 rescued, net 3.84 per 1,000.

Unless stated: 60-minute deadline, 15-point gain, highest chance leaving at most 2 h earlier.

## Decision 1 — What "too uncertain to say" means

**Human design:** the owner asked for options with results (`0013`, Open).

**AI recommended, owner accepted, 6 Nov 2026:** a risk from 40% up to 60%, plus the "advice flips"
check once a cost estimate exists. Thin evidence was not chosen. **The band is measured below. The
flip check and the app's verdict are not built.**

**Consequence:** a firm "consider changing" now starts at 60% risk, not the 50% chosen in `0013`.
From 50% to 60% murphy says too uncertain and still lists the alternatives. The band's 60% end
was an AI assumption.

| Forecast risk | Forecasts | Predicted miss | True miss | Changes offered at the same gain | Rescued | Needless | Made worse | Both missed |
|---|---|---|---|---|---|---|---|---|
| Under 40% | 1,998,588 | 7.1% | 7.9% | 32,237 | 7,842 | 21,929 | 774 | 1,692 |
| 40% to 50% | 22,426 | 44.6% | 49.2% | 6,279 | 2,319 | 3,042 | 221 | 697 |
| 50% to 60% | 14,783 | 54.7% | 59.1% | 4,092 | 1,813 | 1,598 | 134 | 547 |
| 60% and up | 34,656 | 76.9% | 81.3% | 10,737 | 6,465 | 1,835 | 195 | 2,242 |

**Read:** the band holds real coin flips: about half to 59% truly miss, against 81% above it. It
covers 37,209 of the 71,865 forecasts at 40% or more. Firm advice from 60% alone gives 10,737
changes and net 3.03 per 1,000 instead of 14,829 and 3.84. Needless changes fall from 23.2% to
17.1% of changes. These figures include the overlap flights in Finding 1.

## Decision 2 — The baselines

**AI recommended, owner accepted, 6 Nov 2026:** try the schedule-only model and a traveler's rule
of thumb.

**Result: the schedule-only model cannot serve as the baseline.** It flags nothing at 60 or 120
minutes and 159 forecasts at 30, the same problem as climatology (`0013` Finding 5). It stays in
the results as a fact. It was trained at 24, 6 and 2 h, not across lead times like `final_leads`.

**The rule of thumb acts, so it is the baseline.** It first ignored the turn: 60,908 flags with
only 23.1% truly missing, because a plane late many hours ago no longer matters. So it now fires
only when the inbound's delay so far, minus its scheduled turn, exceeds the deadline's minutes.
It then takes the next allowed flight, within the same 2 h limit. It has no forecast, so it
checks no gain. The AI chose both details; no minimum turn is assumed.

| Trips | Rule | Flagged | Changes | Rescued | Needless | Made worse | Both missed | Net per 1,000 | murphy minus rule |
|---|---|---|---|---|---|---|---|---|---|
| All | murphy | 49,439 | 14,829 | 8,278 | 3,433 | 329 | 2,789 | 3.84 (3.58 to 4.07) | |
| All | rule of thumb | 11,695 | 4,772 | 2,527 | 1,119 | 150 | 976 | 1.15 (1.07 to 1.23) | 2.69 (2.49 to 2.89) |
| No overlap | murphy | 29,164 | 7,426 | 4,041 | 1,490 | 164 | 1,731 | 1.90 (1.73 to 2.05) | |
| No overlap | rule of thumb | 1,340 | 540 | 378 | 14 | 1 | 147 | 0.18 (0.16 to 0.21) | 1.71 (1.55 to 1.86) |

Without the overlap, 97.9% of the rule's 1,340 flags truly miss, but it flags about 22 times
less often than murphy. murphy sees 1,272 of those 1,340.
Limit: 45 of the rule's 11,695 flags have no saved alternatives, because they were outside the
trips saved in `0013`; they count as no change.

## Finding 1 — A possible leak: turns that overlap

**Not proven, and large.** In 27,002 forecasts (2% of those with a known tail) the aircraft's previous
leg is scheduled to land after this flight is due to leave. No airline plans that. BTS records the
aircraft that actually flew, so an overlap may show a swap made after trouble began. If so, the model
is using hindsight: 79 to 95% of these inbound legs had not even left at the forecast.

| Scheduled turn (min) | Forecasts | Predicted miss | True miss | Flagged | Inbound not yet departed |
|---|---|---|---|---|---|
| Under -60 | 10,292 | 69.6% | 68.7% | 8,997 | 95.4% |
| -60 to -16 | 11,644 | 64.1% | 66.9% | 9,601 | 87.0% |
| -15 to -1 | 5,066 | 45.9% | 45.5% | 1,677 | 78.8% |
| 0 to 14 | 6,340 | 32.8% | 35.0% | 694 | 72.8% |
| 15 to 29 | 16,702 | 14.0% | 17.0% | 433 | 74.0% |
| 30 to 44 | 234,374 | 8.5% | 10.2% | 4,223 | 65.1% |
| 45 to 59 | 303,280 | 8.1% | 9.1% | 6,294 | 56.4% |
| 60 to 119 | 383,574 | 7.5% | 8.0% | 7,436 | 47.2% |
| 120 and up | 405,736 | 7.0% | 7.3% | 4,299 | 6.3% |

Overlaps are 1.3% of all forecasts but 20,275 of the 49,439 flagged: 10,108 of 30,625 at 2 h, and
10,167 of 16,166 at 6 h. **Removing them and overlapping alternatives halves the measured benefit:**
changes fall from 14,829 to 7,426 and net from 3.84 to 1.90 per 1,000.

**What this does not settle:** the model was not refit, so 1.90 is the same model on fewer trips, not
the result of a clean model. Short positive turns may carry the same effect and were kept. The tail
seen 24 h out is already an assumption (`0006`), and `0008` noted these odd turn values and kept them. The
effect on the forecast scores in `0010` and `0011` is unmeasured. All earlier replay results in `0013` include
these flights.

**Owner, 6 Nov 2026:** chose option (a), blank the overlaps and refit. Finding 2 shows that (a) as
written does not work. Nothing has been changed in the model.

## Finding 2 — Blanks leak too, and "no overlap" is not a fair test

Checked before fitting, because a blank close to departure could itself carry a signal. It does.
`aircraft_kinds.csv`, 60-minute deadline; 24 h has no aircraft inputs for any flight:

| Forecast | Aircraft inputs | Forecasts | Predicted miss | True miss | Cancelled | Flagged |
|---|---|---|---|---|---|---|
| 2 h | Normal | 667,946 | 7.6% | 8.2% | 0.7% | 16,117 |
| 2 h | Overlap | 13,228 | 63.6% | 64.2% | 16.1% | 10,000 |
| 2 h | Prior leg's delay blank | 7,330 | 44.7% | 47.6% | 33.6% | 2,932 |
| 2 h | No tail number | 1,647 | 93.7% | 98.5% | 98.1% | 1,576 |
| 6 h | Normal | 671,250 | 7.5% | 8.5% | 0.9% | 3,741 |
| 6 h | Overlap | 13,500 | 62.3% | 63.6% | 16.1% | 10,167 |
| 6 h | Prior leg's delay blank | 3,754 | 31.8% | 32.2% | 18.2% | 697 |
| 6 h | No tail number | 1,647 | 92.6% | 98.5% | 98.1% | 1,561 |

**A second leak.** A flight with no tail number was almost always cancelled: BTS seems to leave the
tail empty when no aircraft flew. The model reads that at 2 and 6 h, before anyone could know. It adds
3,137 flagged forecasts.

**Why blanking fails.** Blanks are rare near departure, so overlaps blanked this way would simply teach
the model that a blank means trouble. The leak would remain, in a different column.

**Why "no overlap" is unfair.** Finding 1's table removes trips with a 64% true miss, so it removes hard
trips because of what later happened. The 1.90 is therefore not a fair estimate. A fair test keeps every
trip and hides what could not be known. "Prior leg's delay blank" mostly means that leg was cancelled, which
can be known in time; it stays.

## Decision 3 — Hide them by borrowing, and refit

**Human design:** the owner chose to refit (Finding 1) and asked for checks first, so no fit is wasted.

**AI recommended, owner accepted, 6 Nov 2026:** a flight with an overlap or no tail, under 24 h out, takes all
three aircraft inputs from a random other flight. The flight stays in, with its real result. This is meant to
match what a traveler sees: a normal-looking plane that may later go wrong. Applied alike to training, trips and
alternatives (`hide_hindsight` in `murphy/forecast/features.py`). A cancelled prior leg can be known in time, so it stays.

Rejected: dropping these flights from training (it skips hard trips, and forecasts already run low); removing
all aircraft inputs (loses real signal, such as a visibly late inbound plane).

**Modified after a check:** the first match, by split, lead and hour, could still be spotted. A classifier picks
out hidden flights with AUC 1.0000 before hiding. Removing all aircraft inputs gives a floor of 0.7682, because bad
days show in other inputs, which is fair. Of 13 matches tried: hour 0.8187, carrier and hour 0.7809, carrier, hour
and origin **0.7696**, which is used. Others that include the date scored 0.78 to 0.84.

**Other checks before the full fit:**
- Unit tests in `tests/test_features.py`, including no borrowing across splits.
- A scan of all inputs on 500,000 short-lead 2025 rows: after hiding, the worst is a blank prior delay (29.9% cancelled
  or diverted, known in time, kept); no other input passes 5.5%, against 1.8% overall.
- A tenth-size fit with a control. At 2 h, overlap flights: predicted miss 57.2% not hidden, 15.4% hidden; true 64.7%.
  The control shows the check can see the leak. These small fits are a check, not results.
- All steps dry-run at 2 rounds, which found one bug in the scoring step.

## Finding 3 — The refit, `final_clean`

Same rows and settings as `final_leads`: 3,931,131 rows, 300 rounds, 2,320 s. Scored on 2025 only. The best round on the
grid is 225, but the curve is flat there (0.55819 against 0.55822 at 300), and all 300 are used, as with `final_leads`.
**It replaces `final_leads` for recommendations.** Both stay saved.

**The forecast barely changes for normal flights, and it no longer sees the swaps.**

| Forecasts | Number | Brier: `final_clean` | Brier: `final_leads` | Brier: climatology | True late | Predicted late: clean | Predicted late: leads |
|---|---|---|---|---|---|---|---|
| all | 2,070,453 | 0.5612 | 0.5555 | 0.6026 | 23.6% | 22.2% | 22.3% |
| 2 h | 690,151 | 0.5377 | 0.5292 | 0.6026 | 23.6% | 22.9% | 22.9% |
| 6 h | 690,151 | 0.5681 | 0.5599 | 0.6026 | 23.6% | 21.9% | 22.0% |
| 24 h | 690,151 | 0.5777 | 0.5774 | 0.6026 | 23.6% | 21.9% | 21.9% |
| normal, under 24 h | 1,339,196 | 0.5391 | 0.5387 | 0.5859 | 22.0% | 21.9% | 20.8% |
| overlap, under 24 h | 26,728 | 1.1347 | 0.8419 | 1.2057 | 83.2% | 32.3% | 82.4% |
| no tail, under 24 h | 3,294 | 1.0431 | 0.0705 | 1.3690 | 99.1% | 45.2% | 94.1% |
| delay blank, under 24 h | 11,084 | 0.6768 | 0.6774 | 0.9371 | 55.4% | 51.9% | 51.9% |

Against climatology overall, it is better by 0.0414 (0.0395 to 0.0433). Against `final_leads` it is worse by 0.0057
(0.0053 to 0.0060), which is the size of what the leak supplied. Hidden flights are still rated higher than others, since
bad days show in other inputs, but well below their true rate. murphy can rarely warn about them.

**The recommendation benefit falls by more than half.** 60-minute deadline, chosen rule.

| | Flagged | Changes | Rescued | Needless | Made worse | Both missed | Net per 1,000 | No change offered |
|---|---|---|---|---|---|---|---|---|
| `final_leads` (`0013`) | 49,439 | 14,829 | 55.8% | 23.2% | 2.2% | 18.8% | 3.84 (3.58 to 4.07) | 70.0% |
| `final_clean` | 31,187 | 7,466 | 48.4% | 22.8% | 3.3% | 25.5% | 1.63 (1.48 to 1.75) | 76.1% |

**This is the fair number for the chosen rule.** It replaces 3.84 and the 1.90 of Finding 1. **Forecasts run lower on risky
flights:** flagged at 50%, predicted miss averages 66.5% and the true rate is 74.0%, a gap of 7.5 points against 4.4 before.
Hidden flights are in that true rate.

**Baseline.** murphy beats the rule of thumb. The schedule-only model again makes no change at 60 or 120 minutes.

| Trips | Rule | Flagged | Changes | Rescued | Needless | Made worse | Both missed | Net per 1,000 | murphy minus rule |
|---|---|---|---|---|---|---|---|---|---|
| All, hidden | murphy | 31,187 | 7,466 | 3,614 | 1,700 | 248 | 1,904 | 1.63 (1.48 to 1.75) |  |
| All, hidden | schedule | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 1.63 (1.48 to 1.75) |
| All, hidden | thumb | 1,380 | 564 | 387 | 15 | 1 | 161 | 0.19 (0.17 to 0.21) | 1.44 (1.31 to 1.56) |

**The risk cutoffs and the band again**, leaving at most 2 h earlier:

| Risk of missing | Flagged | Changes | Rescued | Needless | Made worse | Both missed | Net per 1,000 |
|---|---|---|---|---|---|---|---|
| 30% | 98,553 | 24,029 | 36.2% | 44.8% | 3.8% | 15.1% | 3.75 (3.43 to 4.07) |
| 40% | 53,813 | 13,104 | 42.4% | 33.2% | 3.9% | 20.5% | 2.44 (2.23 to 2.64) |
| 50% | 31,187 | 7,466 | 48.4% | 22.8% | 3.3% | 25.5% | 1.63 (1.48 to 1.75) |

| Forecast risk | Forecasts | Predicted miss | True miss | Changes offered | Rescued | Needless | Made worse | Both missed |
|---|---|---|---|---|---|---|---|---|
| <40 | 2,016,640 | 7.7% | 8.5% | 29,700 | 7,362 | 19,707 | 910 | 1,721 |
| 40-50 | 22,626 | 44.5% | 49.2% | 5,638 | 1,943 | 2,649 | 260 | 786 |
| 50-60 | 12,475 | 54.5% | 61.6% | 2,977 | 1,169 | 1,063 | 130 | 615 |
| 60+ | 18,712 | 74.4% | 82.3% | 4,489 | 2,445 | 637 | 118 | 1,289 |

Firm advice from 60% alone gives 4,489 changes, net 1.12 per 1,000 and 14.2% needless, against 7,466, 1.63 and 22.8% from 50%.

**Read:** the band still holds real uncertainty, but true miss is higher than the forecast near the top of it. Changes from 50% to 60%
are the weak ones: more of them are needless than at 60% and up.

**Limits:**
- Not proof the leak is gone. BTS shows the aircraft that flew, so a swap that leaves no overlap stays hidden. Turns of 0 to 14 minutes
  still run at 30.1% predicted, 35.0% true, and were kept.
- Borrowed inputs stand in for the plane really planned, which is not in the data.
- The input comparison in `0010` was not rerun, so the gain credited there to current state and aircraft inputs may be overstated.
- The `0011` lead-time scores on the 30,000-flight sample were not rerun.
- The rule-of-thumb baseline also uses hidden inputs. 18 of its 1,380 flags at 60 minutes have no saved alternatives and count as no change.

## Decision 4 — Keep the band after the refit

**AI recommended, owner accepted, 6 Nov 2026:** keep too uncertain from 40% up to 60%, and firm "consider changing" from
60%, with `final_clean`. From 50% to 60% a change rescues the trip about as often as it is needless (1,169 against 1,063), which
does not fit "only when really needed". Those flights still list their alternatives.

**Cost accepted:** 4,489 changes and net 1.12 per 1,000, against 7,466 and 1.63 from 50%; needless falls from 22.8% to 14.2%.
**Rejected:** firm advice from 50% with a 40-50% band, which has the higher net. Gain (15 points) and the 2 h limit stay. **Not built:**
the app's verdict; `MIN_MISS` in `murphy/recommend.py` stays 50% as the replay's setting until then.

## Tests

`tests/test_recommend.py`: the rule counts the turn, a long turn does not fire, and blanks never fire
(the deliberate thin-evidence case). The replay reproduces `0013` as its check; the comparison with a
simpler method is the table above.

## Appendix — All results

Raw files are gitignored, so every figure is recorded. Net is per 1,000 forecasts with a 95% interval
resampling whole days. "No overlap" removes overlapping trips and alternatives.

**A. Rules, by deadline.** "Flags miss" is the rule of thumb's true miss among its flags; "Also murphy" is how many of them murphy flags too.

| Deadline | Trips | Rule | Forecasts | Flagged | Changes | Rescued | Needless | Made worse | Both missed | Net | murphy minus rule | Flags miss | Also murphy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | All | murphy | 2,070,453 | 99,126 | 27,995 | 15,447 | 6,476 | 653 | 5,419 | 7.15 (6.69 to 7.58) |  |  |  |
| 30 min | All | schedule-only | 2,070,453 | 159 | 55 | 17 | 11 | 6 | 21 | 0.01 (-0.00 to 0.01) | 7.14 (6.68 to 7.57) |  |  |
| 30 min | All | rule of thumb | 2,070,453 | 21,926 | 7,446 | 4,421 | 1,149 | 156 | 1,720 | 2.06 (1.95 to 2.16) | 5.09 (4.73 to 5.44) | 82.2% | 20,834 |
| 30 min | No overlap | murphy | 2,043,451 | 73,419 | 19,562 | 10,087 | 4,917 | 496 | 4,062 | 4.69 (4.33 to 5.04) |  |  |  |
| 30 min | No overlap | schedule-only | 2,043,451 | 153 | 52 | 16 | 11 | 4 | 21 | 0.01 (-0.00 to 0.01) | 4.69 (4.32 to 5.03) |  |  |
| 30 min | No overlap | rule of thumb | 2,043,451 | 4,431 | 1,360 | 918 | 18 | 1 | 423 | 0.45 (0.42 to 0.48) | 4.24 (3.89 to 4.57) | 98.9% | 4,320 |
| 60 min | All | murphy | 2,070,453 | 49,439 | 14,829 | 8,278 | 3,433 | 329 | 2,789 | 3.84 (3.58 to 4.07) |  |  |  |
| 60 min | All | schedule-only | 2,070,453 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 3.84 (3.58 to 4.07) |  |  |
| 60 min | All | rule of thumb | 2,070,453 | 11,695 | 4,772 | 2,527 | 1,119 | 150 | 976 | 1.15 (1.07 to 1.23) | 2.69 (2.49 to 2.89) | 72.5% | 10,369 |
| 60 min | No overlap | murphy | 2,043,451 | 29,164 | 7,426 | 4,041 | 1,490 | 164 | 1,731 | 1.90 (1.73 to 2.05) |  |  |  |
| 60 min | No overlap | schedule-only | 2,043,451 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 1.90 (1.73 to 2.05) |  |  |
| 60 min | No overlap | rule of thumb | 2,043,451 | 1,340 | 540 | 378 | 14 | 1 | 147 | 0.18 (0.16 to 0.20) | 1.71 (1.55 to 1.86) | 97.9% | 1,272 |
| 120 min | All | murphy | 2,070,453 | 19,196 | 7,019 | 3,599 | 1,572 | 231 | 1,617 | 1.63 (1.46 to 1.79) |  |  |  |
| 120 min | All | schedule-only | 2,070,453 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 1.63 (1.46 to 1.79) |  |  |
| 120 min | All | rule of thumb | 2,070,453 | 3,390 | 1,966 | 943 | 620 | 65 | 338 | 0.42 (0.38 to 0.48) | 1.20 (1.07 to 1.35) | 64.1% | 2,565 |
| 120 min | No overlap | murphy | 2,043,451 | 11,274 | 3,533 | 1,919 | 570 | 104 | 940 | 0.89 (0.78 to 1.00) |  |  |  |
| 120 min | No overlap | schedule-only | 2,043,451 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 0.89 (0.78 to 1.00) |  |  |
| 120 min | No overlap | rule of thumb | 2,043,451 | 175 | 108 | 80 | 6 | 1 | 21 | 0.04 (0.03 to 0.05) | 0.85 (0.74 to 0.96) | 96.0% | 152 |

**B. Risk ranges, by deadline and forecast time.** Changes are those offered at a 15-point gain at any risk, to show what each range would do. Net here is per 1,000 forecasts in that range.

| Deadline | Trips | Forecast (h) | Risk | Forecasts | Predicted miss | True miss | Changes | Rescued | Needless | Made worse | Both missed | Net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | All | 2 | <40 | 614,776 | 9.4% | 9.7% | 3,526 | 1,080 | 2,146 | 102 | 198 | 1.59 (1.44 to 1.74) |
| 30 min | All | 2 | 40-50 | 17,434 | 44.7% | 47.3% | 2,715 | 1,080 | 1,332 | 91 | 212 | 56.73 (53.20 to 61.39) |
| 30 min | All | 2 | 50-60 | 13,361 | 54.9% | 59.2% | 2,279 | 1,112 | 850 | 52 | 265 | 79.34 (74.57 to 85.31) |
| 30 min | All | 2 | 60+ | 44,580 | 79.9% | 84.3% | 9,179 | 5,905 | 1,207 | 162 | 1,905 | 128.82 (122.14 to 135.38) |
| 30 min | All | 6 | <40 | 644,147 | 11.7% | 12.9% | 10,510 | 3,435 | 6,149 | 291 | 635 | 4.88 (4.45 to 5.34) |
| 30 min | All | 6 | 40-50 | 16,111 | 44.6% | 49.2% | 6,051 | 2,321 | 2,870 | 196 | 664 | 131.90 (126.21 to 139.12) |
| 30 min | All | 6 | 50-60 | 9,747 | 54.6% | 59.8% | 3,797 | 1,736 | 1,391 | 107 | 563 | 167.13 (158.25 to 177.04) |
| 30 min | All | 6 | 60+ | 20,146 | 77.6% | 80.5% | 8,207 | 4,924 | 1,396 | 162 | 1,725 | 236.37 (222.42 to 251.01) |
| 30 min | All | 24 | <40 | 661,768 | 13.4% | 14.8% | 13,798 | 4,078 | 8,293 | 377 | 1,050 | 5.59 (5.03 to 6.13) |
| 30 min | All | 24 | 40-50 | 17,091 | 44.4% | 47.8% | 6,864 | 2,382 | 3,399 | 248 | 835 | 124.86 (118.01 to 130.72) |
| 30 min | All | 24 | 50-60 | 7,553 | 54.3% | 57.0% | 3,151 | 1,206 | 1,241 | 131 | 573 | 142.33 (132.34 to 153.93) |
| 30 min | All | 24 | 60+ | 3,739 | 66.9% | 69.5% | 1,382 | 564 | 391 | 39 | 388 | 140.41 (127.10 to 154.84) |
| 30 min | All | all | <40 | 1,920,691 | 11.5% | 12.5% | 27,834 | 8,593 | 16,588 | 770 | 1,883 | 4.07 (3.72 to 4.42) |
| 30 min | All | all | 40-50 | 50,636 | 44.6% | 48.1% | 15,630 | 5,783 | 7,601 | 535 | 1,711 | 103.64 (100.05 to 108.21) |
| 30 min | All | all | 50-60 | 30,661 | 54.6% | 58.9% | 9,227 | 4,054 | 3,482 | 290 | 1,401 | 122.76 (117.93 to 128.17) |
| 30 min | All | all | 60+ | 68,465 | 78.5% | 82.4% | 18,768 | 11,393 | 2,994 | 363 | 4,018 | 161.10 (152.77 to 169.59) |
| 30 min | No overlap | 2 | <40 | 614,556 | 9.4% | 9.7% | 3,493 | 1,071 | 2,124 | 101 | 197 | 1.58 (1.43 to 1.73) |
| 30 min | No overlap | 2 | 40-50 | 16,940 | 44.7% | 47.4% | 2,629 | 1,040 | 1,289 | 90 | 210 | 56.08 (52.71 to 60.06) |
| 30 min | No overlap | 2 | 50-60 | 12,383 | 54.8% | 59.5% | 2,082 | 1,013 | 763 | 47 | 259 | 78.01 (72.71 to 83.20) |
| 30 min | No overlap | 2 | 60+ | 32,771 | 79.3% | 85.3% | 6,493 | 4,192 | 779 | 104 | 1,418 | 124.74 (118.30 to 131.77) |
| 30 min | No overlap | 6 | <40 | 644,015 | 11.7% | 12.9% | 10,451 | 3,407 | 6,121 | 290 | 633 | 4.84 (4.41 to 5.30) |
| 30 min | No overlap | 6 | 40-50 | 15,662 | 44.5% | 49.2% | 5,873 | 2,244 | 2,788 | 192 | 649 | 131.02 (124.87 to 137.33) |
| 30 min | No overlap | 6 | 50-60 | 8,721 | 54.5% | 60.0% | 3,400 | 1,524 | 1,236 | 105 | 535 | 162.71 (153.60 to 174.48) |
| 30 min | No overlap | 6 | 60+ | 8,252 | 74.6% | 80.6% | 3,093 | 1,597 | 514 | 73 | 909 | 184.68 (166.96 to 202.08) |
| 30 min | No overlap | 24 | <40 | 661,768 | 13.4% | 14.8% | 13,776 | 4,072 | 8,279 | 375 | 1,050 | 5.59 (5.02 to 6.11) |
| 30 min | No overlap | 24 | 40-50 | 17,091 | 44.4% | 47.8% | 6,824 | 2,370 | 3,388 | 246 | 820 | 124.28 (117.38 to 130.05) |
| 30 min | No overlap | 24 | 50-60 | 7,553 | 54.3% | 57.0% | 3,122 | 1,199 | 1,234 | 130 | 559 | 141.53 (131.53 to 153.61) |
| 30 min | No overlap | 24 | 60+ | 3,739 | 66.9% | 69.5% | 1,372 | 562 | 391 | 37 | 382 | 140.41 (127.34 to 154.69) |
| 30 min | No overlap | all | <40 | 1,920,339 | 11.5% | 12.5% | 27,720 | 8,550 | 16,524 | 766 | 1,880 | 4.05 (3.70 to 4.40) |
| 30 min | No overlap | all | 40-50 | 49,693 | 44.5% | 48.1% | 15,326 | 5,654 | 7,465 | 528 | 1,679 | 103.15 (99.56 to 107.56) |
| 30 min | No overlap | all | 50-60 | 28,657 | 54.6% | 59.0% | 8,604 | 3,736 | 3,233 | 282 | 1,353 | 120.53 (115.84 to 125.96) |
| 30 min | No overlap | all | 60+ | 44,762 | 77.4% | 83.1% | 10,958 | 6,351 | 1,684 | 214 | 2,709 | 137.10 (129.72 to 144.86) |
| 60 min | All | 2 | <40 | 649,362 | 5.8% | 6.1% | 5,688 | 1,540 | 3,855 | 110 | 183 | 2.20 (2.05 to 2.35) |
| 60 min | All | 2 | 40-50 | 10,164 | 44.8% | 50.2% | 1,852 | 763 | 891 | 54 | 144 | 69.76 (64.10 to 75.81) |
| 60 min | All | 2 | 50-60 | 7,938 | 54.8% | 60.3% | 1,621 | 781 | 603 | 66 | 171 | 90.07 (82.94 to 97.24) |
| 60 min | All | 2 | 60+ | 22,687 | 77.5% | 82.8% | 5,742 | 3,609 | 853 | 95 | 1,185 | 154.89 (145.11 to 164.40) |
| 60 min | All | 6 | <40 | 666,834 | 7.1% | 8.0% | 12,502 | 3,151 | 8,428 | 299 | 624 | 4.28 (3.93 to 4.62) |
| 60 min | All | 6 | 40-50 | 7,151 | 44.6% | 49.5% | 2,594 | 983 | 1,222 | 97 | 292 | 123.90 (115.95 to 133.03) |
| 60 min | All | 6 | 50-60 | 4,961 | 54.8% | 58.5% | 1,864 | 842 | 736 | 46 | 240 | 160.45 (146.37 to 173.15) |
| 60 min | All | 6 | 60+ | 11,205 | 76.5% | 79.4% | 4,795 | 2,781 | 913 | 93 | 1,008 | 239.89 (220.27 to 261.58) |
| 60 min | All | 24 | <40 | 682,392 | 8.3% | 9.5% | 14,047 | 3,151 | 9,646 | 365 | 885 | 4.08 (3.61 to 4.52) |
| 60 min | All | 24 | 40-50 | 5,111 | 44.2% | 47.0% | 1,833 | 573 | 929 | 70 | 261 | 98.42 (89.21 to 107.70) |
| 60 min | All | 24 | 50-60 | 1,884 | 54.2% | 55.6% | 607 | 190 | 259 | 22 | 136 | 89.17 (74.26 to 106.70) |
| 60 min | All | 24 | 60+ | 764 | 66.0% | 66.0% | 200 | 75 | 69 | 7 | 49 | 89.01 (74.10 to 107.25) |
| 60 min | All | all | <40 | 1,998,588 | 7.1% | 7.9% | 32,237 | 7,842 | 21,929 | 774 | 1,692 | 3.54 (3.24 to 3.83) |
| 60 min | All | all | 40-50 | 22,426 | 44.6% | 49.2% | 6,279 | 2,319 | 3,042 | 221 | 697 | 93.55 (89.00 to 99.02) |
| 60 min | All | all | 50-60 | 14,783 | 54.7% | 59.1% | 4,092 | 1,813 | 1,598 | 134 | 547 | 113.58 (106.28 to 120.40) |
| 60 min | All | all | 60+ | 34,656 | 76.9% | 81.3% | 10,737 | 6,465 | 1,835 | 195 | 2,242 | 180.92 (168.40 to 193.48) |
| 60 min | No overlap | 2 | <40 | 647,627 | 5.8% | 6.1% | 5,290 | 1,419 | 3,595 | 103 | 173 | 2.03 (1.88 to 2.17) |
| 60 min | No overlap | 2 | 40-50 | 8,506 | 44.7% | 51.1% | 1,487 | 624 | 686 | 45 | 132 | 68.07 (61.62 to 74.08) |
| 60 min | No overlap | 2 | 50-60 | 5,805 | 54.6% | 62.3% | 1,059 | 519 | 365 | 40 | 135 | 82.52 (74.27 to 90.33) |
| 60 min | No overlap | 2 | 60+ | 14,712 | 78.4% | 85.9% | 3,540 | 2,291 | 393 | 43 | 813 | 152.80 (142.96 to 163.46) |
| 60 min | No overlap | 6 | <40 | 665,442 | 7.0% | 7.9% | 11,957 | 2,971 | 8,081 | 289 | 616 | 4.03 (3.71 to 4.36) |
| 60 min | No overlap | 6 | 40-50 | 5,209 | 44.3% | 50.7% | 1,811 | 639 | 829 | 84 | 259 | 106.55 (97.61 to 115.37) |
| 60 min | No overlap | 6 | 50-60 | 2,414 | 54.4% | 60.0% | 793 | 290 | 293 | 29 | 181 | 108.12 (94.38 to 123.33) |
| 60 min | No overlap | 6 | 60+ | 3,585 | 81.4% | 87.1% | 1,230 | 677 | 111 | 23 | 419 | 182.43 (154.79 to 214.30) |
| 60 min | No overlap | 24 | <40 | 682,392 | 8.3% | 9.5% | 14,026 | 3,148 | 9,634 | 362 | 882 | 4.08 (3.61 to 4.52) |
| 60 min | No overlap | 24 | 40-50 | 5,111 | 44.2% | 47.0% | 1,818 | 569 | 925 | 69 | 255 | 97.83 (88.74 to 106.72) |
| 60 min | No overlap | 24 | 50-60 | 1,884 | 54.2% | 55.6% | 606 | 190 | 259 | 22 | 135 | 89.17 (74.26 to 106.70) |
| 60 min | No overlap | 24 | 60+ | 764 | 66.0% | 66.0% | 198 | 74 | 69 | 7 | 48 | 87.70 (72.13 to 105.40) |
| 60 min | No overlap | all | <40 | 1,995,461 | 7.0% | 7.9% | 31,273 | 7,538 | 21,310 | 754 | 1,671 | 3.40 (3.10 to 3.69) |
| 60 min | No overlap | all | 40-50 | 18,826 | 44.5% | 49.9% | 5,116 | 1,832 | 2,440 | 198 | 646 | 86.79 (82.16 to 92.69) |
| 60 min | No overlap | all | 50-60 | 10,103 | 54.5% | 60.5% | 2,458 | 999 | 917 | 91 | 451 | 89.87 (82.89 to 96.75) |
| 60 min | No overlap | all | 60+ | 19,061 | 78.5% | 85.3% | 4,968 | 3,042 | 573 | 73 | 1,280 | 155.76 (142.73 to 169.59) |
| 120 min | All | 2 | <40 | 673,479 | 3.3% | 3.6% | 4,625 | 1,257 | 3,140 | 108 | 120 | 1.71 (1.55 to 1.86) |
| 120 min | All | 2 | 40-50 | 4,966 | 44.7% | 51.2% | 1,303 | 549 | 616 | 52 | 86 | 100.08 (91.32 to 109.13) |
| 120 min | All | 2 | 50-60 | 3,599 | 54.7% | 61.0% | 1,068 | 508 | 384 | 53 | 123 | 126.42 (111.82 to 139.95) |
| 120 min | All | 2 | 60+ | 8,107 | 78.1% | 82.4% | 2,656 | 1,447 | 396 | 80 | 733 | 168.62 (150.88 to 188.67) |
| 120 min | All | 6 | <40 | 679,466 | 3.7% | 4.2% | 5,561 | 1,392 | 3,821 | 137 | 211 | 1.85 (1.68 to 2.01) |
| 120 min | All | 6 | 40-50 | 3,682 | 44.7% | 48.5% | 1,344 | 500 | 666 | 56 | 122 | 120.59 (107.03 to 135.09) |
| 120 min | All | 6 | 50-60 | 2,283 | 54.7% | 59.8% | 964 | 435 | 386 | 28 | 115 | 178.27 (154.96 to 201.47) |
| 120 min | All | 6 | 60+ | 4,720 | 79.3% | 81.0% | 2,234 | 1,185 | 355 | 65 | 629 | 237.29 (211.56 to 270.24) |
| 120 min | All | 24 | <40 | 688,550 | 4.4% | 5.1% | 2,434 | 406 | 1,826 | 63 | 139 | 0.50 (0.40 to 0.60) |
| 120 min | All | 24 | 40-50 | 1,114 | 44.0% | 40.9% | 220 | 57 | 121 | 12 | 30 | 40.39 (27.68 to 54.89) |
| 120 min | All | 24 | 50-60 | 350 | 54.1% | 48.6% | 76 | 15 | 43 | 3 | 15 | 34.29 (12.22 to 55.10) |
| 120 min | All | 24 | 60+ | 137 | 65.0% | 62.8% | 21 | 9 | 8 | 2 | 2 | 51.09 (6.89 to 107.07) |
| 120 min | All | all | <40 | 2,041,495 | 3.8% | 4.3% | 12,620 | 3,055 | 8,787 | 308 | 470 | 1.35 (1.23 to 1.45) |
| 120 min | All | all | 40-50 | 9,762 | 44.6% | 49.0% | 2,867 | 1,106 | 1,403 | 120 | 238 | 101.00 (93.06 to 110.11) |
| 120 min | All | all | 50-60 | 6,232 | 54.7% | 59.9% | 2,108 | 958 | 813 | 84 | 253 | 140.24 (127.14 to 155.25) |
| 120 min | All | all | 60+ | 12,964 | 78.4% | 81.7% | 4,911 | 2,641 | 759 | 147 | 1,364 | 192.38 (170.27 to 216.67) |
| 120 min | No overlap | 2 | <40 | 665,798 | 3.1% | 3.4% | 2,986 | 779 | 2,043 | 72 | 92 | 1.06 (0.94 to 1.18) |
| 120 min | No overlap | 2 | 40-50 | 3,301 | 44.6% | 51.7% | 780 | 323 | 370 | 32 | 55 | 88.16 (77.53 to 98.89) |
| 120 min | No overlap | 2 | 50-60 | 2,253 | 54.7% | 63.7% | 590 | 279 | 195 | 37 | 79 | 107.41 (92.25 to 121.21) |
| 120 min | No overlap | 2 | 60+ | 5,298 | 80.4% | 87.0% | 1,635 | 946 | 177 | 33 | 479 | 172.33 (151.24 to 196.87) |
| 120 min | No overlap | 6 | <40 | 671,691 | 3.5% | 4.0% | 2,799 | 611 | 1,960 | 87 | 141 | 0.78 (0.67 to 0.88) |
| 120 min | No overlap | 6 | 40-50 | 1,723 | 44.6% | 46.3% | 462 | 126 | 244 | 31 | 61 | 55.14 (41.67 to 69.55) |
| 120 min | No overlap | 6 | 50-60 | 867 | 54.6% | 61.7% | 221 | 76 | 89 | 11 | 45 | 74.97 (53.88 to 96.89) |
| 120 min | No overlap | 6 | 60+ | 2,369 | 87.1% | 90.4% | 991 | 594 | 58 | 18 | 321 | 243.14 (203.35 to 298.11) |
| 120 min | No overlap | 24 | <40 | 688,550 | 4.4% | 5.1% | 2,427 | 403 | 1,823 | 63 | 138 | 0.49 (0.40 to 0.60) |
| 120 min | No overlap | 24 | 40-50 | 1,114 | 44.0% | 40.9% | 219 | 57 | 121 | 12 | 29 | 40.39 (27.68 to 54.89) |
| 120 min | No overlap | 24 | 50-60 | 350 | 54.1% | 48.6% | 76 | 15 | 43 | 3 | 15 | 34.29 (12.22 to 55.10) |
| 120 min | No overlap | 24 | 60+ | 137 | 65.0% | 62.8% | 20 | 9 | 8 | 2 | 1 | 51.09 (6.89 to 107.07) |
| 120 min | No overlap | all | <40 | 2,026,039 | 3.7% | 4.1% | 8,212 | 1,793 | 5,826 | 222 | 371 | 0.78 (0.69 to 0.86) |
| 120 min | No overlap | all | 40-50 | 6,138 | 44.5% | 48.2% | 1,461 | 506 | 735 | 75 | 145 | 70.22 (62.65 to 79.30) |
| 120 min | No overlap | all | 50-60 | 3,470 | 54.6% | 61.7% | 887 | 370 | 327 | 51 | 139 | 91.93 (79.29 to 106.54) |
| 120 min | No overlap | all | 60+ | 7,804 | 82.2% | 87.6% | 2,646 | 1,549 | 243 | 53 | 801 | 191.70 (164.14 to 224.73) |

**C. `final_clean`: every rule, deadline and scorer.** Replay: 237,450 trips at risk at the widest setting.

| Deadline | Scorer | Risk | Gain | Changes | Per 1,000 | Current missed | Alternative made | Rescued | Needless | Made worse | Both missed | Net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | model | 30% | 10 | 133,675 | 64.56 | 49.0% | 92.4% | 57,618 | 65,889 | 2,237 | 7,931 | 26.75 (24.25 to 29.07) |
| 30 min | model | 30% | 15 | 130,953 | 63.25 | 48.9% | 93.0% | 56,919 | 64,849 | 2,008 | 7,177 | 26.52 (24.05 to 28.84) |
| 30 min | model | 30% | 20 | 126,936 | 61.31 | 48.9% | 93.6% | 55,726 | 63,134 | 1,746 | 6,330 | 26.07 (23.65 to 28.34) |
| 30 min | model | 40% | 10 | 68,960 | 33.31 | 59.0% | 89.0% | 34,383 | 26,991 | 1,308 | 6,278 | 15.97 (14.37 to 17.57) |
| 30 min | model | 40% | 15 | 67,624 | 32.66 | 58.8% | 89.8% | 34,039 | 26,660 | 1,192 | 5,733 | 15.86 (14.27 to 17.46) |
| 30 min | model | 40% | 20 | 65,890 | 31.82 | 58.6% | 90.6% | 33,525 | 26,195 | 1,056 | 5,114 | 15.68 (14.10 to 17.26) |
| 30 min | model | 50% | 10 | 35,109 | 16.96 | 68.9% | 84.4% | 19,431 | 10,207 | 699 | 4,772 | 9.05 (8.03 to 10.01) |
| 30 min | model | 50% | 15 | 34,343 | 16.59 | 68.8% | 85.5% | 19,258 | 10,090 | 640 | 4,355 | 8.99 (7.98 to 9.95) |
| 30 min | model | 50% | 20 | 33,419 | 16.14 | 68.6% | 86.6% | 19,024 | 9,929 | 563 | 3,903 | 8.92 (7.92 to 9.86) |
| 30 min | climatology | 30% | 10 | 398 | 0.19 | 30.4% | 94.5% | 107 | 269 | 8 | 14 | 0.05 (0.03 to 0.06) |
| 30 min | climatology | 30% | 15 | 377 | 0.18 | 29.4% | 94.7% | 99 | 258 | 8 | 12 | 0.04 (0.03 to 0.06) |
| 30 min | climatology | 30% | 20 | 252 | 0.12 | 26.6% | 96.0% | 61 | 181 | 4 | 6 | 0.03 (0.02 to 0.04) |
| 60 min | model | 30% | 10 | 49,623 | 23.97 | 48.2% | 89.0% | 19,743 | 24,401 | 1,300 | 4,179 | 8.91 (7.88 to 9.86) |
| 60 min | model | 30% | 15 | 47,466 | 22.93 | 48.1% | 90.1% | 19,203 | 23,549 | 1,103 | 3,611 | 8.74 (7.73 to 9.67) |
| 60 min | model | 30% | 20 | 44,523 | 21.50 | 48.1% | 91.2% | 18,363 | 22,220 | 905 | 3,035 | 8.43 (7.44 to 9.36) |
| 60 min | model | 40% | 10 | 23,578 | 11.39 | 59.7% | 83.6% | 10,904 | 8,803 | 690 | 3,181 | 4.93 (4.34 to 5.50) |
| 60 min | model | 40% | 15 | 22,633 | 10.93 | 59.5% | 85.0% | 10,684 | 8,565 | 599 | 2,785 | 4.87 (4.28 to 5.43) |
| 60 min | model | 40% | 20 | 21,573 | 10.42 | 59.3% | 86.5% | 10,397 | 8,271 | 511 | 2,394 | 4.77 (4.20 to 5.33) |
| 60 min | model | 50% | 10 | 11,890 | 5.74 | 71.3% | 77.4% | 6,132 | 3,075 | 336 | 2,347 | 2.80 (2.43 to 3.13) |
| 60 min | model | 50% | 15 | 11,397 | 5.50 | 71.0% | 79.2% | 6,021 | 3,011 | 291 | 2,074 | 2.77 (2.41 to 3.10) |
| 60 min | model | 50% | 20 | 10,858 | 5.24 | 70.8% | 81.0% | 5,877 | 2,916 | 255 | 1,810 | 2.72 (2.37 to 3.05) |
| 120 min | model | 30% | 10 | 14,510 | 7.01 | 47.0% | 83.8% | 5,095 | 7,059 | 631 | 1,725 | 2.16 (1.84 to 2.46) |
| 120 min | model | 30% | 15 | 13,004 | 6.28 | 46.7% | 86.0% | 4,747 | 6,437 | 488 | 1,332 | 2.06 (1.77 to 2.36) |
| 120 min | model | 30% | 20 | 11,435 | 5.52 | 46.9% | 87.8% | 4,329 | 5,715 | 356 | 1,035 | 1.92 (1.64 to 2.20) |
| 120 min | model | 40% | 10 | 7,364 | 3.56 | 58.0% | 77.2% | 2,937 | 2,746 | 346 | 1,335 | 1.25 (1.07 to 1.44) |
| 120 min | model | 40% | 15 | 6,712 | 3.24 | 57.4% | 79.9% | 2,790 | 2,576 | 285 | 1,061 | 1.21 (1.04 to 1.40) |
| 120 min | model | 40% | 20 | 6,101 | 2.95 | 57.2% | 82.4% | 2,640 | 2,385 | 228 | 848 | 1.16 (1.00 to 1.35) |
| 120 min | model | 50% | 10 | 3,806 | 1.84 | 68.8% | 69.9% | 1,650 | 1,012 | 177 | 967 | 0.71 (0.60 to 0.81) |
| 120 min | model | 50% | 15 | 3,504 | 1.69 | 67.8% | 73.0% | 1,584 | 975 | 153 | 792 | 0.69 (0.59 to 0.80) |
| 120 min | model | 50% | 20 | 3,229 | 1.56 | 67.4% | 76.1% | 1,528 | 928 | 124 | 649 | 0.68 (0.58 to 0.78) |

**D. `final_clean`: by pick and forecast time.** 60-minute deadline, 15-point gain. `best`, `near` and `latest` as in `0013`.

| Pick | Risk | Forecast | Flagged | Predicted miss | True miss | Changes | No change | Rescued | Needless | Made worse | Both missed | Alternative predicted | Alternative true | Net | Arrives ahead | Leaves | Leaves 2 h+ earlier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best | 30% | 2 | 48,602 | 52.2% | 58.6% | 8,420 | 82.7% | 3,999 | 2,829 | 261 | 1,331 | 83.7% | 81.1% | 5.42 (5.04 to 5.77) | 0.50 | -0.50 | 0.0% |
| best | 30% | 6 | 28,778 | 42.9% | 48.4% | 19,400 | 32.6% | 7,445 | 9,670 | 488 | 1,797 | 90.6% | 88.2% | 10.08 (8.83 to 11.29) | 3.62 | -3.58 | 87.5% |
| best | 30% | 24 | 21,173 | 40.0% | 41.9% | 19,646 | 7.2% | 7,759 | 11,050 | 354 | 483 | 94.1% | 95.7% | 10.73 (9.10 to 12.27) | 9.58 | -9.52 | 99.4% |
| best | 30% | all | 98,553 | 46.9% | 52.0% | 47,466 | 51.8% | 19,203 | 23,549 | 1,103 | 3,611 | 90.8% | 90.1% | 8.74 (7.73 to 9.67) | 4.27 | -4.25 | 76.9% |
| best | 40% | 2 | 32,167 | 61.2% | 68.7% | 6,179 | 80.8% | 3,242 | 1,546 | 187 | 1,204 | 80.9% | 77.5% | 4.43 (4.10 to 4.72) | 0.45 | -0.42 | 0.0% |
| best | 40% | 6 | 13,676 | 52.2% | 59.2% | 9,040 | 33.9% | 3,939 | 3,510 | 267 | 1,324 | 86.8% | 82.4% | 5.32 (4.60 to 6.05) | 3.58 | -3.57 | 86.5% |
| best | 40% | 24 | 7,970 | 49.6% | 50.5% | 7,414 | 7.0% | 3,503 | 3,509 | 145 | 257 | 92.4% | 94.6% | 4.87 (3.99 to 5.77) | 9.52 | -9.47 | 99.5% |
| best | 40% | all | 53,813 | 57.2% | 63.6% | 22,633 | 57.9% | 10,684 | 8,565 | 599 | 2,785 | 87.0% | 85.0% | 4.87 (4.28 to 5.43) | 3.70 | -3.68 | 67.1% |
| best | 50% | 2 | 21,848 | 69.0% | 77.3% | 4,620 | 78.9% | 2,633 | 828 | 113 | 1,046 | 78.6% | 74.9% | 3.65 (3.37 to 3.90) | 0.42 | -0.42 | 0.0% |
| best | 50% | 6 | 6,373 | 61.3% | 69.4% | 4,059 | 36.3% | 1,873 | 1,143 | 131 | 912 | 82.2% | 74.3% | 2.52 (2.07 to 2.92) | 3.58 | -3.53 | 85.4% |
| best | 50% | 24 | 2,966 | 58.7% | 59.2% | 2,718 | 8.4% | 1,515 | 1,040 | 47 | 116 | 90.9% | 94.0% | 2.13 (1.66 to 2.57) | 9.50 | -9.43 | 99.6% |
| best | 50% | all | 31,187 | 66.5% | 74.0% | 11,397 | 63.5% | 6,021 | 3,011 | 291 | 2,074 | 82.8% | 79.2% | 2.77 (2.41 to 3.10) | 2.55 | -2.52 | 54.2% |
| near | 30% | 2 | 48,602 | 52.2% | 58.6% | 8,420 | 82.7% | 3,999 | 2,829 | 261 | 1,331 | 83.7% | 81.1% | 5.42 (5.04 to 5.77) | 0.50 | -0.50 | 0.0% |
| near | 30% | 6 | 28,778 | 42.9% | 48.4% | 9,314 | 67.6% | 2,926 | 4,485 | 398 | 1,505 | 84.9% | 79.6% | 3.66 (3.22 to 4.04) | 1.52 | -1.50 | 5.8% |
| near | 30% | 24 | 21,173 | 40.0% | 41.9% | 6,295 | 70.3% | 1,765 | 3,460 | 266 | 804 | 83.8% | 83.0% | 2.17 (1.80 to 2.46) | 1.60 | -1.58 | 6.6% |
| near | 30% | all | 98,553 | 46.9% | 52.0% | 24,029 | 75.6% | 8,690 | 10,774 | 925 | 3,640 | 84.2% | 81.0% | 3.75 (3.43 to 4.07) | 1.17 | -1.13 | 4.0% |
| near | 40% | 2 | 32,167 | 61.2% | 68.7% | 6,179 | 80.8% | 3,242 | 1,546 | 187 | 1,204 | 80.9% | 77.5% | 4.43 (4.10 to 4.72) | 0.45 | -0.42 | 0.0% |
| near | 40% | 6 | 13,676 | 52.2% | 59.2% | 4,497 | 67.1% | 1,542 | 1,698 | 208 | 1,049 | 79.7% | 72.0% | 1.93 (1.69 to 2.19) | 1.50 | -1.50 | 6.1% |
| near | 40% | 24 | 7,970 | 49.6% | 50.5% | 2,428 | 69.5% | 773 | 1,105 | 113 | 437 | 78.0% | 77.3% | 0.96 (0.77 to 1.14) | 1.58 | -1.58 | 6.5% |
| near | 40% | all | 53,813 | 57.2% | 63.6% | 13,104 | 75.6% | 5,557 | 4,349 | 508 | 2,690 | 80.0% | 75.6% | 2.44 (2.23 to 2.64) | 0.97 | -0.95 | 3.3% |
| near | 50% | 2 | 21,848 | 69.0% | 77.3% | 4,620 | 78.9% | 2,633 | 828 | 113 | 1,046 | 78.6% | 74.9% | 3.65 (3.37 to 3.90) | 0.42 | -0.42 | 0.0% |
| near | 50% | 6 | 6,373 | 61.3% | 69.4% | 2,027 | 68.2% | 714 | 560 | 97 | 656 | 74.2% | 62.9% | 0.89 (0.74 to 1.04) | 1.50 | -1.50 | 6.6% |
| near | 50% | 24 | 2,966 | 58.7% | 59.2% | 819 | 72.4% | 267 | 312 | 38 | 202 | 72.0% | 70.7% | 0.33 (0.25 to 0.42) | 1.58 | -1.57 | 7.9% |
| near | 50% | all | 31,187 | 66.5% | 74.0% | 7,466 | 76.1% | 3,614 | 1,700 | 248 | 1,904 | 76.7% | 71.2% | 1.63 (1.48 to 1.75) | 0.77 | -0.75 | 2.7% |
| latest | 30% | 2 | 48,602 | 52.2% | 58.6% | 8,420 | 82.7% | 3,838 | 2,776 | 314 | 1,492 | 81.5% | 78.6% | 5.11 (4.75 to 5.44) | 0.32 | -0.30 | 0.0% |
| latest | 30% | 6 | 28,778 | 42.9% | 48.4% | 19,400 | 32.6% | 6,516 | 9,431 | 727 | 2,726 | 86.1% | 82.2% | 8.39 (7.35 to 9.39) | 2.08 | -2.07 | 53.6% |
| latest | 30% | 24 | 21,173 | 40.0% | 41.9% | 19,646 | 7.2% | 6,444 | 10,775 | 629 | 1,798 | 86.5% | 87.6% | 8.43 (7.20 to 9.55) | 2.83 | -2.82 | 69.4% |
| latest | 30% | all | 98,553 | 46.9% | 52.0% | 47,466 | 51.8% | 16,798 | 22,982 | 1,670 | 6,016 | 85.4% | 83.8% | 7.31 (6.50 to 8.06) | 2.00 | -2.00 | 50.6% |
| latest | 40% | 2 | 32,167 | 61.2% | 68.7% | 6,179 | 80.8% | 3,091 | 1,501 | 232 | 1,355 | 78.1% | 74.3% | 4.14 (3.85 to 4.43) | 0.23 | -0.22 | 0.0% |
| latest | 40% | 6 | 13,676 | 52.2% | 59.2% | 9,040 | 33.9% | 3,324 | 3,378 | 399 | 1,939 | 80.7% | 74.1% | 4.24 (3.65 to 4.81) | 2.03 | -2.02 | 51.8% |
| latest | 40% | 24 | 7,970 | 49.6% | 50.5% | 7,414 | 7.0% | 2,780 | 3,387 | 267 | 980 | 81.5% | 83.2% | 3.64 (3.04 to 4.24) | 2.85 | -2.83 | 68.6% |
| latest | 40% | all | 53,813 | 57.2% | 63.6% | 22,633 | 57.9% | 9,195 | 8,266 | 898 | 4,274 | 80.3% | 77.1% | 4.01 (3.57 to 4.45) | 1.72 | -1.70 | 43.2% |
| latest | 50% | 2 | 21,848 | 69.0% | 77.3% | 4,620 | 78.9% | 2,488 | 800 | 141 | 1,191 | 75.3% | 71.2% | 3.40 (3.14 to 3.66) | 0.18 | -0.17 | 0.0% |
| latest | 50% | 6 | 6,373 | 61.3% | 69.4% | 4,059 | 36.3% | 1,526 | 1,087 | 187 | 1,259 | 74.8% | 64.4% | 1.94 (1.60 to 2.24) | 2.03 | -2.02 | 51.8% |
| latest | 50% | 24 | 2,966 | 58.7% | 59.2% | 2,718 | 8.4% | 1,152 | 998 | 89 | 479 | 77.1% | 79.1% | 1.54 (1.24 to 1.85) | 3.15 | -3.11 | 71.4% |
| latest | 50% | all | 31,187 | 66.5% | 74.0% | 11,397 | 63.5% | 5,166 | 2,885 | 417 | 2,929 | 75.5% | 70.6% | 2.29 (2.03 to 2.55) | 1.12 | -1.07 | 35.5% |

**E. `final_clean`: rules, by deadline.**

| Deadline | Rule | Flagged | Changes | Rescued | Needless | Made worse | Both missed | Net | murphy minus rule | Flags miss | Also murphy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | murphy | 80,062 | 21,249 | 10,228 | 5,430 | 663 | 4,928 | 4.62 (4.23 to 4.97) |  |  |  |
| 30 min | schedule | 159 | 59 | 18 | 14 | 6 | 21 | 0.01 (-0.00 to 0.01) | 4.61 (4.22 to 4.96) |  |  |
| 30 min | thumb | 4,550 | 1,442 | 941 | 24 | 4 | 473 | 0.45 (0.42 to 0.49) | 4.17 (3.80 to 4.51) | 98.5% | 4,442 |
| 60 min | murphy | 31,187 | 7,466 | 3,614 | 1,700 | 248 | 1,904 | 1.63 (1.48 to 1.75) |  |  |  |
| 60 min | schedule | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 1.63 (1.48 to 1.75) |  |  |
| 60 min | thumb | 1,380 | 564 | 387 | 15 | 1 | 161 | 0.19 (0.17 to 0.21) | 1.44 (1.31 to 1.56) | 97.2% | 1,308 |
| 120 min | murphy | 10,237 | 2,388 | 1,007 | 572 | 132 | 677 | 0.42 (0.38 to 0.48) |  |  |  |
| 120 min | schedule | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 (0.00 to 0.00) | 0.42 (0.38 to 0.48) |  |  |
| 120 min | thumb | 176 | 108 | 79 | 6 | 1 | 22 | 0.04 (0.03 to 0.05) | 0.38 (0.34 to 0.44) | 96.0% | 148 |

**F. `final_clean`: risk ranges, by deadline and forecast time.** Net per 1,000 forecasts in that range.

| Deadline | Forecast (h) | Risk | Forecasts | Predicted miss | True miss | Changes | Rescued | Needless | Made worse | Both missed | Net |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 min | 2 | <40 | 621,387 | 10.4% | 10.6% | 3,889 | 1,152 | 2,370 | 126 | 241 | 1.65 (1.51 to 1.79) |
| 30 min | 2 | 40-50 | 19,678 | 44.7% | 47.7% | 2,918 | 1,167 | 1,374 | 123 | 254 | 53.05 (49.46 to 56.67) |
| 30 min | 2 | 50-60 | 14,304 | 54.7% | 58.9% | 2,385 | 1,094 | 865 | 94 | 332 | 69.91 (64.82 to 75.32) |
| 30 min | 2 | 60+ | 34,782 | 77.9% | 84.2% | 6,834 | 4,176 | 864 | 135 | 1,659 | 116.18 (109.93 to 122.95) |
| 30 min | 6 | <40 | 651,446 | 12.7% | 13.9% | 12,419 | 3,968 | 7,257 | 408 | 786 | 5.46 (4.96 to 5.94) |
| 30 min | 6 | 40-50 | 19,059 | 44.5% | 49.1% | 7,407 | 2,817 | 3,456 | 281 | 853 | 133.06 (127.35 to 139.48) |
| 30 min | 6 | 50-60 | 10,730 | 54.6% | 60.3% | 4,256 | 1,834 | 1,496 | 151 | 775 | 156.85 (148.17 to 167.65) |
| 30 min | 6 | 60+ | 8,916 | 69.8% | 77.1% | 3,390 | 1,455 | 666 | 112 | 1,157 | 150.63 (137.99 to 166.59) |
| 30 min | 24 | <40 | 661,852 | 13.4% | 14.8% | 13,275 | 3,900 | 7,866 | 422 | 1,087 | 5.25 (4.70 to 5.76) |
| 30 min | 24 | 40-50 | 16,969 | 44.4% | 47.6% | 6,759 | 2,310 | 3,271 | 282 | 896 | 119.51 (112.96 to 124.89) |
| 30 min | 24 | 50-60 | 7,449 | 54.3% | 57.0% | 3,022 | 1,141 | 1,172 | 129 | 580 | 135.86 (126.01 to 145.42) |
| 30 min | 24 | 60+ | 3,881 | 67.4% | 69.5% | 1,362 | 528 | 367 | 42 | 425 | 125.23 (113.24 to 137.68) |
| 30 min | all | <40 | 1,934,685 | 12.2% | 13.2% | 29,583 | 9,020 | 17,493 | 956 | 2,114 | 4.17 (3.80 to 4.51) |
| 30 min | all | 40-50 | 55,706 | 44.6% | 48.1% | 17,084 | 6,294 | 8,101 | 686 | 2,003 | 100.67 (96.83 to 104.88) |
| 30 min | all | 50-60 | 32,483 | 54.6% | 58.9% | 9,663 | 4,069 | 3,533 | 374 | 1,687 | 113.75 (108.59 to 119.77) |
| 30 min | all | 60+ | 47,579 | 75.5% | 81.7% | 11,586 | 6,159 | 1,897 | 289 | 3,241 | 123.37 (116.72 to 130.38) |
| 60 min | 2 | <40 | 657,984 | 6.7% | 7.1% | 5,345 | 1,499 | 3,514 | 138 | 194 | 2.07 (1.91 to 2.21) |
| 60 min | 2 | 40-50 | 10,319 | 44.6% | 50.3% | 1,559 | 609 | 718 | 74 | 158 | 51.85 (46.24 to 57.11) |
| 60 min | 2 | 50-60 | 7,048 | 54.8% | 63.0% | 1,226 | 585 | 417 | 49 | 175 | 76.05 (67.65 to 84.63) |
| 60 min | 2 | 60+ | 14,800 | 75.8% | 84.1% | 3,394 | 2,048 | 411 | 64 | 871 | 134.05 (124.30 to 145.00) |
| 60 min | 6 | <40 | 676,475 | 8.0% | 8.9% | 13,780 | 3,446 | 9,120 | 422 | 792 | 4.47 (4.04 to 4.84) |
| 60 min | 6 | 40-50 | 7,303 | 44.4% | 50.4% | 2,470 | 828 | 1,138 | 111 | 393 | 98.18 (89.02 to 109.17) |
| 60 min | 6 | 50-60 | 3,459 | 54.3% | 62.1% | 1,153 | 390 | 408 | 53 | 302 | 97.43 (85.63 to 110.15) |
| 60 min | 6 | 60+ | 2,914 | 69.6% | 78.0% | 874 | 324 | 152 | 44 | 354 | 96.09 (76.97 to 115.34) |
| 60 min | 24 | <40 | 682,181 | 8.3% | 9.5% | 10,575 | 2,417 | 7,073 | 350 | 735 | 3.03 (2.69 to 3.37) |
| 60 min | 24 | 40-50 | 5,004 | 44.2% | 45.4% | 1,609 | 506 | 793 | 75 | 235 | 86.13 (75.05 to 96.32) |
| 60 min | 24 | 50-60 | 1,968 | 54.2% | 55.5% | 598 | 194 | 238 | 28 | 138 | 84.35 (68.97 to 100.81) |
| 60 min | 24 | 60+ | 998 | 67.5% | 66.5% | 221 | 73 | 74 | 10 | 64 | 63.13 (45.40 to 83.66) |
| 60 min | all | <40 | 2,016,640 | 7.7% | 8.5% | 29,700 | 7,362 | 19,707 | 910 | 1,721 | 3.20 (2.90 to 3.47) |
| 60 min | all | 40-50 | 22,626 | 44.5% | 49.2% | 5,638 | 1,943 | 2,649 | 260 | 786 | 74.38 (70.21 to 79.56) |
| 60 min | all | 50-60 | 12,475 | 54.5% | 61.6% | 2,977 | 1,169 | 1,063 | 130 | 615 | 83.29 (76.90 to 90.01) |
| 60 min | all | 60+ | 18,712 | 74.4% | 82.3% | 4,489 | 2,445 | 637 | 118 | 1,289 | 124.36 (114.54 to 135.84) |
| 120 min | 2 | <40 | 678,614 | 3.9% | 4.1% | 2,888 | 742 | 1,943 | 103 | 100 | 0.94 (0.85 to 1.05) |
| 120 min | 2 | 40-50 | 4,410 | 44.6% | 52.9% | 933 | 376 | 416 | 46 | 95 | 74.83 (65.38 to 84.84) |
| 120 min | 2 | 50-60 | 2,840 | 54.7% | 65.1% | 646 | 287 | 212 | 39 | 108 | 87.32 (71.75 to 101.71) |
| 120 min | 2 | 60+ | 4,287 | 73.4% | 82.5% | 1,092 | 526 | 162 | 43 | 361 | 112.67 (97.07 to 132.15) |
| 120 min | 6 | <40 | 685,366 | 4.3% | 4.8% | 3,238 | 660 | 2,244 | 131 | 203 | 0.77 (0.66 to 0.88) |
| 120 min | 6 | 40-50 | 2,475 | 44.4% | 49.4% | 566 | 159 | 267 | 42 | 98 | 47.27 (36.30 to 59.93) |
| 120 min | 6 | 50-60 | 1,196 | 54.4% | 61.9% | 289 | 81 | 105 | 25 | 78 | 46.82 (32.23 to 63.31) |
| 120 min | 6 | 60+ | 1,114 | 69.1% | 77.3% | 248 | 79 | 48 | 17 | 104 | 55.66 (34.21 to 77.82) |
| 120 min | 24 | <40 | 688,100 | 4.3% | 5.1% | 1,204 | 195 | 885 | 50 | 74 | 0.21 (0.16 to 0.28) |
| 120 min | 24 | 40-50 | 1,251 | 44.2% | 41.1% | 166 | 40 | 95 | 7 | 24 | 26.38 (16.09 to 36.90) |
| 120 min | 24 | 50-60 | 506 | 54.3% | 49.4% | 75 | 24 | 29 | 3 | 19 | 41.50 (22.98 to 63.06) |
| 120 min | 24 | 60+ | 294 | 67.7% | 57.5% | 38 | 10 | 16 | 5 | 7 | 17.01 (-10.40 to 43.75) |
| 120 min | all | <40 | 2,052,080 | 4.2% | 4.7% | 7,330 | 1,597 | 5,072 | 284 | 377 | 0.64 (0.58 to 0.72) |
| 120 min | all | 40-50 | 8,136 | 44.5% | 50.0% | 1,665 | 575 | 778 | 95 | 217 | 59.00 (51.45 to 66.08) |
| 120 min | all | 50-60 | 4,542 | 54.6% | 62.5% | 1,010 | 392 | 346 | 67 | 205 | 71.55 (59.75 to 83.91) |
| 120 min | all | 60+ | 5,695 | 72.3% | 80.2% | 1,378 | 615 | 226 | 65 | 472 | 96.58 (83.84 to 114.14) |

**G. `final_clean`: by aircraft inputs as recorded, and by recorded turn.** 60-minute deadline.

| Forecast | Kind | Forecasts | Predicted miss | True miss | Cancelled | Flagged |
|---|---|---|---|---|---|---|
| 2 h | delay blank | 7,330 | 44.4% | 47.6% | 33.6% | 2,817 |
| 2 h | no tail | 1,647 | 32.0% | 98.5% | 98.1% | 458 |
| 2 h | normal | 667,946 | 8.7% | 8.2% | 0.7% | 17,447 |
| 2 h | overlap | 13,228 | 16.8% | 64.2% | 16.1% | 1,126 |
| 6 h | delay blank | 3,754 | 31.6% | 32.2% | 18.2% | 605 |
| 6 h | no tail | 1,647 | 26.9% | 98.5% | 98.1% | 295 |
| 6 h | normal | 671,250 | 8.6% | 8.5% | 0.9% | 4,887 |
| 6 h | overlap | 13,500 | 14.8% | 63.6% | 16.1% | 586 |
| 24 h | no tail | 690,151 | 8.7% | 9.9% | 1.6% | 2,966 |

| Recorded turn (min) | Forecasts | Predicted miss | True miss | Flagged |
|---|---|---|---|---|
| <-60 | 10,292 | 17.9% | 68.7% | 899 |
| -60 to -16 | 11,644 | 14.9% | 66.9% | 620 |
| -15 to -1 | 5,066 | 13.9% | 45.5% | 243 |
| 0-14 | 6,340 | 30.1% | 35.0% | 712 |
| 15-29 | 16,702 | 15.4% | 17.0% | 590 |
| 30-44 | 234,374 | 9.2% | 10.2% | 4,624 |
| 45-59 | 303,280 | 9.1% | 9.1% | 6,795 |
| 60-119 | 383,574 | 8.9% | 8.0% | 8,328 |
| 120+ | 405,736 | 7.9% | 7.3% | 4,657 |

**H. `final_clean`: scores against each comparison.** Lower is better; a positive difference means `final_clean` scored worse.

| Forecasts | Difference from `final_leads` | Difference from climatology |
|---|---|---|
| all | 0.0057 (0.0053 to 0.0060) | -0.0414 (-0.0433 to -0.0395) |
| 2 h | 0.0085 (0.0080 to 0.0090) | -0.0648 (-0.0673 to -0.0620) |
| 6 h | 0.0082 (0.0076 to 0.0087) | -0.0345 (-0.0364 to -0.0326) |
| 24 h | 0.0003 (0.0002 to 0.0004) | -0.0249 (-0.0266 to -0.0233) |
| normal, under 24 h | 0.0004 (0.0003 to 0.0005) | -0.0468 (-0.0484 to -0.0452) |
| overlap, under 24 h | 0.2929 (0.2825 to 0.3066) | -0.0709 (-0.0853 to -0.0565) |
| no tail, under 24 h | 0.9726 (0.9049 to 1.0422) | -0.3258 (-0.4123 to -0.2406) |
| delay blank, under 24 h | -0.0006 (-0.0033 to 0.0016) | -0.2603 (-0.2891 to -0.2327) |

**I. `final_clean`: Brier by round, 2025 grid at trained leads.** 25: 0.56844, 50: 0.56285, 75: 0.56069, 100: 0.55968, 125: 0.55911, 150: 0.55875, 175: 0.55852, 200: 0.55830, 225: 0.55819, 250: 0.55821, 275: 0.55819, 300: 0.55822.
