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

**Open, for the owner:** how to treat these flights in the refit. Not fitted yet.

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
