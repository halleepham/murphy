# 0011 — Forecasts for alternative flights (PLAN layer 3)

Date: 6 Nov 2026. 2025 only; 2026 outcomes were not read. Code:
`scripts/alternatives_measure.py`. Results (gitignored): `data/forecast/alternatives/`,
`data/forecast/logs/alt_counts.log`, `alt_grid.log`. `final_l2` at round 300.

## Finding 1 — How many alternatives a trip has

20,000 sampled 2025 flights. An alternative here: same origin and destination, any
carrier, leaving at least 1 h after the forecast (an AI assumption for time to rebook),
and scheduled to arrive no later than the current flight's scheduled arrival + X h.
Schedule only, so choosing one reveals no outcome. Not covered: connections, nearby
airports, seats.

| Arrives by | Forecast at | Mean alternatives | Median | Share with none |
|---|---|---|---|---|
| Same time | 2 h | 0.2 | 0 | 81.8% |
| Same time | 6 h | 1.5 | 1 | 35.9% |
| Same time | 24 h | 6.3 | 5 | 5.7% |
| + 3 h | 2 h | 1.1 | 1 | 45.3% |
| + 3 h | 6 h | 2.3 | 2 | 25.9% |
| + 3 h | 24 h | 7.2 | 6 | 5.6% |
| + 6 h | 2 h | 2.0 | 1 | 29.4% |
| + 6 h | 6 h | 3.3 | 2 | 18.5% |
| + 6 h | 24 h | 8.1 | 6 | 5.6% |

Read: close to departure, many trips have no direct alternative at all. Murphy must
be able to say so.

An alternative's own lead is rarely one the model trained on. Arriving by + 6 h, all
three forecast times pooled: 21.4% of alternatives fall within an hour of 2, 6 or
24 h; 11.9% are 25 h or more out.

## Finding 2 — What building the inputs costs

| Rows | Climatology | A | B | C | D | E | Total |
|---|---|---|---|---|---|---|---|
| 550 | 275 s | 0 s | 320 s | 125 s | 19 s | 5 s | 12.4 min |
| 330,000 | 259 s | 1 s | 278 s | 112 s | 18 s | 8 s | 11.2 min |

Scoring 330,000 rows took 37 s including loading.

Read: the cost is scanning the flight history, not the number of rows. So a large batch
is cheap, and one live request is not possible with this code. Replaying every sampled
2025 trip means about 5.6M rows at + 6 h (scaled up from the sample, not run). That
looks feasible but is untested at that size.

**Check:** at 24, 6 and 2 h the new build equals the saved inputs in every column of
all five groups, on 90,000 rows.

## Finding 3 — The model at leads it never trained on

30,000 flights, each forecast at all 11 leads, so every row below has the same flights
and the same observed late rate, 0.2337. Late = 15+ minutes, cancelled or diverted.
"Against climatology" is the multiclass difference, with a 95% interval resampling
whole days. Gap is the weighted mean calibration gap for late.

| Lead | Trained? | Multiclass | Against climatology | Late Brier | Mean predicted late | Gap |
|---|---|---|---|---|---|---|
| 1 h | | 0.5167 | −0.0833 (−0.0875 to −0.0790) | 0.1143 | 0.2340 | 0.0108 |
| 2 h | yes | 0.5260 | −0.0741 (−0.0784 to −0.0700) | 0.1211 | 0.2286 | 0.0084 |
| 3 h | | 0.5412 | −0.0589 (−0.0628 to −0.0550) | 0.1326 | 0.2282 | 0.0078 |
| 4 h | | 0.5507 | −0.0494 (−0.0532 to −0.0456) | 0.1411 | 0.2274 | 0.0066 |
| 6 h | yes | 0.5569 | −0.0431 (−0.0467 to −0.0397) | 0.1456 | 0.2221 | 0.0118 |
| 9 h | | 0.5621 | −0.0380 (−0.0413 to −0.0349) | 0.1496 | 0.2188 | 0.0161 |
| 12 h | | 0.5678 | −0.0323 (−0.0352 to −0.0293) | 0.1540 | 0.2304 | 0.0070 |
| 18 h | | 0.5700 | −0.0301 (−0.0332 to −0.0271) | 0.1538 | **0.2615** | 0.0288 |
| 24 h | yes | 0.5767 | −0.0235 (−0.0260 to −0.0210) | 0.1608 | 0.2198 | 0.0142 |
| 30 h | | 0.5915 | −0.0087 (−0.0108 to −0.0067) | 0.1735 | **0.1740** | 0.0598 |
| 36 h | | 0.5952 | −0.0050 (−0.0068 to −0.0032) | 0.1765 | **0.1824** | 0.0526 |

Read: accuracy declines smoothly with lead from 1 to 24 h, so scoring off-lead is not
broken. The problem is **bias that depends on lead**: at 18 h the model overstates risk
by 2.8 points; at 30 h it understates it by 6.0. The change rule compares two flights,
usually at different leads, so the bias alone can make a later flight look safer.
Example: the current flight at 6 h averages 0.222 and an alternative at 30 h 0.174,
though the true late rate is the same 0.2337.

**Likely cause: inferred, not tested.** The model can split lead time only at 4 and 15 h
(printed from the saved model), so 18 h is treated as 24 h and 9 h as 6 h. At "24 h" in
training the aircraft inputs were always blank, and at 18 h they are filled in. Why 30 h
reads so low is not explained.

## Decision 1 — What to do about lead bias

**Options:**
(a) Keep `final_l2`, cap alternatives at 24 h, and correct the bias per lead band using
2025. No refit, but the correction and the recommendation cutoffs would both come from
2025.
(b) Refit with leads spread across 1–24 h: the same 3 rows per flight, with each flight's
3 leads chosen by a fixed hash from 1, 2, 3, 4, 6, 9, 12, 18, 24. Then compare with
`final_l2` on the same 30,000 flights. Keep the 24 h cap.
(c) Compare flights only at matched leads. Rejected by the AI: a replay moment fixes
one as-of time, so the leads differ in practice.

**AI recommended (b). Owner: accepted, 6 Nov 2026,** after asking what a refit is. The
tests were stated before the fit: the average predicted late rate stays close to the
true rate at every lead, and overall accuracy is at least as good, especially at 24, 6
and 2 h. If not, keep `final_l2` and use (a).

**A bug caught before fitting.** The first version chose leads with DuckDB
`hash(flight_id, lead)`, which is not independent across leads: half the flights got
exactly {9, 12, 24} and only 9 of 84 possible sets appeared. Seen in the 2-round test's
row counts. Replaced with `md5(flight_id || '|' || lead)`, which gives all 84 sets
(15,238 to 15,974 flights each), and the script now stops if the spread is uneven.

## Finding 4 — The refit against `final_l2`

**Status:** measured 6 Nov 2026. `scripts/refit_leads.py`, `final_leads` in
`data/forecast/alternatives/` (gitignored): lr 0.05, 63 leaves, l2 10, 3,931,131 rows,
300 rounds, 2,953 s, largest leaf 3.4. Same 30,000 2025 flights at each lead; true late
rate 0.2337. The new model did not train on 5, 8, 15 or 21 h, which check leads between
its training points. Difference is new minus old, multiclass, with a 95% interval
resampling whole days; below zero favors the new model. Bias is mean predicted late
minus the true rate.

| Lead | Multiclass, new | Old | Difference | Bias, new | Old | Gap, new | Old |
|---|---|---|---|---|---|---|---|
| 1 h | 0.5153 | 0.5167 | −0.0014 (−0.0020 to −0.0008) | −0.0026 | +0.0002 | 0.0079 | 0.0108 |
| 2 h | 0.5262 | 0.5260 | +0.0003 (−0.0002 to +0.0008) | −0.0049 | −0.0051 | 0.0083 | 0.0084 |
| 3 h | 0.5413 | 0.5412 | +0.0001 (−0.0004 to +0.0006) | −0.0088 | −0.0055 | 0.0093 | 0.0078 |
| 4 h | 0.5505 | 0.5507 | −0.0002 (−0.0006 to +0.0003) | −0.0108 | −0.0063 | 0.0108 | 0.0066 |
| 5 h, unseen | 0.5549 | 0.5547 | +0.0002 (−0.0002 to +0.0007) | −0.0130 | −0.0074 | 0.0130 | 0.0082 |
| 6 h | 0.5572 | 0.5569 | +0.0002 (−0.0002 to +0.0007) | −0.0135 | −0.0116 | 0.0135 | 0.0118 |
| 8 h, unseen | 0.5594 | 0.5603 | −0.0009 (−0.0014 to −0.0003) | −0.0138 | −0.0156 | 0.0142 | 0.0165 |
| 9 h | 0.5601 | 0.5621 | −0.0020 (−0.0027 to −0.0013) | −0.0140 | −0.0149 | 0.0151 | 0.0161 |
| 12 h | 0.5618 | 0.5678 | −0.0060 (−0.0070 to −0.0051) | −0.0123 | −0.0034 | 0.0126 | 0.0070 |
| 15 h, unseen | 0.5621 | 0.5691 | −0.0071 (−0.0081 to −0.0061) | −0.0072 | +0.0148 | 0.0074 | 0.0171 |
| 18 h | 0.5618 | 0.5700 | −0.0082 (−0.0095 to −0.0068) | −0.0066 | +0.0278 | 0.0066 | 0.0288 |
| 21 h, unseen | 0.5627 | 0.5685 | −0.0058 (−0.0071 to −0.0045) | −0.0094 | +0.0276 | 0.0100 | 0.0286 |
| 24 h | 0.5765 | 0.5767 | −0.0002 (−0.0006 to +0.0002) | −0.0144 | −0.0139 | 0.0147 | 0.0142 |
| 30 h, outside the cap | 0.5896 | 0.5915 | −0.0019 (−0.0023 to −0.0015) | −0.0518 | −0.0597 | 0.0519 | 0.0598 |
| 36 h, outside the cap | 0.5908 | 0.5952 | −0.0044 (−0.0050 to −0.0039) | −0.0450 | −0.0513 | 0.0452 | 0.0526 |

Read:

- **The tests pass.** From 1 to 24 h, the distance between the highest and lowest bias
  falls from 4.3 points to 1.2. No lead is significantly worse in accuracy; 2, 6 and
  24 h are unchanged within their intervals, and 1 h and 8–21 h improve.
- **The swing is gone, but a low bias remains.** Every lead from 1 to 24 h now reads low,
  by 0.3 to 1.4 points. The change rule compares two flights, so a shared low bias
  matters less than a difference, but the chance of missing a deadline is still
  understated. Not corrected yet.
- **Worse in places.** From 3 to 6 h and at 12 h, the old model's bias and gap were
  smaller. A 1.2-point difference between leads can still favor one flight.
- **Beyond 24 h is still poor,** with a 4.5 to 5.2-point understatement, so the cap stays.

**Limits:** 30,000 flights, not all of 2025. The bias figures have no intervals. They
are averages over all flights, so bias could differ on the risky flights where the rule
acts; not checked. 300 rounds is the last round tried, and 2025 chose it, which flatters
the score. Each flight also lost a third of its practice at any one lead, with no
measurable cost at 2, 6 or 24 h. `0010`'s figures on all of 2025 describe `final_l2`,
not this model.

**Result:** by the rule set beforehand, `final_leads` replaces `final_l2` for the
recommendation layer. Alternatives stay capped at 24 h. `final_l2` stays saved. If
layer 4 adds inputs, the single retrain uses this lead design.

## Decision 2 — How the app gets forecasts

**AI recommended:** precompute forecasts in batch for the replay trips and demo days;
the app reads them. Murphy is a replay, so a live build is not needed, and at about 11
minutes one is not possible anyway. Rejected: rewriting inputs as fast lookups, which is
more code than a replay app needs. Limit to state: the app answers only for precomputed
days.

**Owner, 6 Nov 2026:** accepted in principle, but on hold. The app's features are not
decided, so nothing is precomputed yet.
