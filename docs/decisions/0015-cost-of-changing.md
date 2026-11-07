# 0015 — The cost of changing

Date: 6 Nov 2026. **Status: decided by the owner. The fare table and its lookup are built; the app is not.**
Code: `scripts/build_fares.py`, `murphy/cost.py`, `scripts/replay_change_kinds.py`. 2025 only; 2026 stays sealed. The rule measured is `0014` Decision 4 at a
60-minute deadline, with `final_clean`. It reproduces the recorded 4,489 changes.

## Human design

The owner accepted route fares plus a tested range for the last-minute premium, to show how murphy weighs cost
(`0010` Decision 5). It is not an accurate picture of fares, and no result is reported as real dollars saved.
The owner also accepted an "advice flips" check once a cost estimate exists (`0014` Decision 1).

## Finding 1 — Most advised changes move to another airline

| Alternative | Changes | Rescued | Needless | Made worse | Both missed |
|---|---|---|---|---|---|
| Same marketing carrier | 836 | 355 | 153 | 36 | 292 |
| Another carrier | 3,653 | 2,090 | 484 | 82 | 997 |
| All | 4,489 | 2,445 | 637 | 118 | 1,289 |

| Forecast lead | Changes | Same carrier |
|---|---|---|
| 2 h | 3,394 | 10.7% |
| 6 h | 874 | 43.5% |
| 24 h | 221 | 42.1% |

**Read:** 81.4% of changes (3,653 of 4,489) leave the airline, so they usually mean buying another ticket, not paying a
fare difference. The earlier framing, "fare difference or new ticket", treats as equal two cases that are not.
All alternatives are on the same route, so a route fare alone gives the same number for both flights. The estimated cost
therefore comes from the alternative's fare and the premium, not from a difference between two average fares.

**Limits:** carriers are marketing carriers in BTS. Past schedules do not show seats (`0013`). Whether the old ticket can be
credited or refunded is not modeled.

## Finding 2 — The gain sets how much a deadline must be worth

If a change costs C and raises the chance of making the deadline by g, it pays only if missing the deadline would
cost the traveler more than C / g. This assumes the forecast chances are right. Measured on the same 4,489 changes:

| Percentile of gain | 10th | 25th | 50th | 75th | 90th |
|---|---|---|---|---|---|
| Gain in chance, points | 21.2 | 31.4 | 50.6 | 66.1 | 77.7 |
| Break-even, as a multiple of the cost | 4.71 | 3.19 | 1.98 | 1.51 | 1.29 |

The multiples are 1 / gain, listed so each sits under its own gain. Example: at the median, a change is worth it if missing
the deadline would cost the traveler more than 1.98 times the cost of the change. This needs no dollar figure.

## Finding 3 — The data

| Candidate | Status | Note |
|---|---|---|
| BTS DB1BMarket 2025 Q2 | **Checked 6 Nov 2026 by a header request:** exists, 110,296,405 bytes zipped | Not downloaded. Q3 2025 returns 404 |
| BTS DB1BMarket 2025 Q1 | Checked the same way: exists, 94,587,240 bytes | Not downloaded |
| BTS DB1C (40% monthly sample) | **Not verified.** Seen only in search summaries, said to run to Jul 2026; the BTS pages returned 403 to our fetch | Its 2026 months would break the seal |

The DB1B is a 10% sample of tickets on what was paid, mostly bought well ahead. It does not show what a seat costs hours before
departure. **The premium is not measured:** a search found no reliable figure for how far fares rise close to departure. Any range is
therefore an assumption, and stated as one. Change policies vary by fare and airline; the sources found were weak and in conflict, so
none is relied on here. Only 1,116 routes and 1,768 route-and-carrier pairs occur in the changes, so coverage can be measured once
downloaded.

## Finding 4 — How a flight gets a fare

DB1BMarket 2025 Q2 (`data/raw/db1b/`, gitignored; sha256 `e1372718…7db2`): 8,450,420 rows, 16,142,726 passengers. It has route, ticketing
carrier, fare and passengers. **It has no flight number, date or time, only the quarter, so no flight has its own fare.** Every flight
on one route and airline gets the same estimate. `MktFare` is the itinerary's yield times the miles in this market, so a round trip's fare is split across its directions.

Filters: one coupon (nonstop) and not bulk, 4,786,216 rows; of those, 118,247 under $10 removed. Rows, not passengers, are counted. The
result is 13,620 route-and-carrier pairs; 2,742 have under 20 rows and 1,415 under 5.

Largest route in the changes, LGA to ORD:

| Carrier | Rows | Lower quartile | Median | Upper quartile |
|---|---|---|---|---|
| UA | 2,617 | $171.50 | $269.07 | $405.50 |
| DL | 2,464 | $171.88 | $246.50 | $351.00 |
| AA | 2,044 | $155.67 | $231.00 | $334.00 |
| NK | 402 | $59.00 | $92.75 | $146.12 |

Coverage of the 4,489 changes, with a minimum of 20 rows: 4,452 (99.2%) have the pair, 30 fall back to the route, 7 have neither and
get no estimate. Over the 4,452, the median of the alternative's median fare is $221, with 10th and 90th percentiles of $158.78 and $292.50.

**Limit on sorting by price:** flights on one airline and route have the same fare, so the sort compares airlines, not departure times.

## Decision 1 — Show the estimated cost and nothing more

**Human design:** the owner accepted fares plus a tested premium range (`0010` Decision 5) and a flip check (`0014` Decision 1).

**AI recommended:** an optional "what is the deadline worth to you?" box, break-even figures and a verdict line based on them; first with a
premium of 1× to 3×, then with no premium.

**Owner, 6 Nov 2026, rejected all of that:**

- **No premium.** Nothing supports 1× to 3×; it was an AI guess. Research reports that fares rise near departure ([Williams](https://cowles.yale.edu/sites/default/files/2022-09/d2103-u3.pdf),
  [Hortaçsu, Öry and Williams](https://www.ftc.gov/system/files/ftc_gov/pdf/williamshortacsuory.pdf); seen in search summaries, not read in full), but gives no multiple for our median. This reverses the premium in `0010` Decision 5.
- **No verdict line, value box or break-even.** Murphy cannot know what a deadline is worth to someone. Finding 2 stays on record as measured, unused.
- **Cost is shown, not weighed.** Each alternative states the estimated cost of a new ticket. The objective in `docs/PLAN.md` is edited to match, so it no longer says benefit outweighs cost.
- **Kept from the owner's earlier idea:** an option to sort alternatives by price. Not built. The default stays by chance of making the deadline.

**Consequences:** the cost cannot flip the advice, so the flip check in `0014` Decision 1 and the "does cost help?" comparison are dropped. "Holds up" tests the cutoffs only.

**AI choices inside this, for the owner to overrule:**

- One figure, the median, not a range.
- A thin route falls back from airline to route to **no estimate**. The labeled fare by distance is dropped, since it would affect 7 of 4,489 changes and would be one more invented method.
- Every alternative is labeled the same way, as a new ticket. A same-airline change (836 of 4,489) may cost only a fare difference, on which the data says nothing. The label should say so.
- The minimum is 20 rows and the floor is $10. Neither is tuned; both are judgment calls.

## Tests

`tests/test_cost.py`: airline figure with its source, the exact minimum, fallback to the route, a thin route and an unknown route that give no number (the deliberate thin-evidence cases), and direction. The simpler comparison for the recommendation stage stays the rule of thumb (`0014`).
