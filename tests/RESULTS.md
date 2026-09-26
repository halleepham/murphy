# Test results

Eighteen cases covering the path a traveller actually takes — give an itinerary,
correct it, get a forecast, inspect the evidence, compare routes — including the
five where the system is supposed to refuse or hold back.

```
.venv/bin/pytest -q
```

All eighteen pass. Tests that query the flight table skip with a stated reason
until it is built; the parser tests skip without an API key. A missing
prerequisite therefore reads as `skipped`, naming what is absent, rather than as
a wall of errors.

Failure cases are marked **⚠**. There are five.

---

## Retrieval and evidence

### 1. Exact match, reported without caveat
**In** — DL, BOS→ATL, November, 06:00
**Out** — 29 comparable flights, exact match. **p10 −27 / p50 −12 / p90 +16.**
**Why correct** — every returned row is BOS→ATL, 29 clears the threshold of 20,
and the output is a range rather than a point: `p10 < p50 < p90`.

### 2. Small sample, answered with a caution
**In** — WN, MCI→DEN, November, 07:00
**Out** — 19 flights, amber banner, *"treat the range as indicative."*
**Why correct** — 19 sits between the minimum of 10 and the comfortable threshold
of 20. Refusing here would be unhelpful; saying nothing about it would be
dishonest.

### 3. Every month is present
**In** — DL, BOS→ATL, **March**
**Out** — answers on an exact match. All twelve months confirmed present.
**Why correct** — **this test used to assert the opposite.** The Kaggle-derived
source had no March at all, so a March query had to fall down the fallback
ladder. Rebuilding from BTS closed the hole, and the test now guards against a
source silently dropping a month again.

### 3b. Cancellations are excluded from the range and reported separately
**In** — DL, BOS→ATL, November
**Out** — the range covers only flights that operated; the cancellation count and
rate are returned alongside it.
**Why correct** — a cancelled flight never arrived, so it says nothing about how
late you land, but it is something the traveller should know. **The previous
dataset made this impossible** — every cancelled flight had been stripped out of
it.

### 4. ⚠ No evidence at all — refuses to forecast
**In** — DL, BOS→**ANC**, November
**Out** — **no range.** *"Only 0 comparable flights after widening the match as
far as it goes. That is not enough to quote a range, so Murphy will not guess
one."* Ladder trace shown: `exact = 0 · wider = 0 · any month = 0 · any carrier,
any month = 0`.
**Why correct** — there is no Boston–Anchorage service. Every quantile is `None`;
nothing is interpolated or borrowed from a similar route. Refusing is designed
behaviour, and the traveller can see all four rungs were tried.

### 5. ⚠ A larger sample with a worse match gets *lower* confidence
**In** — DL, BOS→**HNL**, November
**Out** — **196 flights**, far more than the 29 behind a confident answer, yet
reported as **limited**.
**Why correct** — reaching 196 required dropping both the carrier and the month,
so those flights describe this itinerary poorly. An earlier build reported this
as high confidence purely because the count was large. That was a real defect,
found by using the app, and this test guards its return.

### 6. Weather is shown as evidence, and withheld when it would be noise
**Out** — the wet/dry split is displayed only when both groups have at least
three flights; the counts reconcile exactly with the sample.
**Why correct** — reporting a statistic drawn from one rainy day would be worse
than reporting nothing.

---

## Validation — the parse is a proposal, not a fact

### 7. ⚠ Codes that do not exist are rejected
**In** — carrier `ZZ`, destination `QQQ`
**Out** — not forecast. Both flagged against the 15 carriers and 305 airports
actually present.
**Why correct** — a plausible-looking but unusable code is caught at the boundary
instead of silently returning zero rows.

### 8. An incomplete leg cannot reach retrieval
**In** — a leg with no origin
**Out** — no query is built; the interface asks for it.
**Why correct** — it is structurally impossible for an incomplete itinerary to
produce a forecast.

---

## The parser

### 9. Multi-leg confirmation with a regional operator
**In** — the two-leg Delta confirmation
**Out** — both legs parsed. Leg 2 is flown by Endeavor but ticketed as Delta:
`carrier` stays **DL**, `operated_by` captures the text verbatim.
**Why correct** — the data is keyed on the operating carrier code and the
confirmation shows the marketing one. Letting the operator overwrite the carrier
would change which flights are retrieved. The passenger name is deliberately not
extracted; the forecast does not need it.

### 10. ⚠ Missing values become null, never guesses
**In** — *"UA 328 Denver to Chicago, Depart 7:45 AM"*
**Out** — `origin`, `dest` and `departure_date` all **null**; the interface asks.
**Why correct** — "Denver" and "Chicago" are city names, not airport codes, and
ORD versus MDW is a real ambiguity. The rule is enforced in code, not only in the
prompt.

### 11. ⚠ Unrelated text produces no itinerary
**In** — an email about dinner
**Out** — zero legs, and a message saying nothing was invented.

### 12. End to end
**In** — the Delta confirmation, through the whole slice
**Out** — both legs produce evidence-backed ranges with real record IDs and the
SQL that produced them. The displayed SQL uses a repository-relative path.

---

## Route planning

### 13. Top-k returns real choices, not near-duplicates
**Out** — at least three itineraries, no two sharing a hub and carrier, including
both a nonstop and a connection.
**Why correct** — the instructor feedback is explicit that k alternatives must be
meaningfully distinct. Two United nonstops an hour apart are one option.

### 14. Every leg of every route is backed by evidence
**Out** — each leg carries ordered quantiles, an evidence count above the
minimum, and a confidence label. Door-to-door times are positive and ordered.
**Why correct** — a route is only as grounded as its legs.

### 15. The traveller's own route is identified exactly once
**In** — a booked Delta nonstop
**Out** — exactly one route marked.
**Why correct** — matching on hub alone marked *every* nonstop as theirs. Keyed on
carrier and hub now.

### 16. Connection risk is counted, never modelled
**Out** — every layover respects the 45-minute minimum and the ceiling; the risk
is a share between 0 and 1 drawn from a stated number of real flights.
**Why correct** — it is the observed proportion of inbound flights that arrived
too late to transfer, not an estimate.

### 17. ⚠ An unknown airport yields no route
**In** — BOS→`QQQ`
**Out** — no itineraries. No path, no invention.

---

## What testing changed

Defects found by running the system rather than reading it. Each is now guarded.

1. **Confidence ignored match quality.** 196 loosely matched flights reported as
   high confidence purely because the count was large. *(Test 5)*
2. **A reduced-confidence answer gave no reason.** A 19-flight exact match turned
   the banner amber while the text explained nothing. *(Test 2)*
3. **Route ranking used delay rather than door-to-door time.** Delay is measured
   against each itinerary's own schedule, so a two-stop route "arriving 20
   minutes early" outranked a nonstop landing two hours sooner.
4. **The traveller's own route was only marked when it happened to place in the
   top five.** On the sample itinerary it never appeared at all.
5. **Switching samples showed the previous forecast.** Streamlit stores widget
   values by key and a stored value beats the one passed in, so clearing the
   itinerary left the old flight's text in the boxes — which then overwrote what
   had just been parsed.
6. **Three flaws in the route evaluation** — an all-trunk pair set where both
   methods always chose the nonstop and the comparison measured nothing;
   candidates drawn from the whole day, comparing trips no traveller would treat
   as substitutes; and missed connections dropped from the time statistics,
   which deleted each method's own failures and flattered the baseline.

## What still needs work

- **Neither forecast is calibrated.** An 80% interval should contain the outcome
  80% of the time; retrieval manages 70.9% and the model 77.0%. Both are
  overconfident.
- **Cancellation is reported, not predicted.** The data now supports predicting
  it; the system does not yet.
- **The model assumes typical weather**, so it will read optimistic on a bad day.
- **Legs are scored independently**, so correlated delay at a shared hub is
  invisible.
