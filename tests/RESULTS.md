# Test results

Twelve cases covering the path a traveller actually takes — paste, parse,
correct, retrieve, inspect — including the four where the system is supposed to
refuse or hold back.

Run them with:

```
.venv/bin/pytest -q
```

All twelve pass. Tests 1–8 run offline against the Parquet table. Tests 9–12
need an API key and skip cleanly without one, so the suite is runnable from a
fresh clone.

Failure cases are marked **⚠**. There are four.

---

## Retrieval: the four evidence bands

### 1. Exact match, confident

**Input** — DL, BOS→ATL, November, departing 06:00
**Result** — 29 comparable flights, exact match on route + carrier + month +
departure-hour bin. Arrival delay **p10 −27 / p50 −12 / p90 +16** minutes.
Reported without caveat.

**Why this is correct** — every returned row is BOS→ATL, and 29 flights clears
the threshold of 20 for an unqualified answer. The output is a range, not a
point: `p10 < p50 < p90`.

### 2. Small sample, answered with a caution

**Input** — WN, MCI→DEN, November, departing 07:00
**Result** — 19 flights. **p10 −22 / p50 −4 / p90 +9.** Amber banner:
*"That is a small sample, so treat the range as indicative."*

**Why this is correct** — 19 sits between the minimum of 10 and the comfortable
threshold of 20. The range is still shown, because refusing here would be
unhelpful, but the traveller is told the evidence is thinner than ideal.

### 3. ⚠ Month missing from the data — the ladder widens and discloses it

**Input** — DL, BOS→ATL, **March**, departing 06:00
**Result** — exact match returns **0**; the ladder drops the month and answers
from **1,181 Delta flights** on the route. Match described to the traveller as
*"DL BOS→ATL, any month, departing within 3h of 06:00."*

**Why this is correct** — March 2024 is absent from the source file entirely.
The system does not silently return nothing, and does not pretend the match was
exact. It also keeps the traveller's own airline: the ladder drops month before
carrier, so Delta survives. Dropping carrier first would have answered from
1,576 all-carrier flights with a worse tail (p90 +25 against +21).

### 4. ⚠ No evidence at all — refuses to forecast

**Input** — DL, BOS→**ANC**, November, departing 06:00
**Result** — **no range shown.**

> Only 0 comparable flights after widening the match as far as it goes. That is
> not enough to quote a range, so Murphy will not guess one.

Ladder trace shown to the traveller: `exact = 0 · wider departure window = 0 ·
any month = 0 · any carrier, any month = 0`.

**Why this is correct** — there is no Boston–Anchorage service in the 2024 data.
Every quantile is `None`; nothing is interpolated or borrowed from a similar
route. Refusing is the designed behaviour, not a crash, and the traveller can
see all four rungs were tried.

---

## Trust behaviour

### 5. ⚠ A larger sample with a worse match gets *lower* confidence

**Input** — DL, BOS→**HNL**, November, departing 06:00
**Result** — **196 flights**, which is far more than the 29 behind the confident
BOS→ATL answer. Confidence is nonetheless reported as **limited**:

> Based on 196 flights, but the match is loose: any carrier BOS→HNL, any month,
> departing within 3h of 06:00. There were too few flights matching this one
> closely, so the comparison is broader than ideal.

**Why this is correct** — sample size alone is a bad proxy for trustworthiness.
Reaching 196 rows required dropping both the carrier and the month, so those
flights describe this itinerary poorly. An earlier build reported this case as
high confidence purely because the count was large; that was a real defect,
found by testing the app by hand, and this test now guards against its return.

### 6. Weather is shown as evidence, and withheld when it would be noise

**Input** — DL, BOS→ATL, November, departing 06:00
**Result** — of the 29 comparable flights, **4 had rain or snow at Boston** and
25 were dry. Because both groups clear the minimum of three, the wet/dry
comparison is displayed. Each evidence row also carries precipitation and wind.

**Why this is correct** — the counts reconcile exactly with the sample
(`wet + dry + unknown == n`). Where one side has fewer than three flights — as
on MCI→DEN, which has a single wet day — the comparison is suppressed and the
app says so rather than reporting a statistic drawn from one flight.

Worth noting: on this route the four wet flights landed *earlier* than the dry
ones. The app reports the split and explicitly declines to claim weather caused
anything, which is why that result embarrasses nothing.

---

## Validation: the parse is a proposal, not a fact

### 7. ⚠ Codes that do not exist in the data are rejected

**Input** — a leg with carrier `ZZ` and destination `QQQ`
**Result** — not forecast. Both flagged: *"carrier ZZ does not appear in the
2024 data"*, *"destination QQQ does not appear in the 2024 data."*

**Why this is correct** — the parse is checked against the 15 carriers and 305
airports actually present before any query runs. A plausible-looking but
unusable code is caught at the boundary instead of silently returning zero rows.

### 8. An incomplete leg cannot reach retrieval

**Input** — a leg with `origin = None`
**Result** — `to_query_args` returns `None`; the app reports *"Still needed
before this leg can be forecast: origin."*

**Why this is correct** — the missing field is surfaced for the traveller to
supply. It is structurally impossible for an incomplete itinerary to produce a
forecast.

---

## The parser

### 9. Multi-leg confirmation, including a regional operator

**Input** — the two-leg Delta confirmation in `tests/fixtures/`
**Result** — both legs parsed. Leg 2 is flown by Endeavor Air but ticketed as
Delta: `carrier` stays **DL**, and `operated_by` captures *"Endeavor Air dba
Delta Connection"* verbatim.

**Why this is correct** — the historical data is keyed on the operating carrier
code, and the traveller's confirmation shows the marketing one. Letting the
operator overwrite the carrier would change which flights are retrieved. The
confirmation code is extracted; the passenger name deliberately is not, because
the forecast does not need it.

### 10. ⚠ Missing values become null, never guesses

**Input** — a truncated United confirmation giving only *"UA 328 Denver to
Chicago, Depart 7:45 AM"*
**Result** — `origin`, `dest` and `departure_date` all **null**. The app asks
the traveller for them.

**Why this is correct** — "Denver" and "Chicago" are city names, not IATA codes.
Converting them to DEN and ORD would be an inference the text does not license,
and ORD versus MDW is a real ambiguity. The rule is enforced in code, not only
in the prompt: a null is a correct answer and a guess is not.

### 11. ⚠ Unrelated text produces no itinerary

**Input** — *"Thanks for dinner last night, it was great catching up."*
**Result** — zero legs. *"No flights found in that text. Nothing was parsed, and
no itinerary was invented."*

**Why this is correct** — the model is not coaxed into fabricating a trip from
text that contains none.

### 12. End to end

**Input** — the Delta confirmation, straight through the full slice
**Result** — both legs produce evidence-backed ranges: BOS→ATL **−27 / −12 /
+16** from 29 flights, ATL→MCI **−14 / −5 / +28** from 28 flights. Each carries
real record IDs and the SQL that produced it.

**Why this is correct** — this is the assignment's required loop end to end:
user input → AI parsing → retrieval → grounded output with evidence → a
correction the traveller can make. Quantiles are ordered, the evidence is
non-empty, and the displayed SQL uses a repository-relative path rather than a
developer's home directory.

---

## What testing changed

Two defects were found by running the app and the suite rather than by reading
the code, and both are now guarded:

1. **Confidence ignored match quality.** BOS→HNL reported high confidence on 196
   loosely matched flights. Confidence is now capped once the ladder gives up
   carrier or month. *(Test 5)*
2. **A reduced-confidence answer gave no reason.** A 19-flight exact match
   turned the banner amber while the text said only *"Based on 19 flights
   matching this route…"*, leaving nothing on screen to explain the warning.
   *(Test 2)*
