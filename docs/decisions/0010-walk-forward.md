# 0010 — Walk-forward retraining (Step 3)

Date: 6 Nov 2026. Follows `0008` Decision 8, which left "scores that rise with
rounds" unexplained, and `docs/PLAN.md` Step 3.

## Finding 1 — Why 2025 scores worsen with more rounds

**Status:** measured from saved models, 6 Nov 2026. No new fit. 2026 rows were
never loaded. The cause in the last section below is inferred, not yet tested.

Samples, by fixed hash of `flight_id`: 245,895 rows the quarter models trained
on; 245,145 rows from 2023–24 flights they did not train on; 258,816 rows from
2025. These are samples, so scores sit about 0.003 below the full-2025 figures
in `0008`. The best round for `final` is 75 on both.

**Short answer:** a small share of forecasts become wildly overconfident in
rare outcomes. Most forecasts do not get worse.

**It is overfitting that shows in unseen 2023–24 flights too.** Multiclass Brier:

| Model | Set | Round 25 | Round 75 | Last round |
|---|---|---|---|---|
| lr 0.1, 63 leaves, quarter | trained on | 0.5304 | 0.5007 | 0.4234 |
| | unseen 2023–24 | 0.5621 | 0.5735 | 0.6072 |
| | 2025 | 0.5820 | 0.6326 | 0.6829 |
| lr 0.05, 63 leaves, quarter | trained on | 0.5387 | 0.5158 | 0.4553 |
| | unseen 2023–24 | 0.5513 | 0.5441 | 0.5409 |
| | 2025 | 0.5657 | 0.5605 | 0.5588 |
| `final`, lr 0.05, 63 leaves, all rows | trained on | 0.5443 | 0.5322 | 0.5132 |
| | 2025 | 0.5634 | 0.5579 | 0.5653 |

Last round is 300, or 275 for `final`. So a shift between years is not the main
cause of the rise, since unseen 2023–24 flights worsen too at lr 0.1. A shift
does exist separately: 2025 is about 0.018 worse than unseen 2023–24 for the
stable model, and worsens faster at lr 0.1.

**The damage is concentrated.** For `final` on 2025, round 75 to 275:

- Brier rises 0.0074. The 1% of rows that worsened most add 0.0101; the other
  99% improve by 0.0027. That split selects the worst rows after the fact, so
  it overstates the contrast somewhat.
- The rise by class sums to 0.0072: `diverted` 0.0033, `150-165` 0.0011, `early`
  0.0010, `165-180` 0.0007, all others 0.0011.

**Forecasts above 90% on one outcome, `final`, 2025 sample:**

| Predicted outcome | Round 75: n | Right | Round 275: n | Right |
|---|---|---|---|---|
| diverted | 83 | 3.6% | 934 | 0.6% |
| 150-165 | 27 | 0% | 184 | 0% |
| 165-180 | 32 | 9.4% | 169 | 1.8% |
| cancelled | 237 | 100% | 376 | 96.5% |
| early | 9 | 88.9% | 671 | 92.0% |

Cancelled and early remain sound. **The saved model, used at round 75, already
has this fault on a small scale**: 83 forecasts say over 90% diverted, and 3 of
them were right.

**Extreme tree leaves appear with it.** Largest leaf value, in log-odds:

| Model | Largest leaf | Trees with a leaf over 20 | First round |
|---|---|---|---|
| lr 0.05, 63 leaves, quarter | 10.9 | 0 | none |
| `final`, same settings, all rows | 2,543 | 398 of 4,400 | 51 |
| lr 0.1, 63 leaves, quarter | 64,673 | not counted by tree (45,018 leaves) | 1 |

In `final`, these are mostly in `diverted` (183), `165-180` (128) and `150-165`
(68). A value of 20 already means near-certainty.

**Likely cause: inferred, not tested.** The library sets a leaf to gradient ÷
hessian. For a rare outcome already given a tiny probability, the hessian is near
zero, so one surprising row produces an enormous value. `l2_regularization`, which
adds to that divisor, is currently 0. Hessians were not measured.

**Not explained:** why `final` blows up and the quarter fit with the same
settings does not.

**Why it matters for later steps:**

1. A best round does not carry over: identical settings gave 275 on a quarter
   and 75 on all rows. Walk-forward folds also cannot pick a round using the
   period they are predicting.
2. Test 1 could pass for a fixable reason. Models that disagree would flag these
   blow-ups, so a trust signal would look useful for catching a bug.

## Decision 1 — What to do about it

**Status:** decided 6 Nov 2026. Stress test and refit measured.

**Options:**
(a) add `l2_regularization`, which targets thin leaves and needs no dependency;
(b) keep the model and stop early, which needs no refit but keeps the fault and
could distort Test 1; (c) merge the rare delay ranges, which changes the 16-outcome
design, loses detail for deadline questions, and leaves `diverted` rare.

**AI recommended (a). Owner: accepted, 6 Nov 2026.** Steps:

1. Try `l2_regularization` 1 and 10 on the worst setting (lr 0.1, 63 leaves, the
   quarter), since the blow-ups are largest there. `gbm.py l2`.
2. If the blow-ups stop, refit on all rows at lr 0.05.
3. Each walk-forward fold picks its round from the last months of its own
   training data, never from the period it predicts.

This uses 2025 again, which is already not pristine. 2026 stays sealed.

**Code:** `scripts/rounds_diag.py` and `scripts/leaf_check.py` reproduce Finding 1.
Results: `data/forecast/gbm/diag/` (gitignored). `gbm.py` now records each new fit's
`largest_leaf`.

**Stress test results, 6 Nov 2026.** lr 0.1, 63 leaves, the quarter (982,293 rows),
all inputs, scored on all of 2025. Multiclass Brier; lower is better.

| `l2_regularization` | Largest leaf | Best | At round | At round 300 | Fit seconds |
|---|---|---|---|---|---|
| 0 | 64,673 | 0.5848 | 25 | 0.6866 | 492 |
| 1 | 7.0 | 0.5591 | 75 | 0.5617 | 515 |
| 10 | 4.6 | 0.5582 | 75 | 0.5603 | 755 |

The largest leaf at 0 comes from `rounds_diag.py` on the same saved model. Others come
from `largest_leaf` in `gbm.py`.

Read: the blow-ups stop, which supports the cause in Finding 1, though hessians
were still not measured. The score is also better than the earlier all-rows `final`
(0.5607), on a quarter of the data. A slow rise remains after round 75 (0.0021 at
l2 10), which looks like ordinary overfitting. Only 1 and 10 were tried, 10 is the
edge of what was tried, and the choice was made on 2025, which flatters these scores.
No confidence intervals are computed.

**Refit:** `gbm.py refit`, lr 0.05, 63 leaves, l2 10, all 3,931,131 rows, 300 rounds.
The earlier `final` stays saved for comparison.

| Model | Largest leaf | Round 75 | Round 150 | Last round | Fit seconds |
|---|---|---|---|---|---|
| `final`, l2 0 | 2,543 | 0.5607 | 0.5644 | 0.5681 (275) | 1,931 |
| `final_l2`, l2 10 | 3.3 | 0.5576 | 0.5560 | **0.5553** (300) | 2,567 |

Multiclass Brier on all 2,070,453 forecasts in 2025. By-origin climatology: 0.6026.

Read: the score now improves at every checkpoint through round 300, by 0.0001 or less
per 25 rounds from round 200 on, so more rounds add little. More data now helps too:
the earlier refit was "barely better" than the quarter fit, perhaps because of the
fault. The largest leaf at l2 0 comes from `rounds_diag.py`. Late and cancelled
Brier, calibration and confidence intervals are not yet computed for `final_l2`, so
`0008`'s calibration results do not carry over. Settings were chosen on 2025, which
flatters these scores. 2026 stays sealed.

## Decision 2 — What the recommendation is

**Status:** the owner's requirements, stated 6 Nov 2026. The details listed as
open are not decided. None of this is built.

**Owner's requirements, summarized:**

1. The human decides. Murphy recommends. Test 2 assumes the traveler took the
   recommendation and asks how that turned out.
2. Murphy may serve only the extreme cases, where a traveler really needs to change
   flights to make a deadline. It recommends a change only when that is really the
   best choice and the risk of not making it is high.
3. Arriving sooner is never a reason to change. If the current flight still arrives
   on time, stay on it, even if alternatives arrive earlier.
4. The cost of changing matters a lot to some people, so murphy is transparent about it.
5. The end product is an application. Backend work must support it, and the owner
   needs to present the methods in the demo and poster.
6. The recommendation should be planned to evolve as the required methods are added.

**This replaces the AI's proposal** to replay "choose any flight" with "keep unless":
every trip starts on a flight, as in the owner's example.

**Proposed by AI, open:**

- Change only if both hold: the current flight's chance of missing is high, and an
  alternative lowers it by a large margin. Thresholds to be tested on 2024–25 and
  frozen before 2026.
- Three answers: keep; consider changing; too uncertain to say. A warning never
  causes a change by itself.
- Superseded by the owner's additions below: "never change" as a competing baseline,
  and the user entering a change cost as the only source.
- Replay limit: a past schedule shows that a flight existed, not that it had a seat.

## Decision 3 — How the methods can change the recommendation

**Status:** AI proposal, 6 Nov 2026, not yet reviewed by the owner. None is built.

Numbers still come from code. A method that changes a number must beat the version
without it on 2024–25, and be frozen, before it counts as tested on 2026.

| Method | Changes the recommendation? | Role |
|---|---|---|
| Community detection | Possibly | An alternative hit by the same disruption as the current flight is not a real escape |
| Knowledge graph | Possibly | Finds alternatives; traces why risk exists, such as the inbound aircraft |
| GNN | Possibly; small example required | Ranks nodes before retrieval |
| Vector retrieval | Possibly | Similar past days as evidence; a possible trust signal input |
| GraphRAG | No | Grounded answers to "why this recommendation?" |
| GenAI | No | Itinerary text to schema; plain-language explanations that match computed values |

## Decision 4 — Owner's additions, 6 Nov 2026

**Decided by the owner:**

- The change rule is accepted: the current flight's chance of missing is high, and an
  alternative lowers it by a large margin. Cutoffs are tested on 2024–25 and the owner
  picks from the results.
- Test 2 asks: did we correctly predict that the current flight would not work out,
  and was the recommended alternative good? Those are the main measures. "No tool"
  is only a reference point, not what murphy competes with.
- Murphy can list several alternatives while marking what it thinks the user should
  do and the best one.
- Murphy is a proof of concept. Estimated costs are acceptable, based on data where
  possible and stated as a limitation.
- The final deliverables are the application, a research poster, a demo, and the
  GitHub repository. The earlier "skip deliverables" meant assignment slides and
  reports only. The only real deadline is 9 Nov 2026, but the owner wants to work as
  though it were not there, so the project does not run short.

**Open, raised by the owner:** what "trust" means. It is not defined or built. Only
candidate signals exist in `docs/PLAN.md`. The owner also asked for a whole-project
overview to prevent backtracking.

**Proposed by AI, open:** estimate change cost from DB1B route fares and a range for
the last-minute premium; define trust through evidence, track record and whether the
recommendation holds up.


## Decision 5 — Trust, cost and the overview

**Owner, 6 Nov 2026:**

- **Trust: accepted** as evidence shown, track record, and whether the advice holds up.
  The claim's wording is to be simplified.
- **Cost: accepted**, DB1B fares plus a tested range for the last-minute premium. The
  point is not an accurate picture of real fares, because that data is not available.
  It is to show how murphy takes cost into account and helps the user decide. So no
  result may be reported as real dollars saved.
- **Overview: accepted**, with two questions raised: whether retrieval should feed
  the prediction, not only explain it; and whether a simple version wastes training
  time compared with building the final version directly.

**AI recommended, open:**

- Retrieval can feed the prediction if code computes the number: find similar past
  flights, take what happened to them, and test that as a model input or as a separate
  source of evidence. The LLM still computes nothing. Screen it with community inputs
  before walk-forward, and keep each only if it beats the version without it.
- Train once, at full quality, after the inputs are chosen. Only the code around the
  model starts simple. Quarter-sized screening fits are for choosing inputs, not
  versions to build on.
- Test only the rule in the claim. The replay assumes the traveler takes the
  recommendation, so it cannot measure whether seeing the evidence changes behavior.

**Owner, 6 Nov 2026, on the above:** accepted one real model with simple code around
it, and screening community and similar-flight inputs before retraining. The owner
asked what trust means; `docs/PLAN.md` now defines it. Working claim wording is in
`docs/PLAN.md`; the owner may still change it. This supersedes the claim and the
warning-based Test 1 in `0009`.

