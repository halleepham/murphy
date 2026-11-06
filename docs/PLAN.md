# Build plan

Rewritten 6 Nov 2026 around the layers of the finished application (`0010`). It
replaces the step list of the same date (see git history). **Everything not marked
built is proposed.**

## What murphy is

**Users:** travelers with a deadline or time constraint, such as a tight
connection, a wedding, or a meeting.

**What they do:** enter a flight and their time constraint, then see the risk and
whether to keep the flight or consider changing. **The traveler always decides.**

**What makes it different:**

1. **Specific to the user's time.** Other flight tools give a number about the
   flight. Murphy answers "will I make *my* connection or *my* deadline?"
2. **It recommends a change only when it is really needed.** Arriving sooner is
   never a reason. If the current flight still arrives on time, keep it. Changing
   costs money, and murphy says so.
3. **Its reasons can be checked.** It shows the evidence, how that kind of forecast
   has done before, and whether the advice holds up.

**Proof of concept.** No real users. Estimated costs show how murphy weighs cost;
they are not an accurate picture of fares, and no result is reported as real dollars
saved.

**Deliverables:** the application, a research poster, a demo, and the GitHub
repository. No assignment slides or reports. The only real deadline is 9 Nov 2026;
the owner wants the work done as though it were not there, so it does not run short.

## The research claim

Working wording (owner may still change it, `0010` Decision 5):

> A tool that recommends a flight change only when its evidence is solid gives
> better advice than one that simply follows its prediction.

**"Better"** means fewer unneeded changes without more missed deadlines. The claim
could fail. If it does, that is the result, and it gets reported.

**What trust means:** not one score, and not a claim that the forecast is right. It
has three parts:

| Part | Measured? | What the user sees |
|---|---|---|
| Evidence shown | Listed, not scored | What was used and what was missing for this flight |
| Track record | A number, from past results | How often similar forecasts came true |
| Holds up | Pass or fail, from defined checks | Whether the advice survives other models and the cost range |

Airline notices, live aircraft position and similar data are missing for every
flight, so they are stated limitations, not per-flight signals.

**Test 1: do the trust checks mean anything?** Among forecasts with similar predicted
risk, do those with weaker evidence or advice that does not hold up turn out worse?
If not, stop and rethink.

**Test 2: is the advice good?** Replay real trips, assuming the traveler takes the
recommendation. Keeping the flight is the default. Main measures: did we correctly
predict that the current flight would not work out, and did the recommended
alternative make it? Compare with no change at all (what happens without murphy), a
recommender using climatology, and the same model without the trust checks. Also
count missed deadlines and unneeded changes, with confidence intervals resampled by
day. Costs are estimates, so test a range.

**Not tested:** whether seeing the evidence changes what people do. That needs real
users. It is stated as a limitation.

**Data use:** 2024–25 to develop, using forecasts from models trained only on
earlier data. **2026 stays sealed** until everything tested on it is committed
to git, then it is opened once. 2026 ends in August, so fall and holiday travel
go untested. 2025 is not pristine, because its results have already been seen.

## How it is built

**One real model, with simple code around it.** Training is the only step that is
slow to redo, so it is done once at full quality, after the inputs are chosen. Other
layers start simple and grow. A late input means retraining and rerunning the scoring
scripts, so it costs machine time, not a restart.

## The layers, in build order

### 1. Data and `as_of` layer. *Built* (`0006`, `0007`).

### 2. Single-flight forecast. *Built, scored on 2025 only* (`0008`, `0010`).

78 inputs, 16 outcomes, gradient boosting with `l2_regularization` 10. Multiclass
Brier 0.5553 on 2025 against 0.6026 for by-origin climatology. Calibration on this
model is not yet computed. Lead times are 24, 6 and 2 hours. More than 24 hours out,
the app says it is too early for a reliable forecast.

### 3. Choose the inputs. *Next.*

Build candidates and screen each on a quarter-sized fit, on 2024–25, against the
version without it. Keep an input only if it helps:

- **Communities: risk is not isolated.** Louvain or Leiden over airports or flights
  whose disruptions move together. *Course asks for:* what the communities mean, and
  an ablation with vs without.
- **Similar past flights.** Retrieval finds them; code computes what happened to
  them. Only flights known before the forecast time count.

An input that does not help the score can still serve as evidence for layer 7.

### 4. Retrain walk-forward. Once.

Train on the past, predict the next six months, across 2024–25. Each fold picks its
round from the last months of its own training data. Measure one fold first.

### 5. Recommendation.

In code, not the LLM: chance of making the deadline, then keep or consider changing.
Change only if the current flight's chance of missing is high and an alternative
lowers it by a large margin. Cutoffs are tested on 2024–25 and the owner picks from
the results. List several alternatives and mark the best. Forecasts cover about 9.4%
of flights, so inputs must be built for alternatives on the replay routes and days;
measure that first. Single-flight deadlines first, then connections, where two legs
can share one storm.

**Cost:** one quarter of DB1B route fares plus a tested range for the last-minute
premium, as fare difference or new ticket. Needs its own decision record. Fallback: a
labeled fare by distance. The user can replace the estimate. A past schedule shows
that a flight existed, not that it had a seat.

### 6. Knowledge graph (Neo4j).

Airports, flights, aircraft, carriers, weather and policies, so users can explore why
a risk exists. Finds alternatives if that proves better than a query. *Course asks
for:* 4–8 node types, 4–10 relationship types, a multi-hop query, one query showing
why a graph helps, and a look at 2–3 existing ontologies.

### 7. Trust.

The three parts above. Run Test 1 and Test 2. Add "do sources agree" (model, similar
flights, the airport's current state) if layers 3 and 6 are ready.

### 8. Explanation.

- **GraphRAG.** Question → Neo4j retrieval → LLM → grounded answer, with the evidence
  shown. *Course asks for:* three question types (basic, multi-hop, analytics), a
  baseline comparison, and a stated limitation.
- **Vector retrieval.** Policies and similar past days as evidence. *Course asks for:*
  a comparison against a simpler method, such as BM25.
- **GNN.** A small message-passing example (GCN, GraphSAGE or GAT), and how it could
  rank nodes before retrieval. *Course says:* a small working example is enough.
- **GenAI.** Itinerary text → validated schema; plain-language explanations; at least
  one presentation beyond a static chart, such as a what-if on the user's deadline.
  Generated content must match the computed findings.

### 9. Application.

Not Streamlit (`0003`). Enter a flight and a time constraint; see:

- the chance of making it, and the range of outcomes;
- keep or consider changing, or too uncertain to say;
- alternatives, with the best marked and the estimated cost of changing;
- the evidence and where it came from, the track record, and whether the advice holds up;
- how the forecast changes at each lead time.

*Human controls:* change the deadline, replace the cost, inspect evidence, compare
alternatives, correct or reject a result. Include a shared-risk or graph view if
ready. A first simple version comes early and grows with the layers.

### 10. Freeze and the sealed test.

Commit everything, then score 2026 once. Anything added after the freeze is reported
as tested on 2024–25 only, never as tested on 2026.

### 11. Deliverables.

Research poster; demo; README with purpose, architecture, data, results and run steps.
Deployed, or a local demo with the limit stated.

## Rules at every step

- **Output shape:** Answer + Evidence + Provenance + Uncertainty.
- **Roles:** big data systems process, models calculate, RAG retrieves, LLMs
  explain. No number comes from an LLM.
- **Tests:** 5–10 cases, a deliberate failure or thin-evidence case, and a
  comparison with something simpler.
- **Decisions:** each records human design → AI recommendation → accepted,
  modified or rejected → why.
- **Honesty:** real results only, and built is kept separate from proposed.
- **Justified by evidence:** say what improved, what did not, and why. The course
  warns against adding AI technologies for their own sake.

## Open

1. **Claim wording.** The owner may change it.
2. **Recommendation details.** The cutoffs, and the cost range.
