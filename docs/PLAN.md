# Build plan

Rewritten 6 Nov 2026 around the finished application and the owner's objective
(`0010`). It replaces earlier versions (see git history). **Everything not marked
built is proposed.**

## Objective

> **Help travelers with a deadline decide whether to keep or change their flight, and
> show the evidence behind each recommendation so they understand why.**

In detail: given an itinerary and a deadline, predict the probability of arriving on
time using only information available at the time of the forecast. Recommend changing
flights only when the expected benefit outweighs the cost, and support each
recommendation with traceable evidence. Measure prediction and recommendation quality
against simpler baselines.

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

## What trust means

Not one score, and not a claim that the forecast is right. The user can check the
reasons:

| Part | Measured? | What the user sees |
|---|---|---|
| Evidence shown | Listed, not scored | What was used and what was missing for this flight |
| Track record | A number, from past results | How often similar forecasts came true |
| Holds up | Pass or fail, from defined checks | Whether the advice survives the cost range and cutoffs |

Airline notices, live aircraft position and similar data are missing for every
flight, so they are stated limitations, not per-flight signals.

## How we show it works

| Question | Compared against | Status |
|---|---|---|
| Is the forecast good? | Climatology | Measured on 2025 (`0010` Finding 2) |
| Which inputs help? | Schedule; then history; current state and aircraft; weather | Measured on 2025 (`0010` Finding 2) |
| Does network information help? | With and without communities; with and without GNN | Not built |
| Are recommendations good? | No change at all; a simple climatology recommender | Not built |
| Do trust checks and cost help? | The recommender with and without each | Not built |
| Is retrieval good? | Vector vs BM25 | Not built |
| Are answers grounded? | GraphRAG vs plain retrieval; every value traces to a source | Not built |

**Are recommendations good?** Replay real trips, assuming the traveler takes the
recommendation. Keeping the flight is the default. Did we correctly predict that the
current flight would not work out, and did the best alternative make it? Also count
missed deadlines and unneeded changes, with confidence intervals resampled by day.
Costs are estimates, so test a range.

A method that does not help is still reported, with its limits and why it may have
failed. A claim that one helps needs evidence.

**Not tested:** whether seeing the evidence changes what people do. That needs real
users. It is stated as a limitation.

**Data use:** train on 2023–24, develop on 2025 (`0010` Decision 6). **2026 stays
sealed** until everything tested on it is committed to git, then it is opened once.
2026 ends in August, so fall and holiday travel go untested. 2025 is not pristine,
because its results have already been seen and settings were chosen on it.

## How it is built

**One real model, with simple code around it.** Training is the only step that is
slow to redo. A first complete path through every layer comes early, then each grows.

## The layers, in build order

### 1. Data and `as_of` layer. *Built* (`0006`, `0007`).

### 2. Single-flight forecast. *Built, scored on 2025 only* (`0008`, `0010`).

78 inputs, 16 outcomes, gradient boosting with `l2_regularization` 10. Multiclass
Brier 0.5553 on 2025 against 0.6026 for by-origin climatology. Lead times are 24, 6
and 2 hours. More than 24 hours out, the app says it is too early for a reliable
forecast. Calibrated within 0.034 in every bin for "late", running low (`0010` Finding 2).

### 3. Recommendation and a first app.

In code, not the LLM: chance of making the deadline, then keep, consider changing, or
too uncertain to say. Change only if the current flight's chance of missing is high
and an alternative lowers it by a large margin. Cutoffs are tested on 2025 and the
owner picks from the results. List several alternatives and mark the best. Forecasts
cover about 9.4% of flights, so inputs must be built for alternatives on the replay
routes and days; measure that first. Single-flight deadlines first, then connections,
where two legs can share one storm.

**Cost:** one quarter of DB1B route fares plus a tested range for the last-minute
premium, as fare difference or new ticket. It shows how murphy weighs cost; it is not
an accurate picture of fares, and no result is reported as real dollars saved. Needs
its own decision record. Fallback: a labeled fare by distance. The user can replace
the estimate. A past schedule shows that a flight existed, not that it had a seat.

**App:** not Streamlit (`0003`). One screen first; layer 7 finishes it.

### 4. Network inputs, tested with and without.

- **Communities.** Louvain or Leiden over airports or flights whose disruptions move
  together, using `networkx`. Also whether an alternative shares the current flight's
  disruption. *Course asks for:* what the communities mean, and an ablation.
- **GNN** (`0010` Decision 6). A model over airports and flights that predicts each
  airport's disruption, fed to the forecast as an input. Start simple; try an STGNN
  only if the simple one shows promise. *Course says:* a small working example is
  enough, including how it could rank nodes before retrieval.

Only information known at the forecast time counts. If either helps, retrain once and
rerun the scoring.

### 5. Knowledge graph and evidence.

- **Neo4j.** Airports, flights, aircraft, carriers, weather and policies, so users can
  explore why a risk exists. *Course asks for:* 4–8 node types, 4–10 relationship
  types, a multi-hop query, one query showing why a graph helps, and a look at 2–3
  existing ontologies.
- **GraphRAG.** Question → Neo4j retrieval → LLM → grounded answer, with the evidence
  shown. *Course asks for:* three question types (basic, multi-hop, analytics), a
  baseline comparison, and a stated limitation.
- **Vector retrieval.** Policies and similar past flights as evidence; code computes
  what happened to them. Test as a model input if time allows. *Course asks for:* a
  comparison against a simpler method, such as BM25.
- **Trust.** The three parts above.

### 6. GenAI.

Itinerary text → validated schema; plain-language explanations; at least one
presentation beyond a static chart, such as a what-if on the user's deadline.
Generated content must match the computed findings.

### 7. Finish the application.

Enter a flight and a time constraint; see:

- the chance of making it, and the range of outcomes;
- keep or consider changing, or too uncertain to say;
- alternatives, with the best marked and the estimated cost of changing;
- the evidence and where it came from, the track record, and whether the advice holds up;
- how the forecast changes at each lead time.

*Human controls:* change the deadline, replace the cost, inspect evidence, compare
alternatives, correct or reject a result. Include a shared-risk or graph view if ready.

### 8. Freeze and the sealed test.

Commit everything, then score 2026 once. Anything added after the freeze is reported
as tested on 2025 only, never as tested on 2026.

### 9. Deliverables.

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

1. **Recommendation details.** The cutoffs, and the cost range.
