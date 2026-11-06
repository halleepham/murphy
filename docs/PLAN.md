# Build plan

Rewritten 6 Nov 2026 around the owner's purpose for murphy. It replaces the
plan of 2 Nov, which listed methods but not the question they serve (see git
history). **Everything not marked built is proposed.**

## What murphy is

**Users:** travelers with a deadline or time constraint, such as a tight
connection, a wedding, or a meeting.

**What they do:** enter a flight and their time constraint, then see the risk.

**What makes it different:**

1. **Specific to the user's time.** Other flight tools give a number about the
   flight. Murphy answers "will I make *my* connection or *my* deadline?"
2. **It shows how far to trust the score.** Not one number about delay: the
   range of outcomes, the evidence behind it, and a warning when the system
   has reason to doubt itself.

## The research claim

> A forecasting system that knows when to distrust itself makes better
> decisions than one that is merely accurate.

The claim could fail. If it does, that is the result, and it gets reported.

**Test 1: do the warnings mean anything?** Compare forecasts with and without a
warning against what happened, among flights with similar predicted risk. The
warned ones should be further off. If they are not, stop and rethink.

**Test 2: do warnings lead to better decisions?** Replay real trips with time
constraints. Compare three policies: climatology; the accurate, calibrated
model alone; and the same model acting on the warnings. Count realized outcomes
such as missed connections, missed deadlines and cost, with confidence intervals
resampled by day. Costs are assumptions, so test a range.

**Data use:** 2024–25 to develop, using forecasts from models trained only on
earlier data. **2026 stays sealed** until everything tested on it is committed
to git, then it is opened once. 2026 ends in August, so fall and holiday travel
go untested. 2025 is not pristine, because its results have already been seen.

## Steps

### Core: the claim

1. **Data and `as_of` layer.** *Built* (`0006`, `0007`).
2. **First forecast.** *Built, scored on 2025 only* (`0008`). Sample, climatology,
   78 inputs, a 16-outcome gradient-boosted model. It stays as the baseline to
   beat. Lead times stay at 24, 6 and 2 hours. More than 24 hours out, the app
   says it is too early for a reliable forecast.
3. **Retrain walk-forward.** Train on the past, predict the next six months,
   across 2024–25. First explain why scores worsen with more rounds. Measure one
   fold's run time before committing to all of them.
4. **Trust signal.** Candidates: disagreement between models trained on separate
   flights; recent error at this airport or carrier; thin evidence (no weather,
   no tail, little route history). Run Test 1 here.
5. **The user's time constraint.** In code, not the LLM: chance of making a
   connection or deadline, from the outcome probabilities; alternatives ranked.
   Develop Test 2 here.

### Required methods: all are built, and need not serve the claim

The class requires every one of these, so none is optional here. Each has its own
job and is evaluated the way the course asks. Any of them can also be tried as an
input to steps 3–5, but it does not have to help the claim.

6. **Knowledge graph (Neo4j).** Airports, flights, aircraft, carriers, weather
   and policies, so users can explore why a risk exists. The professor suggested
   this use; it need not be the only one. *Course asks for:* 4–8 node types, 4–10
   relationship types, a multi-hop query, one query showing why a graph helps,
   and a look at 2–3 existing ontologies.
7. **Community detection: risk is not isolated.** Flights are forecast one at a
   time today, though each sees its airport's recent state and its aircraft's
   prior leg. Find groups of airports or flights whose disruptions move together.
   This could also correct a connection's risk, since two legs can share one storm.
   *Course asks for:* Louvain or Leiden, what the communities mean, and an ablation
   with vs without community information.
8. **GraphRAG.** Question → Neo4j retrieval → LLM → grounded answer, with the
   evidence shown. *Course asks for:* three question types (basic, multi-hop,
   analytics), a baseline comparison, and a stated limitation.
9. **Vector retrieval.** Policies and similar past days as evidence. *Course asks
   for:* a comparison against a simpler method, such as BM25.
10. **GNN.** A small message-passing example (GCN, GraphSAGE or GAT), and how it
    could rank nodes before retrieval. *Course says:* a small working example is
    enough.
11. **GenAI.** Itinerary text → validated schema; plain-language explanations; at
    least one presentation beyond a static chart, such as a what-if on the user's
    deadline. Generated content must match the computed findings.

### The application

12. **App.** Not Streamlit (`0003`). Enter a flight and a time constraint; see:
    - the chance of making it, and the range of outcomes;
    - the evidence and where it came from;
    - the trust warning, and how the forecast changes at each lead time;
    - alternatives to compare.

    *Human controls:* change the deadline, inspect evidence, compare alternatives,
    correct or reject a result. Include a shared-risk or graph view if step 6 or 7
    is ready.
13. **Reproducible repository.** README with purpose, architecture, data, results
    and run steps. Deployed, or a local demo with the limit stated.

### Order and the sealed test

Steps 3–5 come first. **The single 2026 test runs last, on whatever is frozen.** Anything
added after the freeze is reported as tested on 2024–25 only, never as tested on
2026.

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

1. **Decision test details.** Which trip decisions to replay, and the cost range.
   Settle with the owner before Test 2 is built.
2. **Records.** Steps 3–5 each need a decision record as they are made.

Deadlines do not drive this plan (`0009`). Quality does.
