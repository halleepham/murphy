# 0002 — Problem domain: itinerary-specific flight disruption risk

**Date:** 2 Nov 2026
**Status:** accepted (concept). Implementation choices are open; see "Open".

## Decision 1 — The concept carries over from the original idea

The owner's concept, in their words where possible. This is the general idea
behind murphy and is kept. The *implementation* is rebuilt from scratch, and
murphy-v1's methods are still not a reference.

> An application that predicts a traveler's flight and whether there is any
> risk or delay associated with it. If the predicted risk is high, it
> recommends alternatives.

**How it differs from existing flight-delay forecasts:** it is specific to the
user's itinerary and situation, not just the flight.

- A hard arrival deadline ("I have to land by 5 PM").
- Connections: the first leg landed, but the connection is tight. What is the
  best move now?
- Alternatives are weighed on cost and airline policy, not only on time.
- Inputs combine the user's situation with the current environment: weather,
  cost, schedules.

**The name, and the core idea.**

- *Murphy's law*: if something can go wrong, it will. So the app looks for risk.
- *Allan H. Murphy*: a meteorologist who pioneered probability forecasting and
  asked "what is a good forecast?" So a core feature is **how much the user can
  trust the forecast**: a probability, the evidence behind it, and the
  confidence in it.

**The output, stated by the owner:** look at the risks, make a prediction,
back it with evidence and a confidence level, and recommend a next step that
accounts for the user's factors and the current conditions.

This lines up with the course's required output shape,
**Answer + Evidence + Provenance + Uncertainty**, without forcing it.

## Decision 2 — How the coursework is handled

The owner's call, 2 Nov 2026:

- All challenge deadlines have passed. They are not a schedule.
- Deliverables such as slides and reports are skipped. Only the **end
  product** is built.
- That product must still use every method the challenges require: vector
  retrieval, structured generation, a knowledge graph, GraphRAG, a GNN or
  other graph model, community detection, multimodal GenAI output,
  evaluation against a baseline, and an ablation study.
- Work proceeds in challenge order: 2 → 3 → 4.
- "Human Design" stages are **planned together** with the AI, not done by the
  human alone. This replaces the "minimal AI involvement" rule that `AGENTS.md`
  previously stated.

## Open — decided in later records

- What exactly is predicted (the target variable) and how it is evaluated.
- Data sources, licenses, and live data vs historical replay.
- Prediction model, LLM provider, embeddings, and where Neo4j runs.
- Whether to upgrade Python from 3.10 (see the wheel findings in the
  2 Nov planning session).
