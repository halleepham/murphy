# Build plan

Proposed by the AI in the 2 Nov 2026 planning session, after decisions
`0002`–`0005`. It was only in chat until 3 Nov, when it was written here
because a later session could not find it. The steps below are as proposed
then. Status is updated as work lands.

Two days is tight on Claude Pro limits, so the order puts a working
end-to-end slice first, then adds the graph methods.

## Day 1 — Challenge 2 core

1. **Data slice and `as_of` data layer.** The chosen carriers at about 10–15
   hubs, plus IEM weather, DB1B fares and DOT data. Stored as Parquet behind
   an `as_of` data layer, with tests showing no future data leaks in.
   *Status (checked 3 Nov 2026): **done.***
   - *Done (`bd5bc39`, `0006`): BTS flights and the `as_of` layer.
     **Scope changed:** every airport, not 10–15 hubs (owner accepted in
     `0006`), because aircraft rotations leave the hubs.*
   - *Done (`0007`): IEM weather, ASOS observations and TAF forecasts as
     issued, for the 30 busiest airports (64.6% of departures), behind the
     same `as_of` rules. **Scope changed:** 30 airports, chosen by the owner.*
   - *Moved (`0007`, owner accepted):* DB1B fares to Step 3; DOT policy and
     consumer data to Step 4, where each is first used.
2. **Forecasts.** Climatology, then a gradient-boosted model giving per-leg
   delay distributions and cancellation probability. Scored by Brier score
   and calibration on 2026.
3. **Itinerary risk and alternatives.** Missed connection, missing the
   deadline, and ranking of alternatives, all in code.
4. **Parsing and retrieval.** Itinerary parsing (text → Groq → validated
   schema). Retrieval over policy and analog days, plus weather forecast
   text if the IEM archive has it (`0004` Decision 4), compared against BM25.

## Day 2 — Challenges 3 and 4, then the UI

5. **Neo4j graph and GraphRAG** on Aura, comparing hand-written Cypher
   templates against LLM-written Cypher.
6. **Community detection,** plus the ablation with and without community
   features.
7. **A small STGNN,** compared against the simpler models.
8. **UI** with in-browser OCR (Tesseract.js) and charts. The frontend
   framework is not decided yet (`0003` Decision 6: not Streamlit).
9. **Deploy.**

Steps 7 and 9 are the first to shrink if time runs short. The assignment
says a small working GNN example is enough.
