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
   *Status: flight slice and `as_of` layer committed in `bd5bc39`; see
   `0006`. Check `0006` for which sources are actually in.*
2. **Forecasts.** Climatology, then a gradient-boosted model giving per-leg
   delay distributions and cancellation probability. Scored by Brier score
   and calibration on 2026.
3. **Itinerary risk and alternatives.** Missed connection, missing the
   deadline, and ranking of alternatives, all in code.
4. **Parsing and retrieval.** Itinerary parsing (text → Groq → validated
   schema). Retrieval over policy and analog days, compared against BM25.

## Day 2 — Challenges 3 and 4, then the UI

5. **Neo4j graph and GraphRAG** on Aura, comparing hand-written Cypher
   templates against LLM-written Cypher.
6. **Community detection,** plus the ablation with and without community
   features.
7. **A small STGNN,** compared against the simpler models.
8. **UI** with in-browser OCR (Tesseract.js) and charts.
9. **Deploy.**

Steps 7 and 9 are the first to shrink if time runs short. The assignment
says a small working GNN example is enough.
