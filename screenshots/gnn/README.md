# Challenge 3 GNN captures

Ten screenshots: five from Neo4j Browser against the graph after
`src/murphy/kg_congestion.py`, five from the executed `notebooks/04_gnn.ipynb`.

The Cypher and the cell for each is in `docs/challenge-3/screenshot_guide_gnn.md`.

The two result figures are **not** screenshots — they are generated files and should be
placed in the deck directly from `docs/challenge-3/figures/gnn_seed_spread.png` and
`gnn_feature_ablation.png`.

| File | Shows | Slide |
|---|---|---|
| `01_schema_after.png` | the schema with `:AirportHour` and the two new edge types | 6 |
| `02_counts_after.png` | 37,059 nodes · 147,881 relationships, 37% of the Aura cap | 6 |
| `03_flight_neighborhood.png` | one flight, its inbound leg, its two airport-hours — exactly what one layer aggregates | 4 |
| `04_busiest_hours.png` | congestion per airport-hour, ATL 07:00 at 139 movements | 2 |
| `05_cited_chain.png` | the four-leg rotation chain behind the top-ranked flight | 7 |
| `06_guards.png` | the feature build and its five leakage guards passing | 5 |
| `07_message_passing.png` | message → aggregate → update on a real flight | 4 |
| `08_ablation_table.png` | the five rungs, both query slacks, five seeds each | 5, 8 |
| `09_ranking_demo.png` | GNN ranking vs the shipped rule, with outcomes | 7 |
| `10_cited_evidence.png` | the rotation chain the answer rests on | 7 |
