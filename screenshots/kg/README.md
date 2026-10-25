# Challenge 3 knowledge-graph captures

Twelve screenshots from Neo4j, taken against the graph built by
`src/murphy/kg_load.py`. The query behind each one, and what it is meant to show,
is in the screenshot guide under `docs/challenge-3/`.

| File | Shows |
|---|---|
| `01_schema.png` | `db.schema.visualization()` — 9 node types, 10 relationship types |
| `02_node_counts.png` · `03_relationship_counts.png` | 37,059 nodes · 117,489 relationships |
| `04_itinerary_graph.png` | a sample traveller, their trip, and the hub both legs share |
| `05_match_itinerary.png` | the legs in order, with the traveller's own deadline |
| `06_aircraft_day_graph.png` | one aircraft, five flights, the rotation chain |
| `07_cascade_table.png` | the cascade, with BTS cause codes confirming it |
| `08_cascade_multihop.png` | the variable-length `[:PRECEDES*1..4]` traversal |
| `09_correlated_table.png` | nine Atlanta→Newark options, eight already compromised |
| `10_correlated_graph.png` | the same thing as a picture |
| `11_policy_graph.png` | an airline and the commitments it publishes |
| `12_policy_filtered.png` | filter keys discriminating by delay length |
