# Murphy — Challenge 3: Knowledge Graph

**Which flights fail for the same reason?**

Murphy forecasts flight disruption risk and states how much to trust its own forecast.
Challenge 2 answered *"will I land on time?"* from a table of comparable flights.
Challenge 3 builds the thing that table could never express: **how flights are
connected to each other**, and therefore which ones break together.

Course project for CS 5542 (Big Data Analytics and Applications), Fall 2026.

> **This branch is the Challenge 3 AI Design slice.** The Challenge 2 application —
> parser, retrieval, forecast model, route planner and Streamlit interface — is tagged
> [`challenge-2`](../../tree/challenge-2) and deployed at
> [murphy-challenge-2.streamlit.app](https://murphy-challenge-2.streamlit.app/).

---

## Why a graph

The Challenge 2 report named a limitation it could not fix: *"flight legs are scored
independently, so correlated delay at a shared hub is currently invisible."*

A traveller stuck at Atlanta on 19 July 2024 trying to reach Newark had **nine
alternatives**. A forecaster scoring each leg on its own shows nine options, each
carrying some moderate risk. In fact **every one of them was flown by an aircraft
already running 45+ minutes late**, and they arrived an average of **267 minutes**
behind schedule. There was no escape, and nothing in a row-oriented store could say so.

Answering that question means walking backwards from every candidate flight to the
aircraft that will operate it, and from that aircraft to where it currently is. That is
a self-join nobody writes in SQL. It is two hops in a graph.

---

## Contents

- [The schema](#the-schema)
- [Running it](#running-it)
- [The queries](#the-queries)
- [Data and scope](#data-and-scope)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Testing](#testing)
- [Limitations](#limitations)

---

## The schema

**9 node types · 10 relationship types · 37,059 nodes · 117,489 relationships**

| Node | Tier | What it is |
|---|---|---|
| `Flight` | Observed | one operated flight on the loaded day |
| `Service` | Derived | route + carrier + hour, aggregated over the **whole year**, carrying p10/p50/p90 |
| `Aircraft` | Observed | a tail number |
| `Airport` | Observed | built from origin ∪ destination |
| `Airline` | Observed | the nine US majors |
| `Weather` | Observed | one airport-hour observation |
| `PolicyClause` | Documented | a published commitment, with its source URL |
| `Traveler` · `Itinerary` | Session | sample travellers; never real people |

```
(Traveler)-[:BOOKED]->(Itinerary)-[:CONTAINS]->(Flight)
(Aircraft)-[:FLIES]->(Flight)-[:PRECEDES]->(Flight)
(Flight)-[:INSTANCE_OF]->(Service)-[:OPERATED_BY]->(Airline)-[:PUBLISHES]->(PolicyClause)
(Flight|Service)-[:DEPARTS_FROM|:ARRIVES_AT]->(Airport)<-[:OBSERVED_AT]-(Weather)
```

**Provenance tiers are part of the schema, not documentation.** Observed data drives
the forecast; Derived values are computed and labelled as such; Documented clauses are
quoted with a citation and **never numerically combined**; Session data shapes the
question but is never evidence. Folding a policy or a survey score into a risk number
is the mistake the tiers exist to prevent.

**`CONNECTS_TO` is not stored.** Materialising every feasible connection cost 493,824
edges on a single day, against a 400,000 relationship ceiling — and whether a
connection holds is a *question about time*, not a fact. It is computed at query time:

```cypher
MATCH (a:Flight {flight_id: $flight_id})-[:ARRIVES_AT]->(hub:Airport)
      <-[:DEPARTS_FROM]-(b:Flight)
WHERE b.sched_dep_utc >= a.sched_arr_utc + duration({minutes: 45})
  AND b.sched_dep_utc <= a.sched_arr_utc + duration({hours: 4})
RETURN b.flight_id,
       duration.between(a.sched_arr_utc, b.sched_dep_utc).minutes AS slack_min,
       EXISTS { (a)-[:PRECEDES]->(b) } AS same_aircraft
```

---

## Running it

A free [Neo4j Aura](https://console.neo4j.io) instance is enough — the graph uses 19%
of its node cap and 31% of its relationship cap.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

cp .env.example .env     # then paste in the four NEO4J_ values from Aura
                         # note: Aura names the user and database after the
                         # instance id, not "neo4j"

PYTHONPATH=src .venv/bin/python -m murphy.kg            # connection smoke test
PYTHONPATH=src .venv/bin/python -m murphy.kg_build      # extraction counts, no writes
PYTHONPATH=src .venv/bin/python -m murphy.kg_load       # build the graph (~20 s)
PYTHONPATH=src .venv/bin/python -m murphy.kg_queries    # run every demo query
```

The load is **idempotent** — every write is a `MERGE` on a key backed by a uniqueness
constraint, so running it twice leaves the same graph rather than a doubled one.
`--reset` wipes first; `--dry-run` reports counts without writing.

The 2024 flight table is committed, so a clone needs no download.

---

## The queries

`src/murphy/kg_queries.py` — all parameterised, none generated by a model.

| | Shows |
|---|---|
| **Q1 / Q2** | node and relationship creation |
| **Q3** | `MATCH` a traveller's itinerary and its legs |
| **Q4** | traverse a flight → its aircraft → that aircraft's whole day |
| **Q5** | **multi-hop**: the rotation cascade |
| **Q6** | **project-specific**: correlated failure |
| `connection` | connection feasibility, derived |
| `policy` | which published commitments apply to *this* disruption |

**Q5 — the cascade.** One aircraft, five legs, 19 July 2024:

| Flight | | Arrival delay | Carrier | Late aircraft |
|---|---|---|---|---|
| WN4854 | LBB→DAL | 0 | – | – |
| WN4854 | DAL→MCO | −9 | – | – |
| WN1547 | MCO→CMH | **+105** | **105** | 0 |
| WN2637 | CMH→HOU | +115 | 2 | **106** |
| WN2679 | HOU→MDW | +115 | 2 | **113** |

The aircraft starts the day clean, breaks on leg 3 with a *carrier* delay — consistent
with the CrowdStrike outage that grounded flights that morning — and the next two legs
inherit it. The graph traversal finds the propagation; **BTS's own cause codes confirm
it independently.** The path is the explanation.

**Q6 — correlated failure.** The nine Atlanta→Newark options described above, each with
the upstream leg its aircraft was still flying. This is the query that justifies the
graph.

---

## Data and scope

**BTS On-Time Performance + NOAA Integrated Surface Database**, 13.7M flights across
2023–2024, rebuilt from primary sources during Challenge 2 after a preprocessed Kaggle
version was found to have stripped cancellations, broken timestamps and a missing month.

- **Carriers:** nine US majors — `AA DL UA WN AS B6 F9 G4 HA` — by **operating**
  carrier. The rule is "the plane you are actually sitting on."
  Spirit was removed after it ceased operations on 2 May 2026
- **Seed day:** 19 July 2024, the CrowdStrike outage — 59.5% of mainline flights 15+
  minutes late against a 7.6% baseline. A real cascade is what a graph is for
- **Policy sources:** the DOT Airline Customer Service Dashboard (all nine carriers)
  plus the Delta and Southwest contracts of carriage

The graph is a **second index** over the same facts, not a replacement for the Parquet
store. DuckDB answers *"what happened to flights like this one"* by aggregating millions
of rows. Neo4j answers *"what is this flight connected to"* by traversing dozens of
nodes. Neither does the other's job well.

---

## Design decisions worth knowing

- **`Service` aggregates the whole year, not the loaded day.** At day grain it was
  effectively 1:1 with `Flight` and carried no information
- **Numbers are precomputed onto nodes.** An LLM narrating a retrieved subgraph cannot
  derive a quantile from it and will invent one if it is missing
- **Time is an indexed property, not a node.** An `Hour` node at a busy hub is a
  supernode, and two flights sharing an hour is not a relationship
- **Parameterised Cypher templates, never text-to-Cypher.** Three mistakes during this
  build returned *wrong answers rather than errors*: a string timestamp concatenated
  with a duration, a `count {}` subquery collapsed to 1 beside an aggregate, and an
  unbound relationship drew disconnected nodes. A model generating Cypher has no error
  signal to correct against
- **A commitment a carrier declines is an absent edge**, not a property saying "No"

---

## Testing

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q     # 70 tests
```

Neo4j-dependent tests skip without a reachable instance. They assert behaviour rather
than execution — that Q5 shows delay *growing* along the chain with BTS blaming late
aircraft, that every `PRECEDES` edge is spatially continuous, that timestamps have not
regressed to strings, and that no query hardcodes a sample value.

---

## Limitations

- **Regional carriers are out of scope.** 86 airports and 1,152 routes are absent, so a
  traveller connecting to a small city on a regional jet gets nothing
- **One day is loaded.** Rotations are complete within it; chains crossing midnight are not
- **One global 45-minute connection rule.** Real minimum connection times vary by
  airport and terminal and we have no source for them
- **Policy coverage is partial.** Four of the nine carriers block automated retrieval,
  so contract clauses cover Delta and Southwest only. The DOT dashboard had to be saved
  from a browser
- **No GraphRAG yet.** The schema carries the filter keys a retriever will need, but
  retrieval arrives at the Human–AI Co-Design stage
- **No cancellation probability, no fare.** Cancellations are recorded but not predicted;
  no fare data exists in any source currently used
