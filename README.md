# Murphy — Challenge 3: Knowledge Graph + GraphRAG + GNN

**Which flights fail for the same reason?**

Murphy forecasts flight disruption risk and states how much to trust its own forecast.
Challenge 2 answered *"will I land on time?"* from a table of comparable flights.
Challenge 3 builds the thing that table could never express: **how flights are
connected to each other**, and therefore which ones break together — then puts a
natural-language interface on top of it.

Course project for CS 5542 (Big Data Analytics and Applications), Fall 2026.

> **This branch covers all three Challenge 3 stages** — the knowledge graph (AI
> Design), GraphRAG over it, and a GNN ranker on top of that (both Human–AI
> Co-Design). The Challenge 2 application —
> parser, retrieval, forecast model, route planner and Streamlit interface — is tagged
> [`challenge-2`](../../tree/challenge-2) and deployed at
> [murphy-challenge-2.streamlit.app](https://murphy-challenge-2.streamlit.app/).

---

## Why a graph

The Challenge 2 report named a limitation it could not fix: *"flight legs are scored
independently, so correlated delay at a shared hub is currently invisible."*

A traveller stuck at Atlanta on 19 July 2024 trying to reach Newark had **fifteen
options**. A forecaster scoring each leg on its own sees fifteen separate moderate
risks. Eight of them have an inbound flight inside the loaded day — and **every one of
those eight was flown by an aircraft already 45+ minutes late**, arriving an average of
**268 minutes** behind schedule against 212 for the rest. There was no escape, and
nothing in a row-oriented store could say so.

Answering that question means walking backwards from every candidate flight to the
aircraft that will operate it, and from that aircraft to where it currently is. That is
a self-join nobody writes in SQL. It is two hops in a graph.

---

## Contents

- [The schema](#the-schema)
- [GraphRAG](#graphrag)
- [The GNN](#the-gnn)
- [Running it](#running-it)
- [The queries](#the-queries)
- [Data and scope](#data-and-scope)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Testing](#testing)
- [Limitations](#limitations)

---

## The schema

**9 node types · 10 relationship types · 37,059 nodes · 117,489 relationships**

The GNN stage adds one label and two relationship types to this same graph without
changing anything already here, taking it to 12 relationship types and 147,881
relationships. [See below](#what-the-gnn-needed-from-the-graph).

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

## GraphRAG

> question → Neo4j retrieval → retrieved context → LLM → grounded answer

`src/murphy/graphrag.py` puts a natural-language interface on the graph.
`notebooks/03_graphrag.ipynb` demonstrates and evaluates it.

```bash
PYTHONPATH=src .venv/bin/python -m murphy.graphrag   # run every retriever
jupyter notebook notebooks/03_graphrag.ipynb         # the demonstration
```

Needs `GROQ_API_KEY` in `.env` alongside the Neo4j values. The model is
`openai/gpt-oss-120b`, the same one the Challenge 2 parser uses.

### Four retrievers, one per shape of question

| | Mechanism | Answers |
|---|---|---|
| `itinerary_lookup` | Cypher | one trip and its legs |
| `aircraft_cascade` | Cypher, `[:PRECEDES*1..4]` | **multi-hop** — what a delay caused downstream |
| `route_risk_summary` | Cypher aggregate | how many, how often, on average |
| `policy_lookup` | full-text index + graph filters | what the traveller is owed |

An LLM picks between them (`ToolsRetriever`). That is tool selection, **not** a
verification loop — there is no retry, no evidence check and no abstention.

### The graph narrows, the index reads

The policy retriever is where the graph and the documents meet. Asking *"can I get a
hotel or a meal"* **with** the flight's context returns five Delta clauses — Delta
operated the flight, the delay was 186 minutes, and the cause was within the airline's
control. Asking **without** it returns clauses from five other carriers and none from
Delta: fluent, grounded in real published documents, and useless to this traveller.

The graph supplies carrier, cause and delay; those three facts turn a text search into
an answer. A 186-minute delay surfaces a meal commitment with a three-hour threshold;
a 56-minute delay on the same trip does not.

### Two rules, enforced rather than hoped for

- **No Text2Cypher.** Queries are fixed, parameterised templates. Building this graph
  produced three Cypher mistakes that returned *wrong answers rather than errors*, so a
  model generating Cypher has nothing to correct against.
- **The LLM never produces a value.** Every number in an answer is retrieved from the
  graph, and the evidence travels beside the answer so it can be checked.

### What the evaluation found

Three things, all reproducible in the notebook:

- **Grounding stops invention, not omission.** Q2's answer is numerically perfect and
  still drops the carrier attribution and the clean first leg — the two things that
  show the delay *originated* with the airline. Grounding constrains what a model may
  say; it does not make it say the useful part.
- **Ambiguous field names are a hallucination surface.** An aggregate called
  `avg_arrival_delay_min` led the model to report 257 minutes for a subset whose true
  average is 268. Every number it used *was* in the evidence. Renaming the field to say
  which set it covers fixed it with no prompt change. *"The LLM never produces a
  number"* is necessary, not sufficient — the retriever must return values that cannot
  be misread.
- **Routing is unstable, and grounding is the only backstop.** One out-of-scope
  question, eight runs, temperature 0 — two different tools chosen. Every run still
  answered *"the graph does not contain this"* rather than inventing a forecast. The
  safety property survives a moving target, but it is a behaviour rather than a
  guarantee: a verification step confirming the retrieved rows actually address the
  question would make it a property.

---

## The GNN

> candidate flights → one round of message passing → a ranked list, with the rotation
> chain behind each score

The graph answers *"which of these fail together"* only if a human reads the rows. The
GNN asks whether a model can learn that structure instead, and the honest answer here
is **no — for a reason worth more than a win.**

`notebooks/04_gnn.ipynb` is the executed demonstration. `04_gnn_ablation.py` reproduces
every number below.

### The task is ranking, not forecasting

A traveller asks *"will this get me there by 18:40"*, and the same flight is relevant
for a late deadline and irrelevant for an early one. So the model scores a **pair** —
`s(flight, slack)`, where slack is the minutes of margin the query allows. Set slack to
15 minutes and the score becomes ordinary lateness prediction, so forecasting is a
special case rather than a second model.

**Scope is Southwest only.** Across all nine carriers, carrier alone nearly separates
the classes — Delta 91.9% late against Alaska 19.5% on the seed day — so a model can
memorise the carrier, score well, and learn nothing from structure. One carrier leaves
headroom to measure what the graph adds.

### What the GNN needed from the graph

`src/murphy/kg_congestion.py`, additive and idempotent. No existing node, relationship,
property or query was modified.

| | Before | After | Cap |
|---|---|---|---|
| Nodes | 37,059 | 37,059 | 19% of 200,000 |
| Relationships | 117,489 | **147,881** | **37%** of 400,000 |

- `:AirportHour` added as a **second label** on 2,843 existing `Weather` nodes, carrying
  scheduled departures, arrivals and movements
- **15,194** `DEPARTS_IN` and **15,198** `ARRIVES_IN` edges

A second label rather than a rename, because renaming `Weather` would have invalidated
the loader, the queries, the schema diagram and 13 screenshots. Neo4j nodes carry many
labels.

### One architecture at two depths

`Ranker(layers=0)` is an MLP on a flight's own features. `Ranker(layers=1)` adds one
round of GraphSAGE message passing over the rotation chain and the airport-hour.
**Nothing else differs**, so the comparison isolates the convolution rather than the
capacity.

Depth is 1 by correctness, not preference: a second hop would reach flights that had
not yet departed when this one was due out.

### The ablation — 5 seeds, Southwest test window, slack 15 min

| Rung | AUC | ± | AUPRC | Recall@3 origin |
|---|---|---|---|---|
| G0 baseline rate | 0.650 | — | 0.557 | 0.199 |
| **G1 MLP** | **0.792** | 0.011 | 0.695 | 0.258 |
| G2a rotation rule | 0.659 | — | 0.570 | 0.199 |
| G2b congestion rule | 0.589 | — | 0.492 | 0.182 |
| G3 GraphSAGE | 0.791 | 0.016 | 0.688 | 0.245 |

No-skill AUPRC is 0.421.

**Learning beats the hand-set rules decisively** — 0.792 against the rotation rule the
system ships today at 0.659 and the congestion rule at 0.589. That is a result for
Murphy independent of any GNN, and it is the largest and most stable finding here.

**Structure adds nothing.** G3 and G1 are indistinguishable, well inside one standard
deviation. The GNN is not kept.

### Why not — and it is this project's own rule

Zeroing the inbound-delay feature, so the graph is the only remaining route to rotation
information:

| | AUC @15, 5 seeds |
|---|---|
| G1 MLP, inbound delay as a feature | 0.792 ±0.011 |
| G3 GNN, inbound delay as a feature | 0.791 ±0.016 |
| G1 MLP, feature removed | 0.545 ±0.013 |
| G3 GNN, feature removed | 0.533 ±0.007 |

The inbound aircraft's delay carries **essentially all** the predictive signal — hour,
duration, distance, airport and weather together are worth almost nothing on this day.
And the graph does not recover it: G3 without the feature is no better than G1 without
it.

The reason is exact. `outcome_*` properties are excluded from features by standing rule,
so an inbound flight's node vector holds **no delay information at all**. There is
literally nothing delay-related on the graph for message passing to carry. The rotation
signal reaches the model only through an explicitly masked upstream feature — which is a
*feature*, not a traversal.

So the finding is not "GNNs do not help." It is that **the provenance discipline which
makes Murphy's forecasts trustworthy is exactly what leaves the GNN nothing to
propagate.** That is not an argument for relaxing the rule. It is an argument for giving
the graph structural signals that are not outcomes — congestion propagating between
airports, two-leg rotation chains, connection feasibility. Each lives in the topology
rather than on a node, which is the only place a GNN can earn its keep.

Worth noting what survives the negative result: the GNN is the **only** method in the
ablation where two candidates' scores are not computed independently. Both read from the
same `AirportHour` node, so a bad hour moves them together. The mechanism for correlated
risk is right; message passing adding signal is what failed.

### Two defects the measurement caught

- **A single-seed result reversed on reseeding.** The first run had the GNN ahead. Five
  seeds showed the gap was inside the noise.
- **A missing `LayerNorm` surfaced only when the explanation was built.** Asking the
  model to explain one real flight exposed instability that code review and evaluation
  both missed. Every number reported before the fix was re-measured after it, and two
  conclusions changed.

Three of this project's defects — two Cypher errors, the single-seed result, and this
one — were invisible to review and surfaced only when something was built to be
*displayed*.

## Running it

A free [Neo4j Aura](https://console.neo4j.io) instance is enough — the graph uses 19%
of its node cap, and 31% of its relationship cap before the GNN stage or 37% after.

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

The GNN stage needs two more steps and a separate requirements file, kept out of
`requirements.txt` so the deployed app does not pull ~200 MB of PyTorch it never
imports:

```bash
.venv/bin/pip install -r requirements-gnn.txt

PYTHONPATH=src .venv/bin/python -m murphy.kg_congestion  # add :AirportHour + 2 edge types
PYTHONPATH=src .venv/bin/python -m murphy.gnn_export     # export the WN subgraph
PYTHONPATH=src .venv/bin/python -m murphy.gnn_features   # features, labels, split, guards
PYTHONPATH=src .venv/bin/python notebooks/04_gnn_ablation.py   # the five rungs, 5 seeds
jupyter notebook notebooks/04_gnn.ipynb                  # the demonstration
```

The ablation trains 20 models and takes a few minutes, so its output is committed to
[`results/`](results/) — the measured numbers are readable without running anything.

The pins are not free choices: this is an x86_64 Mac, PyTorch dropped Intel macOS
builds after 2.2.2, and that build was compiled against NumPy 1.x — which is why
`requirements.txt` pins `numpy<2`.

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

The aircraft flies two clean legs, then takes a 105-minute delay that BTS attributes
entirely to **carrier** — its category for causes within the airline's control, such as
crew, maintenance or fuelling. The next two legs inherit it, and BTS assigns 106 and 113
minutes of *those* to **late aircraft**.

The traversal finds the propagation; **the cause codes confirm it independently.** The
path is the explanation.

Two things this does *not* claim. BTS records the category, never the incident, so a
carrier delay is not evidence of any particular cause — 19 July was a severe day
network-wide, but Southwest was among the least affected carriers on it, cancelling
0.11% of its flights against a 0.68% baseline. And "starts clean" means within the
loaded day; the graph holds one day, so an aircraft's earlier history is outside it.

**Q6 — correlated failure.** The Atlanta→Newark options described above, each with the
upstream leg its aircraft was still flying. This is the query that justifies the graph.

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
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q           # 85 tests
MURPHY_LLM_TESTS=1 PYTHONPATH=src .venv/bin/python -m pytest tests/ -q   # + 2 LLM tests
```

Neo4j-dependent tests skip without a reachable instance, and the two that spend API
calls are opt-in. They assert behaviour rather than execution — that every `PRECEDES`
edge is spatially continuous, that timestamps have not regressed to strings, that no
query hardcodes a sample value, that narrowing a policy search really does reduce it to
one carrier, and that every aggregate names the set it covers.

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
- **Routing is unstable and unverified.** Tool selection only — no retry, evidence
  check or abstention, and the same question can route differently between runs
- **Four fixed retrievers.** An unanticipated question routes to the nearest one and is
  answered confidently from the wrong evidence, or refused
- **The GNN is a measured negative.** One round of message passing matches a plain MLP,
  because the provenance rule keeps outcomes off the nodes. It is demonstrated and
  evaluated, not shipped
- **No cancellation probability, no fare.** Cancellations are recorded but not predicted;
  no fare data exists in any source currently used
