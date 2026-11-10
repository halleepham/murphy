# Reproducing Murphy

Two ways in. **Run the app** from the release download, which takes minutes.
Or **rebuild everything** from the public sources, which takes hours and about
14 GB of disk.

Every command runs from the repository root with the virtual environment
active and `PYTHONPATH=.`:

```bash
source .venv/bin/activate
export PYTHONPATH=.
```

---

## 1. Run the app

```bash
python scripts/download_data.py
uvicorn murphy.app.main:app --port 8765
```

### Ask tab: the knowledge graph and GraphRAG

1. Create a free [Neo4j Aura](https://neo4j.com/cloud/aura-free/) instance and
   a free [Groq](https://console.groq.com/) API key.
2. `cp .env.example .env` and fill in both.
3. Download the graph data (an 816 MB download of flights and forecast rows):
   `python scripts/download_data.py --graph`
4. Load the graph: `python scripts/build_graph.py load --clear`.
   This deletes everything in the instance first.

### Rules tab: policy retrieval

1. Save the five sources listed in `SOURCES` in `murphy/policies.py` to
   `private/policies/raw/`, using those file names.
2. Build the passages and embeddings: `python -m murphy.policies`

---

## 2. Rebuild everything

The steps are listed in dependency order. Each script reads only what the
steps above it wrote. They were run one at a time during development, not
end-to-end in a single run.

### Fetch the data

```bash
python scripts/fetch_bts.py 2023-01 2026-08
python scripts/fetch_iem.py
```

Then download [`airports.csv`](https://davidmegginson.github.io/ourairports-data/airports.csv) from [OurAirports](https://ourairports.com/data/) to
`data/raw/ourairports/`, and the Q2 2025 [DB1B Market](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK&QO_fu146_anzr=b0-gvzr) file from BTS to
`data/raw/db1b/DB1BMarket_2025_2.zip` (see [DATA.md](DATA.md)).

### Build the data layer

```bash
python -m murphy.datalayer.bts data/raw/bts/*.zip   # flights -> data/flights/
python -m murphy.datalayer.airports                 # time zones
python -m murphy.datalayer.weather                  # ASOS + TAF -> data/weather/
python -m murphy.forecast.sample                    # forecast rows
python -m murphy.forecast.climatology               # climatology baselines
python -m murphy.forecast.features                  # input groups A–E
```

### Train

```bash
python -m murphy.forecast.gbm ablate_l2             # ablation; schedule-only baselines
python scripts/refit_leads.py                       # forecast times spread over 1–24 h
python scripts/refit_clean.py prepare               # hide aircraft hindsight
python scripts/refit_clean.py fit                   # model without network inputs
python scripts/network.py communities               # Louvain communities
python scripts/network.py state                     # regional inputs (group F)
python scripts/network.py gcn                       # train the GCNs
python scripts/network.py gnn_inputs co g           # GCN inputs, co-disruption edges
python scripts/network.py seeds                     # two more GCN seeds
python scripts/network.py average                   # 3-seed average (group M)
python scripts/network.py fit_full                  # the final model: final_network
```

### Replay 2025 and build the app

```bash
export MURPHY_MODEL=final_network                   # the replay scripts read this
python scripts/replay_cutoffs.py
python scripts/replay_baselines.py
python scripts/replay_firm60.py
python scripts/build_fares.py
python scripts/build_app_data.py
python scripts/build_graph.py load --clear          # needs .env
python scripts/export_links.py                      # map links, from Neo4j
python -m murphy.pastdays
python -m murphy.policies                           # needs the policy sources
```

### Evaluate

```bash
python scripts/eval_2026.py inputs
python scripts/eval_2026.py score tune              # must reproduce the 2025 scores
python scripts/eval_2026.py score test              # 2026, scored once
python scripts/eval_2026.py compare                 # tests G1–G7
python scripts/eval_2026_replay.py check
python scripts/eval_2026_replay.py build tune 2025-01
python scripts/eval_2026_replay.py build test
python scripts/eval_2026_replay.py tally test       # tests R1–R6
python scripts/deadline_check.py
```

Method comparisons (section 4 of [RESULTS.md](../results/RESULTS.md)):

```bash
python scripts/gcn_rank_eval.py                     # GCN ranking vs simpler orders
python scripts/graphrag_eval.py                     # needs .env; Groq free tier limits apply
python scripts/pastdays_eval.py
python scripts/policies_eval.py
```

Outputs are written under `data/`. The files kept in `results/` are copies.
