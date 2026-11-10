# Results

Every number on this page comes from a file in this folder. Each section names
its file and the script that produced it. Nothing here is illustrative.

**Split.** Models train on 2023–2024. 2025 is the validation year: every
setting was chosen on it. 2026 (January–August) is the test year. It was held
out until every model and rule was frozen, then scored once.

**Scale.** The 2026 test is 1,439,445 forecasts: 479,815 sampled flights, each
forecast 24, 6 and 2 hours before departure. 2025 is 2,070,453 forecasts.

---

## 1. Forecast: the 2026 test

The model, `final_network`, gives a probability for each of 16 outcomes:
early, on time to 15 minutes late, then 15-minute steps to 180+ minutes,
cancelled and diverted. The score is the multiclass Brier score. Lower is
better, and a uniform guess scores 0.9375.

| Model | 2026 | 2025 |
|---|---|---|
| Uniform guess | 0.9375 | 0.9375 |
| Climatology, all airports | 0.6172 | 0.6043 |
| Climatology, by origin airport | 0.6145 | 0.6026 |
| Schedule only | 0.6024 | 0.5933 |
| Schedule + history | 0.5982 | 0.5900 |
| **Murphy (`final_network`)** | **0.5659** | **0.5602** |

**Murphy minus a simpler baseline**, with 95% intervals from resampling whole
days. Below zero means Murphy is better.

| Compared with | 2026 | 2025 |
|---|---|---|
| Climatology, by origin | −0.0486 (−0.0531 to −0.0446) | −0.0424 (−0.0446 to −0.0402) |
| Schedule only | −0.0366 (−0.0416 to −0.0326) | −0.0331 (−0.0352 to −0.0310) |
| Schedule + history | −0.0323 (−0.0369 to −0.0284) | −0.0297 (−0.0317 to −0.0278) |

By forecast time on 2026, against climatology: −0.0729 at 2 h, −0.0415 at 6 h,
−0.0314 at 24 h. The closer to departure, the more the model knows.

**Are the risk bands honest?** The app sorts each trip into a band by its
chance of missing a 60-minute deadline. True miss rate, with the predicted
rate in brackets:

| Band | 2026 | 2025 |
|---|---|---|
| Keep (below 40%) | 8.9% (8.7%) | 8.5% (7.8%) |
| Too uncertain (40–60%) | 52.7% (48.4%) | 53.3% (48.0%) |
| Consider changing (60% and up) | 81.8% (74.8%) | 82.0% (74.5%) |

The top band misses more often than predicted. The forecast runs slightly low,
so the advice to change is, if anything, cautious.

**Pre-set tests: all seven pass** (`forecast_tests.json`). They were written
before 2026 was scored: G1–G2 beat climatology overall and at every forecast
time, G3 keep at least half of 2025's gap, G4 beat schedule-only, G5
calibration no worse than climatology, G6–G7 the two upper bands hold.

Files: `forecast/2026_test/`, `forecast/2025_validation/`
(`scores`, `diffs`, `bands`, `calibration`, `kinds`). Script: `scripts/eval_2026.py`.

---

## 2. Advice: keep or change?

The replay asks a counterfactual question about past trips: if a traveler had
followed the advice, would they have made their deadline? Each trip has a
deadline: 30, 60 or 120 minutes after scheduled arrival. When the advice says
"change", the outcome is one of four:

- **Rescued:** the original flight would have missed; the new one made it.
- **Needless:** both would have made it.
- **Made worse:** the original would have made it; the new one missed.
- **Both missed.**

**Net** = rescued minus made worse, per 1,000 trips. The baseline is a
**traveler's rule of thumb**: if the inbound aircraft is already later than the
deadline allows, take the next flight.

| Deadline | Rule | Year | Net per 1,000 (95%) | Changes per 1,000 | Rescued : needless | Made worse |
|---|---|---|---|---|---|---|
| 60 min | **Murphy** | 2026 | **1.44** (1.28 to 1.62) | 3.40 | 3.1 | 3.4% |
| | | 2025 | 1.15 (1.05 to 1.24) | 2.22 | 3.8 | 2.7% |
| | Rule of thumb | 2026 | 0.19 (0.16 to 0.21) | 0.28 | 38.7 | 0.7% |
| | | 2025 | 0.19 (0.17 to 0.21) | 0.27 | 25.7 | 0.2% |
| 30 min | Murphy | 2026 | 3.58 (3.26 to 3.97) | 8.01 | 2.8 | 2.8% |
| | Rule of thumb | 2026 | 0.50 (0.45 to 0.54) | 0.77 | 42.2 | 0.2% |
| 120 min | Murphy | 2026 | 0.39 (0.32 to 0.47) | 1.25 | 2.3 | 5.7% |
| | Rule of thumb | 2026 | 0.05 (0.04 to 0.06) | 0.06 | 71.0 | 0.0% |

At 60 minutes in 2026, Murphy rescued 2,235 trips against 712 needless changes
and 165 made worse. The rule of thumb is precise but rarely fires. The
schedule-only model never reaches 60%, so under the same rule it changes
nothing.

**Pre-set tests: all six pass** (`advice_tests_2026.json`). R1 net above 0;
R2–R3 Murphy beats the rule of thumb at every deadline; R4 at least half of
2025's net; R5 rescued at least twice needless; R6 made worse at most 5% of
changes, at 60 minutes.

**Read with care.** 2026's higher net comes from a harder year with more trips
at risk, not from better advice per change. Rescued : needless fell from 3.8 to
3.1. At 120 minutes, made worse is 5.7% of changes, above the 5% line R6 sets
for 60 minutes.

Files: `advice/rules_2026.csv`, `rules_2025.csv`, `advice_tests_2026.json`,
`baselines_2025.csv`, `deadline_band_2025.csv`. Scripts:
`scripts/eval_2026_replay.py`, `replay_firm60.py`, `deadline_check.py`.

---

## 3. Ablation: what each input group adds

2025 validation, multiclass Brier. Each row adds a group of inputs to the row
above.

| Inputs | 2025 score |
|---|---|
| Climatology, by origin | 0.6026 |
| A: schedule (carrier, hour, weekday, month, distance, lead time) | 0.5933 |
| + B: history (trailing 365-day rates by route, carrier, airport) | 0.5900 |
| + C, D: current state (last 2 h at both airports) and the aircraft's previous leg | 0.5636 |
| + E: weather (observation and the forecast in force, 30 busiest airports) | 0.5575 |

Every step helps. Current state and the aircraft help most. This ablation ran
before two later changes, spreading forecast times across 1–24 h and hiding
aircraft hindsight. Its scores therefore don't compare directly with the final
model's. Hiding hindsight costs some score: [BTS](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr) records the aircraft that
actually flew, which a traveler could not have known.

**Network inputs** (communities and the GCN, section 4) on top of the final
inputs: −0.00072 (−0.00101 to −0.00050) on 2026, −0.00093 on 2025. The effect is
small but it holds.

Files: `forecast/ablation_*.json`. Script: `python -m murphy.forecast.gbm ablate_l2`.

---

## 4. Methods compared

One row per method in the app: what was used, the simpler alternative, and the
result.

| Method | Used | Compared with | Result (2025 unless stated) | File |
|---|---|---|---|---|
| **Communities** | Louvain on a co-disruption graph (airports that go bad together), 5 regions | Whole-country input | Forecast change −0.00041 against −0.00022 for the whole country | `methods/network/quick_compare.csv`, `communities.csv` |
| **GNN** | Two-layer GCN over 100 airports, co-disruption edges | No edges, route edges, simple averages | Halves the error of "usual for the hour" (0.0121 vs 0.0273), but edges don't help on its own target: no edges 0.0119, routes 0.0131. As a forecast input, co-disruption beats no edges (−0.00096 vs −0.00061) | `methods/network/gcn_compare.csv`, `gcn_seeds.csv`, `quick_compare.csv` |
| **GCN ranking** | GCN picks which neighbor airports to retrieve | Link weight, last 2 h, usual, random | Top-3 hit share 0.450 vs 0.299 for link weight and 0.294 random (653,549 airport-hours). The no-edge GCN does slightly better (0.460), so the gain is the learned model, not message passing | `methods/gcn_ranking/rank_neighbors.csv` |
| **GraphRAG** | Neo4j queries → evidence → LLM explains (Groq, `gpt-oss-120b`) | Without communities, BM25 over the same facts, no retrieval | 16 answers each (4 flights × 4 questions). Cite evidence: graph 15, BM25 11, none 0. Key fact correct: graph 6 of 7, BM25 5 of 7, none 0 of 7. Evidence holding facts not yet known at forecast time: graph 0, BM25 12. A made-up flight gets "No record of this flight", with no LLM call | `methods/graphrag/eval_run2.txt`, `eval_run2.jsonl` |
| **Policy retrieval** | Embeddings (`bge-small`) over federal rules and contracts of carriage | BM25, random | Relevant passage in top 3: embeddings 14 of 16, BM25 8 of 16, random 2.4 expected | `methods/retrieval/policies_eval.csv` |
| **Similar past days** | Nearest neighbors on standardized forecast numbers | Embeddings, BM25, same-month average, random | Mean abs. error in late share: numbers 0.0821, embeddings 0.0831, BM25 0.0828, month 0.0913, random 0.1015. The top three tie closely; numbers are slightly best | `methods/retrieval/pastdays_eval.csv` |

**GraphRAG before and after.** In the first run the evidence was right but the
LLM misread it. It mixed up whose chance was whose, read a fixed link weight as
today's risk, and decoded `19012G20KT` as "190 kt" (it means wind from 190° at
12 kt). The fix was to label each evidence field, not to change the model.
`eval_run1.txt` is before the fix; `eval_run2.txt` is after.

Scripts: `scripts/network.py`, `gcn_rank_eval.py`, `graphrag_eval.py`,
`policies_eval.py`, `pastdays_eval.py`.

---

## Limits

- **2026 covers January–August only.** No fall or holiday travel was tested.
- **The advice is a replay, not a field trial.** Alternatives come from the real
  schedule, but nobody actually rebooked. The ticket cost is shown in the app,
  but the advice ignores it.
- **Weather covers the 30 busiest airports.** Elsewhere the forecast has none,
  and the app says so.
- **The GraphRAG test is small.** Groq's free daily limit stopped the larger
  run. The counts show how each system fails, not how often.
- **No real users yet**, so whether the evidence changes travelers' decisions is
  untested.
