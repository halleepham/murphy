# Measured results

Committed so the numbers are readable without running anything. Both files are
regenerated, never hand-edited.

| File | Written by | Holds |
|---|---|---|
| `gnn_results.json` | `notebooks/04_gnn_ablation.py` | the five-rung ablation at both query slacks, five seeds each |
| `gnn_figure_data.json` | `notebooks/05_gnn_figures.py` | per-seed AUCs with and without the inbound-delay feature |

Reproducing either needs a loaded Neo4j instance and `requirements-gnn.txt`
installed; the ablation trains 20 models and takes a few minutes. See
[The GNN](../README.md#the-gnn).

`split_boundary` in `gnn_results.json` is the temporal train/test cut — every test
flight was scheduled to depart after it.
