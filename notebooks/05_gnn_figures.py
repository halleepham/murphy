"""Measure the two things worth drawing, and write them for the figure script.

Separate from `04_gnn_ablation.py` because these are per-seed measurements rather than
the headline table, and because the plotting runs under system python3 with matplotlib
(deliberately not a project dependency).

Run:  PYTHONPATH=src .venv/bin/python notebooks/05_gnn_figures.py
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from murphy import gnn_model
from murphy.gnn_features import FLIGHT_FEATURES, build

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "gnn_figure_data.json"
SEEDS = (7, 13, 29, 42, 101)
SLACK = 15.0


def aucs(ds, layers: int, truth, mask) -> list[float]:
    out = []
    for seed in SEEDS:
        gnn_model.SEED = seed
        model, _ = gnn_model.train(ds, layers=layers, verbose=False)
        out.append(float(roc_auc_score(truth[mask], gnn_model.score_all(model, ds, SLACK)[mask])))
    return out


def main():
    ds = build()
    truth = (ds.delay_min <= SLACK).astype(int)
    m = ds.test_mask

    ablated = copy.deepcopy(ds)
    for c in ("up_delay_min_masked", "up_delay_observed"):
        ablated.x_flight[:, FLIGHT_FEATURES.index(c)] = 0.0

    data = {"seeds": list(SEEDS), "slack": SLACK, "no_skill_auc": 0.5}
    for name, d in (("full", ds), ("no_inbound_delay", ablated)):
        for layers, rung in ((0, "G1_MLP"), (1, "G3_GraphSAGE")):
            key = f"{name}__{rung}"
            data[key] = aucs(d, layers, truth, m)
            a = np.array(data[key])
            print(f"  {key:<34} {a.mean():.4f} +/- {a.std():.4f}")

    OUT.write_text(json.dumps(data, indent=2))
    print(f"\nwritten {OUT.name}")


if __name__ == "__main__":
    main()
