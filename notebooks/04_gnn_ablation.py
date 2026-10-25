"""U4: the five-rung ablation, at two candidate scopes.

Does a learned ranker beat the rules already in the system? Five rungs, the same
features, the same temporal split, the same candidate budget:

    G0   stored baseline rate            the lookup table (ladder rung L1)
    G1   MLP, zero message-passing       features without structure
    G2a  rotation rule, inbound 45+ late the structure the system already uses
    G2b  congestion rule, busy hour      the GNN's own information, no learning
    G3   GraphSAGE, one layer            structure with learning

G1 and G3 are one architecture at two depths, so the comparison isolates the structure
rather than two model families. G3 earns its place only by beating **both** rules: G2a
says it beats what ships today, G2b says it beats the simplest use of its own input.

Two scopes, because route-scope sets turned out too small to rank (decision N10):
route scope is the product's real question, origin scope is where slide 20's
"large candidate set" condition is closer to met.

Run:  PYTHONPATH=src .venv/bin/python notebooks/04_gnn_ablation.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from murphy.gnn_features import build
from murphy import gnn_model

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "gnn_results.json"

# The query slacks evaluated. 15 is the BTS convention the project uses everywhere; 45
# is the global connection time from the Challenge 2 decision log.
EVAL_SLACKS = (15.0, 45.0)

ROUTE_MIN = 3    # measured: 99 route sets reach this; none reaches 10
ORIGIN_MIN = 10  # measured: 31 origin sets reach this

# Every learned rung is trained at five seeds and reported as mean +/- std.
#
# This is not thoroughness for its own sake. A single run at seed 7 showed the GNN
# beating the MLP by +0.015 AUC, which reseeding showed to be noise: across five seeds
# the GNN is *behind* by 0.012 and carries twelve times the variance. One seed would
# have produced the opposite conclusion and nothing in the output would have looked
# wrong (decision N11).
SEEDS = (7, 13, 29, 42, 101)


def recall_at_k(ranked: list[int], gold: set[int], k: int) -> float:
    """The course lab's definition, kept for comparability."""
    return len(set(ranked[:k]) & gold) / max(1, len(gold))


def candidate_sets(ds, kind: str) -> dict:
    """Test-window candidates, grouped by the scope being evaluated."""
    idx = np.where(ds.test_mask)[0]
    groups = collections.defaultdict(list)
    for i in idx:
        key = (ds.origin[i], ds.dest[i]) if kind == "route" else ds.origin[i]
        groups[key].append(int(i))
    floor = ROUTE_MIN if kind == "route" else ORIGIN_MIN
    return {k: v for k, v in groups.items() if len(v) >= floor}


def rule_scores(ds, kind: str) -> np.ndarray:
    """The two rule rungs, with ties broken by G0 as the spec requires.

    G0 is subtracted rather than added: pct_late15 is a *risk*, so a higher value means
    a worse candidate. The tie-break term is scaled to stay below the rule's own step,
    so it orders within a block without ever reordering across blocks.
    """
    g0 = -ds.service_pct_late15 / 100.0
    if kind == "g2a":
        late_inbound = (ds.up_observed == 1.0) & (ds.up_delay_raw >= 45.0)
        return -late_inbound.astype(np.float32) + 0.01 * g0
    movements = ds.dep_movements / max(float(ds.dep_movements.max()), 1.0)
    return -movements + 0.01 * g0


def evaluate(ds, scores: np.ndarray, slack: float) -> dict:
    """Classification on the whole test window, ranking within each candidate set."""
    test = ds.test_mask
    # Relevant = arrived within the slack the query allowed. Observed, not invented.
    truth = (ds.delay_min <= slack).astype(int)
    out = {}

    for name, mask in (("all", test), ("with_upstream", test & ds.has_upstream)):
        yt, ys = truth[mask], scores[mask]
        if len(set(yt.tolist())) < 2:
            out[f"auc_{name}"] = out[f"auprc_{name}"] = float("nan")
        else:
            out[f"auc_{name}"] = roc_auc_score(yt, ys)
            out[f"auprc_{name}"] = average_precision_score(yt, ys)
        out[f"base_rate_{name}"] = float(yt.mean())

    for kind, ks in (("route", (1, 3)), ("origin", (3, 5, 10))):
        sets = candidate_sets(ds, kind)
        per_k = {k: [] for k in ks}
        for members in sets.values():
            gold = {i for i in members if truth[i] == 1}
            if not gold or len(gold) == len(members):
                continue          # a set with no answer, or all answers, ranks nothing
            ranked = sorted(members, key=lambda i: -scores[i])
            for k in ks:
                per_k[k].append(recall_at_k(ranked, gold, k))
        for k in ks:
            vals = per_k[k]
            out[f"recall@{k}_{kind}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"n_sets_{kind}"] = len(per_k[ks[0]])
    return out


def mean_std(dicts: list[dict]) -> dict:
    """Average the metrics across seeds, keeping the spread on AUC."""
    keys = dicts[0].keys()
    out = {k: float(np.nanmean([d[k] for d in dicts])) for k in keys}
    out["auc_all_std"] = float(np.nanstd([d["auc_all"] for d in dicts]))
    out["auc_all_min"] = float(np.nanmin([d["auc_all"] for d in dicts]))
    return out


def learned(ds, layers: int, slack: float) -> list[dict]:
    per_seed = []
    for seed in SEEDS:
        gnn_model.SEED = seed          # train() reads this; seeding the global RNG does
        model, _ = gnn_model.train(ds, layers=layers, verbose=False)   # not work
        per_seed.append(evaluate(ds, gnn_model.score_all(model, ds, slack), slack))
    return per_seed


def main():
    print("building features ...")
    ds = build()
    s = ds.summary()
    print(f"  WN {s['flights']:,} labelled | train {s['train']:,} / test {s['test']:,} "
          f"| {s['late_pct']}% late | {s['with_upstream']:,} with an inbound leg")
    print(f"  learned rungs trained at {len(SEEDS)} seeds each\n")

    results = {}
    for slack in EVAL_SLACKS:
        print(f"slack {slack:.0f}m: training ...")
        results[f"slack_{int(slack)}"] = {
            "G0  baseline rate":   evaluate(ds, -ds.service_pct_late15 / 100.0, slack),
            "G1  MLP":             mean_std(learned(ds, 0, slack)),
            "G2a rotation rule":   evaluate(ds, rule_scores(ds, "g2a"), slack),
            "G2b congestion rule": evaluate(ds, rule_scores(ds, "g2b"), slack),
            "G3  GraphSAGE":       mean_std(learned(ds, 1, slack)),
        }
    print()

    for key, table in results.items():
        slack = key.split("_")[1]
        first = next(iter(table.values()))
        print(f"=== query slack {slack} min "
              f"(no-skill AUPRC {first['base_rate_all']:.3f}, "
              f"{first['n_sets_route']:.0f} route sets, "
              f"{first['n_sets_origin']:.0f} origin sets) ===")
        print(f"{'rung':<21}{'AUC':>7}{'+/-':>7}{'AUPRC':>7}"
              f"{'R@1 rt':>8}{'R@3 rt':>8}{'R@3 or':>8}{'R@5 or':>8}{'R@10 or':>9}")
        for name, r in table.items():
            sd = f"{r['auc_all_std']:.3f}" if "auc_all_std" in r else "  --  "
            print(f"{name:<21}{r['auc_all']:>7.3f}{sd:>7}"
                  f"{r['auprc_all']:>7.3f}{r['recall@1_route']:>8.3f}"
                  f"{r['recall@3_route']:>8.3f}{r['recall@3_origin']:>8.3f}"
                  f"{r['recall@5_origin']:>8.3f}{r['recall@10_origin']:>9.3f}")
        print()

    OUT.write_text(json.dumps(
        {"summary": s, "split_boundary": str(ds.dep_cut), "seeds": list(SEEDS),
         "results": results}, indent=2, default=float))
    print(f"written {OUT.name}")


if __name__ == "__main__":
    main()
