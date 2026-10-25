"""U5 and U6: the worked message-passing example, and the ranking demo.

Both exist to be *shown*, so every number here is printed from the trained model rather
than written by hand. If a slide quotes a vector, this is where it came from.

U5 answers the assignment's "show how neighboring nodes contribute to updated node
embeddings" on Murphy's own graph instead of a toy one: a real Southwest flight, the
inbound aircraft that fed it, the airport-hour it departed into, and the three steps of
message -> aggregate -> update.

U6 answers "GNN-enhanced retrieval or ranking": one traveler's question, the candidate
flights ordered by the rule the system ships today and by the GNN, with what actually
happened beside each.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from . import kg
from .gnn_features import HOUR_FEATURES, Dataset
from .gnn_model import Ranker, score_all, tensors

SHOW = 6   # leading dimensions printed; 32 is too wide to read on a slide


def _fmt(v, n=SHOW) -> str:
    return "[" + " ".join(f"{x:+.3f}" for x in v[:n]) + (" ...]" if len(v) > n else "]")


@torch.no_grad()
def worked_example(model: Ranker, ds: Dataset, flight_id: str | None = None) -> dict:
    """One flight's embedding update, traced through message / aggregate / update.

    Picks a flight that actually has both kinds of neighbor, because a flight with no
    inbound leg would show the mechanism doing nothing -- true of 19% of them, and
    honest, but not instructive.
    """
    t = tensors(ds)
    idx = {f: i for i, f in enumerate(ds.flight_ids)}

    rot_in = {}
    for s, d in ds.rot_edges.T:
        rot_in.setdefault(int(d), []).append(int(s))
    hour_in = {}
    for s, d in ds.hour_edges.T:
        hour_in.setdefault(int(d), []).append(int(s))

    if flight_id is None:
        # prefer a flight whose inbound leg was late and observable: the mechanism has
        # something real to carry
        best = None
        for i in range(len(ds.flight_ids)):
            if i in rot_in and i in hour_in and ds.up_observed[i] and ds.up_delay_raw[i] >= 30:
                if best is None or ds.up_delay_raw[i] > ds.up_delay_raw[best]:
                    best = i
        i = best if best is not None else next(iter(rot_in))
    else:
        i = idx[flight_id]

    # --- step 0: the node's own projected state, before any message passing.
    # Mirrors Ranker.embed exactly -- no activation here, because the model normalizes
    # and activates once at the end. Tracing a different computation than the model runs
    # would put numbers on a slide that the model never produced.
    h0 = model.project(torch.cat([
        t["x_flight"], model.origin(t["origin_idx"]), model.dest(t["dest_idx"])
    ], dim=1))

    # --- step 1 MESSAGE: what each neighbor contributes
    inbound = rot_in.get(i, [])
    hours = hour_in.get(i, [])

    # --- step 2 AGGREGATE: SAGEConv means its neighbors before applying weights, so the
    # mean is the quantity the layer actually sees.
    rot_mean = h0[inbound].mean(0) if inbound else torch.zeros(h0.shape[1])
    hour_mean = (t["x_hour"][hours].mean(0) if hours
                 else torch.zeros(t["x_hour"].shape[1]))

    # --- step 3 UPDATE: the layer's own output, then the residual the model uses
    out = model.conv({"flight": h0, "hour": t["x_hour"]}, t["edges"])["flight"]
    h1 = torch.relu(model.norm(h0 + out))
    # The no-message-passing counterfactual: what this node's embedding would be if the
    # graph were not there. The difference between the two is the neighborhood's whole
    # contribution, which is what the assignment asks to be shown.
    h_alone = torch.relu(model.norm(h0))

    return {
        "flight_id": ds.flight_ids[i],
        "route": f"{ds.origin[i]}-{ds.dest[i]}",
        "outcome_arr_delay_min": float(ds.delay_min[i]),
        "inbound": [{
            "flight_id": ds.flight_ids[j],
            "route": f"{ds.origin[j]}-{ds.dest[j]}",
            "delay_min": float(ds.delay_min[j]),
            "observed_in_time": bool(ds.up_observed[i]),
            "h": h0[j].numpy(),
        } for j in inbound],
        "hours": [{
            "weather_id": ds.hour_ids[j],
            "x": t["x_hour"][j].numpy(),
        } for j in hours],
        "h_self_before": h0[i].numpy(),
        "rot_message_mean": rot_mean.numpy(),
        "hour_message_mean": hour_mean.numpy(),
        "conv_output": out[i].numpy(),
        "h_without_graph": h_alone[i].numpy(),
        "h_after": h1[i].numpy(),
        "delta_norm": float(torch.norm(h1[i] - h_alone[i])),
        "self_norm": float(torch.norm(h_alone[i])),
    }


def print_worked_example(w: dict) -> None:
    print(f"FLIGHT  {w['flight_id']}  ({w['route']})")
    print(f"        arrived {w['outcome_arr_delay_min']:+.0f} min  <- the label, never an input\n")

    print("1  MESSAGE -- what the neighbors are")
    for n in w["inbound"]:
        seen = "observable before departure" if n["observed_in_time"] else "MASKED, landed too late"
        print(f"   inbound  {n['flight_id']}  ({n['route']})  arrived {n['delay_min']:+.0f} min")
        print(f"            {seen}")
        print(f"            h = {_fmt(n['h'])}")
    for h in w["hours"]:
        print(f"   hour     {h['weather_id']}")
        print(f"            x = {_fmt(h['x'])}")
        print(f"            ({HOUR_FEATURES[0]} .. standardized)")

    print("\n2  AGGREGATE -- mean over each relation, which is what SAGEConv sees")
    print(f"   rotation  {_fmt(w['rot_message_mean'])}")
    print(f"   hour      {_fmt(w['hour_message_mean'])}")

    print("\n3  UPDATE -- learned weights combine self with aggregate, then normalize")
    print(f"   projected      {_fmt(w['h_self_before'])}")
    print(f"   conv output    {_fmt(w['conv_output'])}")
    print(f"   WITHOUT graph  {_fmt(w['h_without_graph'])}   |h| = {w['self_norm']:.3f}")
    print(f"   WITH graph     {_fmt(w['h_after'])}")
    print(f"\n   the neighborhood moved this embedding by {w['delta_norm']:.3f}, "
          f"{100 * w['delta_norm'] / max(w['self_norm'], 1e-9):.0f}% of its length without the graph")


# ----------------------------------------------------------------- U6: ranking demo

CASCADE = """
MATCH path = (up:Flight)-[:PRECEDES*1..3]->(f:Flight {flight_id: $flight_id})
RETURN [n IN nodes(path) | n.flight_id] AS chain,
       [n IN nodes(path) | n.outcome_arr_delay_min] AS delays
ORDER BY length(path) DESC LIMIT 1
"""


def ranking_demo(model: Ranker, ds: Dataset, origin: str | None = None,
                 slack: float = 15.0, k: int = 5) -> dict:
    """One query, ranked by the shipped rule and by the GNN, with outcomes beside each.

    The query is "rank what leaves <origin> in the test window by whether it will arrive
    within <slack> minutes of schedule". Relevance is observed: a candidate is right if
    it actually did.
    """
    test = np.where(ds.test_mask)[0]
    by_origin = {}
    for i in test:
        by_origin.setdefault(ds.origin[i], []).append(int(i))
    if origin is None:
        origin = max(by_origin, key=lambda o: len(by_origin[o]))
    cands = by_origin[origin]

    gnn = score_all(model, ds, slack)
    # the rule that ships today, with G0 breaking ties exactly as the ablation does
    late_inbound = (ds.up_observed == 1.0) & (ds.up_delay_raw >= 45.0)
    rule = -late_inbound.astype(np.float32) - 0.01 * ds.service_pct_late15 / 100.0

    def order(scores):
        return sorted(cands, key=lambda i: -scores[i])

    def rows(ranked, scores):
        return [{
            "flight_id": ds.flight_ids[i],
            "route": f"{ds.origin[i]}-{ds.dest[i]}",
            "score": float(scores[i]),
            "inbound_late": bool(late_inbound[i]),
            "arrived": float(ds.delay_min[i]),
            "in_time": bool(ds.delay_min[i] <= slack),
        } for i in ranked[:k]]

    gold = {i for i in cands if ds.delay_min[i] <= slack}
    r_gnn, r_rule = order(gnn), order(rule)

    def hits(ranked):
        return len(set(ranked[:k]) & gold)

    # How many *distinct* scores each method produced over the candidate set. A method
    # that emits one value for most candidates is not ranking them -- the order inside
    # that block comes from the tie-break, not from the method. This is the difference
    # between the two approaches that a top-k table alone hides.
    def distinct(scores):
        return len({round(float(scores[i]), 6) for i in cands})

    return {
        "origin": origin, "slack": slack, "k": k,
        "n_candidates": len(cands), "n_relevant": len(gold),
        "gnn": rows(r_gnn, gnn), "rule": rows(r_rule, rule),
        "gnn_hits": hits(r_gnn), "rule_hits": hits(r_rule),
        "gnn_distinct": distinct(gnn), "rule_distinct": distinct(rule),
        # How often the rotation rule's actual condition fires. When it is near zero the
        # rule is not using rotation at all -- it has degenerated into its tie-break,
        # which is the G0 baseline rate. A top-k table alone would hide that.
        "rule_condition_fires": int(sum(1 for i in cands if late_inbound[i])),
    }


def print_ranking_demo(d: dict) -> None:
    print(f'QUESTION  "Which flights out of {d["origin"]} this evening will arrive '
          f'within {d["slack"]:.0f} minutes of schedule?"')
    print(f"          {d['n_candidates']} candidates in the test window, "
          f"{d['n_relevant']} actually did\n")
    for label, key, got, dis in (("GNN ranking (G3)", "gnn", d["gnn_hits"], d["gnn_distinct"]),
                                 ("Shipped rule (G2a)", "rule", d["rule_hits"], d["rule_distinct"])):
        print(f"{label}  --  {got} of top {d['k']} were right  |  "
              f"{dis} distinct scores across {d['n_candidates']} candidates")
        print(f"   {'flight':<30}{'route':<10}{'score':>9}{'inbound late':>14}{'arrived':>9}")
        for r in d[key]:
            mark = "ok " if r["in_time"] else "   "
            print(f" {mark}{r['flight_id']:<30}{r['route']:<10}{r['score']:>9.5f}"
                  f"{str(r['inbound_late']):>14}{r['arrived']:>+8.0f}m")
        print()
    fires = d["rule_condition_fires"]
    print(f"NOTE  the rotation rule's condition -- inbound aircraft 45+ min late -- fires on "
          f"{fires} of {d['n_candidates']} candidates.")
    if fires <= 2:
        print("      With it silent, the rule is ranking purely by its tie-break, the G0")
        print("      baseline rate. Its score here is not evidence that rotation ranks well.")


def cited_path(flight_id: str) -> dict | None:
    """The evidence behind the top-ranked flight: its rotation chain, from the graph.

    This is the division the lecture insists on and the project already enforces -- the
    GNN score decides where to look, and the cited path is what the answer rests on. The
    score is Derived; the chain is Observed.
    """
    rows = kg.run(CASCADE, flight_id=flight_id)
    return dict(rows[0]) if rows else None
