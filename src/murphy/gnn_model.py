"""The query-aware ranker, and the zero-layer model it is compared against.

**The query is part of the input, not an afterthought.** A traveler asks "will this get
me there by 18:40", and the same flight is relevant for a late deadline and irrelevant
for an early one. So the model scores a *pair*: `s(flight, slack)` where slack is the
minutes of margin the query allows. Lecture slide 15 writes this as
`s(v, q) = scorer(h_v, q)`.

Set slack to 15 minutes and the score becomes the ordinary lateness prediction, so the
forecasting use is a special case rather than a separate model (decision N1).

**G1 and G3 are one architecture at two depths.** `Ranker(layers=0)` is an MLP on the
flight's own features; `Ranker(layers=1)` adds one round of message passing over the
rotation chain and the airport-hour. Nothing else differs, so the comparison isolates
the structure rather than two model families.

**Depth is capped at one, for correctness and not for speed** (decision N5). With two
layers the airport-hour node would carry its flights' aggregated features -- including
their inbound delays -- and pass that mixture back, handing a flight outcome-derived
information about flights scheduled after it.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from .gnn_features import FLIGHT_FEATURES, HOUR_FEATURES, Dataset

SEED = 7
EMB_DIM = 8
HIDDEN = 32

ROT = ("flight", "precedes", "flight")
HOSTS = ("hour", "hosts", "flight")


class Ranker(torch.nn.Module):
    """Query-aware relevance scorer over the flight graph.

    `layers=0` is the no-structure baseline (G1); `layers=1` is the GNN (G3).
    """

    def __init__(self, n_airports: int, layers: int = 1):
        super().__init__()
        self.layers = layers
        d_in = len(FLIGHT_FEATURES) + 2 * EMB_DIM
        self.origin = torch.nn.Embedding(n_airports, EMB_DIM)
        self.dest = torch.nn.Embedding(n_airports, EMB_DIM)
        self.project = torch.nn.Linear(d_in, HIDDEN)
        # Applied in BOTH depths, so the only architectural difference between G1 and G3
        # stays the convolution itself. Added after the U5 worked example showed the
        # un-normalized conv output swamping the residual: `relu(h + out)` was zeroing
        # whole embeddings, which is the likeliest source of the seed instability in N11.
        self.norm = torch.nn.LayerNorm(HIDDEN)

        if layers:
            # Only `flight` is updated. The hour nodes are read-only by design, which is
            # why this is HeteroConv and not to_hetero -- the latter requires every node
            # type to be updated during message passing.
            self.conv = HeteroConv({
                ROT:   SAGEConv((HIDDEN, HIDDEN), HIDDEN),
                HOSTS: SAGEConv((len(HOUR_FEATURES), HIDDEN), HIDDEN),
            }, aggr="sum")

        # +1 for the query: the slack the traveler's deadline allows.
        self.score = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN + 1, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, 1),
        )

    def embed(self, x_flight, origin_idx, dest_idx, x_hour, edges) -> torch.Tensor:
        h = torch.cat([x_flight, self.origin(origin_idx), self.dest(dest_idx)], dim=1)
        h = self.project(h)
        if self.layers:
            out = self.conv({"flight": h, "hour": x_hour}, edges)
            # Residual: a flight with no inbound leg and no hour edge still keeps its own
            # features instead of collapsing to zero.
            h = h + out["flight"]
        return F.relu(self.norm(h))

    def forward(self, x_flight, origin_idx, dest_idx, x_hour, edges, slack) -> torch.Tensor:
        h = self.embed(x_flight, origin_idx, dest_idx, x_hour, edges)
        return self.score(torch.cat([h, slack.unsqueeze(1)], dim=1)).squeeze(1)


# ---------------------------------------------------------------- tensor construction

def tensors(ds: Dataset) -> dict:
    return {
        "x_flight": torch.from_numpy(ds.x_flight),
        "x_hour": torch.from_numpy(ds.x_hour),
        "origin_idx": torch.from_numpy(ds.origin_idx),
        "dest_idx": torch.from_numpy(ds.dest_idx),
        "edges": {
            ROT: torch.from_numpy(ds.rot_edges),
            HOSTS: torch.from_numpy(ds.hour_edges),
        },
        "delay": torch.from_numpy(ds.delay_min),
    }


# Slack values sampled as queries during training, in minutes. Chosen to span the
# decisions a traveler actually faces: no margin at all, the BTS 15-minute convention,
# a tight connection, and a comfortable one.
SLACKS = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)


def train(ds: Dataset, layers: int = 1, epochs: int = 300, lr: float = 0.01,
          verbose: bool = True) -> tuple[Ranker, list[float]]:
    """Train on sampled (flight, slack) questions from the training window only."""
    torch.manual_seed(SEED)
    t = tensors(ds)
    model = Ranker(len(ds.airports), layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_idx = torch.from_numpy(np.where(ds.train_mask)[0])
    delay = t["delay"]
    history = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        # One slack per step, cycling: the model sees every query type during training
        # rather than learning a single threshold.
        s = SLACKS[epoch % len(SLACKS)]
        slack = torch.full((ds.x_flight.shape[0],), s, dtype=torch.float32)
        logits = model(t["x_flight"], t["origin_idx"], t["dest_idx"],
                       t["x_hour"], t["edges"], slack)
        # Relevant = arrived within the slack the query allowed. Observed, not invented.
        target = (delay <= s).float()
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], target[train_idx])
        loss.backward()
        opt.step()
        history.append(float(loss))
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:>4}  loss {float(loss):.4f}  (slack {s:.0f}m)")

    return model, history


@torch.no_grad()
def score_all(model: Ranker, ds: Dataset, slack_min: float) -> np.ndarray:
    """Relevance of every flight for one query slack."""
    model.eval()
    t = tensors(ds)
    slack = torch.full((ds.x_flight.shape[0],), slack_min, dtype=torch.float32)
    logits = model(t["x_flight"], t["origin_idx"], t["dest_idx"],
                   t["x_hour"], t["edges"], slack)
    return torch.sigmoid(logits).numpy()
