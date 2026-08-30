"""
Model registry.

`ARCHITECTURES` lists what an experiment configuration may ask for. The
multiscale model is deliberately absent: its forward() needs a cluster
vector and a coarse edge_index on top of the graph, so it cannot be
driven by the standard training loop yet. Import it directly from
gnn_comsol.models.multiscale to work on it.
"""

from .gcn import GCNNet
from .virtual_node import GCNVirtualNodeNet

ARCHITECTURES = {
    "gcn": GCNNet,
    "gcn_virtual_node": GCNVirtualNodeNet
}


def build_model(
    architecture,
    num_in,
    num_out,
    num_neurons,
    num_layers,
    dropout
):
    """Instantiate a network by the name used in the config files."""

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}. "
            "('multiscale' exists in models/multiscale.py but is not "
            "yet wired into the training loop.)"
        )

    return ARCHITECTURES[architecture](
        num_in=num_in,
        num_out=num_out,
        num_neurons=num_neurons,
        num_layers=num_layers,
        dropout=dropout
    )


__all__ = [
    "ARCHITECTURES",
    "GCNNet",
    "GCNVirtualNodeNet",
    "build_model"
]
