"""
Model registry.

Standard graph models and the BSMS pressure model are exposed through
one factory. BSMS additionally requires the fixed multiscale hierarchy
of the mesh.
"""

from .gcn import GCNNet
from .virtual_node import GCNVirtualNodeNet
from .bsms_pressure import PressureBSMSGNN


ARCHITECTURES = {
    "gcn": GCNNet,
    "gcn_virtual_node": GCNVirtualNodeNet,
    "bsms": PressureBSMSGNN,
}


def build_model(
    architecture,
    num_in,
    num_out,
    num_neurons,
    num_layers,
    dropout,
    *,
    edge_indices=None,
    pool_indices=None,
    pos=None,
    unet_depth=None,
    hidden_layers=None,
):
    """
    Instantiate a network by the name used in the config files.

    BSMS additionally requires the fixed multiscale mesh hierarchy.
    """

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}."
        )

    if architecture == "bsms":

        if num_out != 1:
            raise ValueError(
                "The BSMS model is currently implemented only for "
                "pressure prediction (num_out=1)."
            )

        if edge_indices is None:
            raise ValueError("BSMS requires edge_indices.")

        if pool_indices is None:
            raise ValueError("BSMS requires pool_indices.")

        if pos is None:
            raise ValueError("BSMS requires node positions.")

        if unet_depth is None:
            raise ValueError("BSMS requires unet_depth.")

        if hidden_layers is None:
            raise ValueError("BSMS requires hidden_layers.")

        return PressureBSMSGNN(
            num_in=num_in,
            latent_dim=num_neurons,
            unet_depth=unet_depth,
            hidden_layers=hidden_layers,
            edge_indices=edge_indices,
            pool_indices=pool_indices,
            pos=pos,
            pos_dim=pos.shape[-1],
        )

    return ARCHITECTURES[architecture](
        num_in=num_in,
        num_out=num_out,
        num_neurons=num_neurons,
        num_layers=num_layers,
        dropout=dropout,
    )


__all__ = [
    "ARCHITECTURES",
    "GCNNet",
    "GCNVirtualNodeNet",
    "PressureBSMSGNN",
    "build_model",
]