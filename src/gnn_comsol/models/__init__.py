"""
Model registry.

Standard graph models and the BSMS pressure model are exposed through
one factory.

The BSMS model is mesh-independent: the multiscale hierarchy is supplied
at forward time, so the same model can operate on different geometries.
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
    unet_depth=None,
    hidden_layers=None,
    pos_dim=2,
):
    """
    Instantiate a network by the name used in the config files.

    For BSMS, the mesh hierarchy is NOT stored inside the model.
    edge_indices, pool_indices and pos are supplied at forward time.
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

        if unet_depth is None:
            raise ValueError(
                "BSMS requires unet_depth."
            )

        if hidden_layers is None:
            raise ValueError(
                "BSMS requires hidden_layers."
            )

        return PressureBSMSGNN(
            num_in=num_in,
            latent_dim=num_neurons,
            unet_depth=unet_depth,
            hidden_layers=hidden_layers,
            pos_dim=pos_dim,
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