import torch

from .wlsq import (
    wlsq_gradient,
    node_gradient_to_cell,
)


def continuity_residual(
    u,
    v,
    neighbors,
    G_wlsq,
    cell_index,
):
    """
    Compute the incompressible continuity residual

        R = du/dx + dv/dy

    at triangular cell centers.
    """

    # Nodal gradients
    grad_u_node = wlsq_gradient(
        u,
        neighbors,
        G_wlsq,
    )

    grad_v_node = wlsq_gradient(
        v,
        neighbors,
        G_wlsq,
    )

    # Vertex -> cell-center
    grad_u_cell = node_gradient_to_cell(
        grad_u_node,
        cell_index,
    )

    grad_v_cell = node_gradient_to_cell(
        grad_v_node,
        cell_index,
    )

    # Continuity:
    #
    # div(u) = du/dx + dv/dy
    residual = (
        grad_u_cell[:, 0]
        + grad_v_cell[:, 1]
    )

    return residual


def continuity_loss(
    u,
    v,
    neighbors,
    G_wlsq,
    cell_index,
):
    """
    Mean squared incompressibility residual.
    """

    residual = continuity_residual(
        u,
        v,
        neighbors,
        G_wlsq,
        cell_index,
    )

    return torch.mean(residual.square())