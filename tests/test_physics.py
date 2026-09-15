"""
The physics-informed loss: gradient reconstruction and the continuity
residual.

Two things are checked here that no training run would ever tell you
about. First, that the sparse operator really is the per-node WLSQ
formula rearranged, and not merely something of the right shape that
trains. Second, that a field whose divergence is known analytically
gets that divergence back - on a mesh whose corner nodes have exactly
two neighbours, which is the case where the orientation of G_i cannot
be recovered from its shape.
"""

import numpy as np
import pytest
import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics import (
    PhysicsGeometry,
    build_cell_geometry,
    build_wlsq_operators,
    continuity_loss,
    continuity_residual,
    node_gradient_to_cell,
    wlsq_gradient,
    wlsq_gradient_reference,
)

from conftest import grid_cells, grid_mesh, wlsq_operators


@pytest.fixture
def mesh():
    """A small mesh with its WLSQ operators, straight from numpy."""

    edge_index, pos = grid_mesh(4, 5, spacing=0.5)

    neighbors, G_wlsq = wlsq_operators(edge_index, pos)

    cell_index = grid_cells(4, 5)

    return {
        "pos": pos,
        "neighbors": neighbors,
        "G_wlsq": G_wlsq,
        "cell_index": cell_index,
        "num_nodes": pos.shape[0],
        "cells": build_cell_geometry(
            pos,
            cell_index,
            dtype=torch.float64,
        ),
    }


@pytest.fixture
def operators(mesh):

    return build_wlsq_operators(
        mesh["neighbors"],
        mesh["G_wlsq"],
        num_nodes=mesh["num_nodes"],
        dtype=torch.float64,
    )


def test_corner_nodes_have_two_neighbours(mesh):
    """
    The premise of the orientation test below: if this grid ever stops
    having square G_i, the test that depends on it stops testing it.
    """

    sizes = [neigh.size for neigh in mesh["neighbors"]]

    assert min(sizes) == 2


def test_sparse_operator_matches_per_node_reference(mesh, operators):
    """
    The fast path is an algebraic rearrangement of the definition, so
    it must agree with it to round-off on an arbitrary field.
    """

    rng = np.random.default_rng(0)

    phi = torch.as_tensor(
        rng.normal(size=mesh["num_nodes"]),
        dtype=torch.float64,
    )

    reference = wlsq_gradient_reference(
        phi,
        [torch.as_tensor(n) for n in mesh["neighbors"]],
        [torch.as_tensor(G) for G in mesh["G_wlsq"]],
    )

    assert torch.allclose(
        wlsq_gradient(phi, operators),
        reference,
        atol=1e-12,
    )


def test_linear_field_is_reconstructed_exactly(mesh, operators):
    """
    WLSQ is exact for a linear field on any mesh: that is what makes it
    a gradient reconstruction rather than a smoother.
    """

    pos = torch.as_tensor(mesh["pos"], dtype=torch.float64)

    phi = 2.0 * pos[:, 0] + 3.0 * pos[:, 1] + 5.0

    gradient = wlsq_gradient(phi, operators)

    assert torch.allclose(
        gradient,
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        atol=1e-10,
    )


def test_fields_can_be_reconstructed_together(mesh, operators):
    """(N, 2) in one call == two calls on (N,)."""

    pos = torch.as_tensor(mesh["pos"], dtype=torch.float64)

    u = pos[:, 0] ** 2
    v = pos[:, 0] * pos[:, 1]

    together = wlsq_gradient(torch.stack((u, v), dim=1), operators)

    assert together.shape == (mesh["num_nodes"], 2, 2)

    assert torch.allclose(
        together[:, 0],
        wlsq_gradient(u, operators),
    )

    assert torch.allclose(
        together[:, 1],
        wlsq_gradient(v, operators),
    )


def test_gradient_rejects_a_field_of_the_wrong_size(operators):

    with pytest.raises(ValueError, match="built for"):
        wlsq_gradient(torch.zeros(3, dtype=torch.float64), operators)


def test_build_rejects_a_transposed_operator(mesh):
    """
    A (k, 2) operator where a (2, k) one is expected is exactly the
    mistake the .mat orientation rules exist to prevent, and it must
    not be quietly accepted.
    """

    G_wlsq = list(mesh["G_wlsq"])

    # A node in the interior, so the transpose really changes the shape
    interior = next(
        i for i, G in enumerate(G_wlsq) if G.shape[1] == 4
    )

    G_wlsq[interior] = G_wlsq[interior].T

    with pytest.raises(ValueError, match="expected"):
        build_wlsq_operators(
            mesh["neighbors"],
            G_wlsq,
            num_nodes=mesh["num_nodes"],
        )


# =====================================================================
# The continuity residual
# =====================================================================

def test_divergence_free_field_has_zero_residual(mesh, operators):
    """
    u = x, v = -y has du/dx = 1 and dv/dy = -1 everywhere: both terms
    are non-zero, and the residual is their exact cancellation.
    """

    pos = torch.as_tensor(mesh["pos"], dtype=torch.float64)

    residual = continuity_residual(
        pos[:, 0],
        -pos[:, 1],
        operators,
        mesh["cells"],
    )

    assert residual.shape == (mesh["cell_index"].shape[0],)

    assert residual.abs().max() < 1e-10


def test_known_divergence_is_recovered(mesh, operators):
    """
    u = x, v = y has div(u) = 2 in every cell, and the residual is
    integrated over the control volume, so it is 2 * area.
    """

    pos = torch.as_tensor(mesh["pos"], dtype=torch.float64)

    residual = continuity_residual(
        pos[:, 0],
        pos[:, 1],
        operators,
        mesh["cells"],
    )

    assert torch.allclose(
        residual,
        2.0 * mesh["cells"].area,
        rtol=1e-9,
    )


def test_cell_interpolation_averages_the_three_vertices(mesh):

    cell_index = torch.as_tensor(mesh["cell_index"])

    grad_node = torch.arange(
        mesh["num_nodes"] * 2,
        dtype=torch.float64,
    ).reshape(-1, 2)

    grad_cell = node_gradient_to_cell(grad_node, cell_index)

    expected = grad_node[cell_index[0]].mean(dim=0)

    assert torch.allclose(grad_cell[0], expected)


def test_continuity_loss_backpropagates(mesh, operators):
    """
    The whole point of the sparse formulation: the residual must carry
    a gradient back to the values it was computed from.
    """

    rng = np.random.default_rng(1)

    u = torch.as_tensor(
        rng.normal(size=mesh["num_nodes"]),
        dtype=torch.float64,
    ).requires_grad_(True)

    v = torch.as_tensor(
        rng.normal(size=mesh["num_nodes"]),
        dtype=torch.float64,
    ).requires_grad_(True)

    loss = continuity_loss(
        u,
        v,
        operators,
        mesh["cells"],
    )

    loss.backward()

    for gradient in (u.grad, v.grad):

        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0


# =====================================================================
# Loading, and the geometry container
# =====================================================================

def test_load_data_reads_the_wlsq_export(make_dataset):
    """
    The orientation rules of the .mat, end to end: a linear field
    reconstructed with the operators as they come out of load_data.

    A G_i left transposed survives every shape check and only shows up
    here, as a wrong gradient at the corner nodes.
    """

    written = make_dataset(rows=4, cols=5)

    data = load_data(written["path"], skip_initial=0)

    assert len(data.neighbors) == written["num_nodes"]
    assert len(data.G_wlsq) == written["num_nodes"]
    assert data.cell_index.shape == written["cell_index"].shape

    np.testing.assert_array_equal(
        data.cell_index,
        written["cell_index"],
    )

    for loaded, expected in zip(data.G_wlsq, written["G_wlsq"]):
        np.testing.assert_allclose(loaded, expected)

    operators = build_wlsq_operators(
        data.neighbors,
        data.G_wlsq,
        num_nodes=data.num_nodes,
        dtype=torch.float64,
    )

    pos = torch.as_tensor(data.pos, dtype=torch.float64)

    gradient = wlsq_gradient(
        -1.5 * pos[:, 0] + 4.0 * pos[:, 1],
        operators,
    )

    assert torch.allclose(
        gradient,
        torch.tensor([-1.5, 4.0], dtype=torch.float64),
        atol=1e-10,
    )


def test_load_data_without_wlsq_still_works(make_dataset):
    """The datasets that predate the WLSQ export must still load."""

    written = make_dataset(with_wlsq=False)

    data = load_data(written["path"], skip_initial=0)

    assert data.neighbors is None
    assert data.G_wlsq is None
    assert data.cell_index is None

    assert PhysicsGeometry.from_simulation(data) is None


def test_physics_geometry_caches_its_operators(make_dataset):
    """
    Rebuilding the operators every batch is what made the first version
    of the physics loss unusable, so the cache is part of the contract.
    """

    written = make_dataset(rows=4, cols=5)

    data = load_data(written["path"], skip_initial=0)

    geometry = PhysicsGeometry.from_simulation(data)

    assert geometry.num_nodes == written["num_nodes"]
    assert geometry.num_cells == written["cell_index"].shape[0]

    first = geometry.operators(dtype=torch.float64)
    second = geometry.operators(dtype=torch.float64)

    assert first is second

    assert geometry.cells() is geometry.cells()

    # A different dtype is a different operator, not a cache hit
    assert geometry.operators(dtype=torch.float32) is not first
