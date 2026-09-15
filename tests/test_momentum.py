"""
The momentum residual, against fields whose residual is known.

A residual is not something a run can sanity-check for you: it is a
number that is supposed to be small, and a wrong discretization also
produces numbers that are small. So it is checked here against flows
that satisfy the Navier-Stokes equations EXACTLY on this mesh, where
the answer is not "small" but zero to round-off.

The two that do this are:

- a linear velocity field. Its convection term is linear in x and y,
  so the pressure that balances it is quadratic; WLSQ reconstructs the
  gradient of a linear field exactly, the arithmetic mean of the three
  vertex gradients of a linear gradient field is exactly its value at
  the centroid, and the Laplacian of a linear field vanishes term by
  term. Everything cancels, exactly;

- a uniform field changing in time. Every spatial term vanishes and the
  residual is the time derivative alone, whose value is known.
"""

import numpy as np
import pytest
import torch

from gnn_comsol.physics import (
    FluidProperties,
    build_cell_geometry,
    build_wlsq_operators,
    continuity_residual,
    divergence_over_faces,
    find_boundary_nodes,
    interpolate_to_cell,
    momentum_residual,
    wlsq_gradient,
)

from conftest import grid_cells, grid_mesh, wlsq_operators


FLUID = FluidProperties(rho=1.3, mu=0.017, source="test")


@pytest.fixture
def mesh():
    """A mesh, its operators and its control volumes, in float64."""

    rows, cols = 7, 8

    edge_index, pos = grid_mesh(rows, cols, spacing=0.25)

    neighbors, G_wlsq = wlsq_operators(edge_index, pos)

    cell_index = grid_cells(rows, cols)

    geometry = build_cell_geometry(
        pos,
        cell_index,
        dtype=torch.float64,
    )

    operators = build_wlsq_operators(
        neighbors,
        G_wlsq,
        num_nodes=pos.shape[0],
        dtype=torch.float64,
    )

    return {
        "pos": torch.as_tensor(pos, dtype=torch.float64),
        "operators": operators,
        "geometry": geometry,
        "interior": ~geometry.boundary_node,
        "rows": rows,
        "cols": cols,
    }


def interior_cells(mesh):
    """
    Cells all of whose vertices are interior nodes.

    The WLSQ stencil of a boundary node is one-sided, so its gradient
    is only first-order accurate and the exact cancellations below do
    not hold there. That is a property of the method, not a defect of
    this implementation, and the paper handles it by imposing the
    boundary values rather than by trusting the residual there.
    """

    geometry = mesh["geometry"]

    return mesh["interior"][geometry.cell_index].all(dim=1)


# =====================================================================
# The control volumes
# =====================================================================

def test_cell_areas_sum_to_the_domain(mesh):

    area = mesh["geometry"].area.sum()

    width = (mesh["cols"] - 1) * 0.25
    height = (mesh["rows"] - 1) * 0.25

    assert area == pytest.approx(width * height)


def test_face_normals_of_a_cell_sum_to_zero(mesh):
    """
    A closed surface has no net normal. Every flux that follows relies
    on it, and it is exact for straight faces, so it is checked as an
    equality rather than as an approximation.
    """

    total = mesh["geometry"].face_normal.sum(dim=1)

    assert total.abs().max() < 1e-12


def test_face_normals_point_out_of_their_cell(mesh):
    """
    A normal flipped inwards would turn the viscous term into a source
    instead of a sink, and nothing about the shape of the result would
    look wrong.
    """

    geometry = mesh["geometry"]

    corners = mesh["pos"][geometry.cell_index]

    centroid = corners.mean(dim=1)

    for face, (start, end) in enumerate(((0, 1), (1, 2), (2, 0))):

        midpoint = 0.5 * (corners[:, start] + corners[:, end])

        outward = midpoint - centroid

        projection = (
            outward * geometry.face_normal[:, face]
        ).sum(dim=1)

        assert projection.min() > 0


def test_boundary_nodes_are_the_edge_of_the_grid(mesh):
    """
    Found from the connectivity alone - an edge belonging to one cell -
    and checked against the geometry it is supposed to describe.
    """

    pos = mesh["pos"]

    on_edge = (
        (pos[:, 0] == pos[:, 0].min())
        | (pos[:, 0] == pos[:, 0].max())
        | (pos[:, 1] == pos[:, 1].min())
        | (pos[:, 1] == pos[:, 1].max())
    )

    assert torch.equal(mesh["geometry"].boundary_node, on_edge)


def test_a_mesh_with_a_flat_cell_is_refused():

    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    with pytest.raises(ValueError, match="zero area"):
        build_cell_geometry(pos, np.array([[0, 1, 2]]))


def test_boundary_of_a_single_triangle_is_all_of_it():

    boundary = find_boundary_nodes(np.array([[0, 1, 2]]), 3)

    assert boundary.all()


# =====================================================================
# The pieces the residual is built from
# =====================================================================

def test_interpolation_to_the_cell_is_exact_for_a_linear_field(mesh):
    """
    Eq. (A.3) is second order, so a linear field must come out exactly
    at the centroid - which the plain mean of the three vertices would
    also manage. The point of the test is that the gradient correction
    does not BREAK that.
    """

    pos = mesh["pos"]

    field = (2.0 * pos[:, 0] - 0.5 * pos[:, 1] + 1.0).unsqueeze(1)

    gradient = wlsq_gradient(field, mesh["operators"])

    at_cell = interpolate_to_cell(field, gradient, mesh["geometry"])

    centroid = pos[mesh["geometry"].cell_index].mean(dim=1)

    expected = 2.0 * centroid[:, 0] - 0.5 * centroid[:, 1] + 1.0

    assert torch.allclose(at_cell[:, 0], expected, atol=1e-10)


def test_face_divergence_of_a_linear_field_vanishes(mesh):
    """
    The Laplacian of a linear field is zero, and the face sum has to
    reproduce that exactly: the gradient is constant, so the sum
    collapses onto the closed-surface identity above.
    """

    pos = mesh["pos"]

    field = (3.0 * pos[:, 0] + 7.0 * pos[:, 1]).unsqueeze(1)

    gradient = wlsq_gradient(field, mesh["operators"])

    flux = divergence_over_faces(gradient, mesh["geometry"])

    assert flux.abs().max() < 1e-12


def test_face_divergence_recovers_a_known_laplacian(mesh):
    """
    For phi = x^2 + y^2 the Laplacian is 4, so the flux through the
    faces must be 4 * area.
    """

    pos = mesh["pos"]

    field = (pos[:, 0] ** 2 + pos[:, 1] ** 2).unsqueeze(1)

    gradient = wlsq_gradient(field, mesh["operators"])

    flux = divergence_over_faces(gradient, mesh["geometry"])

    expected = 4.0 * mesh["geometry"].area

    inside = interior_cells(mesh)

    assert torch.allclose(
        flux[inside, 0],
        expected[inside],
        rtol=1e-9,
    )


# =====================================================================
# The residuals
# =====================================================================

def test_continuity_residual_is_area_weighted(mesh):
    """u = x, v = y has div(u) = 2, so the residual is 2 * area."""

    pos = mesh["pos"]

    residual = continuity_residual(
        pos[:, 0],
        pos[:, 1],
        mesh["operators"],
        mesh["geometry"],
    )

    assert torch.allclose(
        residual,
        2.0 * mesh["geometry"].area,
        rtol=1e-9,
    )


def test_momentum_residual_vanishes_on_an_exact_solution(mesh):
    """
    A steady, divergence-free, linear velocity field with the pressure
    that balances its own convection:

        u = a x + b y        v = c x - a y
        (u.grad)u = (a^2 + bc) (x, y)
        p = -rho (a^2 + bc) (x^2 + y^2) / 2

    The viscous term vanishes because the field is linear, the time
    term because nothing changes, and the convection term is cancelled
    by the pressure gradient. Every term is individually non-zero, so
    this is a cancellation and not three zeros added up.
    """

    pos = mesh["pos"]

    x, y = pos[:, 0], pos[:, 1]

    a, b, c = 0.7, -0.4, 1.1

    velocity = torch.stack((a * x + b * y, c * x - a * y), dim=1)

    pressure = (
        -FLUID.rho * (a ** 2 + b * c) * (x ** 2 + y ** 2) / 2.0
    )

    residual = momentum_residual(
        velocity,
        velocity,
        pressure,
        0.05,
        FLUID,
        mesh["operators"],
        mesh["geometry"],
    )

    inside = interior_cells(mesh)

    assert inside.sum() > 10, "the test would be vacuous"

    # Against the size of the terms that cancel, not against zero in
    # the abstract: a residual is only ever small compared to something.
    scale = (
        FLUID.rho
        * abs(a ** 2 + b * c)
        * x.abs().max()
        * mesh["geometry"].area.max()
    )

    assert residual[inside].abs().max() < 1e-10 * scale


def test_momentum_residual_recovers_a_known_acceleration(mesh):
    """
    A uniform field that changes in time: every spatial derivative is
    zero, so the residual is rho * du/dt * area and nothing else.
    """

    geometry = mesh["geometry"]

    num_nodes = mesh["pos"].shape[0]

    def uniform(value):

        return torch.full(
            (num_nodes, 2),
            value,
            dtype=torch.float64,
        )

    delta_t = 0.02

    residual = momentum_residual(
        uniform(1.0),
        uniform(1.5),
        torch.zeros(num_nodes, dtype=torch.float64),
        delta_t,
        FLUID,
        mesh["operators"],
        geometry,
    )

    expected = FLUID.rho * (0.5 / delta_t) * geometry.area

    assert torch.allclose(residual[:, 0], expected, rtol=1e-9)
    assert torch.allclose(residual[:, 1], expected, rtol=1e-9)


def test_momentum_residual_sees_the_pressure_gradient(mesh):
    """
    At rest with a linear pressure field, the residual is the pressure
    gradient alone - the term that couples the two networks, and the
    one a sign error would hide in.
    """

    pos = mesh["pos"]

    num_nodes = pos.shape[0]

    at_rest = torch.zeros((num_nodes, 2), dtype=torch.float64)

    pressure = 5.0 * pos[:, 0] - 2.0 * pos[:, 1]

    residual = momentum_residual(
        at_rest,
        at_rest,
        pressure,
        0.1,
        FLUID,
        mesh["operators"],
        mesh["geometry"],
    )

    area = mesh["geometry"].area

    assert torch.allclose(residual[:, 0], 5.0 * area, rtol=1e-9)
    assert torch.allclose(residual[:, 1], -2.0 * area, rtol=1e-9)


def test_momentum_residual_backpropagates(mesh):
    """
    The residual couples both networks, so it must carry a gradient to
    the velocity AND to the pressure.
    """

    num_nodes = mesh["pos"].shape[0]

    rng = np.random.default_rng(3)

    def field(shape):

        return torch.as_tensor(
            rng.normal(size=shape),
            dtype=torch.float64,
        ).requires_grad_(True)

    velocity_next = field((num_nodes, 2))
    pressure_next = field(num_nodes)

    residual = momentum_residual(
        field((num_nodes, 2)).detach(),
        velocity_next,
        pressure_next,
        0.01,
        FLUID,
        mesh["operators"],
        mesh["geometry"],
    )

    residual.square().mean().backward()

    for gradient in (velocity_next.grad, pressure_next.grad):

        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0
