"""
Residuals of the incompressible Navier-Stokes equations on the mesh.

The discretization follows the Gen-FVGN paper (Li et al., 2024): the
variables live at the mesh vertices, the WLSQ gradient is reconstructed
there, and the equations are written on the cell-centered control
volumes, where the vertex quantities are interpolated with Eq. (A.3).

    rho (du/dt + (u.grad) u) = -grad p + mu laplacian(u)
    div(u) = 0

Both residuals are volume-integrated over a cell, as in Eq. (16) and
Eq. (17) of the paper, so they carry a factor of the cell area. That is
not cosmetic: it is what makes a small cell count less than a large one
in the sum, instead of every cell counting the same whatever its size.

The whole thing needs exactly ONE gradient reconstruction pass. The
convection term uses the gradient at the cell, and the viscous term
takes the divergence of that same gradient over the faces rather than
reconstructing it a second time.
"""

import torch

from .mesh import divergence_over_faces, interpolate_to_cell
from .wlsq import node_gradient_to_cell, wlsq_gradient


def continuity_residual(u, v, operators, geometry):
    """
    The incompressible continuity residual, integrated over each cell:

        R = (du/dx + dv/dy) * area

    Parameters
    ----------
    u, v : torch.Tensor, shape (N,)
        Velocity components at the mesh nodes, in physical units.

    operators : WLSQOperators
        Gradient operators of this mesh, see physics.wlsq.

    geometry : CellGeometry
        Control volumes of this mesh, see physics.mesh.

    Returns
    -------
    residual : torch.Tensor, shape (Nc,)
    """

    # u and v are reconstructed in a single pair of sparse products:
    # gradient[:, 0] is grad(u) and gradient[:, 1] is grad(v).
    velocity = torch.stack((u, v), dim=1)

    gradient_cell = node_gradient_to_cell(
        wlsq_gradient(velocity, operators),
        geometry.cell_index,
    )

    divergence = (
        gradient_cell[:, 0, 0]
        + gradient_cell[:, 1, 1]
    )

    return divergence * geometry.area


def continuity_loss(u, v, operators, geometry):
    """Mean squared incompressibility residual."""

    residual = continuity_residual(u, v, operators, geometry)

    return torch.mean(residual.square())


def momentum_terms(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
):
    """
    The four terms of the momentum residual, separately.

    Each one carries the sign it has in the residual and is already
    integrated over the cell, so they sum exactly to
    `momentum_residual`. They are exposed because a residual is only
    ever small compared with the terms that make it up: knowing that
    it is 1e-3 says nothing until you know whether the convection term
    is 1e-3 or 1e+3.

    Returns
    -------
    dict of str -> torch.Tensor, each (Nc, 2)
        "time", "convection", "pressure", "viscous".
    """

    # One reconstruction for all five fields at once: the gradients of
    # u and v at both times, and of the pressure.
    fields = torch.cat(
        (
            velocity_now,
            velocity_next,
            pressure_next.unsqueeze(1),
        ),
        dim=1,
    )

    gradients = wlsq_gradient(fields, operators)

    values_cell = interpolate_to_cell(fields, gradients, geometry)

    # The gradient operator is linear, so the gradient of the midpoint
    # is the midpoint of the gradients - no second reconstruction.
    gradient_mid = 0.5 * (gradients[:, 0:2] + gradients[:, 2:4])

    velocity_now_cell = values_cell[:, 0:2]
    velocity_next_cell = values_cell[:, 2:4]

    velocity_mid_cell = 0.5 * (
        velocity_now_cell + velocity_next_cell
    )

    gradient_mid_cell = node_gradient_to_cell(
        gradient_mid,
        geometry.cell_index,
    )

    area = geometry.area.unsqueeze(1)

    return {
        # rho du/dt
        "time": (
            fluid.rho
            * (velocity_next_cell - velocity_now_cell)
            / delta_t
            * area
        ),

        # rho (u . grad) u, component by component:
        #     u du/dx + v du/dy
        #     u dv/dx + v dv/dy
        "convection": (
            fluid.rho
            * (
                gradient_mid_cell
                * velocity_mid_cell.unsqueeze(1)
            ).sum(dim=-1)
            * area
        ),

        # grad p
        "pressure": (
            node_gradient_to_cell(
                gradients[:, 4],
                geometry.cell_index,
            )
            * area
        ),

        # -mu laplacian(u), as a flux through the faces
        "viscous": (
            -fluid.mu
            * divergence_over_faces(gradient_mid, geometry)
        ),
    }


def momentum_residual(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
):
    """
    The momentum residual of one transition, integrated over each cell.

        R = rho (u^{n+1}_c - u^n_c) / dt * area
          + rho (u'_c . grad) u'_c * area
          + grad p^{n+1}_c * area
          - mu sum_f (grad u'_f . n) dl

    Parameters
    ----------
    velocity_now, velocity_next : torch.Tensor, shape (N, 2)
        Velocity at the nodes at t and t + dt, in physical units.

    pressure_next : torch.Tensor, shape (N,)
        Pressure at t + dt, in physical units.

    delta_t : float or 0-d torch.Tensor
        The PHYSICAL duration of the transition, in seconds. The
        normalized dt the network is fed as a feature is a different
        number and would silently scale the whole residual.

    fluid : FluidProperties

    operators : WLSQOperators
    geometry : CellGeometry

    Returns
    -------
    residual : torch.Tensor, shape (Nc, 2)
        Column 0 is the x-momentum residual, column 1 the y-momentum.

    Notes
    -----
    The convection and viscous terms are evaluated at the IMEX midpoint
    u' = (u^n + u^{n+1}) / 2, as in Eq. (12) of the paper: fully
    explicit is CFL-limited and fully implicit is dissipative, and the
    midpoint is the compromise the paper settles on.

    COMSOL integrates these snapshots with its own BDF scheme, not with
    this one, so even an exact prediction leaves a residual of the order
    of the difference between the two time discretizations. That
    difference is a property of the data, not of the network: measure it
    on the ground truth with scripts/test_momentum_residual.py before
    choosing a weight.
    """

    terms = momentum_terms(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    return sum(terms.values())


def momentum_loss(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
):
    """Mean squared momentum residual, over cells and both components."""

    residual = momentum_residual(
        velocity_now,
        velocity_next,
        pressure_next,
        delta_t,
        fluid,
        operators,
        geometry,
    )

    return torch.mean(residual.square())
