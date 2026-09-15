"""
The acceptance test for the momentum residual, on real COMSOL data.

RUN THIS BEFORE TURNING momentum_weight UP. Everything the residual is
made of is checked by the unit tests, on fields whose residual is known
exactly. What no unit test can tell you is whether THIS discretization,
on THIS mesh, at THIS timestep, is a good description of what COMSOL
actually computed - and if it is not, the physics loss pushes the
network away from the data rather than towards physics.

The number to look at is the residual of the GROUND TRUTH. COMSOL
integrates these snapshots with its own BDF scheme, not with the IMEX
scheme of the residual, and reconstructs gradients with finite elements,
not with WLSQ. So the ground-truth residual is not zero: it is the floor
that a prediction can be pushed down to, and it should be SMALL COMPARED
WITH THE TERMS THAT MAKE IT UP. A residual as large as its own convection
term means the equation is not being satisfied by the data, which means
it is the discretization that is wrong, not the data.

    python scripts/test_momentum_residual.py
"""

import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics import (
    FluidProperties,
    build_cell_geometry,
    build_wlsq_operators,
    continuity_residual,
    momentum_residual,
    momentum_terms,
)


dataset_path = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_manual_stabilization_200R.mat"
)

# Used only if the .mat does not carry rho and mu, i.e. if the COMSOL
# export in first_database.m did not resolve spf.rho / spf.mu.
FALLBACK_FLUID = FluidProperties(rho=1.0, mu=0.005, source="this script")

# Which transition to look at. Python index, MATLAB index minus one.
TRANSITION = 99


data = load_data(dataset_path, skip_initial=0)

fluid = FluidProperties.from_simulation(data) or FALLBACK_FLUID

operators = build_wlsq_operators(
    data.neighbors,
    data.G_wlsq,
    num_nodes=data.num_nodes,
    dtype=torch.float64,
)

geometry = build_cell_geometry(
    data.pos,
    data.cell_index,
    dtype=torch.float64,
)


def tensor(array):
    return torch.as_tensor(array, dtype=torch.float64)


velocity_now = tensor(data.X_input[TRANSITION, :, 0:2])
velocity_next = tensor(data.Y_target[TRANSITION, :, 0:2])
pressure_next = tensor(data.Y_target[TRANSITION, :, 2])

delta_t = float(data.delta_t[TRANSITION])

interior = ~geometry.boundary_node
interior_cells = interior[geometry.cell_index].all(dim=1)


print("\n========================================")
print("SETUP")
print("========================================")

print("nodes          :", data.num_nodes)
print("cells          :", geometry.num_cells)
print("interior cells :", int(interior_cells.sum().item()))
print("dt             :", delta_t)
print("fluid          :", fluid)


# ============================================================
# CROSS-CHECK THE FLUID AGAINST THE REYNOLDS NUMBER
# ============================================================
#
# The dataset file names carry a Reynolds number. If mu and rho are
# what they claim to be, and the reference scales are the obvious ones,
# the two must agree. A factor of five here is worth more than any
# amount of staring at the residual.
# ============================================================

pos = tensor(data.pos)

height = (pos[:, 1].max() - pos[:, 1].min()).item()

speed = tensor(data.X_input[TRANSITION, :, 0:2]).norm(dim=1).max().item()

print("\n========================================")
print("REYNOLDS CROSS-CHECK")
print("========================================")

print(f"channel height (from pos)  : {height:.6g}")
print(f"peak speed  (from the data): {speed:.6g}")
print(
    "implied Re = U H / nu       : "
    f"{fluid.reynolds(speed, height):.6g}"
)
print("compare against the number in the file name.")


# ============================================================
# CONTINUITY
# ============================================================

continuity = continuity_residual(
    velocity_next[:, 0],
    velocity_next[:, 1],
    operators,
    geometry,
)

print("\n========================================")
print("CONTINUITY RESIDUAL OF THE GROUND TRUTH")
print("========================================")

print("abs mean (all cells)     :", continuity.abs().mean().item())
print(
    "abs mean (interior only) :",
    continuity[interior_cells].abs().mean().item(),
)


# ============================================================
# MOMENTUM, TERM BY TERM
# ============================================================

terms = momentum_terms(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
)

residual = momentum_residual(
    velocity_now,
    velocity_next,
    pressure_next,
    delta_t,
    fluid,
    operators,
    geometry,
)

print("\n========================================")
print("MOMENTUM RESIDUAL OF THE GROUND TRUTH")
print("========================================")

print("\nSize of each term (abs mean over interior cells):\n")

largest = 0.0

for name, value in terms.items():

    magnitude = value[interior_cells].abs().mean().item()

    largest = max(largest, magnitude)

    print(f"  {name:12s} {magnitude:.6e}")

interior_residual = residual[interior_cells].abs().mean().item()

print(f"\n  {'RESIDUAL':12s} {interior_residual:.6e}")

print(
    f"\nresidual / largest term = "
    f"{interior_residual / largest:.4%}"
)

print(
    "\nA few percent means the discretization agrees with COMSOL and\n"
    "the residual is usable as a training signal. Of the same order\n"
    "as the largest term means it is not: check rho and mu first,\n"
    "then whether dt is really the gap between these two snapshots."
)

print("\nBoundary cells, for comparison:")

print(
    "  residual abs mean (all cells) :",
    residual.abs().mean().item(),
)

print(
    "  ratio boundary / interior     :",
    (
        residual[~interior_cells].abs().mean()
        / residual[interior_cells].abs().mean()
    ).item(),
)

print(
    "\nThe boundary cells are expected to be worse: their WLSQ\n"
    "stencils are one-sided. enforce_boundary_values handles them\n"
    "during training by imposing the true velocity there."
)
