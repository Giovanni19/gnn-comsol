"""
Check the WLSQ gradient on a real dataset, against MATLAB and against
the one field whose gradient is known exactly.

Run on the machine that has the .mat files:

    python scripts/test_wlsq_gradient.py
"""

import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics import (
    build_wlsq_operators,
    wlsq_gradient,
    wlsq_gradient_reference,
)


dataset_path = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_manual_stabilization_200R.mat"
)


# ============================================================
# LOAD DATASET
# ============================================================

data = load_data(
    dataset_path,
    skip_initial=0,
)


# ============================================================
# BUILD THE SPARSE WLSQ OPERATORS
# ============================================================
#
# float64 here: this script is checking the discretization, not
# imitating the training run, and a float32 operator would put its own
# rounding error on top of the one being measured.
# ============================================================

operators = build_wlsq_operators(
    data.neighbors,
    data.G_wlsq,
    num_nodes=data.num_nodes,
    dtype=torch.float64,
)


# ============================================================
# SELECT SAME TIMESTEP USED IN MATLAB
# ============================================================

# MATLAB:
#
#     test_step = 100
#
# MATLAB is one-based.
#
# Python snapshot index:
#
#     99

test_step = 99

u = torch.as_tensor(
    data.X_input[test_step, :, 0],
    dtype=torch.float64,
)


# ============================================================
# WLSQ GRADIENT
# ============================================================

grad_u = wlsq_gradient(u, operators)

du_dx_wlsq = grad_u[:, 0]
du_dy_wlsq = grad_u[:, 1]


print("\n========================================")
print("PYTORCH WLSQ GRADIENT")
print("========================================")

print("grad_u shape:", grad_u.shape)

print("\ndu/dx:")
print("min :", du_dx_wlsq.min().item())
print("max :", du_dx_wlsq.max().item())

print("\ndu/dy:")
print("min :", du_dy_wlsq.min().item())
print("max :", du_dy_wlsq.max().item())


# ============================================================
# SPARSE OPERATOR vs THE PER-NODE DEFINITION
# ============================================================
#
# The sparse operator is an algebraic rearrangement of the per-node
# formula, so the two must agree to round-off. If they do not, the
# assembly is wrong, and every number above is wrong with it.
# ============================================================

neighbors = [
    torch.as_tensor(neigh, dtype=torch.long)
    for neigh in data.neighbors
]

G_wlsq = [
    torch.as_tensor(G_i, dtype=torch.float64)
    for G_i in data.G_wlsq
]

grad_u_reference = wlsq_gradient_reference(
    u,
    neighbors,
    G_wlsq,
)

print("\n========================================")
print("SPARSE OPERATOR vs PER-NODE REFERENCE")
print("========================================")

print(
    "max abs difference:",
    (grad_u - grad_u_reference).abs().max().item(),
)


# ============================================================
# EXACT LINEAR FIELD TEST
# ============================================================
#
# WLSQ reconstructs a linear field exactly, whatever the mesh, so the
# error here is round-off and nothing else.
# ============================================================

pos = torch.as_tensor(
    data.pos,
    dtype=torch.float64,
)

x = pos[:, 0]
y = pos[:, 1]

phi = 2.0 * x + 3.0 * y + 5.0

grad_phi = wlsq_gradient(phi, operators)

error_x = torch.abs(grad_phi[:, 0] - 2.0)
error_y = torch.abs(grad_phi[:, 1] - 3.0)

print("\n========================================")
print("PYTORCH LINEAR FIELD TEST")
print("========================================")

print("Expected gradient: [2, 3]")

print(
    "max error dphi/dx:",
    error_x.max().item(),
)

print(
    "max error dphi/dy:",
    error_y.max().item(),
)
