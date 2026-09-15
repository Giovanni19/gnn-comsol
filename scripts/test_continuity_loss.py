import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics.navier_stokes import (
    continuity_residual,
    continuity_loss,
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
# CONVERT WLSQ GEOMETRY TO PYTORCH
# ============================================================

neighbors = [
    torch.as_tensor(
        neigh,
        dtype=torch.long,
    )
    for neigh in data.neighbors
]

G_wlsq = [
    torch.as_tensor(
        G_i,
        dtype=torch.float64,
    )
    for G_i in data.G_wlsq
]

cell_index = torch.as_tensor(
    data.cell_index,
    dtype=torch.long,
)


# ============================================================
# SELECT COMSOL SNAPSHOT
# ============================================================
#
# Same snapshot as previous tests:
#
# MATLAB timestep 100
# Python index 99
#
# u and v are cloned because we want them to behave like
# differentiable network outputs.
# ============================================================

test_step = 99

u = torch.tensor(
    data.X_input[test_step, :, 0],
    dtype=torch.float64,
    requires_grad=True,
)

v = torch.tensor(
    data.X_input[test_step, :, 1],
    dtype=torch.float64,
    requires_grad=True,
)


# ============================================================
# CONTINUITY RESIDUAL
# ============================================================

residual = continuity_residual(
    u,
    v,
    neighbors,
    G_wlsq,
    cell_index,
)


print("\n========================================")
print("CONTINUITY RESIDUAL TEST")
print("========================================")

print("Residual shape:", residual.shape)

print("\nResidual statistics:")
print("min      =", residual.min().item())
print("max      =", residual.max().item())
print("mean     =", residual.mean().item())
print("abs mean =", residual.abs().mean().item())
print(
    "RMS      =",
    torch.sqrt(torch.mean(residual.square())).item(),
)


# ============================================================
# CONTINUITY LOSS
# ============================================================

loss = continuity_loss(
    u,
    v,
    neighbors,
    G_wlsq,
    cell_index,
)

print("\nContinuity loss:")
print(loss.item())


# ============================================================
# BACKPROPAGATION TEST
# ============================================================

loss.backward()


print("\n========================================")
print("BACKPROPAGATION TEST")
print("========================================")

print("u.grad exists:", u.grad is not None)
print("v.grad exists:", v.grad is not None)

print("u.grad finite:", torch.isfinite(u.grad).all().item())
print("v.grad finite:", torch.isfinite(v.grad).all().item())

print(
    "u.grad abs mean:",
    u.grad.abs().mean().item(),
)

print(
    "v.grad abs mean:",
    v.grad.abs().mean().item(),
)

print(
    "u.grad max abs:",
    u.grad.abs().max().item(),
)

print(
    "v.grad max abs:",
    v.grad.abs().max().item(),
)