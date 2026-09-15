import torch

from gnn_comsol.data.loading import load_data
from gnn_comsol.physics.wlsq import wlsq_gradient


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
# CONVERT WLSQ GEOMETRY TO TORCH
# ============================================================

neighbors = [
    torch.as_tensor(neigh, dtype=torch.long)
    for neigh in data.neighbors
]

G_wlsq = [
    torch.as_tensor(G_i, dtype=torch.float64)
    for G_i in data.G_wlsq
]


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

grad_u = wlsq_gradient(
    u,
    neighbors,
    G_wlsq,
)

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
# EXACT LINEAR FIELD TEST
# ============================================================

pos = torch.as_tensor(
    data.pos,
    dtype=torch.float64,
)

x = pos[:, 0]
y = pos[:, 1]

phi = 2.0 * x + 3.0 * y + 5.0

grad_phi = wlsq_gradient(
    phi,
    neighbors,
    G_wlsq,
)

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