#!/usr/bin/env python3

"""
One-step evaluation of trained velocity and pressure models
on one snapshot of an unseen test simulation.
"""

import sys
from pathlib import Path

import numpy as np
import torch


# Make src/ importable
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)


from gnn_comsol import data as gdata
from gnn_comsol.checkpoints import load_checkpoint
from gnn_comsol.models import build_model
from gnn_comsol.graph.bsms import BistrideMultiLayerGraph
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
)

# ===============================================================
# Configuration
# ===============================================================

RUN_DIR = Path(
    "outputs/multi_geometry_bsms_predicted_velocity_test_20260902_120159"
)

TEST_DATASET = (
    "C:/Users/giovanni/.comsol/v64/llmatlab/"
    "channel2d_gnn_dataset.mat"
)




# ===============================================================
# Device
# ===============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ===============================================================
# Helper: rebuild a model from checkpoint metadata
# ===============================================================

def build_from_checkpoint(checkpoint_path):

    # First load metadata and normalizer without a model
    _, normalizer, metadata = load_checkpoint(
        checkpoint_path,
        model=None,
        device=device,
    )

    architecture = metadata["architecture"]

    model_kwargs = {
        "architecture": architecture,
        "num_in": metadata["num_in"],
        "num_out": metadata["num_out"],
        "num_neurons": metadata["num_neurons"],
        "num_layers": metadata["num_layers"],
        "dropout": metadata["dropout"],
    }

    if architecture == "bsms":

        model_kwargs.update(
            {
                "unet_depth": metadata["unet_depth"],
                "hidden_layers": metadata["hidden_layers"],
                "pos_dim": 2,
            }
        )

    model = build_model(
        **model_kwargs
    ).to(device)

    model, normalizer, metadata = load_checkpoint(
        checkpoint_path,
        model=model,
        device=device,
    )

    model.eval()

    return model, normalizer, metadata


# ===============================================================
# Load trained networks
# ===============================================================

velocity_model, velocity_normalizer, velocity_metadata = (
    build_from_checkpoint(
        RUN_DIR / "velocity.pth"
    )
)

pressure_model, pressure_normalizer, pressure_metadata = (
    build_from_checkpoint(
        RUN_DIR / "pressure.pth"
    )
)


print("\nVelocity metadata:")
print(velocity_metadata)

print("\nPressure metadata:")
print(pressure_metadata)


# ===============================================================
# Verify that both networks use the same normalization
# ===============================================================

if not np.allclose(
    velocity_normalizer.mean,
    pressure_normalizer.mean,
):
    raise ValueError(
        "Velocity and pressure checkpoints use different means."
    )

if not np.allclose(
    velocity_normalizer.std,
    pressure_normalizer.std,
):
    raise ValueError(
        "Velocity and pressure checkpoints use different stds."
    )

normalizer = velocity_normalizer


# ===============================================================
# Load TEST simulation
# ===============================================================

skip_initial = 0

test_simulation = gdata.load_data(
    TEST_DATASET,
    skip_initial=skip_initial,
    simulation_id=3,
)

print(
    f"\nTest simulation: "
    f"{test_simulation.num_samples} samples | "
    f"{test_simulation.num_nodes} nodes | "
    f"{test_simulation.num_edges} edges"
)


# ===============================================================
# Important: delta_t normalization
#
# These values are taken from the training run output.
# Later we should store them in the checkpoint too.
# ===============================================================

dt_mean = 9.788190e-02
dt_std = 1.326544e-02


# ===============================================================
# Normalize TEST simulation
# ===============================================================

X_norm = normalizer.transform(
    test_simulation.X_input
)

Y_norm = normalizer.transform(
    test_simulation.Y_target
)

dt_norm = (
    test_simulation.delta_t - dt_mean
) / dt_std


# ===============================================================
# Check normalization round trip
# ===============================================================

Y_reconstructed = normalizer.inverse_transform(
    Y_norm
)

round_trip_error = np.max(
    np.abs(
        Y_reconstructed
        - test_simulation.Y_target
    )
)

print(
    "\nNormalization round-trip max error: "
    f"{round_trip_error:.6e}"
)


# ===============================================================
# VELOCITY FEATURES
# ===============================================================

velocity_features = gdata.build_features(
    velocity_metadata["features"],
    X_norm,
    dt_norm,
)




edge_index = torch.as_tensor(
    test_simulation.edge_index,
    dtype=torch.long,
    device=device,
)

edge_weight = torch.as_tensor(
    test_simulation.edge_weight,
    dtype=torch.float32,
    device=device,
)





# ===============================================================
# VELOCITY INFERENCE
# ===============================================================




# ===============================================================
# BUILD TEST BSMS HIERARCHY
# ===============================================================

unet_depth = pressure_metadata[
    "unet_depth"
]

multi_layer_graph = BistrideMultiLayerGraph(
    test_simulation.edge_index,
    unet_depth,
    test_simulation.num_nodes,
    test_simulation.pos,
)

_, m_flat_es, m_ids = (
    multi_layer_graph.get_multi_layer_graphs()
)


edge_indices = [
    torch.as_tensor(
        edges,
        dtype=torch.long,
        device=device,
    )
    for edges in m_flat_es
]

pool_indices = [
    torch.as_tensor(
        indices,
        dtype=torch.long,
        device=device,
    )
    for indices in m_ids
]

pos = torch.as_tensor(
    test_simulation.pos,
    dtype=torch.float32,
    device=device,
)


# ===============================================================
# PRESSURE FEATURES
# ===============================================================

pressure_features = gdata.build_features(
    pressure_metadata["features"],
    X_norm,
    dt_norm,
)

# ===============================================================
# EVALUATE THE WHOLE TEST SET
# ===============================================================

num_samples = test_simulation.num_samples

rmse_u = np.zeros(num_samples)
rmse_v = np.zeros(num_samples)
rmse_p = np.zeros(num_samples)

velocity_model.eval()
pressure_model.eval()

print(
    f"\nEvaluating all {num_samples} test samples..."
)


with torch.no_grad():
    pressure_absolute_error = np.zeros(
    (
        num_samples,
        test_simulation.num_nodes,
    ),
    dtype=np.float32,
    )
    for timestep in range(num_samples):

        # -------------------------------------------------------
        # VELOCITY
        # -------------------------------------------------------

        velocity_x = torch.as_tensor(
            velocity_features[timestep],
            dtype=torch.float32,
            device=device,
        )

        velocity_graph = Data(
            x=velocity_x,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        velocity_pred_norm = velocity_model(
            velocity_graph
        )


        
        # -------------------------------------------------------
        # PRESSURE
        # -------------------------------------------------------

        pressure_x = torch.as_tensor(
            pressure_features[timestep],
            dtype=torch.float32,
            device=device,
        )


        # If the pressure network was trained using the predicted
        # velocity at t+1, append those predictions here as well.
        if pressure_metadata.get(
            "use_predicted_velocity",
            False,
        ):

            pressure_x = torch.cat(
                [
                    pressure_x,
                    velocity_pred_norm,
                ],
                dim=-1,
            )


        # Safety check
        if pressure_x.shape[-1] != pressure_metadata["num_in"]:

            raise ValueError(
                f"Pressure model expects "
                f"{pressure_metadata['num_in']} input features, "
                f"but evaluation constructed "
                f"{pressure_x.shape[-1]}."
            )


        pressure_pred_norm = pressure_model(
            pressure_x,
            pool_indices,
            edge_indices,
            pos,
        )

        


        # -------------------------------------------------------
        # COMPLETE PREDICTION
        # -------------------------------------------------------

        Y_pred_norm = torch.cat(
            [
                velocity_pred_norm,
                pressure_pred_norm,
            ],
            dim=-1,
        )


        # -------------------------------------------------------
        # DENORMALIZE -> PHYSICAL COMSOL UNITS
        # -------------------------------------------------------

        Y_pred = normalizer.inverse_transform(
            Y_pred_norm
        )

        Y_pred = (
            Y_pred
            .detach()
            .cpu()
            .numpy()
        )


        # Ground truth is already in physical units
        Y_true = test_simulation.Y_target[
            timestep
        ]


        # -------------------------------------------------------
        # RMSE FOR THIS TIMESTEP
        # -------------------------------------------------------

        error = Y_pred - Y_true
        pressure_absolute_error[timestep] = np.abs(
            error[:, 2]
        )

        rmse_u[timestep] = np.sqrt(
            np.mean(error[:, 0] ** 2)
        )

        rmse_v[timestep] = np.sqrt(
            np.mean(error[:, 1] ** 2)
        )

        rmse_p[timestep] = np.sqrt(
            np.mean(error[:, 2] ** 2)
        )


        # -------------------------------------------------------
        # PROGRESS
        # -------------------------------------------------------

        if (
            timestep % 100 == 0
            or timestep == num_samples - 1
        ):

            print(
                f"{timestep + 1}/{num_samples} | "
                f"RMSE u={rmse_u[timestep]:.3e} | "
                f"v={rmse_v[timestep]:.3e} | "
                f"p={rmse_p[timestep]:.3e}"
            )


# ===============================================================
# PHYSICAL TIME
# ===============================================================

# Each sample predicts the state at the end of its transition.
time = np.cumsum(
    test_simulation.delta_t
)


# ===============================================================
# SUMMARY
# ===============================================================

print("\n" + "=" * 60)
print("TEST SET RMSE - PHYSICAL UNITS")
print("=" * 60)

for name, values in [
    ("u", rmse_u),
    ("v", rmse_v),
    ("p", rmse_p),
]:

    max_index = np.argmax(values)

    print(f"\n{name}")

    print(
        f"  Mean RMSE:   "
        f"{values.mean():.6e}"
    )

    print(
        f"  Median RMSE: "
        f"{np.median(values):.6e}"
    )

    print(
        f"  Min RMSE:    "
        f"{values.min():.6e}"
    )

    print(
        f"  Max RMSE:    "
        f"{values.max():.6e}"
    )

    print(
        f"  Max at timestep: "
        f"{max_index}"
    )

    print(
        f"  Max at time: "
        f"{time[max_index]:.6e}"
    )


# ===============================================================
# SAVE NUMERICAL RESULTS
# ===============================================================

np.savez(
    RUN_DIR / "test_rmse_over_time.npz",
    time=time,
    rmse_u=rmse_u,
    rmse_v=rmse_v,
    rmse_p=rmse_p,
    pressure_absolute_error=pressure_absolute_error,
)


# ===============================================================
# PLOT U
# ===============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    rmse_u,
)

plt.xlabel("Time")
plt.ylabel("RMSE u")
plt.title("Test set - u RMSE over time")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "test_rmse_u.png",
    dpi=200,
)

plt.show()


# ===============================================================
# PLOT V
# ===============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    rmse_v,
)

plt.xlabel("Time")
plt.ylabel("RMSE v")
plt.title("Test set - v RMSE over time")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "test_rmse_v.png",
    dpi=200,
)

plt.show()


# ===============================================================
# PLOT PRESSURE
# ===============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    rmse_p,
)

plt.xlabel("Time")
plt.ylabel("RMSE pressure")
plt.title("Test set - pressure RMSE over time")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "test_rmse_pressure.png",
    dpi=200,
)

plt.show()


print("\nDone.")

# ===============================================================
# LOG-SCALE RMSE PLOTS
# ===============================================================


# ---------------------------------------------------------------
# U - LOG SCALE
# ---------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.semilogy(
    time,
    rmse_u,
)

plt.xlabel("Time")
plt.ylabel("RMSE u")
plt.title("Test set - u RMSE over time (log scale)")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "test_rmse_u_log.png",
    dpi=200,
)

plt.show()


# ---------------------------------------------------------------
# V - LOG SCALE
# ---------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.semilogy(
    time,
    rmse_v,
)

plt.xlabel("Time")
plt.ylabel("RMSE v")
plt.title("Test set - v RMSE over time (log scale)")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "test_rmse_v_log.png",
    dpi=200,
)

plt.show()


# ---------------------------------------------------------------
# PRESSURE - LOG SCALE
# ---------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.semilogy(
    time,
    rmse_p,
)

plt.xlabel("Time")
plt.ylabel("RMSE pressure")
plt.title("Test set - pressure RMSE over time (log scale)")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RUN_DIR / "test_rmse_pressure_log.png",
    dpi=200,
)

plt.show()

# ===============================================================
# PRESSURE ABSOLUTE ERROR - SPATIAL ANIMATION
# ===============================================================

print("\nCreating pressure error animation...")


# ---------------------------------------------------------------
# Node coordinates
# ---------------------------------------------------------------

x = test_simulation.pos[:, 0]
y = test_simulation.pos[:, 1]


# ---------------------------------------------------------------
# Build a triangulation for visualization
# ---------------------------------------------------------------

triangulation = mtri.Triangulation(
    x,
    y,
)


# ---------------------------------------------------------------
# Fixed color scale for the whole animation
#
# Using the 99th percentile prevents a few extreme nodes from
# making the rest of the error field visually indistinguishable.
# ---------------------------------------------------------------

vmin = 0.0

vmax = np.percentile(
    pressure_absolute_error,
    99,
)

print(
    f"Color scale: "
    f"{vmin:.3e} -> {vmax:.3e}"
)


# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 5)
)


# Initial field, used also to create the fixed colorbar
initial_plot = ax.tripcolor(
    triangulation,
    pressure_absolute_error[0],
    shading="gouraud",
    vmin=vmin,
    vmax=vmax,
    cmap="viridis",
)


colorbar = fig.colorbar(
    initial_plot,
    ax=ax,
)

colorbar.set_label(
    "Absolute pressure error"
)


# ---------------------------------------------------------------
# Animation update
# ---------------------------------------------------------------

def update(frame):

    ax.clear()

    plot = ax.tripcolor(
        triangulation,
        pressure_absolute_error[frame],
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Pressure absolute error | "
        f"t = {time[frame]:.3f} | "
        f"RMSE = {rmse_p[frame]:.3e}"
    )

    return plot,


# ---------------------------------------------------------------
# Use one frame every 5 simulation samples.
#
# 1018 samples -> about 204 animation frames.
# ---------------------------------------------------------------

animation_frames = range(num_samples)


animation = FuncAnimation(
    fig,
    update,
    frames=animation_frames,
    interval=50,
    blit=False,
)


# ---------------------------------------------------------------
# Save GIF
# ---------------------------------------------------------------

animation_path = (
    RUN_DIR / "pressure_error_animation.gif"
)

animation.save(
    animation_path,
    writer=PillowWriter(
        fps=1
    ),
)

plt.close(fig)


print(
    f"Pressure error animation saved to: "
    f"{animation_path}"
)