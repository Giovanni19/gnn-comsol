
import numpy as np
import torch

from torch_geometric.data import Data


def predict_next_timestep(
    velocity_model,
    pressure_model,
    X,
    delta_t,
    edge_index,
    edge_weight,
    x_mean,
    x_std,
    dt_mean,
    dt_std,
    device,
    num_frequencies=4
):

    # ============================================================
    # Normalize physical input
    # ============================================================

    X_norm = (
        X - x_mean
    ) / x_std


    # ============================================================
    # Normalize timestep
    # ============================================================

    delta_t_norm = (
        delta_t - dt_mean
    ) / dt_std


    # ============================================================
    # VELOCITY INPUT
    #
    # [u_norm, v_norm, p_norm, delta_t_norm]
    # ============================================================

    n_nodes = X_norm.shape[0]

    dt_feature = np.full(
        (n_nodes, 1),
        delta_t_norm
    )

    X_velocity = np.concatenate(
        [
            X_norm,
            dt_feature
        ],
        axis=1
    )


    # ============================================================
    # PRESSURE INPUT
    #
    # [
    #   u_norm,
    #   v_norm,
    #   p_norm,
    #   delta_t_norm,
    #   Fourier(delta_t_norm)
    # ]
    # ============================================================

    pressure_features = [
        X_norm,
        dt_feature
    ]

    frequencies = (
        2.0 ** np.arange(num_frequencies)
    )

    for frequency in frequencies:

        sin_value = np.sin(
            2.0
            * np.pi
            * frequency
            * delta_t_norm
        )

        cos_value = np.cos(
            2.0
            * np.pi
            * frequency
            * delta_t_norm
        )

        sin_feature = np.full(
            (n_nodes, 1),
            sin_value
        )

        cos_feature = np.full(
            (n_nodes, 1),
            cos_value
        )

        pressure_features.append(
            sin_feature
        )

        pressure_features.append(
            cos_feature
        )


    X_pressure = np.concatenate(
        pressure_features,
        axis=1
    )


    # ============================================================
    # Convert inputs to tensors
    # ============================================================

    X_velocity = torch.tensor(
        X_velocity,
        dtype=torch.float32,
        device=device
    )

    X_pressure = torch.tensor(
        X_pressure,
        dtype=torch.float32,
        device=device
    )


    # ============================================================
    # Move graph structure to device
    # ============================================================

    edge_index = edge_index.to(
        device
    )

    edge_weight = edge_weight.to(
        device
    )


    # ============================================================
    # Create velocity graph
    # ============================================================

    velocity_graph = Data(
        x=X_velocity,
        edge_index=edge_index,
        edge_weight=edge_weight
    )


    # ============================================================
    # Create pressure graph
    # ============================================================

    pressure_graph = Data(
        x=X_pressure,
        edge_index=edge_index,
        edge_weight=edge_weight
    )


    # ============================================================
    # Inference
    # ============================================================

    velocity_model.eval()
    pressure_model.eval()

    with torch.no_grad():

        velocity_prediction = velocity_model(
            velocity_graph
        )

        pressure_prediction = pressure_model(
            pressure_graph
        )

    pressure_mean = x_mean[2]
    pressure_std = x_std[2]

    pressure_prediction = (
        pressure_prediction * pressure_std
        + pressure_mean
    )
    # ============================================================
    # Combine predictions
    # ============================================================

    Y_pred = torch.cat(
        [
            velocity_prediction,
            pressure_prediction
        ],
        dim=1
    )


    # ============================================================
    # Convert to NumPy
    # ============================================================

    Y_pred = (
        Y_pred
        .cpu()
        .numpy()
    )


    return Y_pred

