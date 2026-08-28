from torch_geometric.data import Data
import torch
import numpy as np

def predict_next_timestep(
    model,
    X,
    delta_t,
    edge_index,
    edge_weight,
    x_mean,
    x_std,
    device
):

    # Normalize u, v, p
    X_norm = (X - x_mean) / x_std

    # Add time as fourth feature
    time_feature = np.full(
        (X.shape[0], 1),
        delta_t
    )

    X_features = np.concatenate(
        (
            X_norm,
            time_feature
        ),
        axis=1
    )

    # Convert to PyTorch
    X_features = torch.tensor(
        X_features,
        dtype=torch.float32
    )

    # Create graph
    graph = Data(
        x=X_features,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

    graph = graph.to(device)

    # Prediction
    model.eval()

    with torch.no_grad():
        prediction = model(graph)

    # Convert to NumPy
    prediction = (
        prediction
        .cpu()
        .numpy()
    )

    # Target is NOT normalized,
    # therefore prediction is already in physical units

    return prediction


def compute_mse(Y_true, Y_pred):

    # Total MSE
    mse_total = np.mean(
        (Y_true - Y_pred) ** 2
    )

    # MSE for each variable
    mse_u = np.mean(
        (Y_true[:, 0] - Y_pred[:, 0]) ** 2
    )

    mse_v = np.mean(
        (Y_true[:, 1] - Y_pred[:, 1]) ** 2
    )

    mse_p = np.mean(
        (Y_true[:, 2] - Y_pred[:, 2]) ** 2
    )

    return (
        mse_total,
        mse_u,
        mse_v,
        mse_p
    )

def compute_relative_error(Y_true, Y_pred):

    # ==========================================
    # RMSE
    # ==========================================

    rmse_u = np.sqrt(
        np.mean((Y_true[:, 0] - Y_pred[:, 0]) ** 2)
    )

    rmse_v = np.sqrt(
        np.mean((Y_true[:, 1] - Y_pred[:, 1]) ** 2)
    )

    rmse_p = np.sqrt(
        np.mean((Y_true[:, 2] - Y_pred[:, 2]) ** 2)
    )


    # ==========================================
    # Mean target values
    # ==========================================

    mean_u = np.mean(Y_true[:, 0])
    mean_v = np.mean(Y_true[:, 1])
    mean_p = np.mean(Y_true[:, 2])


    # ==========================================
    # Relative RMSE [%]
    # ==========================================

    rrmse_u = (rmse_u / mean_u) * 100
    rrmse_v = (rmse_v / mean_v) * 100
    rrmse_p = (rmse_p / mean_p) * 100


    return rrmse_u, rrmse_v, rrmse_p
