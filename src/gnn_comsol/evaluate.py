"""
Evaluation and one-step inference.

The relative error is deliberately NOT the RMSE divided by the mean of
the target, which is what the old inference.py did. In a 2D channel the
mean of v is close to zero and so is the mean of p when the pressure is
only defined up to a constant, so that ratio can be huge, negative or
undefined. Here the RMSE is reported against the spread of the field and
against its L2 norm, both of which are well behaved.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from .data.features import build_features
from .data.normalization import VARIABLE_NAMES


def per_variable_metrics(Y_true, Y_pred):
    """
    RMSE and two well-defined relative errors, per variable.

    Both inputs are (N, 3) arrays in physical units.
    """

    metrics = {}

    for column, name in enumerate(VARIABLE_NAMES):

        true = np.asarray(Y_true)[:, column]
        pred = np.asarray(Y_pred)[:, column]

        error = pred - true

        rmse = float(np.sqrt(np.mean(error ** 2)))

        spread = float(true.max() - true.min())
        norm = float(np.sqrt(np.sum(true ** 2)))

        metrics[name] = {
            "rmse": rmse,
            # fraction of the range the error represents
            "rmse_over_range": rmse / spread if spread > 0 else float("nan"),
            # ||y - y_hat|| / ||y||, the usual relative L2 error
            "relative_l2": (
                float(np.sqrt(np.sum(error ** 2)) / norm)
                if norm > 0 else float("nan")
            )
        }

    return metrics


def format_metrics(metrics, title="INFERENCE RESULTS (physical units)"):

    lines = ["", "=" * 44, title, "=" * 44]

    for name, values in metrics.items():
        lines.append(
            f"{name}: "
            f"RMSE={values['rmse']:.6e} | "
            f"RMSE/range={100 * values['rmse_over_range']:.3f}% | "
            f"rel. L2={100 * values['relative_l2']:.3f}%"
        )

    return "\n".join(lines)


def evaluate_dataset(net, loader, criterion, device, target_columns):
    """
    Mean loss of a network over a loader, in normalized units.

    Supports both:
    - standard PyG Data batches;
    - BSMS tensor batches (X, Y).
    """

    net.eval()

    losses = []

    with torch.no_grad():

        for batch in loader:

            if isinstance(batch, (list, tuple)):

                X, Y = batch

                X = X.to(device)
                Y = Y.to(device)

                preds = net(X)
                target = Y[..., target_columns]

            else:

                batch = batch.to(device)

                preds = net(batch)
                target = batch.y[:, target_columns]

            losses.append(
                criterion(preds, target).item()
            )

    return float(np.mean(losses))


def predict_next_timestep(
    networks,
    X,
    delta_t,
    edge_index,
    edge_weight,
    normalizer,
    dt_mean,
    dt_std,
    device,
    num_frequencies=4
):
    """
    Predict the state at the next timestep, in physical units.

    The velocity network is evaluated first.

    If the pressure network was trained with predicted velocity,
    its input is augmented with the normalized predictions
    u_hat(t+1), v_hat(t+1).

    Returns
    -------
    (N, 3) array in physical units.
    """

    X = np.asarray(X)

    n_nodes = X.shape[0]

    # ---------------------------------------------------------
    # Normalize current state and timestep
    # ---------------------------------------------------------

    X_norm = normalizer.transform(X)

    delta_t_norm = (delta_t - dt_mean) / dt_std

    def features_for(kind):

        return build_features(
            kind,
            X_norm[None, ...],
            np.array([delta_t_norm]),
            num_frequencies=num_frequencies
        )[0]

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Final prediction in normalized units
    prediction_norm = np.zeros(
        (n_nodes, 3),
        dtype=np.float32
    )

    covered = np.zeros(3, dtype=int)

    # =========================================================
    # 1. VELOCITY
    # =========================================================

    if "velocity" not in networks:
        raise ValueError(
            "A velocity network is required before pressure prediction."
        )

    velocity_spec = networks["velocity"]

    velocity_model = velocity_spec["model"]

    velocity_columns = velocity_spec["columns"]

    velocity_x = torch.tensor(
        features_for(velocity_spec["features"]),
        dtype=torch.float32,
        device=device
    )

    velocity_model.eval()

    with torch.no_grad():

        if getattr(
            velocity_model,
            "uses_bsms_tensor_input",
            False
        ):

            velocity_out = velocity_model(velocity_x)

        else:

            velocity_graph = Data(
                x=velocity_x,
                edge_index=edge_index,
                edge_weight=edge_weight
            )

            velocity_out = velocity_model(
                velocity_graph
            )

    # Keep velocity predictions NORMALIZED.
    # These are exactly the quantities used as pressure features
    # during training.
    velocity_pred_norm = (
        velocity_out.detach().cpu().numpy()
    )

    prediction_norm[:, velocity_columns] = (
        velocity_pred_norm
    )

    covered[velocity_columns] += 1

    # =========================================================
    # 2. PRESSURE
    # =========================================================

    if "pressure" not in networks:
        raise ValueError(
            "A pressure network is required."
        )

    pressure_spec = networks["pressure"]

    pressure_model = pressure_spec["model"]

    pressure_columns = pressure_spec["columns"]

    # Base pressure features:
    # [u(t), v(t), p(t), dt, ...]
    pressure_features = features_for(
        pressure_spec["features"]
    )

    # ---------------------------------------------------------
    # Add predicted u(t+1), v(t+1)
    # ---------------------------------------------------------

    use_predicted_velocity = pressure_spec.get(
        "use_predicted_velocity",
        False
    )

    if use_predicted_velocity:

        pressure_features = np.concatenate(
            [
                pressure_features,
                velocity_pred_norm
            ],
            axis=-1
        )

    pressure_x = torch.tensor(
        pressure_features,
        dtype=torch.float32,
        device=device
    )

    pressure_model.eval()

    with torch.no_grad():

        if getattr(
            pressure_model,
            "uses_bsms_tensor_input",
            False
        ):

            pressure_out = pressure_model(
                pressure_x
            )

        else:

            pressure_graph = Data(
                x=pressure_x,
                edge_index=edge_index,
                edge_weight=edge_weight
            )

            pressure_out = pressure_model(
                pressure_graph
            )

    pressure_pred_norm = (
        pressure_out.detach().cpu().numpy()
    )

    prediction_norm[:, pressure_columns] = (
        pressure_pred_norm
    )

    covered[pressure_columns] += 1

    # =========================================================
    # Check coverage
    # =========================================================

    if not np.all(covered == 1):

        raise ValueError(
            "The networks must cover u, v and p exactly once; "
            f"coverage per column is {covered.tolist()}."
        )

    # =========================================================
    # Convert complete prediction back to physical units
    # =========================================================

    prediction = normalizer.inverse_transform(
        prediction_norm
    )

    return prediction

def evaluate_bsms_multi_simulation(
    net,
    loaders,
    hierarchies,
    criterion,
    device,
    target_columns=slice(None),
):
    """
    Evaluate a BSMS network on multiple simulations with different meshes.

    Each simulation has its own DataLoader and its own BSMS hierarchy.
    The returned loss is averaged over samples.
    """

    net.eval()

    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():

        for simulation_id, loader in loaders.items():

            hierarchy = hierarchies[simulation_id]

            edge_indices = [
                edge_index.to(device)
                for edge_index in hierarchy["edge_indices"]
            ]

            pool_indices = [
                indices.to(device)
                for indices in hierarchy["pool_indices"]
            ]

            pos = hierarchy["pos"].to(device)

            for X, Y, _ in loader:

                X = X.to(device)
                Y = Y.to(device)

                preds = net(
                    X,
                    pool_indices,
                    edge_indices,
                    pos,
                )

                target = Y[
                    ...,
                    target_columns
                ]

                loss = criterion(
                    preds,
                    target,
                )

                batch_size = X.shape[0]

                loss_sum += (
                    loss.item() * batch_size
                )

                sample_count += batch_size

    if sample_count == 0:
        raise ValueError(
            "Cannot evaluate BSMS: no samples were found."
        )

    return loss_sum / sample_count
