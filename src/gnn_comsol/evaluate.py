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
    """

    net.eval()

    losses = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            preds = net(batch)
            target = batch.y[:, target_columns]

            losses.append(criterion(preds, target).item())

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

    Parameters
    ----------
    networks : dict
        name -> {"model", "features", "columns"} for each trained
        network. The columns of the different networks must together
        cover u, v and p exactly once.

    normalizer : StateNormalizer
        The one stored in the checkpoints. It defines both how the input
        is scaled and how the output is scaled back, which is what keeps
        training and inference in agreement.

    Returns
    -------
    (N, 3) array in physical units.
    """

    X = np.asarray(X)

    n_nodes = X.shape[0]

    X_norm = normalizer.transform(X)

    delta_t_norm = (delta_t - dt_mean) / dt_std

    # build_features works on a (S, N, C) stack, so add and drop an axis
    def features_for(kind):
        return build_features(
            kind,
            X_norm[None, ...],
            np.array([delta_t_norm]),
            num_frequencies=num_frequencies
        )[0]

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    prediction = np.zeros((n_nodes, 3), dtype=np.float64)

    covered = np.zeros(3, dtype=int)

    for spec in networks.values():

        model = spec["model"]
        columns = spec["columns"]

        x = torch.tensor(
            features_for(spec["features"]),
            dtype=torch.float32,
            device=device
        )

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight
        )

        model.eval()

        with torch.no_grad():
            out = model(graph)

        out = normalizer.inverse_transform(out, columns)

        prediction[:, columns] = out.cpu().numpy()

        covered[columns] += 1

    if not np.all(covered == 1):
        raise ValueError(
            "The networks must cover u, v and p exactly once; "
            f"coverage per column is {covered.tolist()}."
        )

    return prediction
