"""
Turning arrays into PyTorch Geometric graphs.

The mesh topology is the same for every snapshot: only the node features
change. Each snapshot therefore becomes one Data object sharing the same
edge_index and edge_weight.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from .features import build_features
from torch.utils.data import TensorDataset

def to_tensor(array, dtype=torch.float32):
    return torch.tensor(np.asarray(array), dtype=dtype)


def create_graph_dataset(X, Y, edge_index, edge_weight, simulation_id=None):
    """
    Build one Data object per snapshot.
    """

    X = X if isinstance(X, torch.Tensor) else to_tensor(X)
    Y = Y if isinstance(Y, torch.Tensor) else to_tensor(Y)

    edge_index = (
        edge_index
        if isinstance(edge_index, torch.Tensor)
        else to_tensor(edge_index, dtype=torch.long)
    )

    edge_weight = (
        edge_weight
        if isinstance(edge_weight, torch.Tensor)
        else to_tensor(edge_weight)
    )

    return [
        Data(
            x=X[i],
            y=Y[i],
            edge_index=edge_index,
            edge_weight=edge_weight,
            simulation_id=simulation_id
        )
        for i in range(X.shape[0])
    ]





def create_bsms_dataset(X, Y):
    """
    Build a dataset for BSMS models.

    Unlike the standard PyG graph dataset, the mesh topology is not
    duplicated for every snapshot. Each item contains only the node
    features and target.

    The fixed BSMS hierarchy and mesh positions are stored by the model.
    """

    X = X if isinstance(X, torch.Tensor) else to_tensor(X)
    Y = Y if isinstance(Y, torch.Tensor) else to_tensor(Y)

    return TensorDataset(X, Y)

def create_multi_simulation_graph_dataset(
    simulations,
    encoding,
):
    """
    Build one PyG dataset from multiple simulations.

    Each timestep transition becomes an independent PyG Data object.
    Simulations may have different numbers of nodes and edges.
    """

    dataset = []

    for simulation in simulations:

        features = build_features(
            encoding,
            simulation["X"],
            simulation["dt"],
        )

        simulation_dataset = create_graph_dataset(
            features,
            simulation["Y"],
            simulation["edge_index"],
            simulation["edge_weight"],
        )

        dataset.extend(simulation_dataset)

    return dataset
