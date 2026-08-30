"""
Turning arrays into PyTorch Geometric graphs.

The mesh topology is the same for every snapshot: only the node features
change. Each snapshot therefore becomes one Data object sharing the same
edge_index and edge_weight.
"""

import numpy as np
import torch
from torch_geometric.data import Data


def to_tensor(array, dtype=torch.float32):
    return torch.tensor(np.asarray(array), dtype=dtype)


def create_graph_dataset(X, Y, edge_index, edge_weight):
    """
    Build one Data object per snapshot.

    Parameters
    ----------
    X : (S, N, F) array or tensor
        Node features, already normalized and with the time encoding.

    Y : (S, N, C) array or tensor
        Target, already normalized. The full state is kept here and the
        columns of interest are selected inside the training loop, so
        that one dataset can serve networks predicting different
        variables.

    edge_index : (2, E) long tensor
    edge_weight : (E,) float tensor
    """

    X = X if isinstance(X, torch.Tensor) else to_tensor(X)
    Y = Y if isinstance(Y, torch.Tensor) else to_tensor(Y)

    return [
        Data(
            x=X[i],
            y=Y[i],
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        for i in range(X.shape[0])
    ]
