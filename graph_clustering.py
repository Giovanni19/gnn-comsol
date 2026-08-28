

def edge_index_to_metis_adjacency(edge_index, num_nodes):
    """
    Convert PyTorch-Geometric-style edge_index into the adjacency-list
    format required by PyMETIS.

    Parameters
    ----------
    edge_index : numpy.ndarray or torch.Tensor
        Graph connectivity.

        Accepted shapes:
            (E, 2)
        or:
            (2, E)

        Node indices must be zero-based:
            0, 1, ..., num_nodes-1

    num_nodes : int
        Total number of graph nodes.

    Returns
    -------
    adjacency : list[list[int]]
        adjacency[i] contains all nodes connected to node i.
    """

    # Convert torch Tensor to NumPy if necessary
    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()

    # ---------------------------------------------------------
    # Make edge_index have shape (E, 2)
    # ---------------------------------------------------------

    if edge_index.shape[1] == 2:
        edges = edge_index

    elif edge_index.shape[0] == 2:
        edges = edge_index.T

    else:
        raise ValueError(
            f"Unexpected edge_index shape: {edge_index.shape}"
        )

    # ---------------------------------------------------------
    # Build adjacency
    # ---------------------------------------------------------

    adjacency = [set() for _ in range(num_nodes)]

    for i, j in edges:

        i = int(i)
        j = int(j)

        # Ignore self loops
        if i == j:
            continue

        # METIS graph is undirected
        adjacency[i].add(j)
        adjacency[j].add(i)

    # PyMETIS wants lists, not sets
    adjacency = [
        sorted(list(neighbors))
        for neighbors in adjacency
    ]

    return adjacency

import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData


edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4],
    [1, 0, 2, 1, 3, 2, 4, 3]
], dtype=torch.long)

x = torch.randn(5, 3)

data = Data(
    x=x,
    edge_index=edge_index
)

cluster_data = ClusterData(
    data,
    num_parts=2
)

print(cluster_data)