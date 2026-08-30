"""
Coarsening the mesh graph.

The multiscale pressure model needs two things that the dataset does not
provide: a `cluster` vector assigning every fine node to a cluster, and
the connectivity of the resulting coarse graph. This module produces
both.

The previous version of this file had a well-documented PyMETIS adapter
followed by an unrelated toy snippet that ran at import time, and nothing
that actually returned a cluster vector.
"""

from collections import deque

import numpy as np


def edge_index_to_adjacency(edge_index, num_nodes):
    """
    Convert a PyG-style edge_index into an adjacency list.

    Accepts (2, E) or (E, 2), NumPy or torch. Self loops are dropped and
    the relation is symmetrised, since the mesh graph is undirected.

    This is also the format PyMETIS expects.
    """

    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()

    edge_index = np.asarray(edge_index)

    if edge_index.ndim != 2:
        raise ValueError(
            f"edge_index must be 2-D, got shape {edge_index.shape}."
        )

    if edge_index.shape[0] == 2:
        edges = edge_index.T
    elif edge_index.shape[1] == 2:
        edges = edge_index
    else:
        raise ValueError(
            f"Unexpected edge_index shape: {edge_index.shape}. "
            "Expected (2, E) or (E, 2)."
        )

    adjacency = [set() for _ in range(num_nodes)]

    for i, j in edges:

        i, j = int(i), int(j)

        if i == j:
            continue

        adjacency[i].add(j)
        adjacency[j].add(i)

    return [sorted(neighbours) for neighbours in adjacency]


def cluster_graph(
    edge_index,
    num_nodes,
    num_clusters,
    method="bfs"
):
    """
    Assign every node to a cluster.

    Parameters
    ----------
    method : {"bfs", "metis"}

        "bfs"
            Dependency-free region growing: repeatedly pick an
            unassigned node and grow a connected region around it until
            it reaches the target size. Produces contiguous clusters,
            which is what coarsening a mesh needs, and needs nothing
            beyond NumPy.

        "metis"
            Better balanced and better edge cut, via PyMETIS. Optional
            dependency; raises a clear error if it is not installed.

    Returns
    -------
    cluster : (num_nodes,) int64 array
        cluster[i] is the cluster id of node i, contiguous from 0.
    """

    if num_clusters < 1:
        raise ValueError(
            f"num_clusters must be at least 1, got {num_clusters}."
        )

    num_clusters = min(num_clusters, num_nodes)

    adjacency = edge_index_to_adjacency(edge_index, num_nodes)

    if method == "metis":

        try:
            import pymetis
        except ImportError as error:
            raise ImportError(
                "method='metis' needs PyMETIS, which is not installed. "
                "Install it, or use method='bfs' which needs no extra "
                "dependency."
            ) from error

        _, membership = pymetis.part_graph(
            num_clusters,
            adjacency=adjacency
        )

        return _relabel(np.asarray(membership, dtype=np.int64))

    if method != "bfs":
        raise ValueError(
            f"Unknown clustering method {method!r}. "
            "Expected 'bfs' or 'metis'."
        )

    # -----------------------------------------------------------
    # Region growing
    # -----------------------------------------------------------

    target_size = max(1, int(np.ceil(num_nodes / num_clusters)))

    cluster = np.full(num_nodes, -1, dtype=np.int64)

    current = 0

    for seed in range(num_nodes):

        if cluster[seed] != -1:
            continue

        # Grow one connected region from this seed
        queue = deque([seed])
        cluster[seed] = current
        size = 1

        while queue and size < target_size:

            node = queue.popleft()

            for neighbour in adjacency[node]:

                if cluster[neighbour] != -1:
                    continue

                cluster[neighbour] = current
                size += 1
                queue.append(neighbour)

                if size >= target_size:
                    break

        current += 1

    return _relabel(cluster)


def _relabel(cluster):
    """Make cluster ids contiguous from 0, as index_add_ requires."""

    _, relabelled = np.unique(cluster, return_inverse=True)

    return relabelled.astype(np.int64).reshape(cluster.shape)


def build_coarse_graph(edge_index, cluster):
    """
    Connectivity of the coarse graph induced by `cluster`.

    Maps the endpoints of every fine edge onto their clusters, drops
    intra-cluster edges and removes duplicates.

    Works with NumPy arrays or torch tensors, returning the same kind.
    """

    is_torch = hasattr(edge_index, "detach")

    if is_torch:
        import torch

        cluster_t = (
            cluster
            if hasattr(cluster, "detach")
            else torch.as_tensor(cluster, device=edge_index.device)
        )

        source_cluster = cluster_t[edge_index[0]]
        target_cluster = cluster_t[edge_index[1]]

        mask = source_cluster != target_cluster

        coarse = torch.stack(
            [source_cluster[mask], target_cluster[mask]],
            dim=0
        )

        return torch.unique(coarse, dim=1)

    edge_index = np.asarray(edge_index)
    cluster = np.asarray(cluster)

    source_cluster = cluster[edge_index[0]]
    target_cluster = cluster[edge_index[1]]

    mask = source_cluster != target_cluster

    coarse = np.stack(
        [source_cluster[mask], target_cluster[mask]],
        axis=0
    )

    return np.unique(coarse, axis=1)
