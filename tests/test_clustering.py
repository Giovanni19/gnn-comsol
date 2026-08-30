"""Mesh coarsening: clusters must be connected, complete and contiguous."""

import numpy as np
import pytest

from gnn_comsol.graph.clustering import (
    build_coarse_graph,
    cluster_graph,
    edge_index_to_adjacency
)


def grid_graph(rows, cols):
    """4-neighbour grid, a stand-in for a structured mesh."""

    edges = []

    def node(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append((node(r, c), node(r + 1, c)))
            if c + 1 < cols:
                edges.append((node(r, c), node(r, c + 1)))

    edges += [(j, i) for i, j in edges]

    return np.array(edges).T, rows * cols


def test_adjacency_accepts_both_orientations():

    edge_index, num_nodes = grid_graph(3, 3)

    from_2e = edge_index_to_adjacency(edge_index, num_nodes)
    from_e2 = edge_index_to_adjacency(edge_index.T, num_nodes)

    assert from_2e == from_e2

    # Corner of a 3x3 grid has 2 neighbours, centre has 4
    assert len(from_2e[0]) == 2
    assert len(from_2e[4]) == 4


def test_adjacency_drops_self_loops():

    edge_index = np.array([[0, 1, 2], [0, 2, 1]])

    adjacency = edge_index_to_adjacency(edge_index, 3)

    assert adjacency[0] == []
    assert adjacency[1] == [2]


def test_every_node_is_assigned_exactly_once():

    edge_index, num_nodes = grid_graph(10, 10)

    cluster = cluster_graph(edge_index, num_nodes, num_clusters=10)

    assert cluster.shape == (num_nodes,)
    assert cluster.min() >= 0
    assert not np.any(cluster < 0)


def test_cluster_ids_are_contiguous_from_zero():
    """index_add_ in the multiscale pooling requires this."""

    edge_index, num_nodes = grid_graph(10, 10)

    cluster = cluster_graph(edge_index, num_nodes, num_clusters=7)

    assert set(np.unique(cluster)) == set(range(cluster.max() + 1))


def test_clusters_are_connected_regions():
    """Coarsening a mesh only makes sense with contiguous clusters."""

    edge_index, num_nodes = grid_graph(8, 8)

    adjacency = edge_index_to_adjacency(edge_index, num_nodes)
    cluster = cluster_graph(edge_index, num_nodes, num_clusters=8)

    for cluster_id in np.unique(cluster):

        members = set(np.flatnonzero(cluster == cluster_id).tolist())

        # Walk the cluster from one member and check we reach them all
        start = next(iter(members))
        seen = {start}
        stack = [start]

        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour in members and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)

        assert seen == members, f"cluster {cluster_id} is not connected"


def test_disconnected_components_are_never_merged():

    # Two separate triangles
    edge_index = np.array([
        [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
        [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]
    ])

    cluster = cluster_graph(edge_index, 6, num_clusters=2)

    assert set(cluster[:3]) != set(cluster[3:])


def test_coarse_graph_has_no_self_loops():

    edge_index, num_nodes = grid_graph(10, 10)

    cluster = cluster_graph(edge_index, num_nodes, num_clusters=10)
    coarse = build_coarse_graph(edge_index, cluster)

    assert coarse.shape[0] == 2
    assert not np.any(coarse[0] == coarse[1])


def test_coarse_graph_has_no_duplicate_edges():

    edge_index, num_nodes = grid_graph(10, 10)

    cluster = cluster_graph(edge_index, num_nodes, num_clusters=10)
    coarse = build_coarse_graph(edge_index, cluster)

    unique = {tuple(pair) for pair in coarse.T.tolist()}

    assert len(unique) == coarse.shape[1]


def test_more_clusters_than_nodes_is_clamped():

    edge_index, num_nodes = grid_graph(3, 3)

    cluster = cluster_graph(edge_index, num_nodes, num_clusters=100)

    assert cluster.max() + 1 <= num_nodes


def test_invalid_cluster_count_raises():

    edge_index, num_nodes = grid_graph(3, 3)

    with pytest.raises(ValueError):
        cluster_graph(edge_index, num_nodes, num_clusters=0)


def test_unknown_method_raises():

    edge_index, num_nodes = grid_graph(3, 3)

    with pytest.raises(ValueError):
        cluster_graph(edge_index, num_nodes, 2, method="spectral")
