"""Graph coarsening utilities."""

from .clustering import (
    build_coarse_graph,
    cluster_graph,
    edge_index_to_adjacency
)

__all__ = [
    "build_coarse_graph",
    "cluster_graph",
    "edge_index_to_adjacency"
]
