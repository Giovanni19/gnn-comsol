"""
Graph convolutional network with a global virtual node.

Why
---
Pressure in an incompressible flow satisfies a Poisson equation: it is
non-local, and a disturbance anywhere is felt everywhere. A GCN with L
layers only sees a neighbourhood of L hops, so reaching that behaviour
by depth alone needs a very deep and badly conditioned stack.

A virtual node connected to every mesh node puts any two physical nodes
at most 2 hops apart, so global information becomes available in a few
layers. This is why the pressure network here uses 8 layers where the
purely local version needed 25.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNVirtualNodeNet(nn.Module):
    """
    GCN over the mesh graph augmented with one virtual node per graph.

    The virtual node is added inside forward, so the dataset stays the
    plain mesh graph, and it is stripped from the output again so the
    result lines up with the target.
    """

    def __init__(
        self,
        num_in,
        num_out,
        num_neurons,
        num_layers,
        dropout
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNConv(num_in, num_neurons))

        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(num_neurons, num_neurons))

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(num_neurons, num_out)

    def forward(self, graph):

        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        num_nodes = x.size(0)

        # ---------------------------------------------------------
        # How many graphs are we looking at?
        #
        # During training the loader hands us a Batch, which carries a
        # `batch` vector. At inference we get a single Data object,
        # which has no `batch` at all.
        #
        # The previous version read graph.batch and graph.num_graphs
        # BEFORE this check, so a plain Data raised AttributeError and
        # the fallback below was unreachable: inference never ran.
        # ---------------------------------------------------------

        batch = getattr(graph, "batch", None)

        if batch is None:

            batch = torch.zeros(
                num_nodes,
                dtype=torch.long,
                device=x.device
            )

            num_graphs = 1

        else:

            num_graphs = int(batch.max().item()) + 1

        # ---------------------------------------------------------
        # Add one virtual node per graph
        # ---------------------------------------------------------

        virtual_x = torch.zeros(
            num_graphs,
            x.size(1),
            device=x.device,
            dtype=x.dtype
        )

        x = torch.cat([x, virtual_x], dim=0)

        # Virtual nodes are appended after the physical ones
        virtual_indices = torch.arange(
            num_graphs,
            device=x.device
        ) + num_nodes

        # The virtual node each physical node belongs to
        node_virtual = virtual_indices[batch]

        physical_nodes = torch.arange(num_nodes, device=x.device)

        edges_to_virtual = torch.stack(
            [physical_nodes, node_virtual],
            dim=0
        )

        edges_from_virtual = torch.stack(
            [node_virtual, physical_nodes],
            dim=0
        )

        edge_index = torch.cat(
            [edge_index, edges_to_virtual, edges_from_virtual],
            dim=1
        )

        virtual_edge_weight = torch.ones(
            2 * num_nodes,
            device=edge_weight.device,
            dtype=edge_weight.dtype
        )

        edge_weight = torch.cat(
            [edge_weight, virtual_edge_weight],
            dim=0
        )

        # ---------------------------------------------------------
        # Message passing on the augmented graph
        # ---------------------------------------------------------

        for layer in self.layers:

            x = layer(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.out_layer(x)

        # Drop the virtual nodes so the output matches the target
        return x[:num_nodes]
