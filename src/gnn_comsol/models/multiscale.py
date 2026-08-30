"""
Multiscale (U-Net style) network for pressure.

STATUS: architecture only, not yet wired into an experiment.

It solves the same non-locality problem as the virtual node, but with a
coarse graph instead of a single hub: fine encoder, pooling onto
clusters, message passing on the coarse graph, unpooling, skip
connection, fine decoder. A few hops on the coarse graph cover a long
distance on the fine mesh.

What is still missing before it can be trained
----------------------------------------------
1. forward() takes `cluster` and `coarse_edge_index` on top of the graph,
   so it does not match the net(batch) call used by the training loop.
   It needs either a wrapper that carries those tensors on the Data
   object, or a dedicated training loop.

2. Batching. With more than one graph per batch, `cluster` and
   `coarse_edge_index` must be offset per graph, the way PyG does for
   edge_index via __inc__.

The clustering itself is available: see gnn_comsol.graph.clustering.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class MultiscalePressureGNN(nn.Module):

    def __init__(
        self,
        num_in,
        num_neurons,
        num_fine_layers,
        num_coarse_layers,
        num_decoder_layers,
        dropout
    ):

        super().__init__()

        # Fine encoder
        self.fine_encoder = nn.ModuleList()
        self.fine_encoder.append(GCNConv(num_in, num_neurons))

        for _ in range(num_fine_layers - 1):
            self.fine_encoder.append(GCNConv(num_neurons, num_neurons))

        # Coarse graph
        self.coarse_layers = nn.ModuleList()

        for _ in range(num_coarse_layers):
            self.coarse_layers.append(GCNConv(num_neurons, num_neurons))

        # Decoder: input is [fine features, unpooled coarse features]
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(GCNConv(2 * num_neurons, num_neurons))

        for _ in range(num_decoder_layers - 1):
            self.decoder_layers.append(GCNConv(num_neurons, num_neurons))

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(num_neurons, 1)

    def forward(
        self,
        graph,
        cluster,
        coarse_edge_index,
        coarse_edge_weight=None
    ):

        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight

        # 1. Fine message passing
        for layer in self.fine_encoder:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = self.dropout(x)

        x_fine = x

        # 2. Pool fine -> coarse (mean over each cluster)
        num_clusters = int(cluster.max().item()) + 1

        x_coarse = torch.zeros(
            num_clusters,
            x_fine.size(1),
            dtype=x_fine.dtype,
            device=x_fine.device
        )

        x_coarse.index_add_(0, cluster, x_fine)

        cluster_count = torch.zeros(
            num_clusters,
            dtype=x_fine.dtype,
            device=x_fine.device
        )

        cluster_count.index_add_(
            0,
            cluster,
            torch.ones(
                cluster.size(0),
                dtype=x_fine.dtype,
                device=x_fine.device
            )
        )

        x_coarse = x_coarse / cluster_count.unsqueeze(1).clamp(min=1)

        # 3. Coarse message passing
        for layer in self.coarse_layers:
            x_coarse = layer(
                x_coarse,
                coarse_edge_index,
                edge_weight=coarse_edge_weight
            )
            x_coarse = self.activation(x_coarse)
            x_coarse = self.dropout(x_coarse)

        # 4. Unpool coarse -> fine
        x_unpooled = x_coarse[cluster]

        # 5. Skip connection: keep local detail, add global context
        x = torch.cat([x_fine, x_unpooled], dim=1)

        # 6. Fine decoder
        for layer in self.decoder_layers:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = self.dropout(x)

        # 7. Pressure
        return self.out_layer(x)
