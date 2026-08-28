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


        # ============================================================
        # FINE ENCODER
        # ============================================================

        self.fine_encoder = nn.ModuleList()

        self.fine_encoder.append(
            GCNConv(
                num_in,
                num_neurons
            )
        )

        for _ in range(num_fine_layers - 1):

            self.fine_encoder.append(
                GCNConv(
                    num_neurons,
                    num_neurons
                )
            )


        # ============================================================
        # COARSE GNN
        # ============================================================

        self.coarse_layers = nn.ModuleList()

        for _ in range(num_coarse_layers):

            self.coarse_layers.append(
                GCNConv(
                    num_neurons,
                    num_neurons
                )
            )


        # ============================================================
        # DECODER
        #
        # After unpooling:
        #
        # [fine_features, coarse_features]
        #
        # therefore input dimension = 2 * num_neurons
        # ============================================================

        self.decoder_layers = nn.ModuleList()

        self.decoder_layers.append(
            GCNConv(
                2 * num_neurons,
                num_neurons
            )
        )

        for _ in range(num_decoder_layers - 1):

            self.decoder_layers.append(
                GCNConv(
                    num_neurons,
                    num_neurons
                )
            )


        # ============================================================
        # ACTIVATION + DROPOUT
        # ============================================================

        self.activation = nn.LeakyReLU()

        self.dropout = nn.Dropout(
            dropout
        )


        # ============================================================
        # OUTPUT
        # ============================================================

        self.out_layer = nn.Linear(
            num_neurons,
            1
        )


    def forward(
        self,
        graph,
        cluster,
        coarse_edge_index,
        coarse_edge_weight=None
    ):

        # ============================================================
        # ORIGINAL FINE GRAPH
        # ============================================================

        x = graph.x

        edge_index = graph.edge_index

        edge_weight = graph.edge_weight


        # ============================================================
        # 1. FINE MESSAGE PASSING
        # ============================================================

        for layer in self.fine_encoder:

            x = layer(
                x,
                edge_index,
                edge_weight=edge_weight
            )

            x = self.activation(x)

            x = self.dropout(x)


        # Save fine representation for skip connection
        x_fine = x


        # ============================================================
        # 2. POOL FINE -> COARSE
        #
        # Mean of all nodes belonging to the same cluster.
        # ============================================================

        num_clusters = (
            int(cluster.max().item()) + 1
        )

        x_coarse = torch.zeros(
            num_clusters,
            x_fine.size(1),
            dtype=x_fine.dtype,
            device=x_fine.device
        )

        x_coarse.index_add_(
            0,
            cluster,
            x_fine
        )


        # Number of fine nodes inside each cluster
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


        # Mean pooling
        x_coarse = (
            x_coarse
            / cluster_count.unsqueeze(1).clamp(min=1)
        )


        # ============================================================
        # 3. COARSE MESSAGE PASSING
        # ============================================================

        for layer in self.coarse_layers:

            x_coarse = layer(
                x_coarse,
                coarse_edge_index,
                edge_weight=coarse_edge_weight
            )

            x_coarse = self.activation(
                x_coarse
            )

            x_coarse = self.dropout(
                x_coarse
            )


        # ============================================================
        # 4. UNPOOL COARSE -> FINE
        #
        # Every fine node receives the representation
        # of its corresponding cluster.
        # ============================================================

        x_unpooled = x_coarse[
            cluster
        ]


        # ============================================================
        # 5. SKIP CONNECTION
        #
        # Preserve local fine information while adding
        # large-scale information from the coarse graph.
        # ============================================================

        x = torch.cat(
            [
                x_fine,
                x_unpooled
            ],
            dim=1
        )


        # ============================================================
        # 6. FINE DECODER
        # ============================================================

        for layer in self.decoder_layers:

            x = layer(
                x,
                edge_index,
                edge_weight=edge_weight
            )

            x = self.activation(x)

            x = self.dropout(x)


        # ============================================================
        # 7. PRESSURE PREDICTION
        # ============================================================

        x = self.out_layer(
            x
        )

        return x


def build_coarse_graph(
    edge_index,
    cluster
):

    # Fine graph source and destination nodes
    source = edge_index[0]
    target = edge_index[1]

    # Convert fine nodes into cluster IDs
    source_cluster = cluster[source]
    target_cluster = cluster[target]


    # ============================================================
    # Remove edges inside the same cluster
    # ============================================================

    mask = (
        source_cluster
        != target_cluster
    )

    source_cluster = source_cluster[
        mask
    ]

    target_cluster = target_cluster[
        mask
    ]


    # ============================================================
    # Create coarse edges
    # ============================================================

    coarse_edge_index = torch.stack(
        [
            source_cluster,
            target_cluster
        ],
        dim=0
    )


    # ============================================================
    # Remove duplicated edges
    # ============================================================

    coarse_edge_index = torch.unique(
        coarse_edge_index,
        dim=1
    )


    return coarse_edge_index

