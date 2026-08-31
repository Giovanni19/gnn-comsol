import torch
import torch.nn as nn

from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.utils import degree, scatter


def scatter_sum(src, index, dim=-1, dim_size=None):
    return scatter(
        src,
        index,
        dim=dim,
        dim_size=dim_size,
        reduce="sum",
    )


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        output_dim,
        hidden_layers,
        layer_normalized=True,
    ):
        super().__init__()

        modules = []

        for layer in range(hidden_layers):

            if layer == 0:
                modules.append(
                    Linear(input_dim, latent_dim)
                )
            else:
                modules.append(
                    Linear(latent_dim, latent_dim)
                )

            modules.append(ReLU())

        modules.append(
            Linear(latent_dim, output_dim)
        )

        if layer_normalized:
            modules.append(
                LayerNorm(
                    output_dim,
                    elementwise_affine=False,
                )
            )

        self.seq = Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class GMP(nn.Module):

    def __init__(
        self,
        latent_dim,
        hidden_layers,
        pos_dim,
    ):
        super().__init__()

        self.mlp_node = MLP(
            2 * latent_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
        )

        edge_input_dim = (
            2 * latent_dim
            + pos_dim
            + 1
        )

        self.mlp_edge = MLP(
            edge_input_dim,
            latent_dim,
            latent_dim,
            hidden_layers,
        )

    def forward(self, x, edge_index, pos):

        i, j = edge_index[0], edge_index[1]

        if x.dim() == 3:
            batch_size = x.shape[0]
            x_i = x[:, i]
            x_j = x[:, j]

        elif x.dim() == 2:
            x_i = x[i]
            x_j = x[j]

        else:
            raise ValueError(
                "x must have shape [N, C] or [B, N, C]."
            )

        if pos.dim() == 3:
            pos_i = pos[:, i]
            pos_j = pos[:, j]

        elif pos.dim() == 2:
            pos_i = pos[i]
            pos_j = pos[j]

        else:
            raise ValueError(
                "pos must have shape [N, D] or [B, N, D]."
            )

        direction = pos_i - pos_j

        distance = torch.norm(
            direction,
            dim=-1,
            keepdim=True,
        )

        geometry = torch.cat(
            [direction, distance],
            dim=-1,
        )

        if x.dim() == 3 and pos.dim() == 2:
            geometry = geometry.unsqueeze(0).repeat(
                batch_size, 1, 1
            )

        edge_features = torch.cat(
            [
                geometry,
                x_i,
                x_j,
            ],
            dim=-1,
        )

        messages = self.mlp_edge(
            edge_features
        )

        aggregated = scatter_sum(
            messages,
            j,
            dim=-2,
            dim_size=x.shape[-2],
        )

        node_features = torch.cat(
            [x, aggregated],
            dim=-1,
        )

        return self.mlp_node(node_features) + x

class WeightedEdgeConv(nn.Module):

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
        aggregating=True,
    ):

        i, j = edge_index[0], edge_index[1]

        if x.dim() == 3:

            weighted_info = (
                x[:, i]
                if aggregating
                else x[:, j]
            )

        elif x.dim() == 2:

            weighted_info = (
                x[i]
                if aggregating
                else x[j]
            )

        else:
            raise ValueError(
                "x must have shape [N, C] or [B, N, C]."
            )

        weighted_info = (
            weighted_info
            * edge_weight.unsqueeze(-1)
        )

        target_index = (
            j if aggregating else i
        )

        return scatter_sum(
            weighted_info,
            target_index,
            dim=-2,
            dim_size=x.shape[-2],
        )

    @torch.no_grad()
    def calculate_edge_weights(
        self,
        node_weight,
        edge_index,
    ):

        i, j = edge_index[0], edge_index[1]

        deg = degree(
            i,
            dtype=torch.float,
            num_nodes=node_weight.shape[0],
        )

        normalized_weight = (
            node_weight.squeeze(-1) / deg
        )

        weight_to_send = normalized_weight[i]

        eps = 1e-12

        aggregated_weight = (
            scatter_sum(
                weight_to_send,
                j,
                dim=-1,
                dim_size=normalized_weight.size(0),
            )
            + eps
        )

        edge_weight = (
            weight_to_send
            / aggregated_weight[j]
        )

        return edge_weight, aggregated_weight

class Unpool(nn.Module):

    def forward(
        self,
        h,
        previous_num_nodes,
        indices,
    ):

        if h.dim() == 2:

            new_h = h.new_zeros(
                previous_num_nodes,
                h.shape[-1],
            )

            new_h[indices] = h

        elif h.dim() == 3:

            new_h = h.new_zeros(
                h.shape[0],
                previous_num_nodes,
                h.shape[-1],
            )

            new_h[:, indices] = h

        else:
            raise ValueError(
                "h must have shape [N, C] or [B, N, C]."
            )

        return new_h


class BSGMP(nn.Module):
    """
    Bi-Stride Graph Message Passing processor.

    Performs message passing while moving from the fine graph to
    progressively coarser BSMS graphs and then back to the fine graph.

    The module preserves the number of nodes:
        input  -> [N, latent_dim]
        output -> [N, latent_dim]
    """

    def __init__(
        self,
        unet_depth,
        latent_dim,
        hidden_layers,
        pos_dim,
    ):
        super().__init__()

        self.unet_depth = unet_depth

        # Message passing at the coarsest level
        self.bottom_gmp = GMP(
            latent_dim,
            hidden_layers,
            pos_dim,
        )

        # Message passing during fine -> coarse
        self.down_gmps = nn.ModuleList()

        # Message passing during coarse -> fine
        self.up_gmps = nn.ModuleList()

        self.unpools = nn.ModuleList()

        self.edge_conv = WeightedEdgeConv()

        for _ in range(unet_depth):

            self.down_gmps.append(
                GMP(
                    latent_dim,
                    hidden_layers,
                    pos_dim,
                )
            )

            self.up_gmps.append(
                GMP(
                    latent_dim,
                    hidden_layers,
                    pos_dim,
                )
            )

            self.unpools.append(
                Unpool()
            )

    def forward(
        self,
        h,
        pool_indices,
        edge_indices,
        pos,
    ):
        """
        Parameters
        ----------
        h : Tensor
            Latent node features [N, latent_dim]
            or [B, N, latent_dim].

        pool_indices : list[Tensor]
            Nodes retained at every BSMS pooling operation.

        edge_indices : list[Tensor]
            Connectivity at every BSMS level.

            For unet_depth=2:
                edge_indices[0] -> G0
                edge_indices[1] -> G1
                edge_indices[2] -> G2

        pos : Tensor
            Node coordinates [N, pos_dim].
            Coordinates are used by GMP only through relative
            edge geometry.

        Returns
        -------
        Tensor
            Updated latent features on the original fine graph.
        """

        down_outputs = []
        down_positions = []
        transition_weights = []

        # Initial node weights used by the BSMS transition operator
        node_weight = pos.new_ones(
            (pos.shape[-2], 1)
        )

        # --------------------------------------------------
        # DOWN PASS
        # --------------------------------------------------

        for level in range(self.unet_depth):

            edge_index = edge_indices[level]

            # Message passing at current resolution
            h = self.down_gmps[level](
                h,
                edge_index,
                pos,
            )

            # Save information for the skip connection
            down_outputs.append(h)
            down_positions.append(pos)

            # Compute BSMS transition weights
            edge_weight, node_weight = (
                self.edge_conv.calculate_edge_weights(
                    node_weight,
                    edge_index,
                )
            )

            # Aggregate information before removing nodes
            h = self.edge_conv(
                h,
                edge_index,
                edge_weight,
                aggregating=True,
            )

            pos = self.edge_conv(
                pos,
                edge_index,
                edge_weight,
                aggregating=True,
            )

            transition_weights.append(
                edge_weight
            )

            # Nodes retained in the next coarse graph
            indices = pool_indices[level]

            if h.dim() == 3:
                h = h[:, indices]
            else:
                h = h[indices]

            if pos.dim() == 3:
                pos = pos[:, indices]
            else:
                pos = pos[indices]

            node_weight = node_weight[indices]

        # --------------------------------------------------
        # BOTTOM
        # --------------------------------------------------

        h = self.bottom_gmp(
            h,
            edge_indices[self.unet_depth],
            pos,
        )

        # --------------------------------------------------
        # UP PASS
        # --------------------------------------------------

        for level in range(self.unet_depth):

            depth_index = (
                self.unet_depth - level - 1
            )

            edge_index = edge_indices[depth_index]
            indices = pool_indices[depth_index]

            # Restore nodes removed during pooling
            h = self.unpools[level](
                h,
                down_outputs[depth_index].shape[-2],
                indices,
            )

            # Propagate coarse information back to fine nodes
            h = self.edge_conv(
                h,
                edge_index,
                transition_weights[depth_index],
                aggregating=False,
            )

            # Message passing at the restored resolution
            h = self.up_gmps[level](
                h,
                edge_index,
                down_positions[depth_index],
            )

            # U-Net skip connection
            h = (
                h
                + down_outputs[depth_index]
            )

        return h