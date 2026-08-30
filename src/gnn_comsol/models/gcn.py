"""
The plain graph convolutional network.

This class was previously copy-pasted, identical, into three experiment
scripts. It now exists once.
"""

import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNNet(nn.Module):
    """
    A stack of GCNConv layers followed by a linear head.

    Note on depth: GCNConv has no residual connection and no
    normalization here, so very deep stacks (the pressure network once
    used 25 layers) are prone to over-smoothing, where repeated weighted
    averaging makes all node representations converge to each other.
    Adding global context with a virtual node (see virtual_node.py) is a
    cheaper way to widen the receptive field than adding depth.
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

        for layer in self.layers:

            x = layer(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)
            x = self.dropout(x)

        return self.out_layer(x)
