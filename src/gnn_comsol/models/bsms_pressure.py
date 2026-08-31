import torch.nn as nn

from .bsms_ops import BSGMP


class PressureBSMSGNN(nn.Module):
    """
    BSMS-GNN for pressure prediction.

    The BSMS hierarchy and mesh coordinates are fixed for the whole
    simulation and are therefore stored inside the model as buffers.

    Node positions are NOT concatenated to the input node features.
    They are used by BSGMP only to construct relative edge geometry.
    """

    def __init__(
        self,
        num_in,
        latent_dim,
        unet_depth,
        hidden_layers,
        edge_indices,
        pool_indices,
        pos,
        pos_dim=2,
    ):
        super().__init__()
        self.uses_bsms_tensor_input = True
        self.unet_depth = unet_depth

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.encoder = nn.Linear(
            num_in,
            latent_dim,
        )

        # ---------------------------------------------------------
        # BSMS processor
        # ---------------------------------------------------------

        self.processor = BSGMP(
            unet_depth=unet_depth,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            pos_dim=pos_dim,
        )

        # ---------------------------------------------------------
        # Decoder: one pressure value per node
        # ---------------------------------------------------------

        self.decoder = nn.Linear(
            latent_dim,
            1,
        )

        # ---------------------------------------------------------
        # Fixed mesh information
        # ---------------------------------------------------------

        self.register_buffer(
            "pos",
            pos,
        )

        self.num_bsms_levels = len(edge_indices)

        for level, edge_index in enumerate(edge_indices):
            self.register_buffer(
                f"edge_index_{level}",
                edge_index,
            )

        self.num_pool_levels = len(pool_indices)

        for level, indices in enumerate(pool_indices):
            self.register_buffer(
                f"pool_indices_{level}",
                indices,
            )

    def get_edge_indices(self):
        """
        Return the connectivity of all BSMS levels.
        """

        return [
            getattr(self, f"edge_index_{level}")
            for level in range(self.num_bsms_levels)
        ]

    def get_pool_indices(self):
        """
        Return the node indices retained at every pooling level.
        """

        return [
            getattr(self, f"pool_indices_{level}")
            for level in range(self.num_pool_levels)
        ]

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features.

            Single graph:
                [N, F]

            Batch of graphs sharing the same mesh:
                [B, N, F]

        Returns
        -------
        torch.Tensor
            Predicted pressure.

            Single graph:
                [N, 1]

            Batch:
                [B, N, 1]
        """

        # Physical node features -> latent representation
        h = self.encoder(x)

        # Multiscale message passing
        h = self.processor(
            h,
            self.get_pool_indices(),
            self.get_edge_indices(),
            self.pos,
        )

        # Latent representation -> pressure
        return self.decoder(h)