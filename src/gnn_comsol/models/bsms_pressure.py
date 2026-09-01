import torch.nn as nn

from .bsms_ops import BSGMP


class PressureBSMSGNN(nn.Module):
    """
    BSMS-GNN for pressure prediction.

    The neural-network parameters are independent of the mesh.

    The BSMS hierarchy is supplied at forward time, so the same model
    can operate on different geometries and meshes.
    """

    def __init__(
        self,
        num_in,
        latent_dim,
        unet_depth,
        hidden_layers,
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
        # Decoder
        # ---------------------------------------------------------

        self.decoder = nn.Linear(
            latent_dim,
            1,
        )

    def forward(
        self,
        x,
        pool_indices,
        edge_indices,
        pos,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Node features [N, F] or [B, N, F].

        pool_indices : list[torch.Tensor]
            Bi-Stride pooling indices for the current mesh.

        edge_indices : list[torch.Tensor]
            Edge connectivity at every BSMS level for the current mesh.

        pos : torch.Tensor
            Node coordinates [N, pos_dim] for the current mesh.

        Returns
        -------
        torch.Tensor
            Predicted pressure [N, 1] or [B, N, 1].
        """

        # Physical node features -> latent representation
        h = self.encoder(x)

        # Multiscale message passing using the hierarchy
        # belonging to the current mesh
        h = self.processor(
            h,
            pool_indices,
            edge_indices,
            pos,
        )

        # Latent representation -> pressure
        return self.decoder(h)