import torch


def wlsq_gradient(phi, neighbors, G_wlsq):
    """
    Reconstruct the spatial gradient of a scalar nodal field
    using the precomputed WLSQ operators.

    Parameters
    ----------
    phi : torch.Tensor, shape (N,)
        Scalar field at graph nodes.

    neighbors : list[torch.Tensor]
        neighbors[i] contains the node indices belonging to
        the WLSQ stencil of node i.

    G_wlsq : list[torch.Tensor]
        G_wlsq[i] has shape (2, k_i), where k_i is the
        number of neighbors of node i.

    Returns
    -------
    grad_phi : torch.Tensor, shape (N, 2)

        grad_phi[:, 0] = dphi/dx
        grad_phi[:, 1] = dphi/dy
    """

    gradients = []

    for node_i in range(phi.shape[0]):

        neigh = neighbors[node_i]
        G_i = G_wlsq[node_i]

        # phi_j - phi_i
        delta_phi = phi[neigh] - phi[node_i]

        # [dphi/dx, dphi/dy]
        grad_i = G_i @ delta_phi

        gradients.append(grad_i)

    return torch.stack(gradients, dim=0)

def node_gradient_to_cell(grad_node, cell_index):
    """
    Interpolate nodal gradients to triangular cell centers
    using the arithmetic mean of the three vertex gradients.

    Parameters
    ----------
    grad_node : torch.Tensor, shape (N, 2)
        Spatial gradient at graph nodes.

    cell_index : torch.Tensor, shape (Nc, 3)
        Node indices of each triangular cell.

    Returns
    -------
    grad_cell : torch.Tensor, shape (Nc, 2)
        Spatial gradient at cell centers.
    """

    return grad_node[cell_index].mean(dim=1)