"""
Gradient reconstruction at the mesh nodes.

The gradient of a nodal scalar field is reconstructed with the Weighted
Least Squares method (WLSQ), using the per-node operators precomputed by
the MATLAB exporter (`first_database.m`): for every node i,

    grad(phi)_i = G_i @ (phi[neighbors_i] - phi_i)

with G_i of shape (2, k_i) and k_i the size of the stencil of node i.
This is Eq. (8) of the Gen-FVGN paper, with the inverse-distance weight
w_ij = 1 / |x_j - x_i| folded into G_i on the MATLAB side.

Two implementations of exactly that formula live here.

`wlsq_gradient_reference` is the literal transcription: one small
matrix-vector product per node. It is the definition, and the tests
check the fast path against it - but a Python loop over every node of
the mesh is far too slow to sit inside a training step, where it runs
once per field, per graph, per batch, per epoch.

`build_wlsq_operators` assembles the same formula once per mesh into two
sparse N x N matrices. The reconstruction is LINEAR in phi, so expanding
the difference

    grad_i = sum_j G_i[:, j] phi_{n_j} - (sum_j G_i[:, j]) phi_i

gives, row by row, the entries of matrices Dx and Dy with

    dphi/dx = Dx @ phi        dphi/dy = Dy @ phi

Same arithmetic, same result, two sparse products instead of 2N dense
ones. `sparse @ dense` is differentiable with respect to the dense
operand, so the physics loss still backpropagates into the network.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class WLSQOperators:
    """
    The WLSQ gradient of one mesh, as two sparse matrices.

    dx, dy : torch.Tensor
        Sparse (N, N) matrices such that dx @ phi is dphi/dx and
        dy @ phi is dphi/dy at the mesh nodes.
    """

    dx: torch.Tensor
    dy: torch.Tensor

    @property
    def num_nodes(self):
        return self.dx.shape[0]

    @property
    def device(self):
        return self.dx.device

    @property
    def dtype(self):
        return self.dx.dtype

    def to(self, device=None, dtype=None):
        """A copy on another device or in another dtype."""

        if device is None:
            device = self.device

        if dtype is None:
            dtype = self.dtype

        if device == self.device and dtype == self.dtype:
            return self

        return WLSQOperators(
            dx=self.dx.to(device=device, dtype=dtype),
            dy=self.dy.to(device=device, dtype=dtype),
        )


def build_wlsq_operators(
    neighbors,
    G_wlsq,
    num_nodes=None,
    device=None,
    dtype=torch.float32,
):
    """
    Assemble the per-node WLSQ operators into two sparse matrices.

    Parameters
    ----------
    neighbors : list[array-like]
        neighbors[i] holds the node indices in the WLSQ stencil of
        node i, as exported by MATLAB (already zero-based).

    G_wlsq : list[array-like]
        G_wlsq[i] has shape (2, k_i), row 0 for d/dx and row 1 for d/dy.

    num_nodes : int, optional
        Defaults to len(neighbors).

    Returns
    -------
    WLSQOperators

    Notes
    -----
    The matrices are built in float64 and cast at the end: the entries
    of G_i are differences of inverse distances and can span several
    orders of magnitude on a graded mesh, and the diagonal entry is a
    sum of the whole row, which is where that cancellation would bite.
    """

    if len(neighbors) != len(G_wlsq):
        raise ValueError(
            f"neighbors has {len(neighbors)} entries but G_wlsq has "
            f"{len(G_wlsq)}; they must describe the same nodes."
        )

    if num_nodes is None:
        num_nodes = len(neighbors)

    if len(neighbors) != num_nodes:
        raise ValueError(
            f"neighbors has {len(neighbors)} entries, "
            f"but the mesh has {num_nodes} nodes."
        )

    rows = []
    columns = []
    values_x = []
    values_y = []

    for node_i, (neigh, G_i) in enumerate(zip(neighbors, G_wlsq)):

        neigh = np.asarray(neigh, dtype=np.int64).reshape(-1)
        G_i = np.asarray(G_i, dtype=np.float64)

        if G_i.shape != (2, neigh.size):
            raise ValueError(
                f"G_wlsq[{node_i}] has shape {G_i.shape}, expected "
                f"(2, {neigh.size}) for its {neigh.size} neighbors."
            )

        if np.any(neigh < 0) or np.any(neigh >= num_nodes):
            raise ValueError(
                f"WLSQ stencil of node {node_i} references a node "
                f"outside the mesh ({num_nodes} nodes)."
            )

        # phi_j terms
        rows.append(np.full(neigh.size, node_i, dtype=np.int64))
        columns.append(neigh)
        values_x.append(G_i[0])
        values_y.append(G_i[1])

        # -phi_i term, gathered from the whole stencil
        rows.append(np.array([node_i], dtype=np.int64))
        columns.append(np.array([node_i], dtype=np.int64))
        values_x.append(np.array([-G_i[0].sum()]))
        values_y.append(np.array([-G_i[1].sum()]))

    indices = torch.from_numpy(
        np.stack(
            (
                np.concatenate(rows),
                np.concatenate(columns),
            )
        )
    )

    def sparse(values):

        matrix = torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(np.concatenate(values)),
            size=(num_nodes, num_nodes),

            # Once per mesh, against a segfault on a malformed sparse
            # tensor: cheaper than the first training step it precedes.
            check_invariants=True,
        )

        # coalesce() sums duplicate (row, column) entries, which is
        # exactly what is wanted if a node ever appears twice in its own
        # stencil, and is required before the matrix can be multiplied.
        return matrix.coalesce().to(device=device, dtype=dtype)

    return WLSQOperators(
        dx=sparse(values_x),
        dy=sparse(values_y),
    )


def wlsq_gradient(values, operators):
    """
    Spatial gradient of one or more nodal fields.

    Parameters
    ----------
    values : torch.Tensor, shape (N,) or (N, C)
        One scalar field, or C fields side by side. Passing u and v as
        one (N, 2) tensor costs the same two sparse products as passing
        a single field, so prefer it over two calls.

    operators : WLSQOperators

    Returns
    -------
    torch.Tensor
        (N, 2) for a single field, with column 0 = dphi/dx and
        column 1 = dphi/dy.

        (N, C, 2) for C fields, indexed [node, field, direction].
    """

    if values.shape[0] != operators.num_nodes:
        raise ValueError(
            f"values has {values.shape[0]} rows but the WLSQ operators "
            f"were built for {operators.num_nodes} nodes."
        )

    single_field = values.dim() == 1

    matrix = values.unsqueeze(1) if single_field else values

    d_dx = torch.sparse.mm(operators.dx, matrix)
    d_dy = torch.sparse.mm(operators.dy, matrix)

    gradient = torch.stack((d_dx, d_dy), dim=-1)

    return gradient.squeeze(1) if single_field else gradient


def wlsq_gradient_reference(phi, neighbors, G_wlsq):
    """
    The definition of the WLSQ gradient, one node at a time.

    Kept as the reference the fast path is tested against, and as the
    readable statement of what `build_wlsq_operators` encodes. Too slow
    for training: use `wlsq_gradient` there.

    Parameters
    ----------
    phi : torch.Tensor, shape (N,)

    neighbors : list[torch.Tensor]

    G_wlsq : list[torch.Tensor]
        G_wlsq[i] has shape (2, k_i).

    Returns
    -------
    grad_phi : torch.Tensor, shape (N, 2)
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

    This is the interpolation of Appendix A of the Gen-FVGN paper:
    the variables are decoded at the vertices, but the PDE residual is
    written on the cell-centered control volumes.

    Parameters
    ----------
    grad_node : torch.Tensor, shape (N, 2) or (N, C, 2)
        Spatial gradient at graph nodes.

    cell_index : torch.Tensor, shape (Nc, 3)
        Node indices of each triangular cell.

    Returns
    -------
    grad_cell : torch.Tensor, shape (Nc, 2) or (Nc, C, 2)
        Spatial gradient at cell centers.
    """

    return grad_node[cell_index].mean(dim=1)
