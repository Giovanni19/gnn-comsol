"""
The geometry of the control volumes.

Everything the momentum residual needs beyond the WLSQ gradient is here,
and all of it is derived from `pos` and `cell_index` alone - no extra
export from MATLAB. A triangle whose three vertices are known has a
known area, known edge midpoints, known outward normals and known face
lengths, and a mesh whose cells are known knows which of its edges are
on the boundary: they are the ones belonging to a single triangle.

Conventions
-----------
The control volume is the triangular CELL, while the variables live at
the vertices - the arrangement of the Gen-FVGN paper (Li et al., 2024),
where the decoded field is vertex-wise and the PDE residual is
cell-wise.

The three faces of a cell are its edges, taken in the order

    face 0 : vertex 0 -> vertex 1
    face 1 : vertex 1 -> vertex 2
    face 2 : vertex 2 -> vertex 0

and each face carries the vector `n * dl`, the outward unit normal
already multiplied by the length of the face. The two are never needed
apart, and their product is exactly the edge vector rotated by a quarter
turn, so it costs no square root and no division.
"""

from dataclasses import dataclass

import numpy as np
import torch

# Which two vertices of a cell bound each of its three faces
FACE_VERTICES = ((0, 1), (1, 2), (2, 0))


@dataclass(frozen=True)
class CellGeometry:
    """
    The cell-centered control volumes of one mesh.

    cell_index : (Nc, 3) long
        Vertices of each cell.

    area : (Nc,)
        Area of each cell, the 2D volume of the control volume.

    centroid_offset : (Nc, 3, 2)
        Centroid minus vertex, for each vertex of each cell. This is the
        `r` of the second-order interpolation to the cell center.

    face_normal : (Nc, 3, 2)
        Outward normal times face length, for each face of each cell.

    boundary_node : (N,) bool
        True where a node lies on an edge belonging to a single cell.
    """

    cell_index: torch.Tensor
    area: torch.Tensor
    centroid_offset: torch.Tensor
    face_normal: torch.Tensor
    boundary_node: torch.Tensor

    @property
    def num_cells(self):
        return self.cell_index.shape[0]

    @property
    def device(self):
        return self.area.device

    @property
    def dtype(self):
        return self.area.dtype


def build_cell_geometry(pos, cell_index, device=None, dtype=torch.float32):
    """
    Areas, centroid offsets, outward face normals and boundary nodes.

    Parameters
    ----------
    pos : array-like, shape (N, 2)
        Node coordinates.

    cell_index : array-like, shape (Nc, 3)
        Zero-based node indices of each triangle.

    Returns
    -------
    CellGeometry

    Notes
    -----
    The triangles are NOT assumed to be wound consistently. The sign of
    the cross product of two edges says which way each one turns, and
    the normals are flipped accordingly, so a mesh with mixed winding -
    which nothing upstream promises not to produce - still gets normals
    that point out of their own cell rather than into it.
    """

    pos = np.asarray(pos, dtype=np.float64)
    cell_index = np.asarray(cell_index, dtype=np.int64)

    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(
            f"pos has shape {pos.shape}, expected (N, 2)."
        )

    if cell_index.ndim != 2 or cell_index.shape[1] != 3:
        raise ValueError(
            f"cell_index has shape {cell_index.shape}, "
            "expected (Nc, 3): the control volumes are triangles."
        )

    corners = pos[cell_index]

    edge_01 = corners[:, 1] - corners[:, 0]
    edge_02 = corners[:, 2] - corners[:, 0]

    # Twice the signed area; its sign is the winding of the triangle
    signed = (
        edge_01[:, 0] * edge_02[:, 1]
        - edge_01[:, 1] * edge_02[:, 0]
    )

    degenerate = np.abs(signed) <= 0.0

    if np.any(degenerate):
        raise ValueError(
            f"{int(degenerate.sum())} cell(s) have zero area, "
            f"the first being cell {int(np.argmax(degenerate))}. "
            "A control volume cannot be a segment."
        )

    area = np.abs(signed) / 2.0

    orientation = np.sign(signed)[:, None]

    centroid = corners.mean(axis=1)

    centroid_offset = centroid[:, None, :] - corners

    face_normal = np.empty_like(corners)

    for face, (start, end) in enumerate(FACE_VERTICES):

        edge = corners[:, end] - corners[:, start]

        # The edge vector turned a quarter turn is already the outward
        # normal times the face length, for a counter-clockwise cell.
        face_normal[:, face] = orientation * np.stack(
            (edge[:, 1], -edge[:, 0]),
            axis=1,
        )

    return CellGeometry(
        cell_index=torch.as_tensor(
            cell_index,
            dtype=torch.long,
            device=device,
        ),
        area=torch.as_tensor(area, dtype=dtype, device=device),
        centroid_offset=torch.as_tensor(
            centroid_offset,
            dtype=dtype,
            device=device,
        ),
        face_normal=torch.as_tensor(
            face_normal,
            dtype=dtype,
            device=device,
        ),
        boundary_node=torch.as_tensor(
            find_boundary_nodes(cell_index, pos.shape[0]),
            device=device,
        ),
    )


def find_boundary_nodes(cell_index, num_nodes):
    """
    Nodes on the boundary of the meshed domain.

    An interior edge is shared by two triangles and a boundary edge by
    one, which is the whole test: no geometric tolerance, no knowledge
    of what the boundary means physically, and no dependence on the
    inlet/wall/outlet features COMSOL exports separately.

    Returns a (num_nodes,) boolean array.
    """

    cell_index = np.asarray(cell_index, dtype=np.int64)

    edges = np.concatenate(
        [
            cell_index[:, [start, end]]
            for start, end in FACE_VERTICES
        ]
    )

    # Undirected: an edge is the same edge whichever way it is walked
    edges = np.sort(edges, axis=1)

    unique, counts = np.unique(edges, axis=0, return_counts=True)

    boundary = np.zeros(num_nodes, dtype=bool)

    boundary[unique[counts == 1].reshape(-1)] = True

    return boundary


def interpolate_to_cell(values, gradients, geometry):
    """
    Vertex values -> cell center, to second order.

    This is Eq. (A.3) of the paper,

        phi_c = (1/3) sum_v (phi_v + r . grad phi_v)

    with r the vector from the vertex to the centroid. The plain mean of
    the three vertex values would be first order; the gradient is
    already reconstructed for the residual, so the correction is free.

    Parameters
    ----------
    values : torch.Tensor, shape (N, C)
    gradients : torch.Tensor, shape (N, C, 2)
    geometry : CellGeometry

    Returns
    -------
    torch.Tensor, shape (Nc, C)
    """

    at_corners = values[geometry.cell_index]

    gradient_at_corners = gradients[geometry.cell_index]

    # (Nc, 3, C, 2) . (Nc, 3, 1, 2) summed over the last axis
    correction = (
        gradient_at_corners
        * geometry.centroid_offset.unsqueeze(2)
    ).sum(dim=-1)

    return (at_corners + correction).mean(dim=1)


def divergence_over_faces(gradients, geometry):
    """
    The surface integral of a gradient over the faces of each cell,

        sum_f (grad phi)_f . n dl

    which by the divergence theorem is the volume integral of the
    Laplacian over the control volume. The gradient at a face is the
    arithmetic mean of the gradients at the two vertices bounding it -
    for a cell-centered control volume the face center IS the midpoint
    of the edge, so that mean is second-order accurate with no
    non-orthogonality correction, which is the reason the paper places
    the variables at the vertices in the first place.

    Taking the divergence this way, rather than reconstructing the
    gradient a second time, keeps the whole residual to a single WLSQ
    pass.

    Parameters
    ----------
    gradients : torch.Tensor, shape (N, C, 2)
    geometry : CellGeometry

    Returns
    -------
    torch.Tensor, shape (Nc, C)
    """

    total = None

    for face, (start, end) in enumerate(FACE_VERTICES):

        gradient_face = 0.5 * (
            gradients[geometry.cell_index[:, start]]
            + gradients[geometry.cell_index[:, end]]
        )

        # (Nc, C, 2) . (Nc, 1, 2) summed over the last axis
        flux = (
            gradient_face
            * geometry.face_normal[:, face].unsqueeze(1)
        ).sum(dim=-1)

        total = flux if total is None else total + flux

    return total
