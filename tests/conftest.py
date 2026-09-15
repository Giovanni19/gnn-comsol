"""
Shared fixtures: synthetic .mat datasets.

The real dataset is produced by COMSOL and is not in the repository, so
everything here is generated. The generated files mimic the parts of the
format the code actually depends on: MATLAB axis order for X, a mesh
graph, node positions, and a non-uniform time vector.

The time vector is deliberately non-uniform: with a constant step a
misaligned delta_t is invisible, so a test built on one would pass even
with that bug in place.
"""

import h5py
import numpy as np
import pytest

from gnn_comsol.data.normalization import (
    NUM_GEOMETRY_FEATURES,
    NUM_PHYSICS_FEATURES
)


def grid_mesh(rows, cols, spacing=1.0):
    """
    A 4-neighbour grid, standing in for a mesh.

    Returns (edge_index, pos) with edge_index of shape (2, E) carrying
    both directions of every edge, and pos of shape (N, 2).
    """

    def node(r, c):
        return r * cols + c

    edges = []

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                edges.append((node(r, c), node(r + 1, c)))
            if c + 1 < cols:
                edges.append((node(r, c), node(r, c + 1)))

    edges += [(j, i) for i, j in edges]

    pos = np.array(
        [
            [c * spacing, r * spacing]
            for r in range(rows)
            for c in range(cols)
        ],
        dtype=np.float64
    )

    return np.array(edges).T, pos


def grid_cells(rows, cols):
    """
    The grid of `grid_mesh`, cut into triangles.

    Returns (Nc, 3) node indices, two triangles per square, which is
    what the COMSOL mesh gives and what the cell-centered residual is
    written on.
    """

    def node(r, c):
        return r * cols + c

    cells = []

    for r in range(rows - 1):
        for c in range(cols - 1):

            cells.append(
                [node(r, c), node(r, c + 1), node(r + 1, c)]
            )

            cells.append(
                [node(r, c + 1), node(r + 1, c + 1), node(r + 1, c)]
            )

    return np.array(cells, dtype=np.int64)


def wlsq_operators(edge_index, pos):
    """
    The WLSQ stencils and operators, as first_database.m computes them.

    For every node i, with neighbours j and w_ij = 1 / |x_j - x_i|:

        A_i = w_i * [dx_i, dy_i]        (k_i, 2)
        A_i = Q_i R_i                   economy QR
        G_i = (R_i^-1 Q_i^T) * w_i      (2, k_i)

    so that grad(phi)_i = G_i @ (phi_j - phi_i). Duplicated here rather
    than imported because the point of the tests that use it is to check
    the Python side against an independent transcription of the MATLAB.

    Returns (neighbors, G_wlsq).
    """

    num_nodes = pos.shape[0]

    stencils = [[] for _ in range(num_nodes)]

    for source, target in edge_index.T:

        if target not in stencils[source]:
            stencils[source].append(int(target))

    neighbors = []
    G_wlsq = []

    for node_i in range(num_nodes):

        neigh = np.array(sorted(stencils[node_i]), dtype=np.int64)

        delta = pos[neigh] - pos[node_i]

        weight = 1.0 / np.linalg.norm(delta, axis=1)

        A = weight[:, None] * delta

        Q, R = np.linalg.qr(A)

        G = np.linalg.inv(R) @ Q.T * weight[None, :]

        neighbors.append(neigh)
        G_wlsq.append(G)

    return neighbors, G_wlsq


def write_matlab_cell(f, key, arrays):
    """
    Write a list of arrays the way MATLAB writes a cell array to a
    v7.3 .mat: a dataset of object references, one per cell.
    """

    group = f.require_group("#refs#")

    references = []

    for index, array in enumerate(arrays):

        name = f"{key}_{index}"

        group[name] = np.asarray(array)

        references.append(group[name].ref)

    dataset = f.create_dataset(
        key,
        shape=(1, len(references)),
        dtype=h5py.special_dtype(ref=h5py.Reference)
    )

    dataset[0, :] = references


def write_dataset(
    path,
    num_snapshots=16,
    rows=3,
    cols=4,
    seed=0,
    marker_snapshots=False,
    transpose_positions=False,
    with_physics=True,
    with_geometry=True,
    with_wlsq=True,
    fluid=None
):
    """
    Write one synthetic simulation in the layout load_data expects.

    Parameters
    ----------
    marker_snapshots : bool
        Fill snapshot k with the constant value k, so a test can trace
        every sample back to the snapshots it came from. Otherwise a
        drifting field is generated, with pressure on a much larger
        scale than velocity - the ratio that used to break the target
        scaling.

    transpose_positions : bool
        Store P as (2, N) instead of (N, 2), to exercise the orientation
        handling in load_data.

    with_physics : bool
        Include the physics_features array. True by default, because
        that is what a freshly generated .mat now looks like. Pass False
        to write a file in the older layout: the six multi-geometry
        datasets predate the feature and the MATLAB generator in this
        repository still does not produce it, so a run must work without
        it and must fail clearly when a config asks for it anyway.

    with_geometry : bool
        Include the geometry_features array. True by default, for the
        same reason as with_physics. Pass False for the older,
        pre-geometry-features layout.

    with_wlsq : bool
        Include the WLSQ stencils, operators and cell connectivity the
        physics loss needs. Written in the MATLAB orientation, i.e.
        transposed, since that is how h5py hands them back. Pass False
        for a .mat from before the WLSQ export.

    fluid : dict, optional
        {"rho": ..., "mu": ...} to write as the fluid properties, as a
        first_database.m that managed to read them out of COMSOL
        would. Absent by default, which is the case the experiment
        file has to cover.

    Returns
    -------
    dict with the arrays that were written, for assertions.
    """

    rng = np.random.default_rng(seed)

    edge_index, pos = grid_mesh(rows, cols)

    num_nodes = rows * cols

    if marker_snapshots:

        X = np.stack([
            np.full((num_nodes, 3), float(k))
            for k in range(num_snapshots)
        ])

    else:

        base = rng.normal(size=(num_nodes, 3))
        base[:, 2] *= 500.0

        X = np.stack([
            base * (1 + 0.02 * k)
            + 0.01 * rng.normal(size=(num_nodes, 3))
            for k in range(num_snapshots)
        ])

    steps = 0.01 + 0.05 * np.arange(num_snapshots - 1) ** 1.3
    t = np.concatenate([[0.0], np.cumsum(steps)])

    stored_pos = pos.T if transpose_positions else pos

    cell_index = grid_cells(rows, cols)

    neighbors, G_wlsq = wlsq_operators(edge_index, pos)

    # The five physics-derived features COMSOL now exports:
    # du/dx, du/dy, dv/dx, dv/dy and div[(u . grad)u]. They are given
    # deliberately different scales, because the physics normalizer has
    # to bring them onto a common one and a test on identically scaled
    # columns would not notice if it did nothing.
    physics_features = None

    if with_physics:

        scales = np.array([1.0, 2.0, 0.5, 4.0, 100.0])

        physics_features = (
            rng.normal(
                size=(num_snapshots, num_nodes, NUM_PHYSICS_FEATURES)
            )
            * scales
        )

    # Static per-node boundary distance/direction features (wall,
    # inlet, outlet): one row per node, NOT per timestep - unlike
    # physics_features. Deliberately different scales per column, same
    # reason as physics_features.
    geometry_features = None

    if with_geometry:

        geometry_scales = np.array([1.0, 1.0, 10.0, 10.0, 0.1, 0.1])

        geometry_features = (
            rng.normal(size=(num_nodes, NUM_GEOMETRY_FEATURES))
            * geometry_scales
        )

    with h5py.File(path, "w") as f:
        # MATLAB axis order: load_data transposes (3, N, T) -> (T, N, 3)
        f["X"] = np.transpose(X, (2, 1, 0))
        f["edge_index"] = edge_index.astype(np.int64)
        f["edge_weight"] = np.ones(edge_index.shape[1])
        f["t"] = t.reshape(1, -1)
        f["h"] = np.array([[0.1]])
        f["P"] = stored_pos

        if physics_features is not None:
            # Same axis order as X: (T, N, F) -> (F, N, T)
            f["physics_features"] = np.transpose(
                physics_features, (2, 1, 0)
            )

        if geometry_features is not None:
            # Written as (N, NUM_GEOMETRY_FEATURES): load_data accepts
            # this orientation directly, no transpose needed.
            f["geometry_features"] = geometry_features

        if with_wlsq:

            # MATLAB axis order again, and the reason the corner nodes
            # of this grid matter: they have exactly two neighbours, so
            # their G_i is square and its orientation cannot be
            # recovered from its shape.
            #
            #   neighbors_python[i] : MATLAB (1, k) -> (k, 1)
            #   G_wlsq[i]           : MATLAB (2, k) -> (k, 2)
            #   cell_index          : MATLAB (Nc, 3) -> (3, Nc)
            write_matlab_cell(
                f,
                "neighbors_python",
                [neigh.reshape(-1, 1) for neigh in neighbors]
            )

            write_matlab_cell(
                f,
                "G_wlsq",
                [G.T for G in G_wlsq]
            )

            f["cell_index"] = cell_index.T

        if fluid is not None:
            f["rho"] = np.array([[float(fluid["rho"])]])
            f["mu"] = np.array([[float(fluid["mu"])]])

    return {
        "path": path,
        "X": X,
        "t": t,
        "edge_index": edge_index,
        "pos": pos,
        "physics_features": physics_features,
        "geometry_features": geometry_features,
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "num_snapshots": num_snapshots,
        "cell_index": cell_index if with_wlsq else None,
        "neighbors": neighbors if with_wlsq else None,
        "G_wlsq": G_wlsq if with_wlsq else None,
        "fluid": fluid
    }


@pytest.fixture
def make_dataset(tmp_path):
    """Factory writing synthetic .mat files into the test's tmp_path."""

    counter = {"n": 0}

    def factory(**kwargs):

        counter["n"] += 1

        path = tmp_path / f"simulation_{counter['n']}.mat"

        return write_dataset(path, **kwargs)

    return factory
