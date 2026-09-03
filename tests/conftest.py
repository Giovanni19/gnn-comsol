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

from gnn_comsol.data.normalization import NUM_PHYSICS_FEATURES


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


def write_dataset(
    path,
    num_snapshots=16,
    rows=3,
    cols=4,
    seed=0,
    marker_snapshots=False,
    transpose_positions=False,
    with_physics=True
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

    return {
        "path": path,
        "X": X,
        "t": t,
        "edge_index": edge_index,
        "pos": pos,
        "physics_features": physics_features,
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "num_snapshots": num_snapshots
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
