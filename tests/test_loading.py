"""
Reading the dataset: snapshot pairing, time-step alignment, and the
consistency checks on the mesh.
"""

import numpy as np
import pytest

from gnn_comsol.data.loading import load_data, load_simulations


def test_time_vector_is_actually_non_uniform(make_dataset):
    """Guard: without this, the alignment test below proves nothing."""

    info = make_dataset()

    steps = info["t"][1:] - info["t"][:-1]

    assert steps.std() / steps.mean() > 0.5


@pytest.mark.parametrize("skip", [0, 1, 3])
def test_pairs_are_consecutive_snapshots(make_dataset, skip):

    info = make_dataset(marker_snapshots=True)

    raw = load_data(info["path"], skip_initial=skip)

    assert raw.num_samples == info["num_snapshots"] - 1 - skip

    for i in range(raw.num_samples):
        # snapshot k was filled with the constant value k
        assert raw.X_input[i].min() == raw.X_input[i].max() == skip + i
        assert raw.Y_target[i].min() == raw.Y_target[i].max() == skip + i + 1


@pytest.mark.parametrize("skip", [0, 1, 3])
def test_delta_t_is_the_step_being_predicted(make_dataset, skip):
    """
    delta_t[i] must be the duration of the transition sample i has to
    advance, not the one that led into its input state.
    """

    info = make_dataset()

    raw = load_data(info["path"], skip_initial=skip)

    t = info["t"]

    assert len(raw.delta_t) == raw.num_samples

    for i in range(raw.num_samples):
        expected = t[skip + i + 1] - t[skip + i]
        assert raw.delta_t[i] == pytest.approx(expected)


def test_the_old_misalignment_would_be_caught(make_dataset):
    """
    The original code passed the previous step. On a non-uniform time
    vector that is a different number, so the test above discriminates.
    """

    info = make_dataset()

    raw = load_data(info["path"], skip_initial=1)

    t = info["t"]
    old_convention = (t[1:] - t[:-1])[:raw.num_samples]

    assert not np.allclose(raw.delta_t, old_convention)


def test_state_has_three_variables(make_dataset):

    info = make_dataset()

    raw = load_data(info["path"])

    assert raw.X_input.shape[1:] == (info["num_nodes"], 3)
    assert raw.num_nodes == info["num_nodes"]
    assert raw.num_edges == info["num_edges"]


@pytest.mark.parametrize("transposed", [False, True])
def test_positions_are_returned_as_one_row_per_node(
    make_dataset, transposed
):
    """
    P may reach us as (N, 2) or (2, N) depending on how MATLAB stored
    it. Everything downstream indexes it by node, so it must always come
    out as (N, 2).
    """

    info = make_dataset(transpose_positions=transposed)

    raw = load_data(info["path"])

    assert raw.pos.shape == (info["num_nodes"], 2)
    assert np.allclose(raw.pos, info["pos"])


def test_edge_index_out_of_range_is_refused(make_dataset):
    """A 1-based export from MATLAB would land here."""

    import h5py

    info = make_dataset()

    with h5py.File(info["path"], "r+") as f:
        broken = np.array(f["edge_index"]) + 1
        del f["edge_index"]
        f["edge_index"] = broken

    with pytest.raises(ValueError, match="mesh has only"):
        load_data(info["path"])


@pytest.mark.parametrize("skip", [-1, 15, 16])
def test_invalid_skip_initial_raises(make_dataset, skip):

    info = make_dataset(num_snapshots=16)

    with pytest.raises(ValueError):
        load_data(info["path"], skip_initial=skip)


def test_load_simulations_numbers_them_in_order(make_dataset):

    infos = [make_dataset(rows=3, cols=4), make_dataset(rows=4, cols=5)]

    simulations = load_simulations(
        [info["path"] for info in infos]
    )

    assert [s.simulation_id for s in simulations] == [0, 1]

    # Meshes of different sizes must survive side by side
    assert simulations[0].num_nodes == 12
    assert simulations[1].num_nodes == 20

    for simulation, info in zip(simulations, infos):
        assert simulation.file_path == str(info["path"])


@pytest.mark.parametrize("skip", [0, 1, 3])
def test_physics_features_come_back_aligned_with_the_input(
    make_dataset, skip
):
    """
    Same axis order and same alignment as X: physics feature i must
    describe the state the network is given, not the one it predicts.
    """

    info = make_dataset(num_snapshots=12)

    raw = load_data(info["path"], skip_initial=skip)

    assert raw.physics_features.shape == (
        raw.num_samples, raw.num_nodes, 5
    )

    expected = info["physics_features"][skip:-1]

    assert np.allclose(raw.physics_features, expected)


def test_a_dataset_without_physics_features_loads_fine(make_dataset):
    """The six multi-geometry .mat files are still in this layout."""

    info = make_dataset(with_physics=False)

    raw = load_data(info["path"])

    assert raw.physics_features is None
    assert raw.num_samples == info["num_snapshots"] - 1
