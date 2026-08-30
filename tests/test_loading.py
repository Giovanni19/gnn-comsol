"""
Reading the dataset: snapshot pairing and time-step alignment.

The time vector here is deliberately NON uniform. With a constant step
a misaligned delta_t is invisible, so a test built on one would pass
even with the bug in place.
"""

import h5py
import numpy as np
import pytest

from gnn_comsol.data.loading import load_data


NUM_SNAPSHOTS = 12
NUM_NODES = 5
NUM_EDGES = 7


@pytest.fixture
def dataset_file(tmp_path):
    """
    A .mat-like HDF5 file where snapshot k is filled with the value k,
    so every sample can be traced back to the snapshots it came from.
    """

    path = tmp_path / "dataset.mat"

    # MATLAB stores (3, N, T); load_data transposes it to (T, N, 3)
    X = np.zeros((3, NUM_NODES, NUM_SNAPSHOTS))

    for k in range(NUM_SNAPSHOTS):
        X[:, :, k] = k

    # Non-uniform, strictly increasing time
    steps = 0.01 + 0.05 * np.arange(NUM_SNAPSHOTS - 1) ** 1.3
    t = np.concatenate([[0.0], np.cumsum(steps)])

    with h5py.File(path, "w") as f:
        f["X"] = X
        f["edge_index"] = np.zeros((2, NUM_EDGES), dtype=np.int64)
        f["edge_weight"] = np.ones(NUM_EDGES)
        f["t"] = t.reshape(1, -1)      # MATLAB row vector
        f["h"] = np.array([[0.01]])

    return path, t


def test_time_vector_is_actually_non_uniform(dataset_file):
    """Guard: without this the alignment test proves nothing."""

    _, t = dataset_file

    steps = t[1:] - t[:-1]

    assert steps.std() / steps.mean() > 0.5


@pytest.mark.parametrize("skip", [0, 1, 3])
def test_pairs_are_consecutive_snapshots(dataset_file, skip):

    path, _ = dataset_file

    raw = load_data(path, skip_initial=skip)

    assert raw.num_samples == NUM_SNAPSHOTS - 1 - skip

    for i in range(raw.num_samples):
        # snapshot k was filled with the value k
        assert raw.X_input[i].min() == raw.X_input[i].max() == skip + i
        assert raw.Y_target[i].min() == raw.Y_target[i].max() == skip + i + 1


@pytest.mark.parametrize("skip", [0, 1, 3])
def test_delta_t_is_the_step_being_predicted(dataset_file, skip):
    """
    delta_t[i] must be the duration of the transition sample i has to
    advance, not the one that led into its input state.
    """

    path, t = dataset_file

    raw = load_data(path, skip_initial=skip)

    assert len(raw.delta_t) == raw.num_samples

    for i in range(raw.num_samples):

        expected = t[skip + i + 1] - t[skip + i]

        assert raw.delta_t[i] == pytest.approx(expected)


def test_the_old_misalignment_would_be_caught(dataset_file):
    """
    The previous code passed the previous step. On a non-uniform time
    vector that is a different number, so the test above discriminates.
    """

    path, t = dataset_file

    raw = load_data(path, skip_initial=1)

    old_convention = (t[1:] - t[:-1])[:raw.num_samples]

    assert not np.allclose(raw.delta_t, old_convention)


def test_state_has_three_variables(dataset_file):

    path, _ = dataset_file

    raw = load_data(path)

    assert raw.X_input.shape[1:] == (NUM_NODES, 3)
    assert raw.num_nodes == NUM_NODES
    assert raw.num_edges == NUM_EDGES


@pytest.mark.parametrize("skip", [-1, NUM_SNAPSHOTS - 1, NUM_SNAPSHOTS])
def test_invalid_skip_initial_raises(dataset_file, skip):

    path, _ = dataset_file

    with pytest.raises(ValueError):
        load_data(path, skip_initial=skip)
