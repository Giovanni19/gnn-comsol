"""Splitting: contiguity, leakage, and index alignment."""

import numpy as np
import pytest

from gnn_comsol.data.loading import RawDataset
from gnn_comsol.data.splitting import (
    compute_split_indices,
    split_dataset,
    split_simulations
)


def snapshots_touched(indices):
    """Sample i is the pair X[i+1] -> X[i+2]."""
    touched = set()
    for i in indices:
        touched.add(int(i) + 1)
        touched.add(int(i) + 2)
    return touched


def test_temporal_blocks_are_contiguous_and_ordered():

    train, val, test = compute_split_indices(200, mode="temporal")

    assert list(train) == list(range(0, 140))
    assert list(val) == list(range(141, 171))
    assert list(test) == list(range(172, 200))

    assert train.max() < val.min() < test.min()


def test_gap_removes_the_shared_snapshot():

    train, val, test = compute_split_indices(200, mode="temporal", gap=1)

    assert not snapshots_touched(train) & snapshots_touched(val)
    assert not snapshots_touched(val) & snapshots_touched(test)

    # Without the gap, exactly one snapshot is shared
    train0, val0, _ = compute_split_indices(200, mode="temporal", gap=0)

    overlap = snapshots_touched(train0) & snapshots_touched(val0)

    assert overlap == {141}


def test_random_mode_reproduces_the_legacy_split():

    rng = np.random.default_rng(68)
    expected = np.arange(200)
    rng.shuffle(expected)

    train, val, test = compute_split_indices(200, mode="random")

    assert list(train) == list(expected[:140])
    assert list(val) == list(expected[140:170])
    assert list(test) == list(expected[170:])


def test_random_split_leaks_and_temporal_does_not():
    """The reason the default changed, as a number."""

    train_t, _, test_t = compute_split_indices(200, mode="temporal")
    train_r, _, test_r = compute_split_indices(200, mode="random")

    def nearest(test_idx, train_idx):
        return min(
            abs(int(i) - int(j)) for i in test_idx for j in train_idx
        )

    assert nearest(test_r, train_r) == 1
    assert nearest(test_t, train_t) > 10


def test_group_mode_keeps_simulations_whole():

    groups = np.repeat(np.arange(10), 50)

    train, val, test = compute_split_indices(
        len(groups), mode="group", groups=groups
    )

    g_train = set(groups[train])
    g_val = set(groups[val])
    g_test = set(groups[test])

    assert not g_train & g_val
    assert not g_val & g_test
    assert not g_train & g_test

    assert len(train) + len(val) + len(test) == len(groups)
    assert g_val and g_test


def test_group_mode_with_the_minimum_number_of_simulations():

    groups = np.repeat(np.arange(3), 10)

    train, val, test = compute_split_indices(
        len(groups), mode="group", groups=groups
    )

    assert len(train) and len(val) and len(test)


def test_split_dataset_keeps_x_y_and_dt_aligned():

    T, N = 120, 4

    X = np.arange(T * N * 3, dtype=float).reshape(T, N, 3)
    t = np.cumsum(np.full(T, 0.1))

    raw = RawDataset(
        X_input=X[1:-1],
        Y_target=X[2:],
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=(t[1:] - t[:-1])[1:],
        h=np.array(0.0),
        pos=np.zeros((N, 2))
    )

    # delta_t lines up with X_input index by index
    assert len(raw.delta_t) == raw.num_samples

    splits = split_dataset(raw, mode="temporal")

    train_idx, _, _ = compute_split_indices(
        raw.num_samples, mode="temporal"
    )

    assert np.array_equal(splits.train.X, raw.X_input[train_idx])
    assert np.array_equal(splits.train.Y, raw.Y_target[train_idx])
    assert len(splits.train.X) == len(splits.train.dt)


def test_mismatched_delta_t_is_refused():
    """
    A silent truncation here would shift the time step of every sample,
    so the length mismatch must fail loudly.
    """

    X = np.zeros((10, 2, 3))

    raw = RawDataset(
        X_input=X[:-1],
        Y_target=X[1:],
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.ones(5),          # wrong length on purpose
        h=np.array(0.0),
        pos=np.zeros((2, 2))
    )

    with pytest.raises(ValueError, match="line up"):
        split_dataset(raw, mode="temporal")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_samples=5, mode="temporal",
             train_fraction=0.99, val_fraction=0.005),
        dict(n_samples=10, mode="group"),
        dict(n_samples=10, mode="not_a_mode"),
    ],
)
def test_bad_arguments_raise_clearly(kwargs):

    n_samples = kwargs.pop("n_samples")

    with pytest.raises(ValueError):
        compute_split_indices(n_samples, **kwargs)


# ---------------------------------------------------------------------
# Splitting whole simulations
# ---------------------------------------------------------------------

def fake_simulation(simulation_id, num_samples=5, num_nodes=4):
    """A RawDataset with just enough shape to be split."""

    return RawDataset(
        X_input=np.zeros((num_samples, num_nodes, 3)),
        Y_target=np.zeros((num_samples, num_nodes, 3)),
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.ones(num_samples),
        h=np.array(0.0),
        pos=np.zeros((num_nodes, 2)),
        simulation_id=simulation_id
    )


def test_split_simulations_keeps_every_simulation_whole():

    simulations = [fake_simulation(i) for i in range(6)]

    splits = split_simulations(simulations, train_fraction=0.70)

    ids = {
        name: {s.simulation_id for s in getattr(splits, name)}
        for name in ("train", "val", "test")
    }

    assert not ids["train"] & ids["val"]
    assert not ids["val"] & ids["test"]
    assert not ids["train"] & ids["test"]

    # every simulation is used exactly once
    assert ids["train"] | ids["val"] | ids["test"] == set(range(6))
    assert sum(len(v) for v in ids.values()) == 6


def test_split_simulations_never_leaves_val_or_test_empty():
    """With 3 simulations there is exactly one for each block."""

    splits = split_simulations(
        [fake_simulation(i) for i in range(3)],
        train_fraction=0.99
    )

    assert len(splits.train) == 1
    assert len(splits.val) == 1
    assert len(splits.test) == 1


def test_split_simulations_gives_the_odd_one_to_test():

    splits = split_simulations(
        [fake_simulation(i) for i in range(6)],
        train_fraction=0.5
    )

    assert len(splits.train) == 3
    assert len(splits.val) == 1
    assert len(splits.test) == 2


def test_split_simulations_is_reproducible():

    def ids(seed):
        splits = split_simulations(
            [fake_simulation(i) for i in range(6)],
            train_fraction=0.70,
            seed=seed
        )
        return [s.simulation_id for s in splits.train]

    assert ids(68) == ids(68)
    assert ids(68) != ids(1)


def test_split_simulations_needs_at_least_three():

    with pytest.raises(ValueError, match="At least 3"):
        split_simulations([fake_simulation(0), fake_simulation(1)])
