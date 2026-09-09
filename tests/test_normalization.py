"""The scaling contract: transform and inverse_transform must be inverses."""

import numpy as np
import pytest

from gnn_comsol.data.loading import RawDataset
from gnn_comsol.data.normalization import (
    NUM_PHYSICS_FEATURES,
    PHYSICS_FEATURE_NAMES,
    PRESSURE_COLUMNS,
    VELOCITY_COLUMNS,
    PhysicsNormalizer,
    StateNormalizer,
    compute_multi_simulation_delta_normalization_parameters,
    compute_multi_simulation_physics_normalization_parameters,
    compute_normalization_parameters
)


@pytest.fixture
def state():
    """u, v of order 1 and p of order 1000: the ratio that caused the bug."""

    rng = np.random.default_rng(3)

    N, T = 80, 40

    return np.stack([
        np.column_stack([
            rng.normal(0.5, 0.3, N),
            rng.normal(0.0, 0.3, N),
            rng.normal(2000.0, 800.0, N)
        ])
        for _ in range(T)
    ])


@pytest.fixture
def normalizer(state):

    mean, std, _, _ = compute_normalization_parameters(
        state, np.full(len(state), 0.1)
    )

    return StateNormalizer(mean, std)


def test_round_trip_is_exact(normalizer, state):

    back = normalizer.inverse_transform(normalizer.transform(state))

    assert np.allclose(back, state, atol=1e-8)


def test_u_and_v_share_their_scaling(normalizer):
    """Scaling the components separately would destroy isotropy."""

    assert normalizer.mean[0] == normalizer.mean[1]
    assert normalizer.std[0] == normalizer.std[1]


def test_normalized_target_is_order_one(normalizer, state):

    normalized = normalizer.transform(state)

    for column in range(3):
        assert 0.1 < normalized[:, :, column].std() < 10


def test_per_column_inversion_matches_full_inversion(normalizer, state):
    """This is what inference does: each network inverts its own slice."""

    snapshot = state[0]
    normalized = normalizer.transform(snapshot)

    velocity = normalizer.inverse_transform(
        normalized[:, VELOCITY_COLUMNS], VELOCITY_COLUMNS
    )

    pressure = normalizer.inverse_transform(
        normalized[:, PRESSURE_COLUMNS], PRESSURE_COLUMNS
    )

    recombined = np.concatenate([velocity, pressure], axis=1)

    assert np.allclose(recombined, snapshot)


def test_serialization_round_trip(normalizer, state):

    payload = normalizer.to_dict()

    # Must be primitives so the checkpoint stays weights-only loadable
    assert payload["kind"] == "standardize"
    assert isinstance(payload["mean"], list)

    restored = StateNormalizer.from_dict(payload)

    assert np.allclose(
        restored.inverse_transform(restored.transform(state)), state
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "center", "mean": [0, 0, 0], "std": [1, 1, 1]},
        {"mean": [0, 0, 0], "std": [1, 1, 1]},
    ],
)
def test_unknown_convention_is_refused(payload):
    """Guessing the scaling is exactly the bug this prevents."""

    with pytest.raises(ValueError):
        StateNormalizer.from_dict(payload)


def test_degenerate_std_is_refused():

    with pytest.raises(ValueError):
        StateNormalizer([0, 0, 0], [1, 0, 1])


def test_the_old_pressure_bug_would_be_caught(normalizer, state):
    """
    Reproduces B1: the network was trained on physical pressure but
    inference de-standardized its output anyway.
    """

    p_true = state[0][:, PRESSURE_COLUMNS]

    sigma = normalizer.std[2]
    mu = normalizer.mean[2]

    old_prediction = p_true * sigma + mu

    # Off by orders of magnitude, not by a rounding error
    assert np.abs(old_prediction - p_true).mean() > 1e3

    new_prediction = normalizer.inverse_transform(
        normalizer.transform(p_true, PRESSURE_COLUMNS), PRESSURE_COLUMNS
    )

    assert np.allclose(new_prediction, p_true)


# ---------------------------------------------------------------------
# Physics-derived features
# ---------------------------------------------------------------------

def physics_simulation(simulation_id, num_samples, num_nodes, scales,
                       seed=0):
    """A RawDataset carrying only what the physics statistics need."""

    rng = np.random.default_rng(seed)

    features = rng.normal(
        size=(num_samples, num_nodes, NUM_PHYSICS_FEATURES)
    ) * scales

    return RawDataset(
        X_input=np.zeros((num_samples, num_nodes, 3)),
        Y_target=np.zeros((num_samples, num_nodes, 3)),
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.ones(num_samples),
        h=np.array(0.0),
        pos=np.zeros((num_nodes, 2)),
        physics_features=features,
        simulation_id=simulation_id
    )


def test_physics_parameters_pool_meshes_of_different_sizes():
    """
    Every node at every timestep counts once, whatever mesh it is on.

    Simulations have different node counts, so the statistics have to be
    taken over the flattened sample-node axis rather than per simulation
    and then averaged.
    """

    scales = np.array([1.0, 2.0, 0.5, 4.0, 100.0])

    simulations = [
        physics_simulation(0, 12, 7, scales, seed=1),
        physics_simulation(1, 20, 13, scales, seed=2),
        physics_simulation(2, 5, 30, scales, seed=3)
    ]

    mean, std = (
        compute_multi_simulation_physics_normalization_parameters(
            simulations
        )
    )

    assert mean.shape == (NUM_PHYSICS_FEATURES,)
    assert std.shape == (NUM_PHYSICS_FEATURES,)

    # Same answer as pooling the nodes by hand
    pooled = np.concatenate(
        [
            simulation.physics_features.reshape(
                -1, NUM_PHYSICS_FEATURES
            )
            for simulation in simulations
        ]
    )

    assert np.allclose(mean, pooled.mean(axis=0))
    assert np.allclose(std, pooled.std(axis=0) + 1e-8)

    # The columns really are on different scales, so a normalizer that
    # did nothing would not pass the round trip below.
    assert std.max() / std.min() > 10


def test_physics_normalizer_round_trips_and_whitens():

    scales = np.array([1.0, 2.0, 0.5, 4.0, 100.0])

    simulations = [physics_simulation(0, 30, 11, scales, seed=4)]

    mean, std = (
        compute_multi_simulation_physics_normalization_parameters(
            simulations
        )
    )

    physics_normalizer = PhysicsNormalizer(mean, std)

    features = simulations[0].physics_features

    normalized = physics_normalizer.transform(features)

    assert np.allclose(normalized.mean(axis=(0, 1)), 0.0, atol=1e-8)
    assert np.allclose(normalized.std(axis=(0, 1)), 1.0, atol=1e-6)

    assert np.allclose(
        physics_normalizer.inverse_transform(normalized), features
    )


def test_physics_normalizer_serializes_with_its_feature_names():
    """
    A stored normalizer must be readable back without knowing which
    features it was built for, and must refuse a foreign convention.
    """

    physics_normalizer = PhysicsNormalizer(
        np.zeros(NUM_PHYSICS_FEATURES), np.ones(NUM_PHYSICS_FEATURES)
    )

    stored = physics_normalizer.to_dict()

    assert stored["feature_names"] == PHYSICS_FEATURE_NAMES

    restored = PhysicsNormalizer.from_dict(stored)

    assert np.allclose(restored.mean, physics_normalizer.mean)
    assert np.allclose(restored.std, physics_normalizer.std)

    with pytest.raises(ValueError, match="physics normalization"):
        PhysicsNormalizer.from_dict({**stored, "kind": "standardize"})


def test_physics_parameters_name_the_simulation_that_lacks_them():

    scales = np.ones(NUM_PHYSICS_FEATURES)

    good = physics_simulation(0, 5, 4, scales)

    bad = physics_simulation(1, 5, 4, scales)
    bad.physics_features = None

    with pytest.raises(ValueError, match="Simulation 1"):
        compute_multi_simulation_physics_normalization_parameters(
            [good, bad]
        )


def delta_simulation(simulation_id, num_samples, num_nodes, seed=0):
    """A RawDataset with a known one-step increment distribution."""

    rng = np.random.default_rng(seed)

    X_input = rng.normal(0.0, 1.0, size=(num_samples, num_nodes, 3))

    delta = rng.normal(0.0, 1.0, size=(num_samples, num_nodes, 3))
    delta[:, :, :2] *= 0.01  # velocity increments are small
    delta[:, :, 2] *= 5.0    # pressure increments on a different scale

    Y_target = X_input + delta

    return RawDataset(
        X_input=X_input,
        Y_target=Y_target,
        edge_index=np.zeros((2, 1), dtype=int),
        edge_weight=np.ones(1),
        delta_t=np.ones(num_samples),
        h=np.array(0.0),
        pos=np.zeros((num_nodes, 2)),
        simulation_id=simulation_id
    )


def test_delta_parameters_pool_meshes_of_different_sizes():

    simulations = [
        delta_simulation(0, 12, 7, seed=1),
        delta_simulation(1, 20, 13, seed=2),
        delta_simulation(2, 5, 30, seed=3),
    ]

    mean, std = compute_multi_simulation_delta_normalization_parameters(
        simulations
    )

    assert mean.shape == (3,)
    assert std.shape == (3,)

    # du and dv share one mean/std, taken from the magnitude of the
    # velocity increment - same convention as the absolute-state one.
    assert mean[0] == mean[1]
    assert std[0] == std[1]


def test_delta_normalizer_is_not_the_state_normalizer():
    """
    The whole point of a separate delta normalizer: an increment is
    centred near zero and much smaller than the absolute field, so
    reusing StateNormalizer's own parameters on it would neither centre
    nor whiten it.
    """

    simulations = [delta_simulation(0, 200, 15, seed=7)]

    delta_mean, delta_std = (
        compute_multi_simulation_delta_normalization_parameters(
            simulations
        )
    )

    state_mean, state_std, _, _ = compute_normalization_parameters(
        simulations[0].X_input, simulations[0].delta_t
    )

    assert not np.allclose(delta_mean, state_mean)
    assert not np.allclose(delta_std, state_std)

    delta_normalizer = StateNormalizer(delta_mean, delta_std)

    delta = simulations[0].Y_target - simulations[0].X_input

    normalized = delta_normalizer.transform(delta)

    assert np.allclose(normalized.mean(axis=(0, 1)), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=(0, 1)), 1.0, atol=1e-6)

    assert np.allclose(
        delta_normalizer.inverse_transform(normalized), delta
    )


def test_delta_parameters_require_at_least_one_simulation():

    with pytest.raises(ValueError, match="At least one"):
        compute_multi_simulation_delta_normalization_parameters([])
