"""The scaling contract: transform and inverse_transform must be inverses."""

import numpy as np
import pytest

from gnn_comsol.data.normalization import (
    PRESSURE_COLUMNS,
    VELOCITY_COLUMNS,
    StateNormalizer,
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
