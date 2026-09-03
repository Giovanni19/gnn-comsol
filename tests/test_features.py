"""
The pressure feature vector.

Its layout is a property of the trained weights: the training loaders
and the evaluation script both have to produce the same blocks in the
same order. They used to assemble it separately, which meant they could
agree on the total width and disagree on the order - a mismatch that
passes every shape check and silently produces wrong predictions.
"""

import numpy as np
import pytest

from gnn_comsol.data.features import (
    FEATURE_SIZES,
    build_features,
    build_pressure_features,
    pressure_features_size
)
from gnn_comsol.data.normalization import NUM_PHYSICS_FEATURES


SAMPLES, NODES = 6, 5


@pytest.fixture
def inputs():

    rng = np.random.default_rng(0)

    return {
        "X": rng.normal(size=(SAMPLES, NODES, 3)),
        "dt": rng.normal(size=SAMPLES),
        "physics": rng.normal(
            size=(SAMPLES, NODES, NUM_PHYSICS_FEATURES)
        ),
        "velocity": rng.normal(size=(SAMPLES, NODES, 2))
    }


@pytest.mark.parametrize("encoding", sorted(FEATURE_SIZES))
@pytest.mark.parametrize("physics", [False, True])
@pytest.mark.parametrize("velocity", [False, True])
def test_size_matches_what_the_builder_produces(
    inputs, encoding, physics, velocity
):
    """
    The one invariant that matters: the number the model is sized with
    is the number of columns the loader hands it.
    """

    features = build_pressure_features(
        encoding,
        inputs["X"],
        inputs["dt"],
        physics_features=inputs["physics"] if physics else None,
        predicted_velocity=inputs["velocity"] if velocity else None
    )

    expected = pressure_features_size(
        encoding,
        use_physics_features=physics,
        use_predicted_velocity=velocity
    )

    assert features.shape == (SAMPLES, NODES, expected)


def test_blocks_keep_their_documented_order(inputs):
    """
    [ time encoding | physics features | predicted velocity ]

    Getting this wrong does not change any shape, so nothing else would
    catch it.
    """

    features = build_pressure_features(
        "time",
        inputs["X"],
        inputs["dt"],
        physics_features=inputs["physics"],
        predicted_velocity=inputs["velocity"]
    )

    encoding_width = FEATURE_SIZES["time"]

    physics_end = encoding_width + NUM_PHYSICS_FEATURES

    assert np.allclose(
        features[..., :encoding_width],
        build_features("time", inputs["X"], inputs["dt"])
    )

    assert np.allclose(
        features[..., encoding_width:physics_end],
        inputs["physics"]
    )

    assert np.allclose(features[..., physics_end:], inputs["velocity"])


def test_no_extras_is_exactly_the_time_encoding(inputs):

    assert np.allclose(
        build_pressure_features("time", inputs["X"], inputs["dt"]),
        build_features("time", inputs["X"], inputs["dt"])
    )


def test_one_sample_at_a_time_matches_the_whole_block(inputs):
    """
    The evaluation script builds the vector one timestep at a time while
    training builds it for the whole simulation at once. The two must be
    the same array.
    """

    whole = build_pressure_features(
        "time",
        inputs["X"],
        inputs["dt"],
        physics_features=inputs["physics"],
        predicted_velocity=inputs["velocity"]
    )

    for timestep in range(SAMPLES):

        one = build_pressure_features(
            "time",
            inputs["X"][timestep:timestep + 1],
            inputs["dt"][timestep:timestep + 1],
            physics_features=(
                inputs["physics"][timestep:timestep + 1]
            ),
            predicted_velocity=(
                inputs["velocity"][timestep:timestep + 1]
            )
        )[0]

        assert np.allclose(one, whole[timestep])


def test_a_block_with_the_wrong_number_of_nodes_is_refused(inputs):

    with pytest.raises(ValueError, match="node by node"):
        build_pressure_features(
            "time",
            inputs["X"],
            inputs["dt"],
            physics_features=inputs["physics"][:, :NODES - 1]
        )


def test_a_block_with_the_wrong_width_is_refused(inputs):

    with pytest.raises(ValueError, match="physics features"):
        build_pressure_features(
            "time",
            inputs["X"],
            inputs["dt"],
            physics_features=inputs["physics"][..., :2]
        )

    with pytest.raises(ValueError, match="predicted velocity"):
        build_pressure_features(
            "time",
            inputs["X"],
            inputs["dt"],
            predicted_velocity=inputs["physics"]
        )


def test_an_unknown_encoding_is_refused():

    with pytest.raises(ValueError, match="Unknown feature encoding"):
        pressure_features_size("wavelet")
