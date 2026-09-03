"""
Time encodings appended to the node features.

Each network gets the normalized state (u, v, p) plus an encoding of the
time step. Which encoding is chosen by the "features" field of a network
in the experiment configuration.

    "time"          -> 4 features:  u, v, p, dt
    "time_fourier"  -> 12 features: u, v, p, dt, and sin/cos of dt at
                       4 frequencies
"""

import numpy as np

from .normalization import NUM_PHYSICS_FEATURES


# Number of node features produced by each encoding, given the 3 state
# variables. Used to size the first layer of a network.
FEATURE_SIZES = {
    "time": 4,
    "time_fourier": 12
}


def add_time_feature(X, delta_t):
    """
    Append the time step as a fourth feature.

    (S, N, 3) -> (S, N, 4)
    """

    num_timesteps, num_nodes = X.shape[0], X.shape[1]

    time_feature = np.zeros((num_timesteps, num_nodes, 1))

    for i in range(num_timesteps):
        time_feature[i, :, 0] = delta_t[i]

    return np.concatenate((X, time_feature), axis=2)


def add_time_fourier_features(X, delta_t, num_frequencies=4):
    """
    Append the time step plus a Fourier encoding of it.

    (S, N, 3) -> (S, N, 3 + 1 + 2 * num_frequencies)

    The sin/cos pairs at geometrically spaced frequencies give the
    network a multi-scale view of the time step, in the spirit of the
    positional encodings used by Transformers and NeRFs.

    Caveat kept from the original code: this is applied to the
    NORMALIZED delta_t, so the frequencies have no physical meaning.
    Applying it to a time in coherent units would be more defensible,
    but changing it here would alter the numbers, so it belongs in its
    own experiment.
    """

    n_nodes = X.shape[1]

    frequencies = 2.0 ** np.arange(num_frequencies)

    features = [X]

    features.append(
        np.repeat(delta_t[:, None, None], n_nodes, axis=1)
    )

    for frequency in frequencies:

        sin_feature = np.sin(2.0 * np.pi * frequency * delta_t)
        cos_feature = np.cos(2.0 * np.pi * frequency * delta_t)

        features.append(
            np.repeat(sin_feature[:, None, None], n_nodes, axis=1)
        )

        features.append(
            np.repeat(cos_feature[:, None, None], n_nodes, axis=1)
        )

    return np.concatenate(features, axis=2)


def add_time_derivative_features(X_norm, X_original, dt):
    """
    Append the backward time derivative of the state.

    Never used by any experiment so far, kept because it is the natural
    first step towards predicting the increment instead of the absolute
    next state.
    """

    dX_dt = np.zeros_like(X_original)

    dX_dt[1:] = (X_original[1:] - X_original[:-1]) / dt
    dX_dt[0] = dX_dt[1]

    return np.concatenate((X_norm, dX_dt), axis=2)


def build_features(kind, X_norm, dt_norm, num_frequencies=4):
    """
    Dispatch on the encoding name used in the configuration files.
    """

    if kind == "time":
        return add_time_feature(X_norm, dt_norm)

    if kind == "time_fourier":
        return add_time_fourier_features(
            X_norm,
            dt_norm,
            num_frequencies=num_frequencies
        )

    raise ValueError(
        f"Unknown feature encoding {kind!r}. "
        f"Expected one of {sorted(FEATURE_SIZES)}."
    )


# =====================================================================
# The pressure feature vector
#
# The BSMS pressure network is fed more than the state encoding, and the
# layout of what it is fed is decided HERE and nowhere else:
#
#     [ time encoding | physics features | predicted velocity ]
#       4 or 12         5, if used         2, if used
#
# This used to be assembled inline by the training script and rebuilt by
# hand in the evaluation script. The two could disagree on the ORDER of
# the blocks without disagreeing on their total width, which passes
# every shape check and produces confident nonsense - the same failure
# mode as the scaling bug this package was restructured to prevent.
# =====================================================================

def pressure_features_size(
    encoding,
    use_physics_features=False,
    use_predicted_velocity=False
):
    """
    Width of the vector build_pressure_features produces.

    This is what sizes the first layer of the model, so it must be
    derived from the same flags the builder uses - never from the name
    of the network.
    """

    if encoding not in FEATURE_SIZES:
        raise ValueError(
            f"Unknown feature encoding {encoding!r}. "
            f"Expected one of {sorted(FEATURE_SIZES)}."
        )

    size = FEATURE_SIZES[encoding]

    if use_physics_features:
        size += NUM_PHYSICS_FEATURES

    if use_predicted_velocity:
        size += 2

    return size


def build_pressure_features(
    encoding,
    X_norm,
    dt_norm,
    physics_features=None,
    predicted_velocity=None,
    num_frequencies=4
):
    """
    Assemble the input of the pressure network.

    Parameters
    ----------
    X_norm : (S, N, 3)
        Normalized state.

    dt_norm : (S,)
        Normalized time step.

    physics_features : (S, N, NUM_PHYSICS_FEATURES) or None
        Already normalized with the PhysicsNormalizer of the run. None
        when the network does not use them.

    predicted_velocity : (S, N, 2) or None
        u(t+1), v(t+1) from the velocity network, in normalized units.
        None when the network does not use them.

    Returns
    -------
    (S, N, pressure_features_size(...)) array.
    """

    features = build_features(
        encoding,
        X_norm,
        dt_norm,
        num_frequencies=num_frequencies
    )

    # Order matters: it is part of what the trained weights expect.
    blocks = [
        ("physics features", physics_features, NUM_PHYSICS_FEATURES),
        ("predicted velocity", predicted_velocity, 2)
    ]

    for name, block, width in blocks:

        if block is None:
            continue

        block = np.asarray(block)

        if block.shape[:2] != features.shape[:2]:
            raise ValueError(
                f"{name} has shape {block.shape}, which does not line "
                f"up sample by sample and node by node with the state "
                f"encoding of shape {features.shape}."
            )

        if block.shape[-1] != width:
            raise ValueError(
                f"Expected {width} {name}, got {block.shape[-1]}."
            )

        features = np.concatenate([features, block], axis=-1)

    expected = pressure_features_size(
        encoding,
        use_physics_features=physics_features is not None,
        use_predicted_velocity=predicted_velocity is not None
    )

    if features.shape[-1] != expected:
        raise ValueError(
            f"Assembled {features.shape[-1]} pressure features but "
            f"pressure_features_size() says {expected}."
        )

    return features
