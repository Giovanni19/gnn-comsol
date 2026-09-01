"""
Scaling of the state vector (u, v, p).

One object defines the scaling of both the input and the target. Input
and output live in the same physical space, so they must share it: that
is what makes it possible to feed a prediction back in as the next input
without a silent change of units.

The transform is stored inside the checkpoint and read back at inference
instead of being reconstructed by whoever loads the model. Before this,
the two halves lived in different files and drifted apart: the pressure
network was trained on physical units (or on a merely centred target)
while inference always applied a full de-standardization, so predictions
came out wrong by a factor of the pressure standard deviation.
"""

import numpy as np
import torch


# Columns of the state vector, so that no module has to write a bare
# 0:2 / 2:3 slice again.
VELOCITY_COLUMNS = slice(0, 2)
PRESSURE_COLUMNS = slice(2, 3)
STATE_COLUMNS = slice(0, 3)

VARIABLE_NAMES = ["u", "v", "p"]

# Which columns of the target a network is responsible for
TARGET_COLUMNS = {
    "velocity": VELOCITY_COLUMNS,
    "pressure": PRESSURE_COLUMNS,
    "state": STATE_COLUMNS
}


def compute_normalization_parameters(X_train, dt_train):
    """
    Mean and standard deviation, computed on TRAINING data only.

    u and v deliberately share one mean and one standard deviation,
    both taken from the velocity magnitude sqrt(u^2 + v^2). Scaling the
    two components separately would stretch the velocity field along one
    axis and destroy its isotropy.
    """

    u = X_train[:, :, 0]
    v = X_train[:, :, 1]

    velocity_magnitude = np.sqrt(u ** 2 + v ** 2)

    velocity_mean = velocity_magnitude.mean()
    velocity_std = velocity_magnitude.std() + 1e-8

    p = X_train[:, :, 2]

    pressure_mean = p.mean()
    pressure_std = p.std() + 1e-8

    delta_t_mean = dt_train.mean()
    delta_t_std = dt_train.std() + 1e-8

    x_mean = np.array([velocity_mean, velocity_mean, pressure_mean])
    x_std = np.array([velocity_std, velocity_std, pressure_std])

    return x_mean, x_std, delta_t_mean, delta_t_std

def compute_multi_simulation_normalization_parameters(train_simulations):
    """
    Compute normalization parameters using all TRAINING simulations.

    Simulations may have different numbers of nodes and timesteps.

    Every node at every timestep contributes equally to the statistics.
    Validation and test simulations must NOT be passed to this function.

    Velocity
    --------
    u and v share the same mean and standard deviation, computed from
    the velocity magnitude sqrt(u^2 + v^2).

    Pressure
    --------
    Mean and standard deviation are computed globally over all pressure
    values in all training simulations.

    Time step
    ---------
    Mean and standard deviation are computed over all transitions in all
    training simulations.
    """

    if len(train_simulations) == 0:
        raise ValueError(
            "At least one training simulation is required."
        )

    # =========================================================
    # Accumulators for velocity magnitude
    # =========================================================

    velocity_sum = 0.0
    velocity_sum_sq = 0.0
    velocity_count = 0

    # =========================================================
    # Accumulators for pressure
    # =========================================================

    pressure_sum = 0.0
    pressure_sum_sq = 0.0
    pressure_count = 0

    # =========================================================
    # Accumulators for delta_t
    # =========================================================

    dt_sum = 0.0
    dt_sum_sq = 0.0
    dt_count = 0

    # =========================================================
    # Loop over complete training simulations
    # =========================================================

    for simulation in train_simulations:

        X = simulation.X_input

        # -----------------------------------------------------
        # Velocity
        # -----------------------------------------------------

        u = X[:, :, 0]
        v = X[:, :, 1]

        velocity_magnitude = np.sqrt(
            u ** 2 + v ** 2
        )

        velocity_sum += np.sum(
            velocity_magnitude,
            dtype=np.float64
        )

        velocity_sum_sq += np.sum(
            velocity_magnitude ** 2,
            dtype=np.float64
        )

        velocity_count += velocity_magnitude.size

        # -----------------------------------------------------
        # Pressure
        # -----------------------------------------------------

        p = X[:, :, 2]

        pressure_sum += np.sum(
            p,
            dtype=np.float64
        )

        pressure_sum_sq += np.sum(
            p ** 2,
            dtype=np.float64
        )

        pressure_count += p.size

        # -----------------------------------------------------
        # delta_t
        # -----------------------------------------------------

        dt = np.asarray(
            simulation.delta_t,
            dtype=np.float64
        )

        dt_sum += np.sum(dt)

        dt_sum_sq += np.sum(dt ** 2)

        dt_count += dt.size

    # =========================================================
    # Means
    # =========================================================

    velocity_mean = (
        velocity_sum / velocity_count
    )

    pressure_mean = (
        pressure_sum / pressure_count
    )

    delta_t_mean = (
        dt_sum / dt_count
    )

    # =========================================================
    # Variances
    # E[x^2] - E[x]^2
    # =========================================================

    velocity_variance = (
        velocity_sum_sq / velocity_count
        - velocity_mean ** 2
    )

    pressure_variance = (
        pressure_sum_sq / pressure_count
        - pressure_mean ** 2
    )

    delta_t_variance = (
        dt_sum_sq / dt_count
        - delta_t_mean ** 2
    )

    # Protect against tiny negative values caused by
    # floating-point roundoff.
    velocity_variance = max(
        velocity_variance,
        0.0
    )

    pressure_variance = max(
        pressure_variance,
        0.0
    )

    delta_t_variance = max(
        delta_t_variance,
        0.0
    )

    # =========================================================
    # Standard deviations
    # =========================================================

    velocity_std = (
        np.sqrt(velocity_variance) + 1e-8
    )

    pressure_std = (
        np.sqrt(pressure_variance) + 1e-8
    )

    delta_t_std = (
        np.sqrt(delta_t_variance) + 1e-8
    )

    # =========================================================
    # State parameters
    # =========================================================

    x_mean = np.array(
        [
            velocity_mean,
            velocity_mean,
            pressure_mean,
        ],
        dtype=np.float64
    )

    x_std = np.array(
        [
            velocity_std,
            velocity_std,
            pressure_std,
        ],
        dtype=np.float64
    )

    return (
        x_mean,
        x_std,
        delta_t_mean,
        delta_t_std,
    )

class StateNormalizer:
    """
    The single definition of how the state (u, v, p) is scaled.

    The contract
    ------------
        training   uses   transform(y)
        inference  uses   inverse_transform(prediction)

    and inverse_transform(transform(y)) == y.

    Parameters must always be computed on training data only.
    """

    KIND = "standardize"

    def __init__(self, mean, std):

        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean {self.mean.shape} and std {self.std.shape} "
                "must have the same shape."
            )

        if np.any(self.std <= 0):
            raise ValueError(
                f"std must be strictly positive, got {self.std}."
            )

    def transform(self, values, columns=slice(None)):
        """Physical units -> normalized units."""

        return (
            values - self._as_like(self.mean[columns], values)
        ) / self._as_like(self.std[columns], values)

    def inverse_transform(self, values, columns=slice(None)):
        """
        Normalized units -> physical units.

        `columns` selects which part of the state the values refer to:
        VELOCITY_COLUMNS for an (N, 2) velocity prediction,
        PRESSURE_COLUMNS for an (N, 1) pressure prediction.
        """

        return (
            values * self._as_like(self.std[columns], values)
            + self._as_like(self.mean[columns], values)
        )

    @staticmethod
    def _as_like(parameters, values):
        """Match torch tensors (device and dtype), or stay in NumPy."""

        if isinstance(values, torch.Tensor):

            return torch.as_tensor(
                parameters,
                dtype=values.dtype,
                device=values.device
            )

        return parameters

    def to_dict(self):
        """Plain-Python form, safe to store inside a checkpoint."""

        return {
            "kind": self.KIND,
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }

    @classmethod
    def from_dict(cls, state):

        kind = state.get("kind")

        if kind != cls.KIND:
            raise ValueError(
                f"Unknown normalization kind {kind!r}: this checkpoint "
                f"was not produced with '{cls.KIND}'. Refusing to guess "
                "the scaling of its predictions."
            )

        return cls(state["mean"], state["std"])

    def __repr__(self):

        return (
            f"StateNormalizer(kind='{self.KIND}', "
            f"mean={np.array2string(self.mean, precision=4)}, "
            f"std={np.array2string(self.std, precision=4)})"
        )

def normalize_simulation(
    simulation,
    normalizer,
    dt_mean,
    dt_std,
):
    """
    Normalize one complete simulation while preserving its mesh.
    """

    return {
        "X": normalizer.transform(
            simulation.X_input
        ),

        "Y": normalizer.transform(
            simulation.Y_target
        ),

        "dt": (
            simulation.delta_t - dt_mean
        ) / dt_std,

        "edge_index": simulation.edge_index,
        "edge_weight": simulation.edge_weight,
        "pos": simulation.pos,

        "simulation_id": simulation.simulation_id,
        "file_path": simulation.file_path,
    }
