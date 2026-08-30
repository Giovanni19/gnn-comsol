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
